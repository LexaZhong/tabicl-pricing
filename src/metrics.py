"""The single metrics implementation, used by every track and every model.

Scale note: mean Tweedie deviance at p=1.5 on pure premium with exposure weights lands
around 75 on freMTPL2 -- that is the "~75" figure from the team's earlier runs. Poisson
deviance on frequency lands near 0.4. Both are reported explicitly so the two never get
confused again.

Gini convention: predictions are sorted ASCENDING, the Lorenz curve plots cumulative loss
share against cumulative weight share, and Gini = 1 - 2 * area. `normalized` divides by
the Gini of a perfect ranker (the actual losses), so it is capped at 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_tweedie_deviance,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

_EPS = 1e-10

# Two predictions closer than this (relative) are treated as tied in the Lorenz curve.
_GINI_RTOL = 1e-9

# "Full-term" stratum for the artifact-free risk check. Within it 1/Exposure is nearly
# constant, so a model cannot score by ranking on exposure alone.
FIXED_EXPOSURE_MIN = 0.95


# --- deviances ---------------------------------------------------------------


def tweedie_deviance(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray, power: float
) -> float:
    """Mean Tweedie deviance. power=1 is Poisson, 2 is Gamma, 1<p<2 is compound Poisson."""
    return float(
        mean_tweedie_deviance(
            np.asarray(y_true, dtype=float),
            np.clip(np.asarray(y_pred, dtype=float), _EPS, None),
            sample_weight=np.asarray(sample_weight, dtype=float),
            power=power,
        )
    )


def poisson_deviance(y_true, y_pred, sample_weight) -> float:
    return tweedie_deviance(y_true, y_pred, sample_weight, power=1.0)


def gamma_deviance(y_true, y_pred, sample_weight) -> float:
    return tweedie_deviance(y_true, y_pred, sample_weight, power=2.0)


# --- Gini --------------------------------------------------------------------


def gini(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Gini of the loss-vs-risk Lorenz curve.

    y_true is a RATE (pure premium); sample_weight is exposure. Total loss per row is
    y_true * weight, which is what the Lorenz curve accumulates.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.ones_like(y_true) if sample_weight is None else np.asarray(sample_weight, dtype=float)

    losses = y_true * w
    total_loss = losses.sum()
    total_w = w.sum()
    if total_loss <= 0 or total_w <= 0:
        return float("nan")

    order = np.argsort(y_pred, kind="mergesort")
    p_sorted = y_pred[order]

    # Tied predictions must contribute a STRAIGHT segment to the Lorenz curve: a model
    # that cannot separate two risks earns no credit for the order they happen to sit in.
    # Without this the metric reads row order as signal -- a constant predictor scored
    # -0.13 on freMTPL2 instead of 0, and results were not reproducible under reordering.
    #
    # The comparison needs a RELATIVE TOLERANCE, not exact inequality. A constant premium
    # reaches this function as c/e*e, which does not recover c bit-exactly in floating
    # point, and the residual wobble correlates with e -- so exact tie-testing broke the
    # ties in a biased direction and scored 0.0032 instead of 0.
    diffs = np.diff(p_sorted)
    scale = np.maximum(np.abs(p_sorted[:-1]), np.abs(p_sorted[1:]))
    new_group = np.concatenate([[True], diffs > _GINI_RTOL * scale])
    group_id = np.cumsum(new_group) - 1
    n_groups = int(group_id[-1]) + 1

    w_g = np.bincount(group_id, weights=w[order], minlength=n_groups)
    loss_g = np.bincount(group_id, weights=losses[order], minlength=n_groups)

    cum_w = np.concatenate([[0.0], np.cumsum(w_g) / total_w])
    cum_loss = np.concatenate([[0.0], np.cumsum(loss_g) / total_loss])

    area = np.trapezoid(cum_loss, cum_w) if hasattr(np, "trapezoid") else np.trapz(cum_loss, cum_w)
    return float(1.0 - 2.0 * area)


def normalized_gini(y_true, y_pred, sample_weight=None) -> float:
    """Gini relative to a perfect ranker. 1.0 = perfect ordering, 0.0 = random."""
    perfect = gini(y_true, y_true, sample_weight)
    if not np.isfinite(perfect) or abs(perfect) < _EPS:
        return float("nan")
    return gini(y_true, y_pred, sample_weight) / perfect


# --- calibration & lift ------------------------------------------------------


def calibration_ratio(y_true, y_pred, sample_weight) -> float:
    """Sum(predicted loss) / Sum(actual loss). 1.0 means the portfolio level is right."""
    w = np.asarray(sample_weight, dtype=float)
    pred_loss = (np.asarray(y_pred, dtype=float) * w).sum()
    actual_loss = (np.asarray(y_true, dtype=float) * w).sum()
    if actual_loss <= 0:
        return float("nan")
    return float(pred_loss / actual_loss)


def lift_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Equal-EXPOSURE buckets ordered by predicted loss cost.

    Equal-exposure (not equal-count) buckets are the actuarial convention: each bucket
    carries the same amount of risk, so the actual loss costs are comparably stable.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(sample_weight, dtype=float)

    order = np.argsort(y_pred, kind="mergesort")
    y_true, y_pred, w = y_true[order], y_pred[order], w[order]

    cum_w = np.cumsum(w) / w.sum()
    # searchsorted on cumulative exposure gives equal-exposure edges, not equal counts.
    bucket = np.minimum((cum_w * n_buckets).astype(int), n_buckets - 1)

    rows = []
    for b in range(n_buckets):
        m = bucket == b
        if not m.any():
            continue
        exposure = w[m].sum()
        rows.append(
            {
                "bucket": b + 1,
                "n_policies": int(m.sum()),
                "exposure": exposure,
                "actual_loss_cost": (y_true[m] * w[m]).sum() / exposure,
                "predicted_loss_cost": (y_pred[m] * w[m]).sum() / exposure,
            }
        )
    out = pd.DataFrame(rows)
    overall = (y_true * w).sum() / w.sum()
    out["actual_lift"] = out["actual_loss_cost"] / overall
    out["predicted_lift"] = out["predicted_loss_cost"] / overall
    return out


def double_lift_table(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    sample_weight: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Bucket by the ratio pred_a / pred_b -- the standard "is A better than B?" chart.

    Where A prices higher than B, A wins if actual losses are also higher there.
    """
    ratio = np.asarray(pred_a, dtype=float) / np.clip(np.asarray(pred_b, dtype=float), _EPS, None)
    y_true = np.asarray(y_true, dtype=float)
    w = np.asarray(sample_weight, dtype=float)

    order = np.argsort(ratio, kind="mergesort")
    cum_w = np.cumsum(w[order]) / w.sum()
    bucket = np.minimum((cum_w * n_buckets).astype(int), n_buckets - 1)

    rows = []
    for b in range(n_buckets):
        m = bucket == b
        if not m.any():
            continue
        idx = order[m]
        exposure = w[idx].sum()
        rows.append(
            {
                "bucket": b + 1,
                "exposure": exposure,
                "mean_ratio": float(ratio[idx].mean()),
                "actual_loss_cost": (y_true[idx] * w[idx]).sum() / exposure,
                "pred_a_loss_cost": (np.asarray(pred_a)[idx] * w[idx]).sum() / exposure,
                "pred_b_loss_cost": (np.asarray(pred_b)[idx] * w[idx]).sum() / exposure,
            }
        )
    return pd.DataFrame(rows)


# --- stage-level diagnostics -------------------------------------------------


def hurdle_to_frequency(p_hat: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """Convert P(ClaimNb >= 1) to an implied Poisson rate.

    Inverts P(N >= 1) = 1 - exp(-lambda * e). Without this the hurdle stage 1 and the
    Poisson GLM cannot be scored on the same frequency axis at all.
    """
    p = np.clip(np.asarray(p_hat, dtype=float), _EPS, 1.0 - 1e-9)
    return -np.log1p(-p) / np.clip(np.asarray(exposure, dtype=float), _EPS, None)


def stage1_metrics(y_binary: np.ndarray, p_hat: np.ndarray) -> dict[str, float]:
    """Stage 1 is Bernoulli, not a count model -- Poisson deviance does not apply here."""
    y = np.asarray(y_binary, dtype=int)
    p = np.clip(np.asarray(p_hat, dtype=float), _EPS, 1 - _EPS)
    out = {
        "stage1_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "stage1_brier": float(brier_score_loss(y, p)),
        "stage1_mean_pred": float(p.mean()),
        "stage1_base_rate": float(y.mean()),
    }
    out["stage1_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out


def exposure_monotonicity(
    p_hat: np.ndarray, exposure: np.ndarray, n_buckets: int = 10
) -> dict[str, float]:
    """Does stage 1 actually use exposure?

    Exposure is a FEATURE of the classifier, not an offset, so nothing forces p_hat to
    rise with exposure. If it does not, pricing silently breaks on part-year policies
    while the headline Gini looks fine. Returns Spearman rho over exposure deciles and
    the fraction of adjacent deciles that increase.
    """
    p = np.asarray(p_hat, dtype=float)
    e = np.asarray(exposure, dtype=float)
    bucket = pd.qcut(e, q=n_buckets, labels=False, duplicates="drop")
    means = pd.Series(p).groupby(bucket).mean().to_numpy()
    if len(means) < 2:
        return {"exposure_monotonic_frac": float("nan"), "exposure_spearman": float("nan")}
    diffs = np.diff(means)
    ranks_x = np.arange(len(means))
    rho = float(np.corrcoef(ranks_x, np.argsort(np.argsort(means)))[0, 1])
    return {
        "exposure_monotonic_frac": float((diffs > 0).mean()),
        "exposure_spearman": rho,
        "exposure_p_first_decile": float(means[0]),
        "exposure_p_last_decile": float(means[-1]),
    }


# --- the full suite ----------------------------------------------------------


def evaluate_pure_premium(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exposure: np.ndarray,
) -> dict[str, float]:
    """Headline metrics. y_true and y_pred are pure premium (loss per unit exposure)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), _EPS, None)
    w = np.asarray(exposure, dtype=float)

    # Rank by predicted TOTAL LOSS (rate x exposure), not by rate.
    #
    # The rate-based Gini has a degenerate solution: pred = c / Exposure scores wGini
    # 0.4676 on freMTPL2 while carrying zero risk information -- as a price it charges
    # every policy the same premium (rate x exposure = c). Any model of the hurdle form
    # p * S / Exposure inherits that 1/Exposure factor for free, which flatters TabICL's
    # two-stage model and every other hurdle model. Ranking on total loss removes the
    # artifact: a constant premium is constant, so it ties and scores exactly 0.
    actual_loss = y_true * w
    predicted_loss = y_pred * w

    # Third view: rate Gini restricted to near-full-term policies. Here 1/Exposure is
    # almost constant, so it carries no ranking signal and what survives is genuine risk
    # discrimination. The trivial c/e model scores ~0 on this by construction.
    hi = w >= FIXED_EXPOSURE_MIN
    if int(hi.sum()) >= 100:
        fixed = {
            "gini_fixed_exposure": gini(y_true[hi], y_pred[hi], w[hi]),
            "normalized_gini_fixed_exposure": normalized_gini(y_true[hi], y_pred[hi], w[hi]),
            "n_fixed_exposure": float(hi.sum()),
        }
    else:
        fixed = {}

    return {
        **fixed,
        "tweedie_deviance_1.5": tweedie_deviance(y_true, y_pred, w, power=1.5),
        "tweedie_deviance_1.9": tweedie_deviance(y_true, y_pred, w, power=1.9),
        "gini": gini(y_true, y_pred, None),
        "normalized_gini": normalized_gini(y_true, y_pred, None),
        "gini_exposure_weighted": gini(y_true, y_pred, w),
        "normalized_gini_exposure_weighted": normalized_gini(y_true, y_pred, w),
        "gini_total_loss": gini(actual_loss, predicted_loss, None),
        "normalized_gini_total_loss": normalized_gini(actual_loss, predicted_loss, None),
        "calibration_ratio": calibration_ratio(y_true, y_pred, w),
        "mean_predicted_loss_cost": float((y_pred * w).sum() / w.sum()),
        "mean_actual_loss_cost": float((y_true * w).sum() / w.sum()),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exposure: np.ndarray,
    metric: str = "gini_exposure_weighted",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI. Needed to show a gap between models is real, not noise."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    w = np.asarray(exposure, dtype=float)

    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = evaluate_pure_premium(y_true[idx], y_pred[idx], w[idx])[metric]

    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass
class EvalResult:
    """Everything one (model, config, seed) produces."""

    metrics: dict[str, float] = field(default_factory=dict)
    lift: pd.DataFrame | None = None

    def to_rows(self, **keys) -> list[dict]:
        return [{**keys, "metric": k, "value": v} for k, v in self.metrics.items()]
