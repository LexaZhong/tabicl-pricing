"""Automated experiment logic checks.

Motivated by a concrete failure: at N<=5,000 the tuned `xgb_hurdle` collapsed to a single
leaf in both boosters, so its prediction was (p0*S0)/Exposure -- a constant premium with
zero risk information. It scored 0.4676 weighted Gini, bit-identical across four training
sizes and five seeds, and nothing in the pipeline objected. These checks make that class
of failure loud instead of silent.

Two families:
  * per-config  -- run against one model's predictions on the test set
  * cross-config -- run across seeds/N once a sweep has results

Severity: FAIL blocks a headline result; WARN is reported and interpreted; INFO is context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Severity = Literal["INFO", "WARN", "FAIL"]

_EPS = 1e-12

# Models that exist as reference lines, not competitors. Several checks invert for these:
# a constant Gini is correct behaviour for `one_over_exposure` and a symptom for anything
# else, so the checks key on ROLE rather than on the number.
REFERENCE_MODELS = {"intercept", "one_over_exposure"}

# Models whose fitter is deterministic given the data. They vary across draws only because
# the SUBSAMPLE varies -- so when the draw is the whole pool they are identically constant,
# and that is correct rather than a symptom.
DETERMINISTIC_MODELS = {"glm_tweedie", "glm_hurdle", "glm_freqsev", "intercept",
                        "one_over_exposure"}

# The trivial 1/Exposure floor on freMTPL2's fixed test set (see analyze_trackB).
TRIVIAL_RATE_GINI = 0.4676


@dataclass
class Flag:
    check: str
    severity: Severity
    message: str
    context: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.message}"


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without pulling in scipy."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() < _EPS or rb.std() < _EPS:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def check_config(
    model: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    exposure: np.ndarray,
    metrics: dict | None = None,
    info: dict | None = None,
    trivial_gini: float | None = None,
) -> list[Flag]:
    """Per-config checks. `pred` and `y_true` are pure premium (rate).

    `trivial_gini` is the 1/exposure floor for THIS test set. It defaults to the value for
    the fixed 135,603-row freMTPL2 test set, but the floor is test-set specific -- on a
    20k subset it is 0.3606, not 0.4676 -- so pass it explicitly whenever the test set
    differs, or the floor check compares against the wrong line.
    """
    floor = TRIVIAL_RATE_GINI if trivial_gini is None else float(trivial_gini)
    metrics = metrics or {}
    info = info or {}
    flags: list[Flag] = []
    is_reference = model in REFERENCE_MODELS

    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    e = np.asarray(exposure, dtype=float)
    premium = pred * e

    # 4. Prediction sanity -- run first; everything downstream assumes finite positives.
    n_bad = int((~np.isfinite(pred)).sum() + (pred <= 0).sum())
    if n_bad:
        flags.append(Flag("prediction_sanity", "FAIL",
                          f"{n_bad:,} predictions non-finite or <= 0", {"model": model}))
    ceiling = 100.0 * float(np.max(y_true)) if len(y_true) else np.inf
    n_huge = int((pred > ceiling).sum())
    if n_huge:
        flags.append(Flag("prediction_sanity", "FAIL",
                          f"{n_huge:,} predictions exceed 100x the max observed pure premium",
                          {"model": model, "ceiling": ceiling}))

    # 1. Degeneracy -- constant premium means no differentiation at all.
    mean_prem = float(np.mean(premium))
    cv = float(np.std(premium) / mean_prem) if mean_prem > _EPS else 0.0
    if cv < 1e-6:
        sev: Severity = "INFO" if is_reference else "FAIL"
        flags.append(Flag("degeneracy", sev,
                          f"premium CV={cv:.2e} -- collapsed to a constant premium "
                          f"(no risk differentiation)", {"model": model, "cv": cv}))
    else:
        flags.append(Flag("degeneracy", "INFO", f"premium CV={cv:.4f}", {"model": model}))

    # 2. Is the prediction essentially a function of exposure alone?
    rho = _spearman(pred, 1.0 / np.clip(e, _EPS, None))
    if np.isfinite(rho) and abs(rho) > 0.95 and not is_reference:
        flags.append(Flag("exposure_dependence", "WARN",
                          f"Spearman(pred, 1/exposure)={rho:+.4f} -- prediction is nearly a "
                          f"function of exposure alone", {"model": model, "rho": rho}))

    # 3. Does it clear the trivial floor?
    if not is_reference:
        rg = metrics.get("gini_exposure_weighted")
        if rg is not None and rg <= floor + 1e-4:
            flags.append(Flag("trivial_floor", "WARN",
                              f"rate Gini {rg:.4f} does not beat the 1/exposure floor "
                              f"({floor:.4f})", {"model": model}))
        tg = metrics.get("gini_total_loss")
        if tg is not None and tg <= 0.01:
            flags.append(Flag("trivial_floor", "WARN",
                              f"total-loss Gini {tg:.4f} ~ 0 -- no signal on which policies "
                              f"cost most", {"model": model}))

    # 5. Calibration.
    cal = metrics.get("calibration_ratio")
    if cal is not None and np.isfinite(cal) and abs(cal - 1.0) > 0.5:
        flags.append(Flag("calibration", "WARN",
                          f"predicted/actual = {cal:.3f}", {"model": model}))

    # 6. Stage-2 support.
    n_claims = info.get("n_claims_stage2")
    if n_claims is not None and n_claims < 50:
        flags.append(Flag("stage2_support", "WARN",
                          f"severity model fitted on {n_claims} claims -- under-supported",
                          {"model": model}))

    # 7. Severity clipping: predictions pinned to the floor/cap are constrained, not learned.
    frac = info.get("severity_clipped_frac")
    if frac is not None and frac > 0.20:
        flags.append(Flag("severity_clipping", "WARN",
                          f"{frac:.1%} of stage-2 predictions sit on the floor or cap",
                          {"model": model}))

    # 8. Exposure monotonicity of the hurdle probability.
    mono = info.get("exposure_monotonic_frac")
    if mono is not None and mono < 0.7:
        flags.append(Flag("exposure_monotonicity", "WARN",
                          f"stage-1 p_hat rises with exposure in only {mono:.0%} of deciles -- "
                          f"the classifier may be ignoring exposure", {"model": model}))

    return flags


def check_across_seeds(
    model: str,
    metric: str,
    values: list[float],
    reference_value: float | None = None,
    identical_data: bool = False,
) -> list[Flag]:
    """Cross-config checks 9 and 10, for one (model, metric) across its seeds.

    `identical_data=True` means every draw saw the same rows -- which happens at N equal to
    the full training pool, where subsampling is a no-op. A deterministic fitter is then
    expected to return an identical metric, so constancy is INFO rather than FAIL.
    """
    flags: list[Flag] = []
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) < 2:
        return flags
    is_reference = model in REFERENCE_MODELS
    sd = float(np.std(vals, ddof=1))

    # Exact float equality is the wrong test here, for the same reason it was wrong in the
    # Gini tie check: a collapsed model can leave sub-ulp wobble between seeds and still be
    # entirely non-responsive to the data. Treat "varies by less than 1e-9 relative" as
    # constant.
    scale = max(abs(float(np.mean(vals))), _EPS)
    is_constant = sd <= max(1e-12, 1e-9 * scale)

    # 9. Zero variance. A symptom for a competitor; expected for a reference, whose Gini
    # depends only on an ordering that does not change with the data draw.
    expected_constant = is_reference or (identical_data and model in DETERMINISTIC_MODELS)
    if is_constant:
        if expected_constant:
            why = ("expected for a reference line" if is_reference else
                   "expected: deterministic fitter and every draw saw the whole pool, so "
                   "there is no subsample variance to measure at this N")
            flags.append(Flag("zero_variance", "INFO",
                              f"{model}/{metric} constant across {len(vals)} draws ({why})",
                              {"model": model, "metric": metric}))
        else:
            flags.append(Flag("zero_variance", "FAIL",
                              f"{model}/{metric} is bit-identical across {len(vals)} draws "
                              f"({vals[0]:.6f}) -- the model is not responding to the data",
                              {"model": model, "metric": metric, "value": float(vals[0])}))
    elif is_reference and "gini" in metric:
        flags.append(Flag("zero_variance", "FAIL",
                          f"reference {model}/{metric} varies across draws (SD={sd:.2e}) -- "
                          f"it should be ordering-determined and constant",
                          {"model": model, "metric": metric}))

    # 10. Bit-match to the trivial reference.
    if reference_value is not None and not is_reference:
        if np.all(np.abs(vals - reference_value) < 1e-6):
            flags.append(Flag("trivial_match", "FAIL",
                              f"{model}/{metric} matches the trivial 1/exposure model to 1e-6 "
                              f"({reference_value:.6f}) -- it has collapsed to that ordering",
                              {"model": model, "metric": metric}))
    return flags


def check_sample_comparability(samples: list[dict], n: int, z: float = 3.0) -> list[Flag]:
    """Check 11: flag resample draws whose composition is an outlier.

    A single heavy-tail draw can move a whole metric; it must be visible rather than
    averaged away. freMTPL2's max single claim is 4.07M against a 167 portfolio loss cost,
    so this is not hypothetical.
    """
    flags: list[Flag] = []
    if len(samples) < 4:
        return flags
    for key in ["mean_exposure", "claim_rate", "total_loss", "max_loss"]:
        vals = np.asarray([s.get(key, np.nan) for s in samples], dtype=float)
        if not np.isfinite(vals).all() or np.std(vals) < _EPS:
            continue
        mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        for i, v in enumerate(vals):
            if abs(v - mu) > z * sd:
                flags.append(Flag("sample_comparability", "WARN",
                                  f"N={n:,} draw {i}: {key}={v:,.4g} is "
                                  f"{(v - mu) / sd:+.1f} SD from the across-draw mean "
                                  f"({mu:,.4g})", {"n": n, "draw": i, "key": key}))
    return flags


def check_curve_monotonicity(
    model: str, metric: str, by_n: dict[int, tuple[float, float]], higher_better: bool
) -> list[Flag]:
    """Check 12: flag an N where the metric worsens by more than 2 pooled SD."""
    flags: list[Flag] = []
    ns = sorted(by_n)
    for prev, cur in zip(ns, ns[1:]):
        m0, s0 = by_n[prev]
        m1, s1 = by_n[cur]
        if not all(np.isfinite([m0, m1])):
            continue
        pooled = float(np.sqrt(np.nan_to_num(s0) ** 2 + np.nan_to_num(s1) ** 2)) or _EPS
        delta = (m1 - m0) if higher_better else (m0 - m1)
        if delta < -2.0 * pooled:
            flags.append(Flag("curve_monotonicity", "WARN",
                              f"{model}/{metric} worsens from N={prev:,} to N={cur:,} "
                              f"({m0:.4f} -> {m1:.4f}, {abs(delta) / pooled:.1f} pooled SD)",
                              {"model": model, "metric": metric}))
    return flags


def write_report(flags: list[Flag], path) -> dict[str, int]:
    """Write checks_report.md and return counts by severity."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(1 for f in flags if f.severity == s) for s in ("FAIL", "WARN", "INFO")}

    lines = ["# Experiment logic checks", ""]
    lines.append(f"**FAIL {counts['FAIL']}  ·  WARN {counts['WARN']}  ·  INFO {counts['INFO']}**")
    lines.append("")
    if counts["FAIL"] == 0:
        lines.append("No blocking failures.")
    else:
        lines.append("**Blocking failures present — headline results are withheld.**")
    lines.append("")
    for sev in ("FAIL", "WARN", "INFO"):
        sub = [f for f in flags if f.severity == sev]
        if not sub:
            continue
        lines.append(f"## {sev} ({len(sub)})")
        lines.append("")
        for f in sub:
            lines.append(f"- **{f.check}** — {f.message}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return counts
