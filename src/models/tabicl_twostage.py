"""TabICL two-stage hurdle model for pure premium.

Structure (NOT frequency-severity):
  stage 1  TabICLClassifier   -> p = P(ClaimNb >= 1)      trained on all policies
  stage 2  TabICLRegressor    -> S = E[total loss | claim] trained on claims-only rows
  combine  pure_premium = p * S / exposure

Stage 2 targets TOTAL loss per policy rather than loss per claim, so the two stages
compose directly with no E[ClaimNb | ClaimNb>=1] reconstruction term.

log(exposure) is a feature of BOTH stages: stage 1 because claim probability rises with
time on risk, stage 2 because a longer exposure can accumulate more than one claim. TabICL
has no offset mechanism, so this is a learned relationship -- verify it with
metrics.exposure_monotonicity rather than assuming it held.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from ..contexts import correct_prior, enriched_context_indices, random_context_indices

_EPS = 1e-10


@dataclass
class FitInfo:
    """Diagnostics the runner records alongside the metrics."""

    stage1_context_size: int = 0
    stage2_context_size: int = 0
    n_contexts: int = 0
    sampling_odds_ratio: float = 1.0
    severity_cap: float = float("nan")
    peak_vram_mib: float = 0.0
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    extra: dict = field(default_factory=dict)


def _append_log_exposure(X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    log_e = np.log(np.clip(np.asarray(exposure, dtype=float), _EPS, None)).reshape(-1, 1)
    return np.hstack([np.asarray(X, dtype=np.float32), log_e.astype(np.float32)])


class TabICLTwoStage:
    """Fit/predict signature matches the baselines so the runner treats them identically."""

    def __init__(
        self,
        context_size: int = 20_000,
        n_contexts: int = 1,
        sampler: str = "enriched",
        neg_per_pos: float = 3.0,
        stage2_context_size: int = 30_000,
        cap_quantile: float = 0.995,
        cap_mode: str = "empirical",
        fixed_cap: float | None = None,
        fixed_floor: float | None = None,
        n_estimators: int = 8,
        batch_size: int = 2,
        predict_chunk_size: int = 20_000,
        device: str | None = None,
        use_amp="auto",
        use_fa3: bool = False,  # FlashAttention-3 is Hopper-only; Blackwell must not use it
        offload_mode="auto",
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.context_size = context_size
        self.n_contexts = n_contexts
        self.sampler = sampler
        self.neg_per_pos = neg_per_pos
        self.stage2_context_size = stage2_context_size
        self.cap_quantile = cap_quantile
        # "empirical" reproduces Track B: the cap is the draw's own 99.5% quantile. At the
        # sizes Track C runs that is estimated from ~2-4 rows and swings 14.6k-168.2k across
        # seeds -- and since predict_severity hard-clips to it, the ceiling on every
        # prediction moves by an order of magnitude on draw luck alone. "fixed" pins it to a
        # value declared up front so the cap stops being a random variable.
        self.cap_mode = cap_mode
        self.fixed_cap = fixed_cap
        self.fixed_floor = fixed_floor
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.predict_chunk_size = predict_chunk_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.seed = seed
        self.verbose = verbose

        self.info = FitInfo()
        self._stage1: list = []
        self._odds_ratios: list[float] = []
        self._stage2 = None
        self._severity_cap = float("inf")
        self._severity_floor = _EPS
        self._last_p_hat: np.ndarray | None = None

    # --- internals ---
    def _make_classifier(self, seed: int):
        from tabicl import TabICLClassifier

        return TabICLClassifier(
            n_estimators=self.n_estimators,
            batch_size=self.batch_size,
            device=self.device,
            use_amp=self.use_amp,
            use_fa3=self.use_fa3,
            offload_mode=self.offload_mode,
            random_state=seed,
            verbose=self.verbose,
        )

    def _make_regressor(self, seed: int):
        from tabicl import TabICLRegressor

        return TabICLRegressor(
            n_estimators=self.n_estimators,
            batch_size=self.batch_size,
            device=self.device,
            use_amp=self.use_amp,
            use_fa3=self.use_fa3,
            offload_mode=self.offload_mode,
            random_state=seed,
            verbose=self.verbose,
        )

    def fit(
        self,
        X: np.ndarray,
        hurdle_y: np.ndarray,
        total_loss: np.ndarray,
        exposure: np.ndarray,
    ) -> "TabICLTwoStage":
        """`hurdle_y` is the stage-1 indicator -- pass df['has_loss'] (primary) or
        df['has_claim'] (sensitivity). See the note in data.clean: they are different
        events on freMTPL2 and only has_loss makes the hurdle decomposition exact."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        X = np.asarray(X, dtype=np.float32)
        has_claim = np.asarray(hurdle_y, dtype=int)
        total_loss = np.asarray(total_loss, dtype=float)
        exposure = np.asarray(exposure, dtype=float)

        Xe = _append_log_exposure(X, exposure)

        # --- stage 1: one model per context draw, ensembled at predict time ---
        self._stage1, self._odds_ratios = [], []
        pick = enriched_context_indices if self.sampler == "enriched" else random_context_indices
        for k in range(self.n_contexts):
            seed = self.seed * 1000 + k
            if self.sampler == "enriched":
                idx, odds = pick(has_claim, self.context_size, seed, self.neg_per_pos)
            else:
                idx, odds = pick(has_claim, self.context_size, seed)

            clf = self._make_classifier(seed)
            clf.fit(Xe[idx], has_claim[idx])
            self._stage1.append(clf)
            self._odds_ratios.append(odds)

        self.info.stage1_context_size = len(idx)
        self.info.sampling_odds_ratio = float(np.mean(self._odds_ratios))
        self.info.n_contexts = self.n_contexts

        # --- stage 2: claims-only, capped target, single deterministic context ---
        claim_mask = (has_claim == 1) & (total_loss > 0)
        n_claims = int(claim_mask.sum())
        if n_claims < 2:
            # Too few claims to fit a severity model (happens at the smallest N in Track B).
            self._stage2 = None
            fallback = float(total_loss[claim_mask].mean()) if n_claims else 0.0
            self._severity_cap = fallback
            self._severity_floor = fallback
        else:
            y2 = total_loss[claim_mask]
            if self.cap_mode == "fixed":
                if self.fixed_cap is None:
                    raise ValueError("cap_mode='fixed' requires fixed_cap")
                self._severity_cap = float(self.fixed_cap)
            else:
                self._severity_cap = float(np.quantile(y2, self.cap_quantile))
            # Symmetric floor to the 99.5% cap. TabICL's regression head emits an
            # unconstrained average of 999 quantiles and CAN return 0 -- observed at
            # N=500, where a single S_hat=0 against a positive actual loss sent mean
            # Tweedie(1.5) deviance to 3.9e6 via its mu^(-0.5) term. A Gamma GLM/XGBoost
            # log-link cannot do this, so the floor equalises a structural difference in
            # the heads rather than tuning TabICL's accuracy. A loss that occurred is at
            # least as large as the smallest loss the model was trained on.
            if self.cap_mode == "fixed" and self.fixed_floor is not None:
                self._severity_floor = float(self.fixed_floor)
            else:
                self._severity_floor = float(np.quantile(y2, 1.0 - self.cap_quantile))
            y2_capped = np.clip(y2, self._severity_floor, self._severity_cap)

            X2 = Xe[claim_mask]
            if len(X2) > self.stage2_context_size:
                rng = np.random.default_rng(self.seed)
                sub = np.sort(rng.choice(len(X2), size=self.stage2_context_size, replace=False))
                X2, y2_capped = X2[sub], y2_capped[sub]

            reg = self._make_regressor(self.seed)
            reg.fit(X2, y2_capped)
            self._stage2 = reg
            self.info.stage2_context_size = len(X2)

        self.info.severity_cap = self._severity_cap
        self.info.fit_seconds = time.perf_counter() - t0
        if torch.cuda.is_available():
            self.info.peak_vram_mib = torch.cuda.max_memory_allocated() / 2**20
        return self

    def predict_claim_proba(self, X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
        """Prior-corrected P(ClaimNb >= 1), averaged over context draws."""
        Xe = _append_log_exposure(X, exposure)
        acc = np.zeros(len(Xe), dtype=float)
        for clf, odds in zip(self._stage1, self._odds_ratios):
            p = np.empty(len(Xe), dtype=float)
            for s in range(0, len(Xe), self.predict_chunk_size):
                e = min(s + self.predict_chunk_size, len(Xe))
                p[s:e] = clf.predict_proba(Xe[s:e])[:, 1]
            acc += correct_prior(p, odds)
        return acc / max(len(self._stage1), 1)

    def predict_severity(self, X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
        """E[total loss | claim]. Falls back to the training mean if stage 2 could not fit."""
        Xe = _append_log_exposure(X, exposure)
        if self._stage2 is None:
            return np.full(len(Xe), max(self._severity_cap, _EPS), dtype=float)
        out = np.empty(len(Xe), dtype=float)
        for s in range(0, len(Xe), self.predict_chunk_size):
            e = min(s + self.predict_chunk_size, len(Xe))
            out[s:e] = np.asarray(self._stage2.predict(Xe[s:e], output_type="mean"), dtype=float)
        # Constrain to the range the model was actually trained on. Predictions above the
        # cap or below the floor are extrapolation the capped target gives no support for.
        lo, hi = max(self._severity_floor, _EPS), self._severity_cap
        # Track how often the clip binds: a high rate means the reported severities are
        # the constraint speaking, not the model.
        self.info.extra["severity_clipped_frac"] = float(
            np.mean((out <= lo) | (out >= hi))
        )
        return np.clip(out, lo, hi)

    def predict(self, X: np.ndarray, exposure: np.ndarray) -> np.ndarray:
        """Pure premium = p * S / exposure."""
        t0 = time.perf_counter()
        exposure = np.clip(np.asarray(exposure, dtype=float), _EPS, None)
        p = self.predict_claim_proba(X, exposure)
        # Cache it: callers want p_hat for the stage-1 diagnostics, and recomputing it
        # meant a second full ICL pass over the test set for no new information.
        self._last_p_hat = p
        s = self.predict_severity(X, exposure)
        self.info.predict_seconds = time.perf_counter() - t0
        if torch.cuda.is_available():
            self.info.peak_vram_mib = max(
                self.info.peak_vram_mib, torch.cuda.max_memory_allocated() / 2**20
            )
        return np.clip(p * s / exposure, _EPS, None)


class FinetunedTabICLTwoStage(TabICLTwoStage):
    """Track C: same structure, but both stages are fine-tuned before inference.

    Two defaults in the library bite here and are overridden explicitly:
      * `max_data_size=10000` silently truncates the fine-tuning set -- it must be raised
        or the 10k+ rungs of the ladder are not actually testing what they claim to;
      * `eval_metric` offers only roc_auc/log_loss/accuracy and mse/mae/r2, none of which
        is the pricing objective. We use log_loss for stage 1 and mse for stage 2, and
        select the final model on exposure-weighted Tweedie deviance externally.
    """

    def __init__(
        self,
        *args,
        finetune_size: int = 10_000,
        epochs: int = 30,
        learning_rate: float = 1e-5,
        n_estimators_finetune: int = 2,
        patience: int = 8,
        validation_split_ratio: float = 0.1,
        freeze_col: bool = False,
        freeze_row: bool = False,
        time_limit: float | None = None,
        output_dir: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.finetune_size = finetune_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_estimators_finetune = n_estimators_finetune
        self.patience = patience
        self.validation_split_ratio = validation_split_ratio
        self.freeze_col = freeze_col
        self.freeze_row = freeze_row
        self.time_limit = time_limit
        self.output_dir = output_dir

    def _common_ft_kwargs(self, seed: int) -> dict:
        return dict(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            n_estimators_finetune=self.n_estimators_finetune,
            n_estimators_inference=self.n_estimators,
            max_data_size=max(self.finetune_size, self.context_size),
            validation_split_ratio=self.validation_split_ratio,
            early_stopping=True,
            patience=self.patience,
            amp=True,
            freeze_col=self.freeze_col,
            freeze_row=self.freeze_row,
            device=self.device,
            random_state=seed,
            verbose=self.verbose,
            time_limit=self.time_limit,
        )

    def _make_classifier(self, seed: int):
        from tabicl import FinetunedTabICLClassifier

        return FinetunedTabICLClassifier(eval_metric="log_loss", **self._common_ft_kwargs(seed))

    def _make_regressor(self, seed: int):
        from tabicl import FinetunedTabICLRegressor

        return FinetunedTabICLRegressor(eval_metric="mse", **self._common_ft_kwargs(seed))


class _EpochCounter(logging.Handler):
    """Counts fine-tuning epochs by watching the library's per-epoch log line.

    `early_stopping=True` with `patience=8` means a run asking for 30 epochs may stop far
    short of it, and nothing on the estimator records how many actually ran (`global_step`
    is a local in `_finetune/base.py`). Without this, a null result cannot be told apart
    from a fine-tune that halted after three steps.
    """

    _RE = re.compile(r"epoch (\d+)/(\d+) \|.*?time=([0-9.]+)s")

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.epochs: list[tuple[int, float]] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = self._RE.search(record.getMessage())
        if m:
            self.epochs.append((int(m.group(1)), float(m.group(3))))


class _EncodedEstimator:
    """Applies the fine-tuning encoder, then delegates.

    `FinetunedTabICL*.fit` fits a feature encoder and treats it as authoritative for the
    lifetime of the model (base.py:1113-1116). Once we drive the inner estimator ourselves
    we have to honour that, or context and test rows get encoded differently. Wrapping it
    here keeps the parent's predict paths able to pass raw feature arrays.
    """

    def __init__(self, inner, encoder) -> None:
        self._inner = inner
        self._enc = encoder

    def _encode(self, X: np.ndarray) -> np.ndarray:
        if self._enc is None:
            return np.asarray(X, dtype=np.float64)
        return np.asarray(self._enc.transform(X), dtype=np.float64)

    def predict_proba(self, X):
        return self._inner.predict_proba(self._encode(X))

    def predict(self, X, **kw):
        return self._inner.predict(self._encode(X), **kw)


class DisjointFinetunedTabICLTwoStage(FinetunedTabICLTwoStage):
    """Fine-tune the weights on one sample, then run ICL on a DISJOINT context sample.

    The stock estimators fuse the two: `.fit(X, y)` fine-tunes on (X, y) and then installs
    that same (X, y) as the in-context example set (base.py:1119-1121). Any measured gain
    then confounds two different mechanisms -- "the weights learned this distribution" and
    "these particular rows were also visible in context" -- and the second is not what
    Track C is asking about.

    Here the fine-tuning rows and the context rows are disjoint by construction, so the
    contrast against a plain `TabICLTwoStage` on the same context isolates the weight
    update alone.

    `fit` keeps the parent signature for the CONTEXT data and takes the fine-tuning data as
    separate keyword arguments, so the runner can hand it two independent draws.
    """

    def _finetune_then_recontext(self, ft_est, X_ft, y_ft, X_ctx, y_ctx, tag: str = ""):
        """Fine-tune on (X_ft, y_ft); return an estimator holding those weights but
        conditioned on (X_ctx, y_ctx) as its in-context examples."""
        counter = _EpochCounter()
        lib_log = logging.getLogger("tabicl")
        lib_log.addHandler(counter)
        prev_level = lib_log.level
        lib_log.setLevel(logging.INFO)
        try:
            ft_est.fit(X_ft, y_ft)
        finally:
            lib_log.removeHandler(counter)
            lib_log.setLevel(prev_level)

        # Record what actually happened, not what was requested: `epochs` is an upper
        # bound once early stopping is on.
        self.info.extra[f"epochs_run_{tag}"] = len(counter.epochs)
        self.info.extra[f"epochs_requested_{tag}"] = int(self.epochs)
        self.info.extra[f"early_stopped_{tag}"] = int(len(counter.epochs) < int(self.epochs))
        if counter.epochs:
            self.info.extra[f"mean_epoch_s_{tag}"] = float(
                np.mean([t for _, t in counter.epochs])
            )
        self.info.extra[f"best_val_metric_{tag}"] = float(
            getattr(ft_est, "_best_metric_", float("nan"))
        )

        ft_est.model_.eval()
        device = torch.device(self.device)
        inner = ft_est._build_inner_estimator(
            ft_est.model_, ft_est.n_estimators_inference, device
        )
        encoder = getattr(ft_est, "_X_encoder_", None)
        X_ctx_enc = (
            np.asarray(encoder.transform(X_ctx), dtype=np.float64)
            if encoder is not None
            else np.asarray(X_ctx, dtype=np.float64)
        )
        # This .fit only installs the context -- _build_inner_estimator neutralises
        # _load_model, so the fine-tuned weights are kept rather than reloaded.
        inner.fit(X_ctx_enc, y_ctx)
        return _EncodedEstimator(inner, encoder)

    def fit(
        self,
        X: np.ndarray,
        hurdle_y: np.ndarray,
        total_loss: np.ndarray,
        exposure: np.ndarray,
        X_ft: np.ndarray | None = None,
        hurdle_ft: np.ndarray | None = None,
        total_loss_ft: np.ndarray | None = None,
        exposure_ft: np.ndarray | None = None,
    ) -> "DisjointFinetunedTabICLTwoStage":
        if X_ft is None:
            raise ValueError(
                "DisjointFinetunedTabICLTwoStage requires a separate fine-tuning set; "
                "pass X_ft/hurdle_ft/total_loss_ft/exposure_ft"
            )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        X = np.asarray(X, dtype=np.float32)
        has_claim = np.asarray(hurdle_y, dtype=int)
        total_loss = np.asarray(total_loss, dtype=float)
        exposure = np.asarray(exposure, dtype=float)
        Xe = _append_log_exposure(X, exposure)

        has_claim_ft = np.asarray(hurdle_ft, dtype=int)
        loss_ft = np.asarray(total_loss_ft, dtype=float)
        Xe_ft = _append_log_exposure(
            np.asarray(X_ft, dtype=np.float32), np.asarray(exposure_ft, dtype=float)
        )

        # --- stage 1 ---
        self._stage1, self._odds_ratios = [], []
        pick = enriched_context_indices if self.sampler == "enriched" else random_context_indices
        for k in range(self.n_contexts):
            seed = self.seed * 1000 + k
            if self.sampler == "enriched":
                idx, odds = pick(has_claim, self.context_size, seed, self.neg_per_pos)
            else:
                idx, odds = pick(has_claim, self.context_size, seed)
            est = self._finetune_then_recontext(
                self._make_classifier(seed), Xe_ft, has_claim_ft, Xe[idx], has_claim[idx],
                tag="stage1",
            )
            self._stage1.append(est)
            # The prior correction belongs to the CONTEXT composition -- that is what sets
            # the predicted odds -- not to how the fine-tuning set was balanced.
            self._odds_ratios.append(odds)

        self.info.stage1_context_size = len(idx)
        self.info.sampling_odds_ratio = float(np.mean(self._odds_ratios))
        self.info.n_contexts = self.n_contexts
        self.info.extra["finetune_size_stage1"] = int(len(Xe_ft))

        # --- stage 2: claims-only on both sides, capped with the SAME cap ---
        ctx_mask = (has_claim == 1) & (total_loss > 0)
        ft_mask = (has_claim_ft == 1) & (loss_ft > 0)
        if int(ctx_mask.sum()) < 2 or int(ft_mask.sum()) < 2:
            self._stage2 = None
            fallback = float(total_loss[ctx_mask].mean()) if ctx_mask.sum() else 0.0
            self._severity_cap = self._severity_floor = fallback
        else:
            y2 = total_loss[ctx_mask]
            if self.cap_mode == "fixed":
                if self.fixed_cap is None:
                    raise ValueError("cap_mode='fixed' requires fixed_cap")
                self._severity_cap = float(self.fixed_cap)
                self._severity_floor = float(
                    self.fixed_floor
                    if self.fixed_floor is not None
                    else np.quantile(y2, 1.0 - self.cap_quantile)
                )
            else:
                self._severity_cap = float(np.quantile(y2, self.cap_quantile))
                self._severity_floor = float(np.quantile(y2, 1.0 - self.cap_quantile))

            lo, hi = self._severity_floor, self._severity_cap
            X2_ctx = Xe[ctx_mask]
            y2_ctx = np.clip(y2, lo, hi)
            if len(X2_ctx) > self.stage2_context_size:
                rng = np.random.default_rng(self.seed)
                sub = np.sort(rng.choice(len(X2_ctx), size=self.stage2_context_size, replace=False))
                X2_ctx, y2_ctx = X2_ctx[sub], y2_ctx[sub]

            self._stage2 = self._finetune_then_recontext(
                self._make_regressor(self.seed),
                Xe_ft[ft_mask],
                np.clip(loss_ft[ft_mask], lo, hi),
                X2_ctx,
                y2_ctx,
                tag="stage2",
            )
            self.info.stage2_context_size = len(X2_ctx)
            self.info.extra["finetune_size_stage2"] = int(ft_mask.sum())

        self.info.severity_cap = self._severity_cap
        self.info.fit_seconds = time.perf_counter() - t0
        if torch.cuda.is_available():
            self.info.peak_vram_mib = torch.cuda.max_memory_allocated() / 2**20
        return self
