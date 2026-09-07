"""XGBoost baselines, ported from oruppelt/TFM_real_benchmark.

Correction to the reference implementation: it passes exposure as `sample_weight` for the
Poisson frequency model and notes this is "less theoretically rigorous than a GLM offset".
For count:poisson it is genuinely not equivalent -- weighting rescales each row's loss
contribution, an offset shifts its predicted log-mean. XGBoost supports a true offset via
`base_margin = log(exposure)`, so that is the default here, with the weight variant kept
as an explicit sensitivity arm.

Handicapping the incumbent would make any TabICL win contestable in review, which is why
this matters more than it looks.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

_EPS = 1e-10

DEFAULT_PARAMS = dict(
    max_depth=6,
    learning_rate=0.05,
    n_estimators=400,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1e-6,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=0,
    tree_method="hist",
)

# Reference repo's Optuna search space.
SEARCH_SPACE = {
    "max_depth": ("int", 3, 10),
    "learning_rate": ("logfloat", 0.01, 0.3),
    "n_estimators": ("int", 100, 1000),
    "min_child_weight": ("int", 1, 100),
    "subsample": ("float", 0.6, 1.0),
    "colsample_bytree": ("float", 0.6, 1.0),
    "reg_alpha": ("logfloat", 1e-8, 10.0),
    "reg_lambda": ("logfloat", 1e-8, 10.0),
}


def sample_params(trial) -> dict:
    """Draw one Optuna trial from the reference search space."""
    out = {}
    for name, spec in SEARCH_SPACE.items():
        kind, lo, hi = spec
        if kind == "int":
            out[name] = trial.suggest_int(name, lo, hi)
        elif kind == "float":
            out[name] = trial.suggest_float(name, lo, hi)
        else:
            out[name] = trial.suggest_float(name, lo, hi, log=True)
    return out


class XGBPoisson:
    """Frequency. Exposure as a true offset via base_margin (default) or sample_weight."""

    def __init__(self, exposure_mode: str = "base_margin", **params) -> None:
        if exposure_mode not in {"base_margin", "sample_weight"}:
            raise ValueError(exposure_mode)
        self.exposure_mode = exposure_mode
        self.params = {**DEFAULT_PARAMS, **params}
        self.model = xgb.XGBRegressor(objective="count:poisson", **self.params)

    def fit(self, X, y, exposure=None):
        e = np.clip(np.asarray(exposure, dtype=float), _EPS, None)
        if self.exposure_mode == "base_margin":
            # log-link model, so a log(exposure) margin is exactly the GLM offset.
            self.model.fit(X, np.asarray(y, dtype=float), base_margin=np.log(e))
        else:
            self.model.fit(X, np.asarray(y, dtype=float) / e, sample_weight=e)
        return self

    def predict(self, X, exposure=None):
        """Returns the RATE, so it composes with severity the same way the GLM does."""
        if self.exposure_mode == "base_margin":
            zero = np.zeros(len(X))
            return np.clip(self.model.predict(X, base_margin=zero), _EPS, None)
        return np.clip(self.model.predict(X), _EPS, None)


class XGBGamma:
    def __init__(self, **params) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self.model = xgb.XGBRegressor(objective="reg:gamma", **self.params)

    def fit(self, X, y, exposure=None, weights=None):
        self.model.fit(X, np.clip(np.asarray(y, dtype=float), _EPS, None), sample_weight=weights)
        return self

    def predict(self, X, exposure=None):
        return np.clip(self.model.predict(X), _EPS, None)


class XGBTweedie:
    def __init__(self, var_power: float = 1.5, **params) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self.model = xgb.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=var_power, **self.params
        )

    def fit(self, X, y, exposure=None):
        w = None if exposure is None else np.asarray(exposure, dtype=float)
        self.model.fit(X, np.asarray(y, dtype=float), sample_weight=w)
        return self

    def predict(self, X, exposure=None):
        return np.clip(self.model.predict(X), _EPS, None)


class XGBBinary:
    """Hurdle stage 1. log(exposure) is appended as a FEATURE, mirroring TabICL stage 1."""

    def __init__(self, **params) -> None:
        self.params = {**DEFAULT_PARAMS, **params}
        self.model = xgb.XGBClassifier(objective="binary:logistic", **self.params)

    @staticmethod
    def _augment(X, exposure):
        e = np.clip(np.asarray(exposure, dtype=float), _EPS, None).reshape(-1, 1)
        return np.hstack([np.asarray(X, dtype=np.float32), np.log(e).astype(np.float32)])

    def fit(self, X, y, exposure=None):
        self.model.fit(self._augment(X, exposure), np.asarray(y, dtype=int))
        return self

    def predict(self, X, exposure=None):
        return np.clip(self.model.predict_proba(self._augment(X, exposure))[:, 1], _EPS, 1 - _EPS)


class XGBHurdle:
    """binary:logistic x reg:gamma -- the framing-matched control for TabICL two-stage."""

    def __init__(self, cap_quantile: float = 0.995, **params) -> None:
        self.cap_quantile = cap_quantile
        self.stage1 = XGBBinary(**params)
        self.stage2 = XGBGamma(**params)
        self.severity_cap_ = float("nan")

    def fit(self, X, hurdle_y, total_loss, exposure):
        X = np.asarray(X, dtype=np.float32)
        hurdle_y = np.asarray(hurdle_y, dtype=int)
        total_loss = np.asarray(total_loss, dtype=float)

        self.stage1.fit(X, hurdle_y, exposure=exposure)

        mask = (hurdle_y == 1) & (total_loss > 0)
        y2 = total_loss[mask]
        self.severity_cap_ = float(np.quantile(y2, self.cap_quantile)) if len(y2) else float("nan")
        self.stage2.fit(X[mask], np.minimum(y2, self.severity_cap_))
        return self

    def predict(self, X, exposure):
        exposure = np.clip(np.asarray(exposure, dtype=float), _EPS, None)
        p = self.stage1.predict(X, exposure)
        s = self.stage2.predict(X)
        return np.clip(p * s / exposure, _EPS, None)


class XGBFreqSev:
    """Classical Poisson x Gamma."""

    def __init__(self, cap_quantile: float = 0.995, exposure_mode: str = "base_margin", **params):
        self.cap_quantile = cap_quantile
        self.freq = XGBPoisson(exposure_mode=exposure_mode, **params)
        self.sev = XGBGamma(**params)
        self.severity_cap_ = float("nan")

    def fit(self, X, claim_nb, severity, exposure, claims_mask):
        X = np.asarray(X, dtype=np.float32)
        self.freq.fit(X, np.asarray(claim_nb, dtype=float), exposure=exposure)
        y2 = np.asarray(severity, dtype=float)[claims_mask]
        self.severity_cap_ = float(np.quantile(y2, self.cap_quantile)) if len(y2) else float("nan")
        self.sev.fit(
            X[claims_mask],
            np.minimum(y2, self.severity_cap_),
            weights=np.asarray(claim_nb, dtype=float)[claims_mask],
        )
        return self

    def predict(self, X, exposure=None):
        return np.clip(self.freq.predict(X, exposure) * self.sev.predict(X), _EPS, None)
