"""GLM baselines, ported from oruppelt/TFM_real_benchmark (statsmodels, IRLS).

Exposure enters differently by family, and the differences are deliberate:
  * Poisson  -- log-OFFSET on raw counts. Encodes E[N] proportional to exposure structurally.
  * Gamma    -- no exposure; severity per claim does not scale with time on risk.
  * Tweedie  -- no offset; pure premium is already a rate. Exposure enters as var_weights.
  * Binomial -- no offset (there is none for a logit); log(exposure) is added as a FEATURE
                so the hurdle-matched baseline is handicapped exactly like TabICL stage 1.

Features are standardised on train statistics before fitting: IRLS on raw freMTPL2 columns
(Density spans 5 orders of magnitude) converges poorly otherwise.
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

_EPS = 1e-10


class _BaseGLM:
    family = None

    def __init__(self, maxiter: int = 100, add_log_exposure_feature: bool = False) -> None:
        self.maxiter = maxiter
        self.add_log_exposure_feature = add_log_exposure_feature
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self.result_ = None
        self.converged_ = False

    def _design(self, X: np.ndarray, exposure: np.ndarray | None, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.add_log_exposure_feature:
            if exposure is None:
                raise ValueError("exposure required when add_log_exposure_feature=True")
            log_e = np.log(np.clip(np.asarray(exposure, dtype=float), _EPS, None)).reshape(-1, 1)
            X = np.hstack([X, log_e])
        if fit:
            self._mu = X.mean(axis=0)
            self._sd = X.std(axis=0)
            self._sd[self._sd < _EPS] = 1.0
        Z = (X - self._mu) / self._sd
        return sm.add_constant(Z, has_constant="add")

    def _fit_statsmodels(self, Z, y, **kw):
        model = sm.GLM(y, Z, family=self.family, **kw)
        self.result_ = model.fit(method="irls", maxiter=self.maxiter)
        self.converged_ = bool(getattr(self.result_, "converged", True))
        return self


class PoissonGLM(_BaseGLM):
    """Frequency: raw counts with a log-exposure offset."""

    family = sm.families.Poisson(link=sm.families.links.Log())

    def fit(self, X, y, exposure=None):
        Z = self._design(X, exposure, fit=True)
        offset = np.log(np.clip(np.asarray(exposure, dtype=float), _EPS, None))
        return self._fit_statsmodels(Z, np.asarray(y, dtype=float), offset=offset)

    def predict(self, X, exposure=None):
        """Returns the RATE (claims per unit exposure), not the count."""
        Z = self._design(X, exposure, fit=False)
        zero_offset = np.zeros(len(Z))
        return np.clip(self.result_.predict(Z, offset=zero_offset), _EPS, None)


class GammaGLM(_BaseGLM):
    """Severity per claim, weighted by claim count."""

    family = sm.families.Gamma(link=sm.families.links.Log())

    def fit(self, X, y, exposure=None, weights=None):
        Z = self._design(X, None, fit=True)
        y = np.clip(np.asarray(y, dtype=float), _EPS, None)
        kw = {}
        if weights is not None:
            kw["var_weights"] = np.asarray(weights, dtype=float)
        return self._fit_statsmodels(Z, y, **kw)

    def predict(self, X, exposure=None):
        Z = self._design(X, None, fit=False)
        return np.clip(self.result_.predict(Z), _EPS, None)


class TweedieGLM(_BaseGLM):
    """Direct pure premium. No offset -- the target is already a rate."""

    def __init__(self, var_power: float = 1.5, maxiter: int = 100) -> None:
        super().__init__(maxiter=maxiter)
        self.var_power = var_power
        self.family = sm.families.Tweedie(
            link=sm.families.links.Log(), var_power=var_power, eql=True
        )

    def fit(self, X, y, exposure=None):
        Z = self._design(X, None, fit=True)
        kw = {}
        if exposure is not None:
            kw["var_weights"] = np.asarray(exposure, dtype=float)
        return self._fit_statsmodels(Z, np.asarray(y, dtype=float), **kw)

    def predict(self, X, exposure=None):
        Z = self._design(X, None, fit=False)
        return np.clip(self.result_.predict(Z), _EPS, None)


class BinomialGLM(_BaseGLM):
    """Hurdle-matched stage 1: P(loss > 0). log(exposure) is a feature, mirroring TabICL."""

    family = sm.families.Binomial(link=sm.families.links.Logit())

    def __init__(self, maxiter: int = 100) -> None:
        super().__init__(maxiter=maxiter, add_log_exposure_feature=True)

    def fit(self, X, y, exposure=None):
        Z = self._design(X, exposure, fit=True)
        return self._fit_statsmodels(Z, np.asarray(y, dtype=float))

    def predict(self, X, exposure=None):
        Z = self._design(X, exposure, fit=False)
        return np.clip(self.result_.predict(Z), _EPS, 1 - _EPS)


class GLMHurdle:
    """Binomial x Gamma, structurally identical to the TabICL two-stage model.

    This is the control that separates "TabICL is better" from "the hurdle framing is
    better" -- run 1 -> run 2 changed both at once, so without this the comparison is
    confounded.
    """

    def __init__(self, cap_quantile: float = 0.995) -> None:
        self.cap_quantile = cap_quantile
        self.stage1 = BinomialGLM()
        self.stage2 = GammaGLM()
        self.severity_cap_ = float("nan")

    def fit(self, X, hurdle_y, total_loss, exposure):
        X = np.asarray(X, dtype=float)
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


class GLMFreqSev:
    """Classical Poisson x Gamma frequency-severity."""

    def __init__(self, cap_quantile: float = 0.995) -> None:
        self.cap_quantile = cap_quantile
        self.freq = PoissonGLM()
        self.sev = GammaGLM()
        self.severity_cap_ = float("nan")

    def fit(self, X, claim_nb, severity, exposure, claims_mask):
        X = np.asarray(X, dtype=float)
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
        """Pure premium = frequency rate x severity per claim."""
        return np.clip(self.freq.predict(X, exposure) * self.sev.predict(X), _EPS, None)


class InterceptOnly:
    """Pipeline sanity floor. Must score Gini ~ 0; if it does not, scoring is broken."""

    def __init__(self) -> None:
        self.value_ = 0.0

    def fit(self, X, y, exposure=None):
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if exposure is None else np.asarray(exposure, dtype=float)
        self.value_ = float((y * w).sum() / w.sum())
        return self

    def predict(self, X, exposure=None):
        return np.full(len(X), max(self.value_, _EPS), dtype=float)


def balance_to_portfolio(
    pred: np.ndarray, y_true: np.ndarray, exposure: np.ndarray
) -> tuple[np.ndarray, float]:
    """Rescale so sum(pred * w) == sum(actual * w). Returns (balanced, factor).

    Computed on TRAIN and applied to test -- computing it on test would leak the answer.
    """
    w = np.asarray(exposure, dtype=float)
    num = (np.asarray(y_true, dtype=float) * w).sum()
    den = (np.asarray(pred, dtype=float) * w).sum()
    factor = float(num / den) if den > 0 else 1.0
    return np.asarray(pred, dtype=float) * factor, factor
