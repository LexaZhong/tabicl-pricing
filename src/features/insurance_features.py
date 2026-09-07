"""Feature pipelines ported from oruppelt/TFM_real_benchmark.

Two pipelines, matching the reference repo:
  * GBMFeaturePipeline       -- continuous/derived features for XGBoost (and TabICL, Track D)
  * EngineeredFeaturePipeline -- binned features + interactions for the GLM

Exposure is deliberately NOT a feature in either pipeline. It is passed separately to
fit/predict so each model family can use it correctly (log-offset for Poisson GLM,
base_margin for XGBoost Poisson, a plain feature for the TabICL stage-1 classifier).

LEAKAGE: TargetEncoder must be fit on training rows only. In Track A (region censoring)
fitting it on all data leaks the censored region's own target and invalidates the track.
`fit_transform` is the only method that touches y; `transform` never does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

# Area is an ordered density band in freMTPL2, so map it to an integer rather than
# one-hot: the ordering carries real signal.
_AREA_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

_TE_COLS = ["Region", "VehBrand"]

# Bin edges for the GLM pipeline (reference repo values).
_BM_BINS = [-np.inf, 60, 80, 100, 150, np.inf]
_POW_BINS = [-np.inf, 6, 9, 12, np.inf]
_AGE_BINS = [-np.inf, 25, 35, 45, 55, 65, np.inf]
_VEHAGE_BINS = [-np.inf, 1, 5, 10, 15, np.inf]

_VEHAGE_CAP = 20


class _BasePipeline:
    """Shared target-encoding + fit/transform plumbing."""

    def __init__(self) -> None:
        self._target_encoders: dict[str, TargetEncoder] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    # --- subclass hook ---
    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _fit_target_encoders(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self._target_encoders = {}
        for col in _TE_COLS:
            enc = TargetEncoder(cv=5, smooth="auto", target_type="continuous", random_state=0)
            enc.fit(X[[col]].astype(str), y)
            self._target_encoders[col] = enc

    def _apply_target_encoders(self, X: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
        for col, enc in self._target_encoders.items():
            # Unseen categories (e.g. a censored region) fall back to the global target
            # mean -- the column becomes constant on that test set. That is intended and
            # is exactly what Track A measures.
            out[f"{col}_te"] = enc.transform(X[[col]].astype(str)).ravel()
        return out

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        self._fit_target_encoders(X, np.asarray(y, dtype=float))
        out = self._engineer(X)
        out = self._apply_target_encoders(X, out)
        self.feature_names_ = list(out.columns)
        self._fitted = True
        return out.to_numpy(dtype=np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit_transform before transform")
        out = self._engineer(X)
        out = self._apply_target_encoders(X, out)
        out = out[self.feature_names_]  # lock column order
        return out.to_numpy(dtype=np.float32)


class GBMFeaturePipeline(_BasePipeline):
    """Derived continuous features for XGBoost. Also used for TabICL in Track D."""

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        # Raw numerics preserved -- trees can find their own splits.
        out["VehPower"] = X["VehPower"].astype(float)
        out["VehAge"] = X["VehAge"].astype(float)
        out["DrivAge"] = X["DrivAge"].astype(float)
        out["BonusMalus"] = X["BonusMalus"].astype(float)

        # Log compression makes density proportional to the urban/rural gradient.
        out["log_density"] = np.log1p(X["Density"].astype(float))
        out["area_ordinal"] = X["Area"].astype(str).map(_AREA_MAP).astype(float)
        out["is_diesel"] = (X["VehGas"].astype(str) == "Diesel").astype(float)

        # Business features.
        out["bm_excess"] = X["BonusMalus"].astype(float) - 50.0
        out["is_malus"] = (X["BonusMalus"].astype(float) > 100).astype(float)
        young = (X["DrivAge"].astype(float) < 26).astype(float)
        out["young_driver"] = young
        out["senior_driver"] = (X["DrivAge"].astype(float) > 69).astype(float)
        out["young_x_power"] = young * X["VehPower"].astype(float)
        out["vehicle_value_proxy"] = X["VehPower"].astype(float) / (
            X["VehAge"].astype(float) + 1.0
        )
        return out


class EngineeredFeaturePipeline(_BasePipeline):
    """Binned features + interactions for the GLM (linear model needs the structure)."""

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        bm_bin = pd.cut(X["BonusMalus"].astype(float), _BM_BINS, labels=False).astype(float)
        pow_group = pd.cut(X["VehPower"].astype(float), _POW_BINS, labels=False).astype(float)
        age_band = pd.cut(X["DrivAge"].astype(float), _AGE_BINS, labels=False).astype(float)
        vehage_bin = pd.cut(
            X["VehAge"].astype(float).clip(upper=_VEHAGE_CAP), _VEHAGE_BINS, labels=False
        ).astype(float)

        out["bm_bin"] = bm_bin
        out["pow_group"] = pow_group
        out["age_band"] = age_band
        out["vehage_bin"] = vehage_bin

        out["log_density"] = np.log1p(X["Density"].astype(float))
        out["area_ordinal"] = X["Area"].astype(str).map(_AREA_MAP).astype(float)
        out["is_diesel"] = (X["VehGas"].astype(str) == "Diesel").astype(float)

        # Interactions the reference repo found useful.
        out["age_x_power"] = age_band * pow_group
        out["bm_x_age"] = bm_bin * age_band
        return out


class RawFeaturePipeline(_BasePipeline):
    """Track D control: raw cleaned columns, no actuarial feature engineering.

    Categoricals become integer codes fitted on train only, so an unseen level maps to -1
    rather than silently colliding with a seen level.
    """

    def __init__(self) -> None:
        super().__init__()
        self._categories: dict[str, list[str]] = {}

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]:
            out[col] = X[col].astype(float)
        for col in ["Area", "VehBrand", "VehGas", "Region"]:
            cats = self._categories.get(col)
            values = X[col].astype(str)
            if cats is None:
                cats = sorted(values.unique())
                self._categories[col] = cats
            lookup = {c: i for i, c in enumerate(cats)}
            out[col] = values.map(lookup).fillna(-1).astype(float)
        return out

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        self._categories = {}
        out = self._engineer(X)
        self.feature_names_ = list(out.columns)
        self._fitted = True
        return out.to_numpy(dtype=np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("call fit_transform before transform")
        out = self._engineer(X)[self.feature_names_]
        return out.to_numpy(dtype=np.float32)


PIPELINES = {
    "gbm": GBMFeaturePipeline,
    "glm": EngineeredFeaturePipeline,
    "raw": RawFeaturePipeline,
}
