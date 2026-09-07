"""Is the Tweedie GLM baseline actually converging? Its learning curve is non-monotonic."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import metrics as M
from src.data import build_and_cache, exposure_stratified_subsample, random_split
from src.features.insurance_features import EngineeredFeaturePipeline
from src.models.glm import TweedieGLM

df, _ = build_and_cache()
pool, test = random_split(df)
y_te = test["PurePremium"].to_numpy()
e_te = test["Exposure"].to_numpy()

print(f"{'N':>8} {'converged':>10} {'iters':>6} {'wGini':>8} {'dev1.5':>9} {'max|coef|':>10}")
print("-" * 60)
for n in [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 542_410]:
    train = exposure_stratified_subsample(pool, n, seed=0)
    pipe = EngineeredFeaturePipeline()
    X_tr = pipe.fit_transform(train, train["PurePremium"].to_numpy())
    X_te = pipe.transform(test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TweedieGLM().fit(X_tr, train["PurePremium"].to_numpy(),
                             exposure=train["Exposure"].to_numpy())
    pred = m.predict(X_te)
    res = m.result_
    iters = getattr(res, "fit_history", {}).get("iteration", [np.nan])
    iters = iters[-1] if hasattr(iters, "__len__") and len(iters) else np.nan
    mm = M.evaluate_pure_premium(y_te, pred, e_te)
    print(
        f"{n:>8,} {str(m.converged_):>10} {str(iters):>6} "
        f"{mm['gini_exposure_weighted']:>8.4f} {mm['tweedie_deviance_1.5']:>9.2f} "
        f"{np.abs(res.params).max():>10.3f}"
    )
