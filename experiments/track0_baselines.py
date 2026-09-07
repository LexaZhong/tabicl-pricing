"""Track 0.2/0.4: metrics unit tests + GLM/XGBoost baseline reference numbers.

Runs on the fixed 542,410 / 135,603 split. These numbers are the bar TabICL has to clear,
and the metrics tests here are what stop a broken scorer from producing a confident answer.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics as M
from src.data import build_and_cache, random_split
from src.features.insurance_features import EngineeredFeaturePipeline, GBMFeaturePipeline
from src.models.gbm import XGBFreqSev, XGBHurdle, XGBTweedie
from src.models.glm import (
    GLMFreqSev,
    GLMHurdle,
    InterceptOnly,
    TweedieGLM,
    balance_to_portfolio,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def run_metric_tests() -> bool:
    print("=== metrics unit tests ===")
    rng = np.random.default_rng(0)
    n = 20_000
    exposure = rng.uniform(0.05, 1.0, n)
    lam = rng.gamma(2.0, 0.05, n)
    loss = rng.poisson(lam * exposure) * rng.gamma(2.0, 800, n)
    pp = loss / exposure

    tests = {}
    tests["perfect ranker gini == 1"] = abs(M.normalized_gini(pp, pp, exposure) - 1.0) < 1e-9

    # A single random permutation is noisy on a heavy tail; average several.
    rand_ginis = [M.gini(pp, rng.permutation(pp), exposure) for _ in range(20)]
    tests["random ranker gini ~ 0"] = abs(float(np.mean(rand_ginis))) < 0.02

    const = np.full(n, pp.mean())
    tests["constant pred gini == 0"] = abs(M.gini(pp, const, exposure)) < 1e-12
    tests["gini <= perfect"] = M.gini(pp, lam, exposure) <= M.gini(pp, pp, exposure) + 1e-12

    # Ties must not read row order as signal: shuffling rows cannot change the score.
    perm = rng.permutation(n)
    tests["gini order-invariant"] = abs(
        M.gini(pp, lam, exposure) - M.gini(pp[perm], lam[perm], exposure[perm])
    ) < 1e-9
    coarse = np.round(lam, 2)  # deliberately tie-heavy predictor
    tests["gini tie-invariant"] = abs(
        M.gini(pp, coarse, exposure) - M.gini(pp[perm], coarse[perm], exposure[perm])
    ) < 1e-9

    # Tweedie deviance at p=1 must equal Poisson deviance.
    d1 = M.tweedie_deviance(pp, const, exposure, power=1.0)
    d2 = M.poisson_deviance(pp, const, exposure)
    tests["tweedie(p=1) == poisson"] = abs(d1 - d2) < 1e-9

    # Calibration of a perfectly balanced predictor is 1.
    bal, _ = balance_to_portfolio(const, pp, exposure)
    tests["balanced calibration == 1"] = abs(M.calibration_ratio(pp, bal, exposure) - 1.0) < 1e-9

    # Hurdle -> Poisson inversion round-trips.
    lam_true = np.array([0.05, 0.2, 1.0])
    e = np.array([0.5, 1.0, 0.25])
    p = 1 - np.exp(-lam_true * e)
    tests["hurdle->frequency round-trip"] = np.allclose(M.hurdle_to_frequency(p, e), lam_true)

    # Lift table buckets must carry ~equal exposure.
    lt = M.lift_table(pp, lam, exposure, n_buckets=10)
    spread = lt["exposure"].max() / lt["exposure"].min()
    tests["lift buckets equal-exposure"] = spread < 1.05

    for name, ok in tests.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return all(tests.values())


def evaluate(name, pred_test, test, rows, extra=None):
    m = M.evaluate_pure_premium(
        test["PurePremium"].to_numpy(), pred_test, test["Exposure"].to_numpy()
    )
    if extra:
        m.update(extra)
    print(
        f"  {name:24s} dev1.5={m['tweedie_deviance_1.5']:8.2f}  "
        f"wGini={m['gini_exposure_weighted']:.4f}  "
        f"nGini={m['normalized_gini_exposure_weighted']:.4f}  "
        f"cal={m['calibration_ratio']:.3f}"
    )
    rows.extend([{"model": name, "metric": k, "value": v} for k, v in m.items()])
    return m


def main() -> int:
    ok = run_metric_tests()

    df, _ = build_and_cache()
    train, test = random_split(df)
    print(f"\ntrain={len(train):,}  test={len(test):,}")

    y_tr = train["PurePremium"].to_numpy()
    e_tr = train["Exposure"].to_numpy()
    e_te = test["Exposure"].to_numpy()

    # Target encoders are fit on TRAIN ONLY inside fit_transform.
    glm_pipe = EngineeredFeaturePipeline()
    Xg_tr = glm_pipe.fit_transform(train, y_tr)
    Xg_te = glm_pipe.transform(test)

    gbm_pipe = GBMFeaturePipeline()
    Xb_tr = gbm_pipe.fit_transform(train, y_tr)
    Xb_te = gbm_pipe.transform(test)
    print(f"glm features: {glm_pipe.feature_names_}")
    print(f"gbm features: {gbm_pipe.feature_names_}")

    rows: list[dict] = []
    print("\n=== baselines (test set) ===")

    t0 = time.perf_counter()
    floor = InterceptOnly().fit(Xg_tr, y_tr, e_tr)
    m = evaluate("intercept_only", floor.predict(Xg_te), test, rows)
    if abs(m["gini_exposure_weighted"]) > 1e-6:
        print("  [FAIL] intercept-only floor should score Gini ~ 0")
        ok = False
    else:
        print("  [PASS] intercept-only floor scores Gini ~ 0")

    tw = TweedieGLM(var_power=1.5).fit(Xg_tr, y_tr, exposure=e_tr)
    evaluate("glm_tweedie", tw.predict(Xg_te), test, rows)

    fs = GLMFreqSev().fit(
        Xg_tr,
        train["ClaimNb"].to_numpy(),
        train["Severity"].fillna(0).to_numpy(),
        e_tr,
        (train["has_loss"] == 1).to_numpy(),
    )
    evaluate("glm_freqsev", fs.predict(Xg_te, e_te), test, rows)

    hu = GLMHurdle().fit(
        Xg_tr, train["has_loss"].to_numpy(), train["TotalLoss"].to_numpy(), e_tr
    )
    evaluate("glm_hurdle", hu.predict(Xg_te, e_te), test, rows)
    print(f"  (GLMs took {time.perf_counter() - t0:.0f}s)")

    t0 = time.perf_counter()
    xt = XGBTweedie().fit(Xb_tr, y_tr, exposure=e_tr)
    evaluate("xgb_tweedie", xt.predict(Xb_te), test, rows)

    xfs = XGBFreqSev().fit(
        Xb_tr,
        train["ClaimNb"].to_numpy(),
        train["Severity"].fillna(0).to_numpy(),
        e_tr,
        (train["has_loss"] == 1).to_numpy(),
    )
    evaluate("xgb_freqsev", xfs.predict(Xb_te, e_te), test, rows)

    xh = XGBHurdle().fit(Xb_tr, train["has_loss"].to_numpy(), train["TotalLoss"].to_numpy(), e_tr)
    xh_pred = xh.predict(Xb_te, e_te)
    evaluate("xgb_hurdle", xh_pred, test, rows)
    print(f"  (XGBs took {time.perf_counter() - t0:.0f}s)")

    print("\n=== lift chart: xgb_hurdle (10 equal-exposure buckets) ===")
    lt = M.lift_table(test["PurePremium"].to_numpy(), xh_pred, e_te)
    print(lt.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n=== stage-1 diagnostics: xgb_hurdle ===")
    p_hat = xh.stage1.predict(Xb_te, e_te)
    s1 = M.stage1_metrics(test["has_loss"].to_numpy(), p_hat)
    for k, v in s1.items():
        print(f"  {k:24s} {v:.5f}")
    mono = M.exposure_monotonicity(p_hat, e_te)
    for k, v in mono.items():
        print(f"  {k:24s} {v:.5f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "track0_baselines.csv", index=False)
    lt.to_csv(RESULTS / "track0_lift_xgb_hurdle.csv", index=False)
    print(f"\nwrote {RESULTS / 'track0_baselines.csv'}")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
