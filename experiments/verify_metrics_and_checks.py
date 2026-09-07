"""Verification gate: metric correctness + the checks module can re-detect the real bug.

Runs before any GPU time. Two parts:
  1. Metric unit tests, including the new artifact-free metrics.
  2. Replay of the observed failure -- the collapsed `xgb_hurdle` at N<=5,000 -- against
     both the synthetic degenerate predictor and the actual cached results. A check suite
     that cannot re-detect the bug that motivated it is not verified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import checks as C
from src import metrics as M
from src.data import build_and_cache, random_split
from src.runner import CACHE

PASS, FAIL = "PASS", "FAIL"


def _report(results: dict[str, bool]) -> bool:
    for name, ok in results.items():
        print(f"  [{PASS if ok else FAIL}] {name}")
    return all(results.values())


def metric_tests() -> bool:
    print("=== metric unit tests ===")
    rng = np.random.default_rng(0)
    n = 20_000
    exposure = rng.uniform(0.05, 1.0, n)
    lam = rng.gamma(2.0, 0.05, n)
    loss = rng.poisson(lam * exposure) * rng.gamma(2.0, 800, n)
    pp = loss / exposure

    t: dict[str, bool] = {}
    t["perfect ranker normalized gini == 1"] = abs(M.normalized_gini(pp, pp, exposure) - 1) < 1e-9
    t["random ranker gini ~ 0"] = abs(
        float(np.mean([M.gini(pp, rng.permutation(pp), exposure) for _ in range(20)]))
    ) < 0.02
    const = np.full(n, pp.mean())
    t["constant pred gini == 0"] = abs(M.gini(pp, const, exposure)) < 1e-12
    t["gini <= perfect"] = M.gini(pp, lam, exposure) <= M.gini(pp, pp, exposure) + 1e-12

    perm = rng.permutation(n)
    t["gini order-invariant"] = abs(
        M.gini(pp, lam, exposure) - M.gini(pp[perm], lam[perm], exposure[perm])
    ) < 1e-9
    coarse = np.round(lam, 2)
    t["gini tie-invariant"] = abs(
        M.gini(pp, coarse, exposure) - M.gini(pp[perm], coarse[perm], exposure[perm])
    ) < 1e-9

    t["tweedie(p=1) == poisson"] = abs(
        M.tweedie_deviance(pp, const, exposure, 1.0) - M.poisson_deviance(pp, const, exposure)
    ) < 1e-9
    lam_true, e3 = np.array([0.05, 0.2, 1.0]), np.array([0.5, 1.0, 0.25])
    t["hurdle->frequency round-trip"] = np.allclose(
        M.hurdle_to_frequency(1 - np.exp(-lam_true * e3), e3), lam_true
    )
    lt = M.lift_table(pp, lam, exposure, 10)
    t["lift buckets equal-exposure"] = (lt["exposure"].max() / lt["exposure"].min()) < 1.05

    # --- the new artifact-free metrics, on the REAL test set ---
    df, _ = build_and_cache()
    _, test = random_split(df)
    y = test["PurePremium"].to_numpy()
    e = test["Exposure"].to_numpy()
    trivial = (test["TotalLoss"].sum() / e.sum()) * e.mean() / e
    tm = M.evaluate_pure_premium(y, trivial, e)

    print(f"\n  trivial c/e model on the real test set:")
    for k in ["gini_exposure_weighted", "gini_total_loss", "gini_fixed_exposure",
              "tweedie_deviance_1.5"]:
        if k in tm:
            print(f"    {k:32s} {tm[k]:>12.6f}")

    # The tie-tolerance fix: total-loss Gini must now be exactly 0, not 0.0032.
    t["trivial c/e: gini_total_loss == 0"] = abs(tm["gini_total_loss"]) < 1e-9
    t["trivial c/e: rate gini == 0.4676"] = abs(tm["gini_exposure_weighted"] - 0.4676) < 5e-4
    t["trivial c/e: fixed-exposure gini ~ 0"] = abs(tm.get("gini_fixed_exposure", 9)) < 0.05

    # A real ranker must still score on the artifact-free metric.
    good = np.abs(np.asarray(test["BonusMalus"], dtype=float))
    gm = M.evaluate_pure_premium(y, good / e, e)
    t["a real ranker still scores > 0 on total-loss gini"] = gm["gini_total_loss"] > 0.01
    return _report(t)


def replay_synthetic() -> bool:
    print("\n=== replay 1: synthetic degenerate predictor ===")
    df, _ = build_and_cache()
    _, test = random_split(df)
    y = test["PurePremium"].to_numpy()
    e = test["Exposure"].to_numpy()
    pred = (test["TotalLoss"].sum() / e.sum()) * e.mean() / e

    metrics = M.evaluate_pure_premium(y, pred, e)
    flags = C.check_config("xgb_hurdle", pred, y, e, metrics=metrics)
    for f in flags:
        print(f"  {f}")

    kinds = {(f.check, f.severity) for f in flags}
    t = {
        "degeneracy raises FAIL": ("degeneracy", "FAIL") in kinds,
        "exposure_dependence raises WARN": ("exposure_dependence", "WARN") in kinds,
        "trivial_floor raises WARN": ("trivial_floor", "WARN") in kinds,
    }

    # The same predictions labelled as a reference must NOT raise a blocking failure.
    ref_flags = C.check_config("one_over_exposure", pred, y, e, metrics=metrics)
    t["same predictions as a reference line raise no FAIL"] = not any(
        f.severity == "FAIL" for f in ref_flags
    )
    return _report(t)


def replay_cached() -> bool:
    print("\n=== replay 2: the actual cached xgb_hurdle failure ===")
    if not CACHE.exists():
        print("  [skip] no runner cache")
        return True

    by_model: dict[tuple, list[float]] = {}
    for p in CACHE.glob("*.json"):
        rec = json.loads(p.read_text())
        cfg = rec.get("config", {})
        if cfg.get("track") != "B" or rec.get("status") != "ok":
            continue
        v = rec.get("metrics", {}).get("gini_exposure_weighted")
        if v is not None:
            by_model.setdefault((cfg.get("n"), cfg.get("model")), []).append(float(v))

    target = by_model.get((500, "xgb_hurdle"), [])
    if not target:
        print("  [skip] cached B/N500/xgb_hurdle not present")
        return True

    print(f"  cached values across {len(target)} seeds: "
          f"{[f'{v:.17g}' for v in sorted(target)]}")
    print(f"  spread (max-min) = {max(target) - min(target):.3e}")
    flags = C.check_across_seeds("xgb_hurdle", "gini_exposure_weighted", target,
                                 reference_value=0.4676069945192445)
    for f in flags:
        print(f"  {f}")
    kinds = {(f.check, f.severity) for f in flags}

    t = {
        "zero_variance raises FAIL on the real data": ("zero_variance", "FAIL") in kinds,
        "trivial_match raises FAIL on the real data": ("trivial_match", "FAIL") in kinds,
    }

    # A genuinely varying model must not be flagged.
    healthy = by_model.get((20_000, "tabicl"), [])
    if len(healthy) > 1:
        hf = C.check_across_seeds("tabicl", "gini_exposure_weighted", healthy,
                                  reference_value=0.4676069945192445)
        t["a healthy varying model raises no FAIL"] = not any(f.severity == "FAIL" for f in hf)
    return _report(t)


def main() -> int:
    ok = metric_tests()
    ok = replay_synthetic() and ok
    ok = replay_cached() and ok
    print("\nRESULT:", PASS if ok else FAIL)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
