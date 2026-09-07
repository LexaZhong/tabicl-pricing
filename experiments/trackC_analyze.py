"""Track C analysis: the PAIRED contrast between in-context and fine-tuned TabICL.

Every seed runs both modes on the identical 20,000-row context, so the two arms are
matched and the difference must be taken WITHIN seed. This is not a stylistic preference:
Track B's draws show 87-95% of calibration variance is common to the draw, so the paired
contrast detects 1.12 deviance units where the unpaired comparison needs 3.44. Summarising
each mode independently and eyeing the gap would throw that away.

Three questions, reported separately because Track B showed they do not move together:

  1. Does fine-tuning shift the mean?      paired t-test on (finetune - icl)
  2. Does it clear the external targets?   one-sample tests vs fixed reference values
  3. Does it stabilise?                    F-test on the variance ratio (the run-3 claim)

Usage: python experiments/trackC_analyze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

# Fixed reference points from Track B, all scored on the same 135,603-row test set.
BENCH = {
    "tweedie_deviance_1.5": {
        "xgb_hurdle @ full 542,410": 82.675,
        "tabicl @ full 542,410": 87.278,
        "trivial 1/exposure @ 20,000": 94.607,
        "tabicl ICL @ 20,000 (Track B)": 93.256,
    },
    "gini_total_loss": {
        "xgb_hurdle @ full 542,410": 0.543,
        "xgb_hurdle @ 20,000": 0.237,
        "tabicl ICL @ 20,000 (Track B)": 0.411,
    },
    "calibration_ratio": {
        "perfect": 1.000,
        "xgb_hurdle @ full 542,410": 0.539,
        "tabicl ICL @ 20,000 (Track B)": 0.453,
    },
}
METRICS = ["tweedie_deviance_1.5", "gini_total_loss", "calibration_ratio",
           "gini_exposure_weighted"]
LOWER_IS_BETTER = {"tweedie_deviance_1.5"}


def load() -> pd.DataFrame:
    p = ROOT / "results" / "trackC_results.parquet"
    if not p.exists():
        raise SystemExit(f"no results yet at {p}; let trackC_run.py finish first")
    d = pd.read_parquet(p)
    d = d[d["status"] == "ok"]
    return d.pivot_table(index=["cfg_arm", "cfg_seed"], columns=["cfg_mode", "metric"],
                         values="value")


def main() -> int:
    d = load()
    arms = sorted({a for a, _ in d.index})

    for arm in arms:
        sub = d.loc[arm]
        print("=" * 90)
        print(f"ARM: {arm}")
        print("=" * 90)

        for metric in METRICS:
            if ("icl", metric) not in sub or ("finetune", metric) not in sub:
                continue
            pair = sub[[("icl", metric), ("finetune", metric)]].dropna()
            if len(pair) < 3:
                print(f"\n{metric}: only {len(pair)} complete pairs, skipping")
                continue
            icl = pair[("icl", metric)]
            ft = pair[("finetune", metric)]
            diff = ft - icl
            better = "lower" if metric in LOWER_IS_BETTER else "higher"

            t, p = stats.ttest_rel(ft, icl)
            lo, hi = stats.t.interval(0.95, len(diff) - 1,
                                      loc=diff.mean(), scale=stats.sem(diff))

            print(f"\n--- {metric}  ({better} is better, R={len(pair)} paired draws) ---")
            print(f"  icl       {icl.mean():>10.4f}  SD {icl.std():>8.4f}")
            print(f"  finetune  {ft.mean():>10.4f}  SD {ft.std():>8.4f}")
            print(f"  paired difference (finetune - icl): {diff.mean():>+9.4f}"
                  f"  95% CI [{lo:+.4f}, {hi:+.4f}]")
            print(f"  paired t = {t:+.2f}, p = {p:.4f}"
                  f"   {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
            wins = int((diff < 0).sum() if metric in LOWER_IS_BETTER else (diff > 0).sum())
            print(f"  fine-tuning wins on {wins}/{len(diff)} individual draws")

            # 2. External targets.
            if metric in BENCH:
                print("  vs fixed benchmarks (one-sample t on the fine-tuned arm):")
                for name, val in BENCH[metric].items():
                    tt, pp = stats.ttest_1samp(ft, val)
                    gap = ft.mean() - val
                    beat = (gap < 0) if metric in LOWER_IS_BETTER else (gap > 0)
                    verdict = ("BEATS" if beat else "does not beat") if pp < 0.05 else "tie"
                    print(f"    {name:<32} {val:>9.3f}  gap {gap:>+8.3f}  "
                          f"p={pp:.4f}  {verdict}")

            # 3. Stability -- the original run-3 claim was about variance, not means.
            f_stat = (icl.var() / ft.var()) if ft.var() > 0 else np.inf
            df1 = df2 = len(pair) - 1
            p_f = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
            print(f"  stability: SD ratio finetune/icl = {ft.std()/icl.std():.2f}"
                  f"  (variance F = {f_stat:.2f}, p = {p_f:.4f}"
                  f", {'fine-tuning is more stable' if f_stat > 1 and p_f < 0.05 else 'no significant change'})")

    # Cross-arm contrast, if both arms ran.
    if len(arms) > 1:
        print("\n" + "=" * 90)
        print("ARM CONTRAST (tail-balanced vs srs), matched on seed")
        print("=" * 90)
        for metric in METRICS:
            for mode in ["icl", "finetune"]:
                try:
                    a = d.loc["srs"][(mode, metric)]
                    b = d.loc["tail_balanced"][(mode, metric)]
                except KeyError:
                    continue
                pair = pd.concat([a, b], axis=1, keys=["srs", "tb"]).dropna()
                if len(pair) < 3:
                    continue
                diff = pair["tb"] - pair["srs"]
                t, p = stats.ttest_rel(pair["tb"], pair["srs"])
                print(f"  {metric:<24} {mode:<9} diff {diff.mean():>+8.4f}  "
                      f"p={p:.4f}   SD ratio {pair['tb'].std()/pair['srs'].std():.2f}")

    out = ROOT / "results" / "trackC_paired.csv"
    d.to_csv(out)
    print(f"\nPaired table written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
