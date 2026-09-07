"""Does tail coverage in the training draw actually move test performance?

Track B already ran 20 independent draws at each of N=5,000/10,000/20,000 and scored every
one on the SAME fixed 135,603-row test set. So all between-draw variation in the test
metrics is caused by the training draw and nothing else -- which makes those runs a
natural experiment on tail coverage, at zero GPU cost.

For each (N, seed) we reconstruct the draw, measure its tail content, and correlate that
against the test metrics the draw produced. Spearman is used throughout: tail measures are
heavy-tailed and a single 1.4M claim would dominate Pearson.

This is evidence about the ICL arm, not the fine-tuned arm -- but if tail coverage moves
in-context performance, the fine-tuning design has to control it rather than hope.

Usage: python experiments/trackC_tail_effect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.data import build_and_cache, exposure_stratified_subsample, random_split

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["gini_total_loss", "tweedie_deviance_1.5", "calibration_ratio",
           "gini_exposure_weighted"]
TAIL_VARS = ["n_above_q99", "max_claim", "top1_share", "sample_loss_cost", "cap_q995"]


def draw_tail_stats(pool: pd.DataFrame, n: int, seed: int, q99: float) -> dict:
    s = exposure_stratified_subsample(pool, n, seed=seed)
    claims = s.loc[s["has_loss"] == 1, "TotalLoss"].to_numpy()
    if len(claims) == 0:
        return {}
    return {
        "cfg_n": n,
        "cfg_seed": seed,
        "n_claims": len(claims),
        "n_above_q99": int((claims > q99).sum()),
        "max_claim": float(claims.max()),
        "top1_share": float(claims.max() / claims.sum()),
        "sample_loss_cost": float(claims.sum() / s["Exposure"].sum()),
        "cap_q995": float(np.quantile(claims, 0.995)),
    }


def main() -> int:
    res = pd.read_parquet(ROOT / "results" / "trackB_results.parquet")
    res = res[(res["status"] == "ok") & res["metric"].isin(METRICS)]

    df, _ = build_and_cache()
    pool, _test = random_split(df)
    q99 = float(np.quantile(pool.loc[pool["has_loss"] == 1, "TotalLoss"], 0.99))

    # Only the rungs that were actually resampled with many seeds.
    counts = res.groupby("cfg_n")["cfg_seed"].nunique()
    rungs = sorted(counts[counts >= 10].index)
    print(f"portfolio q99 severity = {q99:,.0f}")
    print(f"rungs with >=10 draws: {rungs}\n")

    pairs = res[res["cfg_n"].isin(rungs)][["cfg_n", "cfg_seed"]].drop_duplicates()
    tail = pd.DataFrame([
        draw_tail_stats(pool, int(r.cfg_n), int(r.cfg_seed), q99)
        for r in pairs.itertuples()
    ])

    wide = res.pivot_table(index=["cfg_n", "cfg_model", "cfg_seed"],
                           columns="metric", values="value").reset_index()
    m = wide.merge(tail, on=["cfg_n", "cfg_seed"], how="inner")

    print("=" * 96)
    print("SPEARMAN r BETWEEN TRAINING-DRAW TAIL CONTENT AND TEST METRIC")
    print("(test set is identical across draws, so any association is caused by the draw)")
    print("=" * 96)

    out_rows = []
    for n in rungs:
        for model in sorted(m.loc[m["cfg_n"] == n, "cfg_model"].unique()):
            sub = m[(m["cfg_n"] == n) & (m["cfg_model"] == model)]
            if len(sub) < 8:
                continue
            for metric in METRICS:
                if metric not in sub or sub[metric].isna().all():
                    continue
                for tv in TAIL_VARS:
                    r, p = stats.spearmanr(sub[tv], sub[metric])
                    out_rows.append({"n": n, "model": model, "metric": metric,
                                     "tail_var": tv, "r": r, "p": p, "draws": len(sub)})
    o = pd.DataFrame(out_rows)

    for n in rungs:
        print(f"\n--- N = {n:,} ---")
        for metric in METRICS:
            block = o[(o["n"] == n) & (o["metric"] == metric)]
            if block.empty:
                continue
            piv = block.pivot(index="model", columns="tail_var", values="r")
            print(f"\n  {metric}")
            print(piv.to_string(float_format=lambda x: f"{x:+.2f}"))

    print("\n" + "=" * 96)
    print("STRONGEST ASSOCIATIONS (|r| >= 0.5 and p < 0.05)")
    print("=" * 96)
    strong = o[(o["r"].abs() >= 0.5) & (o["p"] < 0.05)].sort_values("r", key=abs, ascending=False)
    if strong.empty:
        print("  none -- tail content does not detectably move the metrics at these draw counts")
    else:
        print(strong.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    out = ROOT / "results" / "trackC_tail_effect.csv"
    o.to_csv(out, index=False)
    m.to_csv(ROOT / "results" / "trackC_draw_tail_joined.csv", index=False)
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
