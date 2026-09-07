"""Does the tail-balanced arm actually do what it is supposed to, before we spend GPU on it?

Arm 2 exists to remove the draw-to-draw tail variation that Track B showed drives
calibration. That is a claim about the SAMPLER, and it is testable on CPU in seconds. If
tail-balancing does not measurably shrink the spread of sample loss cost and tail counts,
the arm cannot possibly change the fine-tuning result and 20 GPU-configs would be wasted.

Also reports the level bias the arm introduces on purpose (quotas are rounded up to 1 in
sparse tail bands), since that has to be declared and corrected rather than discovered later.

Usage: python experiments/trackC_arm_check.py [--n 20000] [--seeds 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import (
    SEVERITY_BANDS,
    build_and_cache,
    exposure_stratified_subsample,
    random_split,
    severity_stratified_subsample,
)

ROOT = Path(__file__).resolve().parent.parent


def stats_for(s: pd.DataFrame, q99: float, q999: float) -> dict:
    claims = s.loc[s["has_loss"] == 1, "TotalLoss"].to_numpy()
    exp = s["Exposure"].to_numpy()
    return {
        "n_rows": len(s),
        "n_claims": len(claims),
        "mean_exposure": float(exp.mean()),
        "loss_cost": float(claims.sum() / exp.sum()),
        "max_claim": float(claims.max()) if len(claims) else 0.0,
        "n_above_q99": int((claims > q99).sum()),
        "n_above_q999": int((claims > q999).sum()),
        "cap_q995": float(np.quantile(claims, 0.995)) if len(claims) else np.nan,
        "top1_share": float(claims.max() / claims.sum()) if len(claims) else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    df, _ = build_and_cache()
    pool, _test = random_split(df)
    sev = pool.loc[pool["has_loss"] == 1, "TotalLoss"].to_numpy()
    q99, q999 = float(np.quantile(sev, 0.99)), float(np.quantile(sev, 0.999))
    port_loss_cost = sev.sum() / pool["Exposure"].sum()

    print(f"pool: {len(pool):,} policies, {len(sev):,} claims, loss cost {port_loss_cost:,.2f}")
    print(f"severity bands (quantiles): {SEVERITY_BANDS}")
    print(f"  edges: {[f'{v:,.0f}' for v in np.quantile(sev, SEVERITY_BANDS)]}\n")

    rows = []
    for seed in range(args.seeds):
        srs = exposure_stratified_subsample(pool, args.n, seed=seed)
        tb = severity_stratified_subsample(pool, args.n, seed=seed, base=srs)
        rows.append({"seed": seed, "arm": "srs", **stats_for(srs, q99, q999)})
        rows.append({"seed": seed, "arm": "tail_balanced", **stats_for(tb, q99, q999)})
    d = pd.DataFrame(rows)

    cols = ["n_rows", "n_claims", "mean_exposure", "loss_cost", "max_claim",
            "n_above_q99", "n_above_q999", "cap_q995", "top1_share"]

    print("=" * 92)
    print(f"{args.seeds} DRAWS OF N={args.n:,}, MATCHED PAIRS (same non-claim rows, same claim count)")
    print("=" * 92)
    summ = d.groupby("arm")[cols].agg(["mean", "std", "min", "max"])
    for c in cols:
        print(f"\n{c}")
        blk = summ[c].copy()
        blk["cv"] = blk["std"] / blk["mean"].abs()
        print(blk.to_string(float_format=lambda x: f"{x:,.4f}" if abs(x) < 100 else f"{x:,.1f}"))

    print("\n" + "=" * 92)
    print("VARIANCE REDUCTION FROM TAIL-BALANCING (the whole point of arm 2)")
    print("=" * 92)
    print(f"{'measure':>18} {'SRS SD':>12} {'TB SD':>12} {'SD ratio':>10}  verdict")
    for c in ["loss_cost", "max_claim", "n_above_q99", "n_above_q999", "cap_q995", "top1_share"]:
        a = d[d["arm"] == "srs"][c].std()
        b = d[d["arm"] == "tail_balanced"][c].std()
        ratio = b / a if a > 0 else np.nan
        if np.isnan(ratio):
            v = "n/a"
        elif ratio < 0.5:
            v = "large reduction"
        elif ratio < 0.85:
            v = "moderate"
        elif ratio <= 1.15:
            v = "NO REDUCTION"
        else:
            v = "WORSE"
        print(f"{c:>18} {a:>12,.3f} {b:>12,.3f} {ratio:>10.2f}  {v}")

    print("\n" + "=" * 92)
    print("DECLARED BIAS (arm 2 over-represents the tail by rounding sparse quotas up to 1)")
    print("=" * 92)
    for arm in ["srs", "tail_balanced"]:
        lc = d[d["arm"] == arm]["loss_cost"]
        print(f"  {arm:>14}: loss cost {lc.mean():>8,.2f}  "
              f"({lc.mean()/port_loss_cost - 1:+.1%} vs portfolio {port_loss_cost:,.2f})")
    print("  -> correct this with metrics.balance_factor; it is a level shift, not a ranking effect.")

    print("\n" + "=" * 92)
    print("AFTER CAPPING: what the severity model actually trains on")
    print("=" * 92)
    print("The residual max_claim spread above is mostly harmless once the cap is fixed --")
    print("the target is clipped before it reaches stage 2, so a 1.4M and a 153k top-band")
    print("claim enter training as the same number. This is the quantity to judge the arm on.")
    fixed_cap = float(np.quantile(sev, 0.995))
    print(f"\n  fixed cap = portfolio q99.5 = {fixed_cap:,.0f}")
    print(f"{'arm':>16} {'capped loss cost':>18} {'SD':>9} {'min':>9} {'max':>9}")
    capped = {}
    for seed in range(args.seeds):
        srs = exposure_stratified_subsample(pool, args.n, seed=seed)
        tb = severity_stratified_subsample(pool, args.n, seed=seed, base=srs)
        for arm, s in [("srs", srs), ("tail_balanced", tb)]:
            cl = s.loc[s["has_loss"] == 1, "TotalLoss"].to_numpy()
            capped.setdefault(arm, []).append(np.clip(cl, 0, fixed_cap).sum() / s["Exposure"].sum())
    for arm, v in capped.items():
        v = np.asarray(v)
        print(f"{arm:>16} {v.mean():>18,.2f} {v.std(ddof=1):>9,.2f} {v.min():>9,.2f} {v.max():>9,.2f}")
    r = np.std(capped["tail_balanced"], ddof=1) / np.std(capped["srs"], ddof=1)
    print(f"\n  SD ratio (tail_balanced / srs) on the CAPPED target: {r:.2f}")

    out = ROOT / "results" / "trackC_arm_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
