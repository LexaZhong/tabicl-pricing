"""What does a 10k fine-tuning draw actually contain, tail-wise?

Track B established that loss-cost variance across draws is driven by which large claims
land in the sample, not by covariate imbalance (exposure stratification matches means to
0.13%, but loss cost still varied 2.6x across draws at N=5,000). This script quantifies
that for the N=10,000 fine-tuning budget specifically, and -- the part that matters for
Track C -- traces the tail down through the splits the fine-tuner actually applies:

    10,000 policies
      -> ~N_claims rows reach stage 2 at all (claims-only)
      -> x0.9   validation_split_ratio=0.1
      -> x0.8   finetune_ctx_query_ratio=0.2 (the query rows are what the loss is computed on)

so the number of large claims the severity gradient ever sees is much smaller than the
claim count suggests. It also checks the 99.5% severity cap, which at these counts is
estimated from a handful of rows and is itself a tail statistic.

Usage: python experiments/trackC_tail_diag.py [--n 10000] [--seeds 50]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data import build_and_cache, exposure_stratified_subsample, random_split

VAL_SPLIT = 0.1   # FinetunedTabICL* validation_split_ratio
QUERY_RATIO = 0.2  # finetune_ctx_query_ratio


def portfolio_facts(pool: pd.DataFrame) -> dict:
    claims = pool.loc[pool["has_loss"] == 1, "TotalLoss"].to_numpy()
    claims_sorted = np.sort(claims)[::-1]
    total = claims.sum()
    n = len(claims)
    facts = {
        "n_policies": len(pool),
        "n_claim_policies": n,
        "claim_rate": n / len(pool),
        "total_loss": total,
        "mean_severity": claims.mean(),
        "median_severity": float(np.median(claims)),
        "portfolio_loss_cost": total / pool["Exposure"].sum(),
    }
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]:
        facts[f"sev_q{q}"] = float(np.quantile(claims, q))
    # Concentration: how much of all loss sits in the largest k% of claims.
    for pct in [0.1, 0.5, 1.0, 5.0]:
        k = max(1, int(round(n * pct / 100)))
        facts[f"share_top_{pct}pct"] = float(claims_sorted[:k].sum() / total)
    return facts


def draw_facts(pool: pd.DataFrame, n: int, seed: int, port: dict) -> dict:
    s = exposure_stratified_subsample(pool, n, seed=seed)
    claims = s.loc[s["has_loss"] == 1, "TotalLoss"].to_numpy()
    nc = len(claims)
    exp = s["Exposure"].to_numpy()

    # The stage-2 rows that survive the fine-tuner's own splits.
    n_ft_train = int(round(nc * (1 - VAL_SPLIT)))
    n_query = int(round(n_ft_train * QUERY_RATIO))

    row = {
        "seed": seed,
        "n_claims": nc,
        "mean_exposure": float(exp.mean()),
        "loss_cost": float(claims.sum() / exp.sum()),
        "max_claim": float(claims.max()) if nc else 0.0,
        "cap_q995": float(np.quantile(claims, 0.995)) if nc else np.nan,
        "floor_q005": float(np.quantile(claims, 0.005)) if nc else np.nan,
        # How much of THIS sample's loss rides on its single biggest claim.
        "top1_share": float(claims.max() / claims.sum()) if nc else np.nan,
        "top5_share": float(np.sort(claims)[-5:].sum() / claims.sum()) if nc >= 5 else np.nan,
        # Tail coverage against the portfolio's own quantiles.
        "n_above_port_q99": int((claims > port["sev_q0.99"]).sum()),
        "n_above_port_q999": int((claims > port["sev_q0.999"]).sum()),
        "stage2_rows": nc,
        "stage2_ft_train": n_ft_train,
        "stage2_query_rows": n_query,
        # Expected number of >q99 claims that land in the gradient-carrying query split.
        "exp_q99_in_query": (claims > port["sev_q0.99"]).sum() * (1 - VAL_SPLIT) * QUERY_RATIO,
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()

    df, _ = build_and_cache()
    pool, _test = random_split(df)

    port = portfolio_facts(pool)
    print("=" * 78)
    print(f"TRAINING POOL: {port['n_policies']:,} policies, {port['n_claim_policies']:,} with loss "
          f"({port['claim_rate']:.2%})")
    print("=" * 78)
    print(f"  loss cost           {port['portfolio_loss_cost']:>12,.2f}")
    print(f"  mean severity       {port['mean_severity']:>12,.0f}   median {port['median_severity']:>10,.0f}")
    print("  severity quantiles:")
    for q in [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0]:
        print(f"    q{q:<6} {port[f'sev_q{q}']:>14,.0f}")
    print("  loss concentration (share of ALL loss held by the largest claims):")
    for pct in [0.1, 0.5, 1.0, 5.0]:
        k = max(1, int(round(port["n_claim_policies"] * pct / 100)))
        print(f"    top {pct:>4}% ({k:>5,} claims): {port[f'share_top_{pct}pct']:>7.1%}")

    rows = [draw_facts(pool, args.n, s, port) for s in range(args.seeds)]
    d = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(f"{args.seeds} EXPOSURE-STRATIFIED DRAWS OF N={args.n:,}")
    print("=" * 78)
    cols = ["n_claims", "loss_cost", "max_claim", "cap_q995", "top1_share",
            "n_above_port_q99", "n_above_port_q999"]
    summ = d[cols].agg(["mean", "std", "min", "max"]).T
    summ["cv"] = summ["std"] / summ["mean"]
    print(summ.to_string(float_format=lambda x: f"{x:,.4f}" if abs(x) < 10 else f"{x:,.1f}"))

    print("\n  Tail coverage across draws:")
    print(f"    claims above portfolio q99  ({port['sev_q0.99']:,.0f}): "
          f"mean {d['n_above_port_q99'].mean():.1f}, range {d['n_above_port_q99'].min()}-{d['n_above_port_q99'].max()}")
    print(f"    claims above portfolio q99.9 ({port['sev_q0.999']:,.0f}): "
          f"mean {d['n_above_port_q999'].mean():.2f}, "
          f"{(d['n_above_port_q999'] == 0).mean():.0%} of draws contain NONE")

    print("\n  What reaches the severity gradient (stage 2 only):")
    print(f"    claim rows in sample        {d['n_claims'].mean():>8.0f}")
    print(f"    after 10% val split         {d['stage2_ft_train'].mean():>8.0f}")
    print(f"    query rows (loss computed)  {d['stage2_query_rows'].mean():>8.0f}")
    print(f"    expected >q99 claims in the query split: {d['exp_q99_in_query'].mean():.2f}")

    print("\n  The 99.5% severity cap is itself a tail statistic at this size:")
    print(f"    cap value across draws: mean {d['cap_q995'].mean():,.0f}, "
          f"SD {d['cap_q995'].std():,.0f}, range {d['cap_q995'].min():,.0f}-{d['cap_q995'].max():,.0f}")
    print(f"    portfolio q99.5 for reference: {port['sev_q0.995']:,.0f}")
    n_above = np.ceil(d["n_claims"].mean() * 0.005)
    print(f"    it is estimated from ~{n_above:.0f} row(s) above it")

    out = Path(__file__).resolve().parent.parent / "results" / "trackC_tail_diag.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nPer-draw detail written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
