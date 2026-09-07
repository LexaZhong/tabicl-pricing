"""Track 0.2: fetch, clean, split and cache freMTPL2. Prints the assertions a reviewer asks for."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import (
    DATA_DIR,
    build_and_cache,
    claims_only,
    frame_hash,
    random_split,
)


def main() -> int:
    df, report = build_and_cache()

    print("=== cleaning report ===")
    for k, v in report.to_dict().items():
        print(f"  {k:32s} {v:,}")

    print("\n=== assertions ===")
    checks = {
        "Exposure in (0, 1]": bool(((df["Exposure"] > 0) & (df["Exposure"] <= 1)).all()),
        "ClaimNb <= 4": bool((df["ClaimNb"] <= 4).all()),
        "ClaimNb >= 0": bool((df["ClaimNb"] >= 0).all()),
        "TotalLoss >= 0": bool((df["TotalLoss"] >= 0).all()),
        "has_claim matches ClaimNb": bool(
            (df["has_claim"] == (df["ClaimNb"] >= 1).astype(int)).all()
        ),
        "no loss without a claim": bool((df.loc[df["ClaimNb"] == 0, "TotalLoss"] == 0).all()),
        "PurePremium finite": bool(np.isfinite(df["PurePremium"]).all()),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print(f"\n  total rows: {len(df):,}")
    print(f"  claim rate: {df['has_claim'].mean():.4%}")
    print(f"  total exposure: {df['Exposure'].sum():,.0f}")
    print(f"  portfolio loss cost: {df['TotalLoss'].sum() / df['Exposure'].sum():,.2f}")

    co = claims_only(df)
    print(f"\n  stage-2 (claims-only) rows: {len(co):,}")
    print(f"  claims with exactly 1 claim: {(co['ClaimNb'] == 1).mean():.2%}")
    for q in [0.5, 0.9, 0.99, 0.995, 0.999, 1.0]:
        print(f"    TotalLoss q{q:<6} {co['TotalLoss'].quantile(q):>14,.0f}")

    train, test = random_split(df)
    print("\n=== split ===")
    print(f"  train: {len(train):,}  hash={frame_hash(train)}")
    print(f"  test : {len(test):,}  hash={frame_hash(test)}")
    print(f"  train claim rate: {train['has_claim'].mean():.4%}")
    print(f"  test  claim rate: {test['has_claim'].mean():.4%}")
    assert len(set(train['IDpol']) & set(test['IDpol'])) == 0, "train/test policy overlap"
    print("  [PASS] train/test disjoint on IDpol")

    print("\n=== regions ===")
    counts = df.groupby("Region").agg(
        n=("IDpol", "size"),
        exposure=("Exposure", "sum"),
        claim_rate=("has_claim", "mean"),
        loss_cost=("PurePremium", "mean"),
    ).sort_values("n", ascending=False)
    print(counts.to_string(float_format=lambda x: f"{x:,.4f}"))

    (DATA_DIR / "split_hashes.json").write_text(
        json.dumps(
            {
                "train_hash": frame_hash(train),
                "test_hash": frame_hash(test),
                "n_train": len(train),
                "n_test": len(test),
            },
            indent=2,
        )
    )
    counts.to_csv(DATA_DIR / "region_summary.csv")
    print(f"\nwrote {DATA_DIR / 'split_hashes.json'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
