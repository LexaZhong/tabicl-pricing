"""Quantify the has_claim vs has_loss gap -- it decides the stage-1 target."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import build_and_cache

df, _ = build_and_cache()

print("columns:", list(df.columns))
n = len(df)
print(f"\nrows                       {n:,}")
print(f"has_claim rate             {df['has_claim'].mean():.4%}  ({df['has_claim'].sum():,})")
print(f"has_loss  rate             {df['has_loss'].mean():.4%}  ({df['has_loss'].sum():,})")
print(f"claims with no loss record {int(((df['has_claim'] == 1) & (df['has_loss'] == 0)).sum()):,}")
print(
    f"P(loss>0 | claim)          "
    f"{df.loc[df['has_claim'] == 1, 'has_loss'].mean():.4f}"
)

exposure = df["Exposure"].to_numpy()
loss = df["TotalLoss"].to_numpy()
true_pp = loss.sum() / exposure.sum()

# The bias: pairing P(claim) with E[loss | loss>0] instead of P(loss>0) with the same.
p_claim = df["has_claim"].mean()
p_loss = df["has_loss"].mean()
mean_sev = loss[loss > 0].mean()

print(f"\nportfolio loss cost (truth)          {true_pp:,.2f}")
print(f"implied by P(has_loss)  x E[S|loss>0] {p_loss * mean_sev / exposure.mean():,.2f}")
print(f"implied by P(has_claim) x E[S|loss>0] {p_claim * mean_sev / exposure.mean():,.2f}")
print(f"\noverstatement factor if has_claim is used: {p_claim / p_loss:.4f}x")
