"""Is the pure-premium Gini gameable by 1/exposure alone?

Hypothesis: PurePremium = loss / Exposure, so a policy with a claim and a tiny exposure has
an enormous actual pure premium. Any model of the form pred = p * S / Exposure inherits a
1/Exposure factor. If p and S collapse to constants (which Optuna can select at small N,
where a constant is the safest predictor), the prediction becomes c / Exposure -- a ranking
that contains ZERO risk information but may still score a high Gini.

If true, this is a metric artifact that flatters every hurdle-parameterised model, including
TabICL's, and it has to be reported alongside the headline numbers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src import metrics as M
from src.data import build_and_cache, random_split

df, _ = build_and_cache()
_, test = random_split(df)

y = test["PurePremium"].to_numpy()
e = test["Exposure"].to_numpy()
n = len(test)

print(f"test rows {n:,}   exposure min/median/max "
      f"{e.min():.4f} / {np.median(e):.4f} / {e.max():.4f}\n")

candidates = {
    "constant (intercept)": np.full(n, 100.0),
    "1 / exposure": 1.0 / e,
    "constant * (1/exposure)": 250.0 / e,
    "exposure (ascending)": e.copy(),
    "random": np.random.default_rng(0).random(n),
}

print(f"{'predictor':<26} {'gini':>9} {'wGini':>9} {'nGini_w':>9} {'dev1.5':>10}")
print("-" * 68)
for name, pred in candidates.items():
    pred = np.clip(pred, 1e-10, None)
    m = M.evaluate_pure_premium(y, pred, e)
    print(
        f"{name:<26} {m['gini']:>9.4f} {m['gini_exposure_weighted']:>9.4f} "
        f"{m['normalized_gini_exposure_weighted']:>9.4f} {m['tweedie_deviance_1.5']:>10.2f}"
    )

# How much of the actual loss sits on short-exposure policies?
print("\n=== actual pure premium by exposure decile ===")
import pandas as pd

d = pd.DataFrame({"pp": y, "e": e, "loss": y * e})
d["decile"] = pd.qcut(d["e"], 10, labels=False, duplicates="drop")
g = d.groupby("decile").agg(
    exposure_min=("e", "min"),
    exposure_max=("e", "max"),
    loss_cost=("loss", lambda s: s.sum()),
    exposure_sum=("e", "sum"),
)
g["loss_cost_per_exposure"] = g["loss_cost"] / g["exposure_sum"]
g["mean_actual_pp"] = d.groupby("decile")["pp"].mean()
print(g.to_string(float_format=lambda x: f"{x:,.2f}"))
