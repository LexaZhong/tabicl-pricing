"""Diagnose the TabICL blow-up at N=500: is it stage 1, stage 2, or the exposure divide?"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.data import build_and_cache, exposure_stratified_subsample, random_split
from src.features.insurance_features import GBMFeaturePipeline
from src.models.tabicl_twostage import TabICLTwoStage

df, _ = build_and_cache()
pool, test = random_split(df)
test = test.iloc[:20_000].copy()

for n in [500, 2_000, 10_000]:
    train = exposure_stratified_subsample(pool, n, seed=0)
    pipe = GBMFeaturePipeline()
    X_tr = pipe.fit_transform(train, train["PurePremium"].to_numpy())
    X_te = pipe.transform(test)
    e_tr = train["Exposure"].to_numpy()
    e_te = test["Exposure"].to_numpy()

    m = TabICLTwoStage(context_size=n, seed=0, batch_size=4)
    m.fit(X_tr, train["has_loss"].to_numpy(), train["TotalLoss"].to_numpy(), e_tr)

    p = m.predict_claim_proba(X_te, e_te)
    s = m.predict_severity(X_te, e_te)
    pp = p * s / np.clip(e_te, 1e-10, None)

    y2 = train.loc[train["has_loss"] == 1, "TotalLoss"]
    print(f"\n=== N={n:,}  (stage-2 rows = {len(y2)}) ===")
    print(f"  severity cap (train q99.5)  {m.info.severity_cap:>14,.0f}")
    if len(y2):
        print(f"  train severity min/med/max  {y2.min():,.0f} / {y2.median():,.0f} / {y2.max():,.0f}")
    print(f"  p_hat   min/mean/max        {p.min():.5f} / {p.mean():.5f} / {p.max():.5f}")
    print(f"  S_hat   min/med/max         {s.min():,.0f} / {np.median(s):,.0f} / {s.max():,.0f}")
    print(f"  exposure min                {e_te.min():.5f}")
    print(f"  pure premium med/p99/max    {np.median(pp):,.0f} / {np.percentile(pp,99):,.0f} / {pp.max():,.0f}")
    print(f"  actual portfolio loss cost  "
          f"{(test['TotalLoss'].sum() / e_te.sum()):,.0f}")
    print(f"  predicted portfolio cost    {(pp * e_te).sum() / e_te.sum():,.0f}")
