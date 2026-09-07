"""Interim status: summarize whatever is in the runner cache right now."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.runner import collect_results, summarize

pd.set_option("display.width", 200)

res = collect_results()
if not len(res):
    print("no cached results yet")
    sys.exit(0)

done = res[["cfg_label", "status"]].drop_duplicates()
print(f"configs cached: {len(done)}   ok={int((done['status'] == 'ok').sum())}   "
      f"errors={int((done['status'] != 'ok').sum())}")

errs = done[done["status"] != "ok"]
if len(errs):
    print("\n=== ERRORS ===")
    print(errs.to_string(index=False))

for track, idx in [("B", "cfg_n"), ("A", "cfg_region"), ("C", "cfg_size")]:
    sub = res[res.get("cfg_track") == track] if "cfg_track" in res else res.iloc[0:0]
    if not len(sub):
        continue
    print(f"\n{'=' * 70}\nTRACK {track}\n{'=' * 70}")
    for metric in ["gini_exposure_weighted", "tweedie_deviance_1.5"]:
        s = summarize(sub, [idx, "cfg_model" if track != "C" else "cfg_mode"], metric)
        if not len(s):
            continue
        col = "cfg_model" if track != "C" else "cfg_mode"
        print(f"\n--- {metric}: mean over seeds ---")
        print(s.pivot(index=idx, columns=col, values="mean").to_string(
            float_format=lambda x: f"{x:.4f}"))
        print(f"\n--- {metric}: seed SD ---")
        print(s.pivot(index=idx, columns=col, values="sd").to_string(
            float_format=lambda x: f"{x:.4f}"))

# Stage-2 support is the real thin-data stress; reviewers will ask for it.
claims = res[res["metric"] == "info_n_claims_stage2"]
if len(claims):
    print("\n=== stage-2 claim counts by N ===")
    print(
        claims.groupby("cfg_n")["value"]
        .apply(lambda s: pd.to_numeric(s).mean())
        .to_string(float_format=lambda x: f"{x:,.0f}")
    )
