"""Track C: TabICLTwoStage vs FinetunedTabICLTwoStage at N=20,000, fine-tuned on a
DISJOINT 10,000.

Design fixed with the user on 2026-09-07:

  context   N = 20,000 exposure-stratified draw. Track B found this is where in-context
            TabICL first beats xgb_hurdle on ranking (gini_total_loss 0.412 vs 0.237,
            z~7.5) while still losing on price (deviance 93.26 vs the trivial model's
            94.61, z=1.6 n.s.). Scored on the standard fixed 135,603-row test set so the
            numbers are directly comparable to Track B and to the full-data benchmark.

  finetune  N = 10,000, drawn from the training pool but DISJOINT from the 20,000 context
            (asserted per config, on IDpol). This matters: the stock estimator fine-tunes
            on (X, y) and then installs that same (X, y) as the in-context example set
            (_finetune/base.py:1119-1121), so a naive run would confound "the weights
            learned this distribution" with "these rows were also visible in context".
            10,000 is also the size we have a measured cost for -- 71.9 s/epoch.

  contrast  icl       TabICLTwoStage on the 20,000 context, no weight update.
            finetune  DisjointFinetunedTabICLTwoStage: weights fine-tuned on the 10,000,
                      then conditioned on the SAME 20,000 context.
            Same context, same cap, same seed -- so the paired difference isolates the
            weight update and nothing else.

  cap       FIXED at the training pool's q99.5 in both modes. Track B let each draw
            estimate its own cap from ~2-4 rows, which swung 14.6k-168.2k and hard-clips
            every prediction.

  R = 20    Seeds 0-19, matching Track B's resampling depth. Analysis MUST difference
            within seed: 87-95% of calibration variance is common to the draw, so the
            paired contrast detects 1.12 deviance units where unpaired needs 3.44.

  target    xgb_hurdle on the full 542,410: deviance 82.675 (SD 0.164), 10.58 below
            TabICL's in-context 93.256. For scale, 27x more data buys TabICL only 5.98.

Usage: python experiments/trackC_run.py [--arms srs] [--modes icl,finetune]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src import metrics as M
from src.data import (
    ID_COL,
    build_and_cache,
    exposure_stratified_subsample,
    random_split,
    severity_stratified_subsample,
)
from src.features.insurance_features import GBMFeaturePipeline
from src.models.tabicl_twostage import DisjointFinetunedTabICLTwoStage, TabICLTwoStage
from src.runner import RESULTS, collect_results, run_config, summarize

N_CONTEXT = 20_000
N_FINETUNE = 10_000
SEEDS = list(range(20))
CAP_QUANTILE = 0.995

OOM_LADDER = [
    {},
    {"n_estimators_finetune": 1},
    {"n_estimators_finetune": 1, "freeze_col": True},
    {"n_estimators_finetune": 1, "freeze_col": True, "freeze_row": True},
]


def draw_pair(train_pool, n_ctx: int, n_ft: int, seed: int, arm: str):
    """Return (context_df, finetune_df), guaranteed disjoint on IDpol."""
    srs = exposure_stratified_subsample(train_pool, n_ctx, seed=seed)
    ctx = srs if arm == "srs" else severity_stratified_subsample(
        train_pool, n_ctx, seed=seed, base=srs
    )
    # Fine-tuning rows come from what the context did NOT take, so the two are disjoint by
    # construction rather than by luck -- then asserted anyway.
    remaining = train_pool[~train_pool[ID_COL].isin(set(ctx[ID_COL]))]
    ft = exposure_stratified_subsample(remaining, n_ft, seed=seed)
    return ctx, ft


def make_runner(df, n_ctx: int, n_ft: int, epochs: int, time_limit, max_data_size):
    train_pool, test = random_split(df)

    # Declared constant computed on the training pool only -- never the test set.
    pool_claims = train_pool.loc[train_pool["has_loss"] == 1, "TotalLoss"].to_numpy()
    fixed_cap = float(np.quantile(pool_claims, CAP_QUANTILE))
    fixed_floor = float(np.quantile(pool_claims, 1.0 - CAP_QUANTILE))
    print(f"context N   = {n_ctx:,}")
    print(f"finetune N  = {n_ft:,}  (disjoint from context)")
    print(f"fixed cap   = {fixed_cap:,.2f}   floor = {fixed_floor:,.2f}  "
          f"(train-pool q{CAP_QUANTILE})\n")

    y_te = test["PurePremium"].to_numpy()
    e_te = test["Exposure"].to_numpy()
    test_ids = set(test[ID_COL])

    def run(cfg: dict) -> dict:
        seed, arm, mode = cfg["seed"], cfg["arm"], cfg["mode"]
        ctx, ft = draw_pair(train_pool, n_ctx, n_ft, seed, arm)

        ctx_ids, ft_ids = set(ctx[ID_COL]), set(ft[ID_COL])
        assert not (ctx_ids & ft_ids), "fine-tuning set overlaps the context set"
        assert not (ctx_ids & test_ids), "context set overlaps test"
        assert not (ft_ids & test_ids), "fine-tuning set overlaps test"

        # One pipeline, fitted on the context, applied to fine-tune and test alike: the
        # feature encoding must not differ between the three or the comparison is invalid.
        pipe = GBMFeaturePipeline()
        X_ctx = pipe.fit_transform(ctx, ctx["PurePremium"].to_numpy())
        X_ft = pipe.transform(ft)
        X_te = pipe.transform(test)

        e_ctx = ctx["Exposure"].to_numpy()
        hurdle_ctx = ctx["has_loss"].to_numpy()
        loss_ctx = ctx["TotalLoss"].to_numpy()

        cap_kw = dict(cap_mode="fixed", fixed_cap=fixed_cap, fixed_floor=fixed_floor)
        info = {
            "n_context": len(ctx),
            "n_finetune": len(ft) if mode == "finetune" else 0,
            "n_claims_context": int(hurdle_ctx.sum()),
            "n_claims_finetune": int(ft["has_loss"].sum()),
            # arm/mode live in the config, not here: collect_results flattens info into a
            # single numeric `value` column, so a string here breaks the parquet write.
            "overlap_ctx_ft": 0,
            "draw_loss_cost": float(loss_ctx.sum() / e_ctx.sum()),
            "draw_capped_loss_cost": float(np.clip(loss_ctx, 0, fixed_cap).sum() / e_ctx.sum()),
            "draw_max_claim": float(loss_ctx.max()),
        }

        if mode == "icl":
            model = TabICLTwoStage(context_size=n_ctx, seed=seed, batch_size=2, **cap_kw)
            model.fit(X_ctx, hurdle_ctx, loss_ctx, e_ctx)
        else:
            last_err = None
            for rung, patch in enumerate(OOM_LADDER):
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    model = DisjointFinetunedTabICLTwoStage(
                        context_size=n_ctx,
                        seed=seed,
                        batch_size=2,
                        finetune_size=max_data_size or n_ft,
                        epochs=epochs,
                        time_limit=time_limit,
                        **cap_kw,
                        **patch,
                    )
                    model.fit(
                        X_ctx, hurdle_ctx, loss_ctx, e_ctx,
                        X_ft=X_ft,
                        hurdle_ft=ft["has_loss"].to_numpy(),
                        total_loss_ft=ft["TotalLoss"].to_numpy(),
                        exposure_ft=ft["Exposure"].to_numpy(),
                    )
                    info["oom_rung"] = rung
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    last_err = exc
                    print(f"    OOM at rung {rung} ({patch}); escalating", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                raise RuntimeError(f"OOM at every rung: {last_err}")

        pred = model.predict(X_te, e_te)
        info.update({k: v for k, v in model.info.__dict__.items() if k != "extra"})
        info.update({f"x_{k}": v for k, v in model.info.extra.items()})

        met = M.evaluate_pure_premium(y_te, pred, e_te)
        p_hat = model.predict_claim_proba(X_te, e_te)
        info.update(M.stage1_metrics(test["has_loss"].to_numpy(), p_hat))
        info.update(M.exposure_monotonicity(p_hat, e_te))
        return {"metrics": met, "info": info, "predictions": {"pred": pred}}

    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="srs",
                    help="srs (control) and/or tail_balanced; default srs only")
    ap.add_argument("--modes", default="icl,finetune")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--n-context", type=int, default=N_CONTEXT)
    ap.add_argument("--n-finetune", type=int, default=N_FINETUNE)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--time-limit", type=float, default=None,
                    help="seconds per fine-tune fit; None runs all epochs. A limit "
                         "truncates configs at different epoch counts, confounding the "
                         "comparison -- prefer reducing --epochs.")
    ap.add_argument("--max-data-size", type=int, default=0,
                    help="fine-tune chunk size; 0 = the fine-tune set (1 step/epoch)")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    df, _ = build_and_cache()
    run = make_runner(df, args.n_context, args.n_finetune, args.epochs,
                      args.time_limit, args.max_data_size or None)

    configs = []
    # ICL first: ~5 min a config, so an interrupted run still leaves a complete control arm.
    for mode in sorted(modes, key=lambda m: m != "icl"):
        for arm in arms:
            for seed in seeds:
                configs.append({
                    "track": "C", "n": args.n_context, "n_ft": args.n_finetune,
                    "arm": arm, "mode": mode, "seed": seed,
                    "epochs": args.epochs if mode == "finetune" else 0,
                    "cap": "fixed", "disjoint": True,
                    "mds": args.max_data_size or args.n_finetune,
                    "label": f"C/{arm}/{mode}/s{seed}",
                })

    n_ft_cfg = sum(c["mode"] == "finetune" for c in configs)
    print(f"Track C: {len(configs)} configs ({n_ft_cfg} fine-tuned)\n")
    for cfg in configs:
        run_config(cfg, run, require_predictions=True)

    res = collect_results(pattern="C")
    if len(res):
        RESULTS.mkdir(parents=True, exist_ok=True)
        # `value` mixes metrics and info fields; a single non-numeric entry (an error
        # string, or a stale record from before arm/mode were dropped from info) makes the
        # whole parquet write fail. Coerce and set the strays aside rather than lose the run.
        num = pd.to_numeric(res["value"], errors="coerce")
        bad = res[num.isna()]
        if len(bad):
            print(f"  note: {len(bad)} non-numeric rows held out of the parquet "
                  f"({sorted(bad['metric'].unique())[:5]})")
            bad.to_csv(RESULTS / "trackC_nonnumeric.csv", index=False)
        res = res[num.notna()].copy()
        res["value"] = num[num.notna()]
        res.to_parquet(RESULTS / "trackC_results.parquet", index=False)
        for metric in ["tweedie_deviance_1.5", "gini_total_loss", "calibration_ratio"]:
            print(f"\n=== {metric} ===")
            print(summarize(res, ["cfg_arm", "cfg_mode"], metric).to_string(
                index=False, float_format=lambda x: f"{x:.4f}"))
        print("\nBenchmark: xgb_hurdle @ full 542,410 deviance = 82.675")
    return 0


if __name__ == "__main__":
    sys.exit(main())
