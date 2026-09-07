"""Track C: fine-tuning ladder, 3k -> 10k -> OOM boundary.

The claim under test is "fine-tuning made TabICL stable", so the OUTCOME VARIABLE IS THE
SEED SD, not the mean metric. Each size runs several seeds (different fine-tune subsets)
and the summary reports SD alongside the mean.

Two library defaults bite and are overridden in the wrapper:
  * max_data_size=10000 silently truncates the fine-tuning set, which would make the 10k+
    rungs of this ladder measure nothing;
  * eval_metric offers only roc_auc/log_loss/accuracy and mse/mae/r2 -- none is the pricing
    objective -- so final selection is on exposure-weighted Tweedie deviance here.

OOM is a recorded data point, not a crash: the mitigation ladder (n_estimators_finetune=1
-> freeze_col -> freeze_row -> CPU offload) is applied in order and the largest size that
fits is reported honestly.

Usage: python experiments/trackC_finetune.py [--sizes 3000,5000,10000] [--seeds 3]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src import metrics as M
from src.data import build_and_cache, exposure_stratified_subsample, random_split
from src.features.insurance_features import GBMFeaturePipeline
from src.models.tabicl_twostage import FinetunedTabICLTwoStage, TabICLTwoStage
from src.runner import RESULTS, collect_results, run_config, summarize

SIZES = [3_000, 5_000, 8_000, 10_000, 12_000, 16_000, 20_000]

# Applied in order when a size OOMs. Each entry is a kwargs patch.
OOM_LADDER = [
    {},
    {"n_estimators_finetune": 1},
    {"n_estimators_finetune": 1, "freeze_col": True},
    {"n_estimators_finetune": 1, "freeze_col": True, "freeze_row": True},
]


def make_runner(df, test_rows: int | None, epochs: int, time_limit: float | None):
    train_pool, test = random_split(df)
    if test_rows:
        test = test.iloc[:test_rows].copy()

    y_te = test["PurePremium"].to_numpy()
    e_te = test["Exposure"].to_numpy()

    def run(cfg: dict) -> dict:
        size, seed, mode = cfg["size"], cfg["seed"], cfg["mode"]
        train = exposure_stratified_subsample(train_pool, size, seed=seed)

        assert not (set(train["IDpol"]) & set(test["IDpol"])), "train/test overlap"

        pipe = GBMFeaturePipeline()
        X_tr = pipe.fit_transform(train, train["PurePremium"].to_numpy())
        X_te = pipe.transform(test)
        e_tr = train["Exposure"].to_numpy()
        hurdle = train["has_loss"].to_numpy()
        loss = train["TotalLoss"].to_numpy()

        info = {"n_train": len(train), "n_claims_stage2": int(hurdle.sum()), "mode": mode}

        if mode == "icl":
            model = TabICLTwoStage(context_size=size, seed=seed, batch_size=2)
            model.fit(X_tr, hurdle, loss, e_tr)
            pred = model.predict(X_te, e_te)
            info.update({k: v for k, v in model.info.__dict__.items() if k != "extra"})
        else:
            last_err = None
            for rung, patch in enumerate(OOM_LADDER):
                try:
                    torch.cuda.empty_cache()
                    gc.collect()
                    model = FinetunedTabICLTwoStage(
                        context_size=size,
                        seed=seed,
                        batch_size=2,
                        finetune_size=size,
                        epochs=epochs,
                        time_limit=time_limit,
                        **patch,
                    )
                    model.fit(X_tr, hurdle, loss, e_tr)
                    pred = model.predict(X_te, e_te)
                    info.update({k: v for k, v in model.info.__dict__.items() if k != "extra"})
                    info["oom_rung"] = rung
                    info["oom_patch"] = str(patch)
                    break
                except torch.cuda.OutOfMemoryError as exc:
                    last_err = exc
                    print(f"    OOM at rung {rung} ({patch}); escalating", flush=True)
                    torch.cuda.empty_cache()
                    gc.collect()
            else:
                raise RuntimeError(f"OOM at every rung for size={size}: {last_err}")

        metrics = M.evaluate_pure_premium(y_te, pred, e_te)
        p_hat = model.predict_claim_proba(X_te, e_te)
        info.update(M.stage1_metrics(test["has_loss"].to_numpy(), p_hat))
        info.update(M.exposure_monotonicity(p_hat, e_te))
        return {"metrics": metrics, "info": info}

    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default=",".join(str(s) for s in SIZES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--time-limit", type=float, default=900.0, help="seconds per fine-tune fit")
    ap.add_argument("--test-rows", type=int, default=0)
    ap.add_argument("--modes", type=str, default="icl,finetune")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    df, _ = build_and_cache()
    run = make_runner(df, args.test_rows or None, args.epochs, args.time_limit)

    configs = []
    for size in sizes:
        for mode in modes:
            for seed in range(args.seeds):
                configs.append(
                    {"track": "C", "size": size, "mode": mode, "seed": seed,
                     "epochs": args.epochs, "label": f"C/{size}/{mode}/s{seed}"}
                )

    print(f"Track C: {len(configs)} configs; sizes {sizes}\n")
    for cfg in configs:
        run_config(cfg, run)

    res = collect_results(pattern="C")
    if len(res):
        RESULTS.mkdir(parents=True, exist_ok=True)
        res.to_parquet(RESULTS / "trackC_results.parquet", index=False)
        for metric in ["gini_exposure_weighted", "tweedie_deviance_1.5"]:
            print(f"\n=== {metric}: mean and SEED SD (SD is the outcome here) ===")
            s = summarize(res, ["cfg_size", "cfg_mode"], metric)
            print(s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        vram = res[res["metric"] == "info_peak_vram_mib"]
        if len(vram):
            print("\n=== peak VRAM by size/mode (MiB of 12,227) ===")
            print(
                summarize(res, ["cfg_size", "cfg_mode"], "info_peak_vram_mib").to_string(
                    index=False, float_format=lambda x: f"{x:,.0f}"
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
