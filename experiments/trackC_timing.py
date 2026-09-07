"""Track C planning: measure what a fine-tune actually costs at N=10k and 20k.

Wall-clock for the ladder cannot be extrapolated from Track B's inference timings,
because fine-tuning has a different cost structure:

  * `max_data_size` doubles as `max_chunk_size` in the meta-batch iterator
    (_finetune/base.py:893), so with max_data_size=N the whole fine-tune set is ONE
    chunk -- one optimizer step per epoch, over a sequence of T=N rows.
  * That step backprops through dataset-level attention over T, so cost grows
    super-linearly in N and activation memory is the binding constraint, not
    inference VRAM (Track 0.3 measured inference only).
  * Each epoch also runs a full validation pass: ICL over the N-row context
    predicting the 10% holdout.

So this script measures, per stage and per rung: seconds/epoch, seconds for the
one-off setup, and peak VRAM -- and reports OOM as a data point with the mitigation
rung that cleared it, exactly as trackC_finetune.py does.

Usage: python experiments/trackC_timing.py [--sizes 10000,20000] [--epochs 3]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.data import build_and_cache, exposure_stratified_subsample, random_split
from src.features.insurance_features import GBMFeaturePipeline
from src.models.tabicl_twostage import _append_log_exposure

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Same ladder trackC_finetune.py applies, so a rung measured here is the rung the
# real run would use.
OOM_LADDER = [
    {},
    {"n_estimators_finetune": 1},
    {"n_estimators_finetune": 1, "freeze_col": True},
    {"n_estimators_finetune": 1, "freeze_col": True, "freeze_row": True},
]

_EPOCH_RE = re.compile(r"epoch (\d+)/(\d+) \|.*?time=([0-9.]+)s")


class EpochTimeCollector(logging.Handler):
    """The library logs `epoch i/n | train_loss=... | val_x=... | time=Ns` per epoch."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.times: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = _EPOCH_RE.search(record.getMessage())
        if m:
            self.times.append(float(m.group(3)))

    def reset(self) -> None:
        self.times = []


def _ft_kwargs(n: int, seed: int, epochs: int, patch: dict, max_data_size: int | None = None) -> dict:
    """Mirrors FinetunedTabICLTwoStage._common_ft_kwargs so timings transfer.

    `max_data_size` is the one knob that changes the shape of the run, not just its
    speed: it is also `max_chunk_size`, so it sets BOTH the fine-tune context length
    and the number of gradient steps per epoch (ceil(n / max_data_size)). Passing it
    below `n` buys many cheap steps at the price of a shorter context than inference
    will use.
    """
    kw = dict(
        epochs=epochs,
        learning_rate=1e-5,
        n_estimators_finetune=2,
        n_estimators_inference=8,
        max_data_size=max_data_size or n,
        validation_split_ratio=0.1,
        early_stopping=False,  # we want every epoch timed, not an early exit
        patience=8,
        amp=True,
        freeze_col=False,
        freeze_row=False,
        device="cuda",
        random_state=seed,
        verbose=False,
    )
    kw.update(patch)
    return kw


def _fit_timed(make, X, y, collector: EpochTimeCollector) -> dict:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    collector.reset()
    t0 = time.perf_counter()
    model = make()
    model.fit(X, y)
    total = time.perf_counter() - t0
    times = list(collector.times)
    return {
        "model": model,
        "total_s": total,
        "epoch_s": times,
        # Everything not inside a timed epoch: preprocessing, weight download/load,
        # optimizer construction, final best-state restore.
        "setup_s": total - sum(times),
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def _predict_timed(model, X, chunk: int = 20_000) -> dict:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    n = min(len(X), chunk)
    if hasattr(model, "predict_proba"):
        model.predict_proba(X[:n])
    else:
        model.predict(X[:n])
    dt = time.perf_counter() - t0
    return {"rows": n, "seconds": dt, "s_per_1k": dt / n * 1000,
            "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20}


def bench_size(df_train, df_test, n: int, seed: int, epochs: int,
               max_data_size: int | None = None) -> dict:
    from tabicl import FinetunedTabICLClassifier, FinetunedTabICLRegressor

    collector = EpochTimeCollector()
    logging.getLogger("tabicl").addHandler(collector)
    logging.getLogger("tabicl").setLevel(logging.INFO)

    train = exposure_stratified_subsample(df_train, n, seed=seed)
    pipe = GBMFeaturePipeline()
    X_tr = pipe.fit_transform(train, train["PurePremium"].to_numpy())
    X_te = pipe.transform(df_test)
    e_tr = train["Exposure"].to_numpy()
    Xe = _append_log_exposure(X_tr, e_tr)
    Xe_te = _append_log_exposure(X_te, df_test["Exposure"].to_numpy())

    hurdle = train["has_loss"].to_numpy().astype(int)
    loss = train["TotalLoss"].to_numpy()

    mds = max_data_size or n
    steps_per_epoch = max(1, -(-n // mds))
    out: dict = {"n": n, "seed": seed, "epochs_measured": epochs,
                 "max_data_size": mds, "steps_per_epoch": steps_per_epoch,
                 "n_features": Xe.shape[1], "n_claims_stage2": int(hurdle.sum())}
    print(f"\n{'='*70}\nN={n:,}  features={Xe.shape[1]}  stage-2 rows={hurdle.sum():,}  "
          f"max_data_size={mds:,} -> {steps_per_epoch} step(s)/epoch\n{'='*70}", flush=True)

    # ---- stage 1: classifier over all N rows (the expensive stage) ----
    for rung, patch in enumerate(OOM_LADDER):
        try:
            print(f"  stage 1, OOM rung {rung} {patch or '(no mitigation)'} ...", flush=True)
            r = _fit_timed(
                lambda: FinetunedTabICLClassifier(
                    eval_metric="log_loss", **_ft_kwargs(n, seed, epochs, patch, mds)
                ),
                Xe, hurdle, collector,
            )
            clf = r.pop("model")
            r.update({"oom_rung": rung, "oom_patch": str(patch)})
            out["stage1"] = r
            out["stage1_predict"] = _predict_timed(clf, Xe_te)
            del clf
            break
        except torch.cuda.OutOfMemoryError as exc:
            print(f"    OOM: {str(exc)[:120]}", flush=True)
            out.setdefault("stage1_oom_rungs", []).append(rung)
            torch.cuda.empty_cache()
            gc.collect()
    else:
        out["stage1"] = {"failed": "OOM at every rung"}

    torch.cuda.empty_cache()
    gc.collect()

    # ---- stage 2: regressor on claims-only rows (~3.5% of N, so much cheaper) ----
    mask = (hurdle == 1) & (loss > 0)
    y2 = loss[mask]
    cap = float(np.quantile(y2, 0.995))
    floor = float(np.quantile(y2, 0.005))
    X2 = Xe[mask]
    y2c = np.clip(y2, floor, cap)
    n2 = len(X2)
    for rung, patch in enumerate(OOM_LADDER):
        try:
            print(f"  stage 2 ({n2:,} rows), OOM rung {rung} ...", flush=True)
            r = _fit_timed(
                lambda: FinetunedTabICLRegressor(
                    eval_metric="mse", **_ft_kwargs(max(n2, 1), seed, epochs, patch)
                ),
                X2, y2c, collector,
            )
            reg = r.pop("model")
            r.update({"oom_rung": rung, "oom_patch": str(patch), "n_rows": n2})
            out["stage2"] = r
            out["stage2_predict"] = _predict_timed(reg, Xe_te)
            del reg
            break
        except torch.cuda.OutOfMemoryError as exc:
            print(f"    OOM: {str(exc)[:120]}", flush=True)
            out.setdefault("stage2_oom_rungs", []).append(rung)
            torch.cuda.empty_cache()
            gc.collect()
    else:
        out["stage2"] = {"failed": "OOM at every rung"}

    logging.getLogger("tabicl").removeHandler(collector)
    torch.cuda.empty_cache()
    gc.collect()
    return out


def project(out: dict, epochs_real: int, test_rows: int) -> dict:
    """One fine-tuned config = stage1 + stage2 fit, then scoring the full test set."""
    def per_epoch(stage: dict) -> float:
        t = stage.get("epoch_s") or []
        # Epoch 1 carries lazy CUDA/cuDNN warm-up; use the steady-state epochs when
        # there are enough of them.
        return float(np.median(t[1:] if len(t) > 2 else t)) if t else float("nan")

    s1, s2 = out.get("stage1", {}), out.get("stage2", {})
    e1, e2 = per_epoch(s1), per_epoch(s2)
    fit_s = (s1.get("setup_s", 0) + e1 * epochs_real) + (s2.get("setup_s", 0) + e2 * epochs_real)
    pred_s = sum(
        out.get(k, {}).get("s_per_1k", 0) * test_rows / 1000
        for k in ("stage1_predict", "stage2_predict")
    )
    return {
        "stage1_s_per_epoch": e1,
        "stage2_s_per_epoch": e2,
        "fit_s": fit_s,
        "predict_s": pred_s,
        "config_s": fit_s + pred_s,
        "config_min": (fit_s + pred_s) / 60,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000,20000")
    ap.add_argument("--max-data-size", type=int, default=0,
                    help="override the fine-tune chunk size; 0 means use N (one step/epoch)")
    ap.add_argument("--epochs", type=int, default=3, help="epochs to MEASURE (not the real run)")
    ap.add_argument("--epochs-real", type=int, default=30, help="epochs the real ladder would use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-rows", type=int, default=135_603)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/2**20:,.0f} MiB  torch {torch.__version__}")

    df, _ = build_and_cache()
    train_pool, test = random_split(df)

    results = []
    for n in [int(s) for s in args.sizes.split(",")]:
        out = bench_size(train_pool, test, n, args.seed, args.epochs,
                         args.max_data_size or None)
        out["projection"] = project(out, args.epochs_real, args.test_rows)
        results.append(out)
        print(json.dumps(out["projection"], indent=2), flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    # Tag by chunk size so a chunked run does not overwrite the one-step-per-epoch run.
    tag = f"_mds{args.max_data_size}" if args.max_data_size else ""
    path = RESULTS / f"trackC_timing{tag}.json"
    prior = json.loads(path.read_text()) if path.exists() else []
    keep = [r for r in prior if r["n"] not in {o["n"] for o in results}]
    path.write_text(json.dumps(keep + results, indent=2))

    print(f"\n{'='*70}\nSUMMARY (projected to {args.epochs_real} epochs, {args.test_rows:,} test rows)\n{'='*70}")
    hdr = f"{'N':>8} {'s1/epoch':>10} {'s2/epoch':>10} {'fit':>9} {'predict':>9} {'per config':>12} {'peak VRAM':>11} {'rung':>5}"
    print(hdr)
    for out in results:
        p = out["projection"]
        vram = max(out.get("stage1", {}).get("peak_vram_mib", 0),
                   out.get("stage2", {}).get("peak_vram_mib", 0),
                   out.get("stage1_predict", {}).get("peak_vram_mib", 0))
        print(f"{out['n']:>8,} {p['stage1_s_per_epoch']:>10.1f} {p['stage2_s_per_epoch']:>10.1f} "
              f"{p['fit_s']/60:>8.1f}m {p['predict_s']/60:>8.1f}m {p['config_min']:>11.1f}m "
              f"{vram:>10,.0f} {out.get('stage1',{}).get('oom_rung','-'):>5}")
    print(f"\nWritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
