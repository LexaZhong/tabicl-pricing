"""Track 0.3: measure TabICL's real VRAM and wall-time curve on this 12 GB laptop GPU.

The published "50k samples in <10 s" figure is an H100 80 GB number. Every later track's
context size and fine-tune size is set from THIS table, not from that figure.

Each config fits a stage-1 classifier on `context` rows and scores a fixed 10k test slice,
recording peak VRAM and wall time. OOM is caught and recorded as a data point, not a crash.
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.data import build_and_cache, random_split
from src.features.insurance_features import GBMFeaturePipeline

RESULTS = Path(__file__).resolve().parent.parent / "results"

CONTEXT_SIZES = [5_000, 10_000, 20_000, 30_000, 50_000]
BATCH_SIZES = [4, 2, 1]
N_TEST = 10_000
PREDICT_CHUNK = 10_000


def measure(X_ctx, y_ctx, X_test, batch_size, offload_mode, use_amp) -> dict:
    from tabicl import TabICLClassifier

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    rec = {
        "context": len(X_ctx),
        "batch_size": batch_size,
        "offload_mode": str(offload_mode),
        "use_amp": str(use_amp),
        "n_test": len(X_test),
        "status": "ok",
        "fit_s": np.nan,
        "predict_s": np.nan,
        "peak_vram_mib": np.nan,
        "error": "",
    }
    try:
        clf = TabICLClassifier(
            n_estimators=8,
            batch_size=batch_size,
            device="cuda",
            use_amp=use_amp,
            use_fa3=False,  # Hopper-only; must stay off on Blackwell
            offload_mode=offload_mode,
            random_state=0,
            verbose=False,
        )
        t0 = time.perf_counter()
        clf.fit(X_ctx, y_ctx)
        torch.cuda.synchronize()
        rec["fit_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        for s in range(0, len(X_test), PREDICT_CHUNK):
            clf.predict_proba(X_test[s : s + PREDICT_CHUNK])
        torch.cuda.synchronize()
        rec["predict_s"] = time.perf_counter() - t0
        rec["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
        del clf
    except torch.cuda.OutOfMemoryError as exc:
        rec["status"] = "OOM"
        rec["error"] = str(exc)[:200]
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "ERROR"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        traceback.print_exc()
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    return rec


def main() -> int:
    df, _ = build_and_cache()
    train, test = random_split(df)

    pipe = GBMFeaturePipeline()
    X_tr = pipe.fit_transform(train, train["PurePremium"].to_numpy())
    X_te = pipe.transform(test)

    log_e_tr = np.log(train["Exposure"].to_numpy()).astype(np.float32).reshape(-1, 1)
    log_e_te = np.log(test["Exposure"].to_numpy()).astype(np.float32).reshape(-1, 1)
    X_tr = np.hstack([X_tr, log_e_tr])
    X_te = np.hstack([X_te, log_e_te])
    y_tr = train["has_loss"].to_numpy()

    X_test_slice = X_te[:N_TEST]
    print(f"features: {X_tr.shape[1]}  (incl. log_exposure)")
    print(f"scoring slice: {len(X_test_slice):,} rows\n")

    rng = np.random.default_rng(0)
    records = []

    for ctx in CONTEXT_SIZES:
        idx = rng.choice(len(X_tr), size=min(ctx, len(X_tr)), replace=False)
        Xc, yc = X_tr[idx], y_tr[idx]
        any_ok = False
        for bs in BATCH_SIZES:
            rec = measure(Xc, yc, X_test_slice, bs, "auto", "auto")
            records.append(rec)
            status = rec["status"]
            if status == "ok":
                any_ok = True
                print(
                    f"  ctx={ctx:>6,}  bs={bs}  fit={rec['fit_s']:6.1f}s  "
                    f"pred={rec['predict_s']:6.1f}s  peak={rec['peak_vram_mib']:7.0f} MiB"
                )
                break  # largest batch size that fits is the one we want
            print(f"  ctx={ctx:>6,}  bs={bs}  {status}: {rec['error'][:80]}")

        if not any_ok:
            print(f"  ctx={ctx:,} failed at every batch size; trying CPU offload")
            rec = measure(Xc, yc, X_test_slice, 1, "cpu", "auto")
            records.append(rec)
            if rec["status"] == "ok":
                print(
                    f"  ctx={ctx:>6,}  bs=1 offload=cpu  fit={rec['fit_s']:6.1f}s  "
                    f"pred={rec['predict_s']:6.1f}s  peak={rec['peak_vram_mib']:7.0f} MiB"
                )
            else:
                print(f"  ctx={ctx:,} unusable even with CPU offload -- stopping ladder")
                break

    out = pd.DataFrame(records)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "track0_vram_calibration.csv", index=False)

    print("\n=== calibration table ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    ok = out[out["status"] == "ok"]
    if len(ok):
        best = ok.loc[ok["context"].idxmax()]
        print(
            f"\nMAX FEASIBLE CONTEXT: {int(best['context']):,} "
            f"(batch_size={int(best['batch_size'])}, peak {best['peak_vram_mib']:.0f} MiB "
            f"of 12,227 MiB)"
        )
        # Extrapolate cost of scoring the full 135,603-row test set.
        per_row = best["predict_s"] / best["n_test"]
        print(f"scoring 135,603 rows at this context: ~{per_row * 135_603 / 60:.1f} min")
    print(f"\nwrote {RESULTS / 'track0_vram_calibration.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
