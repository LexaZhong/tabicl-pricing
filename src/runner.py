"""Resumable config runner.

A 50 W laptop GPU means the full matrix is a multi-session job, so every (config, seed)
result is cached to disk under a hash of its config. Re-running a track skips everything
already done and only executes what is new or changed. Interrupt and resume freely.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Callable
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE = RESULTS / "cache"


def config_key(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def run_config(
    config: dict,
    fn: Callable[[dict], dict],
    force: bool = False,
    verbose: bool = True,
    require_predictions: bool = False,
    require_keys: tuple[str, ...] = (),
) -> dict:
    """Execute `fn(config)` unless a cached result exists.

    A failure is cached too, as {"status": "error"}. That keeps one broken cell from
    stalling an overnight sweep, while leaving the failure visible in the results table
    instead of silently absent.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    key = config_key(config)
    path = CACHE / f"{key}.json"

    if path.exists() and not force:
        cached = json.loads(path.read_text())
        # A config hash covers the EXPERIMENT, not the code. When the metrics module or
        # the persistence path changes, the hash is unchanged and a stale record is served
        # silently -- which is exactly what happened on the first 3a run. Treat a cached
        # record that is missing required outputs as a miss, so the cache self-heals.
        stale = []
        if require_predictions and not (CACHE / f"{key}_pred.npz").exists():
            stale.append("predictions")
        missing = [k for k in require_keys if k not in cached.get("metrics", {})]
        stale += missing
        if not stale or cached.get("status") != "ok":
            if verbose:
                print(f"  [cached] {config.get('label', key)}")
            return cached
        if verbose:
            print(f"  [stale]  {config.get('label', key)} — missing {', '.join(stale)}")

    label = config.get("label", key)
    if verbose:
        print(f"  [run]    {label}", flush=True)

    try:
        out = fn(config)
        # Persist raw predictions separately from the metrics JSON. Without this, adding
        # ANY new metric later means re-fitting the whole matrix -- which is exactly what
        # the 1/Exposure artifact would have forced. Predictions are float32 and small.
        preds = out.pop("predictions", None)
        if preds is not None:
            import numpy as np

            np.savez_compressed(
                CACHE / f"{key}_pred.npz", **{k: np.asarray(v, dtype=np.float32)
                                              for k, v in preds.items()}
            )
        record = {"status": "ok", "config": config, **out}
    except Exception as exc:  # noqa: BLE001 - a failed cell must not stop the sweep
        record = {
            "status": "error",
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
        if verbose:
            print(f"  [ERROR]  {label}: {record['error']}", flush=True)

    path.write_text(json.dumps(record, indent=2, default=str))
    return record


def collect_results(pattern: str | None = None) -> pd.DataFrame:
    """Flatten every cached record into a tidy (config..., metric, value) frame."""
    rows: list[dict] = []
    if not CACHE.exists():
        return pd.DataFrame(rows)

    for path in sorted(CACHE.glob("*.json")):
        rec = json.loads(path.read_text())
        cfg = rec.get("config", {})
        if pattern and pattern not in str(cfg.get("track", "")):
            continue
        base = {f"cfg_{k}": v for k, v in cfg.items()}
        base["status"] = rec.get("status")
        if rec.get("status") != "ok":
            rows.append({**base, "metric": "error", "value": rec.get("error")})
            continue
        for k, v in rec.get("metrics", {}).items():
            rows.append({**base, "metric": k, "value": v})
        for k, v in rec.get("info", {}).items():
            rows.append({**base, "metric": f"info_{k}", "value": v})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, index: list[str], metric: str) -> pd.DataFrame:
    """Mean +/- SD across seeds -- the instability number is a headline, not a footnote."""
    sub = df[df["metric"] == metric].copy()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    g = sub.groupby(index)["value"]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n_seeds": g.size()}).reset_index()
