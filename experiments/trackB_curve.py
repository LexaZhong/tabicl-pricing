"""Track B: thin-data learning curve -- the headline exhibit.

Trains every model at N = 500 ... 542,410 against the SAME fixed 135,603-row test set,
and reports the crossover N where XGBoost overtakes TabICL. That crossover is the business
answer: "below N ~ X, the foundation model wins".

Fairness: XGBoost is re-tuned with Optuna at each N. An untuned GBM at N=500 would flatter
TabICL and the result would not survive review. Tuning is done once per N on seed 0 and the
params are reused across seeds -- tuning per seed as well would multiply GPU-free CPU cost
without changing the comparison.

Usage: python experiments/trackB_curve.py [--context 20000] [--seeds 5] [--trials 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import checks as C
from src import metrics as M
from src.data import build_and_cache, exposure_stratified_subsample, random_split
from src.features.insurance_features import (
    EngineeredFeaturePipeline,
    GBMFeaturePipeline,
    RawFeaturePipeline,
)
from src.models.gbm import DEFAULT_PARAMS, XGBHurdle, XGBTweedie, sample_params
from src.models.glm import GLMHurdle, InterceptOnly, TweedieGLM, balance_to_portfolio
from src.models.tabicl_twostage import TabICLTwoStage
from src.runner import RESULTS, collect_results, run_config, summarize

# Unified grid shared with the resampling study so the curve and the stability study sit
# on the same sizes. The N <= 2,000 region is already answered (no model there beats the
# trivial 1/exposure floor) and is reported from the completed run rather than re-run.
N_GRID = [5_000, 10_000, 20_000, 100_000, 542_410]
BIG_N = {542_410}  # fewer draws here purely to control wall-clock

_TUNE_CACHE: dict[tuple, dict] = {}
_EPS = 1e-10


def tune_xgb(X, y, exposure, n_trials: int, seed: int, cache_key) -> dict:
    """Optuna over the reference repo's search space, objective = weighted Tweedie deviance."""
    if cache_key in _TUNE_CACHE:
        return _TUNE_CACHE[cache_key]
    if n_trials <= 0 or len(X) < 200:
        _TUNE_CACHE[cache_key] = dict(DEFAULT_PARAMS)
        return _TUNE_CACHE[cache_key]

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, va = idx[:cut], idx[cut:]

    def objective(trial):
        params = sample_params(trial)
        m = XGBTweedie(random_state=seed, **params).fit(X[tr], y[tr], exposure=exposure[tr])
        return M.tweedie_deviance(y[va], m.predict(X[va]), exposure[va], power=1.5)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = {**DEFAULT_PARAMS, **study.best_params}
    _TUNE_CACHE[cache_key] = best
    return best


def make_runner(df: pd.DataFrame, context_size: int, n_trials: int, test_rows: int | None):
    train_pool, test = random_split(df)
    if test_rows:
        test = test.iloc[:test_rows].copy()

    y_te = test["PurePremium"].to_numpy()
    e_te = test["Exposure"].to_numpy()

    def run(cfg: dict) -> dict:
        n, model, seed = cfg["n"], cfg["model"], cfg["seed"]
        train = exposure_stratified_subsample(train_pool, n, seed=seed)

        assert not (set(train["IDpol"]) & set(test["IDpol"])), "train/test overlap"

        # `tabicl_raw` is the feature-engineering ablation: identical model, identical
        # draw (same seed -> same exposure_stratified_subsample), only the feature set
        # differs. TabICL's pitch is that it removes the actuarial feature-engineering
        # step, so the paired gap against `tabicl` measures exactly what that step is worth.
        if model == "tabicl_raw":
            kind = "raw"
        elif model.startswith("glm") or model == "intercept":
            kind = "glm"
        else:
            kind = "gbm"
        pipe = {"gbm": GBMFeaturePipeline, "glm": EngineeredFeaturePipeline,
                "raw": RawFeaturePipeline}[kind]()
        y_tr = train["PurePremium"].to_numpy()
        X_tr = pipe.fit_transform(train, y_tr)
        X_te = pipe.transform(test)
        e_tr = train["Exposure"].to_numpy()

        n_claims = int(((train["has_loss"] == 1)).sum())
        info = {"n_train": len(train), "n_claims_stage2": n_claims,
                "claim_rate": float(train["has_loss"].mean())}

        if model == "one_over_exposure":
            # Trivial floor: zero risk information, charges every policy the same
            # premium. Scores wGini 0.4676 on the rate-based metric, so it is the line
            # every other model's rate-Gini must clear to mean anything.
            base = (train["TotalLoss"].sum() / max(e_tr.sum(), _EPS)) * e_tr.mean()
            pred_te, pred_tr = base / e_te, base / e_tr
        elif model == "intercept":
            m = InterceptOnly().fit(X_tr, y_tr, e_tr)
            pred_te, pred_tr = m.predict(X_te), m.predict(X_tr)
        elif model == "glm_tweedie":
            m = TweedieGLM().fit(X_tr, y_tr, exposure=e_tr)
            pred_te, pred_tr = m.predict(X_te), m.predict(X_tr)
        elif model == "glm_hurdle":
            m = GLMHurdle().fit(
                X_tr, train["has_loss"].to_numpy(), train["TotalLoss"].to_numpy(), e_tr
            )
            pred_te, pred_tr = m.predict(X_te, e_te), m.predict(X_tr, e_tr)
        elif model in ("xgb_tweedie", "xgb_hurdle"):
            params = tune_xgb(X_tr, y_tr, e_tr, n_trials, seed=0, cache_key=("xgb", n))
            params = {k: v for k, v in params.items() if k != "random_state"}
            info["tuned"] = n_trials > 0
            if model == "xgb_tweedie":
                m = XGBTweedie(random_state=seed, **params).fit(X_tr, y_tr, exposure=e_tr)
                pred_te, pred_tr = m.predict(X_te), m.predict(X_tr)
            else:
                m = XGBHurdle(random_state=seed, **params).fit(
                    X_tr, train["has_loss"].to_numpy(), train["TotalLoss"].to_numpy(), e_tr
                )
                pred_te, pred_tr = m.predict(X_te, e_te), m.predict(X_tr, e_tr)
        elif model in ("tabicl", "tabicl_raw"):
            # At small N the whole training set IS the context -- TabICL's home turf.
            m = TabICLTwoStage(
                context_size=min(context_size, len(train)),
                n_contexts=cfg.get("n_contexts", 1),
                seed=seed,
                batch_size=cfg.get("batch_size", 2),
            )
            m.fit(X_tr, train["has_loss"].to_numpy(), train["TotalLoss"].to_numpy(), e_tr)
            pred_te = m.predict(X_te, e_te)
            pred_tr = None
            info.update({k: v for k, v in m.info.__dict__.items() if k != "extra"})
            info.update(m.info.extra)  # carries severity_clipped_frac
            info["n_features"] = int(X_tr.shape[1])
            p_hat = m._last_p_hat  # cached by predict(); recomputing costs a full ICL pass
            info.update(M.stage1_metrics(test["has_loss"].to_numpy(), p_hat))
            info.update(M.exposure_monotonicity(p_hat, e_te))
        else:
            raise ValueError(model)

        metrics = M.evaluate_pure_premium(y_te, pred_te, e_te)
        if pred_tr is not None:
            _, factor = balance_to_portfolio(pred_tr, y_tr, e_tr)
            metrics["balance_factor"] = factor
            metrics["balanced_tweedie_deviance_1.5"] = M.tweedie_deviance(
                y_te, pred_te * factor, e_te, power=1.5
            )

        # Sample characterisation -- lets check 11 spot a draw whose composition, not the
        # model, moved the metric.
        info.update({
            "mean_exposure": float(e_tr.mean()),
            "total_loss": float(train["TotalLoss"].sum()),
            "max_loss": float(train["TotalLoss"].max()),
        })

        flags = C.check_config(model, pred_te, y_te, e_te, metrics=metrics, info=info)
        for f in flags:
            if f.severity in ("FAIL", "WARN"):
                print(f"      {f}", flush=True)

        return {
            "metrics": metrics,
            "info": info,
            "flags": [{"check": f.check, "severity": f.severity, "message": f.message}
                      for f in flags],
            "predictions": {"pred_test": pred_te},
        }

    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--big-seeds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--test-rows", type=int, default=0, help="0 = full 135,603 test set")
    ap.add_argument("--n-grid", type=str, default="")
    ap.add_argument("--models", type=str, default="")
    args = ap.parse_args()

    grid = [int(x) for x in args.n_grid.split(",")] if args.n_grid else N_GRID
    models = ["intercept", "one_over_exposure", "glm_tweedie", "glm_hurdle",
              "xgb_tweedie", "xgb_hurdle", "tabicl", "tabicl_raw"]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in models if m in wanted]

    df, _ = build_and_cache()
    run = make_runner(df, args.context, args.trials, args.test_rows or None)

    configs = []
    for n in grid:
        n_seeds = args.big_seeds if n in BIG_N else args.seeds
        for model in models:
            # Every model at a given N gets the SAME number of draws. The fitter is
            # deterministic for the GLMs, but the subsample is not -- running them once
            # while TabICL/XGBoost got a 5-draw mean gave the incumbent an estimate with
            # ~sqrt(5) larger standard error, which is why its curve looked erratic.
            for seed in range(n_seeds):
                configs.append(
                    {"track": "B", "n": n, "model": model, "seed": seed,
                     "context": args.context, "trials": args.trials,
                     "test_rows": args.test_rows or 135_603,
                     "label": f"B/N{n}/{model}/s{seed}"}
                )

    print(f"Track B: {len(configs)} configs over N grid {grid}\n")
    for cfg in configs:
        # Re-run anything cached without saved predictions or without the artifact-free
        # metric, rather than silently serving a pre-fix record.
        run_config(cfg, run, require_predictions=True,
                   require_keys=("gini_total_loss", "gini_fixed_exposure"))

    res = collect_results(pattern="B")
    if len(res):
        RESULTS.mkdir(parents=True, exist_ok=True)
        res.to_parquet(RESULTS / "trackB_results.parquet", index=False)
        for metric in ["gini_exposure_weighted", "tweedie_deviance_1.5"]:
            print(f"\n=== {metric} vs N (mean over seeds) ===")
            s = summarize(res, ["cfg_n", "cfg_model"], metric)
            print(
                s.pivot(index="cfg_n", columns="cfg_model", values="mean").to_string(
                    float_format=lambda x: f"{x:.4f}"
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
