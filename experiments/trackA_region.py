"""Track A: region censoring (leave-one-region-out).

Train on Region != r, test on Region == r. With TargetEncoder fit on train only, the
censored region is an unseen category and collapses to the global target mean -- so
Region_te is CONSTANT across the test set and GLM/XGBoost lose all region signal. TabICL
sees an unseen level and is handicapped identically. The question this track answers is
which model best recovers regional risk from Area, log_density and VehBrand_te.

Every leakage guard is an assert, not a convention: target-encoder fitting, context
sampling and the balance step all draw from Region != r only.

Usage:  python experiments/trackA_region.py [--context 20000] [--seeds 5] [--regions R24,R82]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import metrics as M
from src.data import build_and_cache, region_split
from src.features.insurance_features import EngineeredFeaturePipeline, GBMFeaturePipeline
from src.models.gbm import DEFAULT_PARAMS, XGBFreqSev, XGBHurdle, XGBTweedie, sample_params
from src.models.glm import GLMHurdle, InterceptOnly, TweedieGLM, balance_to_portfolio
from src.models.tabicl_twostage import TabICLTwoStage
from src.runner import RESULTS, collect_results, run_config, summarize

# Chosen to span volume and character. R21 is deliberately excluded as the "small" pick:
# its loss cost of 1,474 vs a 167 portfolio mean is one huge claim, not a signal any model
# could learn, so it would only add noise. R43 is the clean small-region stress test.
DEFAULT_REGIONS = ["R24", "R82", "R11", "R93", "R43"]

_EPS = 1e-10


def _prep(train: pd.DataFrame, test: pd.DataFrame, kind: str):
    """Fit the feature pipeline on TRAIN ONLY and transform both sides."""
    pipe = GBMFeaturePipeline() if kind == "gbm" else EngineeredFeaturePipeline()
    X_tr = pipe.fit_transform(train, train["PurePremium"].to_numpy())
    X_te = pipe.transform(test)
    return pipe, X_tr, X_te


def _score(pred_te, train, test, pred_tr=None) -> dict:
    y_te = test["PurePremium"].to_numpy()
    e_te = test["Exposure"].to_numpy()
    out = M.evaluate_pure_premium(y_te, pred_te, e_te)

    # Balance factor is computed on TRAIN and applied to test -- computing it on test
    # would hand the model the answer it is being scored on.
    if pred_tr is not None:
        _, factor = balance_to_portfolio(
            pred_tr, train["PurePremium"].to_numpy(), train["Exposure"].to_numpy()
        )
        bal = M.evaluate_pure_premium(y_te, pred_te * factor, e_te)
        out["balance_factor"] = factor
        out["balanced_tweedie_deviance_1.5"] = bal["tweedie_deviance_1.5"]
        out["balanced_calibration_ratio"] = bal["calibration_ratio"]
    return out


CAP_QUANTILE = 0.995

_TUNE_CACHE: dict[tuple, dict] = {}


def tune_xgb(X, y, exposure, n_trials: int, seed: int, cache_key) -> dict:
    """Optuna over the reference search space, objective = weighted Tweedie deviance.

    Ported verbatim from trackB_curve.tune_xgb so the two tracks tune the incumbent
    identically. Without this Track A would fit XGBoost on DEFAULT_PARAMS while Track B
    gave it 30 trials, and every TabICL-vs-XGBoost comparison across the two tracks would
    be against a differently-handicapped baseline.

    Tuned ONCE per (model family, region) with a fixed seed and shared across the 5 run
    seeds, matching Track B's per-N caching -- the seed varies the fit, not the tuning.
    """
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


def make_runner(df: pd.DataFrame, context_size: int, cap_mode: str = "fixed",
                n_trials: int = 30):
    def run(cfg: dict) -> dict:
        region, model, seed = cfg["region"], cfg["model"], cfg["seed"]
        train, test = region_split(df, region)

        # --- leakage guards ---
        assert region not in set(train["Region"]), "censored region present in train"
        assert set(test["Region"]) == {region}, "test set is not the censored region"
        assert not (set(train["IDpol"]) & set(test["IDpol"])), "policy overlap"

        kind = "glm" if model.startswith("glm") or model == "intercept" else "gbm"
        pipe, X_tr, X_te = _prep(train, test, kind)

        # The censored region must be unseen by the encoder, so its encoding is constant.
        te_col = pipe.feature_names_.index("Region_te")
        n_unique_te = len(np.unique(X_te[:, te_col]))
        assert n_unique_te == 1, f"Region_te not constant on censored region ({n_unique_te})"

        e_tr = train["Exposure"].to_numpy()
        e_te = test["Exposure"].to_numpy()
        y_tr = train["PurePremium"].to_numpy()
        info: dict = {"n_train": len(train), "n_test": len(test), "region_te_levels": n_unique_te}

        if model == "one_over_exposure":
            # The trivial floor, and the one Track B proved is indispensable: it charges
            # every policy the SAME premium (pred * e = const), so it carries zero risk
            # information and scores exactly 0 on gini_total_loss. `intercept` does not
            # substitute -- it predicts a constant RATE, so its predicted loss is
            # proportional to exposure and it ranks by exposure alone, which on this data
            # is worth 0.29-0.37 total-loss Gini in some regions.
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
            # Tuning is cached per region and shared by both XGBoost variants, exactly as
            # Track B shares one tuning per N.
            params = tune_xgb(X_tr, y_tr, e_tr, n_trials, seed=0, cache_key=("xgb", region))
            params = {k: v for k, v in params.items() if k != "random_state"}
            info["tuned"] = n_trials > 0
            info["n_trials"] = n_trials
            if model == "xgb_tweedie":
                m = XGBTweedie(random_state=seed, **params).fit(X_tr, y_tr, exposure=e_tr)
                pred_te, pred_tr = m.predict(X_te), m.predict(X_tr)
            else:
                m = XGBHurdle(random_state=seed, **params).fit(
                    X_tr, train["has_loss"].to_numpy(), train["TotalLoss"].to_numpy(), e_tr
                )
                pred_te, pred_tr = m.predict(X_te, e_te), m.predict(X_tr, e_tr)
        elif model == "xgb_freqsev":
            m = XGBFreqSev(random_state=seed).fit(
                X_tr,
                train["ClaimNb"].to_numpy(),
                train["Severity"].fillna(0).to_numpy(),
                e_tr,
                (train["has_loss"] == 1).to_numpy(),
            )
            pred_te, pred_tr = m.predict(X_te, e_te), m.predict(X_tr, e_tr)
        elif model.startswith("tabicl"):
            sampler = "random" if model.endswith("_random") else "enriched"
            cap_kw = {}
            if cap_mode == "fixed":
                # The cap is declared from the TRAINING side only (Region != r). Taking it
                # from the whole frame would leak the censored region's severity tail into
                # the model that is supposed to have never seen that region.
                tr_claims = train.loc[train["has_loss"] == 1, "TotalLoss"].to_numpy()
                cap_kw = dict(
                    cap_mode="fixed",
                    fixed_cap=float(np.quantile(tr_claims, CAP_QUANTILE)),
                    fixed_floor=float(np.quantile(tr_claims, 1.0 - CAP_QUANTILE)),
                )
            m = TabICLTwoStage(
                context_size=context_size,
                n_contexts=cfg.get("n_contexts", 1),
                sampler=sampler,
                seed=seed,
                batch_size=cfg.get("batch_size", 2),
                **cap_kw,
            )
            m.fit(X_tr, train["has_loss"].to_numpy(), train["TotalLoss"].to_numpy(), e_tr)
            pred_te = m.predict(X_te, e_te)
            pred_tr = None  # scoring 500k train rows through ICL is not worth the GPU time

            # Reuse the p_hat computed inside predict(). Calling predict_claim_proba again
            # costs a second full ICL pass over the test region -- up to 160k rows for R24
            # -- for numbers we already have.
            p_hat = m._last_p_hat
            if p_hat is None:
                p_hat = m.predict_claim_proba(X_te, e_te)
            info.update(m.info.__dict__)
            info.pop("extra", None)
            info.update(M.stage1_metrics(test["has_loss"].to_numpy(), p_hat))
            info.update(M.exposure_monotonicity(p_hat, e_te))
        else:
            raise ValueError(f"unknown model {model!r}")

        # Persist predictions. Track B was run without this and adding gini_total_loss
        # later meant it could not be recomputed -- the whole matrix had to be re-fitted.
        # Predictions are float32 and small; a re-fit of this track is hours.
        return {
            "metrics": _score(pred_te, train, test, pred_tr),
            "info": info,
            "predictions": {"pred": pred_te, "actual": test["PurePremium"].to_numpy(),
                            "exposure": e_te},
        }

    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=20_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--regions", type=str, default=",".join(DEFAULT_REGIONS))
    ap.add_argument("--models", type=str, default="")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--trials", type=int, default=30,
                    help="Optuna trials for XGBoost, matching trackB_curve's default; "
                         "0 disables tuning and uses DEFAULT_PARAMS")
    ap.add_argument("--cap-mode", default="fixed", choices=["fixed", "empirical"],
                    help="fixed pins the severity cap to the train side's q99.5; empirical "
                         "reproduces Track B's per-draw cap, which swung 14.6k-168.2k")
    args = ap.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    # Deterministic models take 1 seed because Track A does not subsample: the training set
    # is ALL of Region != r, so a GLM refit with a different seed returns the same numbers.
    # (This differs from Track B, where the seed selected the draw and 1 seed left the
    # incumbent with no error bar.) Only TabICL's context draw and XGBoost's seed vary.
    deterministic = ["one_over_exposure", "intercept", "glm_tweedie", "glm_hurdle"]
    stochastic = ["xgb_tweedie", "xgb_hurdle", "tabicl"]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        deterministic = [m for m in deterministic if m in wanted]
        stochastic = [m for m in stochastic if m in wanted]

    df, _ = build_and_cache()
    run = make_runner(df, args.context, args.cap_mode, args.trials)

    configs = []
    for region in regions:
        for model in deterministic:
            configs.append(
                {"track": "A", "region": region, "model": model, "seed": 0,
                 "context": args.context, "trials": args.trials, "cap": args.cap_mode,
                 "label": f"A/{region}/{model}/s0"}
            )
        for model in stochastic:
            for seed in range(args.seeds):
                configs.append(
                    {"track": "A", "region": region, "model": model, "seed": seed,
                     "context": args.context, "batch_size": args.batch_size,
                     "trials": args.trials, "cap": args.cap_mode,
                     "label": f"A/{region}/{model}/s{seed}"}
                )

    print(f"Track A: {len(configs)} configs across {len(regions)} regions "
          f"(cap={args.cap_mode})\n")
    for cfg in configs:
        run_config(cfg, run, require_predictions=True)

    res = collect_results(pattern="A")
    if len(res):
        RESULTS.mkdir(parents=True, exist_ok=True)
        res.to_parquet(RESULTS / "trackA_results.parquet", index=False)
        for metric in ["gini_exposure_weighted", "tweedie_deviance_1.5"]:
            print(f"\n=== {metric} (mean +/- sd over seeds) ===")
            s = summarize(res, ["cfg_region", "cfg_model"], metric)
            pivot = s.pivot(index="cfg_model", columns="cfg_region", values="mean")
            print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
            sd = s.pivot(index="cfg_model", columns="cfg_region", values="sd")
            print("\n  seed SD:")
            print(sd.to_string(float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
