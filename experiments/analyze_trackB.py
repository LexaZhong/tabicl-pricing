"""Track B analysis: curves, crossover, stability, lift charts, and the checks report.

Read-only over the runner cache -- it never refits anything.

Two conventions that matter for reading the output:
  * COMPETITORS are compared with each other; REFERENCE lines (intercept, one_over_exposure)
    are drawn as horizontal guides, never ranked against models.
  * Results are filtered to a single test set. The 1/exposure floor is test-set specific
    (0.4676 on the full 135,603 rows, 0.3606 on a 20k subset), so mixing them would compare
    numbers against the wrong line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import checks as C
from src import metrics as M
from src.data import build_and_cache, random_split
from src.runner import CACHE, RESULTS, collect_results, summarize

REPORTS = Path(__file__).resolve().parent.parent / "reports"
FULL_TEST_ROWS = 135_603
POOL_SIZE = 542_410  # at this N the subsample is the whole pool: no draw-to-draw variance

COMPETITORS = ["tabicl", "xgb_hurdle", "xgb_tweedie", "glm_hurdle", "glm_tweedie"]
REFERENCES = ["one_over_exposure", "intercept"]
ORDER = COMPETITORS + REFERENCES

COLORS = {
    "tabicl": "#c1440e", "xgb_hurdle": "#1f4e79", "xgb_tweedie": "#5b9bd5",
    "glm_hurdle": "#2e7d32", "glm_tweedie": "#81c784",
    "intercept": "#9e9e9e", "one_over_exposure": "#b39ddb",
}

METRICS = [
    ("gini_exposure_weighted", "exposure-weighted Gini (rate)", True),
    ("gini_total_loss", "Gini on total loss (artifact-free)", True),
    ("gini_fixed_exposure", "Gini on full-term policies (exposure >= 0.95)", True),
    ("tweedie_deviance_1.5", "mean Tweedie deviance (p=1.5)", False),
]


def trivial_floor() -> dict[str, float]:
    """The 1/exposure degenerate solution on the full test set. Not a fitted model."""
    df, _ = build_and_cache()
    _, test = random_split(df)
    y = test["PurePremium"].to_numpy()
    e = test["Exposure"].to_numpy()
    pred = (test["TotalLoss"].sum() / e.sum()) * e.mean() / e
    return M.evaluate_pure_premium(y, pred, e)


def load_results() -> pd.DataFrame:
    res = collect_results(pattern="B")
    if not len(res) or "cfg_track" not in res:
        return pd.DataFrame()
    res = res[res["cfg_track"] == "B"]
    if "cfg_test_rows" in res:
        keep = pd.to_numeric(res["cfg_test_rows"], errors="coerce").fillna(FULL_TEST_ROWS)
        res = res[keep == FULL_TEST_ROWS]
    return res


def load_preds() -> dict[tuple, np.ndarray]:
    """(n, model, seed) -> saved test predictions, full test set only."""
    out: dict[tuple, np.ndarray] = {}
    if not CACHE.exists():
        return out
    for jpath in sorted(CACHE.glob("*.json")):
        npz = jpath.with_name(f"{jpath.stem}_pred.npz")
        if not npz.exists():
            continue
        cfg = json.loads(jpath.read_text()).get("config", {})
        if cfg.get("track") != "B" or cfg.get("test_rows", FULL_TEST_ROWS) != FULL_TEST_ROWS:
            continue
        with np.load(npz) as z:
            if "pred_test" in z:
                out[(cfg.get("n"), cfg.get("model"), cfg.get("seed"))] = z["pred_test"]
    return out


def curve_table(res, metric):
    s = summarize(res, ["cfg_n", "cfg_model"], metric)
    if not len(s):
        return None, None, None
    mean = s.pivot(index="cfg_n", columns="cfg_model", values="mean")
    sd = s.pivot(index="cfg_n", columns="cfg_model", values="sd")
    nseeds = s.pivot(index="cfg_n", columns="cfg_model", values="n_seeds")
    cols = [c for c in ORDER if c in mean.columns]
    return mean[cols], sd[cols], nseeds[cols]


def significance(mean, sd, nseeds, a, b, higher_better):
    """Two-sample comparison per N, so 'A beats B' is not read off noise."""
    rows = []
    for n in mean.index:
        if a not in mean or b not in mean:
            continue
        ma, mb = mean.loc[n, a], mean.loc[n, b]
        sa, sb = sd.loc[n, a], sd.loc[n, b]
        na, nb = nseeds.loc[n, a], nseeds.loc[n, b]
        if not all(np.isfinite([ma, mb, na, nb])) or na < 2 or nb < 2:
            continue
        se = float(np.sqrt(np.nan_to_num(sa) ** 2 / na + np.nan_to_num(sb) ** 2 / nb))
        delta = (ma - mb) if higher_better else (mb - ma)
        z = delta / se if se > 0 else np.nan
        rows.append({
            "N": int(n), a: ma, b: mb, "delta": delta, "SE": se, "z": z,
            "verdict": ("A wins" if z > 2 else "B wins" if z < -2 else "not distinguishable"),
        })
    return pd.DataFrame(rows)


def plot_curve(mean, sd, label, floor, outfile, higher_better):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col in mean.columns:
        m = mean[col].dropna()
        if not len(m):
            continue
        is_ref = col in REFERENCES
        e = sd[col].reindex(m.index).fillna(0)
        ax.plot(m.index, m.values, marker="o" if not is_ref else None, ms=4, label=col,
                color=COLORS.get(col, "#555"), lw=2 if col == "tabicl" else 1.3,
                ls=":" if is_ref else "-", alpha=0.75 if is_ref else 1.0)
        if not is_ref:
            ax.fill_between(m.index, m - e, m + e, color=COLORS.get(col, "#555"), alpha=0.13)

    if floor is not None and np.isfinite(floor):
        ax.axhline(floor, ls="--", lw=1.4, color="#7e57c2")
        ax.text(mean.index.min(), floor, "  1/exposure floor (no risk information)",
                va="bottom", ha="left", fontsize=8, color="#7e57c2")

    ax.set_xscale("log")
    ax.set_xlabel("training rows (N, log scale)")
    ax.set_ylabel(label)
    ax.set_title(f"Track B: {label}  ({'higher' if higher_better else 'lower'} is better)")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f"  wrote {outfile.name}")


def plot_lift(y, e, by_model, n, outfile):
    """Actual vs predicted loss cost, 10 equal-exposure buckets, one panel per model."""
    models = [m for m in ORDER if m in by_model]
    fig, axes = plt.subplots(1, len(models), figsize=(3.9 * len(models), 4.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, models):
        lt = M.lift_table(y, by_model[name], e, n_buckets=10)
        ax.bar(lt["bucket"], lt["actual_loss_cost"], color="#d0d4da", width=0.78, label="actual")
        ax.plot(lt["bucket"], lt["predicted_loss_cost"], marker="o", ms=4, lw=2,
                color=COLORS.get(name, "#c1440e"), label="predicted")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("equal-exposure bucket")
        ax.set_xticks(range(1, 11))
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25, ls=":")
    axes[0].set_ylabel("loss cost per unit exposure")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Track B lift, N={n:,} — 10 equal-exposure buckets", fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f"  wrote {outfile.name}")


def plot_double_lift(y, e, pa, pb, na, nb, n, outfile):
    dl = M.double_lift_table(y, pa, pb, e, n_buckets=10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(dl["bucket"], dl["actual_loss_cost"], color="#d0d4da", width=0.78, label="actual")
    ax.plot(dl["bucket"], dl["pred_a_loss_cost"], marker="o", lw=2,
            color=COLORS.get(na, "#c1440e"), label=na)
    ax.plot(dl["bucket"], dl["pred_b_loss_cost"], marker="s", lw=2,
            color=COLORS.get(nb, "#1f4e79"), label=nb)
    ax.set_xlabel(f"bucket by {na}/{nb} ratio  (left: {nb} prices higher)")
    ax.set_ylabel("loss cost per unit exposure")
    ax.set_title(f"Double lift: {na} vs {nb}, N={n:,}")
    ax.set_xticks(range(1, 11))
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f"  wrote {outfile.name}")


def do_lift(res):
    preds = load_preds()
    print("\n=== lift charts ===")
    if not preds:
        print("  SKIPPED: no saved predictions (configs predate prediction-persistence)")
        return
    df, _ = build_and_cache()
    _, test = random_split(df)
    y, e = test["PurePremium"].to_numpy(), test["Exposure"].to_numpy()

    have = sorted({k[0] for k in preds})
    all_n = sorted({int(x) for x in pd.to_numeric(res["cfg_n"], errors="coerce").dropna()})
    print(f"  predictions available for N: {have}")
    missing = [n for n in all_n if n not in have]
    if missing:
        print(f"  no predictions for N: {missing}")

    for n in have:
        by_model = {}
        for model in ORDER:
            seeds = sorted(k[2] for k in preds if k[0] == n and k[1] == model)
            if seeds:
                # Average across draws: show typical behaviour, not one lucky draw.
                by_model[model] = np.mean([preds[(n, model, s)] for s in seeds], axis=0)
        if by_model:
            plot_lift(y, e, by_model, n, REPORTS / f"trackB_lift_N{n}.png")
            if "tabicl" in by_model and "xgb_hurdle" in by_model:
                plot_double_lift(y, e, by_model["tabicl"], by_model["xgb_hurdle"],
                                 "tabicl", "xgb_hurdle", n,
                                 REPORTS / f"trackB_doublelift_N{n}.png")


def run_checks(res, floor) -> list[C.Flag]:
    """Cross-config checks 9, 10 and 12, plus replay of per-config flags from the cache."""
    flags: list[C.Flag] = []

    for p in sorted(CACHE.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec.get("config", {}).get("track") != "B":
            continue
        for f in rec.get("flags", []):
            if f.get("severity") in ("FAIL", "WARN"):
                cfg = rec["config"]
                flags.append(C.Flag(f["check"], f["severity"],
                                    f"{cfg.get('label')}: {f['message']}"))

    for metric, _, higher in METRICS:
        mean, sd, nseeds = curve_table(res, metric)
        if mean is None:
            continue
        sub = res[res["metric"] == metric].copy()
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        ref = floor.get(metric)
        for (n, model), g in sub.groupby(["cfg_n", "cfg_model"]):
            # At N equal to the full pool the subsample is a no-op, so every draw saw the
            # same rows and a deterministic fitter is legitimately constant.
            flags += C.check_across_seeds(model, metric, list(g["value"].dropna()),
                                          reference_value=ref,
                                          identical_data=int(n) >= POOL_SIZE)
        for model in mean.columns:
            by_n = {int(n): (mean.loc[n, model], sd.loc[n, model])
                    for n in mean.index if np.isfinite(mean.loc[n, model])}
            if len(by_n) > 1:
                flags += C.check_curve_monotonicity(model, metric, by_n, higher)
    return flags


def main() -> int:
    res = load_results()
    if not len(res):
        print("no Track B results")
        return 1

    done = res[["cfg_label", "status"]].drop_duplicates()
    print(f"Track B configs (full test set): {len(done)}  "
          f"ok={int((done['status'] == 'ok').sum())}")

    floor = trivial_floor()
    print("\n=== trivial 1/exposure floor (zero risk information) ===")
    for k, lbl, _ in METRICS:
        if k in floor:
            print(f"  {lbl:48s} {floor[k]:>10.4f}")

    for metric, label, higher in METRICS:
        mean, sd, nseeds = curve_table(res, metric)
        if mean is None or mean.isna().all().all():
            print(f"\n[skip] {metric}: not present in cached results")
            continue
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        print("--- mean over draws (competitors | references) ---")
        print(mean.to_string(float_format=lambda x: f"{x:.4f}"))
        print("\n--- SD across draws ---")
        print(sd.to_string(float_format=lambda x: f"{x:.4f}"))
        print("\n--- draws per cell ---")
        print(nseeds.to_string(float_format=lambda x: f"{x:.0f}"))

        for rival in ["xgb_hurdle", "xgb_tweedie"]:
            sig = significance(mean, sd, nseeds, "tabicl", rival, higher)
            if len(sig):
                print(f"\n--- tabicl vs {rival}: two-sample z per N ---")
                print(sig.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        plot_curve(mean, sd, label, floor.get(metric),
                   REPORTS / f"trackB_{metric}.png", higher)

    do_lift(res)

    flags = run_checks(res, floor)
    counts = C.write_report(flags, RESULTS / "checks_report.md")
    print(f"\n=== checks: FAIL {counts['FAIL']} · WARN {counts['WARN']} ===")
    for f in flags:
        if f.severity == "FAIL":
            print(f"  {f}")
    if counts["FAIL"]:
        print("\n*** Blocking failures present — headline results withheld. ***")
    print(f"wrote {RESULTS / 'checks_report.md'}")

    claims = res[res["metric"] == "info_n_claims_stage2"]
    if len(claims):
        print("\n=== stage-2 claim counts by N ===")
        print(claims.groupby("cfg_n")["value"].apply(lambda s: pd.to_numeric(s).mean())
              .to_string(float_format=lambda x: f"{x:,.0f}"))

    res.to_parquet(RESULTS / "trackB_results.parquet", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
