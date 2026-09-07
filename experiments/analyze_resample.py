"""Analysis for the R=20 resampling study at N in {5,000, 10,000, 20,000}.

Answers "how stable is the performance", which a mean +/- SD alone does not: it shows the
whole distribution across draws, the spread of lift curves, and whether any single draw's
composition (not the model) moved the result.

Read-only over the runner cache.
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
from src.data import build_and_cache, exposure_stratified_subsample, random_split
from src.runner import CACHE, RESULTS

REPORTS = Path(__file__).resolve().parent.parent / "reports"
RESAMPLE_N = [5_000, 10_000, 20_000]
FULL_TEST_ROWS = 135_603

COMPETITORS = ["tabicl", "xgb_hurdle", "xgb_tweedie", "glm_hurdle", "glm_tweedie"]
REFERENCES = ["one_over_exposure", "intercept"]
COLORS = {
    "tabicl": "#c1440e", "xgb_hurdle": "#1f4e79", "xgb_tweedie": "#5b9bd5",
    "glm_hurdle": "#2e7d32", "glm_tweedie": "#81c784",
    "intercept": "#9e9e9e", "one_over_exposure": "#b39ddb",
}
METRICS = [
    ("gini", "Gini (unweighted)"),
    ("gini_exposure_weighted", "exposure-weighted Gini (rate)"),
    ("gini_total_loss", "Gini on total loss"),
    ("tweedie_deviance_1.5", "Tweedie deviance (p=1.5)"),
]


def load_records() -> pd.DataFrame:
    """One row per (n, model, seed) with metrics and sample characteristics."""
    rows = []
    for p in sorted(CACHE.glob("*.json")):
        rec = json.loads(p.read_text())
        cfg = rec.get("config", {})
        if cfg.get("track") != "B" or rec.get("status") != "ok":
            continue
        if cfg.get("n") not in RESAMPLE_N:
            continue
        if cfg.get("test_rows", FULL_TEST_ROWS) != FULL_TEST_ROWS:
            continue
        row = {"n": cfg["n"], "model": cfg["model"], "seed": cfg["seed"], "key": p.stem}
        row.update({k: v for k, v in rec.get("metrics", {}).items()})
        row.update({f"info_{k}": v for k, v in rec.get("info", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def load_preds(n: int, model: str) -> dict[int, np.ndarray]:
    out = {}
    for p in sorted(CACHE.glob("*.json")):
        cfg = json.loads(p.read_text()).get("config", {})
        if (cfg.get("track") == "B" and cfg.get("n") == n and cfg.get("model") == model
                and cfg.get("test_rows", FULL_TEST_ROWS) == FULL_TEST_ROWS):
            npz = p.with_name(f"{p.stem}_pred.npz")
            if npz.exists():
                with np.load(npz) as z:
                    out[cfg["seed"]] = z["pred_test"]
    return out


def stability_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean is not the answer here -- spread and worst case are."""
    g = df.groupby(["n", "model"])[metric]
    out = pd.DataFrame({
        "draws": g.size(), "mean": g.mean(), "sd": g.std(),
        "min": g.min(), "max": g.max(),
        "q25": g.quantile(0.25), "q75": g.quantile(0.75),
    })
    out["range"] = out["max"] - out["min"]
    out["cv"] = out["sd"] / out["mean"].abs()
    return out.reset_index()


def plot_distributions(df: pd.DataFrame, metric: str, label: str, outfile: Path):
    ns = sorted(df["n"].unique())
    models = [m for m in COMPETITORS + REFERENCES if m in set(df["model"])]
    fig, axes = plt.subplots(1, len(ns), figsize=(4.6 * len(ns), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, n in zip(axes, ns):
        data, labels, colors = [], [], []
        for m in models:
            v = df[(df["n"] == n) & (df["model"] == m)][metric].dropna().to_numpy()
            if len(v):
                data.append(v)
                labels.append(m)
                colors.append(COLORS.get(m, "#555"))
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("black")
        # Individual draws on top: the distribution shape matters, not just the box.
        for i, (v, c) in enumerate(zip(data, colors), start=1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(v)), v,
                       s=11, color=c, alpha=0.75, zorder=3, edgecolors="none")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"N = {n:,}  ({len(data[0])} draws)", fontsize=10)
        ax.grid(axis="y", alpha=0.25, ls=":")
    axes[0].set_ylabel(label)
    fig.suptitle(f"Resampling distribution: {label}", fontsize=12)
    fig.tight_layout()
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f"  wrote {outfile.name}")


def plot_spaghetti_lift(y, e, n: int, models: list[str], outfile: Path):
    """All R lift curves overlaid, mean bold. The BAND is the stability result.

    Twenty separate lift charts per cell would be unreadable; the width of the band is
    what the resampling study is actually measuring.
    """
    avail = [(m, load_preds(n, m)) for m in models]
    avail = [(m, p) for m, p in avail if p]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(4.0 * len(avail), 4.4), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (model, preds) in zip(axes, avail):
        curves = []
        for seed in sorted(preds):
            lt = M.lift_table(y, preds[seed], e, n_buckets=10)
            curves.append(lt["predicted_loss_cost"].to_numpy())
            ax.plot(lt["bucket"], curves[-1], color=COLORS.get(model, "#555"),
                    alpha=0.28, lw=0.9)
        actual = M.lift_table(y, preds[sorted(preds)[0]], e, 10)["actual_loss_cost"]
        ax.bar(range(1, 11), actual, color="#d0d4da", width=0.8, zorder=0, label="actual")
        ax.plot(range(1, 11), np.mean(curves, axis=0), color=COLORS.get(model, "#555"),
                lw=2.4, marker="o", ms=4, label=f"mean of {len(curves)}")
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("equal-exposure bucket")
        ax.set_xticks(range(1, 11))
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.25, ls=":")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("loss cost per unit exposure")
    fig.suptitle(f"Lift across {len(curves)} resample draws, N={n:,} "
                 f"(10 equal-exposure buckets)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f"  wrote {outfile.name}")


def sample_characterisation(n_values, n_draws: int = 20):
    """Regenerate each subsample and describe it. No model fitting -- this is cheap.

    A single heavy-tail draw can move a whole metric; it has to be visible rather than
    averaged away. freMTPL2's largest single claim is 4.07M against a 167 portfolio mean.
    """
    df, _ = build_and_cache()
    pool, _ = random_split(df)
    rows, per_n = [], {}
    for n in n_values:
        samples = []
        for seed in range(n_draws):
            s = exposure_stratified_subsample(pool, n, seed=seed)
            rec = {
                "n": n, "seed": seed,
                "mean_exposure": float(s["Exposure"].mean()),
                "total_exposure": float(s["Exposure"].sum()),
                "claim_rate": float(s["has_loss"].mean()),
                "n_claims": int(s["has_loss"].sum()),
                "total_loss": float(s["TotalLoss"].sum()),
                "max_loss": float(s["TotalLoss"].max()),
                "loss_cost": float(s["TotalLoss"].sum() / s["Exposure"].sum()),
            }
            samples.append(rec)
            rows.append(rec)
        per_n[n] = samples
    return pd.DataFrame(rows), per_n


def plot_sample_distributions(per_n: dict, outfile: Path):
    """Exposure ECDF per draw, plus loss-cost spread across draws."""
    df, _ = build_and_cache()
    pool, _ = random_split(df)
    ns = sorted(per_n)
    fig, axes = plt.subplots(2, len(ns), figsize=(4.4 * len(ns), 7.4))
    axes = np.atleast_2d(axes)

    for j, n in enumerate(ns):
        ax = axes[0, j]
        for seed in range(len(per_n[n])):
            s = exposure_stratified_subsample(pool, n, seed=seed)
            x = np.sort(s["Exposure"].to_numpy())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color="#1f4e79",
                    alpha=0.3, lw=0.8)
        xp = np.sort(pool["Exposure"].to_numpy())
        ax.plot(xp, np.arange(1, len(xp) + 1) / len(xp), color="#c1440e", lw=2,
                label="full pool")
        ax.set_title(f"exposure ECDF, N={n:,}", fontsize=10)
        ax.set_xlabel("exposure")
        ax.set_ylabel("cumulative share" if j == 0 else "")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, ls=":")

        ax = axes[1, j]
        lc = [r["loss_cost"] for r in per_n[n]]
        ml = [r["max_loss"] for r in per_n[n]]
        ax.scatter(ml, lc, s=32, color="#c1440e", alpha=0.8)
        ax.axhline(float(pool["TotalLoss"].sum() / pool["Exposure"].sum()),
                   ls="--", color="#555", lw=1.2, label="pool loss cost")
        ax.set_xscale("log")
        ax.set_xlabel("largest single claim in the draw (log)")
        ax.set_ylabel("draw loss cost" if j == 0 else "")
        ax.set_title(f"draw composition, N={n:,}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, ls=":")

    fig.suptitle("Resample composition — exposure stays matched; loss cost is tail-driven",
                 fontsize=12)
    fig.tight_layout()
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=140)
    plt.close(fig)
    print(f"  wrote {outfile.name}")


def main() -> int:
    pd.set_option("display.width", 220)
    recs = load_records()
    if not len(recs):
        print("no resampling results yet")
        return 1

    counts = recs.groupby(["n", "model"]).size().unstack(fill_value=0)
    print("=== draws per cell ===")
    print(counts.to_string())

    for metric, label in METRICS:
        if metric not in recs:
            continue
        print(f"\n{'=' * 90}\n{label}\n{'=' * 90}")
        st = stability_table(recs.dropna(subset=[metric]), metric)
        print(st.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        plot_distributions(recs, metric, label, REPORTS / f"resample_dist_{metric}.png")

    df, _ = build_and_cache()
    _, test = random_split(df)
    y, e = test["PurePremium"].to_numpy(), test["Exposure"].to_numpy()
    print("\n=== spaghetti lift ===")
    for n in sorted(recs["n"].unique()):
        plot_spaghetti_lift(y, e, int(n), ["tabicl", "xgb_hurdle", "xgb_tweedie"],
                            REPORTS / f"resample_lift_N{int(n)}.png")

    print("\n=== sample characterisation ===")
    draws = int(counts.max().max()) if len(counts) else 20
    samp, per_n = sample_characterisation(sorted(recs["n"].unique()), n_draws=draws)
    summary = samp.groupby("n").agg(["mean", "std", "min", "max"])[
        ["mean_exposure", "claim_rate", "loss_cost", "max_loss"]]
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))
    samp.to_csv(RESULTS / "resample_samples.csv", index=False)
    plot_sample_distributions(per_n, REPORTS / "resample_sample_composition.png")

    flags = []
    for n, group in samp.groupby("n"):
        flags += C.check_sample_comparability(group.to_dict("records"), int(n))
    print(f"\n=== sample comparability: {len(flags)} flags ===")
    for f in flags:
        print(f"  {f}")

    C.write_report(flags, RESULTS / "checks_resample.md")
    recs.to_parquet(RESULTS / "resample_results.parquet", index=False)
    print(f"\nwrote {RESULTS / 'resample_results.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
