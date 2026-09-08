"""Export the consolidated TabICL pricing analysis to Excel, ordered as an argument.

The workbook answers three questions in sequence rather than dumping one track per sheet:

  Overview  the three answers on one page, with the numbers that carry them
  Q1        Does TabICL work when data is thin?          (Track B, 5k -> 542k)
  Q2        Does it hold up on a region it never saw?    (Track A, leave-one-region-out)
  Q3        Does feature engineering matter?             (tabicl vs tabicl_raw, paired)

Every chart is a native Excel object wired to cells on the sheet, so each series traces
back to a table the reader can see and edit.

Lift charts follow one convention throughout: LOSS COST AS LINES on the primary axis,
EXPOSURE AS BARS on the secondary axis. Loss cost is what the model is judged on and lines
carry a shape across buckets; exposure is the bucket's weight, which is context, not a
competing quantity -- putting it on its own axis stops a large exposure bar from being read
as a large loss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.chart.marker import Marker
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy import stats

from src import metrics as M
from src.data import build_and_cache, random_split, region_split
from src.runner import CACHE, RESULTS

OUT = RESULTS / "trackB_analysis.xlsx"

RUNGS = [5_000, 10_000, 20_000, 100_000, 542_410]
COMPETITORS = ["tabicl", "xgb_hurdle", "xgb_tweedie", "glm_hurdle", "glm_tweedie"]
REFERENCES = ["one_over_exposure", "intercept"]
MODELS = COMPETITORS + REFERENCES
REGIONS = ["R24", "R82", "R93", "R11", "R43"]
FULL_TEST_ROWS = 135_603

# Fixed colour per model across every artefact of this study. tabicl_raw sits next to
# tabicl in the Q3 charts, and blue-vs-violet is the pair that collapses under
# deuteranopia -- so it also gets a dashed line. Colour is never the only cue.
SERIES_HEX = {
    "tabicl": "2A78D6", "tabicl_raw": "6D28D9", "xgb_hurdle": "EB6834",
    "xgb_tweedie": "1BAF7A", "glm_hurdle": "EDA100", "glm_tweedie": "E87BA4",
    "one_over_exposure": "7B7A75", "intercept": "B9B9B2", "actual": "0B0B0B",
}
DASHED = {"tabicl_raw"}

METRICS = [
    ("gini_total_loss", "Gini on total loss (artifact-free)", "higher"),
    ("gini_exposure_weighted", "Exposure-weighted Gini (rate)", "higher"),
    ("gini_fixed_exposure", "Gini, full-term policies (exp >= 0.95)", "higher"),
    ("tweedie_deviance_1.5", "Mean Tweedie deviance (p=1.5)", "lower"),
    ("calibration_ratio", "Calibration (predicted / actual)", "one"),
]

CHART_ROWS = 18
INK = "0B0B0B"
MUTED = "7B7A75"
RULE = "DCDCD7"
HEAD_FILL = PatternFill("solid", fgColor="ECECEA")
THIN = Side(style="thin", color=RULE)
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


# --------------------------------------------------------------------------- data


def _ok_records(track: str):
    for p in sorted(CACHE.glob("*.json")):
        rec = json.loads(p.read_text())
        cfg = rec.get("config", {})
        if cfg.get("track") == track and rec.get("status") == "ok":
            yield p, cfg, rec


def load_trackB() -> tuple[dict, dict]:
    """(n, model) -> {metric: [values]}, and (n, model) -> mean lift table."""
    recs: dict = {}
    preds: dict = {}
    for p, cfg, rec in _ok_records("B"):
        if cfg.get("n") not in RUNGS or cfg.get("test_rows", FULL_TEST_ROWS) != FULL_TEST_ROWS:
            continue
        key = (cfg["n"], cfg["model"])
        # At N=100,000 legacy tabicl records carry trials=25 and the reruns trials=30.
        # `trials` is inert for TabICL, but a seed must not be counted twice.
        seen = recs.setdefault(key, {}).setdefault("_seeds", [])
        if cfg["seed"] in seen:
            continue
        seen.append(cfg["seed"])
        for k, v in rec.get("metrics", {}).items():
            recs[key].setdefault(k, []).append(float(v))
        npz = p.with_name(f"{p.stem}_pred.npz")
        if npz.exists():
            with np.load(npz) as z:
                preds.setdefault(key, []).append(z["pred_test"])

    df, _ = build_and_cache()
    _, test = random_split(df)
    y, e = test["PurePremium"].to_numpy(), test["Exposure"].to_numpy()
    lifts = {k: M.lift_table(y, np.mean(v, axis=0), e, n_buckets=10) for k, v in preds.items()}
    return recs, lifts


def load_trackA() -> tuple[dict, dict]:
    """(region, model) -> {metric: [values]}, and (region, model) -> mean lift table."""
    recs: dict = {}
    preds: dict = {}
    for p, cfg, rec in _ok_records("A"):
        if cfg.get("region") not in REGIONS:
            continue
        key = (cfg["region"], cfg["model"])
        for k, v in rec.get("metrics", {}).items():
            recs.setdefault(key, {}).setdefault(k, []).append(float(v))
        recs.setdefault(key, {}).setdefault("_seeds", []).append(cfg["seed"])
        npz = p.with_name(f"{p.stem}_pred.npz")
        if npz.exists():
            with np.load(npz) as z:
                preds.setdefault(key, []).append((z["pred"], z["actual"], z["exposure"]))

    lifts = {}
    for key, arrs in preds.items():
        pred = np.mean([a[0] for a in arrs], axis=0)
        actual, exposure = arrs[0][1], arrs[0][2]
        lifts[key] = M.lift_table(actual, pred, exposure, n_buckets=10)
    return recs, lifts


def region_table() -> pd.DataFrame:
    """Every region's size and risk profile, computed from the cleaned frame.

    data/region_summary.csv disagrees with the cleaned data (it lists R24 at 449.8 where
    the frame gives 185.7, and reverses R11/R93), so it is deliberately not used.
    """
    df, _ = build_and_cache()
    g = df.groupby("Region")
    out = pd.DataFrame({
        "policies": g.size(),
        "exposure": g["Exposure"].sum(),
        "claim_rate": g["has_loss"].mean(),
        "loss_cost": g["TotalLoss"].sum() / g["Exposure"].sum(),
    })
    out["share_of_exposure"] = out["exposure"] / out["exposure"].sum()
    return out.sort_values("policies", ascending=False)


# ------------------------------------------------------------------- sheet helpers


def title(ws, row, text, size=13):
    c = ws.cell(row=row, column=1, value=text)
    c.font = Font(bold=True, size=size, color=INK)
    return row + 1


def note(ws, row, text):
    c = ws.cell(row=row, column=1, value=text)
    c.font = Font(size=9, italic=True, color=MUTED)
    return row + 1


def write_table(ws, row, col, headers, rows, widths=None, number_format="0.0000"):
    for j, h in enumerate(headers):
        c = ws.cell(row=row, column=col + j, value=h)
        c.font = Font(bold=True, size=9, color=INK)
        c.fill = HEAD_FILL
        c.border = BOX
        c.alignment = Alignment(horizontal="center" if j else "left", wrap_text=True)
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r):
            c = ws.cell(row=row + i, column=col + j, value=v)
            c.border = BOX
            if j == 0:
                c.font = Font(size=9, color=INK)
            else:
                c.font = Font(size=9, name="Consolas")
                c.alignment = Alignment(horizontal="right")
                if isinstance(v, (int, float)):
                    c.number_format = number_format
    if widths:
        for j, w in enumerate(widths):
            ws.column_dimensions[get_column_letter(col + j)].width = w
    return row + len(rows) + 1


def style_series(s, hexcolor, *, line=True, marker="circle", width=22000, dashed=False):
    if line:
        lp = LineProperties(solidFill=hexcolor, w=width)
        if dashed:
            lp.dashStyle = "dash"
        s.graphicalProperties.line = lp
    else:
        s.graphicalProperties.line.noFill = True
    s.marker = Marker(symbol=marker, size=6)
    s.marker.graphicalProperties.solidFill = hexcolor
    s.marker.graphicalProperties.line.solidFill = hexcolor
    s.smooth = False


def lift_chart(ws, hdr_row, n_rows, models, chart_title):
    """Loss cost as LINES (primary) + exposure as BARS (secondary).

    Table layout assumed: col1 Bucket | col2 Exposure | col3 Actual | col4.. model preds.

    openpyxl builds a combined chart by adding one chart to another; the BASE chart owns
    the primary axis, so the line chart must be the base. The added chart needs its own
    axId or the two collapse onto one scale. Note a BarChart cannot be combined with a
    ScatterChart at all -- it fails silently and drops the overlay -- which is why this is
    LineChart + BarChart.
    """
    last = hdr_row + n_rows
    line = LineChart()
    line.title = chart_title
    line.height, line.width = 8.4, 16.5
    line.y_axis.title = "loss cost per unit exposure"
    line.x_axis.title = "equal-exposure bucket (ordered by predicted)"
    line.legend.position = "b"
    line.add_data(Reference(ws, min_col=3, max_col=3 + len(models),
                            min_row=hdr_row, max_row=last), titles_from_data=True)
    line.set_categories(Reference(ws, min_col=1, min_row=hdr_row + 1, max_row=last))
    for s, name in zip(line.series, ["actual"] + models):
        style_series(s, SERIES_HEX.get(name, "7B7A75"), dashed=name in DASHED)

    bar = BarChart()
    bar.type = "col"
    bar.add_data(Reference(ws, min_col=2, max_col=2, min_row=hdr_row, max_row=last),
                 titles_from_data=True)
    bar.y_axis.axId = 200
    bar.y_axis.title = "exposure in bucket"
    bar.gapWidth = 60
    for s in bar.series:
        s.graphicalProperties.solidFill = "D8D8D3"
        s.graphicalProperties.line.noFill = True

    # Put the secondary axis on the right-hand side.
    line.y_axis.crosses = "max"
    line += bar
    return line


def strip_chart(ws, hdr_row, ndraw, cols, xcol, chart_title, y_title, x_max):
    """Every draw as a point, one vertical track per model."""
    ch = ScatterChart()
    ch.title = chart_title
    ch.style = 2
    ch.height, ch.width = 8.4, 16.5
    ch.y_axis.title = y_title
    ch.x_axis.scaling.min, ch.x_axis.scaling.max = 0, x_max
    ch.x_axis.majorGridlines = None
    ch.x_axis.delete = True  # track positions are plumbing, not data
    ch.legend.position = "b"
    for ci, m in enumerate(cols, start=1):
        xc = xcol + ci
        ws.cell(row=hdr_row, column=xc, value=m).font = Font(bold=True, size=8)
        for i in range(ndraw):
            ws.cell(row=hdr_row + 1 + i, column=xc, value=ci)
        xref = Reference(ws, min_col=xc, min_row=hdr_row + 1, max_row=hdr_row + ndraw)
        yref = Reference(ws, min_col=1 + ci, min_row=hdr_row, max_row=hdr_row + ndraw)
        s = Series(yref, xref, title_from_data=True)
        style_series(s, SERIES_HEX.get(m, "7B7A75"), line=False)
        ch.series.append(s)
    for c in range(xcol, xcol + len(cols) + 1):
        ws.column_dimensions[get_column_letter(c)].hidden = True
    return ch


def mean_of(recs, key, metric):
    v = recs.get(key, {}).get(metric)
    return round(float(np.mean(v)), 4) if v else None


# ------------------------------------------------------------------ sheet: overview


def sheet_overview(wb, B, A):
    ws = wb.create_sheet("Overview")
    ws.sheet_view.showGridLines = False
    r = 1
    r = title(ws, r, "TabICL for motor pricing — does it work?", 16)
    r = note(ws, r, "freMTPL2, 678,013 policies. Track B: 542,410 train / 135,603 fixed test, "
                    "20 resample draws per rung. Track A: leave-one-region-out, 21 regions "
                    "train, held-out region tested, 5 seeds.")
    r = note(ws, r, "All ranking claims use Gini on TOTAL LOSS. The rate-based Gini has a "
                    "degenerate solution: a c/exposure model charges every policy the same "
                    "premium yet scores 0.4676 exposure-weighted Gini. On total loss it scores 0.")
    r += 2

    r = title(ws, r, "The three answers", 13)
    r += 1
    answers = [
        ["1. Does TabICL work when data is thin?",
         "Ranking yes, pricing no.",
         "Leads total-loss Gini from N=5,000 and decisively at N=20,000 (0.412 vs 0.239, "
         "z~7.5). But does not out-price a constant premium until N=100,000; at N=5,000 the "
         "trivial model is better. Calibration ~0.45 — it prices the book at half its cost."],
        ["2. Does it hold up on a region it never saw?",
         "No — it degrades where shift is worst.",
         "xgb_hurdle takes 3 of 5 regions. On R11 (Ile-de-France, most distinct) TabICL "
         "loses ranking at z=-6.39, over-predicts 2.01x, and is the ONLY model beaten by "
         "charging every policy the same premium."],
        ["3. Does feature engineering matter?",
         "Yes below 20k, and it reverses by 100k.",
         "Raw features cost 0.060 total-loss Gini at N=5,000 (p=0.0009, losing 17/20 draws). "
         "At N=100,000 raw features WIN by 0.034 (p=0.0002, 18/20). The crossover sits where "
         "TabICL first prices better than a constant premium."],
    ]
    r = write_table(ws, r, 1, ["Question", "Answer", "Evidence"], answers,
                    widths=[42, 34, 76], number_format="General")
    r += 2

    r = title(ws, r, "Headline numbers", 13)
    r = note(ws, r, "Gini on total loss (higher better) and Tweedie deviance p=1.5 (lower better).")
    r += 1
    rows = []
    for n in RUNGS:
        rows.append([f"N = {n:,}",
                     mean_of(B, (n, "tabicl"), "gini_total_loss"),
                     mean_of(B, (n, "xgb_hurdle"), "gini_total_loss"),
                     mean_of(B, (n, "one_over_exposure"), "gini_total_loss"),
                     mean_of(B, (n, "tabicl"), "tweedie_deviance_1.5"),
                     mean_of(B, (n, "xgb_hurdle"), "tweedie_deviance_1.5"),
                     mean_of(B, (n, "one_over_exposure"), "tweedie_deviance_1.5")])
    r = write_table(ws, r, 1,
                    ["Rung", "Gini tabicl", "Gini xgb_hurdle", "Gini trivial",
                     "Dev tabicl", "Dev xgb_hurdle", "Dev trivial"], rows,
                    widths=[16] + [15] * 6)
    r += 2

    r = title(ws, r, "How to read every lift chart in this workbook", 12)
    r = note(ws, r, "Loss cost is drawn as LINES on the left axis — that is the quantity the "
                    "model is judged on, and a line carries its shape across buckets. Exposure "
                    "is drawn as BARS on the right axis: it is the weight behind each bucket, "
                    "context rather than a competing quantity. A well-ranking model's predicted "
                    "line rises with the actual line from bucket 1 to 10.")
    ws.freeze_panes = "A2"
    return ws


# ------------------------------------------------------------------ sheet: Q1


def sheet_q1(wb, recs, lifts):
    ws = wb.create_sheet("Q1 Thin data")
    ws.sheet_view.showGridLines = False
    r = 1
    r = title(ws, r, "Q1 · Does TabICL work when data is thin?", 15)
    r = note(ws, r, "Same fixed 135,603-row test set at every rung, so all variation comes from "
                    "the training draw. 20 draws at 5k/10k/20k/100k, 3 at full data.")
    r = note(ws, r, "Reference lines: one_over_exposure charges every policy the same premium "
                    "(total-loss Gini 0 by construction); intercept charges a constant RATE, so "
                    "its predicted loss tracks exposure and it is NOT a zero floor.")
    r += 1

    r = title(ws, r, "1 · Evaluation metrics by rung", 12)
    r = note(ws, r, "Mean across draws. Best competitor per rung in bold blue.")
    r += 1
    for key, label, better in METRICS:
        r = title(ws, r, label, 10)
        headers = ["Model"] + [f"N = {n:,}" for n in RUNGS]
        rows = [[m] + [mean_of(recs, (n, m), key) for n in RUNGS] for m in MODELS]
        start = r
        r = write_table(ws, r, 1, headers, rows, widths=[24] + [13] * len(RUNGS))
        for j in range(len(RUNGS)):
            best_i, best_v = None, None
            for i, m in enumerate(COMPETITORS):
                v = rows[i][1 + j]
                if v is None:
                    continue
                ok = (best_v is None
                      or (v > best_v if better == "higher"
                          else v < best_v if better == "lower"
                          else abs(v - 1) < abs(best_v - 1)))
                if ok:
                    best_i, best_v = i, v
            if best_i is not None:
                ws.cell(row=start + 1 + best_i, column=2 + j).font = Font(
                    size=9, name="Consolas", bold=True, color="2A78D6")
        r += 1

    r += 1
    r = title(ws, r, "2 · Variance across draws", 12)
    r = note(ws, r, "At N=542,410 the subsample is the whole pool, so SD there is "
                    "model-internal randomness only and is not comparable to smaller rungs.")
    r += 1
    headers = ["Model", "Rung", "Draws", "Mean", "SD", "CV", "Min", "Max"]
    rows = []
    for m in MODELS:
        for n in RUNGS:
            v = recs.get((n, m), {}).get("gini_total_loss")
            if not v:
                continue
            a = np.asarray(v, dtype=float)
            sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
            rows.append([m, f"{n:,}", len(a), round(float(a.mean()), 4), round(sd, 4),
                         round(sd / abs(a.mean()), 4) if abs(a.mean()) > 1e-9 else None,
                         round(float(a.min()), 4), round(float(a.max()), 4)])
    r = write_table(ws, r, 1, headers, rows, widths=[24, 12, 8, 11, 11, 11, 11, 11])
    r += 2

    r = title(ws, r, "3 · Distribution and lift, per rung", 12)
    r = note(ws, r, "Distribution charts plot every draw as a point, so the spread is visible "
                    "rather than summarised.")
    r += 1

    for n in RUNGS:
        r = title(ws, r, f"N = {n:,}", 12)
        top = r
        key = "gini_total_loss"
        cols = [m for m in COMPETITORS if recs.get((n, m), {}).get(key)]
        ndraw = max(len(recs[(n, m)][key]) for m in cols) if cols else 0
        ws.cell(row=r, column=1, value=f"Total-loss Gini — {ndraw} draws").font = Font(
            bold=True, size=10)
        r += 1
        hdr = r
        rows = []
        for i in range(ndraw):
            row = [i + 1]
            for m in cols:
                v = recs[(n, m)][key]
                row.append(round(float(v[i]), 4) if i < len(v) else None)
            rows.append(row)
        end = write_table(ws, r, 1, ["Draw"] + cols, rows, widths=[8] + [14] * len(cols))
        xcol = 1 + len(cols) + 2
        ch = strip_chart(ws, hdr, ndraw, cols, xcol,
                         f"Total-loss Gini distribution, N = {n:,}", "Gini on total loss",
                         len(cols) + 1)
        anchor_col = get_column_letter(xcol + len(cols) + 2)
        ws.add_chart(ch, f"{anchor_col}{top}")

        lift_models = [m for m in ["tabicl", "xgb_hurdle"] if (n, m) in lifts]
        if not lift_models:
            r = max(end, top + CHART_ROWS) + 2
            continue
        lr = end + 1
        ws.cell(row=lr, column=1, value="Lift — 10 equal-exposure buckets").font = Font(
            bold=True, size=10)
        lr += 1
        base = lifts[(n, lift_models[0])]
        headers = ["Bucket", "Exposure", "Actual"] + [f"{m} pred" for m in lift_models]
        lrows = []
        for i in range(len(base)):
            row = [int(base["bucket"].iloc[i]), round(float(base["exposure"].iloc[i]), 1),
                   round(float(base["actual_loss_cost"].iloc[i]), 2)]
            for m in lift_models:
                row.append(round(float(lifts[(n, m)]["predicted_loss_cost"].iloc[i]), 2))
            lrows.append(row)
        lhdr = lr
        end2 = write_table(ws, lr, 1, headers, lrows,
                           widths=[9, 12, 12] + [15] * len(lift_models),
                           number_format="0.00")
        lc = lift_chart(ws, lhdr, len(lrows), lift_models,
                        f"Lift, N = {n:,} — loss cost (lines) vs exposure (bars)")
        lift_row = max(lr, top + CHART_ROWS)
        ws.add_chart(lc, f"{anchor_col}{lift_row}")
        r = max(end2, lift_row + CHART_ROWS) + 2
    return ws


# ------------------------------------------------------------------ sheet: Q2


def sheet_q2(wb, recs, lifts, regions):
    ws = wb.create_sheet("Q2 Censored regions")
    ws.sheet_view.showGridLines = False
    r = 1
    r = title(ws, r, "Q2 · Does TabICL hold up on a region it never saw?", 15)
    r = note(ws, r, "Leave-one-region-out: train on the other 21 regions, test only on the "
                    "held-out one. Five independent experiments.")
    r = note(ws, r, "With TargetEncoder fit on train only, the censored region is an unseen "
                    "category and Region_te is CONSTANT across its test set — every model "
                    "loses region signal equally. The question is which recovers most of it "
                    "from Area, log_density and VehBrand_te.")
    r += 1

    # ---- region distribution ----
    r = title(ws, r, "1 · Distribution of regions in the portfolio", 12)
    r = note(ws, r, "All 22 regions by size. The five tested are marked. Computed from the "
                    "cleaned frame — data/region_summary.csv disagrees with it and is not used.")
    r += 1
    top = r
    hdr = r
    rows = []
    for reg, row in regions.iterrows():
        rows.append([reg + ("  *tested" if reg in REGIONS else ""), int(row["policies"]),
                     round(float(row["exposure"]), 1), round(float(row["claim_rate"]), 4),
                     round(float(row["loss_cost"]), 1),
                     round(float(row["share_of_exposure"]), 4)])
    end = write_table(ws, r, 1,
                      ["Region", "Policies", "Exposure", "Claim rate", "Loss cost",
                       "Share of exposure"], rows,
                      widths=[16, 12, 13, 12, 12, 16], number_format="0.0000")

    bar = BarChart()
    bar.type = "col"
    bar.title = "Exposure by region (bars) with loss cost (line)"
    bar.height, bar.width = 8.4, 18
    bar.y_axis.title = "exposure"
    bar.x_axis.title = "region"
    bar.legend.position = "b"
    bar.add_data(Reference(ws, min_col=3, max_col=3, min_row=hdr, max_row=hdr + len(rows)),
                 titles_from_data=True)
    bar.set_categories(Reference(ws, min_col=1, min_row=hdr + 1, max_row=hdr + len(rows)))
    for s in bar.series:
        s.graphicalProperties.solidFill = "D8D8D3"
        s.graphicalProperties.line.noFill = True
    lc = LineChart()
    lc.add_data(Reference(ws, min_col=5, max_col=5, min_row=hdr, max_row=hdr + len(rows)),
                titles_from_data=True)
    lc.y_axis.axId = 200
    lc.y_axis.title = "loss cost"
    for s in lc.series:
        style_series(s, SERIES_HEX["xgb_hurdle"])
    bar.y_axis.crosses = "max"
    bar += lc
    ws.add_chart(bar, f"{get_column_letter(9)}{top}")
    r = max(end, top + CHART_ROWS) + 2

    # ---- metrics per region ----
    r = title(ws, r, "2 · Performance on the censored region", 12)
    r = note(ws, r, "Mean over 5 seeds. Best competitor per region in bold blue.")
    r += 1
    a_models = [m for m in COMPETITORS if any((reg, m) in recs for reg in REGIONS)]
    a_refs = [m for m in REFERENCES if any((reg, m) in recs for reg in REGIONS)]
    for key, label, better in METRICS:
        if not any(recs.get((reg, m), {}).get(key) for reg in REGIONS for m in a_models):
            continue
        r = title(ws, r, label, 10)
        headers = ["Model"] + REGIONS
        rows = [[m] + [mean_of(recs, (reg, m), key) for reg in REGIONS]
                for m in a_models + a_refs]
        start = r
        r = write_table(ws, r, 1, headers, rows, widths=[24] + [13] * len(REGIONS))
        for j in range(len(REGIONS)):
            best_i, best_v = None, None
            for i, m in enumerate(a_models):
                v = rows[i][1 + j]
                if v is None:
                    continue
                ok = (best_v is None
                      or (v > best_v if better == "higher"
                          else v < best_v if better == "lower"
                          else abs(v - 1) < abs(best_v - 1)))
                if ok:
                    best_i, best_v = i, v
            if best_i is not None:
                ws.cell(row=start + 1 + best_i, column=2 + j).font = Font(
                    size=9, name="Consolas", bold=True, color="2A78D6")
        r += 1

    r += 1
    r = title(ws, r, "3 · Head-to-head on the censored region", 12)
    r += 1
    hh = []
    for reg in REGIONS:
        t = recs.get((reg, "tabicl"), {}).get("gini_total_loss")
        if not t:
            continue
        best_name, best_mean = None, -9
        for m in a_models:
            if m == "tabicl":
                continue
            v = recs.get((reg, m), {}).get("gini_total_loss")
            if v and np.mean(v) > best_mean:
                best_name, best_mean = m, float(np.mean(v))
        o = recs[(reg, best_name)]["gini_total_loss"]
        gap = float(np.mean(t)) - best_mean
        se = np.sqrt(np.var(t, ddof=1) / len(t) + (np.var(o, ddof=1) / len(o) if len(o) > 1 else 0))
        dev_t = mean_of(recs, (reg, "tabicl"), "tweedie_deviance_1.5")
        dev_f = mean_of(recs, (reg, "one_over_exposure"), "tweedie_deviance_1.5")
        hh.append([reg, round(float(np.mean(t)), 4), best_name, round(best_mean, 4),
                   round(gap, 4), round(gap / se, 2) if se > 0 else None,
                   dev_t, dev_f,
                   "BEATEN BY TRIVIAL" if (dev_t and dev_f and dev_t > dev_f) else "clears"])
    r = write_table(ws, r, 1,
                    ["Region", "tabicl Gini", "Best rival", "Rival Gini", "Gap", "z",
                     "tabicl deviance", "Trivial deviance", "Price vs trivial floor"], hh,
                    widths=[10, 13, 14, 12, 10, 8, 15, 16, 20])
    r += 2

    # ---- per-region distribution + lift ----
    r = title(ws, r, "4 · Distribution and lift, per censored region", 12)
    r += 1
    for reg in REGIONS:
        r = title(ws, r, f"Region {reg}", 12)
        top = r
        cols = [m for m in a_models if recs.get((reg, m), {}).get("gini_total_loss")]
        if not cols:
            r += 2
            continue
        ndraw = max(len(recs[(reg, m)]["gini_total_loss"]) for m in cols)
        ws.cell(row=r, column=1, value=f"Total-loss Gini — {ndraw} seeds").font = Font(
            bold=True, size=10)
        r += 1
        hdr = r
        rows = []
        for i in range(ndraw):
            row = [i + 1]
            for m in cols:
                v = recs[(reg, m)]["gini_total_loss"]
                row.append(round(float(v[i]), 4) if i < len(v) else None)
            rows.append(row)
        end = write_table(ws, r, 1, ["Seed"] + cols, rows, widths=[8] + [14] * len(cols))
        xcol = 1 + len(cols) + 2
        ch = strip_chart(ws, hdr, ndraw, cols, xcol,
                         f"{reg} — total-loss Gini by seed", "Gini on total loss",
                         len(cols) + 1)
        anchor_col = get_column_letter(xcol + len(cols) + 2)
        ws.add_chart(ch, f"{anchor_col}{top}")

        lift_models = [m for m in ["tabicl", "xgb_hurdle"] if (reg, m) in lifts]
        if not lift_models:
            r = max(end, top + CHART_ROWS) + 2
            continue
        lr = end + 1
        ws.cell(row=lr, column=1, value="Lift — 10 equal-exposure buckets").font = Font(
            bold=True, size=10)
        lr += 1
        base = lifts[(reg, lift_models[0])]
        headers = ["Bucket", "Exposure", "Actual"] + [f"{m} pred" for m in lift_models]
        lrows = []
        for i in range(len(base)):
            row = [int(base["bucket"].iloc[i]), round(float(base["exposure"].iloc[i]), 1),
                   round(float(base["actual_loss_cost"].iloc[i]), 2)]
            for m in lift_models:
                row.append(round(float(lifts[(reg, m)]["predicted_loss_cost"].iloc[i]), 2))
            lrows.append(row)
        lhdr = lr
        end2 = write_table(ws, lr, 1, headers, lrows,
                           widths=[9, 12, 12] + [15] * len(lift_models),
                           number_format="0.00")
        lc2 = lift_chart(ws, lhdr, len(lrows), lift_models,
                         f"{reg} lift — loss cost (lines) vs exposure (bars)")
        lift_row = max(lr, top + CHART_ROWS)
        ws.add_chart(lc2, f"{anchor_col}{lift_row}")
        r = max(end2, lift_row + CHART_ROWS) + 2
    return ws


# ------------------------------------------------------------------ sheet: Q3


def sheet_q3(wb, recs, lifts):
    ws = wb.create_sheet("Q3 Feature engineering")
    ws.sheet_view.showGridLines = False
    r = 1
    r = title(ws, r, "Q3 · Does feature engineering matter?", 15)
    r = note(ws, r, "tabicl (15 engineered features: derived business terms plus target "
                    "encodings) vs tabicl_raw (9 raw cleaned columns, categoricals as integer "
                    "codes, no target encoding).")
    r = note(ws, r, "PAIRED: both arms use the same seed, so exposure_stratified_subsample "
                    "returns the identical draw and only the feature set differs. Differences "
                    "are taken within seed, which removes the draw variance that dominates "
                    "everything else in this study.")
    r += 1

    pair_rungs = [n for n in RUNGS
                  if recs.get((n, "tabicl_raw"), {}).get("gini_total_loss")]

    r = title(ws, r, "1 · Paired comparison by rung", 12)
    r += 1
    for key, label, better in METRICS:
        rows = []
        for n in pair_rungs:
            a = recs.get((n, "tabicl"), {})
            b = recs.get((n, "tabicl_raw"), {})
            if not (a.get(key) and b.get(key)):
                continue
            sa = dict(zip(a["_seeds"], a[key]))
            sb = dict(zip(b["_seeds"], b[key]))
            common = sorted(set(sa) & set(sb))
            if len(common) < 3:
                continue
            x = np.array([sa[s] for s in common])
            y = np.array([sb[s] for s in common])
            d = y - x
            t, p = stats.ttest_rel(y, x)
            wins = int((d < 0).sum() if better == "lower" else (d > 0).sum())
            rows.append([f"N = {n:,}", len(common), round(float(x.mean()), 4),
                         round(float(y.mean()), 4), round(float(d.mean()), 4),
                         round(float(p), 4), f"{wins}/{len(common)}"])
        if not rows:
            continue
        r = title(ws, r, f"{label} — {'lower' if better == 'lower' else 'higher'} is better", 10)
        r = write_table(ws, r, 1,
                        ["Rung", "Paired draws", "Engineered", "Raw", "Gap (raw − eng)",
                         "p (paired t)", "Raw wins"], rows,
                        widths=[14, 13, 13, 13, 16, 13, 12])
        r += 1

    r += 1
    r = title(ws, r, "2 · The crossover", 12)
    r = note(ws, r, "Feature engineering is an asset below 20k and a liability by 100k. Three "
                    "metrics agree on direction and significance at both ends.")
    r += 1
    top = r
    hdr = r
    rows = []
    for n in pair_rungs:
        a, b = recs.get((n, "tabicl"), {}), recs.get((n, "tabicl_raw"), {})
        if not (a.get("gini_total_loss") and b.get("gini_total_loss")):
            continue
        rows.append([f"{n:,}", round(float(np.mean(a["gini_total_loss"])), 4),
                     round(float(np.mean(b["gini_total_loss"])), 4)])
    end = write_table(ws, r, 1, ["N", "engineered", "raw"], rows, widths=[12, 14, 14])
    ch = LineChart()
    ch.title = "Total-loss Gini — engineered vs raw features"
    ch.height, ch.width = 8.4, 16.5
    ch.y_axis.title = "Gini on total loss"
    ch.x_axis.title = "training rows (N)"
    ch.legend.position = "b"
    ch.add_data(Reference(ws, min_col=2, max_col=3, min_row=hdr, max_row=hdr + len(rows)),
                titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=hdr + 1, max_row=hdr + len(rows)))
    for s, name in zip(ch.series, ["tabicl", "tabicl_raw"]):
        style_series(s, SERIES_HEX[name], dashed=name in DASHED)
    ws.add_chart(ch, f"{get_column_letter(6)}{top}")
    r = max(end, top + CHART_ROWS) + 2

    # Lift, engineered vs raw, per rung.
    r = title(ws, r, "3 · Lift by rung — engineered vs raw", 12)
    r += 1
    for n in pair_rungs:
        if (n, "tabicl") not in lifts or (n, "tabicl_raw") not in lifts:
            continue
        r = title(ws, r, f"N = {n:,}", 11)
        top = r
        base = lifts[(n, "tabicl")]
        models = ["tabicl", "tabicl_raw"]
        headers = ["Bucket", "Exposure", "Actual"] + [f"{m} pred" for m in models]
        rows = []
        for i in range(len(base)):
            row = [int(base["bucket"].iloc[i]), round(float(base["exposure"].iloc[i]), 1),
                   round(float(base["actual_loss_cost"].iloc[i]), 2)]
            for m in models:
                row.append(round(float(lifts[(n, m)]["predicted_loss_cost"].iloc[i]), 2))
            rows.append(row)
        hdr = r
        end = write_table(ws, r, 1, headers, rows, widths=[9, 12, 12, 15, 15],
                          number_format="0.00")
        lc = lift_chart(ws, hdr, len(rows), models,
                        f"N = {n:,} — loss cost (lines) vs exposure (bars)")
        ws.add_chart(lc, f"{get_column_letter(8)}{top}")
        r = max(end, top + CHART_ROWS) + 2
    return ws


def main() -> int:
    B, B_lifts = load_trackB()
    print(f"Track B: {len(B)} (rung, model) cells, {len(B_lifts)} lift tables")
    A, A_lifts = load_trackA()
    print(f"Track A: {len(A)} (region, model) cells, {len(A_lifts)} lift tables")
    regions = region_table()
    print(f"regions: {len(regions)}")

    wb = Workbook()
    wb.remove(wb.active)
    sheet_overview(wb, B, A)
    sheet_q1(wb, B, B_lifts)
    sheet_q2(wb, A, A_lifts, regions)
    sheet_q3(wb, B, B_lifts)
    RESULTS.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
