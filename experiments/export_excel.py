"""Export the Track B analysis to Excel.

Sheet 1 "Metrics by rung"  - evaluation metrics per rung, the summary read, and the
                             variance analysis.
Sheet 2 "Charts"           - a distribution chart and a lift chart for every rung, each
                             sitting immediately beside the table it is drawn from.

Charts are native Excel objects, not pasted images, so every series traces back to cells
on the sheet and stays editable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference, ScatterChart, Series
from openpyxl.chart.marker import Marker
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from src import metrics as M
from src.data import build_and_cache, random_split
from src.runner import CACHE, RESULTS

OUT = RESULTS / "trackB_analysis.xlsx"

RUNGS = [5_000, 10_000, 20_000, 100_000, 542_410]
COMPETITORS = ["tabicl", "xgb_hurdle", "xgb_tweedie", "glm_hurdle", "glm_tweedie"]
REFERENCES = ["one_over_exposure", "intercept"]
MODELS = COMPETITORS + REFERENCES
FULL_TEST_ROWS = 135_603

# Same validated categorical slots as the published page, in the same fixed order, so a
# model wears one colour across every artefact of this study.
SERIES_HEX = {
    "tabicl": "2A78D6", "xgb_hurdle": "EB6834", "xgb_tweedie": "1BAF7A",
    "glm_hurdle": "EDA100", "glm_tweedie": "E87BA4",
    "one_over_exposure": "7B7A75", "intercept": "B9B9B2",
}

METRICS = [
    ("gini_total_loss", "Gini on total loss (artifact-free)", "higher"),
    ("gini_exposure_weighted", "Exposure-weighted Gini (rate)", "higher"),
    ("gini", "Gini (unweighted)", "higher"),
    ("gini_fixed_exposure", "Gini, full-term policies (exp >= 0.95)", "higher"),
    ("tweedie_deviance_1.5", "Mean Tweedie deviance (p=1.5)", "lower"),
    ("calibration_ratio", "Calibration (predicted / actual)", "one"),
]

# Vertical rows spanned by an 8.4cm chart at default row height, used to keep chart
# anchors from colliding on rungs whose tables are short.
CHART_ROWS = 18

INK = "0B0B0B"
MUTED = "7B7A75"
RULE = "DCDCD7"
HEAD_FILL = PatternFill("solid", fgColor="ECECEA")
THIN = Side(style="thin", color=RULE)
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


# --------------------------------------------------------------------------- data


def load_records() -> dict:
    """(n, model) -> {metric: [values across draws]} plus the per-draw key list."""
    out: dict = {}
    for p in sorted(CACHE.glob("*.json")):
        rec = json.loads(p.read_text())
        cfg = rec.get("config", {})
        if (cfg.get("track") != "B" or rec.get("status") != "ok"
                or cfg.get("n") not in RUNGS
                or cfg.get("test_rows", FULL_TEST_ROWS) != FULL_TEST_ROWS):
            continue
        d = out.setdefault((cfg["n"], cfg["model"]), {})
        for k, v in rec.get("metrics", {}).items():
            d.setdefault(k, []).append(float(v))
        d.setdefault("_seeds", []).append(cfg["seed"])
    return out


def load_lift() -> dict:
    """(n, model) -> lift table averaged over draws, from saved predictions."""
    df, _ = build_and_cache()
    _, test = random_split(df)
    y = test["PurePremium"].to_numpy()
    e = test["Exposure"].to_numpy()

    preds: dict = {}
    for p in sorted(CACHE.glob("*.json")):
        cfg = json.loads(p.read_text()).get("config", {})
        npz = p.with_name(f"{p.stem}_pred.npz")
        if (cfg.get("track") == "B" and npz.exists() and cfg.get("n") in RUNGS
                and cfg.get("test_rows", FULL_TEST_ROWS) == FULL_TEST_ROWS):
            with np.load(npz) as z:
                preds.setdefault((cfg["n"], cfg["model"]), []).append(z["pred_test"])

    out = {}
    for key, arrs in preds.items():
        lt = M.lift_table(y, np.mean(arrs, axis=0), e, n_buckets=10)
        out[key] = lt
    return out


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
                c.font = Font(size=9, bold=False, color=INK)
            else:
                c.font = Font(size=9, name="Consolas")
                c.alignment = Alignment(horizontal="right")
                if isinstance(v, (int, float)):
                    c.number_format = number_format
    if widths:
        for j, w in enumerate(widths):
            ws.column_dimensions[get_column_letter(col + j)].width = w
    return row + len(rows) + 1


def style_series(s, hexcolor, *, line=True, marker="circle", width=22000):
    """openpyxl defaults to thick lines and no markers; set both explicitly."""
    if line:
        s.graphicalProperties.line = LineProperties(solidFill=hexcolor, w=width)
    else:
        s.graphicalProperties.line.noFill = True
    s.marker = Marker(symbol=marker, size=6)
    s.marker.graphicalProperties.solidFill = hexcolor
    s.marker.graphicalProperties.line.solidFill = hexcolor
    s.smooth = False


# ------------------------------------------------------------------ sheet 1


def sheet_metrics(wb, recs):
    ws = wb.create_sheet("Metrics by rung")
    ws.sheet_view.showGridLines = False
    r = 1
    r = title(ws, r, "Track B — TabICL pricing benchmark (freMTPL2)", 15)
    r = note(ws, r, "678,013 policies · 542,410 train / 135,603 test · 614 fitted configs · "
                    "20 resample draws at N=5k–20k, 5 at 100k, 3 at full")
    r = note(ws, r, "Trivial 1/exposure reference: rate Gini 0.4676, total-loss Gini 0.000, "
                    "deviance ~93.5–95.5. Any model at or below that line has learned nothing.")
    r += 1

    # ---- block 1: evaluation metrics per rung ----
    r = title(ws, r, "1 · Evaluation metrics by rung", 12)
    r = note(ws, r, "Mean across resample draws. Best competitor per rung in bold.")
    r += 1

    for key, label, better in METRICS:
        r = title(ws, r, label, 10)
        headers = ["Model"] + [f"N = {n:,}" for n in RUNGS]
        rows = []
        for m in MODELS:
            vals = []
            for n in RUNGS:
                v = recs.get((n, m), {}).get(key)
                vals.append(round(float(np.mean(v)), 4) if v else None)
            rows.append([m] + vals)
        start = r
        r = write_table(ws, r, 1, headers, rows, widths=[24] + [13] * len(RUNGS))

        # Bold the best competitor in each rung column.
        for j, n in enumerate(RUNGS):
            best_i, best_v = None, None
            for i, m in enumerate(COMPETITORS):
                v = rows[i][1 + j]
                if v is None:
                    continue
                if better == "higher":
                    ok = best_v is None or v > best_v
                elif better == "lower":
                    ok = best_v is None or v < best_v
                else:
                    ok = best_v is None or abs(v - 1) < abs(best_v - 1)
                if ok:
                    best_i, best_v = i, v
            if best_i is not None:
                c = ws.cell(row=start + 1 + best_i, column=2 + j)
                c.font = Font(size=9, name="Consolas", bold=True, color="2A78D6")
        r += 1

    # ---- block 2: summary read ----
    r += 1
    r = title(ws, r, "2 · Summary analysis", 12)
    r = note(ws, r, "Two-sample z on the difference of means, using the per-rung SD and draw count.")
    r += 1

    summary = [
        ["Ranking — first rung TabICL clears the trivial floor", "N = 5,000",
         "total-loss Gini 0.331 vs 0.000; z ≈ 2.6 vs best rival"],
        ["Ranking — TabICL's strongest rung", "N = 20,000",
         "0.412 vs 0.239 for the next model; z ≈ 7.5"],
        ["Ranking — where TabICL loses", "N = 542,410",
         "xgb_hurdle 0.543 vs TabICL 0.451; z = −3.5"],
        ["Pricing — trivial model beats every fitted model", "N = 5,000",
         "1/exposure 95.52 vs TabICL 98.64 deviance"],
        ["Pricing — first rung TabICL leads, but not significantly", "N = 20,000",
         "93.26 vs 94.61; z = 1.6 (not significant)"],
        ["Pricing — first significant win over the trivial model", "N = 100,000",
         "88.08 vs 94.00; z = 3.1"],
        ["Calibration across all rungs", "0.39 – 0.45",
         "Models price the portfolio at roughly half its actual cost"],
        ["Degenerate fit detected", "xgb_hurdle, N ≤ 5,000",
         "Gini 0.46760699 bit-identical across seeds = the 1/exposure ordering"],
    ]
    r = write_table(ws, r, 1, ["Finding", "Where", "Evidence"], summary,
                    widths=[46, 20, 58], number_format="General")

    # ---- block 3: variance analysis ----
    r += 1
    r = title(ws, r, "3 · Variance analysis", 12)
    r = note(ws, r, "Spread across resample draws. At N = 542,410 the subsample is the whole "
                    "pool, so deterministic models are constant by construction and SD there "
                    "reflects model-internal randomness only — not comparable to smaller rungs.")
    r += 1

    for key, label, _ in METRICS[:2] + [METRICS[4]]:
        r = title(ws, r, label, 10)
        headers = ["Model", "Rung", "Draws", "Mean", "SD", "CV", "Min", "Max", "Range"]
        rows = []
        for m in MODELS:
            for n in RUNGS:
                v = recs.get((n, m), {}).get(key)
                if not v:
                    continue
                a = np.asarray(v, dtype=float)
                mean = float(a.mean())
                sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
                rows.append([m, f"{n:,}", len(a), round(mean, 4), round(sd, 4),
                             round(sd / abs(mean), 4) if abs(mean) > 1e-9 else None,
                             round(float(a.min()), 4), round(float(a.max()), 4),
                             round(float(a.max() - a.min()), 4)])
        r = write_table(ws, r, 1, headers, rows,
                        widths=[24, 12, 8, 11, 11, 11, 11, 11, 11])
        r += 1

    ws.freeze_panes = "A2"
    return ws


# ------------------------------------------------------------------ sheet 2


def sheet_charts(wb, recs, lifts):
    ws = wb.create_sheet("Charts")
    ws.sheet_view.showGridLines = False
    r = 1
    r = title(ws, r, "Distribution and lift by rung", 15)
    r = note(ws, r, "Each chart sits beside the table it is drawn from. Distribution charts "
                    "plot every resample draw as a point, so the spread is visible rather "
                    "than summarised.")
    r += 1

    for n in RUNGS:
        r = title(ws, r, f"N = {n:,}", 13)
        top = r

        # ---------- distribution: every draw, per model ----------
        key = "gini_total_loss"
        cols = [m for m in COMPETITORS if recs.get((n, m), {}).get(key)]
        ndraw = max(len(recs[(n, m)][key]) for m in cols) if cols else 0

        ws.cell(row=r, column=1, value=f"Total-loss Gini — {ndraw} draws").font = \
            Font(bold=True, size=10)
        r += 1
        hdr_row = r
        headers = ["Draw"] + cols
        rows = []
        for i in range(ndraw):
            row = [i + 1]
            for m in cols:
                v = recs[(n, m)][key]
                row.append(round(float(v[i]), 4) if i < len(v) else None)
            rows.append(row)
        end = write_table(ws, r, 1, headers, rows, widths=[8] + [14] * len(cols))

        # X positions for a strip plot: each model on its own vertical track.
        xcol = 1 + len(cols) + 2
        ws.cell(row=hdr_row, column=xcol, value="x").font = Font(bold=True, size=9)
        for i in range(ndraw):
            ws.cell(row=hdr_row + 1 + i, column=xcol, value=1)

        ch = ScatterChart()
        ch.title = f"Total-loss Gini distribution, N = {n:,}"
        ch.style = 2
        ch.height, ch.width = 8.4, 15.5
        ch.x_axis.title = "model"
        ch.y_axis.title = "Gini on total loss"
        ch.x_axis.scaling.min, ch.x_axis.scaling.max = 0, len(cols) + 1
        ch.x_axis.majorGridlines = None
        # The x positions are arbitrary tracks, not a measured quantity -- showing the
        # numbers would invite reading them as data. The legend carries identity.
        ch.x_axis.delete = True
        ch.legend.position = "b"

        for ci, m in enumerate(cols, start=1):
            # One x-column per model so points sit on separate tracks.
            xc = xcol + ci
            ws.cell(row=hdr_row, column=xc, value=m).font = Font(bold=True, size=8)
            for i in range(ndraw):
                ws.cell(row=hdr_row + 1 + i, column=xc, value=ci)
            xref = Reference(ws, min_col=xc, min_row=hdr_row + 1, max_row=hdr_row + ndraw)
            yref = Reference(ws, min_col=1 + ci, min_row=hdr_row, max_row=hdr_row + ndraw)
            s = Series(yref, xref, title_from_data=True)
            style_series(s, SERIES_HEX[m], line=False, marker="circle")
            ch.series.append(s)

        ws.add_chart(ch, f"{get_column_letter(xcol + len(cols) + 2)}{top}")

        # The x-position helper columns are chart plumbing, not results -- hide them so
        # the sheet reads as the tables it is meant to show.
        for c in range(xcol, xcol + len(cols) + 1):
            ws.column_dimensions[get_column_letter(c)].hidden = True

        # ---------- lift: table beside its chart ----------
        lift_row = end + 1
        ws.cell(row=lift_row, column=1,
                value="Lift — 10 equal-exposure buckets").font = Font(bold=True, size=10)
        lift_row += 1

        lift_models = [m for m in ["tabicl", "xgb_hurdle", "xgb_tweedie"] if (n, m) in lifts]
        if not lift_models:
            r = lift_row + 2
            continue

        base = lifts[(n, lift_models[0])]
        headers = ["Bucket", "Exposure", "Actual"] + [f"{m} pred" for m in lift_models]
        rows = []
        for i in range(len(base)):
            row = [int(base["bucket"].iloc[i]), round(float(base["exposure"].iloc[i]), 1),
                   round(float(base["actual_loss_cost"].iloc[i]), 2)]
            for m in lift_models:
                row.append(round(float(lifts[(n, m)]["predicted_loss_cost"].iloc[i]), 2))
            rows.append(row)
        lhdr = lift_row
        end2 = write_table(ws, lift_row, 1, headers, rows,
                           widths=[9, 12, 12] + [15] * len(lift_models),
                           number_format="0.00")

        # Grouped columns for actual + the two headline models. A BarChart combined with
        # a ScatterChart drops the overlay silently in openpyxl, and combining chart
        # types would also put the predictions on a secondary axis -- both series are
        # loss cost on the same scale, so they must share one axis.
        plotted = lift_models[:2]
        bc = BarChart()
        bc.type = "col"
        bc.title = f"Lift, N = {n:,} — actual vs predicted loss cost"
        bc.height, bc.width = 8.4, 15.5
        bc.y_axis.title = "loss cost per unit exposure"
        bc.x_axis.title = "equal-exposure bucket (ordered by predicted)"
        bc.legend.position = "b"
        bc.gapWidth = 60
        bc.overlap = -10
        data = Reference(ws, min_col=3, max_col=3 + len(plotted),
                         min_row=lhdr, max_row=lhdr + len(rows))
        bc.add_data(data, titles_from_data=True)
        bc.set_categories(Reference(ws, min_col=1, min_row=lhdr + 1, max_row=lhdr + len(rows)))
        fills = ["C9C9C3"] + [SERIES_HEX[m] for m in plotted]
        for s, hexc in zip(bc.series, fills):
            s.graphicalProperties.solidFill = hexc
            s.graphicalProperties.line.noFill = True

        # An 8.4cm chart spans roughly CHART_ROWS rows. The rungs with few draws have
        # short tables, so anchoring the lift chart at its table would overlap the
        # distribution chart above it -- hold the anchors CHART_ROWS apart regardless.
        lift_chart_row = max(lift_row, top + CHART_ROWS)
        ws.add_chart(bc, f"{get_column_letter(xcol + len(cols) + 2)}{lift_chart_row}")

        r = max(end2, lift_chart_row + CHART_ROWS) + 2

    return ws


def main() -> int:
    recs = load_records()
    if not recs:
        print("no cached Track B results")
        return 1
    print(f"loaded {len(recs)} (rung, model) cells")
    lifts = load_lift()
    print(f"computed {len(lifts)} lift tables from saved predictions")

    wb = Workbook()
    wb.remove(wb.active)
    sheet_metrics(wb, recs)
    sheet_charts(wb, recs, lifts)
    RESULTS.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
