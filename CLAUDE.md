# CLAUDE.md — TabICL pricing POC

Benchmarks TabICL (a tabular foundation model) against tuned XGBoost and GLM baselines for
motor pure-premium pricing on freMTPL2. The question driving it is a build/buy decision:
**can TabICL rank risk and price competitively, especially when data is thin?**

Repo: https://github.com/LexaZhong/tabicl-pricing (private). 777 cached configs.

## Run commands

```powershell
.venv\Scripts\python.exe experiments\trackA_region.py      # region censoring
.venv\Scripts\python.exe experiments\trackB_curve.py       # learning curve
.venv\Scripts\python.exe experiments\trackC_run.py         # fine-tuning contrast
.venv\Scripts\python.exe experiments\export_excel.py       # rebuild the workbook
```

Everything is resumable: `results/cache/` is keyed by a hash of the config dict, so a
re-run skips completed work. **Any field you add to a config changes its hash and forces a
re-run** — that is the intended way to supersede stale results.

`pip` is not installed in the venv (it was created by `uv`). Use
`importlib.metadata` to regenerate `requirements.txt`, not `pip freeze`.

---

## Rules that must not be broken

These were each learned by getting them wrong first. Violating any one invalidates results.

1. **Rank on `gini_total_loss`, never `gini_exposure_weighted`.** The rate-based Gini has a
   degenerate solution: a `c/exposure` model charges every policy an identical premium —
   zero risk information — yet scores **0.4676** exposure-weighted Gini. `xgb_hurdle`
   reproduced this exactly at N≤5,000. Ranking by predicted *total loss* scores it 0.000.
2. **`one_over_exposure` is the trivial floor. `intercept` is NOT.** `intercept` predicts a
   constant *rate*, so its predicted loss tracks exposure and it scores −0.14 to +0.37
   across regions. Every track needs `one_over_exposure` as a reference line.
3. **XGBoost must be Optuna-tuned wherever it is compared.** Tuning lives in
   `trackB_curve.tune_xgb` and was ported to `trackA_region.py`; a new track must port it
   too. An untuned incumbent flatters TabICL and will not survive review. `trials` belongs
   in the config hash.
4. **Persist predictions** (`run_config(..., require_predictions=True)`). Track B was run
   without this and adding `gini_total_loss` later forced a full re-fit.
5. **Analyse paired, within seed.** The same seed gives the same draw. 87–95% of
   calibration variance is common to the draw, so paired contrasts detect ~3× smaller
   effects. Summarising arms independently throws that away.
6. **Compute caps/encoders on the training side only.** In Track A the severity cap and the
   TargetEncoder must come from `Region != r`, or the censored region leaks.
7. **Draw-to-draw tail variation drives calibration, not ranking.** Spearman +0.91 for
   TabICL calibration vs draw tail content, against +1.00 for a featureless model. Never
   report a single-draw calibration number.

## Environment gotchas

- RTX 5070 Ti Laptop, 12 GB, **50 W power cap**. Sustained load settles to ~40% of peak
  clock; the first configs of a run are misleadingly fast.
- sm_120 requires torch **cu128**; a default PyPI cu121 wheel installs cleanly then fails at
  runtime. `use_fa3=False` (FlashAttention-3 is Hopper-only).
- **Windows spills GPU→host memory instead of raising OOM.** Fine-tuning at N=10,000 peaks
  at 13.9 GB on a 12.2 GB card. The OOM ladder in `trackC_run.py` will never fire —
  degradation appears as slowness, not an exception.
- TabICL cost is superlinear in context. Scoring the fixed test set: ~95 s / 153 s / 265 s
  at context 5k / 10k / 20k — but **6,412 s at N=542,410**, because stage 2's context
  becomes the whole ~19,907-claim subset.
- openpyxl: **`BarChart` + `ScatterChart` fails silently** and drops the overlay. Combined
  charts must be `LineChart` + `BarChart`. Always verify series counts on the saved file.

---

## Done

**Track 0 — setup.** Environment, data cleaning, VRAM curve, baselines. Reproduces the
team's 542,410 / 135,603 split exactly.

**Track B — thin-data learning curve.** 5 rungs (5k/10k/20k/100k/542,410), 20 resample
draws at the first four. *Ranking yes, pricing no*: TabICL leads `gini_total_loss` from
N=5,000 and decisively at 20,000 (0.412 vs 0.239, z≈7.5), but does not out-price a constant
premium until N=100,000; at 5,000 the trivial model is better. Calibration ~0.45. At full
data XGBoost wins (82.68 vs 87.28 deviance).

**Track A — region censoring.** 95 configs, leave-one-region-out over R24/R82/R93/R11/R43,
train on the other 21 regions. `xgb_hurdle` takes 3 of 5. **R11 (Île-de-France, the most
distinct region) is where TabICL breaks**: ranking z=−6.39, calibration 2.01× over-predicted,
and the only model in the study beaten on price by charging every policy the same premium
(84.14 vs the floor's 79.32). Context-draw SD is 5–35× `xgb_hurdle`'s even with 600k rows
available.

**Feature-engineering ablation** (`tabicl_raw`, 9 raw columns vs 15 engineered), 20 paired
draws per rung. Engineering is worth 0.060 total-loss Gini at N=5,000 (p=0.0009, raw loses
17/20) and 1.56 deviance at 20,000 (p=0.0004) — then **reverses by 100,000**, where raw wins
by 0.034 Gini (p=0.0002, 18/20). This supersedes the originally planned Track D.

**Deliverables.** `results/trackB_analysis.xlsx` — 4 sheets (Overview / Q1 thin data /
Q2 censored regions / Q3 feature engineering), 26 native charts; all lift charts plot loss
cost as lines on the primary axis and exposure as bars on the secondary. Published page for
Track B: https://claude.ai/code/artifact/65e777e1-7de8-41d3-a62b-3fd5ca1b5234

---

## TODO

**1. Track C — finish the fine-tuned arm.** *(~5.5h, 19 configs)*
ICL arm is complete (20 configs). The fine-tuned arm is paused at 1 of 20.
`DisjointFinetunedTabICLTwoStage` fine-tunes on a disjoint 10,000 then conditions on the
20,000 context — necessary because the stock estimator installs the fine-tuning set *as* the
in-context examples (`_finetune/base.py:1119-1121`), confounding weight learning with
in-context visibility.
**Restart required before resuming**: `_EpochCounter` is committed but has never run, so we
cannot yet distinguish a genuine null from a fine-tune that early-stopped around epoch 11.
Seed 0 currently shows degradation worsening with training (96.65 ICL → 97.65 at 2 epochs →
99.85 at ~11).

**2. Re-plan Track C's objective.** The library trains stage 1 on cross-entropy and stage 2
on pinball loss, offers no Tweedie option, and supports neither sample weights nor an
exposure offset — exposure enters only as a feature. **Nothing in fine-tuning optimises the
pricing objective**, which is the most likely reason it degrades. Also: stage 2 fine-tunes on
only ~409 claim rows with a 41-row validation split driving `patience=8`, and gets seconds of
compute against stage 1's 13.8 minutes.

**3. Tail-balanced arm — built, validated, never run.** `severity_stratified_subsample` and
`experiments/trackC_arm_check.py` exist. On the capped target it cuts draw SD 12.44 → 4.86
(61%). Pairs with the fixed cap, which neutralises the residual `max_claim` spread.

**4. `tabicl_raw` at N=542,410.** *(~5.3h, 3 configs at 107 min each)* Deliberately skipped;
the Q3 crossover is established at 100,000 without it.

**5. Fix or delete `data/region_summary.csv`.** It contradicts the cleaned frame — lists R24
loss cost 449.8 where the frame gives 185.7, and reverses R11/R93. Currently unused and
flagged in the workbook, but it partly justified the region picks.

**6. Caveat to carry into any write-up.** R43 has **36 test claims**; its Track A column is
indicative only and must not sit unmarked beside the other four.
