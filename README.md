# TabICL pricing POC — freMTPL2

Can [TabICL](https://github.com/soda-inria/tabicl), a tabular foundation model, rank risk and
price motor insurance competitively — particularly when data is thin? This repo is the
evidence base for that decision, benchmarked against tuned XGBoost and GLM baselines on the
French motor dataset (freMTPL2).

**Status: Tracks 0, A and B complete. Track C (fine-tuning) partially run and paused.**

## Headline findings

**Ranking and pricing give different answers, and the distinction is the whole result.**

- **Ranking — TabICL is competitive from N=5,000.** On the artifact-free `gini_total_loss`
  it leads every classical baseline at N=5,000–100,000, decisively at N=20,000
  (0.412 vs 0.237, z ≈ 7.5 over 20 draws).
- **Pricing — it is not.** It does not significantly out-price a *constant premium* until
  N=100,000. At N=5,000 the trivial model is genuinely better. Calibration sits at
  0.45 against a target of 1.0 — under-predicting loss cost by more than half.
- **Under regional distribution shift it gets worse, not better.** Leave-one-region-out
  (Track A): `xgb_hurdle` takes three of five regions. On R11 (Île-de-France, the most
  distinct region) TabICL loses ranking at z = −6.39, over-predicts by 2.01×, and is the
  only model whose price is *beaten by charging every policy the same premium*.
- **At full data XGBoost wins clearly**: 82.68 vs 87.28 deviance.

Two methodological findings shaped everything above:

- **The rate-based Gini has a degenerate solution.** A model predicting `c/exposure` charges
  every policy an identical premium — zero risk information — yet scores 0.4676 exposure-weighted
  Gini. `xgb_hurdle` reproduced this exactly at small N. All ranking claims here use
  `gini_total_loss`, which ranks by predicted *total loss* and scores such a model at 0.000.
- **Draw-to-draw tail variation drives calibration, not ranking.** Across Track B's 20 draws,
  tail content correlates with calibration at Spearman +0.91 for TabICL — against +1.00 for a
  featureless model — while showing no association with `gini_total_loss` at all.

## Tracks

| Track | Question | Status |
|---|---|---|
| 0 | Environment, data, VRAM, baselines | complete |
| A | Region censoring — train on 21 regions, test on the held-out one | complete, 95 configs |
| B | Thin-data learning curve + R=20 resampling study | complete, 614 configs |
| C | Does fine-tuning close the pricing gap? | ICL arm complete, fine-tuned arm paused |
| D | Does TabICL need feature engineering? | not started |

Out-of-time validation was **dropped by decision**: freMTPL2 carries no date column, so a
2011-12 / 2013 split cannot be constructed.

## Layout

```
src/data.py                 load, clean, split, subsamplers (exposure- and severity-stratified)
src/metrics.py              the single metrics implementation used by every track
src/checks.py               12 automated logic checks (degeneracy, trivial-floor, calibration...)
src/contexts.py             ICL context construction + prior correction
src/models/                 glm.py, gbm.py, tabicl_twostage.py
src/runner.py               resumable config runner, per-config caching
experiments/                one script per track, plus diagnostics and analysis
results/                    metrics parquet + results/cache/ (cached configs and predictions)
reports/                    charts and write-ups
```

## Reproducing

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # see note below
.venv\Scripts\python experiments\track0_env_check.py    # verifies sm_120 + cu128
.venv\Scripts\python experiments\trackA_region.py       # ~1 h on an RTX 5070 Ti Laptop
.venv\Scripts\python experiments\trackB_curve.py
```

Every run is resumable: results are cached under `results/cache/` keyed by a hash of the config,
so re-running skips completed work. Predictions are persisted alongside the metrics, so a new
metric can be computed without re-fitting.

`data/freMTPL2_clean.parquet` and `data/split_hashes.json` are committed, so a clone reproduces
the identical 542,410 / 135,603 split. `build_and_cache()` will otherwise fetch from OpenML
(`data_id` 41214 and 41215).

## Environment

RTX 5070 Ti Laptop (12 GB, sm_120, **50 W power cap**), torch 2.11.0+cu128. Notes that cost real
time to learn:

- sm_120 needs cu128 or newer; older cu121 wheels install cleanly then fail at runtime.
- `use_fa3=False` — FlashAttention-3 is Hopper-only.
- Fine-tuning at N=10,000 peaks at ~13.9 GB against a 12.2 GB card. Windows spills to host
  memory rather than raising, so the OOM ladder never fires — degradation shows up as
  slowness, not an exception.

## Caveats carried forward

- **R43 has 36 test claims.** Its Track A column is indicative only.
- **`data/region_summary.csv` disagrees with the cleaned frame** on loss cost (it lists R24 at
  449.8; the data gives 185.7) and reverses the R11/R93 ordering. Do not cite it until reconciled.
- **Track C's fine-tuning objective is not the pricing objective.** The library trains stage 1 on
  cross-entropy and stage 2 on pinball loss, offers no Tweedie option, and supports neither
  sample weights nor an exposure offset. Exposure enters only as a feature.
