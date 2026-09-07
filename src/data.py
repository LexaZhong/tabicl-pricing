"""freMTPL2 loading, cleaning and splitting.

Cleaning follows the canonical Noll-Salzmann-Wuthrich / scikit-learn convention so our
numbers stay comparable to published benchmarks on this dataset.

Targets differ by model family -- see the plan's target table:
  * TabICL stage 1 (classifier): has_claim = 1{ClaimNb >= 1}, all rows, exposure as a FEATURE
  * TabICL stage 2 (regressor):  total loss, claims-only rows
  * Evaluation (all models):     PurePremium = loss / Exposure, weight = Exposure
  * Poisson baselines:           ClaimNb with log(Exposure) offset
  * Gamma baselines:             Severity = loss / ClaimNb, claims-only, weight = ClaimNb

`Frequency` is NOT a TabICL two-stage target; it exists only for the Poisson baselines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FREQ_OPENML_ID = 41214
SEV_OPENML_ID = 41215

# Column groups shared across the codebase.
RAW_NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
RAW_CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
ID_COL = "IDpol"

MAX_EXPOSURE = 1.0
MAX_CLAIM_NB = 4

# The split the team already ran: 678,013 -> 542,410 train / 135,603 test.
DEFAULT_TEST_SIZE = 135_603
DEFAULT_SPLIT_SEED = 0


@dataclass
class CleaningReport:
    """Counts a reviewer will ask about. Persisted next to the cleaned parquet."""

    n_freq_rows: int
    n_sev_rows: int
    n_sev_unique_policies: int
    claimnb_sum_before_clip: int
    n_exposure_clipped: int
    n_claimnb_clipped: int
    # The known freMTPL2 join mismatch:
    n_claim_no_severity: int  # ClaimNb > 0 but no severity record
    n_severity_no_claim: int  # severity record but ClaimNb == 0
    n_claims_only_subset: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _strip_quotes(s: pd.Series) -> pd.Series:
    """OpenML ships freMTPL2 categoricals as "'B12'" / "'Regular'" with literal quotes."""
    return s.astype(str).str.strip().str.strip("'\"")


def load_raw(cache_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch freMTPL2freq/sev from OpenML, caching to parquet."""
    from sklearn.datasets import fetch_openml

    cache_dir.mkdir(parents=True, exist_ok=True)
    freq_path = cache_dir / "freMTPL2freq_raw.parquet"
    sev_path = cache_dir / "freMTPL2sev_raw.parquet"

    if freq_path.exists() and sev_path.exists():
        return pd.read_parquet(freq_path), pd.read_parquet(sev_path)

    freq = fetch_openml(data_id=FREQ_OPENML_ID, as_frame=True).data
    sev = fetch_openml(data_id=SEV_OPENML_ID, as_frame=True).data

    freq.to_parquet(freq_path)
    sev.to_parquet(sev_path)
    return freq, sev


def clean(freq: pd.DataFrame, sev: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clip exposure/counts, join severity, build every target column."""
    df = freq.copy()
    df[ID_COL] = df[ID_COL].astype(np.int64)

    for col in RAW_CATEGORICAL:
        df[col] = _strip_quotes(df[col])

    for col in RAW_NUMERIC + ["Exposure", "ClaimNb"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)

    claimnb_sum_before = int(df["ClaimNb"].sum())
    n_exposure_clipped = int((df["Exposure"] > MAX_EXPOSURE).sum())
    n_claimnb_clipped = int((df["ClaimNb"] > MAX_CLAIM_NB).sum())

    # Exposure must be strictly positive: it is a denominator and a log() argument.
    df = df[df["Exposure"] > 0].copy()
    df["Exposure"] = df["Exposure"].clip(upper=MAX_EXPOSURE)
    df["ClaimNb"] = df["ClaimNb"].clip(upper=MAX_CLAIM_NB).astype(int)

    # Severity: sum all claim amounts per policy.
    sev = sev.copy()
    sev[ID_COL] = sev[ID_COL].astype(np.int64)
    sev_by_policy = sev.groupby(ID_COL)["ClaimAmount"].sum()

    df["ClaimAmount"] = df[ID_COL].map(sev_by_policy).fillna(0.0).astype(float)

    # The documented freMTPL2 join mismatch -- log it, do not silently patch it.
    n_claim_no_sev = int(((df["ClaimNb"] > 0) & (df["ClaimAmount"] <= 0)).sum())
    n_sev_no_claim = int(((df["ClaimNb"] == 0) & (df["ClaimAmount"] > 0)).sum())

    # A severity record with ClaimNb == 0 is unusable for a hurdle model: the claim
    # indicator and the loss disagree. Promote the count to 1 so the two stages stay
    # consistent with each other and with the pure-premium target.
    promoted = (df["ClaimNb"] == 0) & (df["ClaimAmount"] > 0)
    df.loc[promoted, "ClaimNb"] = 1

    # --- targets -------------------------------------------------------------
    df["TotalLoss"] = df["ClaimAmount"]                          # TabICL stage 2
    df["has_claim"] = (df["ClaimNb"] >= 1).astype(int)           # reported-claim indicator

    # PRIMARY stage-1 target. ~9.1k policies (27% of all claim policies) have ClaimNb > 0
    # but no severity record in freMTPL2sev, so has_claim and "a loss was observed" are
    # NOT the same event. The hurdle decomposition of pure premium is exact only for the
    # latter:
    #     E[loss] = P(loss > 0) * E[loss | loss > 0]
    # Using has_claim instead pairs P(claim) with E[loss | loss > 0], which overstates
    # expected loss by ~1/0.73 and distorts the ranking too. `has_claim` is retained as a
    # sensitivity arm, not as the default.
    df["has_loss"] = (df["ClaimAmount"] > 0).astype(int)
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]       # evaluation
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]             # Poisson baselines only
    df["Severity"] = np.where(                                   # Gamma baselines
        df["ClaimNb"] > 0, df["ClaimAmount"] / df["ClaimNb"].clip(lower=1), np.nan
    )
    df["log_exposure"] = np.log(df["Exposure"])

    df = df.reset_index(drop=True)

    report = CleaningReport(
        n_freq_rows=len(freq),
        n_sev_rows=len(sev),
        n_sev_unique_policies=int(sev_by_policy.shape[0]),
        claimnb_sum_before_clip=claimnb_sum_before,
        n_exposure_clipped=n_exposure_clipped,
        n_claimnb_clipped=n_claimnb_clipped,
        n_claim_no_severity=n_claim_no_sev,
        n_severity_no_claim=n_sev_no_claim,
        n_claims_only_subset=int((df["ClaimNb"] > 0).sum()),
    )
    return df, report


def claims_only(df: pd.DataFrame) -> pd.DataFrame:
    """Stage-2 training subset: policies with a claim AND a positive recorded loss.

    Policies with ClaimNb > 0 but no severity record stay in the frequency/stage-1 data
    and are excluded here -- we cannot fit a loss model on a loss we never observed.
    """
    return df[(df["ClaimNb"] > 0) & (df["TotalLoss"] > 0)].copy()


def cap_target(y: np.ndarray, quantile: float = 0.995) -> tuple[np.ndarray, float]:
    """Cap the stage-2 TRAINING target. Never applied to test data.

    The cap must be computed on the training claims subset for the current
    fold/region/N -- computing it on the full data leaks the held-out tail.
    """
    cap = float(np.quantile(y, quantile))
    return np.minimum(y, cap), cap


# --- splitting ---------------------------------------------------------------


def random_split(
    df: pd.DataFrame, test_size: int = DEFAULT_TEST_SIZE, seed: int = DEFAULT_SPLIT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The team's existing 542,410 / 135,603 split. Held fixed across every track."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    test_idx = np.sort(idx[:test_size])
    train_idx = np.sort(idx[test_size:])
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def region_split(df: pd.DataFrame, region: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Track A: train on Region != r, test on Region == r."""
    test = df[df["Region"] == region].copy()
    train = df[df["Region"] != region].copy()
    if len(test) == 0:
        raise ValueError(f"region {region!r} not present; have {sorted(df['Region'].unique())}")
    assert region not in set(train["Region"].unique()), "censored region leaked into train"
    return train, test


def exposure_stratified_subsample(
    df: pd.DataFrame, n: int, seed: int, n_strata: int = 10
) -> pd.DataFrame:
    """Track B: subsample to n rows keeping the exposure distribution intact.

    Plain random sampling would let small-N draws drift in mean exposure, which moves the
    pure-premium scale and confounds the learning curve with an exposure-mix effect.
    """
    if n >= len(df):
        return df.copy()

    rng = np.random.default_rng(seed)
    strata = pd.qcut(df["Exposure"], q=n_strata, labels=False, duplicates="drop")

    picks: list[np.ndarray] = []
    positions = np.arange(len(df))
    for s in np.unique(strata):
        in_s = positions[strata.values == s]
        take = int(round(n * len(in_s) / len(df)))
        take = min(take, len(in_s))
        if take > 0:
            picks.append(rng.choice(in_s, size=take, replace=False))

    chosen = np.concatenate(picks) if picks else np.array([], dtype=int)

    # Rounding can leave us a few short or long; top up / trim at random.
    if len(chosen) < n:
        remaining = np.setdiff1d(positions, chosen)
        extra = rng.choice(remaining, size=min(n - len(chosen), len(remaining)), replace=False)
        chosen = np.concatenate([chosen, extra])
    elif len(chosen) > n:
        chosen = rng.choice(chosen, size=n, replace=False)

    return df.iloc[np.sort(chosen)].copy()


# Severity band edges as quantiles of the POOL's claim distribution. The tail is split
# finely because that is where the variance lives: a single band spanning q99->max would
# hold claims from 18k to 1.4M and fixing its count would not fix its magnitude.
SEVERITY_BANDS = (0.0, 0.5, 0.9, 0.99, 0.995, 0.999, 1.0)


def severity_stratified_subsample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    base: pd.DataFrame | None = None,
    bands: tuple[float, ...] = SEVERITY_BANDS,
    n_strata: int = 10,
) -> pd.DataFrame:
    """Track C arm 2: same size and claim rate as the SRS draw, but tail coverage is fixed.

    `exposure_stratified_subsample` matches the covariate side almost perfectly (mean
    exposure to 0.13%, claim count to 5%) yet still lets sample loss cost vary 4x across
    seeds, because it does not control WHICH claims land in the draw: 74% of 10k draws hold
    no claim above the portfolio q99.9, and the ones that do can hold a 1.4M claim. Track B
    showed that variation moves calibration (Spearman +0.91 for TabICL, against +1.00 for a
    featureless model) while leaving Gini untouched -- i.e. it acts on exactly the pricing
    axis Track C is testing.

    This sampler fills each severity band to a fixed quota, so every seed sees the same
    tail SHAPE and only the identity of the claims changes.

    Passing `base` (the SRS draw for the same seed) makes the two arms a MATCHED PAIR: the
    non-claim rows and the claim count are taken from it unchanged, so the arms differ in
    claim composition and nothing else. That turns the arm contrast into a paired
    comparison, which is where most of its power comes from.

    Note this is outcome-dependent selection: quotas are rounded UP to 1 in sparse tail
    bands, so the tail is mildly over-represented relative to the portfolio and the fitted
    level is biased high. That is intentional and correctable -- report it with
    `metrics.balance_factor` rather than pretending the draw is representative.
    """
    pool_claims = df[df["has_loss"] == 1]
    if len(pool_claims) == 0:
        raise ValueError("pool contains no claims; cannot severity-stratify")

    rng = np.random.default_rng(seed + 900_000)  # disjoint from the SRS seed stream

    if base is not None:
        non_claims = base[base["has_loss"] == 0]
        n_claims = int((base["has_loss"] == 1).sum())
    else:
        srs = exposure_stratified_subsample(df, n, seed, n_strata=n_strata)
        non_claims = srs[srs["has_loss"] == 0]
        n_claims = int((srs["has_loss"] == 1).sum())

    sev = pool_claims["TotalLoss"].to_numpy()
    edges = np.quantile(sev, bands)
    edges[0], edges[-1] = -np.inf, np.inf

    # Quota per band = its portfolio share of claims, but never 0 for a band that exists:
    # a quota of 0 in the top band would silently turn this into a tail-censored arm.
    shares = np.diff(bands)
    quotas = np.maximum(1, np.round(n_claims * shares).astype(int))
    # Rounding up the sparse bands overshoots; take the excess off the largest band, which
    # is the one least sensitive to losing a few rows.
    while quotas.sum() > n_claims:
        quotas[int(np.argmax(quotas))] -= 1

    picks = []
    for i, quota in enumerate(quotas):
        in_band = pool_claims[(sev >= edges[i]) & (sev < edges[i + 1])]
        if len(in_band) == 0:
            continue
        take = min(int(quota), len(in_band))
        # Exposure-stratify inside the band so the tail rows do not drag the exposure mix.
        picks.append(exposure_stratified_subsample(in_band, take, int(rng.integers(1 << 31)),
                                                   n_strata=min(n_strata, max(1, take))))

    claims = pd.concat(picks) if picks else pool_claims.iloc[:0]
    out = pd.concat([non_claims, claims]).sort_values(ID_COL)
    return out.reset_index(drop=True)


# --- provenance --------------------------------------------------------------


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash so every track can prove it scored the identical test set."""
    ids = np.ascontiguousarray(df[ID_COL].to_numpy(dtype=np.int64))
    return hashlib.sha256(ids.tobytes()).hexdigest()[:16]


def build_and_cache(cache_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, CleaningReport]:
    """Entry point: load, clean, cache, and persist the cleaning report."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean_path = cache_dir / "freMTPL2_clean.parquet"
    report_path = cache_dir / "cleaning_report.json"

    if clean_path.exists() and report_path.exists():
        df = pd.read_parquet(clean_path)
        report = CleaningReport(**json.loads(report_path.read_text()))
        return df, report

    freq, sev = load_raw(cache_dir)
    df, report = clean(freq, sev)
    df.to_parquet(clean_path, index=False)
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    return df, report
