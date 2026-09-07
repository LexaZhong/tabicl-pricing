"""In-context-learning context construction for TabICL stage 1.

TabICL has no training loop: the "context" IS the model. With a 5% claim rate, a random
50k context carries only ~2.5k positives, and which 2.5k you happen to draw moves the
predictions a lot -- that is the instability the team observed.

Two levers here:
  * claim-enriched sampling: take every available positive, then fill with negatives to a
    target ratio, so the context is dense in the signal that matters;
  * prior correction: undersampling negatives inflates p_hat, so shift it back to the
    population base rate analytically. Without this the model is badly miscalibrated and
    the pure-premium level is wrong even when the ranking is fine.

Stage 2 needs none of this: the whole claims subset (~25k rows) fits in one context, so it
has zero sampling variance. That -- not the two-stage framing by itself -- is most of the
stability gain seen in the team's run 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-12


def enriched_context_indices(
    y_binary: np.ndarray,
    context_size: int,
    seed: int,
    neg_per_pos: float = 3.0,
) -> tuple[np.ndarray, float]:
    """Pick context rows dense in positives; return (indices, sampling_odds_ratio).

    `sampling_odds_ratio` is (ctx_pos_rate/ctx_neg_rate) / (pop_pos_rate/pop_neg_rate),
    i.e. how much the sampling inflated the odds. Feed it to `correct_prior`.
    """
    y = np.asarray(y_binary, dtype=int)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        idx = rng.choice(len(y), size=min(context_size, len(y)), replace=False)
        return np.sort(idx), 1.0

    # Budget positives first, then fill with negatives at the requested ratio.
    n_pos = int(min(len(pos_idx), context_size / (1.0 + neg_per_pos)))
    n_pos = max(n_pos, 1)
    n_neg = int(min(len(neg_idx), context_size - n_pos))
    n_neg = max(n_neg, 1)

    take_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    take_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = np.sort(np.concatenate([take_pos, take_neg]))

    pop_odds = len(pos_idx) / len(neg_idx)
    ctx_odds = n_pos / n_neg
    return idx, float(ctx_odds / max(pop_odds, _EPS))


def random_context_indices(
    y_binary: np.ndarray, context_size: int, seed: int
) -> tuple[np.ndarray, float]:
    """Plain random context -- the ablation arm. Odds ratio is 1 by construction."""
    rng = np.random.default_rng(seed)
    n = len(y_binary)
    idx = rng.choice(n, size=min(context_size, n), replace=False)
    return np.sort(idx), 1.0


def stratified_context_indices(
    y_binary: np.ndarray, exposure: np.ndarray, context_size: int, seed: int, n_strata: int = 10
) -> tuple[np.ndarray, float]:
    """Exposure-stratified random context (what the team's original batching approximated)."""
    rng = np.random.default_rng(seed)
    n = len(y_binary)
    if context_size >= n:
        return np.arange(n), 1.0
    strata = pd.qcut(np.asarray(exposure, dtype=float), q=n_strata, labels=False, duplicates="drop")
    positions = np.arange(n)
    picks = []
    for s in np.unique(strata):
        in_s = positions[strata == s]
        take = min(int(round(context_size * len(in_s) / n)), len(in_s))
        if take > 0:
            picks.append(rng.choice(in_s, size=take, replace=False))
    idx = np.sort(np.concatenate(picks)) if picks else np.array([], dtype=int)
    return idx, 1.0


def correct_prior(p_hat: np.ndarray, sampling_odds_ratio: float) -> np.ndarray:
    """Undo the effect of negative undersampling on predicted probabilities.

    If the context over-represents positives by an odds factor r, then
        odds_ctx = r * odds_pop  =>  odds_pop = odds_ctx / r
    which in probability terms is the expression below. r == 1 is a no-op.
    """
    r = float(sampling_odds_ratio)
    if not np.isfinite(r) or abs(r - 1.0) < 1e-12:
        return np.asarray(p_hat, dtype=float)
    p = np.clip(np.asarray(p_hat, dtype=float), _EPS, 1 - _EPS)
    odds = p / (1.0 - p) / r
    return odds / (1.0 + odds)


SAMPLERS = {
    "enriched": enriched_context_indices,
    "random": random_context_indices,
}
