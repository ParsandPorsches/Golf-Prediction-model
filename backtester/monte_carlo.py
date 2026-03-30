"""
backtester/monte_carlo.py
--------------------------
Monte Carlo simulation engine.
Simulates N tournaments, returning win/top5/top10/top20/make_cut probabilities.
"""

import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config.settings import N_SIMULATIONS, RANDOM_SEED

log = logging.getLogger(__name__)

ROUND_SIGMA = 3.5
LUCK_SIGMA  = 2.0
TOTAL_SIGMA = np.sqrt(ROUND_SIGMA**2 + LUCK_SIGMA**2)


def get_cut_model():
    """Lazy import to avoid circular deps."""
    from backtester.cut_simulator import get_cut_model as _get
    return _get()


def simulate_tournament(
    field_scores: pd.DataFrame,
    n_sims: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
    apply_cut: bool = True,
    cut_top_n: int = 70,
) -> pd.DataFrame:
    """
    Run N Monte Carlo tournament simulations.

    Parameters
    ----------
    field_scores : DataFrame with columns [dg_id, player_name, sg_composite]
                   Optionally: [dg_win_prob, dg_make_cut_prob] from DataGolf
    n_sims       : Number of simulations
    seed         : Random seed
    apply_cut    : Whether to simulate the 36-hole cut
    cut_top_n    : Players who make the cut

    Returns
    -------
    DataFrame: dg_id, player_name, sg_composite, win_prob, top5_prob,
               top10_prob, top20_prob, make_cut_prob
    """
    rng = np.random.default_rng(seed)

    n_players  = len(field_scores)
    dg_ids     = field_scores["dg_id"].values
    skill      = field_scores["sg_composite"].values
    mean_score = -skill  # higher skill = lower (better) score

    wins      = np.zeros(n_players, dtype=np.int32)
    top5      = np.zeros(n_players, dtype=np.int32)
    top10     = np.zeros(n_players, dtype=np.int32)
    top20     = np.zeros(n_players, dtype=np.int32)
    made_cuts = np.zeros(n_players, dtype=np.int32)

    # Generate all random scores at once: (n_sims, n_players, 4 rounds)
    noise = rng.normal(0, TOTAL_SIGMA, size=(n_sims, n_players, 4))

    for sim_idx in range(n_sims):
        round_scores = noise[sim_idx] + mean_score[:, np.newaxis]

        r36 = round_scores[:, 0] + round_scores[:, 1]

        if apply_cut:
            cut_idx        = min(cut_top_n - 1, n_players - 1)
            cut_line_score = np.sort(r36)[cut_idx]
            survived       = r36 <= cut_line_score
        else:
            survived = np.ones(n_players, dtype=bool)

        r72 = r36 + round_scores[:, 2] + round_scores[:, 3]
        r72[~survived] = np.inf

        ranks = np.argsort(r72, kind="stable")

        for pos, idx in enumerate(ranks):
            if r72[idx] == np.inf:
                break
            finish = pos + 1
            if survived[idx]:
                made_cuts[idx] += 1
            if finish == 1:
                wins[idx] += 1
            if finish <= 5:
                top5[idx] += 1
            if finish <= 10:
                top10[idx] += 1
            if finish <= 20:
                top20[idx] += 1

    results = pd.DataFrame({
        "dg_id":         dg_ids,
        "win_prob":      wins      / n_sims,
        "top5_prob":     top5      / n_sims,
        "top10_prob":    top10     / n_sims,
        "top20_prob":    top20     / n_sims,
        "make_cut_prob": made_cuts / n_sims,
    })

    # Merge player names and sg_composite back in
    merge_cols = ["dg_id", "player_name", "sg_composite"]
    # Also carry through DataGolf's own probs if present
    for col in ["dg_win_prob", "dg_top5_prob", "dg_top10_prob",
                "dg_top20_prob", "dg_make_cut_prob"]:
        if col in field_scores.columns:
            merge_cols.append(col)

    results = results.merge(
        field_scores[[c for c in merge_cols if c in field_scores.columns]],
        on="dg_id", how="left"
    )

    return results.sort_values("win_prob", ascending=False).reset_index(drop=True)


def simulate_tournament_fast(
    skill_array: np.ndarray,
    n_sims: int = N_SIMULATIONS,
    seed: int = RANDOM_SEED,
    apply_cut: bool = True,
    cut_top_n: int = 70,
) -> np.ndarray:
    """
    Vectorized Monte Carlo simulation for backtesting (no DataFrame overhead).

    Parameters
    ----------
    skill_array : 1D array of sg_composite scores (one per player)
    n_sims      : Number of simulations
    seed        : Random seed
    apply_cut   : Whether to simulate the 36-hole cut
    cut_top_n   : Players who survive the cut

    Returns
    -------
    np.ndarray of shape (n_players, 5):
        columns → [win_prob, top5_prob, top10_prob, top20_prob, make_cut_prob]
    """
    rng = np.random.default_rng(seed)
    n_players = len(skill_array)
    mean_score = -skill_array  # higher skill = lower score

    # All noise at once: (n_sims, n_players, 4)
    noise = rng.normal(0, TOTAL_SIGMA, size=(n_sims, n_players, 4))
    # round_scores: (n_sims, n_players, 4)
    round_scores = noise + mean_score[np.newaxis, :, np.newaxis]

    # 36-hole scores: (n_sims, n_players)
    r36 = round_scores[:, :, 0] + round_scores[:, :, 1]

    if apply_cut:
        cut_top_n_actual = min(cut_top_n, n_players)
        cut_lines = np.sort(r36, axis=1)[:, cut_top_n_actual - 1]  # (n_sims,)
        survived = r36 <= cut_lines[:, np.newaxis]                  # (n_sims, n_players)
    else:
        survived = np.ones((n_sims, n_players), dtype=bool)

    # 72-hole scores; missed-cut players set to inf
    r72 = r36 + round_scores[:, :, 2] + round_scores[:, :, 3]
    r72[~survived] = np.inf

    # Finish positions via inverse argsort: positions[s, p] = 0-based rank of player p in sim s
    order = np.argsort(r72, axis=1, kind="stable")           # (n_sims, n_players)
    positions = np.empty_like(order)
    positions[np.arange(n_sims)[:, None], order] = np.arange(n_players)[None, :]

    finished = r72 != np.inf  # (n_sims, n_players)

    return np.column_stack([
        ((positions == 0) & finished).sum(axis=0) / n_sims,   # win
        ((positions <  5) & finished).sum(axis=0) / n_sims,   # top5
        ((positions < 10) & finished).sum(axis=0) / n_sims,   # top10
        ((positions < 20) & finished).sum(axis=0) / n_sims,   # top20
        survived.sum(axis=0) / n_sims,                         # make_cut
    ])
