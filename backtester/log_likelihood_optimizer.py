"""
backtester/log_likelihood_optimizer.py
----------------------------------------
Phase 2 (ESPN variant): Optimize SG weights by maximizing log-likelihood
of actual PGA Tour finish positions given our model's predicted probabilities.

Instead of Pinnacle ROI (requires upgraded DataGolf plan), we score each
weight combo by how well it predicts who actually wins / finishes top-10.

Higher log-likelihood = weights that better explain real outcomes.

Usage:
    python backtester/log_likelihood_optimizer.py
    python backtester/log_likelihood_optimizer.py --fine-grid
"""

import sys
import logging
import argparse
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from pymongo import MongoClient
from tqdm import tqdm

sys.path.insert(0, ".")
from config.settings import (
    MONGODB_URI, DB_NAME, COLLECTIONS,
    WEIGHT_GRID, HOLDOUT_YEAR, N_SIMULATIONS, RANDOM_SEED,
)
from model.sg_composite import SGWeights
from backtester.monte_carlo import simulate_tournament_fast

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

PROB_FLOOR = 1e-4  # avoid log(0)


# =============================================================================
# Data loading
# =============================================================================

def load_espn_results(exclude_year: int = HOLDOUT_YEAR) -> pd.DataFrame:
    """
    Load all ESPN results excluding the holdout year.
    Returns DataFrame: [year, espn_event_id, event_name, player_name,
                        finish, made_cut, withdrew]
    """
    rows = []
    cursor = db[COLLECTIONS["espn_results"]].find(
        {"year": {"$ne": exclude_year}},
        {"year": 1, "espn_event_id": 1, "event_name": 1, "results": 1}
    )
    for doc in cursor:
        for r in doc.get("results", []):
            rows.append({
                "year":          doc["year"],
                "espn_event_id": doc["espn_event_id"],
                "event_name":    doc["event_name"],
                "player_name":   r.get("player_name"),
                "finish":        r.get("finish"),
                "made_cut":      r.get("made_cut", False),
                "withdrew":      r.get("withdrew", False),
            })

    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df)} player-results across "
             f"{df['espn_event_id'].nunique()} events")
    return df


def load_skill_ratings() -> pd.DataFrame:
    """Load current skill ratings with name normalization."""
    docs = list(db[COLLECTIONS["skill_ratings"]].find(
        {}, {"dg_id": 1, "player_name": 1, "sg_ott": 1, "sg_app": 1, "sg_arg": 1, "sg_putt": 1}
    ))
    df = pd.DataFrame(docs).dropna(subset=["sg_ott", "sg_app", "sg_arg", "sg_putt"])
    df["name_norm"] = df["player_name"].apply(_norm)
    return df


def _norm(name: str) -> str:
    name = name.strip().lower().replace(".", "").replace("-", " ")
    if "," in name:
        parts = name.split(",", 1)
        return parts[1].strip() + " " + parts[0].strip()
    return name


def build_name_lookup(skill_df: pd.DataFrame) -> dict:
    """Build name -> row lookup with last-name fallback."""
    exact = {r["name_norm"]: r for r in skill_df.to_dict("records")}
    last_map = {}
    for norm, row in exact.items():
        last = norm.split()[-1]
        last_map.setdefault(last, []).append((norm, row))
    return exact, last_map


def match_player(name: str, exact: dict, last_map: dict, median_sg: float) -> float | None:
    """Return pre-computed sg_composite for a player name, or None if unmatched."""
    norm = _norm(name)
    row = exact.get(norm)
    if not row:
        last = norm.split()[-1]
        candidates = last_map.get(last, [])
        if len(candidates) == 1:
            row = candidates[0][1]
    return row  # may be None


# =============================================================================
# Log-likelihood scoring
# =============================================================================

def score_event_ll(
    event_results: pd.DataFrame,
    weights: SGWeights,
    exact: dict,
    last_map: dict,
    median_sg: float,
    n_sims: int = 2000,
) -> dict:
    """
    Score a single event using log-likelihood.

    For each player in the actual results:
      - win:     log(win_prob)     if finish == 1
      - top10:   log(top10_prob)   if finish <= 10
      - cut:     log(make_cut)     if made_cut
      - missed:  log(1-make_cut)   if not made_cut and not withdrew

    Returns {ll_win, ll_top10, ll_cut, n_players}
    """
    # Build field from skill ratings
    field = []
    for _, row in event_results.iterrows():
        sr = match_player(row["player_name"], exact, last_map, median_sg)
        if sr is not None:
            sg = (weights.sg_ott  * sr["sg_ott"]  +
                  weights.sg_app  * sr["sg_app"]  +
                  weights.sg_arg  * sr["sg_arg"]  +
                  weights.sg_putt * sr["sg_putt"])
        else:
            sg = median_sg
        field.append({"player_name": row["player_name"], "sg": sg})

    if len(field) < 10:
        return {"ll_win": 0.0, "ll_top10": 0.0, "ll_cut": 0.0, "n_players": 0}

    skill_array = np.array([p["sg"] for p in field])
    sim = simulate_tournament_fast(skill_array, n_sims=n_sims, seed=RANDOM_SEED)
    # sim columns: [win, top5, top10, top20, make_cut]

    name_to_idx = {p["player_name"]: i for i, p in enumerate(field)}

    ll_win = ll_top10 = ll_cut = 0.0
    n = 0

    for _, row in event_results.iterrows():
        idx = name_to_idx.get(row["player_name"])
        if idx is None:
            continue

        win_p    = max(float(sim[idx, 0]), PROB_FLOOR)
        top10_p  = max(float(sim[idx, 2]), PROB_FLOOR)
        cut_p    = max(float(sim[idx, 4]), PROB_FLOOR)

        finish   = row.get("finish")
        made_cut = row.get("made_cut", False)
        withdrew = row.get("withdrew", False)

        if finish == 1:
            ll_win += np.log(win_p)
        if finish and finish <= 10:
            ll_top10 += np.log(top10_p)
        if not withdrew:
            if made_cut:
                ll_cut += np.log(cut_p)
            else:
                ll_cut += np.log(max(1 - cut_p, PROB_FLOOR))

        n += 1

    return {"ll_win": ll_win, "ll_top10": ll_top10, "ll_cut": ll_cut, "n_players": n}


# =============================================================================
# Grid search
# =============================================================================

def generate_weight_combos(grid: dict, step_override: float = None) -> list:
    if step_override:
        values = [
            [round(v, 3) for v in np.arange(0.0, 1.01, step_override)]
            for _ in grid
        ]
        combos_raw = list(itertools.product(*values))
    else:
        combos_raw = list(itertools.product(*grid.values()))

    valid = []
    for combo in combos_raw:
        if abs(sum(combo) - 1.0) < 1e-6 and all(v >= 0 for v in combo):
            try:
                valid.append(SGWeights(*combo))
            except ValueError:
                pass
    log.info(f"Generated {len(valid)} valid weight combinations")
    return valid


def run_grid_search(
    combos: list,
    espn_results: pd.DataFrame,
    skill_df: pd.DataFrame,
    save_to_db: bool = True,
) -> pd.DataFrame:
    exact, last_map = build_name_lookup(skill_df)
    median_sg = (
        0.20 * skill_df["sg_ott"]  +
        0.40 * skill_df["sg_app"]  +
        0.25 * skill_df["sg_arg"]  +
        0.15 * skill_df["sg_putt"]
    ).median()

    events = espn_results.groupby("espn_event_id")
    event_list = list(events)
    run_id = datetime.utcnow().isoformat()
    results = []

    for weights in tqdm(combos, desc="Grid search"):
        total_ll_win = total_ll_top10 = total_ll_cut = 0.0
        total_players = 0

        for _, event_df in event_list:
            out = score_event_ll(event_df, weights, exact, last_map, median_sg)
            total_ll_win    += out["ll_win"]
            total_ll_top10  += out["ll_top10"]
            total_ll_cut    += out["ll_cut"]
            total_players   += out["n_players"]

        # Combined score: equal weight on win + top10 + cut accuracy
        combined_ll = total_ll_win + total_ll_top10 + total_ll_cut

        row = {
            "run_id":       run_id,
            "sg_ott":       weights.sg_ott,
            "sg_app":       weights.sg_app,
            "sg_arg":       weights.sg_arg,
            "sg_putt":      weights.sg_putt,
            "ll_win":       total_ll_win,
            "ll_top10":     total_ll_top10,
            "ll_cut":       total_ll_cut,
            "ll_combined":  combined_ll,
            "n_players":    total_players,
            "optimizer":    "log_likelihood_espn",
            "tested_at":    datetime.utcnow(),
        }
        results.append(row)

        if save_to_db:
            # Store with roi=ll_combined so pre_tournament.py load_best_weights() works
            db_row = {**row, "roi": combined_ll / max(total_players, 1)}
            db[COLLECTIONS["backtest_runs"]].insert_one(db_row)

    df = pd.DataFrame(results).sort_values("ll_combined", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-grid", action="store_true",
                        help="Run fine grid (0.025 steps) around best combo")
    args = parser.parse_args()

    log.info("Phase 2 (ESPN): SG Weight Optimizer via Log-Likelihood")
    log.info(f"  Hold-out year: {HOLDOUT_YEAR}")
    log.info("=" * 60)

    log.info("Loading ESPN results...")
    espn_df = load_espn_results(exclude_year=HOLDOUT_YEAR)

    if espn_df.empty:
        log.error("No ESPN results found. Run: python extraction/espn_results.py")
        return

    log.info("Loading skill ratings...")
    skill_df = load_skill_ratings()
    log.info(f"  {len(skill_df)} players with skill ratings")

    log.info("\nRunning coarse grid search...")
    combos = generate_weight_combos(WEIGHT_GRID)
    results = run_grid_search(combos, espn_df, skill_df)

    best = results.iloc[0]
    print(f"\n{'='*60}")
    print("  TOP 20 WEIGHT COMBINATIONS")
    print(f"{'='*60}")
    print(results.head(20)[[
        "sg_ott", "sg_app", "sg_arg", "sg_putt", "ll_combined", "n_players"
    ]].to_string(index=False))

    print(f"\n  Best: OTT={best.sg_ott}, APP={best.sg_app}, "
          f"ARG={best.sg_arg}, PUTT={best.sg_putt}")
    print(f"  Combined LL={best.ll_combined:.1f}  "
          f"(win={best.ll_win:.1f}, top10={best.ll_top10:.1f}, cut={best.ll_cut:.1f})")

    if args.fine_grid:
        log.info("\nRunning fine grid around best region...")
        fine_grid = {
            "sg_ott":  [round(best.sg_ott  + d, 3) for d in np.arange(-0.10, 0.11, 0.025)
                        if 0 <= best.sg_ott  + d <= 1],
            "sg_app":  [round(best.sg_app  + d, 3) for d in np.arange(-0.10, 0.11, 0.025)
                        if 0 <= best.sg_app  + d <= 1],
            "sg_arg":  [round(best.sg_arg  + d, 3) for d in np.arange(-0.10, 0.11, 0.025)
                        if 0 <= best.sg_arg  + d <= 1],
            "sg_putt": [round(best.sg_putt + d, 3) for d in np.arange(-0.10, 0.11, 0.025)
                        if 0 <= best.sg_putt + d <= 1],
        }
        fine_combos = generate_weight_combos(fine_grid)
        fine_results = run_grid_search(fine_combos, espn_df, skill_df)
        fine_best = fine_results.iloc[0]
        print(f"\n  Fine grid best: OTT={fine_best.sg_ott}, APP={fine_best.sg_app}, "
              f"ARG={fine_best.sg_arg}, PUTT={fine_best.sg_putt} "
              f"LL={fine_best.ll_combined:.1f}")

    log.info("\nWeights saved to backtest_runs collection.")
    log.info("pre_tournament.py will now use these weights automatically.")


if __name__ == "__main__":
    main()
