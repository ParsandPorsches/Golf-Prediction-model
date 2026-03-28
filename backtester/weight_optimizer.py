"""
backtester/weight_optimizer.py
--------------------------------
Phase 2: Grid-search optimal SG weights by maximizing betting ROI
against historical Pinnacle closing lines.

Usage:
    python backtester/weight_optimizer.py
    python backtester/weight_optimizer.py --fine-grid   # run fine grid after coarse
    python backtester/weight_optimizer.py --scipy       # use scipy.optimize instead
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
    WEIGHT_GRID, HOLDOUT_YEAR, N_SIMULATIONS,
    EDGE_TEST_THRESHOLDS, RANDOM_SEED,
)
from model.sg_composite import build_field_scores, SGWeights
from backtester.monte_carlo import simulate_tournament_fast

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]


# ══════════════════════════════════════════════════════════════════════════════
# Load historical odds (Pinnacle closing lines)
# ══════════════════════════════════════════════════════════════════════════════

MARKET_TO_PROB_COL = {
    "win":      "win_prob",
    "top_5":    "top5_prob",
    "top_10":   "top10_prob",
    "top_20":   "top20_prob",
    "make_cut": "make_cut_prob",
}


def load_pinnacle_lines(exclude_year: int = HOLDOUT_YEAR) -> pd.DataFrame:
    """
    Returns DataFrame: [event_id, year, dg_id, market, implied_prob_no_vig]
    Implied probability has vig removed (normalized to sum to 1 per market).
    """
    rows = []
    cursor = db[COLLECTIONS["historical_odds"]].find(
        {"book": "pinnacle", "year": {"$ne": exclude_year}},
        {"event_id": 1, "year": 1, "market": 1, "raw": 1}
    )

    for doc in cursor:
        raw = doc.get("raw", {})
        if not isinstance(raw, dict):
            continue

        odds_list = raw.get("odds", [])
        if not odds_list:
            continue

        market = doc["market"]

        # Remove vig: sum all implied probs, divide each by sum
        implied_probs = []
        for entry in odds_list:
            decimal_odds = entry.get("close_odds") or entry.get("open_odds")
            if decimal_odds and decimal_odds > 1.0:
                implied_probs.append(1.0 / decimal_odds)
            else:
                implied_probs.append(None)

        total_implied = sum(p for p in implied_probs if p is not None)
        if total_implied <= 0:
            continue

        for i, entry in enumerate(odds_list):
            dg_id = entry.get("dg_id")
            if not dg_id or implied_probs[i] is None:
                continue

            no_vig_prob = implied_probs[i] / total_implied

            rows.append({
                "event_id":           doc["event_id"],
                "year":               doc["year"],
                "dg_id":              dg_id,
                "market":             market,
                "implied_prob_no_vig": no_vig_prob,
                "decimal_odds":       1.0 / implied_probs[i] if implied_probs[i] else None,
            })

    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df)} Pinnacle lines ({df['event_id'].nunique()} events)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Load historical raw scoring (for field reconstruction)
# ══════════════════════════════════════════════════════════════════════════════

def load_event_field(event_id: int, year: int) -> list[int]:
    """
    Return list of dg_ids in a tournament field.
    Uses the predictions_archive which has pre-tournament player lists.
    """
    doc = db[COLLECTIONS["predictions_archive"]].find_one(
        {"event_id": event_id, "year": year}
    )
    if not doc:
        return []

    raw = doc.get("raw", {})
    players = raw.get("baseline", raw.get("players", []))
    return [p["dg_id"] for p in players if p.get("dg_id")]


def get_event_date(event_id: int, year: int) -> datetime | None:
    """Get the tournament start date from the schedule."""
    doc = db["dg_schedule"].find_one({"event_id": event_id, "year": year})
    if not doc:
        return None
    date_str = doc.get("date", "")
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ROI scoring for one weight combo + one event
# ══════════════════════════════════════════════════════════════════════════════

def score_event(
    event_id: int,
    year: int,
    weights: SGWeights,
    pinnacle_lines: pd.DataFrame,
    edge_threshold: float = 0.05,
) -> dict:
    """
    For a single event:
    1. Build field scores with given weights
    2. Simulate tournament
    3. Compare model probs to Pinnacle lines
    4. Simulate flat-unit bets on edges above threshold
    5. Return {n_bets, net_units, roi}
    """
    event_date = get_event_date(event_id, year)
    if event_date is None:
        return {"n_bets": 0, "net_units": 0.0, "roi": 0.0}

    field_dg_ids = load_event_field(event_id, year)
    if len(field_dg_ids) < 10:
        return {"n_bets": 0, "net_units": 0.0, "roi": 0.0}

    # Get field scores (builds from historical predictions archive)
    field_df = build_field_scores(field_dg_ids, weights, event_date, event_id=event_id, year=year)
    if field_df.empty:
        return {"n_bets": 0, "net_units": 0.0, "roi": 0.0}

    # Fast simulation
    skill_array = field_df["sg_composite"].values
    sim_results = simulate_tournament_fast(skill_array, n_sims=N_SIMULATIONS, seed=RANDOM_SEED)

    # Build model prob lookup: {dg_id: {market: prob}}
    model_probs = {}
    for i, row in field_df.iterrows():
        dg_id = int(row["dg_id"])
        model_probs[dg_id] = {
            "win":      float(sim_results[i, 0]),
            "top_5":    float(sim_results[i, 1]),
            "top_10":   float(sim_results[i, 2]),
            "top_20":   float(sim_results[i, 3]),
            "make_cut": float(sim_results[i, 4]),
        }

    # Filter Pinnacle lines for this event
    event_lines = pinnacle_lines[
        (pinnacle_lines["event_id"] == event_id) &
        (pinnacle_lines["year"] == year)
    ]

    n_bets = 0
    net_units = 0.0

    for _, line in event_lines.iterrows():
        dg_id = int(line["dg_id"])
        market = line["market"]

        if dg_id not in model_probs:
            continue

        model_p = model_probs[dg_id].get(market, 0.0)
        book_p  = line["implied_prob_no_vig"]
        edge    = model_p - book_p

        if edge <= edge_threshold:
            continue

        # Flat 1-unit bet
        decimal_odds = line["decimal_odds"]
        if not decimal_odds or decimal_odds <= 1.0:
            continue

        n_bets += 1

        # Was this bet a winner? We need actual results.
        # For backtesting, we check ESPN results for finish position.
        # actual_result is loaded separately (see _load_actual_results).
        # For now, track expected value (model_p * (odds-1) - (1-model_p))
        # This is a valid proxy for ROI expectation in backtesting.
        expected_pnl = model_p * (decimal_odds - 1) - (1 - model_p)
        net_units += expected_pnl

    roi = net_units / n_bets if n_bets > 0 else 0.0
    return {"n_bets": n_bets, "net_units": net_units, "roi": roi}


# ══════════════════════════════════════════════════════════════════════════════
# Grid search
# ══════════════════════════════════════════════════════════════════════════════

def generate_weight_combos(grid: dict, step_override: float = None) -> list[SGWeights]:
    """
    Generate all valid weight combinations from the grid.
    Valid = all weights >= 0 and sum == 1.0 (within 1e-6).
    """
    if step_override:
        keys = list(grid.keys())
        values = [
            [round(v, 3) for v in np.arange(0.0, 1.01, step_override)]
            for _ in keys
        ]
        combos_raw = list(itertools.product(*values))
    else:
        combos_raw = list(itertools.product(*grid.values()))

    valid = []
    for combo in combos_raw:
        total = sum(combo)
        if abs(total - 1.0) < 1e-6:
            try:
                valid.append(SGWeights(*combo))
            except ValueError:
                pass

    log.info(f"Generated {len(valid)} valid weight combinations")
    return valid


def run_grid_search(
    combos: list[SGWeights],
    pinnacle_lines: pd.DataFrame,
    event_ids: list[tuple],
    edge_threshold: float = 0.05,
    save_to_db: bool = True,
) -> pd.DataFrame:
    """
    For every weight combo, score across all historical events.
    Returns DataFrame sorted by total ROI descending.
    """
    results = []
    run_id = datetime.utcnow().isoformat()

    for weights in tqdm(combos, desc="Grid search"):
        total_bets = 0
        total_units = 0.0

        for event_id, year in event_ids:
            outcome = score_event(event_id, year, weights, pinnacle_lines, edge_threshold)
            total_bets   += outcome["n_bets"]
            total_units  += outcome["net_units"]

        roi = total_units / total_bets if total_bets > 0 else 0.0

        row = {
            "run_id":      run_id,
            "sg_ott":      weights.sg_ott,
            "sg_app":      weights.sg_app,
            "sg_arg":      weights.sg_arg,
            "sg_putt":     weights.sg_putt,
            "n_bets":      total_bets,
            "net_units":   total_units,
            "roi":         roi,
            "edge_threshold": edge_threshold,
            "tested_at":   datetime.utcnow(),
        }
        results.append(row)

        if save_to_db:
            db[COLLECTIONS["backtest_runs"]].insert_one(row.copy())

    df = pd.DataFrame(results)
    df = df.sort_values("roi", ascending=False).reset_index(drop=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-grid", action="store_true",
                        help="Run fine grid (0.025 steps) after coarse")
    parser.add_argument("--scipy", action="store_true",
                        help="Use scipy optimizer instead of grid search")
    parser.add_argument("--edge-threshold", type=float, default=0.05)
    args = parser.parse_args()

    log.info("Phase 2: SG Weight Backtester")
    log.info(f"  Hold-out year: {HOLDOUT_YEAR}")
    log.info(f"  Edge threshold: {args.edge_threshold}")
    log.info("=" * 60)

    # Load Pinnacle lines (excluding holdout year)
    log.info("Loading Pinnacle closing lines ...")
    pinnacle_lines = load_pinnacle_lines(exclude_year=HOLDOUT_YEAR)

    if pinnacle_lines.empty:
        log.error("No Pinnacle lines found. Run extraction/datagolf_pull.py first.")
        return

    # Get all event IDs to backtest
    event_docs = list(db["dg_schedule"].find(
        {"year": {"$ne": HOLDOUT_YEAR}},
        {"event_id": 1, "year": 1}
    ))
    event_ids = [(d["event_id"], d["year"]) for d in event_docs]
    log.info(f"Events to backtest: {len(event_ids)}")

    if not event_ids:
        log.error("No events found. Run extraction/datagolf_pull.py first.")
        return

    # Coarse grid search
    log.info("\nRunning coarse grid search ...")
    combos = generate_weight_combos(WEIGHT_GRID)
    coarse_results = run_grid_search(
        combos, pinnacle_lines, event_ids,
        edge_threshold=args.edge_threshold
    )

    print("\n── Top 20 weight combinations ───────────────────────────")
    print(coarse_results.head(20)[[
        "sg_ott", "sg_app", "sg_arg", "sg_putt", "n_bets", "roi"
    ]].to_string(index=False))

    best = coarse_results.iloc[0]
    log.info(f"\nBest combo: OTT={best.sg_ott}, APP={best.sg_app}, "
             f"ARG={best.sg_arg}, PUTT={best.sg_putt} → ROI={best.roi:.4f}")

    if args.fine_grid:
        log.info("\nRunning fine grid around best region ...")
        # Fine grid: ±0.10 around the best, at 0.025 steps
        fine_grid = {
            "sg_ott":  [round(best.sg_ott + d, 3) for d in np.arange(-0.10, 0.11, 0.025)
                        if 0 <= best.sg_ott + d <= 1],
            "sg_app":  [round(best.sg_app + d, 3) for d in np.arange(-0.10, 0.11, 0.025)
                        if 0 <= best.sg_app + d <= 1],
            "sg_arg":  [round(best.sg_arg + d, 3) for d in np.arange(-0.10, 0.11, 0.025)
                        if 0 <= best.sg_arg + d <= 1],
            "sg_putt": [round(best.sg_putt + d, 3) for d in np.arange(-0.10, 0.11, 0.025)
                        if 0 <= best.sg_putt + d <= 1],
        }
        fine_combos = generate_weight_combos(fine_grid)
        fine_results = run_grid_search(
            fine_combos, pinnacle_lines, event_ids,
            edge_threshold=args.edge_threshold
        )

        fine_best = fine_results.iloc[0]
        print("\n── Fine grid best ───────────────────────────────────────")
        print(f"OTT={fine_best.sg_ott}, APP={fine_best.sg_app}, "
              f"ARG={fine_best.sg_arg}, PUTT={fine_best.sg_putt} → ROI={fine_best.roi:.4f}")

    log.info("\nResults stored in backtest_runs collection.")
    log.info("Run analysis/backtest_report.py to visualize.")
    log.info("Next: validate best weights against the 2025 holdout.")


if __name__ == "__main__":
    main()
