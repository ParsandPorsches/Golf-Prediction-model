"""
analysis/kelly_sizing.py
-------------------------
Kelly Criterion bet sizing for confirmed value bets.
Mirrors the trading journal structure in bet_log collection.

Usage:
    python analysis/kelly_sizing.py --bankroll 1000
    python analysis/kelly_sizing.py --bankroll 1000 --event_id 28 --year 2025
"""

import sys
import logging
import argparse
from datetime import datetime

import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import MONGODB_URI, DB_NAME, COLLECTIONS, KELLY_FRACTION, MAX_BET_PCT

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]


def full_kelly(model_prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction. Usually too aggressive — use fractional."""
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    return max(0.0, (b * model_prob - (1 - model_prob)) / b)


def fractional_kelly(model_prob: float, decimal_odds: float,
                     fraction: float = KELLY_FRACTION) -> float:
    """Fractional Kelly (default 25%). Much more conservative."""
    return full_kelly(model_prob, decimal_odds) * fraction


def size_bets(bankroll: float, event_id: int = None, year: int = None) -> pd.DataFrame:
    """
    Load pending value bets from bet_log and compute stake recommendations.
    """
    query = {"outcome": "pending"}
    if event_id:
        query["event_id"] = event_id
    if year:
        query["year"] = year

    docs = list(db[COLLECTIONS["bet_log"]].find(query))
    if not docs:
        log.info("No pending bets in bet_log.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)

    # Kelly sizing
    df["full_kelly_pct"]      = df.apply(
        lambda r: full_kelly(r["model_prob"], r["decimal_odds"]), axis=1
    )
    df["frac_kelly_pct"]      = df["full_kelly_pct"] * KELLY_FRACTION
    df["capped_kelly_pct"]    = df["frac_kelly_pct"].clip(upper=MAX_BET_PCT)
    df["recommended_stake"]   = (df["capped_kelly_pct"] * bankroll).round(2)
    df["expected_profit"]     = df["recommended_stake"] * (
        df["model_prob"] * (df["decimal_odds"] - 1) - (1 - df["model_prob"])
    )

    # Format output
    display_cols = [
        "player_name", "market", "book",
        "model_prob", "decimal_odds", "edge",
        "frac_kelly_pct", "capped_kelly_pct", "recommended_stake", "expected_profit"
    ]
    return df[[c for c in display_cols if c in df.columns]].sort_values(
        "expected_profit", ascending=False
    ).reset_index(drop=True)


def update_bet_outcome(event_id: int, year: int, player_name: str,
                       market: str, won: bool, decimal_odds: float, stake: float):
    """
    Update a bet in bet_log after the tournament completes.
    won: True if bet won, False if lost.
    """
    pnl = stake * (decimal_odds - 1) if won else -stake

    fk = {
        "event_id": event_id,
        "year":     year,
        "market":   market,
    }
    # Try to match by player name
    doc = db[COLLECTIONS["bet_log"]].find_one({**fk, "player_name": player_name})
    if not doc:
        log.warning(f"Bet not found: {player_name} / {market}")
        return

    db[COLLECTIONS["bet_log"]].update_one(
        {"_id": doc["_id"]},
        {"$set": {
            "outcome":      "won" if won else "lost",
            "actual_stake": stake,
            "pnl":          pnl,
            "settled_at":   datetime.utcnow(),
        }}
    )
    log.info(f"Settled: {player_name} / {market} → {'WON' if won else 'LOST'} ${abs(pnl):.2f}")


def print_bankroll_summary(bankroll: float, event_id: int = None, year: int = None):
    df = size_bets(bankroll, event_id, year)
    if df.empty:
        print("No pending bets to size.")
        return

    print(f"\n{'═'*75}")
    print(f"Kelly Bet Sizing  |  Bankroll: ${bankroll:,.2f}  |  "
          f"Kelly fraction: {KELLY_FRACTION*100:.0f}%  |  "
          f"Max per bet: {MAX_BET_PCT*100:.0f}%")
    print(f"{'═'*75}")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nTotal recommended exposure: ${df['recommended_stake'].sum():,.2f} "
          f"({df['recommended_stake'].sum()/bankroll*100:.1f}% of bankroll)")
    print(f"Total expected profit: ${df['expected_profit'].sum():,.2f}")

    # P&L history
    settled = list(db[COLLECTIONS["bet_log"]].find({"outcome": {"$in": ["won", "lost"]}}))
    if settled:
        settled_df = pd.DataFrame(settled)
        total_pnl   = settled_df["pnl"].sum()
        n_won       = (settled_df["outcome"] == "won").sum()
        n_lost      = (settled_df["outcome"] == "lost").sum()
        win_rate    = n_won / len(settled_df) if len(settled_df) > 0 else 0
        print(f"\nHistorical P&L: ${total_pnl:+,.2f}  |  "
              f"Record: {n_won}W / {n_lost}L  |  "
              f"Win rate: {win_rate*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bankroll", type=float, required=True)
    parser.add_argument("--event_id", type=int, default=None)
    parser.add_argument("--year",     type=int, default=datetime.utcnow().year)
    args = parser.parse_args()

    print_bankroll_summary(args.bankroll, args.event_id, args.year)
