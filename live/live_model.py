"""
live/live_model.py
------------------
Fetches DataGolf's live in-play predictions and runs a value scan
against current bookmaker odds using real-time win/top10 probabilities.

Usage:
    python live/live_model.py --event_id 20              # fetch + value scan
    python live/live_model.py --event_id 20 --fetch-only # just store predictions
    python live/live_model.py --event_id 20 --no-discord
"""

import sys
import logging
import argparse
from datetime import datetime

import requests
import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import (
    MONGODB_URI, DB_NAME, COLLECTIONS,
    DATAGOLF_API_KEY, EDGE_THRESHOLDS,
)
from live.value_detector import (
    load_live_odds, detect_value, kelly_fraction, decimal_to_american,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

DG_INPLAY_URL = "https://feeds.datagolf.com/preds/in-play"
COLLECTION    = "live_model_predictions"


# =============================================================================
# Fetch + store
# =============================================================================

def fetch_live_predictions(event_id: int) -> pd.DataFrame:
    """Pull DataGolf in-play predictions and upsert into MongoDB."""
    try:
        resp = requests.get(DG_INPLAY_URL, params={
            "tour":        "pga",
            "dead_heat":   "no",
            "odds_format": "percent",
            "key":         DATAGOLF_API_KEY,
        }, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.error(f"Failed to fetch live predictions: {exc}")
        return pd.DataFrame()

    players  = data.get("data", [])
    info     = data.get("info", {})
    round_no = info.get("current_round")
    updated  = info.get("last_update")
    event_name = info.get("event_name", "")

    log.info(f"Live data: {event_name} | Round {round_no} | Updated {updated}")
    log.info(f"Players: {len(players)}")

    rows = []
    for p in players:
        rows.append({
            "event_id":      event_id,
            "year":          datetime.now().year,
            "dg_id":         p.get("dg_id"),
            "player_name":   p.get("player_name"),
            "current_pos":   p.get("current_pos"),
            "current_score": p.get("current_score"),
            "round":         p.get("round"),
            "thru":          p.get("thru"),
            "today":         p.get("today"),
            "R1":            p.get("R1"),
            "R2":            p.get("R2"),
            "R3":            p.get("R3"),
            "R4":            p.get("R4"),
            "win_prob":      p.get("win", 0) or 0,
            "top5_prob":     p.get("top_5", 0) or 0,
            "top10_prob":    p.get("top_10", 0) or 0,
            "top20_prob":    p.get("top_20", 0) or 0,
            "make_cut_prob": p.get("make_cut", 0) or 0,
            "fetched_at":    datetime.utcnow(),
            "last_update":   updated,
            "current_round": round_no,
        })

    if not rows:
        log.warning("No player data returned.")
        return pd.DataFrame()

    # Upsert each player
    for row in rows:
        fk = {"event_id": event_id, "dg_id": row["dg_id"]}
        db[COLLECTION].update_one(fk, {"$set": row}, upsert=True)

    log.info(f"Stored {len(rows)} live predictions in '{COLLECTION}'")
    return pd.DataFrame(rows)


def load_stored_live_predictions(event_id: int) -> pd.DataFrame:
    """Load most recently stored live predictions from MongoDB."""
    docs = list(db[COLLECTION].find({"event_id": event_id}, {"_id": 0}))
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame(docs)


# =============================================================================
# Live leaderboard print
# =============================================================================

def print_leaderboard(preds: pd.DataFrame, top_n: int = 20):
    preds = preds.copy()

    # Normalise name to First Last
    def fmt_name(name):
        if isinstance(name, str) and "," in name:
            last, first = name.split(",", 1)
            return f"{first.strip()} {last.strip()}"
        return str(name)

    preds["display_name"] = preds["player_name"].apply(fmt_name)
    preds = preds.sort_values("win_prob", ascending=False).reset_index(drop=True)

    round_no = preds["current_round"].iloc[0] if "current_round" in preds.columns else "?"
    updated  = preds["last_update"].iloc[0]   if "last_update"   in preds.columns else "?"

    print(f"\n{'='*72}")
    print(f"  LIVE LEADERBOARD  |  Round {round_no}  |  Updated {updated}")
    print(f"{'='*72}")
    print(f"{'PLAYER':<26} {'POS':<6} {'SCORE':<7} {'THRU':<6} {'WIN%':>5} {'TOP10%':>7}")
    print(f"{'-'*72}")

    for _, row in preds.head(top_n).iterrows():
        thru  = str(row.get("thru", "")) if row.get("thru") else "F"
        score = row.get("current_score", 0)
        score_str = f"{score:+d}" if isinstance(score, (int, float)) else str(score)
        print(
            f"  {row['display_name']:<24}"
            f"  {str(row.get('current_pos','')):<5}"
            f"  {score_str:<6}"
            f"  {thru:<5}"
            f"  {row['win_prob']*100:>4.1f}%"
            f"  {row['top10_prob']*100:>6.1f}%"
        )
    print(f"{'-'*72}")


# =============================================================================
# Live value scan
# =============================================================================

def run_live_value_scan(event_id: int, year: int, send_discord: bool = True):
    """Compare live DG predictions against current bookmaker odds."""

    # Fetch fresh live predictions
    preds = fetch_live_predictions(event_id)
    if preds.empty:
        log.error("No live predictions available.")
        return

    # Print leaderboard
    print_leaderboard(preds)

    # Load live odds from MongoDB (loaded via odds_scraper)
    live_odds = load_live_odds(event_id)
    if live_odds.empty:
        log.error("No live odds found. Run live/odds_scraper.py --csv first.")
        return

    # Run value detection using live probabilities
    value_bets = detect_value(preds, live_odds, thresholds=EDGE_THRESHOLDS)

    if value_bets.empty:
        log.info("No value bets found above threshold.")
        return

    # Add Kelly sizing
    value_bets["kelly_pct"] = value_bets.apply(
        lambda r: kelly_fraction(r["model_prob"], r["decimal_odds"]), axis=1
    )

    # Flag source as live
    value_bets["source"] = "live"

    print(f"\n{'='*75}")
    print(f"  LIVE VALUE BETS --- Event {event_id} / Round {preds['current_round'].iloc[0]}")
    print(f"{'='*75}")

    for _, bet in value_bets.iterrows():
        print(
            f"\n  {bet['player_name']:<28} | {bet['market'].upper():<10}"
            f"\n  Model: {bet['model_prob']*100:.1f}%  |  "
            f"Book: {bet['book_prob']*100:.1f}%  |  "
            f"Edge: +{bet['edge']*100:.1f}%"
            f"\n  Odds: {bet['decimal_odds']:.2f} ({bet['american_odds']})  |  "
            f"Fair: {bet['fair_odds']:.2f}  |  "
            f"Kelly: {bet['kelly_pct']*100:.1f}% bankroll"
        )

    # Upsert to bet_log with source=live
    for _, bet in value_bets.iterrows():
        fk  = {"event_id": event_id, "dg_id": bet["dg_id"],
               "market": bet["market"], "book": bet["book"], "source": "live"}
        doc = {**fk, **bet.to_dict(), "year": year, "outcome": "pending", "pnl": None}
        db[COLLECTIONS["bet_log"]].update_one(fk, {"$set": doc}, upsert=True)

    log.info(f"{len(value_bets)} live value bet(s) saved to bet_log.")

    if send_discord:
        try:
            from live.discord_alerts import send_value_bets_alert
            send_value_bets_alert(value_bets, event_id, year)
        except Exception:
            pass


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id",   type=int, required=True)
    parser.add_argument("--year",       type=int, default=datetime.utcnow().year)
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch and store predictions, skip value scan")
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args()

    if args.fetch_only:
        fetch_live_predictions(args.event_id)
    else:
        run_live_value_scan(args.event_id, args.year, send_discord=not args.no_discord)


if __name__ == "__main__":
    main()
