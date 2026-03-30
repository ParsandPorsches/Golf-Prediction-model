"""
live/odds_scraper.py
---------------------
Loads odds into MongoDB from either The Odds API (majors) or a CSV file (regular events).

Usage:
    # Majors only (uses The Odds API):
    python live/odds_scraper.py --event_id 28 --sport_key golf_masters_tournament_winner

    # Regular events (CSV):
    python live/odds_scraper.py --event_id 20 --csv outputs/2026_event-name/odds.csv

CSV format (player_name, market, book, decimal_odds):
    player_name, market, book, decimal_odds
    Min Woo Lee, win, draftkings, 18.00
    Min Woo Lee, top_10, draftkings, 3.50
    ...

Valid markets: win, top_5, top_10, top_20, make_cut
"""

import sys
import csv
import logging
import argparse
from datetime import datetime

import requests
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import (
    MONGODB_URI, DB_NAME, COLLECTIONS,
    ODDS_API_KEY, ODDS_API_BASE,
    ODDS_MARKETS,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

VALID_MARKETS = set(ODDS_MARKETS)


# =============================================================================
# Snapshot history (line movement tracking)
# =============================================================================

def _save_snapshot(event_id: int, market: str, book: str, odds_list: list):
    """
    Append a timestamped odds snapshot to odds_snapshots.
    The first snapshot for a given (event_id, market, book) is tagged as
    the opening line. Subsequent snapshots are tagged as updates.
    """
    col = db[COLLECTIONS["odds_snapshots"]]
    fk = {"event_id": event_id, "market": market, "book": book}

    is_opening = col.count_documents(fk, limit=1) == 0

    col.insert_one({
        **fk,
        "odds":        odds_list,
        "snapshot_at": datetime.utcnow(),
        "is_opening":  is_opening,
    })

    tag = "opening" if is_opening else "update"
    log.debug(f"  Snapshot ({tag}): {market}/{book} — {len(odds_list)} players")


# =============================================================================
# CSV import
# =============================================================================

def load_from_csv(csv_path: str, event_id: int) -> int:
    """
    Read odds from a CSV and upsert into live_odds collection.

    Groups rows by (market, book) and stores as a single odds document per group,
    matching the same schema used by the API path.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = row.get("player_name", "").strip()
            market = row.get("market", "").strip().lower()
            book   = row.get("book", "").strip().lower()
            odds   = row.get("decimal_odds", "").strip()

            if not player or not market or not odds:
                continue
            if market not in VALID_MARKETS:
                log.warning(f"  Skipping unknown market '{market}' for {player}")
                continue
            try:
                decimal_odds = float(odds)
            except ValueError:
                continue
            if decimal_odds <= 1.0:
                continue

            rows.append({
                "player_name":  player,
                "market":       market,
                "book":         book or "manual",
                "decimal_odds": decimal_odds,
            })

    if not rows:
        log.error(f"No valid rows found in {csv_path}")
        return 0

    # Match dg_ids from player map by name
    # player_map stores names as "Last, First" — build both-direction lookup
    player_map = {}
    for doc in db[COLLECTIONS["player_map"]].find({}, {"dg_name": 1, "dg_id": 1}):
        dg_name = doc["dg_name"]
        dg_id   = doc["dg_id"]
        player_map[dg_name] = dg_id                    # "Woodland, Gary"
        if ", " in dg_name:
            last, first = dg_name.split(", ", 1)
            player_map[f"{first} {last}"] = dg_id      # "Gary Woodland"

    # Group into (market, book) buckets
    buckets: dict[tuple, list] = {}
    for row in rows:
        key = (row["market"], row["book"])
        buckets.setdefault(key, []).append(row)

    stored = 0
    for (market, book), entries in buckets.items():
        odds_list = []
        for e in entries:
            dg_id = player_map.get(e["player_name"])
            odds_list.append({
                "player_name":  e["player_name"],
                "decimal_odds": e["decimal_odds"],
                "dg_id":        dg_id,
            })

        doc = {
            "event_id":  event_id,
            "market":    market,
            "book":      book,
            "odds":      odds_list,
            "pulled_at": datetime.utcnow(),
        }
        fk = {"event_id": event_id, "market": market, "book": book}
        db[COLLECTIONS["live_odds"]].update_one(fk, {"$set": doc}, upsert=True)
        _save_snapshot(event_id, market, book, odds_list)
        stored += 1
        log.info(f"  {market} / {book}: {len(odds_list)} players stored")

    return stored


# =============================================================================
# The Odds API (majors)
# =============================================================================

def fetch_odds(sport_key: str, market: str) -> list:
    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us",
        "markets":    market,
        "oddsFormat": "decimal",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.warning(f"Odds API fetch failed for market={market}: {exc}")
        return []


def parse_and_store_api(raw_events: list, market: str, event_id: int) -> int:
    if not raw_events:
        return 0

    stored = 0
    for event in raw_events:
        for bookmaker in event.get("bookmakers", []):
            book = bookmaker.get("key")
            for market_data in bookmaker.get("markets", []):
                if market_data.get("key") != market:
                    continue

                odds_list = [
                    {
                        "player_name":  o.get("name"),
                        "decimal_odds": o.get("price"),
                        "dg_id":        None,
                    }
                    for o in market_data.get("outcomes", [])
                ]

                doc = {
                    "event_id":  event_id,
                    "market":    market,
                    "book":      book,
                    "odds":      odds_list,
                    "pulled_at": datetime.utcnow(),
                }
                fk = {"event_id": event_id, "market": market, "book": book}
                db[COLLECTIONS["live_odds"]].update_one(fk, {"$set": doc}, upsert=True)
                _save_snapshot(event_id, market, book, odds_list)
                stored += 1

    return stored


def run_api_scrape(event_id: int, sport_key: str):
    log.info(f"Scraping Odds API: sport={sport_key}, event_id={event_id}")
    total = 0
    for market in ODDS_MARKETS:
        raw = fetch_odds(sport_key, market)
        n = parse_and_store_api(raw, market, event_id)
        log.info(f"  {market}: {n} book(s) stored")
        total += n
    log.info(f"Done. {total} total odds documents upserted.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id", type=int, required=True,
                        help="Event ID (from dg_schedule or dg_current_field)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file with odds (for non-major events)")
    parser.add_argument("--sport_key", type=str, default="golf_masters_tournament_winner",
                        help="The Odds API sport key (majors only)")
    args = parser.parse_args()

    if args.csv:
        log.info(f"Loading odds from CSV: {args.csv}")
        n = load_from_csv(args.csv, args.event_id)
        log.info(f"Done. {n} market/book combinations stored.")
    else:
        run_api_scrape(args.event_id, args.sport_key)

    log.info("Next: python live/value_detector.py --event_id <id>")


if __name__ == "__main__":
    main()
