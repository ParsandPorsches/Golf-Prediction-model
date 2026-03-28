"""
extraction/espn_results.py
---------------------------
Pulls historical PGA Tour tournament results from ESPN's public API.
Stores finish positions in MongoDB for use by the log-likelihood backtester.

Usage:
    python extraction/espn_results.py
    python extraction/espn_results.py --years 2023 2024
"""

import sys
import time
import argparse
import logging
import datetime

import requests
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import (
    MONGODB_URI, DB_NAME, COLLECTIONS,
    REQUEST_DELAY_SECONDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/golf/pga/scoreboard"
HEADERS = {"user-agent": "Mozilla/5.0"}


def espn_get(url: str, params: dict = None) -> dict:
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY_SECONDS)
        return resp.json()
    except requests.RequestException as exc:
        log.warning(f"ESPN request failed: {exc}")
        return {}


def fetch_events_for_year(year: int) -> dict:
    """
    Walk through every week of the year collecting unique event IDs.
    Returns {espn_event_id: {id, name, date}} dict.
    """
    events = {}
    date = datetime.date(year, 1, 1)
    end  = datetime.date(year, 12, 31)

    while date <= end:
        data = espn_get(ESPN_SCOREBOARD, params={"dates": date.strftime("%Y%m%d")})
        for e in data.get("events", []):
            eid = e.get("id")
            if eid and eid not in events:
                events[eid] = {
                    "id":   eid,
                    "name": e.get("name", "Unknown"),
                    "date": e.get("date", "")[:10],
                }
        date += datetime.timedelta(days=7)

    return events


def parse_event_results(espn_event: dict, competitors: list) -> list:
    """Extract finish positions from a list of ESPN competitor objects."""
    results = []
    for c in competitors:
        name = c.get("athlete", {}).get("displayName", "")
        if not name:
            continue

        finish   = c.get("order")          # 1-based finish position
        score    = c.get("score")          # total score vs par (string or int)
        status   = c.get("status", {})

        # Detect WD/DQ/MDF
        withdrew = False
        if isinstance(status, dict):
            abbr = status.get("type", {}).get("abbreviation", "")
            withdrew = abbr in ("WD", "DQ", "MDF", "CUT")

        # A finish position > field size or missing = missed cut
        made_cut = (finish is not None) and (not withdrew)

        results.append({
            "player_name": name,
            "finish":      int(finish) if finish else None,
            "score":       score,
            "made_cut":    made_cut,
            "withdrew":    withdrew,
        })

    return results


def pull_results_for_year(year: int) -> int:
    """Pull all event results for a year and store in MongoDB."""
    log.info(f"  Scanning {year} for events...")
    event_map = fetch_events_for_year(year)
    log.info(f"  Found {len(event_map)} unique events in {year}")

    stored = 0
    for eid, meta in event_map.items():
        fk = {"espn_event_id": eid, "year": year}
        if db[COLLECTIONS["espn_results"]].count_documents(fk, limit=1):
            continue

        # Re-fetch with the event's actual date to get full competitor list
        data = espn_get(ESPN_SCOREBOARD, params={"dates": meta["date"].replace("-", "")})
        target_event = next((e for e in data.get("events", []) if e.get("id") == eid), None)
        if not target_event:
            continue

        competitions = target_event.get("competitions", [])
        if not competitions:
            continue

        competitors = competitions[0].get("competitors", [])
        if not competitors:
            continue

        results = parse_event_results(target_event, competitors)
        if not results:
            continue

        doc = {
            **fk,
            "event_name": meta["name"],
            "event_date": meta["date"],
            "results":    results,
            "pulled_at":  datetime.datetime.utcnow(),
        }
        db[COLLECTIONS["espn_results"]].update_one(fk, {"$set": doc}, upsert=True)
        stored += 1
        log.info(f"  {year} / {meta['name']} — {len(results)} players ✓")

    return stored


def main():
    parser = argparse.ArgumentParser(description="Pull historical PGA Tour results from ESPN")
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2026)))
    args = parser.parse_args()

    log.info("ESPN Results Extraction")
    log.info(f"Years: {args.years}")
    log.info("=" * 60)

    db[COLLECTIONS["espn_results"]].create_index(
        [("espn_event_id", 1), ("year", 1)], unique=True
    )

    total = 0
    for year in args.years:
        log.info(f"── {year} ──────────────────────────────────────────────────")
        n = pull_results_for_year(year)
        log.info(f"  {year}: {n} events stored")
        total += n

    log.info("=" * 60)
    log.info(f"Done. {total} total events stored in {COLLECTIONS['espn_results']}.")
    log.info("Next: python backtester/log_likelihood_optimizer.py")


if __name__ == "__main__":
    main()
