"""
extraction/datagolf_pull.py
---------------------------
Phase 1: Data extraction using basic DataGolf subscription.

Pulls:
  - Player list
  - Tour schedule (event IDs)
  - Skill ratings (current SG decompositions)
  - Pre-tournament prediction archive (DataGolf's own model probabilities)

Skips (requires upgraded plan):
  - historical-raw-data/rounds  (raw SG per round)
  - historical-odds/outrights   (Pinnacle/DK/FD closing lines)

Usage:
    python extraction/datagolf_pull.py
    python extraction/datagolf_pull.py --years 2023 2024
"""

import sys
import time
import argparse
import logging
from datetime import datetime

import requests
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError

sys.path.insert(0, ".")
from config.settings import (
    MONGODB_URI, DB_NAME, COLLECTIONS,
    DATAGOLF_API_KEY, EXTRACTION_YEARS,
    REQUEST_DELAY_SECONDS, MAX_RETRIES, RETRY_DELAY_SECONDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

DG_BASE = "https://feeds.datagolf.com"


def dg_get(endpoint: str, params: dict = None) -> dict:
    if params is None:
        params = {}
    params["key"] = DATAGOLF_API_KEY
    url = f"{DG_BASE}/{endpoint}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY_SECONDS)
            return resp.json()
        except requests.RequestException as exc:
            log.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {endpoint}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                raise RuntimeError(f"All retries exhausted for {endpoint}") from exc


def upsert_many(collection_name: str, docs: list, id_fields: list) -> int:
    if not docs:
        return 0
    col = db[collection_name]
    ops = []
    for doc in docs:
        filter_key = {f: doc[f] for f in id_fields if f in doc}
        ops.append(UpdateOne(filter_key, {"$set": doc}, upsert=True))
    try:
        result = col.bulk_write(ops, ordered=False)
        return result.upserted_count + result.modified_count
    except BulkWriteError as bwe:
        log.error(f"Bulk write error in {collection_name}: {bwe.details}")
        return 0


def doc_exists(collection_name: str, filter_key: dict) -> bool:
    return db[collection_name].count_documents(filter_key, limit=1) > 0


def pull_player_list():
    log.info("── Pulling player list ──────────────────────────────────")
    data = dg_get("get-player-list")
    players = data if isinstance(data, list) else data.get("players", [])
    docs = []
    for p in players:
        docs.append({
            "dg_id":      p.get("dg_id"),
            "dg_name":    p.get("player_name"),
            "country":    p.get("country"),
            "amateur":    p.get("amateur", False),
            "updated_at": datetime.now(),
        })
    n = upsert_many(COLLECTIONS["player_map"], docs, id_fields=["dg_id"])
    log.info(f"  Player list: {len(docs)} players upserted ({n} new/updated)")


def pull_schedule() -> list:
    log.info("── Pulling tour schedule ────────────────────────────────")
    try:
        data = dg_get("get-schedule", params={"tour": "pga", "file_format": "json"})
        events = data if isinstance(data, list) else data.get("schedule", [])
        docs = []
        for e in events:
            docs.append({
                "event_id":   e.get("event_id"),
                "event_name": e.get("event_name"),
                "course":     e.get("course"),
                "date":       e.get("date"),
                "tour":       "pga",
                "year":       datetime.now().year,
            })
        upsert_many("dg_schedule", docs, id_fields=["event_id"])
        log.info(f"  Schedule: {len(docs)} events stored")
        return events
    except RuntimeError as exc:
        log.error(f"  Schedule pull failed: {exc}")
        return []


def get_stored_event_ids() -> list:
    docs = list(db["dg_schedule"].find({}, {"event_id": 1}))
    return [d["event_id"] for d in docs if "event_id" in d]


def pull_skill_ratings():
    log.info("── Pulling skill ratings ────────────────────────────────")
    data = dg_get("preds/skill-ratings", params={"display": "value", "file_format": "json"})
    players = data if isinstance(data, list) else data.get("players", [])
    docs = []
    for p in players:
        docs.append({
            "dg_id":       p.get("dg_id"),
            "player_name": p.get("player_name"),
            "sg_ott":      p.get("sg_ott"),
            "sg_app":      p.get("sg_app"),
            "sg_arg":      p.get("sg_arg"),
            "sg_putt":     p.get("sg_putt"),
            "sg_total":    p.get("sg_total"),
            "dg_rank":     p.get("dg_rank"),
            "pulled_at":   datetime.now(),
        })
    n = upsert_many(COLLECTIONS["skill_ratings"], docs, id_fields=["dg_id"])
    log.info(f"  Skill ratings: {len(docs)} players upserted ({n} new/updated)")


def pull_predictions_archive(years: list):
    log.info("── Pulling predictions archive ──────────────────────────")
    log.info("  (Primary dataset — DataGolf model probs per event/year)")
    total = 0
    event_ids = get_stored_event_ids()
    if not event_ids:
        log.warning("  No event IDs — pulling schedule first")
        pull_schedule()
        event_ids = get_stored_event_ids()

    for year in years:
        year_count = 0
        for event_id in event_ids:
            fk = {"event_id": event_id, "year": year}
            if doc_exists(COLLECTIONS["predictions_archive"], fk):
                continue
            try:
                url = f"https://feeds.datagolf.com/preds/pre-tournament-archive?tour=pga&event_id={event_id}&year={year}&file_format=json&key={DATAGOLF_API_KEY}"
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                time.sleep(REQUEST_DELAY_SECONDS)
                data = resp.json()
                if not data or (isinstance(data, dict) and data.get("error")):
                    continue
                event_name = data.get("event_name", event_id) if isinstance(data, dict) else event_id
                doc = {**fk, "event_name": event_name, "raw": data, "pulled_at": datetime.now()}
                db[COLLECTIONS["predictions_archive"]].update_one(fk, {"$set": doc}, upsert=True)
                total += 1
                year_count += 1
                log.info(f"  {year} / {event_name} ✓")
            except (RuntimeError, requests.RequestException) as exc:
                log.warning(f"  Pred archive {year}/{event_id}: {exc}")
        log.info(f"  {year}: {year_count} events stored")

    log.info(f"  Prediction archive total: {total} event-year documents")


def pull_current_predictions():
    log.info("── Pulling current week predictions ─────────────────────")
    try:
        url = f"https://feeds.datagolf.com/preds/pre-tournament?tour=pga&file_format=json&key={DATAGOLF_API_KEY}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        event_name = data.get("event_name", "current")
        event_id   = data.get("event_id")
        players    = data.get("baseline", data.get("players", []))
        if not players:
            log.warning("  No current predictions available")
            return
        doc = {"event_id": event_id, "event_name": event_name,
               "players": players, "pulled_at": datetime.now()}
        db["dg_current_predictions"].update_one({"event_id": event_id}, {"$set": doc}, upsert=True)
        log.info(f"  Current predictions: {event_name} — {len(players)} players ✓")
    except Exception as exc:
        log.warning(f"  Current predictions pull failed: {exc}")


def pull_current_field():
    log.info("── Pulling current tournament field ─────────────────────")
    try:
        data = dg_get("field-updates", params={"tour": "pga", "file_format": "json"})
        event_name = data.get("event_name", "current")
        event_id   = data.get("event_id")
        players    = data.get("field", [])
        if not players:
            log.warning("  No current field data available")
            return
        doc = {"event_id": event_id, "event_name": event_name,
               "field": players, "pulled_at": datetime.now()}
        db["dg_current_field"].update_one({"event_id": event_id}, {"$set": doc}, upsert=True)
        log.info(f"  Current field: {event_name} — {len(players)} players ✓")
    except RuntimeError as exc:
        log.warning(f"  Current field pull failed: {exc}")


def create_indexes():
    log.info("── Creating MongoDB indexes ──────────────────────────────")
    db[COLLECTIONS["player_map"]].create_index("dg_id", unique=True)
    db[COLLECTIONS["predictions_archive"]].create_index(
        [("event_id", 1), ("year", 1)], unique=True)
    db[COLLECTIONS["skill_ratings"]].create_index("dg_id", unique=True)
    db[COLLECTIONS["model_predictions"]].create_index(
        [("event_id", 1), ("year", 1), ("dg_id", 1)])
    db[COLLECTIONS["bet_log"]].create_index(
        [("event_id", 1), ("dg_id", 1), ("market", 1)])
    log.info("  Indexes created ✓")


def main():
    parser = argparse.ArgumentParser(description="DataGolf Phase 1 extraction (basic plan)")
    parser.add_argument("--years", nargs="+", type=int, default=EXTRACTION_YEARS)
    parser.add_argument("--skip-archive", action="store_true")
    args = parser.parse_args()

    log.info("Golf Model v3 — Phase 1 Extraction (Basic Plan)")
    log.info(f"Years: {args.years}")
    log.info(f"MongoDB: {MONGODB_URI} / {DB_NAME}")
    log.info("=" * 60)

    create_indexes()
    pull_player_list()
    pull_schedule()
    pull_skill_ratings()
    pull_current_field()
    pull_current_predictions()

    if not args.skip_archive:
        pull_predictions_archive(args.years)

    log.info("=" * 60)
    log.info("Phase 1 complete.")
    log.info("Collections populated:")
    log.info("  dg_player_map          — player IDs and names")
    log.info("  dg_skill_ratings       — current SG decompositions (438 players)")
    log.info("  dg_predictions_archive — DataGolf model probs per event/year")
    log.info("  dg_current_predictions — this week's predictions")
    log.info("")
    log.info("Next: python extraction/espn_results.py")


if __name__ == "__main__":
    main()