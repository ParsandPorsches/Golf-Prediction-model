"""
model/sg_composite.py
---------------------
Computes player skill scores using DataGolf's prediction archive
as the primary signal (Option B — no raw SG rounds needed).
"""

import sys
import math
import logging
import numpy as np
import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import MONGODB_URI, DB_NAME, COLLECTIONS

log = logging.getLogger(__name__)
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]


class SGWeights:
    """Holds one set of SG weights. Kept for interface compatibility."""
    def __init__(self, sg_ott: float, sg_app: float, sg_arg: float, sg_putt: float):
        total = sg_ott + sg_app + sg_arg + sg_putt
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")
        self.sg_ott  = sg_ott
        self.sg_app  = sg_app
        self.sg_arg  = sg_arg
        self.sg_putt = sg_putt

    def as_dict(self):
        return {"sg_ott": self.sg_ott, "sg_app": self.sg_app,
                "sg_arg": self.sg_arg, "sg_putt": self.sg_putt}

    def __repr__(self):
        return (f"SGWeights(ott={self.sg_ott:.2f}, app={self.sg_app:.2f}, "
                f"arg={self.sg_arg:.2f}, putt={self.sg_putt:.2f})")


def get_current_predictions() -> pd.DataFrame:
    """Load this week's predictions from MongoDB."""
    doc = db["dg_current_predictions"].find_one({}, sort=[("pulled_at", -1)])
    if not doc:
        log.error("No current predictions in MongoDB. Run datagolf_pull.py first.")
        return pd.DataFrame()

    players = doc.get("players", [])
    rows = []
    for p in players:
        win_prob = max(p.get("win", 0.01), 0.0001)
        # Convert win prob to skill score via log-odds
        sg_composite = math.log(win_prob / (1 - win_prob)) + 4.5
        rows.append({
            "dg_id":          p.get("dg_id"),
            "player_name":    p.get("player_name"),
            "dg_win_prob":    win_prob,
            "dg_top5_prob":   p.get("top_5", 0),
            "dg_top10_prob":  p.get("top_10", 0),
            "dg_top20_prob":  p.get("top_20", 0),
            "dg_make_cut_prob": p.get("make_cut", 0),
            "sg_composite":   sg_composite,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("sg_composite", ascending=False).reset_index(drop=True)


def build_live_field_scores(player_dg_ids: list, weights: SGWeights) -> pd.DataFrame:
    """
    Build field scores for current week.
    Primary: DataGolf predictions (win prob -> sg_composite)
    Fallback: skill_ratings + weights
    """
    preds = get_current_predictions()

    if not preds.empty:
        field_preds = preds[preds["dg_id"].isin(player_dg_ids)].copy()
        if len(field_preds) >= 10:
            log.info(f"  Using DataGolf predictions for {len(field_preds)} players")
            return field_preds

    log.warning("Falling back to skill ratings")
    return _build_from_skill_ratings(player_dg_ids, weights)


def _build_from_skill_ratings(player_dg_ids: list, weights: SGWeights) -> pd.DataFrame:
    docs = list(db[COLLECTIONS["skill_ratings"]].find(
        {"dg_id": {"$in": player_dg_ids}},
        {"dg_id": 1, "player_name": 1, "sg_ott": 1, "sg_app": 1, "sg_arg": 1, "sg_putt": 1}
    ))
    if not docs:
        log.error("No skill ratings found.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df = df.dropna(subset=["sg_ott", "sg_app", "sg_arg", "sg_putt"])
    df["sg_composite"] = (
        weights.sg_ott  * df["sg_ott"]  +
        weights.sg_app  * df["sg_app"]  +
        weights.sg_arg  * df["sg_arg"]  +
        weights.sg_putt * df["sg_putt"]
    )

    missing_ids = set(player_dg_ids) - set(df["dg_id"].tolist())
    if missing_ids:
        median = df["sg_composite"].median()
        extras = pd.DataFrame([
            {"dg_id": i, "player_name": f"Unknown ({i})", "sg_composite": median}
            for i in missing_ids
        ])
        df = pd.concat([df, extras], ignore_index=True)

    return df[["dg_id", "player_name", "sg_composite"]].sort_values(
        "sg_composite", ascending=False).reset_index(drop=True)


def build_field_scores(
    player_dg_ids: list,
    weights: SGWeights,
    event_date=None,
    event_id: int = None,
    year: int = None,
) -> pd.DataFrame:
    """
    Build field scores for backtesting using historical predictions archive.

    Tries (in order):
      1. predictions_archive SG components for the given event
      2. Current skill_ratings weighted by `weights`

    Returns DataFrame with [dg_id, player_name, sg_composite].
    """
    # Try predictions archive if event context is provided
    if event_id and year:
        doc = db[COLLECTIONS["predictions_archive"]].find_one(
            {"event_id": event_id, "year": year}
        )
        if doc:
            raw = doc.get("raw", {})
            players = raw.get("baseline", raw.get("players", []))
            rows = []
            for p in players:
                dg_id = p.get("dg_id")
                if not dg_id or dg_id not in player_dg_ids:
                    continue
                sg_ott  = p.get("sg_ott")
                sg_app  = p.get("sg_app")
                sg_arg  = p.get("sg_arg")
                sg_putt = p.get("sg_putt")
                if all(v is not None for v in [sg_ott, sg_app, sg_arg, sg_putt]):
                    composite = (
                        weights.sg_ott  * sg_ott  +
                        weights.sg_app  * sg_app  +
                        weights.sg_arg  * sg_arg  +
                        weights.sg_putt * sg_putt
                    )
                elif p.get("win"):
                    # Fall back to log-odds transform of win prob
                    win_prob  = max(p["win"], 0.0001)
                    composite = math.log(win_prob / (1 - win_prob)) + 4.5
                else:
                    continue
                rows.append({
                    "dg_id":        dg_id,
                    "player_name":  p.get("player_name", f"Unknown ({dg_id})"),
                    "sg_composite": composite,
                })
            if len(rows) >= 10:
                df = pd.DataFrame(rows)
                return df.sort_values("sg_composite", ascending=False).reset_index(drop=True)

    # Fall back to current skill ratings
    return _build_from_skill_ratings(player_dg_ids, weights)


def get_current_field_ids() -> tuple:
    """Returns (list of dg_ids, event_name) for the current tournament."""
    doc = db["dg_current_field"].find_one({}, sort=[("pulled_at", -1)])
    if not doc:
        pred_doc = db["dg_current_predictions"].find_one({}, sort=[("pulled_at", -1)])
        if pred_doc:
            ids = [p["dg_id"] for p in pred_doc.get("players", []) if p.get("dg_id")]
            return ids, pred_doc.get("event_name", "Unknown Event")
        return [], "Unknown Event"

    field = doc.get("field", [])
    ids = [p.get("dg_id") for p in field if p.get("dg_id")]
    return ids, doc.get("event_name", "Unknown Event")
