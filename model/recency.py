"""
model/recency.py
----------------
Recency-weighted skill ratings built by blending multiple DataGolf time windows.

DataGolf provides skill ratings for rolling look-back periods (l24, l36, l72
months). This module pulls all three, blends them with heavier weight on the
most-recent window, stores the result in MongoDB, and exposes a function that
adjusts sg_composite to incorporate recency-weighted skill.

Usage:
    from model.recency import pull_recency_ratings, apply_recency
    recency_df = pull_recency_ratings(db)
    field_df   = apply_recency(field_df, db, weights)
"""

import sys
import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, ".")
from config.settings import DATAGOLF_API_KEY, COLLECTIONS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# DataGolf skill ratings endpoint
_DG_SKILL_URL = (
    "https://feeds.datagolf.com/preds/skill-ratings"
    "?display=value&period={period}&file_format=json&key={key}"
)

# Blend weights for each look-back period (normalised automatically if a
# period is unavailable)
_PERIOD_WEIGHTS = {
    "l24": 0.50,   # most recent 24 months
    "l36": 0.30,   # most recent 36 months
    "l72": 0.20,   # most recent 72 months
}

# MongoDB collection that stores blended recency ratings
_RECENCY_COLLECTION = "dg_skill_ratings_recency"

# How many days before the stored recency ratings are considered stale
_STALE_DAYS = 7

# SG component columns expected in DataGolf responses
_SG_COLS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_period(period: str) -> pd.DataFrame | None:
    """
    Pull skill ratings for one DataGolf look-back period.

    Returns a DataFrame with columns:
        dg_id, player_name, sg_ott, sg_app, sg_arg, sg_putt

    Returns None if the request fails (404, network error, etc.).
    """
    url = _DG_SKILL_URL.format(period=period, key=DATAGOLF_API_KEY)
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            log.warning(f"  recency: period '{period}' returned 404, skipping.")
            return None
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.warning(f"  recency: request failed for period '{period}': {exc}")
        return None
    except ValueError as exc:
        log.warning(f"  recency: JSON decode error for period '{period}': {exc}")
        return None

    # DataGolf returns either a top-level list or {"players": [...]}
    players = data if isinstance(data, list) else data.get("players", [])
    if not players:
        log.warning(f"  recency: empty player list for period '{period}'.")
        return None

    rows = []
    for p in players:
        dg_id = p.get("dg_id")
        if not dg_id:
            continue
        sg_ott  = p.get("sg_ott")
        sg_app  = p.get("sg_app")
        sg_arg  = p.get("sg_arg")
        sg_putt = p.get("sg_putt")
        if any(v is None for v in [sg_ott, sg_app, sg_arg, sg_putt]):
            continue
        rows.append({
            "dg_id":       int(dg_id),
            "player_name": p.get("player_name", f"Unknown ({dg_id})"),
            "sg_ott":      float(sg_ott),
            "sg_app":      float(sg_app),
            "sg_arg":      float(sg_arg),
            "sg_putt":     float(sg_putt),
        })

    if not rows:
        log.warning(f"  recency: no valid rows for period '{period}'.")
        return None

    df = pd.DataFrame(rows).set_index("dg_id")
    log.info(f"  recency: fetched {len(df)} players for period '{period}'.")
    return df


def _blend_periods(period_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine skill rating DataFrames from multiple periods into a single
    blended DataFrame.

    Each period's data is weighted according to _PERIOD_WEIGHTS, with
    normalisation applied if some periods are missing.

    Returns a DataFrame with:
        dg_id (index), player_name, sg_ott_recency, sg_app_recency,
        sg_arg_recency, sg_putt_recency
    """
    available = {k: v for k, v in period_frames.items() if v is not None}
    if not available:
        log.error("  recency: no period data available to blend.")
        return pd.DataFrame()

    # Normalise weights for available periods
    raw_total = sum(_PERIOD_WEIGHTS[p] for p in available)
    norm_weights = {p: _PERIOD_WEIGHTS[p] / raw_total for p in available}
    log.info(f"  recency: blending periods {list(available.keys())} "
             f"with normalised weights {norm_weights}")

    # Collect all dg_ids that appear in at least one period
    all_ids = set()
    for df in available.values():
        all_ids.update(df.index.tolist())

    records = []
    for dg_id in all_ids:
        # Accumulate weighted SG values across available periods
        sg_ott = sg_app = sg_arg = sg_putt = 0.0
        weight_used = 0.0
        player_name = f"Unknown ({dg_id})"

        for period, df in available.items():
            if dg_id not in df.index:
                continue
            row = df.loc[dg_id]
            w = norm_weights[period]
            sg_ott  += w * row["sg_ott"]
            sg_app  += w * row["sg_app"]
            sg_arg  += w * row["sg_arg"]
            sg_putt += w * row["sg_putt"]
            weight_used += w
            if "player_name" in df.columns:
                player_name = row["player_name"]

        if weight_used < 1e-9:
            continue

        # Re-normalise to handle players not appearing in all periods
        factor = 1.0 / weight_used
        records.append({
            "dg_id":             int(dg_id),
            "player_name":       player_name,
            "sg_ott_recency":    sg_ott  * factor,
            "sg_app_recency":    sg_app  * factor,
            "sg_arg_recency":    sg_arg  * factor,
            "sg_putt_recency":   sg_putt * factor,
        })

    if not records:
        log.error("  recency: blending produced no records.")
        return pd.DataFrame()

    return pd.DataFrame(records)


def _is_recency_stale(db) -> bool:
    """
    Return True if the recency collection is empty or the newest document
    is older than _STALE_DAYS days.
    """
    doc = db[_RECENCY_COLLECTION].find_one({}, sort=[("pulled_at", -1)])
    if not doc:
        return True
    pulled_at = doc.get("pulled_at")
    if not pulled_at:
        return True
    cutoff = datetime.now(timezone.utc) - timedelta(days=_STALE_DAYS)
    # pulled_at may be naive; treat it as UTC for comparison
    if pulled_at.tzinfo is None:
        pulled_at = pulled_at.replace(tzinfo=timezone.utc)
    return pulled_at < cutoff


def _load_recency_from_mongo(db) -> pd.DataFrame:
    """
    Load the stored recency ratings from MongoDB.

    Returns a DataFrame with columns:
        dg_id, player_name, sg_ott_recency, sg_app_recency,
        sg_arg_recency, sg_putt_recency
    """
    docs = list(db[_RECENCY_COLLECTION].find(
        {},
        {"_id": 0, "dg_id": 1, "player_name": 1,
         "sg_ott_recency": 1, "sg_app_recency": 1,
         "sg_arg_recency": 1, "sg_putt_recency": 1}
    ))
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame(docs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pull_recency_ratings(db) -> pd.DataFrame:
    """
    Pull skill ratings for l24, l36, and l72 periods from DataGolf, blend
    them, store in MongoDB, and return the result.

    Periods that return a 404 or network error are skipped gracefully;
    blend weights are normalised across whichever periods succeed.

    Parameters
    ----------
    db : pymongo.database.Database
        Active MongoDB database connection.

    Returns
    -------
    pd.DataFrame
        Columns: dg_id, player_name, sg_ott_recency, sg_app_recency,
                 sg_arg_recency, sg_putt_recency
        Returns an empty DataFrame if all period fetches fail.
    """
    log.info("  recency: pulling multi-period skill ratings from DataGolf...")

    period_frames: dict[str, pd.DataFrame | None] = {}
    for period in _PERIOD_WEIGHTS:
        period_frames[period] = _fetch_period(period)
        # Respect API rate limit between calls
        time.sleep(1.0)

    blended = _blend_periods(period_frames)
    if blended.empty:
        log.error("  recency: blending failed. No data stored.")
        return pd.DataFrame()

    # Upsert each player into MongoDB
    now = datetime.now(timezone.utc)
    for _, row in blended.iterrows():
        doc = row.to_dict()
        doc["pulled_at"] = now
        db[_RECENCY_COLLECTION].update_one(
            {"dg_id": int(doc["dg_id"])},
            {"$set": doc},
            upsert=True,
        )

    log.info(f"  recency: upserted {len(blended)} players into '{_RECENCY_COLLECTION}'.")
    return blended


def apply_recency(
    field_df: pd.DataFrame,
    db,
    weights,
    alpha: float = 0.35,
) -> pd.DataFrame:
    """
    Adjust sg_composite by blending in a recency-weighted skill composite.

    Recency data is loaded from MongoDB if available and not stale. If stale
    or absent, a fresh pull from DataGolf is attempted first.

    Parameters
    ----------
    field_df : pd.DataFrame
        Must contain columns: dg_id, player_name, sg_composite.
    db : pymongo.database.Database
        Active MongoDB database connection.
    weights : SGWeights
        The same SGWeights instance used to build field_scores, providing
        ott/app/arg/putt component weights.
    alpha : float
        Blend weight for recency signal. Default 0.35 means:
        adjusted = 0.65 * sg_composite + 0.35 * recency_composite_normalised

    Returns
    -------
    pd.DataFrame
        field_df with additional columns:
            - sg_composite_original : original sg_composite (if not already set)
            - recency_composite     : raw recency weighted composite per player
        The sg_composite column is updated in-place.

    Notes
    -----
    If no recency data is available after the pull attempt, field_df is
    returned unchanged.
    """
    if field_df.empty:
        log.warning("  apply_recency: received empty DataFrame, skipping.")
        return field_df

    # Load recency ratings, refreshing if stale
    if _is_recency_stale(db):
        log.info("  Recency ratings are stale or missing — pulling fresh data.")
        recency_df = pull_recency_ratings(db)
    else:
        recency_df = _load_recency_from_mongo(db)
        log.info(f"  Loaded {len(recency_df)} recency ratings from MongoDB.")

    if recency_df.empty:
        log.warning("  No recency data available. Returning field_df unchanged.")
        return field_df

    recency_cols = ["sg_ott_recency", "sg_app_recency",
                    "sg_arg_recency", "sg_putt_recency"]
    if not all(c in recency_df.columns for c in recency_cols):
        log.warning("  Recency DataFrame is missing expected columns. Skipping.")
        return field_df

    # Merge recency data into field_df
    df = field_df.copy()
    recency_lookup = recency_df[["dg_id"] + recency_cols].copy()
    recency_lookup["dg_id"] = recency_lookup["dg_id"].astype(int)
    df["dg_id"] = df["dg_id"].astype(int)

    df = df.merge(recency_lookup, on="dg_id", how="left")

    has_recency = df[recency_cols].notna().all(axis=1)
    n_matched = has_recency.sum()

    if n_matched < 5:
        log.warning(
            f"  Only {n_matched} players matched recency ratings. "
            "Skipping recency adjustment."
        )
        return field_df

    # Compute recency composite using the same SG weights as the main model
    df["recency_composite"] = np.nan
    df.loc[has_recency, "recency_composite"] = (
        weights.sg_ott  * df.loc[has_recency, "sg_ott_recency"]  +
        weights.sg_app  * df.loc[has_recency, "sg_app_recency"]  +
        weights.sg_arg  * df.loc[has_recency, "sg_arg_recency"]  +
        weights.sg_putt * df.loc[has_recency, "sg_putt_recency"]
    )

    # Normalise recency_composite to the same mean/std as sg_composite so
    # the blend is numerically consistent
    rec_mean = df.loc[has_recency, "recency_composite"].mean()
    rec_std  = df.loc[has_recency, "recency_composite"].std(ddof=1)

    if rec_std < 1e-9:
        log.warning("  recency_composite has near-zero variance. Skipping adjustment.")
        return field_df

    sg_mean = df["sg_composite"].mean()
    sg_std  = df["sg_composite"].std(ddof=1)

    if sg_std < 1e-9:
        log.warning("  sg_composite has near-zero variance. Skipping adjustment.")
        return field_df

    # Rescale recency z-scores to sg_composite scale
    recency_z = (df.loc[has_recency, "recency_composite"] - rec_mean) / rec_std
    recency_normalised = recency_z * sg_std + sg_mean

    # Preserve original composite only if not already saved by course_fit
    if "sg_composite_original" not in df.columns:
        df["sg_composite_original"] = df["sg_composite"]

    # Blend
    df.loc[has_recency, "sg_composite"] = (
        (1 - alpha) * df.loc[has_recency, "sg_composite"]
        + alpha * recency_normalised
    )

    log.info(
        f"  Recency adjustment applied: alpha={alpha:.2f}, "
        f"{n_matched} players updated."
    )

    # Drop the temporary recency component columns to keep the DataFrame tidy
    df = df.drop(columns=recency_cols, errors="ignore")

    return df
