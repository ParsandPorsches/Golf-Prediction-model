"""
model/course_history.py
------------------------
Adjusts sg_composite based on each player's historical performance at the
current venue. Players who consistently finish well at a course get a boost;
those who consistently underperform get faded.

Data source: ESPN historical results (MongoDB espn_tournament_results).
Requires at least 2 appearances at the venue to apply any adjustment.

Approach:
  1. Map the current tournament name to all historical ESPN event name variants
     at the same physical venue (e.g., "Houston Open", "Cadence Bank Houston
     Open", and "Texas Children's Houston Open" all map to Memorial Park).
  2. For each player, find every historical finish at that venue.
  3. Convert raw finishes to percentile rank within that field (0=best, 1=worst).
  4. Weight recent appearances more heavily (year decay).
  5. Convert each player's weighted-average percentile into a z-score, then
     blend into sg_composite with a small alpha (default 0.15).

Usage:
    from model.course_history import apply_course_history
    field_df = apply_course_history(field_df, "Texas Children's Houston Open", db)
"""

import sys
import logging
from difflib import get_close_matches

import numpy as np
import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import MONGODB_URI, DB_NAME, COLLECTIONS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Venue -> ESPN event name groupings
# Each key is a canonical venue; the list contains every ESPN event_name
# variant that was played at that venue.
# ---------------------------------------------------------------------------
VENUE_EVENTS = {
    "Memorial Park": [
        "Houston Open",
        "Cadence Bank Houston Open",
        "Texas Children's Houston Open",
    ],
    "Augusta National": [
        "Masters Tournament",
        "2021 Masters Tournament",
        "Masters",
    ],
    "Pebble Beach": [
        "AT&T Pebble Beach Pro-Am",
    ],
    "TPC Sawgrass": [
        "THE PLAYERS Championship",
        "The Players Championship",
    ],
    "Riviera": [
        "The Genesis Invitational",
        "Genesis Invitational",
    ],
    "Torrey Pines": [
        "Farmers Insurance Open",
    ],
    "Bay Hill": [
        "Arnold Palmer Invitational pres. by Mastercard",
        "Arnold Palmer Invitational Pres. By Mastercard",
    ],
    "Muirfield Village": [
        "the Memorial Tournament pres. by Workday",
        "Memorial Tournament",
    ],
    "Colonial": [
        "Charles Schwab Challenge",
    ],
    "Quail Hollow": [
        "Wells Fargo Championship",
    ],
    "East Lake": [
        "TOUR Championship",
    ],
    "Kapalua": [
        "Sentry Tournament of Champions",
    ],
    "TPC Scottsdale": [
        "WM Phoenix Open",
    ],
    "Harbour Town": [
        "RBC Heritage",
    ],
    "Sedgefield": [
        "Wyndham Championship",
    ],
    "TPC Twin Cities": [
        "3M Open",
    ],
    "TPC Southwind": [
        "FedEx St. Jude Championship",
        "WGC-FedEx St. Jude Invitational",
    ],
    "Waialae": [
        "Sony Open in Hawaii",
    ],
    "TPC San Antonio": [
        "Valero Texas Open",
    ],
    "Innisbrook": [
        "Valspar Championship",
    ],
    "Country Club of Jackson": [
        "Sanderson Farms Championship",
    ],
    "Silverado Resort": [
        "Fortinet Championship",
    ],
    "TPC Summerlin": [
        "Shriners Children's Open",
        "Shriners Hospitals for Children Open",
    ],
    "PGA West": [
        "The American Express",
        "American Express",
    ],
    "Hamilton Golf": [
        "RBC Canadian Open",
    ],
    "Renaissance Club": [
        "Genesis Scottish Open",
    ],
    "TPC River Highlands": [
        "Travelers Championship",
    ],
    "Detroit Golf Club": [
        "Rocket Mortgage Classic",
    ],
    "TPC Deere Run": [
        "John Deere Classic",
    ],
    "Country Club of Jackson": [
        "Sanderson Farms Championship",
    ],
    "Congaree": [
        "Palmetto Championship at Congaree",
    ],
    "El Cardonal": [
        "World Wide Technology Championship",
        "World Wide Technology Championship at Mayakoba",
        "Mayakoba Golf Classic",
    ],
    "Vidanta Vallarta": [
        "Mexico Open",
        "Mexico Open at Vidanta",
    ],
    "Grand Reserve": [
        "Puerto Rico Open",
    ],
    "Bermuda": [
        "Butterfield Bermuda Championship",
    ],
    "Barbasol": [
        "Barbasol Championship",
    ],
    "Barracuda": [
        "Barracuda Championship",
    ],
    "Myrtle Beach": [
        "Myrtle Beach Classic",
    ],
    "Black Desert": [
        "Black Desert Championship",
    ],
    "Procore": [
        "Procore Championship",
    ],
    "Corales": [
        "Corales Puntacana Championship",
        "Corales Puntacana Resort & Club Championship",
    ],
    "Byron Nelson": [
        "AT&T Byron Nelson",
        "THE CJ CUP Byron Nelson",
    ],
    "CJ Cup": [
        "THE CJ CUP @ SUMMIT",
        "THE CJ CUP in South Carolina",
    ],
    "Cognizant": [
        "Cognizant Classic",
    ],
    "ISCO": [
        "ISCO Championship",
    ],
    "Northern Trust": [
        "THE NORTHERN TRUST",
    ],
    "ZOZO": [
        "ZOZO CHAMPIONSHIP",
    ],
    "Accordia Narashino": [
        "ZOZO CHAMPIONSHIP",
    ],
}

# Flat lookup: lowercase event_name -> canonical venue key
_EVENT_TO_VENUE: dict[str, str] = {}
for _venue, _events in VENUE_EVENTS.items():
    for _ev in _events:
        _EVENT_TO_VENUE[_ev.lower()] = _venue

# Year decay weights: more recent = higher weight
_YEAR_WEIGHTS = {
    2025: 1.00,
    2024: 0.85,
    2023: 0.70,
    2022: 0.55,
    2021: 0.40,
    2020: 0.30,
    2019: 0.20,
}


def _resolve_venue(event_name: str) -> str | None:
    """Map an event name to its canonical venue. Returns None if not found."""
    key = event_name.strip().lower()
    if key in _EVENT_TO_VENUE:
        return _EVENT_TO_VENUE[key]
    close = get_close_matches(key, list(_EVENT_TO_VENUE.keys()), n=1, cutoff=0.75)
    if close:
        return _EVENT_TO_VENUE[close[0]]
    return None


def _build_dg_id_name_map(db) -> dict[str, int]:
    """
    Build a lookup from lowercase 'first last' player name -> dg_id
    using the dg_player_map collection (which stores 'Last, First').
    """
    docs = list(db[COLLECTIONS["player_map"]].find(
        {}, {"_id": 0, "dg_id": 1, "dg_name": 1}
    ))
    mapping = {}
    for d in docs:
        raw = d.get("dg_name", "")
        dg_id = d.get("dg_id")
        if not raw or not dg_id:
            continue
        # "Last, First" -> "first last"
        if "," in raw:
            parts = raw.split(",", 1)
            normalized = f"{parts[1].strip()} {parts[0].strip()}".lower()
        else:
            normalized = raw.lower().strip()
        mapping[normalized] = dg_id
    return mapping


def _load_venue_history(venue: str, db) -> pd.DataFrame:
    """
    Load all ESPN results for the given venue (across all name variants).

    Returns a long DataFrame with columns:
        year, player_name, finish_pct
    where finish_pct is the within-field percentile rank (0=best, 1=worst).
    """
    event_names = VENUE_EVENTS.get(venue, [])
    if not event_names:
        return pd.DataFrame()

    docs = list(db[COLLECTIONS["espn_results"]].find(
        {"event_name": {"$in": event_names}},
        {"_id": 0, "year": 1, "results": 1}
    ))
    if not docs:
        return pd.DataFrame()

    rows = []
    for doc in docs:
        year = doc.get("year")
        results = doc.get("results", [])
        field_size = len(results)
        if field_size < 5:
            continue

        for r in results:
            finish = r.get("finish")
            made_cut = r.get("made_cut", True)
            withdrew = r.get("withdrew", False)
            if not made_cut or withdrew or finish is None:
                finish = field_size + 1
            rows.append({
                "year":        year,
                "player_name": r.get("player_name", "").strip(),
                "finish":      int(finish),
                "field_size":  field_size,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["player_name"] != ""]

    # Percentile rank within each year's field (0=best, 1=worst)
    df["finish_pct"] = df.groupby("year")["finish"].transform(
        lambda x: x.rank(method="average", pct=True)
    )
    return df


def _compute_player_history_scores(
    history_df: pd.DataFrame,
    dg_id_map: dict[str, int],
    min_appearances: int = 2,
) -> dict[int, float]:
    """
    For each player in history_df, compute a year-weighted average
    finish percentile. Returns {dg_id: weighted_avg_pct} for players
    with >= min_appearances.
    """
    scores = {}
    for player_name, group in history_df.groupby("player_name"):
        key = player_name.lower()
        dg_id = dg_id_map.get(key)

        if dg_id is None:
            # Try fuzzy match
            close = get_close_matches(key, list(dg_id_map.keys()), n=1, cutoff=0.82)
            if close:
                dg_id = dg_id_map[close[0]]

        if dg_id is None:
            continue

        weighted_sum = 0.0
        weight_total = 0.0
        for _, row in group.iterrows():
            w = _YEAR_WEIGHTS.get(int(row["year"]), 0.15)
            weighted_sum  += w * row["finish_pct"]
            weight_total  += w

        if weight_total < 1e-9:
            continue

        n_apps = len(group)
        if n_apps < min_appearances:
            continue

        scores[dg_id] = weighted_sum / weight_total

    return scores


def apply_course_history(
    field_df: pd.DataFrame,
    event_name: str,
    db,
    alpha: float = 0.15,
    min_appearances: int = 2,
) -> pd.DataFrame:
    """
    Adjust sg_composite for course-specific player history.

    Parameters
    ----------
    field_df : pd.DataFrame
        Must contain columns: dg_id, player_name, sg_composite.
    event_name : str
        Current tournament name — fuzzy-matched to a venue group.
    db : pymongo.database.Database
    alpha : float
        Blend weight (default 0.15). Small because history samples are limited.
    min_appearances : int
        Minimum appearances at the venue to qualify for adjustment (default 2).

    Returns
    -------
    pd.DataFrame with additional columns:
        course_history_score   : raw weighted percentile (lower = historically better)
        course_history_adj     : z-score rescaled to sg_composite spread
    sg_composite is updated in-place for qualifying players.
    """
    if field_df.empty:
        return field_df

    venue = _resolve_venue(event_name)
    if venue is None:
        log.warning(
            f"  apply_course_history: no venue mapping for '{event_name}'. "
            "Course history skipped."
        )
        return field_df

    log.info(f"  Course history: '{event_name}' -> venue '{venue}'")

    history_df = _load_venue_history(venue, db)
    if history_df.empty:
        log.warning(f"  No ESPN history found for venue '{venue}'.")
        return field_df

    n_years = history_df["year"].nunique()
    log.info(
        f"  Found {len(history_df)} player-results over {n_years} years at {venue}."
    )

    dg_id_map = _build_dg_id_name_map(db)
    history_scores = _compute_player_history_scores(
        history_df, dg_id_map, min_appearances=min_appearances
    )

    if not history_scores:
        log.warning("  Could not match any historical players to dg_ids.")
        return field_df

    df = field_df.copy()

    # Map history scores onto field using dg_id
    df["course_history_score"] = df["dg_id"].map(history_scores)

    n_matched = df["course_history_score"].notna().sum()
    log.info(
        f"  Course history matched {n_matched}/{len(df)} players "
        f"({min_appearances}+ appearances at {venue})."
    )

    if n_matched < 3:
        log.warning("  Too few history matches. Skipping course history adjustment.")
        return field_df

    # Invert: lower finish_pct = historically better -> higher sg adjustment
    # So we negate: good history -> negative pct -> positive z-score -> composite boost
    mask = df["course_history_score"].notna()

    hist_mean = df.loc[mask, "course_history_score"].mean()
    hist_std  = df.loc[mask, "course_history_score"].std(ddof=1)
    sg_std    = df["sg_composite"].std(ddof=1)

    if hist_std < 1e-9 or sg_std < 1e-9:
        log.warning("  Near-zero variance in history/composite. Skipping.")
        return field_df

    # Negate z-score so lower historical percentile (=better finishes) gives positive adj
    z = -(df.loc[mask, "course_history_score"] - hist_mean) / hist_std
    df["course_history_adj"] = np.nan
    df.loc[mask, "course_history_adj"] = z * sg_std + df["sg_composite"].mean()

    df["sg_composite_pre_history"] = df["sg_composite"]
    df.loc[mask, "sg_composite"] = (
        (1 - alpha) * df.loc[mask, "sg_composite_pre_history"]
        + alpha      * df.loc[mask, "course_history_adj"]
    )

    log.info(
        f"  Course history applied: alpha={alpha:.2f}, "
        f"{n_matched} players adjusted."
    )

    return df
