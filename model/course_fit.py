"""
model/course_fit.py
-------------------
Course DNA database for PGA Tour venues and a function that adjusts
sg_composite for course fit.

Each course entry stores SG component weights (ott, app, arg, putt) that
reflect the relative importance of each skill category at that venue.
Weights sum to 1.0.

Usage:
    from model.course_fit import apply_course_fit, get_course_weights
    field_df = apply_course_fit(field_df, "TPC Sawgrass", db)
"""

import sys
import logging
import difflib

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config.settings import COLLECTIONS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Course DNA Database
# ---------------------------------------------------------------------------
# Keys are canonical course names. Values are dicts with keys:
#   ott  - strokes gained off the tee weight
#   app  - strokes gained approach weight
#   arg  - strokes gained around the green weight
#   putt - strokes gained putting weight
# All four weights sum to exactly 1.0 for each course.
# ---------------------------------------------------------------------------

COURSE_DATABASE = {
    # Tight bermuda, short par-70 — approach and short game dominate
    "Memorial Park": {
        "aliases": ["Memorial Park", "Houston Open", "Texas Children's Houston Open"],
        "ott": 0.05, "app": 0.45, "arg": 0.30, "putt": 0.20,
        "notes": "tight bermuda, short par-70",
    },

    # Long, sloped greens — balanced but approach and distance matter
    "Augusta National": {
        "aliases": ["Augusta National", "Masters", "Augusta"],
        "ott": 0.20, "app": 0.40, "arg": 0.20, "putt": 0.20,
        "notes": "long, sloped greens",
    },

    # Coastal wind, firm conditions — approach under pressure
    "Pebble Beach": {
        "aliases": ["Pebble Beach", "AT&T Pebble Beach", "ATNT Pebble Beach",
                    "Pebble Beach Pro-Am"],
        "ott": 0.15, "app": 0.40, "arg": 0.25, "putt": 0.20,
        "notes": "coastal wind, firm",
    },

    # Tight, water, approach-heavy — accuracy over distance
    "TPC Sawgrass": {
        "aliases": ["TPC Sawgrass", "Players Championship", "The Players",
                    "Players", "Sawgrass"],
        "ott": 0.10, "app": 0.45, "arg": 0.25, "putt": 0.20,
        "notes": "tight, water, approach",
    },

    # Long rough, approach crucial — premium on iron play
    "Riviera": {
        "aliases": ["Riviera", "Genesis", "Genesis Invitational",
                    "LA Open", "Los Angeles CC"],
        "ott": 0.15, "app": 0.45, "arg": 0.25, "putt": 0.15,
        "notes": "long rough, approach crucial",
    },

    # Long, rough, distance — US Open setup
    "Torrey Pines": {
        "aliases": ["Torrey Pines", "Farmers Insurance", "US Open Torrey"],
        "ott": 0.25, "app": 0.40, "arg": 0.20, "putt": 0.15,
        "notes": "long, rough, distance",
    },

    # Long, windy — Arnold Palmer layout rewards length
    "Bay Hill": {
        "aliases": ["Bay Hill", "Arnold Palmer", "Arnold Palmer Invitational",
                    "Bay Hill Club"],
        "ott": 0.20, "app": 0.40, "arg": 0.20, "putt": 0.20,
        "notes": "long, windy",
    },

    # Long, demanding approach — Jack's course
    "Muirfield Village": {
        "aliases": ["Muirfield Village", "Memorial Tournament", "Memorial",
                    "Jack Memorial"],
        "ott": 0.20, "app": 0.45, "arg": 0.20, "putt": 0.15,
        "notes": "long, demanding approach",
    },

    # Accuracy over distance — putting also rewarded
    "Colonial": {
        "aliases": ["Colonial", "Charles Schwab", "Colonial Country Club",
                    "Charles Schwab Challenge"],
        "ott": 0.05, "app": 0.40, "arg": 0.30, "putt": 0.25,
        "notes": "accuracy over distance",
    },

    # Long, demanding — big hitters have advantage
    "Quail Hollow": {
        "aliases": ["Quail Hollow", "Wells Fargo", "Wells Fargo Championship",
                    "PGA Championship Quail"],
        "ott": 0.25, "app": 0.40, "arg": 0.20, "putt": 0.15,
        "notes": "long, demanding",
    },

    # Long, tough greens — Tour Championship conditions
    "East Lake": {
        "aliases": ["East Lake", "Tour Championship", "Tour Champ", "FedEx Cup Final"],
        "ott": 0.20, "app": 0.40, "arg": 0.20, "putt": 0.20,
        "notes": "long, tough greens",
    },

    # Long, birdie fest — scoring environment rewards length
    "Kapalua": {
        "aliases": ["Kapalua", "Sentry", "Sentry Tournament", "Plantation Course",
                    "Sentry TOC"],
        "ott": 0.25, "app": 0.35, "arg": 0.20, "putt": 0.20,
        "notes": "long, birdie fest, scoring",
    },

    # Birdie fest, putting matters — party stadium hole
    "TPC Scottsdale": {
        "aliases": ["TPC Scottsdale", "WM Phoenix", "Waste Management Phoenix",
                    "Phoenix Open", "WM Phoenix Open", "Phoenix"],
        "ott": 0.10, "app": 0.30, "arg": 0.25, "putt": 0.35,
        "notes": "birdie fest, putting matters",
    },

    # Short, accuracy, putting — classic Hilton Head layout
    "Harbour Town": {
        "aliases": ["Harbour Town", "RBC Heritage", "Heritage", "Hilton Head",
                    "RBC Heritage Classic"],
        "ott": 0.05, "app": 0.35, "arg": 0.30, "putt": 0.30,
        "notes": "short, accuracy, putting",
    },

    # Short, accuracy, bermuda — end of season venue
    "Sedgefield": {
        "aliases": ["Sedgefield", "Wyndham", "Wyndham Championship",
                    "Sedgefield Country Club"],
        "ott": 0.05, "app": 0.35, "arg": 0.35, "putt": 0.25,
        "notes": "short, accuracy, bermuda",
    },

    # Scoring, putting — FedEx playoffs first stop
    "TPC Twin Cities": {
        "aliases": ["TPC Twin Cities", "3M Open", "3M", "Twin Cities",
                    "TPC Twin Cities Open"],
        "ott": 0.10, "app": 0.35, "arg": 0.25, "putt": 0.30,
        "notes": "scoring, putting",
    },

    # Long, brutal rough — US Open at its hardest
    "Oakmont": {
        "aliases": ["Oakmont", "US Open Oakmont", "Oakmont Country Club"],
        "ott": 0.25, "app": 0.45, "arg": 0.20, "putt": 0.10,
        "notes": "long, brutal rough",
    },

    # Long, rough — typical major setup
    "Bethpage Black": {
        "aliases": ["Bethpage Black", "Bethpage", "PGA Bethpage",
                    "PGA Championship Bethpage"],
        "ott": 0.25, "app": 0.40, "arg": 0.20, "putt": 0.15,
        "notes": "long, rough",
    },

    # Long, approach — BMW Championship venue
    "Wilmington CC": {
        "aliases": ["Wilmington CC", "BMW Championship", "BMW Wilmington",
                    "Wilmington Country Club"],
        "ott": 0.20, "app": 0.40, "arg": 0.25, "putt": 0.15,
        "notes": "long, approach",
    },

    # Long, scoring — Deutsche Bank / Northern Trust history
    "TPC Boston": {
        "aliases": ["TPC Boston", "Deutsche Bank", "Northern Trust TPC",
                    "TPC Boston Championship"],
        "ott": 0.20, "app": 0.38, "arg": 0.22, "putt": 0.20,
        "notes": "long, scoring",
    },
}

# Build a flat lookup: lowercase alias -> canonical key
_ALIAS_MAP: dict[str, str] = {}
for _canonical, _data in COURSE_DATABASE.items():
    for _alias in _data.get("aliases", []):
        _ALIAS_MAP[_alias.lower()] = _canonical


def _get_weights_from_mongo(course_name: str, db) -> dict | None:
    """
    Look up data-driven course weights from the course_fit_profiles MongoDB
    collection (populated by analysis/course_fit_builder.py).

    Tries exact match first, then difflib fuzzy match (cutoff=0.65).
    Returns {"ott", "app", "arg", "putt"} or None.
    """
    if db is None:
        return None

    try:
        collection = db["course_fit_profiles"]
        query_lower = course_name.strip().lower()

        # Load all stored event names for matching
        all_docs = list(collection.find({}, {"_id": 0, "event_name": 1,
                                             "ott": 1, "app": 1, "arg": 1, "putt": 1}))
        if not all_docs:
            return None

        names_lower = [d["event_name"].lower() for d in all_docs]

        # Exact match
        if query_lower in names_lower:
            idx = names_lower.index(query_lower)
            d = all_docs[idx]
            log.info(f"  Course fit: '{course_name}' -> MongoDB profile (exact).")
            return {"ott": d["ott"], "app": d["app"], "arg": d["arg"], "putt": d["putt"]}

        # Fuzzy match
        close = difflib.get_close_matches(query_lower, names_lower, n=1, cutoff=0.80)
        if close:
            idx = names_lower.index(close[0])
            d = all_docs[idx]
            log.info(
                f"  Course fit: '{course_name}' -> MongoDB profile "
                f"(fuzzy via '{all_docs[idx]['event_name']}')."
            )
            return {"ott": d["ott"], "app": d["app"], "arg": d["arg"], "putt": d["putt"]}

    except Exception as exc:
        log.debug(f"  MongoDB course profile lookup failed: {exc}")

    return None


def get_course_weights(course_name: str, db=None) -> dict | None:
    """
    Return SG component weights for a given course.

    Lookup order:
      1. MongoDB course_fit_profiles (data-driven, if db provided and populated)
      2. Hardcoded COURSE_DATABASE (exact alias match)
      3. Hardcoded COURSE_DATABASE (difflib fuzzy match, cutoff=0.60)

    Returns a dict {"ott": float, "app": float, "arg": float, "putt": float}
    or None if no match is found.
    """
    if not course_name or not course_name.strip():
        return None

    # 1. Data-driven MongoDB profile
    mongo_weights = _get_weights_from_mongo(course_name, db)
    if mongo_weights is not None:
        return mongo_weights

    query = course_name.strip().lower()

    # 2. Exact alias match against hardcoded database
    if query in _ALIAS_MAP:
        canonical = _ALIAS_MAP[query]
        data = COURSE_DATABASE[canonical]
        log.info(f"  Course match: '{course_name}' -> '{canonical}' (hardcoded exact)")
        return {"ott": data["ott"], "app": data["app"],
                "arg": data["arg"], "putt": data["putt"]}

    # 3. Fuzzy match across all aliases
    all_aliases = list(_ALIAS_MAP.keys())
    close = difflib.get_close_matches(query, all_aliases, n=1, cutoff=0.60)
    if close:
        canonical = _ALIAS_MAP[close[0]]
        data = COURSE_DATABASE[canonical]
        log.info(f"  Course match: '{course_name}' -> '{canonical}' (hardcoded fuzzy via '{close[0]}')")
        return {"ott": data["ott"], "app": data["app"],
                "arg": data["arg"], "putt": data["putt"]}

    log.warning(f"  No course match found for '{course_name}'. Course fit will not be applied.")
    return None


def _fetch_skill_components_from_mongo(dg_ids: list, db) -> pd.DataFrame:
    """
    Load sg_ott, sg_app, sg_arg, sg_putt from the dg_skill_ratings collection
    for a given list of dg_ids.

    Returns a DataFrame indexed by dg_id with the four SG columns.
    """
    collection_name = "dg_skill_ratings"
    docs = list(db[collection_name].find(
        {"dg_id": {"$in": dg_ids}},
        {"_id": 0, "dg_id": 1, "sg_ott": 1, "sg_app": 1, "sg_arg": 1, "sg_putt": 1}
    ))
    if not docs:
        return pd.DataFrame(columns=["dg_id", "sg_ott", "sg_app", "sg_arg", "sg_putt"])
    return pd.DataFrame(docs).set_index("dg_id")


def apply_course_fit(
    field_df: pd.DataFrame,
    course_name: str,
    db,
    alpha: float = 0.30,
    avg_wind_mph: float = 0.0,
) -> pd.DataFrame:
    """
    Adjust sg_composite for course fit by blending a course-specific composite
    with the existing sg_composite score. Optionally further adjusts weights
    for wind conditions.

    Parameters
    ----------
    field_df : pd.DataFrame
        Must contain columns: dg_id, player_name, sg_composite.
        Optionally contains: sg_ott, sg_app, sg_arg, sg_putt (from skill ratings).
    course_name : str
        Human-readable course name, fuzzy-matched against the database.
    db : pymongo.database.Database
        Active MongoDB connection for skill rating lookups.
    alpha : float
        Blend weight for course fit. 0.0 = no adjustment, 1.0 = pure course fit.
        Default 0.30 means 70% original composite + 30% course-fit adjustment.
    avg_wind_mph : float
        Average wind speed during playing hours (mph). When > 8 mph, weights are
        blended toward approach-heavy profile. 0.0 = no wind adjustment.

    Returns
    -------
    pd.DataFrame
        field_df with additional columns:
            - sg_composite_original : original sg_composite before adjustment
            - course_fit_score      : raw weighted composite for this course
            - course_fit_adj        : course-fit z-score rescaled to sg_composite scale
        The sg_composite column is updated in-place.

    Notes
    -----
    If the course is not in the database, field_df is returned unchanged with
    a warning logged. If SG components are not present in field_df, they are
    loaded from MongoDB's dg_skill_ratings collection.
    """
    if field_df.empty:
        log.warning("  apply_course_fit: received empty DataFrame, skipping.")
        return field_df

    # Resolve course weights — try MongoDB data-driven first, then hardcoded
    weights = get_course_weights(course_name, db=db)
    if weights is None:
        log.warning(f"  Course '{course_name}' not in database. Returning field unchanged.")
        return field_df

    # Apply wind adjustment if wind data provided
    if avg_wind_mph > 0:
        from data.weather import wind_adjust_weights
        orig_weights = dict(weights)
        weights = wind_adjust_weights(weights, avg_wind_mph)
        if weights != orig_weights:
            log.info(
                f"  Wind adjustment ({avg_wind_mph:.1f} mph): "
                f"OTT {orig_weights['ott']:.3f}->{weights['ott']:.3f}  "
                f"APP {orig_weights['app']:.3f}->{weights['app']:.3f}  "
                f"ARG {orig_weights['arg']:.3f}->{weights['arg']:.3f}  "
                f"PUT {orig_weights['putt']:.3f}->{weights['putt']:.3f}"
            )

    df = field_df.copy()
    sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
    has_components = all(c in df.columns for c in sg_cols)

    if not has_components:
        # Attempt to load from MongoDB
        log.info("  SG components not in field_df — loading from dg_skill_ratings.")
        ids_needed = df["dg_id"].dropna().astype(int).tolist()
        skill_df = _fetch_skill_components_from_mongo(ids_needed, db)

        if skill_df.empty:
            log.warning("  No skill ratings found in MongoDB. Course fit cannot be computed.")
            return field_df

        df = df.join(skill_df, on="dg_id", how="left")
        fetched = df[sg_cols].notna().all(axis=1).sum()
        log.info(f"  Loaded SG components for {fetched}/{len(df)} players from MongoDB.")

    # Compute raw course-fit composite for each player
    # Rows with any missing SG component are excluded from course adjustment
    component_mask = df[sg_cols].notna().all(axis=1)
    n_adjustable = component_mask.sum()

    if n_adjustable < 5:
        log.warning(
            f"  Only {n_adjustable} players have complete SG components. "
            "Skipping course fit."
        )
        return field_df

    df["course_fit_score"] = np.nan
    df.loc[component_mask, "course_fit_score"] = (
        weights["ott"]  * df.loc[component_mask, "sg_ott"]  +
        weights["app"]  * df.loc[component_mask, "sg_app"]  +
        weights["arg"]  * df.loc[component_mask, "sg_arg"]  +
        weights["putt"] * df.loc[component_mask, "sg_putt"]
    )

    # Normalize course_fit_score relative to field mean/std (z-score among
    # the players for whom we computed it)
    fit_mean = df.loc[component_mask, "course_fit_score"].mean()
    fit_std  = df.loc[component_mask, "course_fit_score"].std(ddof=1)

    if fit_std < 1e-9:
        log.warning("  course_fit_score has near-zero variance. Returning field unchanged.")
        return field_df

    # Rescale z-score to match the scale of sg_composite so that the blend
    # is numerically meaningful.
    sg_std = df["sg_composite"].std(ddof=1)
    if sg_std < 1e-9:
        log.warning("  sg_composite has near-zero variance. Returning field unchanged.")
        return field_df

    z_scores = (df.loc[component_mask, "course_fit_score"] - fit_mean) / fit_std
    # Rescale to same spread as sg_composite
    df["course_fit_adj"] = np.nan
    df.loc[component_mask, "course_fit_adj"] = (
        z_scores * sg_std + df["sg_composite"].mean()
    )

    # Preserve original composite
    df["sg_composite_original"] = df["sg_composite"]

    # Blend: adjusted = (1-alpha)*sg_composite + alpha*course_fit_adj
    # Only update rows where course_fit_adj is available
    df.loc[component_mask, "sg_composite"] = (
        (1 - alpha) * df.loc[component_mask, "sg_composite_original"]
        + alpha * df.loc[component_mask, "course_fit_adj"]
    )

    log.info(
        f"  Course fit applied for '{course_name}' "
        f"(alpha={alpha:.2f}, {n_adjustable} players adjusted)."
    )

    return df
