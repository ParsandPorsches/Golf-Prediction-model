"""
analysis/course_fit_builder.py
-------------------------------
Builds data-driven course DNA profiles from ESPN historical results
and DataGolf skill ratings stored in MongoDB.

For each course (keyed by tournament name), loads finish positions
(2019-2025) and joins them with current player skill ratings. Computes
Spearman rank correlation between each SG component (OTT/APP/ARG/PUTT)
and finish position. Normalizes absolute correlations into weights that
sum to 1.0 — venues where long hitters win show high OTT weight, etc.

Saves results to MongoDB `course_fit_profiles` collection and prints a
comparison table vs. the hardcoded profiles in model/course_fit.py.

Usage:
    python analysis/course_fit_builder.py
    python analysis/course_fit_builder.py --min-events 2 --min-players 15
    python analysis/course_fit_builder.py --save          # write to MongoDB
    python analysis/course_fit_builder.py --save --compare
"""

import sys
import argparse
import logging
import datetime
from difflib import get_close_matches

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import MONGODB_URI, DB_NAME, COLLECTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SG_COLS = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_espn_results(db) -> pd.DataFrame:
    """
    Pull all ESPN tournament results from MongoDB.

    Returns a long DataFrame with columns:
        event_name, year, player_name, finish, field_size
    Players who missed the cut or withdrew get finish = field_size + 1.
    """
    docs = list(db[COLLECTIONS["espn_results"]].find(
        {},
        {"_id": 0, "event_name": 1, "year": 1, "results": 1}
    ))
    if not docs:
        log.error("No ESPN results found in MongoDB. Run extraction/espn_results.py first.")
        return pd.DataFrame()

    rows = []
    for doc in docs:
        event_name = doc.get("event_name", "Unknown")
        year = doc.get("year")
        results = doc.get("results", [])
        field_size = len(results)
        for r in results:
            finish = r.get("finish")
            made_cut = r.get("made_cut", True)
            withdrew = r.get("withdrew", False)

            # Penalty finish for non-finishers
            if not made_cut or withdrew or finish is None:
                finish = field_size + 1

            rows.append({
                "event_name":  event_name,
                "year":        year,
                "player_name": r.get("player_name", ""),
                "finish":      int(finish),
                "field_size":  field_size,
            })

    df = pd.DataFrame(rows)
    df = df[df["player_name"].str.strip() != ""]
    log.info(f"Loaded {len(df):,} player-results across {df['event_name'].nunique()} events.")
    return df


def _normalize_name(name: str) -> str:
    """
    Convert 'Last, First' -> 'first last' (lowercase).
    If no comma, just lowercase and strip.
    """
    name = name.strip()
    if "," in name:
        parts = name.split(",", 1)
        last = parts[0].strip()
        first = parts[1].strip()
        return f"{first} {last}".lower()
    return name.lower()


def load_skill_ratings(db) -> pd.DataFrame:
    """
    Load SG component ratings from MongoDB dg_skill_ratings.

    DataGolf stores names as 'Last, First'. We normalize to 'first last'
    (lowercase) so they can be fuzzy-matched against ESPN 'First Last' names.

    Returns a DataFrame indexed by normalized player name with columns:
        sg_ott, sg_app, sg_arg, sg_putt
    """
    docs = list(db[COLLECTIONS["skill_ratings"]].find(
        {},
        {"_id": 0, "player_name": 1, "dg_id": 1,
         "sg_ott": 1, "sg_app": 1, "sg_arg": 1, "sg_putt": 1}
    ))
    if not docs:
        log.error("No skill ratings found. Run extraction/datagolf_pull.py first.")
        return pd.DataFrame()

    df = pd.DataFrame(docs).dropna(subset=SG_COLS)
    df["player_name_lower"] = df["player_name"].apply(_normalize_name)
    df = df.set_index("player_name_lower")
    log.info(f"Loaded skill ratings for {len(df):,} players.")
    return df


def build_name_map(espn_names: list, dg_names: list) -> dict:
    """
    Fuzzy-match ESPN player names to DataGolf player names.

    Returns {espn_name_lower: dg_name_lower} dict.
    ESPN and DG both use "First Last" format.
    """
    dg_lower = [n.lower() for n in dg_names]
    name_map = {}

    for name in espn_names:
        key = name.lower().strip()
        if key in dg_lower:
            name_map[key] = key
            continue
        close = get_close_matches(key, dg_lower, n=1, cutoff=0.80)
        if close:
            name_map[key] = close[0]

    matched = sum(1 for v in name_map.values() if v)
    log.info(f"  Name matching: {matched}/{len(espn_names)} ESPN names matched to DG names.")
    return name_map


# ---------------------------------------------------------------------------
# Profile computation
# ---------------------------------------------------------------------------

def percentile_finish(group: pd.DataFrame) -> pd.Series:
    """
    Convert raw finish positions to a 0-1 percentile rank within the field
    (0 = best, 1 = worst). This normalises across different field sizes.
    """
    return group["finish"].rank(method="average", pct=True)


def compute_course_profile(
    joined: pd.DataFrame,
    min_players: int = 15,
) -> dict | None:
    """
    Given a DataFrame with columns [finish_pct, sg_ott, sg_app, sg_arg, sg_putt],
    compute Spearman correlations and convert to normalised weights.

    Returns {"ott": float, "app": float, "arg": float, "putt": float, "n": int}
    or None if not enough data.
    """
    clean = joined[["finish_pct"] + SG_COLS].dropna()
    if len(clean) < min_players:
        return None

    corrs = {}
    for col in SG_COLS:
        rho, _ = spearmanr(clean["finish_pct"], clean[col])
        # Negative rho = higher SG -> better finish -> component matters
        # We want the magnitude of the negative correlation
        corrs[col] = max(0.0, -rho)  # floor at 0 (ignore positive rho)

    total = sum(corrs.values())
    if total < 1e-9:
        # All components uncorrelated — return equal weights
        return {"ott": 0.25, "app": 0.25, "arg": 0.25, "putt": 0.25, "n": len(clean)}

    weights = {col.replace("sg_", ""): round(v / total, 4) for col, v in corrs.items()}
    weights["n"] = len(clean)
    return weights


def build_all_profiles(
    espn_df: pd.DataFrame,
    skill_df: pd.DataFrame,
    min_events: int = 2,
    min_players: int = 15,
) -> dict:
    """
    Iterate over every unique event in the ESPN data and build a course profile.

    Returns {event_name: {ott, app, arg, putt, n, events}} dict.
    """
    # Build name map once
    espn_names = espn_df["player_name"].unique().tolist()
    # skill_df index is already normalized "first last" (lowercase)
    dg_names = skill_df.index.tolist()

    log.info(f"Building name map ({len(espn_names)} ESPN players -> {len(dg_names)} DG players)...")
    name_map = build_name_map(espn_names, dg_names)

    # Add DG name column to ESPN df
    espn_df = espn_df.copy()
    espn_df["dg_name"] = espn_df["player_name"].str.lower().str.strip().map(name_map)

    # Add percentile finish per event-year group
    espn_df["finish_pct"] = (
        espn_df.groupby(["event_name", "year"], group_keys=False)
        .apply(percentile_finish)
    )

    # Join with skill ratings
    joined_full = espn_df.join(skill_df[SG_COLS], on="dg_name", how="left")

    profiles = {}
    grouped = joined_full.groupby("event_name")

    for event_name, group in grouped:
        n_events = group["year"].nunique()
        if n_events < min_events:
            continue

        profile = compute_course_profile(group, min_players=min_players)
        if profile is None:
            continue

        profile["events"] = n_events
        profiles[event_name] = profile
        log.info(
            f"  {event_name[:45]:<45}  "
            f"OTT={profile['ott']:.2f}  APP={profile['app']:.2f}  "
            f"ARG={profile['arg']:.2f}  PUT={profile['putt']:.2f}  "
            f"(n={profile['n']}, years={n_events})"
        )

    log.info(f"\nBuilt {len(profiles)} data-driven course profiles.")
    return profiles


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_profiles_to_mongo(profiles: dict, db) -> None:
    """Upsert data-driven profiles to course_fit_profiles collection."""
    collection = db["course_fit_profiles"]
    collection.create_index("event_name", unique=True)

    for event_name, profile in profiles.items():
        doc = {
            "event_name": event_name,
            "ott":        profile["ott"],
            "app":        profile["app"],
            "arg":        profile["arg"],
            "putt":       profile["putt"],
            "n_players":  profile.get("n", 0),
            "n_events":   profile.get("events", 0),
            "built_at":   datetime.datetime.utcnow(),
            "source":     "espn_spearman",
        }
        collection.update_one(
            {"event_name": event_name},
            {"$set": doc},
            upsert=True,
        )
    log.info(f"Saved {len(profiles)} profiles to MongoDB course_fit_profiles.")


# ---------------------------------------------------------------------------
# Comparison display
# ---------------------------------------------------------------------------

def compare_with_hardcoded(profiles: dict) -> None:
    """Print a side-by-side comparison vs. the hardcoded COURSE_DATABASE."""
    try:
        from model.course_fit import COURSE_DATABASE, get_course_weights
    except ImportError:
        log.warning("Could not import COURSE_DATABASE for comparison.")
        return

    print("\n" + "=" * 100)
    print(f"{'Tournament':<45}  {'Src':>3}  {'OTT':>5}  {'APP':>5}  {'ARG':>5}  {'PUT':>5}  {'n':>5}")
    print("-" * 100)

    for event_name, profile in sorted(profiles.items(), key=lambda x: x[0]):
        print(
            f"{event_name[:44]:<45}  {'DRV':>3}  "
            f"{profile['ott']:.3f}  {profile['app']:.3f}  "
            f"{profile['arg']:.3f}  {profile['putt']:.3f}  "
            f"{profile.get('n', 0):>5}"
        )
        # Try to find matching hardcoded entry
        hw = get_course_weights(event_name)
        if hw:
            print(
                f"{'  ^ hardcoded':<45}  {'HDC':>3}  "
                f"{hw['ott']:.3f}  {hw['app']:.3f}  "
                f"{hw['arg']:.3f}  {hw['putt']:.3f}"
            )

    print("=" * 100)
    print("DRV = data-driven (this script)   HDC = hardcoded (course_fit.py)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build data-driven course DNA profiles from ESPN + DataGolf ratings"
    )
    parser.add_argument("--min-events",  type=int, default=2,
                        help="Minimum distinct years a course must appear (default: 2)")
    parser.add_argument("--min-players", type=int, default=15,
                        help="Minimum matched players per course (default: 15)")
    parser.add_argument("--save",    action="store_true",
                        help="Write profiles to MongoDB course_fit_profiles")
    parser.add_argument("--compare", action="store_true",
                        help="Print comparison vs hardcoded profiles")
    args = parser.parse_args()

    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]

    log.info("=" * 60)
    log.info("Course Fit Builder")
    log.info(f"  min_events  : {args.min_events}")
    log.info(f"  min_players : {args.min_players}")
    log.info("=" * 60)

    espn_df = load_espn_results(db)
    if espn_df.empty:
        return

    skill_df = load_skill_ratings(db)
    if skill_df.empty:
        return

    profiles = build_all_profiles(
        espn_df, skill_df,
        min_events=args.min_events,
        min_players=args.min_players,
    )

    if not profiles:
        log.warning("No profiles built — try lowering --min-events or --min-players.")
        return

    if args.save:
        save_profiles_to_mongo(profiles, db)

    if args.compare:
        compare_with_hardcoded(profiles)

    if not args.save and not args.compare:
        # Default: show profiles
        compare_with_hardcoded(profiles)

    client.close()
    log.info("Done.")


if __name__ == "__main__":
    main()
