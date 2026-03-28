"""
model/pre_tournament.py
------------------------
Full pre-tournament prediction pipeline.
Run Tuesday/Wednesday before a PGA Tour event.

Usage:
    python model/pre_tournament.py                        # current week
    python model/pre_tournament.py --event_name "houston"
    python model/pre_tournament.py --top 20
    python model/pre_tournament.py --course "TPC Sawgrass"
    python model/pre_tournament.py --course "Augusta National" --no-recency
"""

import sys
import logging
import argparse
from datetime import datetime

import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import (
    MONGODB_URI, DB_NAME, COLLECTIONS,
    N_SIMULATIONS, RANDOM_SEED, EDGE_THRESHOLDS,
)
from model.sg_composite import SGWeights, build_live_field_scores, get_current_field_ids
from model.course_fit import apply_course_fit
from model.recency import apply_recency
from model.course_history import apply_course_history
from backtester.monte_carlo import simulate_tournament

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]


def load_best_weights() -> SGWeights:
    """Load backtested weights, or use research-based defaults."""
    best = db[COLLECTIONS["backtest_runs"]].find_one({}, sort=[("roi", -1)])
    if best:
        log.info(f"Using backtested weights (ROI={best['roi']:.4f})")
        return SGWeights(best["sg_ott"], best["sg_app"], best["sg_arg"], best["sg_putt"])
    # Research-based defaults: APP dominant, PUTT low
    log.info("Using default weights (no backtest run yet)")
    return SGWeights(sg_ott=0.20, sg_app=0.40, sg_arg=0.25, sg_putt=0.15)


def save_predictions(
    sim_results: pd.DataFrame,
    event_id,
    event_name: str,
    weights: SGWeights,
    course_name: str = None,
):
    """Store predictions in MongoDB."""
    docs = []
    for _, row in sim_results.iterrows():
        doc = {
            "event_id":      event_id,
            "event_name":    event_name,
            "year":          datetime.now().year,
            "dg_id":         int(row.get("dg_id", 0)),
            "player_name":   row.get("player_name"),
            "sg_composite":  float(row.get("sg_composite", 0)),
            "win_prob":      float(row.get("win_prob", 0)),
            "top5_prob":     float(row.get("top5_prob", 0)),
            "top10_prob":    float(row.get("top10_prob", 0)),
            "top20_prob":    float(row.get("top20_prob", 0)),
            "make_cut_prob": float(row.get("make_cut_prob", 0)),
            "dg_win_prob":   float(row.get("dg_win_prob", 0)),
            "weights":       weights.as_dict(),
            "n_sims":        N_SIMULATIONS,
            "generated_at":  datetime.now(),
        }
        # Persist course name if course fit was applied
        if course_name:
            doc["course_name"] = course_name
        # Persist original sg_composite before any adjustments
        if "sg_composite_original" in row:
            doc["sg_composite_original"] = float(row.get("sg_composite_original", 0))
        docs.append(doc)

    if docs:
        db[COLLECTIONS["model_predictions"]].delete_many({"event_id": event_id})
        db[COLLECTIONS["model_predictions"]].insert_many(docs)
        log.info(f"  Saved {len(docs)} predictions to MongoDB")


def run_prediction(
    event_name_filter: str = None,
    top_n: int = 30,
    course_name: str = None,
    use_recency: bool = True,
    weather_date: str = None,
    use_course_history: bool = True,
):
    """
    Full pipeline: load field -> score -> (recency adjust) -> (course adjust)
    -> simulate -> print -> save.

    Parameters
    ----------
    event_name_filter : str, optional
        Partial event name for display confirmation only (does not filter field).
    top_n : int
        Number of players to display in the output table.
    course_name : str, optional
        Course name for course-fit adjustment. If None, course fit is skipped.
    use_recency : bool
        Whether to apply recency-weighted skill adjustment. Default True.
    weather_date : str, optional
        First round date 'YYYY-MM-DD'. If provided (along with course_name),
        fetches wind forecast and adjusts course-fit weights accordingly.
    """

    # Get current field
    dg_ids, event_name = get_current_field_ids()

    if not dg_ids:
        log.error("No field found. Run: python extraction/datagolf_pull.py")
        return pd.DataFrame()

    # Optional name filter (just for display confirmation)
    if event_name_filter and event_name_filter.lower() not in event_name.lower():
        log.warning(f"Current event is '{event_name}', not '{event_name_filter}'")
        log.warning("Running prediction for current event anyway...")

    log.info(f"\nEvent: {event_name}")
    log.info(f"Field size: {len(dg_ids)} players")

    # Load weights
    weights = load_best_weights()
    log.info(f"Weights: {weights}")

    # Build field scores from DataGolf predictions
    field_scores = build_live_field_scores(dg_ids, weights)
    if field_scores.empty:
        log.error("Could not build field scores.")
        return pd.DataFrame()

    # --- Additive adjustments (applied before simulation) ---

    # Recency decay: blend multi-period DataGolf ratings into composite
    if use_recency:
        field_scores = apply_recency(field_scores, db, weights)
        log.info("Recency decay applied")

    # Course fit: adjust composite using venue-specific SG weights
    if course_name:
        avg_wind_mph = 0.0
        if weather_date:
            try:
                from data.weather import get_tournament_wind
                wind_data = get_tournament_wind(course_name, weather_date)
                if wind_data:
                    avg_wind_mph = wind_data["avg_mph"]
                    log.info(
                        f"Weather: {wind_data['condition'].upper()} "
                        f"({avg_wind_mph} mph avg, {wind_data['max_mph']} mph max gusts)"
                    )
                else:
                    log.warning("Weather data unavailable — using course fit weights only.")
            except Exception as exc:
                log.warning(f"Weather fetch failed: {exc}")

        field_scores = apply_course_fit(field_scores, course_name, db,
                                        avg_wind_mph=avg_wind_mph)
        log.info(f"Course fit applied for {course_name}")

    # Course history: reward/fade players based on venue track record
    if use_course_history and event_name:
        field_scores = apply_course_history(field_scores, event_name, db)
        log.info("Course history applied")

    # Run Monte Carlo
    log.info(f"Running {N_SIMULATIONS:,} simulations...")
    sim_results = simulate_tournament(field_scores, n_sims=N_SIMULATIONS, seed=RANDOM_SEED)

    # --- Print results table ---
    separator = "=" * 72
    divider   = "-" * 72

    print(f"\n{separator}")
    print(f"  {event_name}")
    if course_name:
        print(f"  Course: {course_name}")
    adjustments = []
    if use_recency:
        adjustments.append("recency")
    if course_name:
        adjustments.append("course fit")
    adj_label = (f"  Adjustments: {', '.join(adjustments)}" if adjustments
                 else "  Adjustments: none")
    print(adj_label)
    print(f"  Model: {N_SIMULATIONS:,} simulations  |  {len(dg_ids)} players")
    print(separator)
    print(f"{'PLAYER':<28} {'WIN%':>6} {'TOP5%':>6} {'TOP10%':>7} {'TOP20%':>7} {'CUT%':>6} {'SG':>6}")
    print(divider)

    for _, row in sim_results.head(top_n).iterrows():
        name = str(row.get("player_name", "?"))
        # Clean up "Last, First" format to "First Last"
        if "," in name:
            parts = name.split(",", 1)
            name = f"{parts[1].strip()} {parts[0].strip()}"

        print(
            f"{name:<28}"
            f"  {row['win_prob']*100:>5.1f}%"
            f"  {row['top5_prob']*100:>5.1f}%"
            f"  {row['top10_prob']*100:>6.1f}%"
            f"  {row['top20_prob']*100:>6.1f}%"
            f"  {row['make_cut_prob']*100:>5.1f}%"
            f"  {row.get('sg_composite', 0):>+5.2f}"
        )

    print(divider)
    print(f"  Win probs sum: {sim_results['win_prob'].sum()*100:.1f}%")

    # Compare model vs DataGolf where available
    if "dg_win_prob" in sim_results.columns:
        print(f"\n{divider}")
        print("  MODEL vs DATAGOLF WIN PROBABILITY (top 15 divergences)")
        print(divider)
        print(f"  {'PLAYER':<28} {'MODEL':>7} {'DATAGOLF':>9} {'DIFF':>7}")
        diff = sim_results.copy()
        diff["diff"] = diff["win_prob"] - diff["dg_win_prob"]
        diff = diff.reindex(diff["diff"].abs().sort_values(ascending=False).index)
        for _, row in diff.head(15).iterrows():
            name = str(row.get("player_name", "?"))
            if "," in name:
                parts = name.split(",", 1)
                name = f"{parts[1].strip()} {parts[0].strip()}"
            arrow = "^" if row["diff"] > 0 else "v"
            print(f"  {name:<28}  {row['win_prob']*100:>5.1f}%   "
                  f"{row['dg_win_prob']*100:>7.1f}%   "
                  f"{arrow}{abs(row['diff'])*100:>4.1f}%")

    # Resolve event_id: field doc first, then schedule lookup by name
    field_doc = db["dg_current_field"].find_one({}, sort=[("pulled_at", -1)])
    event_id = field_doc.get("event_id") if field_doc else None
    if not event_id:
        sched = db["dg_schedule"].find_one(
            {"event_name": {"$regex": event_name, "$options": "i"}}
        )
        event_id = sched.get("event_id") if sched else None

    save_predictions(sim_results, event_id, event_name, weights, course_name=course_name)

    log.info("\nDone. Run live/odds_scraper.py to compare against bookmaker odds.")
    return sim_results


def main():
    parser = argparse.ArgumentParser(
        description="Run the pre-tournament golf prediction model."
    )
    parser.add_argument(
        "--event_name",
        type=str,
        default=None,
        help="Partial event name for confirmation (does not filter field).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of players to display (default: 30).",
    )
    parser.add_argument(
        "--course",
        type=str,
        default=None,
        help=(
            "Course name to apply course-fit adjustment "
            "(e.g., 'TPC Sawgrass', 'Augusta National')."
        ),
    )
    parser.add_argument(
        "--no-recency",
        action="store_true",
        default=False,
        help="Disable recency-weighted skill adjustment.",
    )
    parser.add_argument(
        "--weather",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help=(
            "First round date for wind forecast (requires --course). "
            "Fetches Open-Meteo forecast and adjusts course-fit weights for wind. "
            "Example: --weather 2026-04-10"
        ),
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        default=False,
        help="Disable course history adjustment.",
    )
    args = parser.parse_args()

    run_prediction(
        event_name_filter=args.event_name,
        top_n=args.top,
        course_name=args.course,
        use_recency=not args.no_recency,
        weather_date=args.weather,
        use_course_history=not args.no_history,
    )


if __name__ == "__main__":
    main()
