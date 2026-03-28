"""
live/sharp_money.py
-------------------
Detects sharp vs. public money divergence using multi-book odds from
The Odds API.

Sharp benchmark: Betfair Exchange (betfair_ex_uk)
  - A betting exchange where bettors wager against each other.
  - No built-in bookmaker margin on the exchange side.
  - Prices reflect true market consensus — used as the "sharp" line.

Recreational books: DraftKings, FanDuel, BetMGM, BetRivers, BetOnline

Logic:
  - Remove vig from each book's odds to get implied probabilities.
  - Compare each recreational book's implied prob vs Betfair's.
  - Large positive gap (recr. > sharp) = public money pushed that player
    shorter at the recreational book = potential fade signal.
  - Large negative gap (recr. < sharp) = sharp money on that player,
    recreational book hasn't moved = potential follow signal.

Also cross-references with our model predictions (MongoDB) to find where
sharp signal AND model edge align — highest conviction plays.

Available events (Odds API free tier):
  - Masters Tournament
  - PGA Championship
  - The Open Championship
  - US Open

Usage:
    python live/sharp_money.py
    python live/sharp_money.py --event masters
    python live/sharp_money.py --event masters --top 30 --min-gap 3
"""

import sys
import os
import argparse
import logging
import datetime
from difflib import get_close_matches

import requests
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")
from config.settings import MONGODB_URI, DB_NAME, COLLECTIONS, ODDS_API_KEY, ODDS_API_BASE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event key map
# ---------------------------------------------------------------------------
EVENT_KEYS = {
    "masters":         "golf_masters_tournament_winner",
    "pga":             "golf_pga_championship_winner",
    "open":            "golf_the_open_championship_winner",
    "usopen":          "golf_us_open_winner",
    "us open":         "golf_us_open_winner",
    "the open":        "golf_the_open_championship_winner",
    "pga championship":"golf_pga_championship_winner",
    "masters tournament": "golf_masters_tournament_winner",
}

SHARP_BOOKS  = {"betfair_ex_uk"}
RECR_BOOKS   = {"draftkings", "fanduel", "betmgm", "betrivers", "betonlineag"}

# Labels for display
BOOK_LABELS = {
    "betfair_ex_uk": "Betfair",
    "draftkings":    "DraftKings",
    "fanduel":       "FanDuel",
    "betmgm":        "BetMGM",
    "betrivers":     "BetRivers",
    "betonlineag":   "BetOnline",
}


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------

def american_to_implied(american: int | float) -> float:
    """Convert American odds to raw implied probability (no vig removed)."""
    if american > 0:
        return 100 / (american + 100)
    else:
        return abs(american) / (abs(american) + 100)


def remove_vig(probs: list[float]) -> list[float]:
    """Normalize a list of implied probs so they sum to 1.0."""
    total = sum(probs)
    if total < 1e-9:
        return probs
    return [p / total for p in probs]


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_odds(event_key: str) -> dict | None:
    """
    Fetch all book odds for a golf outright market from The Odds API.
    Returns raw API response dict or None on failure.
    """
    url = f"{ODDS_API_BASE}/sports/{event_key}/odds"
    params = {
        "apiKey":      ODDS_API_KEY,
        "regions":     "us,uk",
        "markets":     "outrights",
        "oddsFormat":  "american",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"  Odds API requests remaining: {remaining}")
        resp.raise_for_status()
        data = resp.json()
        if not data:
            log.warning("  No events returned from Odds API.")
            return None
        return data[0]   # First (and only) event
    except requests.RequestException as exc:
        log.error(f"  Odds API request failed: {exc}")
        return None


def parse_book_probs(event_data: dict) -> dict[str, dict[str, float]]:
    """
    Parse Odds API event dict into {book_key: {player_name: raw_implied_prob}}.

    Uses raw (vig-included) implied probabilities for each player.
    This keeps comparisons consistent regardless of field size per book —
    we only care about the relative gap between books, not exact fair values.
    Duplicate entries (Betfair back/lay) are averaged per player.
    """
    result = {}
    for book in event_data.get("bookmakers", []):
        key = book["key"]
        player_probs: dict[str, list[float]] = {}
        for market in book.get("markets", []):
            if market.get("key") != "outrights":
                continue
            for outcome in market.get("outcomes", []):
                name  = outcome["name"]
                price = outcome["price"]
                prob  = american_to_implied(price)
                player_probs.setdefault(name, []).append(prob)
        # Average duplicates (e.g. Betfair back/lay spread)
        result[key] = {name: sum(probs) / len(probs)
                       for name, probs in player_probs.items()}
    return result


# ---------------------------------------------------------------------------
# Sharp money analysis
# ---------------------------------------------------------------------------

def compute_divergence(book_probs: dict[str, dict[str, float]],
                       min_gap: float = 2.5) -> pd.DataFrame:
    """
    Build a DataFrame of per-player sharp vs. recreational divergence.

    Columns:
        player          : str
        betfair_prob    : float  (no-vig implied, sharp benchmark)
        recr_avg_prob   : float  (average across recreational books)
        best_recr_prob  : float  (recreational book with highest implied prob)
        best_recr_book  : str
        gap_pct         : float  (+ve = public pushed recr shorter = fade signal)
                                 (-ve = sharp shorter = follow signal)
        signal          : str    "FADE" / "FOLLOW" / "NEUTRAL"
    """
    # Get Betfair as sharp baseline (average both betfair entries if duplicated)
    sharp_probs: dict[str, float] = {}
    sharp_count: dict[str, int]   = {}
    for book_key in SHARP_BOOKS:
        if book_key in book_probs:
            for player, prob in book_probs[book_key].items():
                sharp_probs[player]  = sharp_probs.get(player, 0) + prob
                sharp_count[player]  = sharp_count.get(player, 0) + 1

    if not sharp_probs:
        log.error("  No sharp book (Betfair) data found.")
        return pd.DataFrame()

    sharp_avg = {p: sharp_probs[p] / sharp_count[p] for p in sharp_probs}

    # Collect recreational book probs
    rows = []
    for player, sharp_p in sharp_avg.items():
        recr_vals = []
        recr_book_probs = {}
        for book_key in RECR_BOOKS:
            if book_key in book_probs and player in book_probs[book_key]:
                p = book_probs[book_key][player]
                recr_vals.append(p)
                recr_book_probs[book_key] = p

        if not recr_vals:
            continue

        recr_avg = sum(recr_vals) / len(recr_vals)
        best_book = max(recr_book_probs, key=recr_book_probs.get)
        best_prob = recr_book_probs[best_book]

        # gap: recr_avg - sharp (positive = public pushed recr shorter)
        gap = (recr_avg - sharp_p) * 100   # in percentage points

        if gap > min_gap:
            signal = "FADE"      # public inflated this player at recreational books
        elif gap < -min_gap:
            signal = "FOLLOW"    # sharp money on this player, recr not caught up
        else:
            signal = "neutral"

        rows.append({
            "player":         player,
            "betfair_prob":   round(sharp_p * 100, 2),
            "recr_avg_prob":  round(recr_avg * 100, 2),
            "best_recr_book": BOOK_LABELS.get(best_book, best_book),
            "gap_pct":        round(gap, 2),
            "signal":         signal,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("gap_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cross-reference with model predictions
# ---------------------------------------------------------------------------

def load_model_predictions(db) -> pd.DataFrame:
    """Load most recent model predictions from MongoDB."""
    docs = list(db[COLLECTIONS["model_predictions"]].find(
        {}, {"_id": 0, "player_name": 1, "win_prob": 1, "generated_at": 1}
    ).sort("generated_at", -1).limit(500))
    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    # Keep only most recent run (group by player_name, take latest)
    df = df.sort_values("generated_at", ascending=False).drop_duplicates("player_name")
    df["model_prob"] = (df["win_prob"] * 100).round(2)
    return df[["player_name", "model_prob"]]


def merge_model_signal(div_df: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to join model predictions onto divergence table by player name.
    Uses fuzzy matching for name format differences.
    """
    if model_df.empty:
        div_df["model_prob"] = None
        return div_df

    model_names = model_df["player_name"].str.lower().tolist()

    def match_name(odds_name: str) -> float | None:
        key = odds_name.lower()
        # Try direct match
        if key in model_names:
            idx = model_names.index(key)
            return model_df.iloc[idx]["model_prob"]
        # Try "Last, First" -> "First Last" normalization from model
        for i, mn in enumerate(model_names):
            if "," in mn:
                parts = mn.split(",", 1)
                normalized = f"{parts[1].strip()} {parts[0].strip()}"
                if normalized == key:
                    return model_df.iloc[i]["model_prob"]
        # Fuzzy match
        close = get_close_matches(key, model_names, n=1, cutoff=0.82)
        if close:
            idx = model_names.index(close[0])
            return model_df.iloc[idx]["model_prob"]
        return None

    div_df = div_df.copy()
    div_df["model_prob"] = div_df["player"].apply(match_name)

    # Model edge vs Betfair (positive = model thinks they're underpriced by sharp market)
    mask = div_df["model_prob"].notna()
    div_df.loc[mask, "model_vs_sharp"] = (
        div_df.loc[mask, "model_prob"] - div_df.loc[mask, "betfair_prob"]
    ).round(2)

    return div_df


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame, event_title: str, top_n: int, min_gap: float):
    print()
    print("=" * 90)
    print(f"  Sharp Money Report — {event_title}")
    print(f"  Betfair Exchange (sharp) vs Recreational Book Average")
    print(f"  FADE = public money shortened recreational odds | FOLLOW = sharp money signal")
    print("=" * 90)

    has_model = "model_prob" in df.columns and df["model_prob"].notna().any()

    header = (
        f"{'Player':<28}  {'Betfair':>8}  {'Recr Avg':>8}  "
        f"{'Gap':>7}  {'Signal':<8}"
    )
    if has_model:
        header += f"  {'Model':>6}  {'vs Sharp':>8}"
    sep = "-" * (90 if not has_model else 110)

    def fmt_row(row):
        signal = row["signal"]
        tag = "<<" if signal == "FOLLOW" else (">>" if signal == "FADE" else "  ")
        line = (
            f"{tag} {row['player'][:26]:<27}  "
            f"{row['betfair_prob']:>7.1f}%  "
            f"{row['recr_avg_prob']:>7.1f}%  "
            f"{row['gap_pct']:>+6.1f}pp  "
            f"{signal:<8}"
        )
        if has_model and pd.notna(row.get("model_prob")):
            vs = row.get("model_vs_sharp", 0) or 0
            line += f"  {row['model_prob']:>5.1f}%  {vs:>+7.2f}pp"
        return line

    # Full ranked table (most public-inflated at top, sharpest follows at bottom)
    print(f"\n  {'Ranked by gap (>> FADE = public inflated | << FOLLOW = sharp money)'}")
    print(header)
    print(sep)
    for _, row in df[df["betfair_prob"] >= 0.5].head(top_n).iterrows():
        print(fmt_row(row))

    # High conviction: model edge + sharp signal alignment
    if has_model and "model_vs_sharp" in df.columns:
        conviction = df[
            (df["signal"] == "FOLLOW") &
            (df["model_vs_sharp"].notna()) &
            (df["model_vs_sharp"] > 0.5)
        ].sort_values("model_vs_sharp", ascending=False).head(8)

        if not conviction.empty:
            print(f"\n  -- HIGH CONVICTION: Sharp FOLLOW + Model Edge --")
            print(f"  (sharp money AND our model both like these players)")
            print(header)
            print(sep)
            for _, row in conviction.iterrows():
                print(fmt_row(row))

    print("=" * 90)
    print(f"  Gap = Recreational avg implied% - Betfair implied% (both no-vig)")
    print(f"  Positive gap = public money pushed recreational book shorter than sharp market")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sharp money divergence analysis using multi-book odds"
    )
    parser.add_argument(
        "--event", type=str, default="masters",
        help="Event shortcode: masters / pga / open / usopen (default: masters)"
    )
    parser.add_argument("--top",     type=int,   default=20,
                        help="Players to show per section (default: 20)")
    parser.add_argument("--min-gap", type=float, default=2.0,
                        help="Minimum gap pp to flag (default: 2.0)")
    parser.add_argument("--no-model", action="store_true",
                        help="Skip loading model predictions from MongoDB")
    args = parser.parse_args()

    event_key = EVENT_KEYS.get(args.event.lower())
    if not event_key:
        log.error(f"Unknown event '{args.event}'. Use: masters / pga / open / usopen")
        return

    log.info(f"Fetching odds for: {args.event}")
    event_data = fetch_odds(event_key)
    if not event_data:
        return

    event_title = event_data.get("sport_title", args.event.title())
    books_found = [b["key"] for b in event_data.get("bookmakers", [])]
    log.info(f"  Books: {books_found}")

    book_probs = parse_book_probs(event_data)
    div_df = compute_divergence(book_probs, min_gap=args.min_gap)

    if div_df.empty:
        log.error("No divergence data computed.")
        return

    if not args.no_model:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        model_df = load_model_predictions(db)
        if not model_df.empty:
            div_df = merge_model_signal(div_df, model_df)
            log.info(
                f"  Model predictions matched for "
                f"{div_df['model_prob'].notna().sum()}/{len(div_df)} players."
            )
        client.close()

    print_report(div_df, event_title, top_n=args.top, min_gap=args.min_gap)


if __name__ == "__main__":
    main()
