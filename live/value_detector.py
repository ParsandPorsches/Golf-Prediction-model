"""
live/value_detector.py
-----------------------
Compares model predictions to live odds and identifies value bets.
Sends Discord alerts when edges appear above threshold.

Usage:
    python live/value_detector.py --event_id 28 --year 2025
"""

import sys
import logging
import argparse
from datetime import datetime

import pandas as pd
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import (
    MONGODB_URI, DB_NAME, COLLECTIONS, EDGE_THRESHOLDS,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]


# ══════════════════════════════════════════════════════════════════════════════
# Load model predictions + live odds
# ══════════════════════════════════════════════════════════════════════════════

def load_model_predictions(event_id: int, year: int) -> pd.DataFrame:
    """Load our model's pre-tournament predictions from MongoDB."""
    docs = list(db[COLLECTIONS["model_predictions"]].find(
        {"event_id": event_id, "year": year},
        {"dg_id": 1, "player_name": 1,
         "win_prob": 1, "top5_prob": 1, "top10_prob": 1,
         "top20_prob": 1, "make_cut_prob": 1}
    ))
    return pd.DataFrame(docs)


def load_live_odds(event_id: int) -> pd.DataFrame:
    """Load current live odds from MongoDB."""
    docs = list(db[COLLECTIONS["live_odds"]].find({"event_id": event_id}))
    if not docs:
        return pd.DataFrame()

    rows = []
    for doc in docs:
        market = doc.get("market")
        book   = doc.get("book")
        for player in doc.get("odds", []):
            rows.append({
                "dg_id":        player.get("dg_id"),
                "player_name":  player.get("player_name"),
                "market":       market,
                "book":         book,
                "decimal_odds": player.get("decimal_odds"),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Remove vig per market/book
    implied_no_vig = []
    for (market_key, book_key), grp in df.groupby(["market", "book"]):
        imp = 1.0 / grp["decimal_odds"].clip(lower=1.001)
        total = imp.sum()
        implied_no_vig.append(imp / total if total > 0 else imp)

    df["implied_no_vig"] = pd.concat(implied_no_vig).reindex(df.index)
    return df.dropna(subset=["dg_id", "decimal_odds"])


# ══════════════════════════════════════════════════════════════════════════════
# Value detection
# ══════════════════════════════════════════════════════════════════════════════

MARKET_MODEL_COL = {
    "win":      "win_prob",
    "top_5":    "top5_prob",
    "top_10":   "top10_prob",
    "top_20":   "top20_prob",
    "make_cut": "make_cut_prob",
}


def detect_value(
    predictions: pd.DataFrame,
    live_odds: pd.DataFrame,
    thresholds: dict = EDGE_THRESHOLDS,
) -> pd.DataFrame:
    """
    Cross-reference model predictions with no-vig odds.
    Returns all bets with edge above threshold.
    """
    if predictions.empty or live_odds.empty:
        return pd.DataFrame()

    value_bets = []

    for _, odds_row in live_odds.iterrows():
        dg_id  = odds_row.get("dg_id")
        market = odds_row.get("market")

        if not dg_id or not market:
            continue

        model_col = MARKET_MODEL_COL.get(market)
        if not model_col:
            continue

        player_pred = predictions[predictions["dg_id"] == dg_id]
        if player_pred.empty:
            continue

        model_prob = float(player_pred[model_col].iloc[0])
        book_prob  = float(odds_row.get("implied_no_vig", 0))
        edge       = model_prob - book_prob
        threshold  = thresholds.get(market, 0.05)

        if edge <= threshold:
            continue

        decimal_odds = float(odds_row.get("decimal_odds", 0))
        fair_odds    = round(1.0 / model_prob, 2) if model_prob > 0 else None

        # Convert to American odds for reference
        american_odds = decimal_to_american(decimal_odds)

        value_bets.append({
            "player_name":   odds_row.get("player_name") or player_pred["player_name"].iloc[0],
            "dg_id":         dg_id,
            "market":        market,
            "model_prob":    round(model_prob, 4),
            "book_prob":     round(book_prob, 4),
            "edge":          round(edge, 4),
            "decimal_odds":  decimal_odds,
            "american_odds": american_odds,
            "fair_odds":     fair_odds,
            "book":          odds_row.get("book"),
            "detected_at":   datetime.utcnow(),
        })

    if not value_bets:
        return pd.DataFrame()

    df = pd.DataFrame(value_bets)
    return df.sort_values("edge", ascending=False).reset_index(drop=True)


def decimal_to_american(decimal_odds: float) -> str:
    """Convert decimal odds to American (+/- format)."""
    if decimal_odds <= 1.0:
        return "N/A"
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    else:
        return f"{int(-100 / (decimal_odds - 1))}"


# ══════════════════════════════════════════════════════════════════════════════
# Kelly sizing
# ══════════════════════════════════════════════════════════════════════════════

def kelly_fraction(model_prob: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """
    Fractional Kelly bet size as % of bankroll.
    fraction = 0.25 → quarter-Kelly (conservative)
    """
    b = decimal_odds - 1  # net odds (profit per unit staked)
    q = 1 - model_prob
    full_kelly = (b * model_prob - q) / b
    return max(0.0, full_kelly * fraction)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_value_scan(event_id: int, year: int, send_discord: bool = True):
    log.info(f"Value scan: event_id={event_id}, year={year}")

    predictions = load_model_predictions(event_id, year)
    if predictions.empty:
        log.error("No model predictions found. Run model/pre_tournament.py first.")
        return

    live_odds = load_live_odds(event_id)
    if live_odds.empty:
        log.error("No live odds found. Run live/odds_scraper.py first.")
        return

    value_bets = detect_value(predictions, live_odds)

    if value_bets.empty:
        log.info("No value bets found above threshold.")
        return

    # Add Kelly sizing
    value_bets["kelly_pct"] = value_bets.apply(
        lambda r: kelly_fraction(r["model_prob"], r["decimal_odds"]), axis=1
    )

    print(f"\n{'='*75}")
    print(f"VALUE BETS --- Event {event_id} / {year}")
    print(f"{'='*75}")
    for _, bet in value_bets.iterrows():
        print(
            f"\n  {bet['player_name']:<28} | {bet['market'].upper():<10}"
            f"\n  Model: {bet['model_prob']*100:.1f}%  |  "
            f"Book: {bet['book_prob']*100:.1f}%  |  "
            f"Edge: +{bet['edge']*100:.1f}%"
            f"\n  Odds: {bet['decimal_odds']:.2f} ({bet['american_odds']})  |  "
            f"Fair: {bet['fair_odds']:.2f}  |  "
            f"Book: {bet['book'].upper()}  |  "
            f"Kelly: {bet['kelly_pct']*100:.1f}% bankroll"
        )

    # Send Discord alert
    if send_discord:
        from live.discord_alerts import send_value_bets_alert
        send_value_bets_alert(value_bets, event_id, year)

    # Store in bet_log (as pending/paper bets)
    for _, bet in value_bets.iterrows():
        fk = {"event_id": event_id, "dg_id": bet["dg_id"], "market": bet["market"],
              "book": bet["book"]}
        doc = {**fk, **bet.to_dict(), "year": year, "outcome": "pending", "pnl": None}
        db[COLLECTIONS["bet_log"]].update_one(fk, {"$set": doc}, upsert=True)

    log.info(f"\n{len(value_bets)} value bet(s) saved to bet_log.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id", type=int, required=True)
    parser.add_argument("--year",     type=int, default=datetime.utcnow().year)
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args()

    run_value_scan(args.event_id, args.year, send_discord=not args.no_discord)
