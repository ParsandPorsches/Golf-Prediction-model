"""
config/settings.py
------------------
Central config. Fill in your real values here or use a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ────────────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "golf_model_v3"

# Collection names
COLLECTIONS = {
    "player_map":            "dg_player_map",
    "historical_odds":       "dg_historical_odds",
    "predictions_archive":   "dg_predictions_archive",
    "skill_ratings":         "dg_skill_ratings",
    "raw_scoring":           "dg_raw_scoring",
    "espn_results":          "espn_tournament_results",
    "backtest_runs":         "backtest_runs",
    "model_predictions":     "model_predictions",
    "live_odds":             "live_odds",
    "bet_log":               "bet_log",
}

# ── DataGolf API ───────────────────────────────────────────────────────────────
DATAGOLF_API_KEY = os.getenv("DATAGOLF_API_KEY", "")

# Historical extraction range
EXTRACTION_YEARS = list(range(2019, 2026))   # 2019–2025 inclusive
HOLDOUT_YEAR = 2025                           # Never touch until Phase 2 is complete

# Markets to pull historical odds for
ODDS_MARKETS = ["win", "top_5", "top_10", "top_20", "make_cut"]

# Books to pull — Pinnacle is the sharp benchmark
ODDS_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm"]

# ── Backtester ─────────────────────────────────────────────────────────────────

# SG weight search space (Phase 2 grid search)
WEIGHT_GRID = {
    "sg_ott":  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "sg_app":  [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "sg_arg":  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    "sg_putt": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],  # 0.00 is a valid candidate
}

# Edge thresholds for flagging value bets
EDGE_THRESHOLDS = {
    "win":     0.03,
    "top_5":   0.05,
    "top_10":  0.05,
    "top_20":  0.08,
    "make_cut": 0.05,
}

# Minimum edge tests (coarse, medium, fine)
EDGE_TEST_THRESHOLDS = [0.03, 0.05, 0.08]

# ── Monte Carlo ────────────────────────────────────────────────────────────────
N_SIMULATIONS = 10_000
RANDOM_SEED = 42

# ── Bet sizing ─────────────────────────────────────────────────────────────────
KELLY_FRACTION = 0.25   # Fractional Kelly — conservative, use 0.25–0.50
MAX_BET_PCT = 0.05      # Never bet more than 5% of bankroll on one bet

# ── The Odds API ───────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "YOUR_KEY_HERE")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# ── Discord ────────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ── Rate limiting ──────────────────────────────────────────────────────────────
REQUEST_DELAY_SECONDS = 1.0   # Sleep between DataGolf API calls
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5.0
