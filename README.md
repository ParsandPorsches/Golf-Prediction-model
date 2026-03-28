# Golf Betting Model v3

Pre-tournament golf betting model using DataGolf API, MongoDB, and Monte Carlo simulation.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start MongoDB locally
```bash
# Make sure MongoDB is running on localhost:27017
mongod --dbpath /your/data/path
```

### 3. Configure secrets
Copy `config/settings.py` and fill in your keys:
- `DATAGOLF_API_KEY` — from datagolf.com
- `MONGODB_URI` — default: `mongodb://localhost:27017`
- `DISCORD_WEBHOOK_URL` — optional, for alerts

### 4. Run Phase 1 (data extraction)
```bash
python extraction/datagolf_pull.py
```
This dumps everything to MongoDB. Takes ~1-2 hours on first run.

### 5. Run Phase 2 (backtester)
```bash
python backtester/weight_optimizer.py
```
Grid-searches SG weight combinations against historical Pinnacle odds.

### 6. Run the live model
```bash
python model/pre_tournament.py --event_id 28 --year 2025
```

## Execution Order

1. Phase 1 — Data extraction (run once, then cancel DataGolf subscription)
2. Phase 2 — Backtest SG weight optimizer
3. Phase 3 — Cut simulation is built into monte_carlo.py
4. Phase 4 — Odds scraping + value detection
5. Phase 5 — Sunday closing power (optional, low priority)

## File Structure

```
golf-model-v3/
├── config/settings.py           # API keys, thresholds, MongoDB URI
├── extraction/
│   ├── datagolf_pull.py         # One-time historical data extraction
│   ├── espn_results.py          # ESPN tournament results
│   └── player_mapping.py        # Cross-source ID fuzzy matching
├── backtester/
│   ├── weight_optimizer.py      # Grid search + ROI scoring
│   ├── monte_carlo.py           # Simulation engine
│   └── cut_simulator.py         # Cut line modeling
├── live/
│   ├── odds_scraper.py          # Multi-book odds ingestion
│   ├── value_detector.py        # Edge calculation
│   └── discord_alerts.py        # Webhook alerts
├── model/
│   ├── pre_tournament.py        # Full pre-tourney pipeline
│   └── sg_composite.py          # SG composite scoring
├── analysis/
│   ├── backtest_report.py       # Backtest visualization
│   └── kelly_sizing.py          # Bet sizing calculator
└── requirements.txt
```
