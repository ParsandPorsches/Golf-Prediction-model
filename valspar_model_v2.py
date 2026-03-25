"""
Golf Betting Model v2 — Valspar Championship 2026
==================================================
100% FREE, no API keys required.

What's in this model:
  1. Live leaderboard          — ESPN public API
  2. Strokes-Gained splits     — PGA Tour stats (SG:OTT, SG:APP, SG:ARG, SG:PUTT)
  3. Recent form (8 weeks)     — decay-weighted average of last 8 weeks of results
  4. Course history            — past Valspar finishes (2022-2025)
  5. Monte Carlo simulation    — 10,000 runs of remaining rounds
  6. CSV export                — saved to same folder as this script

Install:
    pip install requests pandas numpy tabulate colorama

Run:
    python valspar_model_v2.py
"""

import re
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate
from colorama import Fore, Style, init
import os

init(autoreset=True)

# ================================================================
#  CONFIG
# ================================================================

TOURNAMENT_NAME  = "Valspar Championship 2026"
COURSE           = "Innisbrook Resort (Copperhead), Par 71"
MONTE_CARLO_SIMS = 10_000

# Copperhead SG weights - approach and scrambling dominate here
SG_WEIGHTS = {
    "sg_app":  0.40,
    "sg_arg":  0.25,
    "sg_putt": 0.20,
    "sg_ott":  0.15,
}

FORM_WEEKS   = 8
FORM_DECAY   = 0.80   # each week back is worth 80% of the previous

PAST_VALSPAR_DATES = ["20250323", "20240324", "20230319", "20220320"]

# PGA Tour strokes-gained stat IDs
PGA_SG_IDS = {
    "sg_ott":  "02567",
    "sg_app":  "02568",
    "sg_arg":  "02569",
    "sg_putt": "02564",
}

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga"


# ================================================================
#  HELPERS
# ================================================================

def header(text):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}")


def parse_score(score_str):
    s = str(score_str).strip()
    if s in ("E", "", "--"):
        return 0
    try:
        return int(s.replace("+", ""))
    except ValueError:
        return 0


def normalise(series, invert=False):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    norm = (series - mn) / (mx - mn)
    return (1 - norm) if invert else norm


def american_odds(prob):
    if prob <= 0 or prob >= 1:
        return "N/A"
    dec = 1.0 / prob
    if dec >= 2.0:
        return f"+{int((dec - 1) * 100)}"
    return f"{int(-100 / (dec - 1))}"


def get_espn_event(date_str=None):
    """Get the Valspar event, or first in-progress event when no date given."""
    url    = f"{ESPN_BASE}/scoreboard"
    params = {"dates": date_str} if date_str else {}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception:
        return None

    for event in events:
        if "valspar" in event.get("name", "").lower():
            return event

    if not date_str:
        for event in events:
            if event.get("status", {}).get("type", {}).get("state") == "in":
                return event
        return events[0] if events else None

    return None


def get_espn_event_any(date_str):
    """
    Get ANY completed PGA Tour event for a given date (YYYYMMDD).
    Used for recent form — we want whatever tournament was happening that week,
    not just Valspar.
    """
    url = f"{ESPN_BASE}/scoreboard"
    try:
        r = requests.get(url, params={"dates": date_str}, timeout=15)
        r.raise_for_status()
        events = r.json().get("events", [])
    except Exception:
        return None, None

    # skip the current Valspar (still in progress)
    for event in events:
        name  = event.get("name", "")
        state = event.get("status", {}).get("type", {}).get("state", "")
        if "valspar" in name.lower() and state == "in":
            continue
        # return first event that has competitors
        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        if competitors:
            return event, name

    return None, None


# ================================================================
#  1. LIVE LEADERBOARD
# ================================================================

def fetch_leaderboard():
    header("[ 1/4 ]  Fetching live leaderboard from ESPN...")

    event = get_espn_event()
    if not event:
        print(f"{Fore.RED}  x No active event found.")
        return pd.DataFrame()

    print(f"  + Event: {event.get('name', '?')}")
    competitors = event.get("competitions", [{}])[0].get("competitors", [])
    rows = []

    for p in competitors:
        athlete = p.get("athlete", {})
        status  = p.get("status", {})
        if status.get("type", {}).get("description") == "Cut":
            continue

        total = parse_score(p.get("score", "E"))
        round_scores = []
        for ls in p.get("linescores", []):
            try:
                val = int(ls.get("value", 0))
                if val > 0:
                    round_scores.append(val)
            except (TypeError, ValueError):
                pass

        rows.append({
            "player":        athlete.get("displayName", ""),
            "athlete_id":    athlete.get("id", ""),
            "position":      status.get("position", {}).get("displayName", ""),
            "total":         total,
            "rounds_played": max(len(round_scores), 1),
            "round_scores":  round_scores,
        })

    df = pd.DataFrame(rows)
    print(f"  + {len(df)} active players loaded")
    return df


# ================================================================
#  2. STROKES-GAINED SPLITS
# ================================================================

# ESPN stat keys that map to SG-equivalent metrics
# ── PGA Tour SG stat IDs → local CSV filenames ───────────────────────────
# Download these from pgatour.com/stats/detail/<ID> using the Download button
# Save the CSVs into the same folder as this script
PGA_SG_CSV_IDS = {
    "sg_ott":  "02567",   # SG: Off-the-Tee
    "sg_app":  "02568",   # SG: Approach-the-Green
    "sg_arg":  "02569",   # SG: Around-the-Green
    "sg_putt": "02564",   # SG: Putting
}

# Column names PGA Tour uses in their CSV exports
PGA_CSV_NAME_COLS   = ["PLAYER NAME", "Player Name", "player_name", "NAME", "Name"]
PGA_CSV_VALUE_COLS  = ["AVG", "Avg", "avg", "VALUE", "Value", "AVERAGE", "SG AVG"]


# Friendly name keywords to match against CSV filenames (case-insensitive)
SG_CSV_NAME_PATTERNS = {
    "sg_ott":  ["02567", "off-the-tee", "off the tee", "sg off", "sg_ott"],
    "sg_app":  ["02568", "approach",    "sg app",       "sg_app"],
    "sg_arg":  ["02569", "around",      "sg arg",       "sg_arg"],
    "sg_putt": ["02564", "putting",     "sg putt",      "sg_putt"],
}

def find_sg_csvs(script_dir: str) -> dict:
    """
    Search script_dir for PGA Tour stat CSVs.
    Matches by stat ID number OR friendly name keywords in the filename.
    Returns {sg_col: filepath}.
    """
    found = {}
    try:
        files = os.listdir(script_dir)
    except Exception:
        return found

    for col, patterns in SG_CSV_NAME_PATTERNS.items():
        for fname in files:
            if not fname.lower().endswith(".csv"):
                continue
            fname_lower = fname.lower()
            if any(p in fname_lower for p in patterns):
                found[col] = os.path.join(script_dir, fname)
                break
    return found


def load_sg_csv(filepath: str, col: str) -> dict:
    """
    Load a PGA Tour stat CSV and return {player_name: value}.
    Tries multiple skiprow values and column name patterns.
    """
    # keywords to identify the player name column and value column
    name_keywords  = ["player", "name"]
    value_keywords = ["avg", "sg", "value", "average", "total"]

    for skiprows in [0, 1, 2, 3]:
        try:
            df = pd.read_csv(filepath, skiprows=skiprows)
            df.columns = df.columns.str.strip()

            # find name col
            name_col = next(
                (c for c in df.columns
                 if any(k in c.lower() for k in name_keywords)),
                None
            )
            # find value col — prefer SG/avg columns, avoid rank/year cols
            val_col = next(
                (c for c in df.columns
                 if any(k in c.lower() for k in value_keywords)
                 and c != name_col
                 and "rank" not in c.lower()
                 and "year" not in c.lower()
                 and "event" not in c.lower()),
                None
            )

            if not name_col or not val_col:
                continue

            result = {}
            for _, row in df.iterrows():
                name = str(row[name_col]).strip()
                try:
                    value = float(str(row[val_col]).replace(",", "").replace("+", ""))
                    if name and name.lower() not in ("nan", "player name", "player", ""):
                        result[name] = value
                except (ValueError, TypeError):
                    continue

            if len(result) > 10:   # need at least 10 players to be valid
                return result

        except Exception:
            continue

    # last resort: print columns so user can debug
    try:
        df = pd.read_csv(filepath, nrows=3)
        print(f"    Debug — columns found: {list(df.columns)}")
        print(f"    First row: {df.iloc[0].tolist() if len(df) > 0 else 'empty'}")
    except Exception as e:
        print(f"    Debug — could not read file: {e}")

    return {}


def fetch_sg_stats(leaderboard: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load SG stats from locally saved PGA Tour CSV files.
    
    To get these files:
      1. Go to pgatour.com/stats/detail/02567 (and 02568, 02569, 02564)
      2. Click the Download button on each page
      3. Save the CSV files into the same folder as this script
    """
    header("[ 2/4 ]  Loading SG stats from PGA Tour CSV files...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files  = find_sg_csvs(script_dir)

    if not csv_files:
        print(f"  {Fore.YELLOW}x No SG CSV files found in {script_dir}")
        print(f"  {Fore.WHITE}  To fix: download CSVs from pgatour.com/stats/detail/")
        print(f"           02567 (SG:OTT)  02568 (SG:APP)  02569 (SG:ARG)  02564 (SG:PUTT)")
        print(f"           Save them in the same folder as this script")
        print(f"  {Fore.YELLOW}  Running without SG signal for now...")
        return pd.DataFrame()

    all_stats = {}
    for col, filepath in csv_files.items():
        result = load_sg_csv(filepath, col)
        if result:
            for player, value in result.items():
                all_stats.setdefault(player, {})[col] = value
            print(f"  + {col}: {len(result)} players  ({os.path.basename(filepath)})")
        else:
            print(f"  {Fore.YELLOW}x {col}: could not parse {os.path.basename(filepath)}")

    if not all_stats:
        print(f"  {Fore.YELLOW}x Could not load any SG data from CSVs")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(all_stats, orient="index").reset_index()
    df.rename(columns={"index": "player"}, inplace=True)
    sg_cols = [c for c in ["sg_ott","sg_app","sg_arg","sg_putt"] if c in df.columns]
    print(f"  + SG data ready: {len(df)} players, columns: {sg_cols}")
    return df


# ================================================================
#  3. RECENT FORM (last 8 weeks)
# ================================================================

# One date per week (any day that week works) for the 8 events before Valspar 2026
# ESPN scoreboard returns whatever tournament was running that week
RECENT_FORM_DATES = [
    "20260314",  # PLAYERS Championship week
    "20260307",  # Arnold Palmer week
    "20260228",  # Cognizant Classic week
    "20260221",  # Genesis Invitational week
    "20260214",  # AT&T Pebble Beach week
    "20260207",  # WM Phoenix Open week
    "20260131",  # Farmers Insurance week
    "20260124",  # The American Express week
]


def fetch_recent_form():
    header("[ 3/4 ]  Building recent form (last 8 weeks)...")

    records = []

    for i, date_str in enumerate(RECENT_FORM_DATES):
        week_weight = FORM_DECAY ** i
        event, event_name = get_espn_event_any(date_str)

        if not event:
            print(f"  {Fore.YELLOW}x {date_str}: no event found")
            continue

        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        loaded = 0

        for p in competitors:
            name      = p.get("athlete", {}).get("displayName", "")
            pos_str   = p.get("status", {}).get("position", {}).get("displayName", "")
            pos_clean = pos_str.replace("T", "").strip()
            try:
                finish = int(pos_clean)
            except ValueError:
                finish = 999
            if name:
                records.append({
                    "player":      name,
                    "finish":      finish,
                    "week_weight": week_weight,
                    "event":       event_name,
                })
                loaded += 1

        if loaded:
            print(f"  + {event_name[:48]:<48}  weight={week_weight:.2f}  ({loaded} players)")
        else:
            print(f"  {Fore.YELLOW}x {event_name}: no competitors found")

    if not records:
        print(f"  {Fore.YELLOW}x No recent results loaded — form signal will be neutral")
        return pd.DataFrame()

    form_df = pd.DataFrame(records)
    form_df["finish_score"]   = form_df["finish"].apply(lambda f: 1.0/f if f < 999 else 0.001)
    form_df["weighted_score"] = form_df["finish_score"] * form_df["week_weight"]

    summary = (
        form_df.groupby("player")
        .agg(
            form_raw      = ("weighted_score", "sum"),
            events_played = ("event", "count"),
            best_recent   = ("finish", "min"),
        )
        .reset_index()
    )
    summary["form_score"] = normalise(summary["form_raw"])
    print(f"  + Form scores built for {len(summary)} players")
    return summary[["player", "form_score", "events_played", "best_recent"]]


# ================================================================
#  4. COURSE HISTORY
# ================================================================

def fetch_course_history():
    header("[ 4/4 ]  Scraping Valspar course history (2022-2025)...")

    records = []

    for date_str in PAST_VALSPAR_DATES:
        year  = int(date_str[:4])
        event = get_espn_event(date_str)
        if not event:
            print(f"  {Fore.YELLOW}x {year}: no data")
            continue

        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        loaded = 0
        for p in competitors:
            name      = p.get("athlete", {}).get("displayName", "")
            pos_str   = p.get("status", {}).get("position", {}).get("displayName", "")
            pos_clean = pos_str.replace("T", "").strip()
            try:
                finish = int(pos_clean)
            except ValueError:
                finish = 999
            if name:
                records.append({"player": name, "finish": finish, "year": year})
                loaded += 1
        print(f"  + {year}: {loaded} players")

    if not records:
        print(f"  {Fore.YELLOW}x No history found - signal will be neutral")
        return pd.DataFrame()

    hist = pd.DataFrame(records)
    year_w = {2025: 4, 2024: 3, 2023: 2, 2022: 1}
    hist["year_weight"]    = hist["year"].map(year_w).fillna(1)
    hist["finish_score"]   = hist["finish"].apply(lambda f: 1.0/f if f < 999 else 0.001)
    hist["weighted_score"] = hist["finish_score"] * hist["year_weight"]

    summary = (
        hist.groupby("player")
        .agg(
            history_raw = ("weighted_score", "sum"),
            appearances = ("year", "count"),
            best_finish = ("finish", "min"),
            avg_finish  = ("finish", lambda x: x[x < 999].mean() if (x < 999).any() else 999),
        )
        .reset_index()
    )
    summary["history_raw"]   += np.log1p(summary["appearances"]) * 0.05
    summary["history_score"]  = normalise(summary["history_raw"])
    print(f"  + History scores built for {len(summary)} players")
    return summary[["player", "history_score", "appearances", "best_finish", "avg_finish"]]


# ================================================================
#  5. BUILD MODEL
# ================================================================

def build_model(leaderboard, sg_stats, form, history):
    df = leaderboard.copy()

    if not sg_stats.empty:
        df = df.merge(sg_stats, on="player", how="left")
    if not form.empty:
        df = df.merge(form,     on="player", how="left")
    if not history.empty:
        df = df.merge(history,  on="player", how="left")

    rounds   = df["rounds_played"].max()
    progress = rounds / 4.0

    # blend weights fade as live scores accumulate
    w_score   = progress
    w_sg      = max(0.0, 0.30 * (1 - progress))
    w_form    = max(0.0, 0.25 * (1 - progress * 0.8))
    w_history = max(0.0, 0.20 * (1 - progress * 1.4))
    total_w   = w_score + w_sg + w_form + w_history

    weights = {
        "score":   round(w_score   / total_w, 3),
        "sg":      round(w_sg      / total_w, 3),
        "form":    round(w_form    / total_w, 3),
        "history": round(w_history / total_w, 3),
    }

    # live score signal
    df["score_norm"] = normalise(df["total"], invert=True)

    # SG composite
    sg_comp  = pd.Series(0.0, index=df.index)
    sg_tot_w = 0.0
    for col, w in SG_WEIGHTS.items():
        if col in df.columns:
            filled   = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            sg_comp += w * normalise(filled)
            sg_tot_w += w
    df["sg_score"] = sg_comp / sg_tot_w if sg_tot_w > 0 else 0.5

    # form & history (fillna to neutral if player not in data)
    if "form_score" not in df.columns:
        df["form_score"] = 0.5
    else:
        df["form_score"] = pd.to_numeric(df["form_score"], errors="coerce").fillna(0.5)

    if "history_score" not in df.columns:
        df["history_score"] = 0.3
    else:
        df["history_score"] = pd.to_numeric(df["history_score"], errors="coerce").fillna(0.3)

    # composite
    df["composite"] = (
        weights["score"]   * df["score_norm"]   +
        weights["sg"]      * df["sg_score"]      +
        weights["form"]    * df["form_score"]    +
        weights["history"] * df["history_score"]
    )

    # momentum nudge
    def get_momentum(scores):
        return (scores[-1] - np.mean(scores[:-1])) if len(scores) >= 2 else 0.0

    df["momentum"] = df["round_scores"].apply(get_momentum)
    df["composite"] += df["momentum"].apply(
        lambda x: -0.025 if x < -1.5 else (0.025 if x > 1.5 else 0.0)
    )
    df["trend"] = df["momentum"].apply(
        lambda x: "HOT" if x < -1.5 else ("COLD" if x > 1.5 else "")
    )

    return df, weights


# ================================================================
#  6. MONTE CARLO
# ================================================================

def monte_carlo(df, rounds_complete):
    rounds_left = 4 - rounds_complete
    n           = len(df)
    par         = 71.0

    means  = np.zeros(n)
    sigmas = np.zeros(n)
    totals = df["total"].values.astype(float)

    for i, (_, row) in enumerate(df.iterrows()):
        valid = [s for s in row["round_scores"] if s > 0]
        if valid:
            actual_mean = np.mean(valid) - par
            means[i]    = 0.70 * actual_mean + 0.30 * 0.5
            sigmas[i]   = max(np.std(valid, ddof=1), 1.5) if len(valid) >= 2 else 2.8
        else:
            means[i]  = 0.5
            sigmas[i] = 2.8

    if rounds_left > 0:
        sim = np.random.normal(
            loc   = means[np.newaxis, :, np.newaxis],
            scale = sigmas[np.newaxis, :, np.newaxis],
            size  = (MONTE_CARLO_SIMS, n, rounds_left),
        )
        sim_remaining = sim.sum(axis=2)
    else:
        sim_remaining = np.zeros((MONTE_CARLO_SIMS, n))

    sim_totals = totals[np.newaxis, :] + sim_remaining
    ranks      = np.argsort(np.argsort(sim_totals, axis=1), axis=1) + 1
    winners    = np.argmin(sim_totals, axis=1)

    df = df.copy()
    df["mc_win"]   = np.bincount(winners, minlength=n) / MONTE_CARLO_SIMS
    df["mc_top5"]  = (ranks <= 5).mean(axis=0)
    df["mc_top10"] = (ranks <= 10).mean(axis=0)
    df["mc_top20"] = (ranks <= 20).mean(axis=0)

    temp    = 0.3
    softmax = np.exp(df["composite"].values / temp)
    softmax /= softmax.sum()

    df["win_prob"]  = 0.5 * softmax + 0.5 * df["mc_win"]
    df["win_prob"] /= df["win_prob"].sum()

    df["fair_win_odds"]   = df["win_prob"].apply(american_odds)
    df["fair_top5_odds"]  = df["mc_top5"].apply(american_odds)
    df["fair_top10_odds"] = df["mc_top10"].apply(american_odds)

    return df.sort_values("win_prob", ascending=False).reset_index(drop=True)


# ================================================================
#  7. DISPLAY
# ================================================================

def display(df, weights, rounds_complete):
    blend = (
        f"Score {weights['score']*100:.0f}%  |  "
        f"SG {weights['sg']*100:.0f}%  |  "
        f"Form {weights['form']*100:.0f}%  |  "
        f"History {weights['history']*100:.0f}%"
    )

    print(f"\n{Fore.WHITE}{Style.BRIGHT}{'='*82}")
    print(f"  {TOURNAMENT_NAME}")
    print(f"  {COURSE}   |   After Round {rounds_complete}")
    print(f"  Signal blend: {blend}")
    print(f"{'='*82}{Style.RESET_ALL}")

    # Main table
    print(f"\n{Fore.GREEN}{Style.BRIGHT}  WIN PROBABILITIES - TOP 20{Style.RESET_ALL}")
    rows = []
    for i, r in df.head(20).iterrows():
        total_str = f"{r['total']:+d}" if r["total"] != 0 else "E"
        scores    = "  ".join(str(s) for s in r["round_scores"]) if r["round_scores"] else "-"
        rows.append([
            i + 1, r["player"], r["position"], total_str, scores,
            f"{r['win_prob']*100:.1f}%", r["fair_win_odds"],
            f"{r['mc_top5']*100:.0f}%", r["fair_top5_odds"],
            f"{r['mc_top10']*100:.0f}%", r.get("trend", ""),
        ])
    print(tabulate(
        rows,
        headers=["#", "Player", "Pos", "Total", "Rounds", "Win%", "Fair Win",
                 "Top5%", "Fair Top5", "Top10%", "Trend"],
        tablefmt="rounded_outline",
        numalign="right",
    ))

    # SG leaders
    if "sg_score" in df.columns and df["sg_score"].std() > 0.01:
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}  STROKES-GAINED LEADERS - TOP 10{Style.RESET_ALL}")
        sg_rows = []
        for _, r in df.nlargest(10, "sg_score").iterrows():
            total_str = f"{r['total']:+d}" if r["total"] != 0 else "E"
            sg_rows.append([
                r["player"], r["position"], total_str,
                f"{r.get('sg_ott',  0):.3f}",
                f"{r.get('sg_app',  0):.3f}",
                f"{r.get('sg_arg',  0):.3f}",
                f"{r.get('sg_putt', 0):.3f}",
                f"{r['sg_score']:.3f}",
                f"{r['win_prob']*100:.1f}%",
            ])
        print(tabulate(
            sg_rows,
            headers=["Player", "Pos", "Total", "SG:OTT", "SG:APP", "SG:ARG", "SG:PUTT", "Composite", "Win%"],
            tablefmt="rounded_outline",
        ))

    # Form leaders
    if "form_score" in df.columns and df["form_score"].std() > 0.01:
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}  RECENT FORM LEADERS - TOP 10  (last {FORM_WEEKS} weeks){Style.RESET_ALL}")
        form_rows = []
        for _, r in df.nlargest(10, "form_score").iterrows():
            total_str = f"{r['total']:+d}" if r["total"] != 0 else "E"
            best_r    = str(int(r["best_recent"])) if r.get("best_recent", 999) < 999 else "MC"
            form_rows.append([
                r["player"], r["position"], total_str,
                f"{r['form_score']:.3f}",
                int(r.get("events_played", 0)),
                best_r,
                f"{r['win_prob']*100:.1f}%",
            ])
        print(tabulate(
            form_rows,
            headers=["Player", "Pos", "Total", "Form Score", "Events", "Best (8wk)", "Win%"],
            tablefmt="rounded_outline",
        ))

    # Course history
    if "history_score" in df.columns and "appearances" in df.columns:
        hist_players = df[df["appearances"] > 0]
        if not hist_players.empty:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}  COPPERHEAD VETERANS - TOP 10{Style.RESET_ALL}")
            hist_rows = []
            for _, r in hist_players.nlargest(10, "history_score").iterrows():
                total_str = f"{r['total']:+d}" if r["total"] != 0 else "E"
                avg_f  = f"{r['avg_finish']:.1f}" if r.get("avg_finish", 999) < 999 else "MC"
                best_f = str(int(r["best_finish"])) if r.get("best_finish", 999) < 999 else "MC"
                hist_rows.append([
                    r["player"], r["position"], total_str,
                    int(r["appearances"]), best_f, avg_f,
                    f"{r['history_score']:.3f}",
                    f"{r['win_prob']*100:.1f}%",
                ])
            print(tabulate(
                hist_rows,
                headers=["Player", "Pos", "Total", "Apps", "Best", "Avg", "Hist Score", "Win%"],
                tablefmt="rounded_outline",
            ))

    # Momentum
    hot  = df[df["momentum"] < -1.5].head(5)
    cold = df[df["momentum"] >  1.5].head(5)
    if not hot.empty:
        print(f"\n{Fore.RED}{Style.BRIGHT}  HEATING UP{Style.RESET_ALL}")
        for _, r in hot.iterrows():
            print(f"   {r['player']:<26} {r['position']:<6}  {r['momentum']:+.1f} vs avg  "
                  f"Win: {r['win_prob']*100:.1f}%  Top5: {r['mc_top5']*100:.0f}%")
    if not cold.empty:
        print(f"\n{Fore.BLUE}{Style.BRIGHT}  FADING{Style.RESET_ALL}")
        for _, r in cold.iterrows():
            print(f"   {r['player']:<26} {r['position']:<6}  {r['momentum']:+.1f} vs avg  "
                  f"Win: {r['win_prob']*100:.1f}%  Top5: {r['mc_top5']*100:.0f}%")

    # Sportsbook guide
    print(f"\n{Fore.WHITE}{Style.BRIGHT}  HOW TO USE AT YOUR SPORTSBOOK{Style.RESET_ALL}")
    print("""
  OUTRIGHT WINNER
    Compare "Fair Win" to your book's winner market.
    Book shows bigger + number than Fair Win  =  VALUE, consider betting.
    Book shows smaller + number               =  skip, you're overpaying.

  TOP 5 / TOP 10 (place markets - often most mispriced)
    Use Top5% / Fair Top5 and Top10% / Fair Top10 columns.
    Same rule: book offering more than fair price = value.

  BEST HUNTING GROUND: model ranks #4-#10 are most often mispriced.
  Leaders get heavily bet down - value lives further back in the field.
    """)

    print(f"{Fore.WHITE}Monte Carlo: {MONTE_CARLO_SIMS:,} simulations, {4-rounds_complete} round(s) remaining")
    print(f"Final win prob = 50% softmax (composite) + 50% Monte Carlo")
    print(f"{Fore.RED}  WARNING: Model output only - not financial advice. Bet responsibly.\n")


# ================================================================
#  8. CSV EXPORT
# ================================================================

def save_csv(df):
    export = df.copy()

    max_r = max((len(r) for r in export["round_scores"]), default=4)
    for i in range(max_r):
        export[f"R{i+1}"] = export["round_scores"].apply(
            lambda x, i=i: x[i] if i < len(x) else ""
        )

    col_map = {
        "player":          "Player",
        "position":        "Position",
        "total":           "Total (vs Par)",
        **{f"R{i+1}": f"Round {i+1}" for i in range(max_r)},
        "win_prob":        "Win %",
        "fair_win_odds":   "Fair Win Odds",
        "mc_top5":         "Top 5 %",
        "fair_top5_odds":  "Fair Top5 Odds",
        "mc_top10":        "Top 10 %",
        "fair_top10_odds": "Fair Top10 Odds",
        "mc_top20":        "Top 20 %",
        "sg_ott":          "SG: Off-the-Tee",
        "sg_app":          "SG: Approach",
        "sg_arg":          "SG: Around-Green",
        "sg_putt":         "SG: Putting",
        "sg_score":        "SG Composite",
        "form_score":      "Recent Form Score",
        "best_recent":     "Best Finish (8wk)",
        "history_score":   "Course History Score",
        "appearances":     "Valspar Appearances",
        "best_finish":     "Best Valspar Finish",
        "avg_finish":      "Avg Valspar Finish",
        "momentum":        "Momentum",
        "trend":           "Trend",
    }

    col_map = {k: v for k, v in col_map.items() if k in export.columns}
    export  = export[list(col_map.keys())].rename(columns=col_map)

    for c in ["Win %", "Top 5 %", "Top 10 %", "Top 20 %"]:
        if c in export.columns:
            export[c] = (export[c] * 100).round(1).astype(str) + "%"

    ts       = datetime.now().strftime("%H%M")
    filename = f"valspar_v2_{ts}.csv"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    export.to_csv(filepath, index=False)
    print(f"{Fore.GREEN}  + CSV saved -> {filepath}\n")


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    print(f"\n{Fore.WHITE}{Style.BRIGHT}  Golf Betting Model v2  -  {TOURNAMENT_NAME}{Style.RESET_ALL}")
    print(f"  {datetime.now().strftime('%A %d %B %Y, %H:%M')}\n")

    leaderboard = fetch_leaderboard()
    if leaderboard.empty:
        print(f"{Fore.RED}Could not load leaderboard. Check your internet connection.")
        exit(1)

    sg_stats = fetch_sg_stats()
    form     = fetch_recent_form()
    history  = fetch_course_history()

    df, weights = build_model(leaderboard, sg_stats, form, history)

    rounds_complete = int(leaderboard["rounds_played"].max())
    print(f"\n{Fore.CYAN}Running Monte Carlo ({MONTE_CARLO_SIMS:,} simulations)...")
    df = monte_carlo(df, rounds_complete)
    print(f"{Fore.GREEN}  + Done")

    display(df, weights, rounds_complete)
    save_csv(df)