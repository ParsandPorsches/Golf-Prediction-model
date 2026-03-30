"""
model/recent_form.py
---------------------
Adjusts sg_composite based on each player's most recent tournament finishes.

The recency module (model/recency.py) blends long-term SG rolling averages
(l24/l36/l72). This module captures *short-term momentum* — a player who has
3 top-10s in their last 5 starts is "hot" in a way that rolling SG averages
won't reflect, and a player who has missed 3 cuts in a row should be faded.

Data source: ESPN results (MongoDB espn_tournament_results), same as course_history.
Requires at least 3 recent starts to apply any adjustment.

Approach:
  1. Load all ESPN results for the current year (+ late prior year for early season).
  2. For each player in the field, find their last N finishes sorted by date.
  3. Convert each finish to a percentile rank within that field (0=best, 1=worst).
  4. Weight most recent starts heaviest (exponential decay).
  5. Convert to z-score, blend into sg_composite with alpha (default 0.12).
  6. Flag hot streaks (3+ consecutive top-20s) and cold streaks (3+ missed cuts).

Usage:
    from model.recent_form import apply_recent_form
    field_df = apply_recent_form(field_df, db)
"""

import sys
import logging
from difflib import get_close_matches

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config.settings import COLLECTIONS

log = logging.getLogger(__name__)

# Number of most recent starts to consider
_LOOKBACK_EVENTS = 5

# Exponential decay weights for last N events (index 0 = most recent)
_EVENT_WEIGHTS = [1.00, 0.75, 0.55, 0.40, 0.30]

# Streak thresholds
_HOT_STREAK_TOP_N = 20      # top-20 finish counts as "good"
_COLD_STREAK_MISSED_CUT = True
_STREAK_LENGTH = 3           # consecutive events to flag


def _build_dg_id_name_map(db) -> dict[str, int]:
    """
    Build a lookup from lowercase 'first last' player name -> dg_id
    using the dg_player_map collection.
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
        if "," in raw:
            parts = raw.split(",", 1)
            normalized = f"{parts[1].strip()} {parts[0].strip()}".lower()
        else:
            normalized = raw.lower().strip()
        mapping[normalized] = dg_id
    return mapping


def _load_recent_results(db, max_years_back: int = 2) -> pd.DataFrame:
    """
    Load ESPN results from the most recent years, sorted by event_date.

    Returns a DataFrame with columns:
        event_name, event_date, year, player_name, finish, field_size,
        made_cut, finish_pct
    """
    from datetime import datetime
    current_year = datetime.now().year
    years = list(range(current_year - max_years_back + 1, current_year + 1))

    docs = list(db[COLLECTIONS["espn_results"]].find(
        {"year": {"$in": years}},
        {"_id": 0, "event_name": 1, "event_date": 1, "year": 1, "results": 1}
    ))

    if not docs:
        return pd.DataFrame()

    rows = []
    for doc in docs:
        event_name = doc.get("event_name", "")
        event_date = doc.get("event_date", "")
        year = doc.get("year")
        results = doc.get("results", [])
        field_size = len(results)

        if field_size < 10:
            continue

        for r in results:
            player_name = r.get("player_name", "").strip()
            if not player_name:
                continue

            finish = r.get("finish")
            made_cut = r.get("made_cut", True)
            withdrew = r.get("withdrew", False)

            if not made_cut or withdrew or finish is None:
                finish = field_size + 1

            rows.append({
                "event_name":  event_name,
                "event_date":  event_date,
                "year":        year,
                "player_name": player_name,
                "finish":      int(finish),
                "field_size":  field_size,
                "made_cut":    bool(made_cut) and not withdrew,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Percentile rank within each event (0=best, 1=worst)
    df["finish_pct"] = df.groupby(["event_name", "year"])["finish"].transform(
        lambda x: x.rank(method="average", pct=True)
    )

    return df.sort_values("event_date", ascending=False)


def _compute_form_scores(
    recent_df: pd.DataFrame,
    dg_id_map: dict[str, int],
    field_dg_ids: set[int],
    n_events: int = _LOOKBACK_EVENTS,
    min_starts: int = 3,
) -> dict[int, dict]:
    """
    For each player in the field, compute a form score from their last N events.

    Returns {dg_id: {"form_score": float, "hot": bool, "cold": bool, "n_starts": int}}
    """
    scores = {}

    for player_name, group in recent_df.groupby("player_name"):
        key = player_name.lower()
        dg_id = dg_id_map.get(key)

        if dg_id is None:
            close = get_close_matches(key, list(dg_id_map.keys()), n=1, cutoff=0.82)
            if close:
                dg_id = dg_id_map[close[0]]

        if dg_id is None or dg_id not in field_dg_ids:
            continue

        # Take last N events sorted by date (already sorted desc)
        last_n = group.head(n_events)
        n_starts = len(last_n)

        if n_starts < min_starts:
            continue

        # Weighted form score (lower = better recent form)
        weighted_sum = 0.0
        weight_total = 0.0
        for i, (_, row) in enumerate(last_n.iterrows()):
            w = _EVENT_WEIGHTS[i] if i < len(_EVENT_WEIGHTS) else 0.20
            weighted_sum += w * row["finish_pct"]
            weight_total += w

        form_score = weighted_sum / weight_total if weight_total > 0 else 0.5

        # Detect streaks from most recent events
        recent_finishes = last_n.head(_STREAK_LENGTH)
        hot = False
        cold = False

        if len(recent_finishes) >= _STREAK_LENGTH:
            # Hot: last 3 events all top-20 finishes
            top_n_pcts = recent_finishes["finish"].values
            field_sizes = recent_finishes["field_size"].values
            hot = all(
                f <= _HOT_STREAK_TOP_N and f <= fs
                for f, fs in zip(top_n_pcts, field_sizes)
            )
            # Cold: last 3 events all missed cut
            cold = all(not mc for mc in recent_finishes["made_cut"].values)

        scores[dg_id] = {
            "form_score": form_score,
            "hot": hot,
            "cold": cold,
            "n_starts": n_starts,
        }

    return scores


def apply_recent_form(
    field_df: pd.DataFrame,
    db,
    alpha: float = 0.12,
    n_events: int = _LOOKBACK_EVENTS,
    min_starts: int = 3,
) -> pd.DataFrame:
    """
    Adjust sg_composite for recent tournament form.

    Parameters
    ----------
    field_df : pd.DataFrame
        Must contain columns: dg_id, player_name, sg_composite.
    db : pymongo.database.Database
    alpha : float
        Blend weight (default 0.12). Kept small — form is noisy but real.
    n_events : int
        Number of most recent starts to consider (default 5).
    min_starts : int
        Minimum starts required to apply adjustment (default 3).

    Returns
    -------
    pd.DataFrame with additional columns:
        recent_form_score : weighted percentile (lower = hotter form)
        recent_form_adj   : z-score rescaled to sg_composite spread
        form_streak       : "HOT", "COLD", or None
    sg_composite is updated in-place for qualifying players.
    """
    if field_df.empty:
        return field_df

    log.info("  Recent form: loading ESPN results...")
    recent_df = _load_recent_results(db)

    if recent_df.empty:
        log.warning("  No ESPN results found. Run: python extraction/espn_results.py --years 2026")
        return field_df

    n_events_loaded = recent_df.groupby(["event_name", "year"]).ngroups
    log.info(f"  Recent form: {n_events_loaded} events loaded from ESPN results.")

    dg_id_map = _build_dg_id_name_map(db)
    field_dg_ids = set(field_df["dg_id"].dropna().astype(int).tolist())

    form_data = _compute_form_scores(
        recent_df, dg_id_map, field_dg_ids,
        n_events=n_events, min_starts=min_starts,
    )

    if not form_data:
        log.warning("  Could not match any players to recent form data.")
        return field_df

    df = field_df.copy()

    # Map form scores onto field
    df["recent_form_score"] = df["dg_id"].map(
        {k: v["form_score"] for k, v in form_data.items()}
    )
    df["form_streak"] = df["dg_id"].map(
        lambda did: (
            "HOT" if form_data.get(did, {}).get("hot") else
            "COLD" if form_data.get(did, {}).get("cold") else
            None
        )
    )

    mask = df["recent_form_score"].notna()
    n_matched = mask.sum()

    log.info(f"  Recent form: matched {n_matched}/{len(df)} players ({min_starts}+ starts).")

    if n_matched < 5:
        log.warning("  Too few form matches. Skipping recent form adjustment.")
        return field_df

    # Log streaks
    hot_players = df.loc[df["form_streak"] == "HOT", "player_name"].tolist()
    cold_players = df.loc[df["form_streak"] == "COLD", "player_name"].tolist()
    if hot_players:
        names = ", ".join(str(n) for n in hot_players[:5])
        log.info(f"  HOT STREAK ({_STREAK_LENGTH}+ top-{_HOT_STREAK_TOP_N}s): {names}")
    if cold_players:
        names = ", ".join(str(n) for n in cold_players[:5])
        log.info(f"  COLD STREAK ({_STREAK_LENGTH}+ missed cuts): {names}")

    # Z-score transformation
    # Invert: lower form_score (=better recent finishes) -> positive z-score -> boost
    form_mean = df.loc[mask, "recent_form_score"].mean()
    form_std = df.loc[mask, "recent_form_score"].std(ddof=1)
    sg_std = df["sg_composite"].std(ddof=1)

    if form_std < 1e-9 or sg_std < 1e-9:
        log.warning("  Near-zero variance in form/composite. Skipping.")
        return field_df

    z = -(df.loc[mask, "recent_form_score"] - form_mean) / form_std
    df["recent_form_adj"] = np.nan
    df.loc[mask, "recent_form_adj"] = z * sg_std + df["sg_composite"].mean()

    # Preserve original composite if not already saved
    if "sg_composite_original" not in df.columns:
        df["sg_composite_original"] = df["sg_composite"]

    # Blend
    df.loc[mask, "sg_composite"] = (
        (1 - alpha) * df.loc[mask, "sg_composite"]
        + alpha * df.loc[mask, "recent_form_adj"]
    )

    log.info(f"  Recent form applied: alpha={alpha:.2f}, {n_matched} players adjusted.")

    return df
