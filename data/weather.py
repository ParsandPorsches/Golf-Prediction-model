"""
data/weather.py
---------------
Free weather forecasts via Open-Meteo API (no API key required).
Fetches hourly wind speed and direction for each round of a tournament
(Thursday-Sunday, 7 AM - 6 PM local time) and returns a summary dict.

Used by model/pre_tournament.py to adjust SG component weights:
  - High wind  -> approach (APP) matters more, distance (OTT) matters less
  - Calm wind  -> weights unchanged

Usage:
    from data.weather import get_tournament_wind
    wind = get_tournament_wind("THE PLAYERS Championship", "2026-03-12")
    # wind = {"avg_mph": 14.2, "max_mph": 22.1, "rounds": [...], "course": ...}

    python data/weather.py --course "TPC Sawgrass" --date 2026-03-12
"""

import sys
import math
import logging
import argparse
import datetime
from difflib import get_close_matches

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Course coordinate database
# lat/lng for the clubhouse / 1st tee at each PGA Tour venue
# ---------------------------------------------------------------------------
COURSE_COORDS = {
    "Memorial Park": {
        "aliases": ["Memorial Park", "Houston Open", "Texas Children's Houston Open",
                    "Texas Children's"],
        "lat": 29.7654, "lng": -95.4224, "tz": "America/Chicago",
    },
    "Augusta National": {
        "aliases": ["Augusta National", "Masters", "Masters Tournament", "Augusta"],
        "lat": 33.5021, "lng": -82.0232, "tz": "America/New_York",
    },
    "Pebble Beach": {
        "aliases": ["Pebble Beach", "AT&T Pebble Beach", "AT&T Pebble Beach Pro-Am",
                    "Pebble Beach Pro-Am"],
        "lat": 36.5680, "lng": -121.9500, "tz": "America/Los_Angeles",
    },
    "TPC Sawgrass": {
        "aliases": ["TPC Sawgrass", "THE PLAYERS Championship", "The Players Championship",
                    "Players Championship", "The Players", "Players"],
        "lat": 30.1975, "lng": -81.3956, "tz": "America/New_York",
    },
    "Riviera": {
        "aliases": ["Riviera", "Genesis Invitational", "The Genesis Invitational",
                    "LA Open", "Genesis"],
        "lat": 34.0461, "lng": -118.5058, "tz": "America/Los_Angeles",
    },
    "Torrey Pines": {
        "aliases": ["Torrey Pines", "Farmers Insurance Open", "Farmers Insurance",
                    "US Open Torrey"],
        "lat": 32.8991, "lng": -117.2531, "tz": "America/Los_Angeles",
    },
    "Bay Hill": {
        "aliases": ["Bay Hill", "Arnold Palmer Invitational", "Arnold Palmer",
                    "Arnold Palmer Invitational pres. by Mastercard",
                    "Bay Hill Club"],
        "lat": 28.4534, "lng": -81.4934, "tz": "America/New_York",
    },
    "Muirfield Village": {
        "aliases": ["Muirfield Village", "Memorial Tournament",
                    "the Memorial Tournament pres. by Workday",
                    "Memorial", "Jack Memorial"],
        "lat": 40.1534, "lng": -83.1416, "tz": "America/New_York",
    },
    "Colonial": {
        "aliases": ["Colonial", "Charles Schwab Challenge", "Colonial Country Club"],
        "lat": 32.7262, "lng": -97.3687, "tz": "America/Chicago",
    },
    "Quail Hollow": {
        "aliases": ["Quail Hollow", "Wells Fargo Championship", "Wells Fargo",
                    "PGA Championship Quail"],
        "lat": 35.1673, "lng": -80.8543, "tz": "America/New_York",
    },
    "East Lake": {
        "aliases": ["East Lake", "Tour Championship", "TOUR Championship",
                    "FedEx Cup Final", "Tour Champ"],
        "lat": 33.7248, "lng": -84.3007, "tz": "America/New_York",
    },
    "Kapalua": {
        "aliases": ["Kapalua", "Sentry Tournament of Champions", "Sentry TOC",
                    "Sentry", "Plantation Course"],
        "lat": 20.9998, "lng": -156.6706, "tz": "Pacific/Honolulu",
    },
    "TPC Scottsdale": {
        "aliases": ["TPC Scottsdale", "WM Phoenix Open", "Waste Management Phoenix",
                    "Phoenix Open", "WM Phoenix"],
        "lat": 33.6594, "lng": -111.8890, "tz": "America/Phoenix",
    },
    "Harbour Town": {
        "aliases": ["Harbour Town", "RBC Heritage", "Heritage", "Hilton Head",
                    "RBC Heritage Classic"],
        "lat": 32.1385, "lng": -80.8071, "tz": "America/New_York",
    },
    "Sedgefield": {
        "aliases": ["Sedgefield", "Wyndham Championship", "Sedgefield Country Club"],
        "lat": 36.0388, "lng": -79.8765, "tz": "America/New_York",
    },
    "TPC Twin Cities": {
        "aliases": ["TPC Twin Cities", "3M Open", "Twin Cities"],
        "lat": 45.0875, "lng": -93.5582, "tz": "America/Chicago",
    },
    "Oakmont": {
        "aliases": ["Oakmont", "US Open Oakmont", "Oakmont Country Club"],
        "lat": 40.5206, "lng": -79.8335, "tz": "America/New_York",
    },
    "Bethpage Black": {
        "aliases": ["Bethpage Black", "Bethpage", "PGA Bethpage",
                    "PGA Championship Bethpage"],
        "lat": 40.7540, "lng": -73.4557, "tz": "America/New_York",
    },
    "Wilmington CC": {
        "aliases": ["Wilmington CC", "BMW Championship", "Wilmington Country Club"],
        "lat": 39.7915, "lng": -75.5279, "tz": "America/New_York",
    },
    "TPC Boston": {
        "aliases": ["TPC Boston", "Deutsche Bank", "Northern Trust TPC"],
        "lat": 42.0909, "lng": -71.2653, "tz": "America/New_York",
    },
    "Valero Texas Open": {
        "aliases": ["TPC San Antonio", "Valero Texas Open", "Valero"],
        "lat": 29.5958, "lng": -98.6544, "tz": "America/Chicago",
    },
    "Valspar Championship": {
        "aliases": ["Copperhead Course", "Valspar Championship", "Valspar",
                    "Innisbrook Resort"],
        "lat": 28.1928, "lng": -82.7177, "tz": "America/New_York",
    },
    "Sanderson Farms": {
        "aliases": ["Country Club of Jackson", "Sanderson Farms Championship",
                    "Sanderson Farms"],
        "lat": 32.3182, "lng": -90.1818, "tz": "America/Chicago",
    },
    "FedEx St. Jude": {
        "aliases": ["TPC Southwind", "FedEx St. Jude Championship",
                    "FedEx St. Jude", "St. Jude"],
        "lat": 35.0456, "lng": -89.8736, "tz": "America/Chicago",
    },
    "Sony Open": {
        "aliases": ["Waialae Country Club", "Sony Open in Hawaii", "Sony Open"],
        "lat": 21.2817, "lng": -157.7983, "tz": "Pacific/Honolulu",
    },
    "RBC Canadian Open": {
        "aliases": ["Hamilton Golf", "RBC Canadian Open", "Canadian Open"],
        "lat": 43.2557, "lng": -79.8711, "tz": "America/Toronto",
    },
    "The Open": {
        "aliases": ["The Open", "The Open Championship", "British Open"],
        "lat": 56.3426, "lng": -2.8016, "tz": "Europe/London",
    },
    "Scottish Open": {
        "aliases": ["Genesis Scottish Open", "Scottish Open",
                    "Renaissance Club"],
        "lat": 56.0028, "lng": -2.5989, "tz": "Europe/London",
    },
    "Travelers Championship": {
        "aliases": ["TPC River Highlands", "Travelers Championship", "Travelers"],
        "lat": 41.5970, "lng": -72.6526, "tz": "America/New_York",
    },
    "Rocket Mortgage Classic": {
        "aliases": ["Detroit Golf Club", "Rocket Mortgage Classic",
                    "Rocket Mortgage"],
        "lat": 42.3841, "lng": -83.0988, "tz": "America/Detroit",
    },
    "John Deere Classic": {
        "aliases": ["TPC Deere Run", "John Deere Classic", "John Deere"],
        "lat": 41.4958, "lng": -90.5099, "tz": "America/Chicago",
    },
    "Fortinet Championship": {
        "aliases": ["Silverado Resort", "Fortinet Championship", "Fortinet"],
        "lat": 38.3220, "lng": -122.2926, "tz": "America/Los_Angeles",
    },
    "Shriners": {
        "aliases": ["TPC Summerlin", "Shriners Children's Open", "Shriners Open",
                    "Shriners"],
        "lat": 36.1836, "lng": -115.3228, "tz": "America/Los_Angeles",
    },
    "ZOZO Championship": {
        "aliases": ["Accordia Golf Narashino", "ZOZO Championship", "ZOZO"],
        "lat": 35.7220, "lng": 140.0620, "tz": "Asia/Tokyo",
    },
    "American Express": {
        "aliases": ["PGA West", "The American Express", "American Express"],
        "lat": 33.7295, "lng": -116.2896, "tz": "America/Los_Angeles",
    },
    "Puerto Rico Open": {
        "aliases": ["Grand Reserve CC", "Puerto Rico Open"],
        "lat": 18.4661, "lng": -65.9060, "tz": "America/Puerto_Rico",
    },
    "WWT Championship": {
        "aliases": ["El Cardonal at Diamante", "World Wide Technology Championship",
                    "WWT Championship"],
        "lat": 22.8798, "lng": -109.9162, "tz": "America/Mazatlan",
    },
}

# Build lowercase alias map
_COORD_MAP: dict[str, str] = {}
for _canonical, _data in COURSE_COORDS.items():
    for _alias in _data.get("aliases", []):
        _COORD_MAP[_alias.lower()] = _canonical

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
PLAYING_HOURS = range(7, 19)   # 7 AM to 6 PM inclusive


# ---------------------------------------------------------------------------
# Coordinate lookup
# ---------------------------------------------------------------------------

def get_course_coords(course_name: str) -> dict | None:
    """Return {'lat', 'lng', 'tz'} for a course name or None."""
    query = course_name.strip().lower()

    if query in _COORD_MAP:
        return COURSE_COORDS[_COORD_MAP[query]]

    close = get_close_matches(query, list(_COORD_MAP.keys()), n=1, cutoff=0.60)
    if close:
        return COURSE_COORDS[_COORD_MAP[close[0]]]

    return None


# ---------------------------------------------------------------------------
# Open-Meteo API fetch
# ---------------------------------------------------------------------------

def _ms_to_mph(ms: float) -> float:
    return ms * 2.23694


def fetch_forecast(lat: float, lng: float, start_date: str, end_date: str,
                   timezone: str = "auto") -> dict:
    """
    Fetch hourly wind speed (m/s) and direction (degrees) from Open-Meteo.

    Parameters
    ----------
    lat, lng      : coordinates
    start_date    : 'YYYY-MM-DD'
    end_date      : 'YYYY-MM-DD' (inclusive)
    timezone      : IANA timezone string or 'auto'

    Returns raw Open-Meteo JSON response dict.
    """
    params = {
        "latitude":        lat,
        "longitude":       lng,
        "hourly":          "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "wind_speed_unit": "mph",
        "start_date":      start_date,
        "end_date":        end_date,
        "timezone":        timezone,
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        log.error(f"Open-Meteo request failed: {exc}")
        return {}


def _parse_forecast(raw: dict) -> list[dict]:
    """
    Parse Open-Meteo hourly response into a list of hourly dicts:
        [{"time": datetime, "wind_mph": float, "gust_mph": float,
          "direction": float}, ...]
    """
    hourly = raw.get("hourly", {})
    times  = hourly.get("time", [])
    speeds = hourly.get("wind_speed_10m", [])
    gusts  = hourly.get("wind_gusts_10m", [])
    dirs   = hourly.get("wind_direction_10m", [])

    rows = []
    for t, s, g, d in zip(times, speeds, gusts, dirs):
        try:
            dt = datetime.datetime.fromisoformat(t)
        except ValueError:
            continue
        rows.append({
            "time":      dt,
            "wind_mph":  float(s) if s is not None else 0.0,
            "gust_mph":  float(g) if g is not None else 0.0,
            "direction": float(d) if d is not None else 0.0,
        })
    return rows


# ---------------------------------------------------------------------------
# Tournament wind summary
# ---------------------------------------------------------------------------

def get_tournament_wind(
    course_name: str,
    start_date: str,
    rounds: int = 4,
) -> dict | None:
    """
    Fetch wind forecast for a tournament and return a summary.

    Parameters
    ----------
    course_name : str
        Fuzzy-matched against COURSE_COORDS.
    start_date  : str
        'YYYY-MM-DD' for the first round (Thursday).
    rounds      : int
        Number of rounds (default 4 = Thu-Sun).

    Returns
    -------
    dict with keys:
        course       : str  (matched course name)
        avg_mph      : float  (mean wind during playing hours across all rounds)
        max_mph      : float  (maximum gust during playing hours)
        rounds       : list of per-day dicts {date, avg_mph, max_mph}
        condition    : str  "calm" / "breezy" / "windy" / "very_windy"
    or None if course not found or API failed.
    """
    coords = get_course_coords(course_name)
    if coords is None:
        log.warning(f"No coordinates found for '{course_name}'. Weather adjustment skipped.")
        return None

    start_dt  = datetime.date.fromisoformat(start_date)
    end_dt    = start_dt + datetime.timedelta(days=rounds - 1)
    today     = datetime.date.today()
    max_date  = today + datetime.timedelta(days=15)   # Open-Meteo free limit

    if start_dt > max_date:
        log.warning(
            f"  Tournament starts {start_dt}, which is beyond the 16-day forecast "
            f"window (max: {max_date}). Weather adjustment unavailable."
        )
        return None

    # Clamp end_date to max forecast date
    fetch_end = min(end_dt, max_date)
    if fetch_end < end_dt:
        log.info(
            f"  Forecast only available through {fetch_end} "
            f"(tournament ends {end_dt}). Basing wind on available rounds."
        )

    log.info(
        f"  Fetching wind forecast for {course_name} "
        f"({start_dt} to {fetch_end})..."
    )
    raw = fetch_forecast(
        coords["lat"], coords["lng"],
        start_date, fetch_end.isoformat(),
        timezone=coords["tz"],
    )
    if not raw:
        return None

    hourly = _parse_forecast(raw)
    if not hourly:
        return None

    # Filter to playing hours (7 AM – 6 PM) for each round day
    round_summaries = []
    all_playing = []

    for day_offset in range(rounds):
        day = start_dt + datetime.timedelta(days=day_offset)
        day_hours = [
            h for h in hourly
            if h["time"].date() == day and h["time"].hour in PLAYING_HOURS
        ]
        if not day_hours:
            continue

        winds = [h["wind_mph"] for h in day_hours]
        gusts = [h["gust_mph"] for h in day_hours]
        round_summaries.append({
            "date":    day.isoformat(),
            "avg_mph": round(sum(winds) / len(winds), 1),
            "max_mph": round(max(gusts), 1),
        })
        all_playing.extend(day_hours)

    if not all_playing:
        log.warning("No playing-hours forecast data found.")
        return None

    all_winds = [h["wind_mph"] for h in all_playing]
    all_gusts = [h["gust_mph"] for h in all_playing]
    avg_mph = sum(all_winds) / len(all_winds)
    max_mph = max(all_gusts)

    if avg_mph < 8:
        condition = "calm"
    elif avg_mph < 15:
        condition = "breezy"
    elif avg_mph < 25:
        condition = "windy"
    else:
        condition = "very_windy"

    return {
        "course":    course_name,
        "avg_mph":   round(avg_mph, 1),
        "max_mph":   round(max_mph, 1),
        "rounds":    round_summaries,
        "condition": condition,
    }


# ---------------------------------------------------------------------------
# Weight adjustment
# ---------------------------------------------------------------------------

# Target SG weights under very windy conditions (avg > 25 mph):
# Approach dominates — you need precision into greens from unpredictable lies.
# OTT distance advantage shrinks — wind equalises driving.
# ARG up — more missed greens require scrambling.
# Putting down — greens are slower to break in wind.
WINDY_TARGET_WEIGHTS = {
    "ott":  0.10,
    "app":  0.50,
    "arg":  0.28,
    "putt": 0.12,
}


def wind_adjust_weights(
    base_weights: dict,
    avg_wind_mph: float,
    calm_threshold: float = 8.0,
    max_wind: float = 30.0,
) -> dict:
    """
    Blend base course weights toward WINDY_TARGET_WEIGHTS based on wind speed.

    At or below calm_threshold mph  -> no adjustment (wind_factor = 0)
    At or above max_wind mph        -> full target weights (wind_factor = 1)
    Between                         -> linear interpolation

    Returns adjusted {"ott", "app", "arg", "putt"} that sum to 1.0.
    """
    if avg_wind_mph <= calm_threshold:
        return base_weights

    wind_factor = min(1.0, (avg_wind_mph - calm_threshold) / (max_wind - calm_threshold))
    adjusted = {}
    for k in ("ott", "app", "arg", "putt"):
        adjusted[k] = round(
            (1 - wind_factor) * base_weights.get(k, 0.25)
            + wind_factor * WINDY_TARGET_WEIGHTS[k],
            4,
        )

    # Renormalize to ensure exact sum = 1.0 (floating point safety)
    total = sum(adjusted.values())
    return {k: round(v / total, 4) for k, v in adjusted.items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Fetch tournament wind forecast")
    parser.add_argument("--course", required=True, help="Course / tournament name")
    parser.add_argument("--date",   required=True,
                        help="First round date YYYY-MM-DD (Thursday)")
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()

    result = get_tournament_wind(args.course, args.date, args.rounds)
    if result is None:
        print("Failed to get wind data.")
        return

    print(f"\nWind Forecast — {result['course']}")
    print(f"Condition : {result['condition'].upper()}")
    print(f"Avg wind  : {result['avg_mph']} mph")
    print(f"Max gust  : {result['max_mph']} mph")
    print()
    print(f"{'Date':<12}  {'Avg (mph)':>10}  {'Max Gust':>10}")
    print("-" * 36)
    for r in result["rounds"]:
        print(f"{r['date']:<12}  {r['avg_mph']:>10}  {r['max_mph']:>10}")

    print()
    # Show how weights would shift from equal weights
    base = {"ott": 0.25, "app": 0.25, "arg": 0.25, "putt": 0.25}
    adj  = wind_adjust_weights(base, result["avg_mph"])
    print("Weight shift (from equal baseline):")
    print(f"  OTT  {base['ott']:.3f} -> {adj['ott']:.3f}  ({adj['ott']-base['ott']:+.3f})")
    print(f"  APP  {base['app']:.3f} -> {adj['app']:.3f}  ({adj['app']-base['app']:+.3f})")
    print(f"  ARG  {base['arg']:.3f} -> {adj['arg']:.3f}  ({adj['arg']-base['arg']:+.3f})")
    print(f"  PUTT {base['putt']:.3f} -> {adj['putt']:.3f}  ({adj['putt']-base['putt']:+.3f})")


if __name__ == "__main__":
    main()
