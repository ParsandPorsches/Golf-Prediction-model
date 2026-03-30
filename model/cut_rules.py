"""
model/cut_rules.py
-------------------
PGA Tour cut rule database. Maps events to their actual cut format
so the Monte Carlo sim uses the right threshold instead of a flat 70.

Cut rules as of 2024-2026 PGA Tour season:
  - Regular full-field events (120-156 players): top 65 + ties
  - Signature events (~70-80 player fields):     top 50 + ties
  - Majors: varies per championship
  - Playoffs / invitationals / no-cut formats:   no cut

Usage:
    from model.cut_rules import get_cut_rule
    rule = get_cut_rule("Valero Texas Open", field_size=144)
    sim = simulate_tournament(field, cut_top_n=rule.cut_n, apply_cut=rule.apply_cut)
"""

import logging
from dataclasses import dataclass
from difflib import get_close_matches

log = logging.getLogger(__name__)


@dataclass
class CutRule:
    cut_n: int          # number of players who make the cut (before ties)
    apply_cut: bool     # whether a cut exists at all
    label: str          # human-readable description


# ── No-cut events ────────────────────────────────────────────────────────────
NO_CUT = CutRule(cut_n=0, apply_cut=False, label="no cut")

# ── Standard cut rules ───────────────────────────────────────────────────────
TOP_65 = CutRule(cut_n=65, apply_cut=True, label="top 65 + ties")
TOP_50 = CutRule(cut_n=50, apply_cut=True, label="top 50 + ties")
TOP_60 = CutRule(cut_n=60, apply_cut=True, label="top 60 + ties")
TOP_70 = CutRule(cut_n=70, apply_cut=True, label="top 70 + ties")


# ── Event-specific overrides ─────────────────────────────────────────────────
# Keys are lowercase event name fragments matched via substring/fuzzy search.
# More specific entries are checked first.
#
# Sources:
#   PGA Tour competition regulations (2024-2026)
#   https://www.pgatour.com/tournaments

EVENT_CUT_RULES: dict[str, CutRule] = {
    # ── Majors ────────────────────────────────────────────────────────────
    "masters":                          TOP_50,
    "pga championship":                 TOP_65,
    "u.s. open":                        TOP_60,
    "us open":                          TOP_60,
    "the open":                         TOP_70,
    "open championship":                TOP_70,

    # ── Signature Events (smaller fields, top 50 + ties) ─────────────────
    "sentry":                           NO_CUT,     # Sentry TOC — 60-player no-cut
    "at&t pebble beach":                TOP_50,
    "genesis invitational":             TOP_50,
    "arnold palmer invitational":       TOP_50,
    "rbc heritage":                     TOP_50,
    "wells fargo":                      TOP_50,
    "memorial tournament":              TOP_50,
    "travelers championship":           TOP_50,
    "the players championship":         TOP_65,     # Players is top 65 (full field)
    "fedex st. jude":                   TOP_50,

    # ── No-cut events ────────────────────────────────────────────────────
    "tour championship":                NO_CUT,     # 30-player field, no cut
    "hero world challenge":             NO_CUT,     # 20-player invitational
    "wgc":                              NO_CUT,
    "match play":                       NO_CUT,

    # ── Opposite-field / regular events (top 65 + ties) ──────────────────
    # These use the default, but listing a few explicitly for clarity
    "valero texas open":                TOP_65,
    "houston open":                     TOP_65,
    "sony open":                        TOP_65,
    "farmers insurance":                TOP_65,
    "wm phoenix open":                  TOP_65,
    "valspar championship":             TOP_65,
    "cognizant classic":                TOP_65,
    "puerto rico open":                 TOP_65,
    "john deere classic":               TOP_65,
    "rocket mortgage classic":          TOP_65,
    "3m open":                          TOP_65,
    "wyndham championship":             TOP_65,
    "barbasol championship":            TOP_65,
    "barracuda championship":           TOP_65,
    "american express":                 TOP_65,
    "shriners":                         TOP_65,
    "sanderson farms":                  TOP_65,
    "bermuda":                          TOP_65,
    "rbc canadian open":                TOP_65,
    "byron nelson":                     TOP_65,
    "charles schwab challenge":         TOP_65,
    "zozo":                             TOP_65,
    "mexico open":                      TOP_65,
    "genesis scottish open":            TOP_65,
    "myrtle beach":                     TOP_65,
}


def get_cut_rule(event_name: str, field_size: int = 0) -> CutRule:
    """
    Resolve the cut rule for a given event.

    Lookup order:
      1. Substring match against EVENT_CUT_RULES keys
      2. Fuzzy match (difflib, cutoff=0.65)
      3. Field-size heuristic: <= 80 players -> top 50, else top 65

    Parameters
    ----------
    event_name : str
        Tournament name (e.g., "Valero Texas Open", "Masters Tournament").
    field_size : int
        Number of players in the field. Used as fallback heuristic.

    Returns
    -------
    CutRule with cut_n, apply_cut, and label.
    """
    if not event_name:
        return TOP_65

    query = event_name.strip().lower()

    # 1. Substring match (check longer keys first for specificity)
    for key in sorted(EVENT_CUT_RULES.keys(), key=len, reverse=True):
        if key in query:
            rule = EVENT_CUT_RULES[key]
            log.info(f"  Cut rule: '{event_name}' -> {rule.label} (matched '{key}')")
            return rule

    # 2. Fuzzy match
    all_keys = list(EVENT_CUT_RULES.keys())
    close = get_close_matches(query, all_keys, n=1, cutoff=0.65)
    if close:
        rule = EVENT_CUT_RULES[close[0]]
        log.info(f"  Cut rule: '{event_name}' -> {rule.label} (fuzzy via '{close[0]}')")
        return rule

    # 3. Field-size heuristic
    if field_size > 0 and field_size <= 80:
        log.info(f"  Cut rule: '{event_name}' -> top 50 + ties (small field heuristic, {field_size} players)")
        return TOP_50

    log.info(f"  Cut rule: '{event_name}' -> top 65 + ties (default)")
    return TOP_65
