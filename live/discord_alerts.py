"""
live/discord_alerts.py
-----------------------
Sends value bet alerts to a Discord webhook.
Set DISCORD_WEBHOOK_URL in your .env to enable.
"""

import logging
from datetime import datetime

import requests
import pandas as pd

import sys
sys.path.insert(0, ".")
from config.settings import DISCORD_WEBHOOK_URL

log = logging.getLogger(__name__)


def send_value_bets_alert(value_bets: pd.DataFrame, event_id: int, year: int):
    """Post a value bet summary to Discord."""
    if not DISCORD_WEBHOOK_URL:
        log.info("No Discord webhook configured — skipping alert.")
        return

    if value_bets.empty:
        return

    lines = [f"**Value Bets — Event {event_id} / {year}**", ""]
    for _, bet in value_bets.iterrows():
        lines.append(
            f"**{bet['player_name']}** | {bet['market'].upper()}"
            f"\nModel: {bet['model_prob']*100:.1f}% | "
            f"Book: {bet['book_prob']*100:.1f}% | "
            f"Edge: +{bet['edge']*100:.1f}%"
            f"\nOdds: {bet['decimal_odds']:.2f} ({bet['american_odds']}) | "
            f"Kelly: {bet['kelly_pct']*100:.1f}% bankroll | "
            f"Book: {bet['book'].upper()}"
        )
        lines.append("")

    content = "\n".join(lines)
    # Discord message limit is 2000 chars
    if len(content) > 1900:
        content = content[:1900] + "\n... (truncated)"

    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": content, "username": "Golf Model v3"},
            timeout=10,
        )
        resp.raise_for_status()
        log.info("Discord alert sent.")
    except requests.RequestException as exc:
        log.warning(f"Discord alert failed: {exc}")


def send_simple_message(text: str):
    """Send any plain text message to Discord."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": text, "username": "Golf Model v3"},
            timeout=10,
        )
    except requests.RequestException as exc:
        log.warning(f"Discord message failed: {exc}")
