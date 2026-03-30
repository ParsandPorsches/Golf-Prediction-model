"""
analysis/bet_report.py
-----------------------
Generates a clean HTML report from the bet_log collection.

Output is written to:
    outputs/{year}_{event-slug}/bet_report.html

Usage:
    python analysis/bet_report.py --event_id 20
    python analysis/bet_report.py --event_id 20 --open   # auto-opens in browser
"""

import re
import sys
import argparse
import webbrowser
from datetime import datetime
from pathlib import Path

from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import MONGODB_URI, DB_NAME, COLLECTIONS

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]


def event_output_dir(event_id: int) -> Path:
    """Return (and create) outputs/{year}_{event-slug}/ for the given event."""
    year = datetime.now().year
    # Try to get event name from model_predictions
    pred = db[COLLECTIONS["model_predictions"]].find_one({"event_id": event_id})
    event_name = pred.get("event_name", f"event{event_id}") if pred else f"event{event_id}"
    slug = re.sub(r"[^a-z0-9]+", "-", event_name.lower()).strip("-")
    folder = Path("outputs") / f"{year}_{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def fetch_bets(event_id: int) -> list:
    bets = list(db[COLLECTIONS["bet_log"]].find(
        {"event_id": event_id},
        {"_id": 0, "player_name": 1, "market": 1, "model_prob": 1,
         "book_prob": 1, "edge": 1, "american_odds": 1, "kelly_pct": 1,
         "outcome": 1, "pnl": 1}
    ))
    return sorted(bets, key=lambda x: x.get("edge", 0), reverse=True)


def outcome_badge(outcome: str, pnl) -> str:
    if outcome == "win":
        return f'<span class="badge win">WIN</span>'
    elif outcome == "loss":
        return f'<span class="badge loss">LOSS</span>'
    elif outcome == "void":
        return f'<span class="badge void">VOID</span>'
    else:
        return f'<span class="badge pending">PENDING</span>'


def pnl_cell(pnl) -> str:
    if pnl is None:
        return '<td class="neutral">—</td>'
    color = "profit" if pnl >= 0 else "loss-text"
    sign = "+" if pnl >= 0 else ""
    return f'<td class="{color}">{sign}{pnl:.2f}u</td>'


def build_rows(bets: list) -> str:
    rows = []
    for b in bets:
        kelly = b.get("kelly_pct", 0) or 0
        edge  = b.get("edge", 0) or 0
        row_class = "actionable" if kelly > 0 else ""

        rows.append(f"""
        <tr class="{row_class}">
            <td class="player">{b.get('player_name', '')}</td>
            <td><span class="market-tag">{b.get('market','').upper().replace('_',' ')}</span></td>
            <td>{b.get('model_prob', 0)*100:.1f}%</td>
            <td>{b.get('book_prob',  0)*100:.1f}%</td>
            <td class="edge">+{edge*100:.1f}%</td>
            <td class="odds">{b.get('american_odds','')}</td>
            <td class="{'kelly-pos' if kelly > 0 else 'neutral'}">{kelly*100:.1f}%</td>
            <td>{outcome_badge(b.get('outcome','pending'), b.get('pnl'))}</td>
            {pnl_cell(b.get('pnl'))}
        </tr>""")
    return "\n".join(rows)


def build_summary(bets: list) -> dict:
    actionable = [b for b in bets if (b.get("kelly_pct") or 0) > 0]
    wins   = [b for b in bets if b.get("outcome") == "win"]
    losses = [b for b in bets if b.get("outcome") == "loss"]
    settled_pnl = sum(b.get("pnl") or 0 for b in bets if b.get("pnl") is not None)
    return {
        "total":      len(bets),
        "actionable": len(actionable),
        "wins":       len(wins),
        "losses":     len(losses),
        "pnl":        settled_pnl,
    }


def generate_html(event_id: int, bets: list) -> str:
    summary = build_summary(bets)
    rows    = build_rows(bets)
    pnl_color = "profit" if summary["pnl"] >= 0 else "loss-text"
    pnl_sign  = "+" if summary["pnl"] >= 0 else ""
    generated = datetime.now().strftime("%B %d, %Y %I:%M %p")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Golf Model — Bet Report (Event {event_id})</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f1923;
    color: #e0e6ed;
    padding: 32px;
  }}
  h1 {{
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 4px;
  }}
  .subtitle {{
    color: #7a8a99;
    font-size: 0.85rem;
    margin-bottom: 28px;
  }}

  /* Summary cards */
  .summary {{
    display: flex;
    gap: 16px;
    margin-bottom: 28px;
    flex-wrap: wrap;
  }}
  .card {{
    background: #1a2634;
    border: 1px solid #253545;
    border-radius: 10px;
    padding: 16px 24px;
    min-width: 130px;
    text-align: center;
  }}
  .card .label {{ font-size: 0.72rem; color: #7a8a99; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; color: #fff; }}
  .card .value.profit  {{ color: #2ecc71; }}
  .card .value.loss-text {{ color: #e74c3c; }}

  /* Legend */
  .legend {{
    font-size: 0.78rem;
    color: #7a8a99;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .legend-dot {{
    width: 10px; height: 10px;
    border-radius: 2px;
    background: rgba(46,204,113,0.15);
    border: 1px solid rgba(46,204,113,0.4);
    display: inline-block;
  }}

  /* Table */
  .table-wrap {{ overflow-x: auto; border-radius: 10px; border: 1px solid #253545; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  thead tr {{ background: #1a2634; }}
  thead th {{
    padding: 12px 14px;
    text-align: left;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #7a8a99;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}
  thead th:hover {{ color: #c0cdd8; }}
  thead th::after {{ content: ' \\2195'; opacity: .3; }}
  tbody tr {{ border-top: 1px solid #1e2d3d; transition: background .12s; }}
  tbody tr:hover {{ background: #1e2d3d; }}
  tbody tr.actionable {{ background: rgba(46,204,113,0.05); }}
  tbody tr.actionable:hover {{ background: rgba(46,204,113,0.10); }}
  td {{ padding: 10px 14px; vertical-align: middle; }}

  .player {{ font-weight: 600; color: #fff; white-space: nowrap; }}
  .market-tag {{
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    background: #253545;
    color: #a0b0c0;
    white-space: nowrap;
  }}
  .edge      {{ color: #2ecc71; font-weight: 700; }}
  .odds      {{ color: #f39c12; font-weight: 600; font-family: monospace; }}
  .kelly-pos {{ color: #2ecc71; font-weight: 700; }}
  .neutral   {{ color: #7a8a99; }}
  .profit    {{ color: #2ecc71; font-weight: 700; }}
  .loss-text {{ color: #e74c3c; font-weight: 700; }}

  .badge {{
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: .05em;
  }}
  .badge.win     {{ background: rgba(46,204,113,.2); color: #2ecc71; }}
  .badge.loss    {{ background: rgba(231,76, 60,.2); color: #e74c3c; }}
  .badge.void    {{ background: rgba(149,165,166,.2); color: #95a5a6; }}
  .badge.pending {{ background: rgba(241,196,15,.15); color: #f1c40f; }}

  .footer {{
    margin-top: 20px;
    font-size: 0.75rem;
    color: #3d5066;
    text-align: right;
  }}
</style>
</head>
<body>

<h1>Golf Model &mdash; Value Bet Report</h1>
<div class="subtitle">Event ID {event_id} &nbsp;&bull;&nbsp; Generated {generated}</div>

<div class="summary">
  <div class="card">
    <div class="label">Total Bets</div>
    <div class="value">{summary['total']}</div>
  </div>
  <div class="card">
    <div class="label">Actionable</div>
    <div class="value" style="color:#f1c40f">{summary['actionable']}</div>
  </div>
  <div class="card">
    <div class="label">Wins</div>
    <div class="value profit">{summary['wins']}</div>
  </div>
  <div class="card">
    <div class="label">Losses</div>
    <div class="value loss-text">{summary['losses']}</div>
  </div>
  <div class="card">
    <div class="label">Net P&amp;L</div>
    <div class="value {pnl_color}">{pnl_sign}{summary['pnl']:.2f}u</div>
  </div>
</div>

<div class="legend">
  <span class="legend-dot"></span>
  Highlighted rows = positive Kelly (actionable bets)
  &nbsp;&bull;&nbsp; Click any column header to sort
</div>

<div class="table-wrap">
<table id="betTable">
  <thead>
    <tr>
      <th onclick="sortTable(0)">Player</th>
      <th onclick="sortTable(1)">Market</th>
      <th onclick="sortTable(2)">Model %</th>
      <th onclick="sortTable(3)">Book %</th>
      <th onclick="sortTable(4)">Edge</th>
      <th onclick="sortTable(5)">Odds</th>
      <th onclick="sortTable(6)">Kelly</th>
      <th onclick="sortTable(7)">Outcome</th>
      <th onclick="sortTable(8)">P&amp;L</th>
    </tr>
  </thead>
  <tbody>
{rows}
  </tbody>
</table>
</div>

<div class="footer">Golf Model v3 &nbsp;&bull;&nbsp; DraftKings odds &nbsp;&bull;&nbsp; Quarter-Kelly sizing</div>

<script>
  let sortDir = {{}};
  function sortTable(col) {{
    const table = document.getElementById('betTable');
    const tbody = table.querySelector('tbody');
    const rows  = Array.from(tbody.querySelectorAll('tr'));
    const asc   = !sortDir[col];
    sortDir = {{}};
    sortDir[col] = asc;

    rows.sort((a, b) => {{
      const va = a.cells[col].innerText.trim().replace(/[+%u]/g,'');
      const vb = b.cells[col].innerText.trim().replace(/[+%u]/g,'');
      const na = parseFloat(va), nb = parseFloat(vb);
      if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_id", type=int, required=True)
    parser.add_argument("--open", action="store_true", help="Auto-open in browser")
    args = parser.parse_args()

    bets = fetch_bets(args.event_id)
    if not bets:
        print(f"No bets found for event_id={args.event_id}")
        return

    html     = generate_html(args.event_id, bets)
    out_dir  = event_output_dir(args.event_id)
    out_path = out_dir / "bet_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Report saved: {out_path.resolve()}")

    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
