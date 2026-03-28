"""
analysis/backtest_report.py
----------------------------
Visualize backtest results: ROI heatmaps, weight sensitivity,
comparison vs baselines.

Usage:
    python analysis/backtest_report.py
    python analysis/backtest_report.py --save   # save plots to files
"""

import sys
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pymongo import MongoClient

sys.path.insert(0, ".")
from config.settings import MONGODB_URI, DB_NAME, COLLECTIONS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = MongoClient(MONGODB_URI)
db = client[DB_NAME]

# Style
plt.style.use("dark_background")
ACCENT_COLOR = "#00D4AA"
WARN_COLOR   = "#FF6B35"


# ══════════════════════════════════════════════════════════════════════════════
# Load backtest data
# ══════════════════════════════════════════════════════════════════════════════

def load_backtest_results() -> pd.DataFrame:
    docs = list(db[COLLECTIONS["backtest_runs"]].find(
        {},
        {"sg_ott": 1, "sg_app": 1, "sg_arg": 1, "sg_putt": 1,
         "roi": 1, "n_bets": 1, "net_units": 1, "edge_threshold": 1}
    ))
    if not docs:
        log.error("No backtest results found. Run backtester/weight_optimizer.py first.")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    log.info(f"Loaded {len(df)} backtest runs")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ROI Heatmap: APP vs OTT (best ROI at each pairing)
# ══════════════════════════════════════════════════════════════════════════════

def plot_roi_heatmap(df: pd.DataFrame, save: bool = False):
    """2D heatmap of ROI across sg_app (y-axis) vs sg_ott (x-axis)."""
    pivot = (
        df.groupby(["sg_ott", "sg_app"])["roi"]
        .max()
        .unstack("sg_ott")
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "ROI (flat unit)"},
    )
    ax.set_title("Backtest ROI: SG:APP vs SG:OTT weight combinations\n"
                 "(each cell = max ROI across all ARG/PUTT combos)", pad=15)
    ax.set_xlabel("SG:Off the Tee weight")
    ax.set_ylabel("SG:Approach weight")
    plt.tight_layout()

    if save:
        plt.savefig("analysis/roi_heatmap_app_ott.png", dpi=150)
        log.info("Saved roi_heatmap_app_ott.png")
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Putting weight sensitivity: ROI vs sg_putt
# ══════════════════════════════════════════════════════════════════════════════

def plot_putting_sensitivity(df: pd.DataFrame, save: bool = False):
    """Show ROI distribution at each sg_putt value. Tests putting-noise hypothesis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Box plot: ROI distribution per putting weight
    putt_groups = df.groupby("sg_putt")["roi"].apply(list)
    ax = axes[0]
    bp = ax.boxplot(
        [putt_groups[k] for k in sorted(putt_groups.index)],
        labels=[f"{k:.2f}" for k in sorted(putt_groups.index)],
        patch_artist=True,
        medianprops=dict(color=ACCENT_COLOR, linewidth=2),
        boxprops=dict(facecolor="#1E293B", color="#475569"),
        whiskerprops=dict(color="#475569"),
        capprops=dict(color="#475569"),
        flierprops=dict(marker=".", color="#94A3B8", alpha=0.4),
    )
    ax.set_title("ROI Distribution by SG:Putt Weight\n(supports putting-noise hypothesis if 0.0 is best)")
    ax.set_xlabel("SG:Putt weight")
    ax.set_ylabel("ROI")
    ax.axhline(0, color=WARN_COLOR, linestyle="--", alpha=0.7, label="Break-even")
    ax.legend()

    # Mean ROI line by putting weight
    ax2 = axes[1]
    mean_roi_by_putt = df.groupby("sg_putt")["roi"].agg(["mean", "std"]).reset_index()
    ax2.plot(mean_roi_by_putt["sg_putt"], mean_roi_by_putt["mean"],
             color=ACCENT_COLOR, marker="o", linewidth=2, label="Mean ROI")
    ax2.fill_between(
        mean_roi_by_putt["sg_putt"],
        mean_roi_by_putt["mean"] - mean_roi_by_putt["std"],
        mean_roi_by_putt["mean"] + mean_roi_by_putt["std"],
        alpha=0.2, color=ACCENT_COLOR, label="±1 std"
    )
    ax2.axhline(0, color=WARN_COLOR, linestyle="--", alpha=0.7)
    ax2.set_title("Mean ROI vs SG:Putt Weight")
    ax2.set_xlabel("SG:Putt weight")
    ax2.set_ylabel("Mean ROI")
    ax2.legend()

    plt.tight_layout()

    if save:
        plt.savefig("analysis/putting_sensitivity.png", dpi=150)
        log.info("Saved putting_sensitivity.png")
    else:
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Top combos table
# ══════════════════════════════════════════════════════════════════════════════

def print_top_combos(df: pd.DataFrame, n: int = 20):
    top = df.nlargest(n, "roi")[["sg_ott", "sg_app", "sg_arg", "sg_putt", "roi", "n_bets"]]
    top["roi_pct"] = (top["roi"] * 100).round(2)
    top = top.drop(columns="roi")

    print(f"\n{'═'*65}")
    print(f"Top {n} SG Weight Combinations by Backtest ROI")
    print(f"{'═'*65}")
    print(top.to_string(index=False))

    best = top.iloc[0]
    print(f"\n✓ Best: OTT={best.sg_ott} | APP={best.sg_app} | "
          f"ARG={best.sg_arg} | PUTT={best.sg_putt} → {best.roi_pct:.2f}% ROI")

    # Putting hypothesis check
    best_with_putting    = df[df["sg_putt"] > 0]["roi"].max()
    best_without_putting = df[df["sg_putt"] == 0]["roi"].max()
    print(f"\nPutting hypothesis:")
    print(f"  Best ROI with putting weight > 0:  {best_with_putting*100:.2f}%")
    print(f"  Best ROI with putting weight = 0:  {best_without_putting*100:.2f}%")
    if best_without_putting >= best_with_putting:
        print("  → Hypothesis CONFIRMED: SG:Putt = 0 is optimal or tied")
    else:
        print("  → Hypothesis REJECTED: Putting adds predictive value")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save plots to files")
    args = parser.parse_args()

    df = load_backtest_results()
    if df.empty:
        return

    print_top_combos(df)
    plot_roi_heatmap(df, save=args.save)
    plot_putting_sensitivity(df, save=args.save)


if __name__ == "__main__":
    main()
