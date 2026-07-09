"""Does gating entries on IV percentile improve the variance risk premium?

Builds on vrp_analysis.py's daily VRP series (IV_30 - realized forward 30d
vol) and joins it against the SAME IV-percentile definition the live engine
already uses for its entry gate (backtest.py: near-month ATM call IV,
rolling 252-trading-day percentile rank, min_periods=60 -- see
Rules.iv_pct_min/max). Buckets VRP outcomes by that percentile to answer:
does only selling premium when IV percentile is high (a) raise the average
premium captured, (b) raise the win rate, and/or (c) shrink the worst-case
tail losses -- vs. the unconditional baseline from vrp_analysis.py?

Usage:
    python iv_percentile_filter.py --underlying 510050
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import backtest as bt
import data_store as ds
import vrp_analysis as vrp

OUT_DIR = Path(__file__).resolve().parent / "out"

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"
C_AQUA = "#1baf7a"

BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
BUCKET_LABELS = ["Q1\n(lowest IV)", "Q2", "Q3", "Q4", "Q5\n(highest IV)"]
THRESHOLDS = [0.0, 0.5, 0.7, 0.8]


def iv_percentile_series(underlying: str) -> pd.Series:
    """Same definition as the live engine's entry gate (backtest.Rules)."""
    chain = ds.load_chain(underlying)
    d = bt._prep(chain, bt.Rules())
    iv_atm = bt._atm_iv_series(d)
    return iv_atm.rolling(252, min_periods=60).rank(pct=True)


def bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = pd.cut(df["iv_pct"], BUCKET_EDGES, labels=BUCKET_LABELS,
                          include_lowest=True)
    g = df.groupby("bucket", observed=True)
    return g.agg(
        n=("vrp", "size"), mean_vrp=("vrp", "mean"), median_vrp=("vrp", "median"),
        win=("vrp", lambda s: (s > 0).mean()),
        tail5=("vrp", lambda s: s.quantile(0.05)),
        mean_iv=("iv30", "mean"), mean_rv=("rv30_fwd", "mean"))


def threshold_stats(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        s = df.loc[df["iv_pct"] >= t, "vrp"]
        rows.append({
            "iv_pct >=": t, "n": len(s), "coverage": len(s) / len(df),
            "mean_vrp": s.mean(), "median_vrp": s.median(),
            "win": (s > 0).mean(), "tail5": s.quantile(0.05),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, underlying: str) -> pd.DataFrame:
    print(f"{underlying}: {len(df)} days with both a VRP reading and an "
          f"IV-percentile reading (needs 60+ trading days of IV history)\n")

    r, p = stats.pearsonr(df["iv_pct"], df["vrp"])
    print(f"Correlation(IV percentile, VRP): r={r:+.3f}  p={p:.4f}\n")

    print("By IV-percentile quintile at entry:")
    bstats = bucket_stats(df)
    for label, r_ in bstats.iterrows():
        print(f"  {label.replace(chr(10), ' '):<16} n={int(r_['n']):>4}  "
              f"IV={r_['mean_iv']:.2%}  RV={r_['mean_rv']:.2%}  "
              f"VRP mean={r_['mean_vrp']:+.2%} median={r_['median_vrp']:+.2%}  "
              f"win%={r_['win']:.0%}  worst-5%={r_['tail5']:+.2%}")

    print("\nSimple threshold filters (only sell when IV percentile >= X):")
    tstats = threshold_stats(df, THRESHOLDS)
    for _, r_ in tstats.iterrows():
        print(f"  iv_pct>={r_['iv_pct >=']:.1f}  coverage={r_['coverage']:.0%}  "
              f"n={int(r_['n']):>4}  VRP mean={r_['mean_vrp']:+.2%} "
              f"median={r_['median_vrp']:+.2%}  win%={r_['win']:.0%}  "
              f"worst-5%={r_['tail5']:+.2%}")
    return bstats


def plot(bstats: pd.DataFrame, underlying: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True, facecolor=C_SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C_AXIS)
        ax.spines["bottom"].set_color(C_AXIS)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    x = np.arange(len(bstats))
    colors = [C_RED if v < 0 else C_BLUE for v in bstats["mean_vrp"]]
    ax1.bar(x, bstats["mean_vrp"], color=colors, width=0.6)
    ax1.axhline(0, color=C_AXIS, linewidth=1)
    ax1.set_ylabel("mean VRP", color=C_INK2, fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_title(f"{underlying}: mean VRP by IV-percentile quintile at entry",
                 color=C_INK, fontsize=12, loc="left", pad=10)

    ax2.bar(x, bstats["win"], color=C_AQUA, width=0.6)
    ax2.axhline(0.5, color=C_AXIS, linewidth=1, linestyle=(0, (3, 2)))
    ax2.set_ylabel("win rate", color=C_INK2, fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xticks(x)
    ax2.set_xticklabels(bstats.index, color=C_MUTED, fontsize=9)
    ax2.set_title("win rate (VRP > 0) by quintile -- dashed line = 50%",
                 color=C_INK, fontsize=10, loc="left", pad=8)

    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def main(underlying: str) -> None:
    vrp_df = vrp.build_vrp_series(underlying)
    iv_pct = iv_percentile_series(underlying)
    df = vrp_df.join(iv_pct.rename("iv_pct"), how="inner").dropna(subset=["iv_pct"])

    bstats = report(df, underlying)
    out_path = OUT_DIR / f"iv_pct_filter_{underlying}.png"
    plot(bstats, underlying, out_path)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    args = ap.parse_args()
    main(args.underlying)
