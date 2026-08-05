"""Variance risk premium (implied vs. realized vol) on ETF options.

For each trading day, builds a VIX-style constant-30-calendar-day-maturity
ATM implied vol by interpolating TOTAL VARIANCE (iv^2 * days) between the two
listed expiries bracketing 30 days, then compares it to the realized vol that
actually shows up over the matching forward 30 calendar days.

    VRP = IV_30 - RV_30_fwd

Positive VRP means option sellers were, on average, overpaid for the
volatility that materialized -- the raw edge a premium-selling operation
(iron condor / butterfly) is trying to harvest. This does NOT model
transaction costs, bid/ask, or gamma-hedging error, so treat the sign and
rough magnitude as a first-order gut-check, not a tradeable P&L estimate.

Usage:
    python vrp_analysis.py --underlying 510050
    python vrp_analysis.py --underlying 510300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # archived: options_lab/ modules live one level up

import data_store as ds

TARGET_DAYS = 30  # constant-maturity target, calendar days
OUT_DIR = Path(__file__).resolve().parent / "out"

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"    # categorical slot 1 -- IV_30 / positive VRP
C_AQUA = "#1baf7a"    # categorical slot 2 -- RV_30_fwd
C_RED = "#e34948"     # diverging pole -- negative VRP


# ------------------------------------------------------------------ VRP series
def atm_iv_by_expiry(chain_day: pd.DataFrame, spot: float) -> pd.DataFrame:
    """One ATM IV per expiry for one trading day (avg of call+put IV at the
    strike closest to spot), plus each expiry's DTE."""
    out = []
    for expiry, g in chain_day.groupby(ds.C_EXPIRY):
        strikes = g[ds.C_STRIKE].unique()
        atm_strike = strikes[np.argmin(np.abs(strikes - spot))]
        atm = g[g[ds.C_STRIKE] == atm_strike]
        iv = atm[ds.C_IV].mean()
        dte = atm[ds.C_DTE].iloc[0]
        if pd.notna(iv) and iv > 0 and pd.notna(dte) and dte > 0:
            out.append({"dte": dte, "iv": iv})
    return pd.DataFrame(out).sort_values("dte")


def constant_maturity_iv(exp_df: pd.DataFrame,
                         target_days: int = TARGET_DAYS) -> float | None:
    """VIX-style: interpolate TOTAL VARIANCE (iv^2 * days) linearly between
    the two expiries bracketing the target maturity, then re-annualize."""
    if len(exp_df) < 2:
        return None
    near = exp_df[exp_df["dte"] <= target_days].tail(1)
    nxt = exp_df[exp_df["dte"] >= target_days].head(1)
    if near.empty or nxt.empty:
        return None
    d1, iv1 = near["dte"].iloc[0], near["iv"].iloc[0]
    d2, iv2 = nxt["dte"].iloc[0], nxt["iv"].iloc[0]
    if d1 == d2:
        return float(iv1)
    tv1, tv2 = iv1 ** 2 * d1, iv2 ** 2 * d2
    w = (d2 - target_days) / (d2 - d1)
    tv_t = w * tv1 + (1 - w) * tv2
    return float(np.sqrt(max(tv_t, 0) / target_days))


def build_vrp_series(underlying: str) -> pd.DataFrame:
    chain = ds.load_chain(underlying)
    etf = ds.load_etf_ohlc(underlying).set_index("date")["close"].sort_index()
    logret = np.log(etf).diff()

    rows = []
    for d, day_df in chain.groupby(ds.C_DATE):
        spot = day_df[ds.C_UND_CLOSE].iloc[0]
        if pd.isna(spot):
            continue
        exp_df = atm_iv_by_expiry(day_df, spot)
        iv30 = constant_maturity_iv(exp_df)
        if iv30 is not None:
            rows.append({"date": d, "iv30": iv30})
    iv_df = pd.DataFrame(rows).set_index("date").sort_index()

    rv = pd.Series(index=iv_df.index, dtype=float)
    for d in iv_df.index:
        end = d + pd.Timedelta(days=TARGET_DAYS)
        window = logret[(logret.index > d) & (logret.index <= end)]
        if len(window) < TARGET_DAYS * 0.5:  # require reasonable trading-day coverage
            continue
        rv.loc[d] = window.std(ddof=1) * np.sqrt(252)
    iv_df["rv30_fwd"] = rv
    iv_df["vrp"] = iv_df["iv30"] - iv_df["rv30_fwd"]
    return iv_df.dropna(subset=["vrp"])


# ------------------------------------------------------------------ report
def report(df: pd.DataFrame, underlying: str) -> pd.DataFrame:
    print(f"{underlying}: {len(df)} days with a valid 30d VRP reading, "
          f"{df.index.min().date()} -> {df.index.max().date()}")

    mean, median = df["vrp"].mean(), df["vrp"].median()
    tstat, pval = stats.ttest_1samp(df["vrp"], 0)
    winrate = (df["vrp"] > 0).mean()
    print("\nFull-sample VRP (IV_30 - RV_30_fwd, annualized vol points):")
    print(f"  mean={mean:+.2%}  median={median:+.2%}  win%={winrate:.0%}  "
          f"t={tstat:+.2f}  p={pval:.4f}")
    print(f"  mean IV_30={df['iv30'].mean():.2%}  "
          f"mean RV_30_fwd={df['rv30_fwd'].mean():.2%}")

    yearly = df.groupby(df.index.year).agg(
        n=("vrp", "size"), iv30=("iv30", "mean"), rv30=("rv30_fwd", "mean"),
        vrp_mean=("vrp", "mean"), win=("vrp", lambda s: (s > 0).mean()))
    print("\nBy year:")
    for yr, r in yearly.iterrows():
        print(f"  {yr}: n={int(r['n']):>4}  IV={r['iv30']:.2%}  RV={r['rv30']:.2%}  "
              f"VRP={r['vrp_mean']:+.2%}  win%={r['win']:.0%}")

    yrs = yearly.index.to_numpy(dtype=float)
    slope, intercept, r, p, se = stats.linregress(yrs, yearly["vrp_mean"])
    if p >= 0.10:
        verdict = "no clear trend"
    else:
        verdict = "increasing" if slope > 0 else "decaying"
    print(f"\nYearly VRP trend: slope={slope:+.4%}/yr  p={p:.3f}  ({verdict})")
    return yearly


# ------------------------------------------------------------------ plot
def plot(df: pd.DataFrame, underlying: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1]}, facecolor=C_SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C_AXIS)
        ax.spines["bottom"].set_color(C_AXIS)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    ax1.plot(df.index, df["iv30"], color=C_BLUE, linewidth=1.6,
             label="IV (30d constant maturity)")
    ax1.plot(df.index, df["rv30_fwd"], color=C_AQUA, linewidth=1.6,
             label="Realized vol (forward 30d)")
    ax1.set_ylabel("annualized vol", color=C_INK2, fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=C_INK2)
    ax1.set_title(f"{underlying}: implied vs. realized vol", color=C_INK,
                 fontsize=12, loc="left", pad=10)

    pos = df["vrp"].clip(lower=0)
    neg = df["vrp"].clip(upper=0)
    ax2.fill_between(df.index, 0, pos, color=C_BLUE, alpha=0.55, linewidth=0)
    ax2.fill_between(df.index, 0, neg, color=C_RED, alpha=0.55, linewidth=0)
    ax2.axhline(0, color=C_AXIS, linewidth=1)
    ax2.set_ylabel("VRP = IV − RV", color=C_INK2, fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_title("Variance risk premium (blue = sellers overpaid, "
                 "red = sellers underpaid)", color=C_INK, fontsize=10,
                 loc="left", pad=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(underlying: str) -> None:
    df = build_vrp_series(underlying)
    report(df, underlying)
    out_path = OUT_DIR / f"vrp_{underlying}.png"
    plot(df, underlying, out_path)
    print(f"\nChart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    args = ap.parse_args()
    main(args.underlying)
