# -*- coding: utf-8 -*-
"""
expiry_week_atm_straddle_chart.py -- 到期周平值双买 vs. context
=============================================================================
3-panel interactive chart (monthly return on premium deployed / index spot /
SSE Composite) for expiry_week_atm_straddle_backtest.py, one file per
underlying.

Config charted -- the concluding one, i.e. the strategy exactly as posed:
all three entry days (Mon + Wed + Fri of the expiry week) held to expiry on
the 当月 contract, `calendar` entry basis, zero slippage (this lab's
convention).

Denominator is the debit (premium paid), not margin -- a long option
position posts none, and cffex_margin.py has verified parameters for MO
only. This is a portfolio of up to three simultaneously open straddles, so
it also does not use chart_common.attach_daily_pnl_margin (a last-writer-
wins per-trade mask, right for one position at a time) or
monthly_return_series: see monthly_on_deployed_debit() below for why the
mean-daily-capital denominator those imply understates committed premium
here and produced impossible sub--100% months. compounding=False for the
same reason long_straddle_chart.py uses it: premium at risk is small and
flat per cycle, not a reinvested pool, so chaining monthly percentages
would fabricate growth that never happened.

Monthly bucketing is exact rather than approximate: every trade opens and
settles inside its own expiry week, so no position crosses a month
boundary.

The Friday leg is included because the question asked for it and because it
contributes almost nothing either way (total P&L -269 / -15 / +344 yuan on
HO / IO / MO), but it is not a tradeable result -- see the backtest
module's docstring on CFFEX settling at the last two hours' index average.

Panel 2 is the near-month put-call-parity forward, not a stored index
level: there is no CSI300/CSI1000/SSE50 daily OHLC file in Data/index/
(only 000001), so this rebuilds the spot series from the chain itself the
same way csi1000_skew_risk_reversal_backtest.build_underlying_price_series
does.

Output:
    analysis/options_lab/out/expiry_week_atm_straddle_000300_vs_underlying_sse.html
    analysis/options_lab/out/expiry_week_atm_straddle_000852_vs_underlying_sse.html
    analysis/options_lab/out/expiry_week_atm_straddle_000016_vs_underlying_sse.html

Usage:
    python expiry_week_atm_straddle_chart.py            # all three
    python expiry_week_atm_straddle_chart.py --underlying 000852
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chart_common as cc
import consolidated_store as cs
import data_store as ds
import expiry_week_atm_straddle_backtest as ew

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "out"
SSE_CSV = ROOT / "Data" / "index" / "index_daily_000001.csv"

EXIT_RULE = "to_expiry"
BASIS = "calendar"
PANEL2_LABELS = {"000300": "CSI300 (near-month forward)",
                 "000852": "CSI1000 (near-month forward)",
                 "000016": "SSE50 (near-month forward)"}


def monthly_on_deployed_debit(trades: pd.DataFrame) -> pd.DataFrame:
    """Monthly frame in chart_common's shape (month / pnl_yuan /
    margin_yuan / return_pct / month_start / month_end), but with the
    denominator being the total premium DEPLOYED that month -- the sum of
    the debits of the straddles bought in it.

    chart_common.monthly_return_series divides by the mean daily capital at
    risk instead, which is right for a single position held continuously
    but wrong here: the three legs stack up across the expiry week (one
    straddle open on Monday, two by Wednesday, three by Friday), so the
    daily mean sits well below the premium actually committed and produced
    monthly returns past -119% -- impossible for a long straddle, whose
    loss is bounded by the debit. On deployed premium the floor is exactly
    -100%, and the number answers the question an options trader actually
    asks: of the money I paid out this cycle, how much came back?

    Every trade opens and settles inside one expiry week, so no position
    crosses a month boundary and the attribution is exact, not smeared."""
    t = trades.copy()
    t["month"] = t["entry_date"].dt.to_period("M")
    g = t.groupby("month").agg(pnl_yuan=("pnl_yuan", "sum"),
                               margin_yuan=("debit_yuan", "sum")).reset_index()
    g["return_pct"] = 100.0 * g["pnl_yuan"] / g["margin_yuan"]
    g["month_start"] = g["month"].dt.to_timestamp()
    g["month_end"] = (g["month"] + 1).dt.to_timestamp()
    return g


def _forward_series(underlying: str, start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.DataFrame:
    """date/close frame of the nearest-expiry parity forward -- panel 2's
    spot proxy, since Data/index/ has no file for these three indices."""
    chain = cs.load_chain(underlying).dropna(subset=[ds.C_UND_CLOSE])
    chain = chain[chain[ds.C_DTE] >= 0]
    nearest = chain.sort_values([ds.C_DATE, ds.C_DTE]).groupby(ds.C_DATE).first()
    s = nearest[ds.C_UND_CLOSE]
    s = s[(s.index >= start) & (s.index <= end)]
    return pd.DataFrame({"date": s.index, "close": s.values})


def load(underlying: str):
    trades = ew.run(underlying, basis=BASIS, slippage=0.0)
    trades = trades[trades["exit_rule"] == EXIT_RULE].copy()
    if trades.empty:
        raise ValueError(f"no {EXIT_RULE} trades for {underlying}")

    monthly = monthly_on_deployed_debit(trades)
    start, end = trades["entry_date"].min(), trades["exit_date"].max()
    return (monthly, _forward_series(underlying, start, end),
            cc.load_price_series(SSE_CSV, start, end), trades)


def build(underlying: str) -> Path:
    monthly, fwd, sse, trades = load(underlying)
    n = len(trades)
    mean_r = trades["return_on_debit"].mean()
    title = (f"{ew.PLOT_NAMES[underlying]}: expiry-week ATM long straddle "
             f"(Mon+Wed+Fri, front month, held to expiry) &mdash; "
             f"n={n}, mean return on debit {mean_r:+.1%}")
    fig, annotation_idx = cc.build_three_panel_figure(
        monthly, fwd, sse,
        chart_title=title,
        panel1_title="monthly return on premium deployed (%)",
        panel2_label=PANEL2_LABELS[underlying],
        compounding=False)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"expiry_week_atm_straddle_{underlying}_vs_underlying_sse.html"
    cc.write_interactive_html(fig, monthly, annotation_idx, out_path,
                              div_id=f"ewstraddle_{underlying}", compounding=False)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default=None, choices=list(ew.UNDERLYINGS),
                    help="default: all three")
    args = ap.parse_args()
    for u in ([args.underlying] if args.underlying else list(ew.UNDERLYINGS)):
        print(f"  {ew.UNDERLYINGS[u]} -> {build(u)}")
