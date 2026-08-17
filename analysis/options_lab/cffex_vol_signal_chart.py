# -*- coding: utf-8 -*-
"""
cffex_vol_signal_chart.py -- signal-gated long straddle on CFFEX index options
=============================================================================
4-panel interactive chart (monthly return on debit / monthly P&L in yuan /
index spot / SSE Composite) for `cffex_vol_signal_transfer.py`'s stage-B
frozen pick on IO / MO / HO. Each panel has a show/hide button that
collapses its row rather than leaving blank space -- see
chart_common.build_panel_toggle_script().

Config charted -- the one the verdict is about: `vrp <= 0.00, max_hold=30,
profit_target=0.5`, IO's own stage-B frozen pick (see cffex_vol_signal_
transfer.py and the memo). MO and HO are built with the SAME config rather
than a fitted one, because the memo's finding is precisely that it does not
carry over to them -- the charts are meant to show that contrast, not to
present three separate strategies.

This is a separate script from `long_straddle_chart.py` rather than an
extension of it because that one charts the six live ETF picks off
`alerts_core.VOL_SIGNAL_ALERTS` and hardcodes `MULTIPLIER = ls.ds.MULTIPLIER`
(10000, the ETF options multiplier). CFFEX index options are 100 yuan per
index point, so every yuan figure would be 100x wrong; and there is no
frozen CFFEX pick in VOL_SIGNAL_ALERTS for it to resolve against.

Panel 2 is the REAL index close from `Data/index/index_daily_{code}.csv`
(written by `index_price_updater.py`), not the put-call-parity forward proxy
that `expiry_week_atm_straddle_chart.py` had to reconstruct -- that cache did
not exist when this lab last charted a CFFEX strategy.

Denominator is the debit (premium paid); `compounding=False`, same reasoning
as `long_straddle_chart.py` (premium at risk is small and flat per trade, not
a reinvested pool).

Output:
    analysis/options_lab/out/cffex_vol_signal_000300_vrp_vs_underlying_sse.html
    analysis/options_lab/out/cffex_vol_signal_000852_vrp_vs_underlying_sse.html
    analysis/options_lab/out/cffex_vol_signal_000016_vrp_vs_underlying_sse.html

Usage:
    python cffex_vol_signal_chart.py                     # all three
    python cffex_vol_signal_chart.py --underlying 000300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chart_common as cc  # noqa: E402
import data_store as ds  # noqa: E402
import long_straddle_rv_contraction_backtest as ls  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent.parent
SSE_CSV = ROOT / "Data" / "index" / "index_daily_000001.csv"

# The configuration the verdict is about: IO's own stage-B frozen pick --
# best discovery mean_r with n>=MIN_CELL_N over the 64-cell vrp grid, then
# scored once on holdout (+11.05% disc -> +19.47% holdout). vrp is the only
# signal whose grid-level discovery/holdout agreement survives the strict
# lag-1 convention (30 agree-positive vs 4 agree-negative of 48 eligible
# cells) and whose threshold neighbourhood is monotone rather than a spike.
SIGNAL = "vrp"
THRESHOLD = 0.00
MAX_HOLD = 30
PROFIT_TARGET = 0.5

PLOT_NAMES = {"000300": "IO (CSI300)", "000852": "MO (CSI1000)", "000016": "HO (SSE50)"}
PANEL2 = {"000300": "CSI300 index", "000852": "CSI1000 index", "000016": "SSE50 index"}


def _daily_pnl_debit(trades: pd.DataFrame, equity: pd.DataFrame,
                     multiplier: int) -> pd.DataFrame:
    """Same entry/exit-window mask as chart_common.attach_daily_pnl_margin,
    keyed on "debit" instead of "margin_yuan" -- a long straddle posts no
    margin, so percentage return is on the premium at risk. Identical to
    long_straddle_chart._daily_pnl_debit; only the multiplier differs."""
    equity = equity.copy()
    equity["date"] = pd.to_datetime(equity["date"])
    equity = equity.sort_values("date").reset_index(drop=True)
    equity["pnl_yuan"] = equity["equity"].diff().fillna(0.0) * multiplier

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    equity["margin_yuan"] = np.nan
    for _, t in trades.iterrows():
        mask = (equity["date"] >= t["entry_date"]) & (equity["date"] <= t["exit_date"])
        equity.loc[mask, "margin_yuan"] = t["debit"] * multiplier
    return equity


def load(underlying: str):
    res = ls.run(underlying, signal=SIGNAL, threshold=THRESHOLD,
                 max_hold_days=MAX_HOLD, profit_target=PROFIT_TARGET)
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        raise SystemExit(f"no trades for {underlying}/{SIGNAL}")
    mult = ls._multiplier(underlying)
    equity = _daily_pnl_debit(trades, equity, mult)
    monthly = cc.monthly_return_series(equity)
    start, end = equity["date"].iloc[0], equity["date"].iloc[-1]
    index_px = cc.load_price_series(ds.index_ohlc_path(underlying), start, end)
    sse = cc.load_price_series(SSE_CSV, start, end)
    return monthly, index_px, sse, trades


def build(underlying: str) -> Path:
    monthly, index_px, sse, trades = load(underlying)
    r = trades["pnl"] / trades["debit"]
    title = (f"{PLOT_NAMES[underlying]}: long straddle gated on "
             f"{SIGNAL}&le;{THRESHOLD:+.2f}, max_hold={MAX_HOLD}d, "
             f"profit_target={PROFIT_TARGET:+.0%} &mdash; n={len(trades)}, "
             f"mean return on debit {r.mean():+.1%} "
             f"[IO stage-B frozen pick; MO/HO show the same config, not a refit]")
    # 4-panel variant (return % / P&L yuan / spot / SSE) with collapsible
    # show-hide buttons -- trying this out here first before rolling the
    # pattern out to chart_common's other 6 callers, per the discussion that
    # prompted it (a debit-driven month can show a big % on a small yuan
    # figure or vice versa, e.g. the 2024-10 trade: -84.9% on a 42,680-yuan
    # debit -- 3-4x this run's typical size -- see cffex_vol_signal_
    # investigation.md's discussion of that trade).
    div_id = f"cffex-vol-{underlying}-{SIGNAL}-chart"
    panel1_title = f"long straddle [{SIGNAL}]: monthly return on debit (mark-to-market)"
    panel2_title = "monthly P&L (yuan)"
    fig, annotation_idx = cc.build_four_panel_figure(
        monthly, index_px, sse,
        chart_title=title,
        panel1_title=panel1_title,
        panel2_title=panel2_title,
        panel3_label=PANEL2[underlying],
        compounding=False)
    OUT.mkdir(exist_ok=True)
    out_html = OUT / f"cffex_vol_signal_{underlying}_{SIGNAL}_vs_underlying_sse.html"
    toggle_script = cc.build_panel_toggle_script(
        [panel1_title, panel2_title, PANEL2[underlying], "SSE Composite (000001)"], div_id)
    cc.write_interactive_html(fig, monthly, annotation_idx, out_html,
                              div_id=div_id, compounding=False,
                              axis_names=["xaxis", "xaxis2", "xaxis3", "xaxis4"],
                              extra_script=toggle_script, responsive=False)
    print(f"wrote {out_html} ({len(monthly)} months, "
          f"{len(monthly[monthly['return_pct'] < 0])} losing)")
    return out_html


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default=None, choices=list(ls.CFFEX_UNDERLYINGS),
                    help="default: all three")
    args = ap.parse_args()
    for u in ([args.underlying] if args.underlying else list(ls.CFFEX_UNDERLYINGS)):
        build(u)
