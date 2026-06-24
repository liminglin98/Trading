# -*- coding: utf-8 -*-
"""
oneside_scan.py -- how often does 50ETF trend one-sided inside the holding window?
=================================================================================
A long butterfly / iron condor is short realized vol and wants the underlying to
CHOP. Its enemy is a sustained one-directional move. This is a census: for every
monthly option cycle since 2015, how "one-sided" was 50ETF over the window the
position actually lives through -- from ~30 DTE to the usual exit (14 or 7 DTE)?

Two readings of "one-sided" (both reported, at 50/70/90% thresholds):

  Efficiency Ratio (Kaufman):  |P_end - P_start| / sum(|daily change|)
     = the fraction of the total path length that was directional.
     ER = 0.90  <->  "90% one-sided" (a clean trend).  ER ~ 0  <->  chop.

  Dominant-day share:  max(up_days, down_days) / total_days
     = the share of sessions that went the dominant way ("90% of days up").

Each trending month is also tagged up / down, because a one-sided DOWN move is
the one that runs through a condor's put wing.

Window: anchored on the SSE 50ETF option expiry (4th Wednesday of the month).
entry = trading day on/before (expiry - 30d); cut = on/before (expiry - 14d) and
(expiry - 7d). Pure price census -- no option data needed.

Run (from inside macro_model\, needs internet for the 50ETF feed):
    python oneside_scan.py
Outputs:
    out/oneside_scan.csv   -- one row per (cycle, cut_dte) with ER + day-share
    printed summary tables  -- frequency above each threshold, up vs down
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import butterfly_timing_backtest as B
ak = B.ak

OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

CUTS = [14, 7]                      # DTE at which the position is usually exited
THRESHOLDS = [0.5, 0.7, 0.9]       # "one-sided" cutoffs to tabulate
ENTRY_DTE = 30                     # window starts ~here


def fourth_wednesday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    first_wed = first + pd.Timedelta(days=(2 - first.weekday()) % 7)   # Wed = 2
    return first_wed + pd.Timedelta(weeks=3)


def monthly_expiries(start="2015-02", end=None) -> list[pd.Timestamp]:
    end = pd.Timestamp.now() if end is None else pd.Timestamp(end)
    months = pd.period_range(start=start, end=end, freq="M")
    return [fourth_wednesday(p.year, p.month) for p in months]


def load_etf() -> pd.Series:
    e = ak.fund_etf_hist_sina(symbol="sh510050").copy()
    e["date"] = pd.to_datetime(e["date"])
    return e.set_index("date")["close"].astype(float).sort_index()


def asof_trading_day(idx: pd.DatetimeIndex, target: pd.Timestamp):
    """Last trading day on or before target, or None if before the data starts."""
    prior = idx[idx <= target]
    return prior[-1] if len(prior) else None


def scan(etf: pd.Series) -> pd.DataFrame:
    idx = etf.index
    rows = []
    for exp in monthly_expiries():
        entry = asof_trading_day(idx, exp - pd.Timedelta(days=ENTRY_DTE))
        if entry is None:
            continue
        for cut_dte in CUTS:
            cut = asof_trading_day(idx, exp - pd.Timedelta(days=cut_dte))
            if cut is None or cut <= entry:
                continue
            path = etf.loc[entry:cut]
            rets = path.diff().dropna()
            n = len(rets)
            if n < 5:
                continue
            net = float(path.iloc[-1] - path.iloc[0])
            gross = float(rets.abs().sum())
            er = abs(net) / gross if gross > 0 else np.nan
            up = int((rets > 0).sum()); dn = int((rets < 0).sum())
            day_share = max(up, dn) / n
            rows.append(dict(
                expiry=exp.date(), cut_dte=cut_dte,
                entry=entry.date(), cut=cut.date(), days=n,
                net_pct=net / float(path.iloc[0]) * 100.0,
                eff_ratio=er, day_share=day_share,
                direction="up" if net > 0 else "down"))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    for cut_dte in CUTS:
        sub = df[df["cut_dte"] == cut_dte]
        if not len(sub):
            continue
        print(f"\n=== window 30DTE -> {cut_dte}DTE   ({len(sub)} monthly cycles, "
              f"{sub['entry'].min()} .. {sub['cut'].max()}) ===")
        print(f"  efficiency ratio: mean {sub['eff_ratio'].mean():.2f}  "
              f"median {sub['eff_ratio'].median():.2f}  "
              f"(avg window {sub['days'].mean():.0f} sessions)")
        print(f"  {'threshold':>9} | {'ER >= t':>16} | {'day-share >= t':>16} | "
              f"{'ER & down':>10}")
        for t in THRESHOLDS:
            er_hit = sub["eff_ratio"] >= t
            ds_hit = sub["day_share"] >= t
            er_dn = int((er_hit & (sub["direction"] == "down")).sum())
            print(f"  {t:>9.0%} | {er_hit.sum():>4} / {len(sub)} "
                  f"({er_hit.mean():>5.1%}) | "
                  f"{ds_hit.sum():>4} / {len(sub)} ({ds_hit.mean():>5.1%}) | "
                  f"{er_dn:>4} down")


def main():
    print("=" * 72)
    print("One-sided / trend census of 50ETF over the short-vol holding window")
    print(f"ENTRY_DTE={ENTRY_DTE}  CUTS={CUTS}  thresholds={THRESHOLDS}")
    print("=" * 72)
    etf = load_etf()
    print(f"50ETF daily: {len(etf)} sessions ({etf.index[0].date()} .. {etf.index[-1].date()})")
    df = scan(etf)
    df.to_csv(OUTDIR / "oneside_scan.csv", index=False)
    summarize(df)
    print(f"\nsaved -> {OUTDIR / 'oneside_scan.csv'}")


if __name__ == "__main__":
    main()
