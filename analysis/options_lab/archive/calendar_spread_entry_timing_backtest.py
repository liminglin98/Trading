"""Calendar spread (日历价差): sell 1x near-month (当月) call/put at strike K,
buy 1x next-month (次月) call/put at the SAME strike K, same side. Entry cost
is normally a net debit (the longer-dated leg costs more time value). Hold
both legs until the near-month leg expires, then close out: near leg settles
at intrinsic, next leg is marked at its close price that day (still ~1 month
of life left).

Mechanism note (read before trusting "big move" framing): this is fundamentally
a PINNING / low-realized-vol bet through the near-month expiry, not a big-move
bet -- max value at near expiry is when spot lands AT strike K (short leg
worthless, long leg still has full time value). A big move before near expiry
hurts this structure (short leg intrinsic eats the spread). ATM (moneyness=0)
is the pure form of this; ITM strikes (moneyness>0) add a directional lean
(you're betting spot ends up near that ITM level) at the cost of extra debit.

Per-trade, not a continuous equity curve: unlike the other backtests in this
repo, this study intentionally samples MANY overlapping candidate entry dates
per monthly cycle (to compare early/mid/late-month entries), so trades from
the same cycle share the same expiry-date outcome path -- read n as "distinct
entry-day observations," not independent episodes; cycles (~1 every month)
are the real independent unit.

Entry timing, take 2 -- signal-gated instead of day-of-month: --signal
{rv_pct,vrp} gates entry on the same "cheap/quiet -> bet on expansion"
signals validated for long straddle in long_straddle_rv_contraction_backtest.py
(rv_pct) and iv_low_long_straddle_backtest.py (vrp, real IV for 510050/510300).
Day-of-month bucketing (early/mid/late) already tested with NO signal gate --
found no edge, closed out in memory -- and stays only as a secondary
breakdown column on gated trades, not a sweep dimension.

Deliberate rigor choice, NOT yet applied to sibling scripts: gates on
YESTERDAY's signal value (the prior trading date in the chain's own date
list), not the entry date's own same-day value. Every other signal-gated
backtest in this repo (long_straddle_rv_contraction_backtest.py, the closed
backspread investigation) gates on the same-day value while pricing the
trade at that same day's close -- a subtle look-ahead consistency issue
(you'd need to know the close before it happens to act on it). This script
avoids that; the others haven't been revisited.

Entry timing, take 3 -- pre-holiday: a paid blog post (Data/日历价差.pdf)
claims opening 1-2 trading days before a long holiday is favorable, on the
theory that IV gets marked down going into days the market can't trade
(near-month leg loses relatively more than next-month). This is a genuinely
different, falsifiable hypothesis from the day-of-month/signal-gated tests
above -- not something already closed out -- so it gets its own comparison:
--holiday-report buckets the unrestricted trade set into "1-2 trading days
before a >=4-calendar-day gap" vs "everything else" and Welch-t-tests the
difference, same discovery/holdout split as everywhere else in this repo.
Long-holiday dates are DERIVED from gaps in the chain's own trading-date
list (>=4 calendar days between consecutive trading dates), not a hardcoded
calendar -- catches mini 3-day holidays (清明/端午/中秋) as well as the major
ones (春节/国庆/五一) the blog post specifically means by "长假"; the report
also splits pre-holiday trades by gap size (major >=7 calendar days vs minor)
so a real effect concentrated in the major holidays doesn't get diluted by
lumping in the mini ones, or vice versa.

Usage:
    python calendar_spread_entry_timing_backtest.py --underlying 510500
    python calendar_spread_entry_timing_backtest.py --underlying 510500 --signal rv_pct
    python calendar_spread_entry_timing_backtest.py --underlying all --signal vrp --sweep
    python calendar_spread_entry_timing_backtest.py --underlying all --holiday-report
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # archived: options_lab/ modules live one level up

import data_store as ds
import quarterly_atm_short_straddle as qs
import long_straddle_rv_contraction_backtest as ls
import iv_low_long_straddle_backtest as il

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = ls.SPLIT_DATE
# 510050/510300 go through data_store's original CSV feed (real exchange
# quotes/Greeks), same convention as quarterly_atm_short_straddle.py and the
# macro_model MA-signal investigation -- NOT consolidated_store, which
# ls._prepped_chain uses for the other five. See reference_data_sources.
LEGACY_UNDERLYINGS = {"510050": "50ETF", "510300": "300ETF"}
UNDERLYINGS = {**LEGACY_UNDERLYINGS, **ls.UNDERLYINGS}

MONEYNESS_PCTS = [0.0, 0.03, 0.05]  # 0.0 = ATM; >0 = ITM by this %% of spot
SIDES = ["C", "P"]
HOLIDAY_MIN_GAP_DAYS = 4  # calendar days; plain weekend = 3, so this excludes weekends
HOLIDAY_MAJOR_GAP_DAYS = 7  # >= this = 春节/国庆/五一-scale; below = mini holidays (清明/端午/中秋)
# Longest real combined holiday seen (中秋+国庆 2023) is 11 calendar days. A gap
# bigger than this in one underlying's own chain (confirmed: 510500 has three
# ~22-23 day gaps around 2024-04, 2024-12, 2025-12 while OTHER underlyings
# trade normally through those windows) is a DATA GAP in that chain, not a
# holiday -- must be excluded or it corrupts the pre-holiday bucket.
HOLIDAY_MAX_GAP_DAYS = 13


def _prepped_chain(underlying: str) -> pd.DataFrame:
    if underlying in LEGACY_UNDERLYINGS:
        chain = ds.load_chain(underlying)
        chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
        return chain[chain[ds.C_CLOSE] > 0]
    return ls._prepped_chain(underlying)


def _signal_series(underlying: str, signal: str) -> pd.Series:
    if signal == "rv_pct":
        return ls.rv_pct_series(underlying)  # ds.load_etf_ohlc-based, works for all 7
    if signal == "vrp":
        if underlying in LEGACY_UNDERLYINGS:
            return il.vrp_trailing_series(underlying)  # real exchange IV
        return ls.vrp_trailing_series(underlying)  # BS-inverted IV via consolidated_store
    raise ValueError(f"unknown signal {signal!r}")


def _nearest_strike_row(day: pd.DataFrame, expiry: pd.Timestamp, cp: str,
                         target_strike: float) -> pd.Series | None:
    m = day[(day[ds.C_EXPIRY] == expiry) & (day[ds.C_CP] == cp) & (day[ds.C_CLOSE] > 0)]
    if m.empty:
        return None
    return m.iloc[(m[ds.C_STRIKE] - target_strike).abs().argsort()].iloc[0]


def _bucket(day_of_month: int) -> str:
    if day_of_month <= 10:
        return "early(1-10)"
    if day_of_month <= 20:
        return "mid(11-20)"
    return "late(21-31)"


def _holiday_offsets(dates: list[pd.Timestamp], min_gap_days: int = HOLIDAY_MIN_GAP_DAYS,
                      max_gap_days: int = HOLIDAY_MAX_GAP_DAYS) -> dict[pd.Timestamp, tuple[int, int]]:
    """Maps a trading date -> (days_before_holiday, gap_days) for the last two
    trading dates before a detected long-holiday gap (min_gap_days <= gap <=
    max_gap_days calendar days to the next trading date -- the upper bound
    excludes data gaps in a single underlying's own chain, which aren't real
    holidays and would otherwise contaminate this bucket, see module
    docstring / HOLIDAY_MAX_GAP_DAYS). days_before_holiday: 0 = last trading
    day before the gap ("1 day before" in the blog's count), 1 = the day
    before that ("2 days before"). Dates not near a detected gap are simply
    absent from the returned dict."""
    offsets: dict[pd.Timestamp, tuple[int, int]] = {}
    for i in range(len(dates) - 1):
        gap = (pd.Timestamp(dates[i + 1]) - pd.Timestamp(dates[i])).days
        if min_gap_days <= gap <= max_gap_days:
            offsets[pd.Timestamp(dates[i])] = (0, gap)
            if i > 0:
                offsets[pd.Timestamp(dates[i - 1])] = (1, gap)
    return offsets


def _compute_trade(day: pd.DataFrame, by_date: dict, entry_date: pd.Timestamp,
                    spot: float, cp: str, moneyness_pct: float,
                    holiday_offsets: dict | None = None) -> dict | None:
    expiries_today = sorted(e for e in day[ds.C_EXPIRY].unique()
                             if pd.Timestamp(e) >= entry_date)
    if len(expiries_today) < 2:
        return None
    front_e, next_e = pd.Timestamp(expiries_today[0]), pd.Timestamp(expiries_today[1])

    if moneyness_pct == 0.0:
        strike = qs._atm_strike(day, front_e, spot)
        if strike is None:
            return None
    else:
        target = spot * (1 - moneyness_pct) if cp == "C" else spot * (1 + moneyness_pct)
        row = _nearest_strike_row(day, front_e, cp, target)
        if row is None:
            return None
        strike = float(row[ds.C_STRIKE])

    near_row = qs._leg_row(day, front_e, cp, strike)
    next_row = qs._leg_row(day, next_e, cp, strike)
    if near_row is None or next_row is None:
        return None
    entry_cost = float(next_row[ds.C_CLOSE]) - float(near_row[ds.C_CLOSE])

    exit_day = by_date.get(front_e)
    if exit_day is None:
        return None
    exit_spot = float(exit_day[ds.C_UND_CLOSE].iloc[0])
    near_intrinsic = qs._intrinsic(cp, strike, exit_spot)
    next_exit = exit_day[exit_day[ds.C_CODE] == next_row[ds.C_CODE]]
    if next_exit.empty or float(next_exit[ds.C_CLOSE].iloc[0]) <= 0:
        return None
    next_exit_px = float(next_exit[ds.C_CLOSE].iloc[0])

    exit_value = next_exit_px - near_intrinsic
    pnl = exit_value - entry_cost
    days_before_holiday, holiday_gap_days = (holiday_offsets or {}).get(entry_date, (None, None))
    return {
        "entry_date": entry_date, "front_expiry": front_e, "next_expiry": next_e,
        "cp": cp, "moneyness_pct": moneyness_pct, "strike": strike,
        "entry_spot": spot, "exit_spot": exit_spot,
        "dte_at_entry": (front_e - entry_date).days,
        "day_of_month": entry_date.day, "bucket": _bucket(entry_date.day),
        "days_before_holiday": days_before_holiday, "holiday_gap_days": holiday_gap_days,
        "entry_cost": entry_cost, "pnl": pnl,
    }


def run(underlying: str, signal: str | None = None,
        threshold: float | None = None) -> pd.DataFrame:
    """signal=None: unrestricted (every trading date is a candidate entry) --
    the original day-of-month-only study. signal={'rv_pct','vrp'}: only dates
    where YESTERDAY's signal value (prior date in `dates`, not `sig.shift(1)`
    -- see module docstring) is <= threshold are candidates."""
    chain = _prepped_chain(underlying)
    dates = sorted(chain[ds.C_DATE].unique())
    by_date = dict(tuple(chain.groupby(ds.C_DATE)))
    sig = _signal_series(underlying, signal) if signal is not None else None
    holiday_offsets = _holiday_offsets(dates)  # cheap, always computed

    rows = []
    for i, dt in enumerate(dates):
        dt = pd.Timestamp(dt)
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        signal_val = np.nan
        if sig is not None:
            prev_dt = pd.Timestamp(dates[i - 1]) if i > 0 else None
            signal_val = sig.get(prev_dt, np.nan) if prev_dt is not None else np.nan
            if not (signal_val == signal_val) or signal_val > threshold:
                continue

        for cp in SIDES:
            for mp in MONEYNESS_PCTS:
                trade = _compute_trade(day, by_date, dt, spot, cp, mp, holiday_offsets)
                if trade is not None:
                    if sig is not None:
                        trade["signal_at_entry"] = float(signal_val)
                    rows.append(trade)
    return pd.DataFrame(rows)


def report(trades: pd.DataFrame, underlying: str, signal: str | None = None,
           threshold: float | None = None) -> None:
    name = UNDERLYINGS.get(underlying, underlying)
    if trades.empty:
        print(f"{underlying} ({name}): no trades.")
        return
    trades = trades.copy()
    # NOT pnl/entry_cost: entry_cost (near/next time-value gap) is frequently
    # a few fen, sometimes near-zero/negative -- dividing by it blows up into
    # meaningless four-digit "returns". Normalize by spot instead (stable,
    # comparable across dates/underlyings) -- pnl as a %% of underlying price.
    trades["r"] = trades["pnl"] / trades["entry_spot"]
    trades["period"] = np.where(trades["entry_date"] < SPLIT_DATE, "discovery", "holdout")

    sig_label = "unrestricted (no signal gate)" if signal is None else \
        f"signal={signal}<=yesterday's {threshold:+.2f}"
    print(f"\n{underlying} ({name}): calendar spread entry-timing study [{sig_label}] "
          f"({len(trades)} total entry-day observations)")
    for cp in SIDES:
        for mp in MONEYNESS_PCTS:
            sub = trades[(trades["cp"] == cp) & (trades["moneyness_pct"] == mp)]
            if sub.empty:
                continue
            label = "ATM" if mp == 0.0 else f"ITM{mp:.0%}"
            print(f"  {cp} {label}:")
            for period in ["discovery", "holdout"]:
                psub = sub[sub["period"] == period]
                if psub.empty:
                    continue
                print(f"    {period:>9}:", end=" ")
                for bucket in ["early(1-10)", "mid(11-20)", "late(21-31)"]:
                    b = psub[psub["bucket"] == bucket]
                    if b.empty:
                        print(f"{bucket}=n/a", end="  ")
                        continue
                    print(f"{bucket}: n={len(b):>3} mean_pnl_yuan={b['pnl'].mean()*ds.MULTIPLIER:>+7,.0f} "
                          f"mean_r(%spot)={b['r'].mean():+.2%} win%={(b['pnl']>0).mean():.0%}", end="  |  ")
                print()

    OUT_DIR.mkdir(exist_ok=True)
    suffix = "" if signal is None else f"_{signal}"
    out_path = OUT_DIR / f"calendar_spread_timing_{underlying}{suffix}.csv"
    trades.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Trades saved to {out_path}")


# ------------------------------------------------------------------ holiday
def _welch_tstat(a: np.ndarray, b: np.ndarray) -> float:
    """Same definition as macro_model's welch_tstat (ma_signal_put_backtest.py
    etc.): t-stat for mean(a) - mean(b), unequal variance."""
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return (a.mean() - b.mean()) / se if se > 0 else np.nan


def report_holiday_timing(trades: pd.DataFrame, underlying: str) -> None:
    """Pre-holiday (1-2 trading days before a detected long-holiday gap) vs
    everything else, per (cp, moneyness), full-sample + discovery/holdout,
    plus a major (>=HOLIDAY_MAJOR_GAP_DAYS calendar-day gap)/minor split
    within the pre-holiday group."""
    name = UNDERLYINGS.get(underlying, underlying)
    if trades.empty:
        print(f"{underlying} ({name}): no trades.")
        return
    trades = trades.copy()
    trades["r"] = trades["pnl"] / trades["entry_spot"]
    trades["period"] = np.where(trades["entry_date"] < SPLIT_DATE, "discovery", "holdout")
    trades["is_pre_holiday"] = trades["days_before_holiday"].isin([0, 1])

    n_holidays = trades.loc[trades["days_before_holiday"] == 0, "entry_date"].nunique()
    print(f"\n{underlying} ({name}): pre-holiday entry-timing study "
          f"({n_holidays} detected long-holiday gaps >= {HOLIDAY_MIN_GAP_DAYS}d, "
          f"{len(trades)} total entry-day observations)")

    for cp in SIDES:
        for mp in MONEYNESS_PCTS:
            sub = trades[(trades["cp"] == cp) & (trades["moneyness_pct"] == mp)]
            if sub.empty:
                continue
            label = "ATM" if mp == 0.0 else f"ITM{mp:.0%}"
            print(f"  {cp} {label}:")
            for period in ["full", "discovery", "holdout"]:
                psub = sub if period == "full" else sub[sub["period"] == period]
                pre = psub[psub["is_pre_holiday"]]
                rest = psub[~psub["is_pre_holiday"]]
                if pre.empty or rest.empty:
                    print(f"    {period:>9}: insufficient data (pre_n={len(pre)}, rest_n={len(rest)})")
                    continue
                t = _welch_tstat(pre["r"].to_numpy(), rest["r"].to_numpy())
                print(f"    {period:>9}: pre-holiday n={len(pre):>3} mean_r={pre['r'].mean():+.3%} "
                      f"win%={(pre['pnl']>0).mean():.0%}  |  rest n={len(rest):>4} "
                      f"mean_r={rest['r'].mean():+.3%} win%={(rest['pnl']>0).mean():.0%}  |  "
                      f"excess={pre['r'].mean()-rest['r'].mean():+.3%}  t={t:+.2f}")
            major = sub[sub["is_pre_holiday"] & (sub["holiday_gap_days"] >= HOLIDAY_MAJOR_GAP_DAYS)]
            minor = sub[sub["is_pre_holiday"] & (sub["holiday_gap_days"] < HOLIDAY_MAJOR_GAP_DAYS)]
            maj_r = f"{major['r'].mean():+.3%}" if not major.empty else "n/a"
            min_r = f"{minor['r'].mean():+.3%}" if not minor.empty else "n/a"
            print(f"    (pre-holiday split: major/>={HOLIDAY_MAJOR_GAP_DAYS}d n={len(major)} mean_r={maj_r}  "
                  f"vs minor n={len(minor)} mean_r={min_r})")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"calendar_spread_holiday_{underlying}.csv"
    trades.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Trades saved to {out_path}")


# ------------------------------------------------------------------ sweep
def _cell_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "mean_r": np.nan}
    return {"n": len(trades), "mean_r": (trades["pnl"] / trades["entry_spot"]).mean()}


def sweep(underlying: str, signal: str) -> None:
    """Signal doesn't depend on cp/moneyness_pct, so run() once per
    threshold (not once per grid cell), then slice by cp x moneyness_pct."""
    thresholds = ls.SIGNAL_MAXES[signal]
    name = UNDERLYINGS.get(underlying, underlying)
    grid_rows = []
    for threshold in thresholds:
        trades = run(underlying, signal=signal, threshold=threshold)
        if trades.empty:
            for cp, mp in product(SIDES, MONEYNESS_PCTS):
                grid_rows.append({"threshold": threshold, "cp": cp, "moneyness_pct": mp,
                                   "disc_n": 0, "disc_mean_r": np.nan,
                                   "hold_n": 0, "hold_mean_r": np.nan})
            continue
        trades = trades.copy()
        trades["period"] = np.where(trades["entry_date"] < SPLIT_DATE, "discovery", "holdout")
        for cp, mp in product(SIDES, MONEYNESS_PCTS):
            sub = trades[(trades["cp"] == cp) & (trades["moneyness_pct"] == mp)]
            d = _cell_stats(sub[sub["period"] == "discovery"])
            h = _cell_stats(sub[sub["period"] == "holdout"])
            grid_rows.append({"threshold": threshold, "cp": cp, "moneyness_pct": mp,
                               "disc_n": d["n"], "disc_mean_r": d["mean_r"],
                               "hold_n": h["n"], "hold_mean_r": h["mean_r"]})

    grid = pd.DataFrame(grid_rows)
    print(f"{underlying} ({name}) [signal={signal}]: sweep grid ({len(grid)} cells), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={ls.MIN_CELL_N}. "
          f"n = entry-day observations gated on YESTERDAY's signal, not independent cycles.")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"calendar_spread_sweep_{underlying}_{signal}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"Full grid saved to {grid_out}")

    eligible = grid[grid["disc_n"] >= ls.MIN_CELL_N]
    if eligible.empty:
        print("No cell has >= MIN_CELL_N discovery trades -- cannot freeze a pick "
              f"(max disc_n in grid = {grid['disc_n'].max():.0f}).\n")
        return
    both_pos = eligible[(eligible["disc_mean_r"] > 0) & (eligible["hold_mean_r"] > 0)]
    both_neg = eligible[(eligible["disc_mean_r"] < 0) & (eligible["hold_mean_r"] < 0)]
    print(f"Of {len(eligible)} eligible cells: {len(both_pos)} agree-positive, "
          f"{len(both_neg)} agree-negative, {len(eligible) - len(both_pos) - len(both_neg)} mixed sign")

    best = eligible.loc[eligible["disc_mean_r"].idxmax()]
    print(f"Frozen pick (best discovery mean_r, n>={ls.MIN_CELL_N}): "
          f"threshold={best['threshold']:+.2f}  cp={best['cp']}  moneyness_pct={best['moneyness_pct']:.2f}")
    print(f"  discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.3%}")
    print(f"  holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.3%}\n")


# ------------------------------------------------------------------ main
def main(underlying: str, signal: str | None = None, threshold: float | None = None,
          do_sweep: bool = False, holiday_report: bool = False) -> None:
    targets = list(UNDERLYINGS) if underlying == "all" else [underlying]
    for u in targets:
        if holiday_report:
            report_holiday_timing(run(u), u)  # always unrestricted -- holiday timing is the only gate
            continue
        if do_sweep:
            sweep(u, signal)
            continue
        trades = run(u, signal=signal, threshold=threshold)
        report(trades, u, signal, threshold)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="all", choices=list(UNDERLYINGS) + ["all"])
    ap.add_argument("--signal", choices=["rv_pct", "vrp"], default=None,
                    help="gate entries on the PRIOR trading day's signal <= --threshold. "
                         "Default: no gating (original unrestricted day-of-month study).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="entry when yesterday's signal <= this "
                         "(default: ls.DEFAULT_THRESHOLD[signal] when --signal given)")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search threshold x side x moneyness on discovery, "
                         "score frozen pick on holdout (requires --signal)")
    ap.add_argument("--holiday-report", action="store_true", dest="holiday_report",
                    help="pre-holiday (1-2 trading days before a detected long-holiday "
                         "gap) vs rest comparison, Welch t-test, discovery/holdout split")
    args = ap.parse_args()
    if args.sweep and args.signal is None:
        ap.error("--sweep requires --signal")
    threshold = args.threshold
    if args.signal is not None and threshold is None:
        threshold = ls.DEFAULT_THRESHOLD[args.signal]
    main(args.underlying, args.signal, threshold, args.sweep, args.holiday_report)
