"""Naked short straddle (sell 1x near-month ATM call + 1x ATM put, same
expiry, no calendar leg) triggered by rv_pct/vrp being HIGH -- the mirror
image of long_straddle_rv_contraction_backtest.py's "cheap/quiet -> bet on
expansion" long-straddle trigger, flipped to "rich/hot -> sell premium,
bet on mean-reversion." Sequential single-position tracking (one straddle
open at a time, exit before a new one can enter), same convention as
quarterly_atm_short_straddle.py / iv_spike_short_straddle_backtest.py -- NOT
the calendar spread script's parallel entry-day sampling. n here means
genuinely independent episodes, not overlapping observations.

Two signals, reused (not reimplemented) from sibling modules:
  rv_pct: rolling 252d percentile of the underlying's own trailing 20d
      realized vol (long_straddle_rv_contraction_backtest.rv_pct_series).
      CAUTION on mechanism fit: this is backward-looking realized vol, not
      implied vol -- "realized vol has been high lately" doesn't necessarily
      mean current IMPLIED vol is rich enough to be worth selling (could be
      chasing a regime shift already underway rather than harvesting a
      genuine premium). Included for symmetry with the long-side signal set
      and because it's cheap to test, not because the mechanism is obviously
      sound -- treat any "edge" here with extra skepticism versus vrp below.
  vrp: near-month ATM IV minus trailing realized vol -- IS the classical
      premium-selling rationale (IV rich relative to what's actually
      happened). Real exchange IV for 510050/510300
      (iv_low_long_straddle_backtest.vrp_trailing_series), BS-inverted via
      consolidated_store for the other five
      (long_straddle_rv_contraction_backtest.vrp_trailing_series).

Same deliberate rigor choice as the calendar spread script: gates on
YESTERDAY's signal value (prior trading date in the chain's own date list),
not the entry date's own same-day value.

Exit: stop-loss (loss >= stop_loss_mult x credit received, uncapped naked
risk needs this), max holding period, or expiry (settled at intrinsic) --
same three-way exit as iv_spike_short_straddle_backtest.py. Margin-sized via
margin.py's compute_deposit; returns normalized on margin (pnl/margin), not
notional or spot, since capital-at-risk for a naked short is the margin.

Follows this repo's discovery/holdout discipline: discovery < 2024-01-01,
holdout >= 2024-01-01, split fixed once.

Usage:
    python vol_rich_short_straddle_backtest.py --underlying 510500 --signal vrp
    python vol_rich_short_straddle_backtest.py --underlying all --signal rv_pct --sweep
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qs
import long_straddle_rv_contraction_backtest as ls
import iv_low_long_straddle_backtest as il

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = ls.SPLIT_DATE
LEGACY_UNDERLYINGS = {"510050": "50ETF", "510300": "300ETF"}
UNDERLYINGS = {**LEGACY_UNDERLYINGS, **ls.UNDERLYINGS}

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"

# sweep grids -- MINIMUMS (enter when signal >= threshold), mirror of
# ls.SIGNAL_MAXES's cheap-side grid but re-tuned for the rich side, not a
# blind sign-flip of the same numbers
SIGNAL_MINS = {
    "rv_pct": [0.70, 0.80, 0.90, 0.95],
    "vrp": [0.0, 0.03, 0.06, 0.10],  # ATM IV minus trailing RV, vol points
    "iv_pct": [0.70, 0.80, 0.90, 0.95],  # rolling percentile of raw ATM IV, mirror of rv_pct's grid
}
MAX_HOLDS = [3, 5, 10, 20]
STOP_LOSS_MULT = 2.0  # fixed by convention (matches iv_spike_short_straddle_backtest.py), not swept
MIN_CELL_N = 15

# DTE bucket grid for the theta/vega-motivated "does shorter DTE (more decay
# per unit vega, more gamma/pin risk) outperform" question -- see
# _theta_vega_ratio's docstring for why a per-day dynamic selector can't test
# this (theta/vega for an ATM straddle is ~1/DTE, a Black-Scholes identity,
# so "max theta/vega" always just means "shortest available expiry"; the only
# real lever is the window itself). (dte_min, dte_max) pairs; [10,45] is the
# script's original default, kept in the grid as the reference point.
DTE_BUCKETS = [(3, 10), (10, 20), (10, 45), (20, 45), (45, 90), (45, 120)]
DEFAULT_THRESHOLD = {"rv_pct": 0.80, "vrp": 0.03, "iv_pct": 0.80}


# ------------------------------------------------------------------ signal
def _prepped_chain(underlying: str) -> pd.DataFrame:
    if underlying in LEGACY_UNDERLYINGS:
        chain = ds.load_chain(underlying)
        chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
        return chain[chain[ds.C_CLOSE] > 0]
    return ls._prepped_chain(underlying)


def _signal_series(underlying: str, signal: str) -> pd.Series:
    if signal == "rv_pct":
        return ls.rv_pct_series(underlying)  # works for all 7
    if signal == "vrp":
        if underlying in LEGACY_UNDERLYINGS:
            return il.vrp_trailing_series(underlying)  # real exchange IV
        return ls.vrp_trailing_series(underlying)  # BS-inverted via consolidated_store
    if signal == "iv_pct":
        # rolling percentile of raw ATM IV -- distinct from vrp (IV minus
        # trailing RV): this only asks "is IV rich vs its OWN history",
        # never compares to realized vol at all. Never tested as a SHORT-side
        # (high-percentile) trigger before -- the long-straddle project only
        # ever used it as a LOW-percentile ("cheap") entry.
        if underlying in LEGACY_UNDERLYINGS:
            return il.iv_pct_series(underlying)  # real exchange IV
        return ls.iv_pct_series(underlying)  # BS-inverted via consolidated_store
    raise ValueError(f"unknown signal {signal!r}")


DTE_SELECT_MODES = ("nearest", "max_theta_vega")


def _theta_vega_ratio(call_row: pd.Series, put_row: pd.Series) -> float:
    """abs(combined theta) / combined vega for an ATM straddle -- theta
    harvested per unit of vega (vol) risk carried. Higher = more decay
    collected for the same vol exposure, at the cost of the gamma/pin risk
    that comes with whatever DTE produces that ratio (short-dated expiries
    mechanically score high here; the backtest, not this formula, is what
    tells us whether the extra theta actually compensates for that risk)."""
    theta = float(call_row[ds.C_THETA]) + float(put_row[ds.C_THETA])
    vega = float(call_row[ds.C_VEGA]) + float(put_row[ds.C_VEGA])
    if vega <= 0:
        return -np.inf
    return abs(theta) / vega


def _select_expiry(day: pd.DataFrame, candidate_expiries: list, spot: float,
                    dte_select: str) -> tuple | None:
    """Pick one (expiry, strike, call_row, put_row) among candidate_expiries
    (already DTE-window-filtered, ordered nearest-first). 'nearest' = old
    behaviour (first candidate with valid legs). 'max_theta_vega' = evaluate
    every candidate's ATM straddle and keep the highest theta/vega ratio."""
    valid = []
    for e in candidate_expiries:
        expiry = pd.Timestamp(e)
        strike = qs._atm_strike(day, expiry, spot)
        if strike is None:
            continue
        call_row = qs._leg_row(day, expiry, "C", strike)
        put_row = qs._leg_row(day, expiry, "P", strike)
        if call_row is None or put_row is None:
            continue
        valid.append((expiry, strike, call_row, put_row))
        if dte_select == "nearest":
            return valid[0]
    if not valid:
        return None
    if dte_select == "max_theta_vega":
        return max(valid, key=lambda v: _theta_vega_ratio(v[2], v[3]))
    raise ValueError(f"unknown dte_select {dte_select!r}")


# ------------------------------------------------------------------ engine
def run(underlying: str, signal: str, threshold: float, max_hold_days: int,
        stop_loss_mult: float | None = STOP_LOSS_MULT, dte_min: int = 10,
        dte_max: int = 45, dte_select: str = "nearest") -> dict:
    chain = _prepped_chain(underlying)
    dates = sorted(chain[ds.C_DATE].unique())
    by_date = dict(tuple(chain.groupby(ds.C_DATE)))
    sig = _signal_series(underlying, signal)

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_pos: dict | None = None
    realized = 0.0

    for i, dt in enumerate(dates):
        dt = pd.Timestamp(dt)
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            call = day[day[ds.C_CODE] == open_pos["call_code"]]
            put = day[day[ds.C_CODE] == open_pos["put_code"]]
            call_px = float(call[ds.C_CLOSE].iloc[0]) if not call.empty else open_pos["last_call_px"]
            put_px = float(put[ds.C_CLOSE].iloc[0]) if not put.empty else open_pos["last_put_px"]
            open_pos["last_call_px"], open_pos["last_put_px"] = call_px, put_px

            expired = pd.Timestamp(dt) >= open_pos["expiry"]
            days_held = i - open_pos["entry_idx"]
            reason = None
            if expired:
                call_px = qs._intrinsic("C", open_pos["strike"], spot)
                put_px = qs._intrinsic("P", open_pos["strike"], spot)
                reason = "expiry"

            mark = -(call_px + put_px)  # cost to close a short straddle
            pnl = mark - open_pos["cost"]  # cost = -(credit received), negative
            basis = abs(open_pos["cost"])

            if reason is None:
                if stop_loss_mult is not None and pnl <= -stop_loss_mult * basis:
                    reason = "stop_loss"
                elif days_held >= max_hold_days:
                    reason = "max_hold"

            if reason:
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "exit_date": dt,
                    "expiry": open_pos["expiry"], "exit_reason": reason,
                    "strike": open_pos["strike"], "entry_spot": open_pos["entry_spot"],
                    "exit_spot": spot, "signal_at_entry": open_pos["signal_at_entry"],
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": days_held, "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: only when YESTERDAY's signal >= threshold ("rich/hot")
        # and a near-month expiry is in range, and no position is open
        prev_dt = pd.Timestamp(dates[i - 1]) if i > 0 else None
        p = sig.get(prev_dt, np.nan) if prev_dt is not None else np.nan
        ok = (p == p) and p >= threshold
        if ok:
            expiries = sorted(day[ds.C_EXPIRY].unique())
            target = [e for e in expiries if dte_min <= (e - dt).days <= dte_max]
            if target:
                picked = _select_expiry(day, target, spot, dte_select)
                if picked is not None:
                    expiry, strike, call_row, put_row = picked
                    call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
                    legs_df = pd.DataFrame([
                        {"expiry": expiry, "cp": "C", "strike": strike, "direction": -1,
                         "qty": 1, "entry_price": call_px},
                        {"expiry": expiry, "cp": "P", "strike": strike, "direction": -1,
                         "qty": 1, "entry_price": put_px},
                    ])
                    margin_yuan, _ = mg.compute_deposit(legs_df, spot)
                    open_pos = {
                        "entry_date": dt, "entry_idx": i, "entry_spot": spot,
                        "expiry": expiry, "strike": strike,
                        "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
                        "last_call_px": call_px, "last_put_px": put_px,
                        "cost": -(call_px + put_px), "margin": margin_yuan,
                        "signal_at_entry": float(p),
                    }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, signal: str, threshold: float,
           max_hold_days: int, dte_select: str = "nearest") -> None:
    trades, equity = res["trades"], res["equity"]
    name = UNDERLYINGS.get(underlying, underlying)
    suffix = "" if dte_select == "nearest" else f"_{dte_select}"
    if trades.empty:
        print(f"{underlying} ({name}): no trades.")
        return
    pnl_yuan = trades["pnl"] * ds.MULTIPLIER
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan
    avg_margin = trades["margin_yuan"].mean()
    total_pnl = pnl_yuan.sum()
    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    cagr_on_margin = (total_pnl / avg_margin) / span_years if span_years > 0 else np.nan

    print(f"{underlying} ({name}) [{signal}]: vol-rich short straddle "
          f"(yesterday's {signal}>={threshold:.2f}, max_hold={max_hold_days}d, "
          f"stop_loss={STOP_LOSS_MULT:.1f}x credit, dte_select={dte_select})")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  avg_days_held={trades['days_held'].mean():.1f}  "
          f"mean signal at entry={trades['signal_at_entry'].mean():+.2f}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={(pnl_yuan/trades['margin_yuan']).mean():+.2%} (pnl/margin per trade)")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin (naked)={avg_margin:,.0f} yuan  return_on_margin/yr~={cagr_on_margin:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    trades["year"] = trades["entry_date"].dt.year
    print("  by year:")
    for yr, g in trades.groupby("year"):
        print(f"    {yr}: n={len(g)}  pnl={g['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"vol_rich_short_straddle_trades_{underlying}_{signal}{suffix}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"  Trades saved to {trades_out}\n")


def plot(res: dict, underlying: str, signal: str, out_path: Path) -> None:
    equity = res["equity"]
    name = UNDERLYINGS.get(underlying, underlying)
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]}, facecolor=C_SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C_AXIS)
        ax.spines["bottom"].set_color(C_AXIS)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    ax1.plot(eq.index, eq.values, color=C_BLUE, linewidth=1.6)
    ax1.set_ylabel("cumulative P&L (yuan)", color=C_INK2, fontsize=10)
    ax1.set_title(f"{underlying} ({name}): vol-rich short straddle [{signal}]", color=C_INK,
                  fontsize=12, loc="left", pad=10)

    ax2.fill_between(dd.index, 0, dd.values, color=C_RED, alpha=0.55, linewidth=0)
    ax2.set_ylabel("drawdown (yuan)", color=C_INK2, fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ sweep
def _cell_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "mean_r": np.nan}
    r = trades["pnl"] * ds.MULTIPLIER / trades["margin_yuan"]
    return {"n": len(trades), "mean_r": r.mean()}


def sweep(underlying: str, signal: str, dte_select: str = "nearest") -> None:
    thresholds = SIGNAL_MINS[signal]
    name = UNDERLYINGS.get(underlying, underlying)
    suffix = "" if dte_select == "nearest" else f"_{dte_select}"
    rows = []
    for threshold, hold in product(thresholds, MAX_HOLDS):
        res = run(underlying, signal=signal, threshold=threshold, max_hold_days=hold,
                   dte_select=dte_select)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        rows.append({"threshold": threshold, "hold": hold,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    print(f"{underlying} ({name}) [{signal}]: sweep grid ({len(grid)} cells), dte_select={dte_select}, "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"vol_rich_short_straddle_sweep_{underlying}_{signal}{suffix}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"Full grid saved to {grid_out}")

    eligible = grid[grid["disc_n"] >= MIN_CELL_N]
    if eligible.empty:
        print("No cell has >= MIN_CELL_N discovery trades -- cannot freeze a pick "
              f"(max disc_n in grid = {grid['disc_n'].max():.0f}).\n")
        return
    both_pos = eligible[(eligible["disc_mean_r"] > 0) & (eligible["hold_mean_r"] > 0)]
    both_neg = eligible[(eligible["disc_mean_r"] < 0) & (eligible["hold_mean_r"] < 0)]
    print(f"Of {len(eligible)} eligible cells: {len(both_pos)} agree-positive, "
          f"{len(both_neg)} agree-negative, {len(eligible) - len(both_pos) - len(both_neg)} mixed sign")

    best = eligible.loc[eligible["disc_mean_r"].idxmax()]
    print(f"Frozen pick (best discovery mean_r, n>={MIN_CELL_N}): "
          f"threshold={best['threshold']:.2f}  max_hold={best['hold']:.0f}d")
    print(f"  discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.2%}")
    print(f"  holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.2%}\n")


def compare_dte_select(underlying: str, signal: str, threshold: float, max_hold_days: int,
                        dte_min: int = 10, dte_max: int = 45) -> None:
    """Matched-sample comparison at a fixed (threshold, max_hold) cell: same
    entry gate, only the expiry-selection rule differs. Reports whether entry
    dates line up (composition check per feedback-backtest-rigor) before
    trusting any mean_r difference between 'nearest' and 'max_theta_vega'.
    Note the default 10-45 dte window rarely offers >1 candidate expiry (mean
    1.12, only 12% of days with >=2 on 510050) -- 'max_theta_vega' degenerates
    to 'nearest' there. Widen dte_max (e.g. 120) for a non-degenerate test."""
    name = UNDERLYINGS.get(underlying, underlying)
    res_a = run(underlying, signal=signal, threshold=threshold, max_hold_days=max_hold_days,
                dte_select="nearest", dte_min=dte_min, dte_max=dte_max)
    res_b = run(underlying, signal=signal, threshold=threshold, max_hold_days=max_hold_days,
                dte_select="max_theta_vega", dte_min=dte_min, dte_max=dte_max)
    ta, tb = res_a["trades"], res_b["trades"]
    print(f"{underlying} ({name}) [{signal}] threshold={threshold:.2f} max_hold={max_hold_days}d "
          f"dte_window=[{dte_min},{dte_max}]: nearest vs max_theta_vega")
    if ta.empty or tb.empty:
        print(f"  nearest n={len(ta)}  max_theta_vega n={len(tb)} (at least one empty, nothing to compare)\n")
        return
    dates_a, dates_b = set(ta["entry_date"]), set(tb["entry_date"])
    common = dates_a & dates_b
    print(f"  entry dates: nearest n={len(ta)}  max_theta_vega n={len(tb)}  "
          f"common={len(common)} ({len(common)/max(len(dates_a), len(dates_b)):.0%} overlap)")
    if dates_a != dates_b:
        print("  NOTE: entry-date sets differ -- some candidate expiries must have had "
              "missing legs on one side; treat mean_r comparison as approximate, not "
              "strictly matched-sample.")
    for label, t in (("nearest", ta), ("max_theta_vega", tb)):
        r = t["pnl"] * ds.MULTIPLIER / t["margin_yuan"]
        avg_dte = (t["expiry"] - t["entry_date"]).dt.days.mean()
        print(f"  {label:>15}: n={len(t)}  mean_r={r.mean():+.2%}  win_rate={(t['pnl']>0).mean():.0%}  "
              f"avg_dte_at_entry={avg_dte:.1f}d")
    if common:
        # same entry date on both sides -> isolate the pure effect of expiry choice
        ma = ta.set_index("entry_date").loc[sorted(common)]
        mb = tb.set_index("entry_date").loc[sorted(common)]
        ra = ma["pnl"] * ds.MULTIPLIER / ma["margin_yuan"]
        rb = mb["pnl"] * ds.MULTIPLIER / mb["margin_yuan"]
        print(f"  matched-only (n={len(common)}): nearest mean_r={ra.mean():+.2%}  "
              f"max_theta_vega mean_r={rb.mean():+.2%}  diff={rb.mean()-ra.mean():+.2%}\n")
    else:
        print()


def dte_bucket_sweep(underlying: str, signal: str, threshold: float, max_hold_days: int,
                      buckets: list | None = None) -> pd.DataFrame:
    """Holds the entry signal/threshold/max_hold fixed (the already-validated
    VRP pick) and varies only the DTE window -- the real empirical question
    behind 'theta/vega DTE selection' once the per-day dynamic version was
    shown degenerate (see _theta_vega_ratio docstring): does selling a
    shorter-DTE straddle (more theta harvested per unit vega, but more
    gamma/pin risk near expiry) systematically do better or worse than the
    current 10-45d default? discovery/holdout split per feedback-backtest-rigor."""
    buckets = buckets or DTE_BUCKETS
    name = UNDERLYINGS.get(underlying, underlying)
    rows = []
    for dte_min, dte_max in buckets:
        res = run(underlying, signal=signal, threshold=threshold, max_hold_days=max_hold_days,
                   dte_select="nearest", dte_min=dte_min, dte_max=dte_max)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        avg_dte = (trades["expiry"] - trades["entry_date"]).dt.days.mean() if not trades.empty else np.nan
        rows.append({"dte_min": dte_min, "dte_max": dte_max, "avg_dte_at_entry": avg_dte,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    print(f"{underlying} ({name}) [{signal}] threshold={threshold:.2f} max_hold={max_hold_days}d: "
          f"DTE-bucket sweep ({len(grid)} buckets), discovery < {SPLIT_DATE.date()} <= holdout, "
          f"MIN_CELL_N={MIN_CELL_N}")
    for _, row in grid.iterrows():
        flag = "" if row["disc_n"] >= MIN_CELL_N else "  (< MIN_CELL_N, low confidence)"
        print(f"  [{row['dte_min']:>3.0f},{row['dte_max']:>3.0f}]  avg_dte={row['avg_dte_at_entry']:>5.1f}d  "
              f"disc: n={row['disc_n']:>3.0f} mean_r={row['disc_mean_r']:+7.2%}   "
              f"hold: n={row['hold_n']:>3.0f} mean_r={row['hold_mean_r']:+7.2%}{flag}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"vol_rich_short_straddle_dte_bucket_sweep_{underlying}_{signal}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"  Full grid saved to {grid_out}\n")
    return grid


# ------------------------------------------------------------------ main
def main(underlying: str, signal: str, threshold: float, max_hold_days: int,
         do_sweep: bool, dte_select: str = "nearest", do_compare: bool = False,
         dte_min: int = 10, dte_max: int = 45, do_bucket_sweep: bool = False) -> None:
    targets = list(UNDERLYINGS) if underlying == "all" else [underlying]
    for u in targets:
        if do_bucket_sweep:
            dte_bucket_sweep(u, signal, threshold, max_hold_days)
            continue
        if do_compare:
            compare_dte_select(u, signal, threshold, max_hold_days, dte_min, dte_max)
            continue
        if do_sweep:
            sweep(u, signal, dte_select)
            continue
        res = run(u, signal=signal, threshold=threshold, max_hold_days=max_hold_days,
                   dte_select=dte_select, dte_min=dte_min, dte_max=dte_max)
        report(res, u, signal, threshold, max_hold_days, dte_select)
        suffix = "" if dte_select == "nearest" else f"_{dte_select}"
        out_path = OUT_DIR / f"vol_rich_short_straddle_{u}_{signal}{suffix}.png"
        plot(res, u, signal, out_path)
        print(f"  Chart saved to {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="all", choices=list(UNDERLYINGS) + ["all"])
    ap.add_argument("--signal", choices=["rv_pct", "vrp", "iv_pct"], default="vrp",
                    help="rv_pct: past realized vol high vs its own history (weaker "
                         "mechanism fit -- see module docstring). "
                         "vrp: ATM IV rich relative to trailing realized vol (the "
                         "classical premium-selling rationale). "
                         "iv_pct: raw ATM IV high vs its OWN rolling history, no RV "
                         "comparison at all -- untested on the short side until now.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="entry when yesterday's signal >= this "
                         "(default: 0.80 for rv_pct/iv_pct, 0.03 for vrp)")
    ap.add_argument("--max-hold", type=int, default=10, dest="max_hold_days",
                    help="max trading days to hold before closing regardless of expiry")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search (threshold, max_hold) on discovery, "
                         "score the frozen pick on holdout")
    ap.add_argument("--dte-select", choices=list(DTE_SELECT_MODES), default="nearest",
                    dest="dte_select",
                    help="'nearest' (default, original behaviour): first expiry in the "
                         "DTE window. 'max_theta_vega': pick whichever expiry in-window "
                         "has the highest theta/vega ATM straddle (most decay per unit "
                         "vega risk) instead.")
    ap.add_argument("--compare", action="store_true",
                    help="matched-sample comparison of nearest vs max_theta_vega at a "
                         "fixed (threshold, max_hold) cell, instead of running/sweeping "
                         "a single dte_select mode")
    ap.add_argument("--dte-min", type=int, default=10, dest="dte_min")
    ap.add_argument("--dte-max", type=int, default=45, dest="dte_max",
                    help="widen this (e.g. 120) so >1 candidate expiry is usually "
                         "available -- at the default 45 'max_theta_vega' degenerates "
                         "to 'nearest' almost always")
    ap.add_argument("--dte-bucket-sweep", action="store_true", dest="do_bucket_sweep",
                    help="fix signal/threshold/max_hold at the given values and sweep "
                         "DTE_BUCKETS instead -- the actual 'does shorter DTE do better' "
                         "question once the per-day theta/vega selector was shown degenerate")
    args = ap.parse_args()
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD[args.signal]
    main(args.underlying, args.signal, threshold, args.max_hold_days, args.sweep,
         args.dte_select, args.compare, args.dte_min, args.dte_max, args.do_bucket_sweep)
