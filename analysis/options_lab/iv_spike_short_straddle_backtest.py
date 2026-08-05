"""IV-spike-triggered short straddle: sell a near-month ATM straddle right
after ATM IV jumps abnormally over a short trailing window, betting on
mean-reversion (vol crush) rather than harvesting VRP on a fixed calendar
schedule like quarterly_atm_short_straddle.py. Naked (uncapped risk, sized
via margin.py's conservative exchange-minimum formula) -- exits after a
fixed short holding period (trading days), not held to expiry, with a
stop-loss safety net.

Entry trigger: near-month ATM call IV (same series backtest.py's entry gate
and iv_percentile_filter.py use) rises by >= --spike-pct (relative) over the
trailing --spike-window trading days. Only evaluated when no position is
open (one position at a time, same convention as quarterly_atm_short_straddle.py).

Follows this repo's established discovery/holdout discipline
(condor_delta_investigation_summary.md): discovery 2015-02-09 -> 2023-12-31,
holdout 2024-01-01 -> 2026-07-08, touched once. Use --sweep to grid-search
(spike_window, spike_pct, max_hold_days) on discovery and score the frozen
pick on holdout; a plain --underlying run uses one fixed config for
inspection/plotting.

Usage:
    python iv_spike_short_straddle_backtest.py --underlying 510050
    python iv_spike_short_straddle_backtest.py --underlying 510050 --sweep
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

import backtest as bt
import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qs

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = pd.Timestamp("2024-01-01")  # same split as condor_delta_investigation_summary.md

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

# sweep grid (discovery-only search, frozen pick scored once on holdout)
SPIKE_WINDOWS = [3, 5, 10]
SPIKE_PCTS = [0.15, 0.25, 0.40]
MAX_HOLDS = [3, 5, 10, 20]
STOP_LOSS_MULT = 2.0  # fixed by convention (condor investigation's typical range), not swept
MIN_CELL_N = 15


# ------------------------------------------------------------------ signal
def iv_atm_series(underlying: str) -> pd.Series:
    """Same near-month ATM call IV definition as backtest.py's entry gate."""
    chain = ds.load_chain(underlying)
    d = bt._prep(chain, bt.Rules())
    return bt._atm_iv_series(d)


def spike_flags(iv_atm: pd.Series, window: int, pct_threshold: float) -> pd.Series:
    """True on days where ATM IV rose >= pct_threshold (relative) over the
    trailing `window` trading days."""
    jump = iv_atm / iv_atm.shift(window) - 1.0
    return jump >= pct_threshold


# ------------------------------------------------------------------ engine
def run(underlying: str, spike_window: int, spike_pct: float, max_hold_days: int,
       stop_loss_mult: float | None = STOP_LOSS_MULT, dte_min: int = 10,
       dte_max: int = 45) -> dict:
    chain = ds.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    chain = chain[chain[ds.C_CLOSE] > 0]

    iv_atm = bt._atm_iv_series(chain)
    trigger = spike_flags(iv_atm, spike_window, spike_pct)

    dates = sorted(chain[ds.C_DATE].unique())
    by_date = dict(tuple(chain.groupby(ds.C_DATE)))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_pos: dict | None = None
    realized = 0.0

    for i, dt in enumerate(dates):
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

            mark = -(call_px + put_px)
            pnl = mark - open_pos["cost"]
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
                    "exit_spot": spot, "iv_at_entry": open_pos["iv_at_entry"],
                    "iv_jump_at_entry": open_pos["iv_jump_at_entry"],
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": days_held, "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: only when a spike just fired and a near-month expiry is in range
        expiries = sorted(day[ds.C_EXPIRY].unique())
        target = [e for e in expiries if dte_min <= (e - pd.Timestamp(dt)).days <= dte_max]
        ok_spike = bool(trigger.get(dt, False))
        if target and ok_spike:
            expiry = pd.Timestamp(target[0])
            strike = qs._atm_strike(day, expiry, spot)
            call_row = qs._leg_row(day, expiry, "C", strike) if strike is not None else None
            put_row = qs._leg_row(day, expiry, "P", strike) if strike is not None else None
            if call_row is not None and put_row is not None:
                call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
                legs_df = pd.DataFrame([
                    {"expiry": expiry, "cp": "C", "strike": strike, "direction": -1,
                     "qty": 1, "entry_price": call_px},
                    {"expiry": expiry, "cp": "P", "strike": strike, "direction": -1,
                     "qty": 1, "entry_price": put_px},
                ])
                margin_yuan, _ = mg.compute_deposit(legs_df, spot)
                open_pos = {
                    "entry_date": pd.Timestamp(dt), "entry_idx": i, "entry_spot": spot,
                    "expiry": expiry, "strike": strike,
                    "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
                    "last_call_px": call_px, "last_put_px": put_px,
                    "cost": -(call_px + put_px), "margin": margin_yuan,
                    "iv_at_entry": float(iv_atm.get(dt, np.nan)),
                    "iv_jump_at_entry": float(iv_atm.get(dt, np.nan) / iv_atm.get(dates[max(i - spike_window, 0)], np.nan) - 1.0),
                }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, spike_window: int, spike_pct: float,
          max_hold_days: int) -> None:
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        print("No trades.")
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

    print(f"{underlying}: IV-spike short straddle "
          f"(window={spike_window}d, spike>={spike_pct:.0%}, max_hold={max_hold_days}d)")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  avg_days_held={trades['days_held'].mean():.1f}  "
          f"mean IV jump at entry={trades['iv_jump_at_entry'].mean():+.0%}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={(pnl_yuan/(trades['margin_yuan'])).mean():+.2%} (pnl/margin per trade)")
    print(f"  avg_win={pnl_yuan[pnl_yuan>0].mean():,.0f}  "
          f"avg_loss={pnl_yuan[pnl_yuan<=0].mean():,.0f} yuan")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin (naked)={avg_margin:,.0f} yuan  return_on_margin/yr~={cagr_on_margin:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    trades["year"] = trades["entry_date"].dt.year
    print("\n  By year:")
    for yr, g in trades.groupby("year"):
        print(f"    {yr}: n={len(g)}  pnl={g['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"iv_spike_short_straddle_trades_{underlying}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\n  Trades saved to {trades_out}")


def plot(res: dict, underlying: str, out_path: Path) -> None:
    equity, trades = res["equity"], res["trades"]
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
    if not trades.empty:
        for _, t in trades.iterrows():
            ax1.axvline(t["entry_date"], color=C_AXIS, linewidth=0.6, alpha=0.5)
    ax1.set_ylabel("cumulative P&L (yuan)", color=C_INK2, fontsize=10)
    ax1.set_title(f"{underlying}: IV-spike short straddle", color=C_INK,
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
    return {"n": len(trades), "mean_r": r.mean(), "win": (trades["pnl"] > 0).mean()}


def sweep(underlying: str) -> None:
    rows = []
    cache: dict[tuple, dict] = {}
    for window, pct, hold in product(SPIKE_WINDOWS, SPIKE_PCTS, MAX_HOLDS):
        res = run(underlying, spike_window=window, spike_pct=pct, max_hold_days=hold)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        rows.append({"window": window, "pct": pct, "hold": hold,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})
        cache[(window, pct, hold)] = res

    grid = pd.DataFrame(rows)
    print(f"{underlying}: sweep grid ({len(grid)} cells), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")
    print(grid.to_string(index=False, formatters={
        "disc_mean_r": lambda v: f"{v:+.2%}" if v == v else "n/a",
        "hold_mean_r": lambda v: f"{v:+.2%}" if v == v else "n/a"}))

    eligible = grid[grid["disc_n"] >= MIN_CELL_N]
    if eligible.empty:
        print("\nNo cell has >= MIN_CELL_N discovery trades -- cannot freeze a pick.")
        return
    best = eligible.loc[eligible["disc_mean_r"].idxmax()]
    print(f"\nFrozen pick (best discovery mean_r, n>={MIN_CELL_N}): "
          f"window={best['window']:.0f}d  spike>={best['pct']:.0%}  max_hold={best['hold']:.0f}d")
    print(f"  discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.2%}")
    print(f"  holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.2%}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"iv_spike_sweep_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"\nGrid saved to {grid_out}")


# ------------------------------------------------------------------ main
def main(underlying: str, spike_window: int, spike_pct: float, max_hold_days: int,
        do_sweep: bool) -> None:
    if do_sweep:
        sweep(underlying)
        return
    res = run(underlying, spike_window=spike_window, spike_pct=spike_pct,
             max_hold_days=max_hold_days)
    report(res, underlying, spike_window, spike_pct, max_hold_days)
    out_path = OUT_DIR / f"iv_spike_short_straddle_{underlying}.png"
    plot(res, underlying, out_path)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--spike-window", type=int, default=5,
                    help="trailing trading-day window for the IV jump")
    ap.add_argument("--spike-pct", type=float, default=0.25,
                    help="minimum relative ATM IV increase over the window to trigger entry")
    ap.add_argument("--max-hold", type=int, default=10, dest="max_hold_days",
                    help="max trading days to hold before closing regardless of expiry")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search (window, pct, max_hold) on discovery, "
                         "score the frozen pick on holdout")
    args = ap.parse_args()
    main(args.underlying, args.spike_window, args.spike_pct, args.max_hold_days, args.sweep)
