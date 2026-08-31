"""Rolling short STRANGLE on the exchange's quarterly (季月) contract, legs
selected by target Delta (independently per side) instead of ATM moneyness.

Same always-on cadence as quarterly_atm_short_straddle.py: no entry signal or
threshold -- open a strangle on the nearest quarterly-cycle expiry (expiry
month in {3,6,9,12}), hold to expiry with no profit target / stop loss,
settle at intrinsic value, and immediately roll into the new nearest
quarterly contract the next trading day. Only the strike-selection mechanism
differs from that sibling script: the call leg is whichever listed strike
has Delta closest to +target_delta, the put leg closest to -target_delta,
picked independently per side (real skew can put them at different
moneyness distances from spot) -- so this is a strangle, not a same-strike
straddle. Reuses `delta_strangle_short_backtest._delta_leg` unmodified for
that selection, and `quarterly_atm_short_straddle._target_expiries`/
`_intrinsic` unmodified for expiry choice and settlement.

Motivation: delta_strangle_short_backtest.py already found that delta-
targeted OTM strikes trade some premium for materially less gamma exposure
than ATM -- but only under a VRP-gated, opportunistic entry (dozens of
trades/year at ~20-45 DTE). This script isolates the pure strike-selection
effect on the SAME always-on quarterly cadence as the ATM baseline, so the
two are directly comparable apples-to-apples: identical expiry selection,
hold-to-expiry/settlement/margin logic -- only the strike lever (ATM vs.
delta-target) differs. target_delta=0.5 is the closest analog to the ATM
sibling's straddle (not identical -- see that script's own docstring caveat
about the one-strike-rung difference between "nearest-to-spot" and
"nearest-to-0.5-delta").

Requires real per-leg Delta (only validated for 510050/510300 and the other
SSE names' officially-replaced Greeks; SZSE Greeks are BS-derived and
unvalidated -- see delta_strangle_short_backtest.py's docstring and
Data/买断日更/README.md).

Usage:
    python quarterly_delta_strangle_short_backtest.py --underlying 510050 --target-delta 0.30
    python quarterly_delta_strangle_short_backtest.py --underlying 510050 --sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qs
import delta_strangle_short_backtest as dss

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = dss.SPLIT_DATE
TARGET_DELTAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
# Quarterly cadence gives ~4 cycles/year vs. dozens/year for dss's daily-
# signal-gated strangle, so a discovery/holdout cell needs a lower bar.
MIN_CELL_N = 8

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"


# ------------------------------------------------------------------ engine
def run(underlying: str, target_delta: float = 0.30, roll_freq: str = "quarterly",
       start_date: str | None = None) -> dict:
    df = ds.load_chain(underlying)
    df = df[~df[ds.C_NAME].astype(str).str.endswith("A")]  # exclude adjusted
    df = df[df[ds.C_CLOSE] > 0]
    if start_date is not None:
        df = df[df[ds.C_DATE] >= pd.Timestamp(start_date)]
    dates = sorted(df[ds.C_DATE].unique())
    by_date = dict(tuple(df.groupby(ds.C_DATE)))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_pos: dict | None = None
    realized = 0.0

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            call = day[day[ds.C_CODE] == open_pos["call_code"]]
            put = day[day[ds.C_CODE] == open_pos["put_code"]]
            expired = pd.Timestamp(dt) >= open_pos["expiry"]
            if expired or call.empty or put.empty:
                call_px = qs._intrinsic("C", open_pos["call_strike"], spot)
                put_px = qs._intrinsic("P", open_pos["put_strike"], spot)
                mark = -(call_px + put_px)
                pnl = mark - open_pos["cost"]
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "expiry": open_pos["expiry"],
                    "exit_date": dt, "call_strike": open_pos["call_strike"],
                    "put_strike": open_pos["put_strike"],
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "call_entry_px": open_pos["call_px"], "put_entry_px": open_pos["put_px"],
                    "call_delta_at_entry": open_pos["call_delta_at_entry"],
                    "put_delta_at_entry": open_pos["put_delta_at_entry"],
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": (pd.Timestamp(dt) - open_pos["entry_date"]).days,
                    "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            call_px = float(call[ds.C_CLOSE].iloc[0])
            put_px = float(put[ds.C_CLOSE].iloc[0])
            mark = -(call_px + put_px)
            pnl = mark - open_pos["cost"]
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: roll into nearest quarterly contract, delta-target legs
        qexp = qs._target_expiries(day, roll_freq)
        if not qexp:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue
        expiry = pd.Timestamp(qexp[0])
        call_row = dss._delta_leg(day, expiry, "C", target_delta)
        put_row = dss._delta_leg(day, expiry, "P", target_delta)
        if call_row is None or put_row is None:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue

        call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
        call_strike, put_strike = float(call_row[ds.C_STRIKE]), float(put_row[ds.C_STRIKE])
        legs_df = pd.DataFrame([
            {"expiry": expiry, "cp": "C", "strike": call_strike, "direction": -1,
             "qty": 1, "entry_price": call_px},
            {"expiry": expiry, "cp": "P", "strike": put_strike, "direction": -1,
             "qty": 1, "entry_price": put_px},
        ])
        margin_yuan, _ = mg.compute_deposit(legs_df, spot)

        open_pos = {
            "entry_date": pd.Timestamp(dt), "entry_spot": spot, "expiry": expiry,
            "call_strike": call_strike, "put_strike": put_strike,
            "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
            "call_px": call_px, "put_px": put_px, "cost": -(call_px + put_px),
            "margin": margin_yuan,
            "call_delta_at_entry": float(call_row[ds.C_DELTA]),
            "put_delta_at_entry": float(put_row[ds.C_DELTA]),
        }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, target_delta: float,
          baseline: dict | None = None) -> None:
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        print(f"{underlying}: no trades.")
        return
    pnl_yuan = trades["pnl"] * ds.MULTIPLIER
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan

    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    avg_margin = trades["margin_yuan"].mean()
    total_pnl = pnl_yuan.sum()
    cagr_on_margin = (total_pnl / avg_margin) / span_years if span_years > 0 else np.nan

    print(f"{underlying}: quarterly short strangle (target_delta={target_delta:.2f}), "
          f"{equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades(quarters)={len(trades)}  avg_days_held={trades['days_held'].mean():.0f}  "
          f"avg call/put delta at entry={trades['call_delta_at_entry'].mean():+.2f}/"
          f"{trades['put_delta_at_entry'].mean():+.2f}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={(pnl_yuan/trades['margin_yuan']).mean():+.2%} (pnl/margin per trade)")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin (naked)={avg_margin:,.0f} yuan  return_on_margin/yr~={cagr_on_margin:.1%}")

    trades["year"] = trades["entry_date"].dt.year
    print("\nBy year:")
    for yr, g in trades.groupby("year"):
        print(f"  {yr}: n={len(g)}  pnl={g['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  "
              f"win%={(g['pnl']>0).mean():.0%}")

    if baseline is not None and not baseline["trades"].empty:
        bt = baseline["trades"]
        common_start = max(trades["entry_date"].min(), bt["entry_date"].min())
        b = bt[bt["entry_date"] >= common_start]
        t = trades[trades["entry_date"] >= common_start]
        print(f"\n  vs. static ATM baseline (quarterly_atm_short_straddle.py), "
              f"same window from {common_start.date()}:")
        print(f"    delta strangle:   n={len(t)}  total_pnl={t['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  "
              f"mean_r={(t['pnl']*ds.MULTIPLIER/t['margin_yuan']).mean():+.2%}")
        print(f"    ATM straddle:     n={len(b)}  total_pnl={b['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  "
              f"mean_r={(b['pnl']*ds.MULTIPLIER/b['margin_yuan']).mean():+.2%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"quarterly_delta_strangle_short_trades_{underlying}_{target_delta:.2f}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\n  Trades saved to {trades_out}")


def plot(res: dict, underlying: str, target_delta: float, out_path: Path) -> None:
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
    for _, t in trades.iterrows():
        ax1.axvline(t["entry_date"], color=C_AXIS, linewidth=0.6, alpha=0.5)
    ax1.set_ylabel("cumulative P&L (yuan)", color=C_INK2, fontsize=10)
    ax1.set_title(f"{underlying}: rolling quarterly short strangle (target_delta={target_delta:.2f})",
                  color=C_INK, fontsize=12, loc="left", pad=10)

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


def delta_sweep(underlying: str, target_deltas: list | None = None) -> pd.DataFrame:
    """Same always-on quarterly cadence throughout -- isolates the strike-
    selection (target_delta) effect the same way dss.delta_sweep isolates it
    for the signal-gated strangle."""
    target_deltas = target_deltas or TARGET_DELTAS
    rows = []
    for td in target_deltas:
        res = run(underlying, target_delta=td)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        avg_call_d = trades["call_delta_at_entry"].mean() if not trades.empty else np.nan
        avg_put_d = trades["put_delta_at_entry"].mean() if not trades.empty else np.nan
        rows.append({"target_delta": td, "avg_call_delta": avg_call_d, "avg_put_delta": avg_put_d,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    print(f"{underlying}: quarterly delta-strangle sweep ({len(grid)} target_deltas), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")
    for _, row in grid.iterrows():
        flag = "" if row["disc_n"] >= MIN_CELL_N else "  (< MIN_CELL_N, low confidence)"
        print(f"  target_delta={row['target_delta']:.2f}  actual C/P delta="
              f"{row['avg_call_delta']:+.2f}/{row['avg_put_delta']:+.2f}  "
              f"disc: n={row['disc_n']:>3.0f} mean_r={row['disc_mean_r']:+7.2%}   "
              f"hold: n={row['hold_n']:>3.0f} mean_r={row['hold_mean_r']:+7.2%}{flag}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"quarterly_delta_strangle_short_sweep_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"  Full grid saved to {grid_out}\n")
    return grid


# ------------------------------------------------------------------ main
def main(underlying: str, target_delta: float, do_sweep: bool) -> None:
    if do_sweep:
        delta_sweep(underlying)
        return
    res = run(underlying, target_delta=target_delta)
    baseline = qs.run(underlying)
    report(res, underlying, target_delta, baseline=baseline)
    out_path = OUT_DIR / f"quarterly_delta_strangle_short_{underlying}_{target_delta:.2f}.png"
    plot(res, underlying, target_delta, out_path)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--target-delta", type=float, default=0.30, dest="target_delta",
                    help="sell the strike closest to +/- this delta on each side")
    ap.add_argument("--sweep", action="store_true", dest="do_sweep",
                    help="fix cadence, sweep TARGET_DELTAS instead")
    args = ap.parse_args()
    main(args.underlying, args.target_delta, args.do_sweep)
