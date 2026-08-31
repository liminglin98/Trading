"""Rolling ATM/delta short straddle on the exchange's quarterly (季月)
contract, periodically RE-STRIKED within the same fixed expiry.

Expiry selection is identical to quarterly_atm_short_straddle.py: hold the
nearest listed quarterly-cycle contract (expiry month in {3,6,9,12}) from
entry to expiry, and never change which expiry is held mid-cycle. Unlike
that script, the strike is not frozen for the whole cycle -- on a
pre-committed calendar/counter cadence (never conditioned on realized P&L,
per the standing "no hindsight-gated overlay triggers" rule), close the
current legs and reopen new ones at the then-current ATM strike (or
delta-target strike, reusing delta_strangle_short_backtest.py's
_delta_leg) for the SAME expiry.

Open question this backtest exists to answer: re-striking later in the
cycle sells a straddle with much less extrinsic value (fewer days left to
decay) while ATM gamma is *rising* as DTE falls -- so each successive
re-strike plausibly has worse risk/reward than the last, and a late-cycle
re-strike (thin premium, high gamma) could be a net negative versus just
holding the original strike to expiry. --min-dte-for-restrike lets you cut
off re-striking before that high-gamma tail; --restrike-cadence "none"
reproduces the static baseline exactly (as a built-in sanity check against
quarterly_atm_short_straddle.py). The --sweep grid spans from "restrike
every month all the way through" down to "one early restrike only, then
hold" so this is tested as a spectrum, not a single cherry-picked config.

Same conventions as the sibling scripts: fills at daily close. --slippage
charges an adverse half-spread per leg on entry (sell at bid: raw - slip)
and on any non-settlement close (buy back at ask: raw + slip) -- including
BOTH sides of a restrike (2 closes + 2 opens = 4 leg-fills), following
expiry_week_atm_straddle_backtest.py's convention. No slippage at cash
settlement (expiry) -- there's no spread to cross. Margin sized via
margin.py.

Requires real per-leg Delta for --strike-mode delta (only validated for
510050/510300 and the other SSE names; SZSE Greeks are BS-derived and
unvalidated -- see delta_strangle_short_backtest.py's docstring and
Data/买断日更/README.md). Delta mode picks call/put strikes independently
so it generally produces a strangle (unequal strikes), not a straddle.

Usage:
    python quarterly_restrike_short_straddle.py --underlying 510050
    python quarterly_restrike_short_straddle.py --underlying 510050 \\
        --restrike-cadence trading_days --restrike-days 21 --min-dte-for-restrike 30
    python quarterly_restrike_short_straddle.py --underlying 510050 --sweep
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
# Far fewer independent trades exist here than in dss's daily-signal-gated
# strangle (a ~90-day quarterly cadence gives ~4/year vs. dozens/year), so a
# discovery/holdout cell needs a lower bar to be usable at all.
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

# (cadence, restrike_n_days, min_dte_for_restrike, label) -- spans the full
# spectrum from "restrike monthly all the way to expiry" down to "one early
# restrike only, then hold" and includes "none" as a built-in sanity check
# that should reproduce quarterly_atm_short_straddle.py's ATM result exactly.
RESTRIKE_SPECS = [
    ("none", None, 0, "never restrike (static baseline)"),
    ("calendar_monthly", None, 0, "monthly, no DTE floor"),
    ("calendar_monthly", None, 15, "monthly, stop restriking inside 15 DTE"),
    ("calendar_monthly", None, 30, "monthly, stop restriking inside 30 DTE"),
    ("trading_days", 30, 40, "one restrike ~30 trading days (~1mo) in, then hold"),
]


# ------------------------------------------------------------------ trigger
def _should_restrike(open_pos: dict, dt: pd.Timestamp, i: int, cadence: str,
                     n_days: int | None, min_dte_for_restrike: int) -> bool:
    """Pure function of calendar/index state (expiry, last restrike date/idx)
    vs. (dt, i) -- never reads P&L or the day's mark price, so the trigger is
    a pre-committed rule, not a hindsight-gated overlay decision."""
    if cadence == "none":
        return False
    dte_remaining = (open_pos["expiry"] - dt).days
    if dte_remaining < min_dte_for_restrike:
        return False
    if cadence == "trading_days":
        return (i - open_pos["last_restrike_idx"]) >= n_days
    last = open_pos["last_restrike_date"]
    return (dt.year, dt.month) != (last.year, last.month)


# ------------------------------------------------------------------ strikes
def _select_strike(day: pd.DataFrame, expiry: pd.Timestamp, spot: float,
                   mode: str, target_delta: float) -> tuple[float | None, float | None]:
    """(call_strike, put_strike) -- equal for mode='atm' (true straddle),
    generally unequal for mode='delta' (independent per-side selection)."""
    if mode == "atm":
        k = qs._atm_strike(day, expiry, spot)
        return k, k
    call_row = dss._delta_leg(day, expiry, "C", target_delta)
    put_row = dss._delta_leg(day, expiry, "P", target_delta)
    call_k = float(call_row[ds.C_STRIKE]) if call_row is not None else None
    put_k = float(put_row[ds.C_STRIKE]) if put_row is not None else None
    return call_k, put_k


def _entry_price(raw_px: float, slippage: float) -> float:
    return max(raw_px - slippage, 0.0)


def _exit_price(raw_px: float, slippage: float, at_settlement: bool) -> float:
    return raw_px if at_settlement else raw_px + slippage


def _open_position(day: pd.DataFrame, expiry: pd.Timestamp, spot: float,
                   call_k: float | None, put_k: float | None, dt: pd.Timestamp,
                   i: int, slippage: float, cycle_id: int, n_restrikes: int) -> dict | None:
    if call_k is None or put_k is None:
        return None
    call_row = qs._leg_row(day, expiry, "C", call_k)
    put_row = qs._leg_row(day, expiry, "P", put_k)
    if call_row is None or put_row is None:
        return None
    call_px = _entry_price(float(call_row[ds.C_CLOSE]), slippage)
    put_px = _entry_price(float(put_row[ds.C_CLOSE]), slippage)
    legs_df = pd.DataFrame([
        {"expiry": expiry, "cp": "C", "strike": call_k, "direction": -1,
         "qty": 1, "entry_price": call_px},
        {"expiry": expiry, "cp": "P", "strike": put_k, "direction": -1,
         "qty": 1, "entry_price": put_px},
    ])
    margin_yuan, _ = mg.compute_deposit(legs_df, spot)
    return {
        "cycle_id": cycle_id, "n_restrikes": n_restrikes,
        "entry_date": dt, "entry_spot": spot, "expiry": expiry,
        "call_strike": call_k, "put_strike": put_k,
        "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
        "call_px": call_px, "put_px": put_px, "cost": -(call_px + put_px),
        "margin": margin_yuan,
        "last_restrike_date": dt, "last_restrike_idx": i,
    }


def _trade_row(open_pos: dict, exit_date: pd.Timestamp, exit_spot: float,
               call_exit_px: float, put_exit_px: float, pnl: float,
               exit_reason: str) -> dict:
    return {
        "cycle_id": open_pos["cycle_id"], "segment_index": open_pos["n_restrikes"],
        "entry_date": open_pos["entry_date"], "exit_date": exit_date,
        "expiry": open_pos["expiry"], "exit_reason": exit_reason,
        "call_strike": open_pos["call_strike"], "put_strike": open_pos["put_strike"],
        "entry_spot": open_pos["entry_spot"], "exit_spot": exit_spot,
        "dte_at_entry": (open_pos["expiry"] - open_pos["entry_date"]).days,
        "dte_at_exit": (open_pos["expiry"] - exit_date).days,
        "call_entry_px": open_pos["call_px"], "put_entry_px": open_pos["put_px"],
        "call_exit_px": call_exit_px, "put_exit_px": put_exit_px,
        "credit": -open_pos["cost"], "pnl": pnl,
        "days_held": (exit_date - open_pos["entry_date"]).days,
        "margin_yuan": open_pos["margin"],
    }


# ------------------------------------------------------------------ engine
def run(underlying: str, roll_freq: str = "quarterly",
       restrike_cadence: str = "calendar_monthly", restrike_n_days: int | None = 21,
       strike_mode: str = "atm", target_delta: float = 0.30,
       min_dte_for_restrike: int = 10, slippage: float = 0.0,
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
    cycle_id = 0

    for i, dt in enumerate(dates):
        dt = pd.Timestamp(dt)
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            call = day[day[ds.C_CODE] == open_pos["call_code"]]
            put = day[day[ds.C_CODE] == open_pos["put_code"]]
            expired = dt >= open_pos["expiry"]
            missing = call.empty or put.empty

            if expired or missing:
                call_px = qs._intrinsic("C", open_pos["call_strike"], spot)
                put_px = qs._intrinsic("P", open_pos["put_strike"], spot)
                mark = -(call_px + put_px)
                pnl = mark - open_pos["cost"]
                realized += pnl
                trades.append(_trade_row(open_pos, dt, spot, call_px, put_px, pnl, "expiry"))
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue

            call_raw = float(call[ds.C_CLOSE].iloc[0])
            put_raw = float(put[ds.C_CLOSE].iloc[0])

            if _should_restrike(open_pos, dt, i, restrike_cadence, restrike_n_days,
                                min_dte_for_restrike):
                call_px = _exit_price(call_raw, slippage, at_settlement=False)
                put_px = _exit_price(put_raw, slippage, at_settlement=False)
                mark = -(call_px + put_px)
                pnl = mark - open_pos["cost"]
                realized += pnl
                trades.append(_trade_row(open_pos, dt, spot, call_px, put_px, pnl, "restrike"))

                expiry = open_pos["expiry"]
                next_cycle_id, next_n_restrikes = open_pos["cycle_id"], open_pos["n_restrikes"] + 1
                call_k, put_k = _select_strike(day, expiry, spot, strike_mode, target_delta)
                new_pos = _open_position(day, expiry, spot, call_k, put_k, dt, i, slippage,
                                         next_cycle_id, next_n_restrikes)
                open_pos = new_pos  # None if no valid strike -- stays flat until next entry scan
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue

            mark = -(call_raw + put_raw)
            pnl = mark - open_pos["cost"]
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: open a new quarterly cycle
        qexp = qs._target_expiries(day, roll_freq)
        if not qexp:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue
        expiry = pd.Timestamp(qexp[0])
        call_k, put_k = _select_strike(day, expiry, spot, strike_mode, target_delta)
        new_pos = _open_position(day, expiry, spot, call_k, put_k, dt, i, slippage,
                                 cycle_id + 1, 0)
        if new_pos is not None:
            cycle_id += 1
        open_pos = new_pos
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


def _cycle_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """One row per quarterly cycle: summed pnl/credit across all its
    segments, plus how many times it was re-struck -- the fair unit to
    compare against quarterly_atm_short_straddle.py's per-quarter trades."""
    if trades.empty:
        return pd.DataFrame(columns=["cycle_id", "entry_date", "exit_date", "pnl",
                                     "margin_yuan", "n_restrikes"])
    g = trades.groupby("cycle_id")
    return g.agg(entry_date=("entry_date", "min"), exit_date=("exit_date", "max"),
                pnl=("pnl", "sum"), margin_yuan=("margin_yuan", "mean"),
                n_restrikes=("segment_index", "max")).reset_index()


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, restrike_cadence: str, strike_mode: str,
          min_dte_for_restrike: int, baseline: dict | None = None) -> None:
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        print(f"{underlying}: no trades.")
        return
    cycles = _cycle_pnl(trades)
    pnl_yuan = cycles["pnl"] * ds.MULTIPLIER
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan
    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    avg_margin = cycles["margin_yuan"].mean()
    total_pnl = pnl_yuan.sum()
    cagr_on_margin = (total_pnl / avg_margin) / span_years if span_years > 0 else np.nan

    print(f"{underlying}: quarterly short straddle, restrike={restrike_cadence} "
          f"(min_dte_for_restrike={min_dte_for_restrike}), strike_mode={strike_mode}")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_cycles={len(cycles)}  avg_restrikes/cycle={cycles['n_restrikes'].mean():.1f}  "
          f"n_segments={len(trades)}")
    print(f"  total_pnl={total_pnl:,.0f} yuan (per-cycle)  win_rate(cycle)={(cycles['pnl']>0).mean():.0%}")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin={avg_margin:,.0f} yuan  return_on_margin/yr~={cagr_on_margin:.1%}")

    print("\n  Segment premium vs. remaining DTE (tests: does re-striking late "
          "sell thin premium into rising gamma?):")
    for seg, g in trades.groupby("segment_index"):
        print(f"    segment {seg}: n={len(g):>3d}  avg_dte_at_entry={g['dte_at_entry'].mean():>5.0f}d  "
              f"avg_credit={g['credit'].mean()*ds.MULTIPLIER:>8,.0f} yuan  "
              f"avg_pnl={g['pnl'].mean()*ds.MULTIPLIER:>8,.0f} yuan")

    if baseline is not None and not baseline["trades"].empty:
        bt = baseline["trades"]
        common_start = max(cycles["entry_date"].min(), bt["entry_date"].min())
        b = bt[bt["entry_date"] >= common_start]
        c = cycles[cycles["entry_date"] >= common_start]
        b_pnl = b["pnl"].sum() * ds.MULTIPLIER
        c_pnl = c["pnl"].sum() * ds.MULTIPLIER
        print(f"\n  vs. static ATM baseline (quarterly_atm_short_straddle.py), "
              f"same window from {common_start.date()}:")
        print(f"    restrike variant: n={len(c)}  total_pnl={c_pnl:>10,.0f}  "
              f"mean_r={(c['pnl']*ds.MULTIPLIER/c['margin_yuan']).mean():+.2%}")
        print(f"    static baseline:  n={len(b)}  total_pnl={b_pnl:>10,.0f}  "
              f"mean_r={(b['pnl']*ds.MULTIPLIER/b['margin_yuan']).mean():+.2%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"quarterly_restrike_short_straddle_trades_{underlying}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\n  Trades saved to {trades_out}")


def plot(res: dict, underlying: str, out_path: Path, restrike_cadence: str) -> None:
    equity, trades = res["equity"], res["trades"]
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    cycles = _cycle_pnl(trades)

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
    for _, c in cycles.iterrows():
        ax1.axvline(c["entry_date"], color=C_AXIS, linewidth=0.6, alpha=0.5)
    ax1.set_ylabel("cumulative P&L (yuan)", color=C_INK2, fontsize=10)
    ax1.set_title(f"{underlying}: quarterly short straddle, restrike={restrike_cadence}",
                  color=C_INK, fontsize=12, loc="left", pad=10)

    ax2.fill_between(dd.index, 0, dd.values, color=C_RED, alpha=0.55, linewidth=0)
    ax2.set_ylabel("drawdown (yuan)", color=C_INK2, fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ sweep
def _cell_stats(cycles: pd.DataFrame) -> dict:
    if cycles.empty:
        return {"n": 0, "mean_r": np.nan}
    r = cycles["pnl"] * ds.MULTIPLIER / cycles["margin_yuan"]
    return {"n": len(cycles), "mean_r": r.mean()}


def sweep(underlying: str, strike_modes: list[str] | None = None,
         slippages: list[float] | None = None) -> pd.DataFrame:
    """Grid over RESTRIKE_SPECS x strike_mode x slippage. Isolates the
    restrike-cadence/DTE-floor effect the same way dss.delta_sweep isolates
    the delta effect: discovery/holdout split, MIN_CELL_N flag."""
    strike_modes = strike_modes or ["atm"]
    slippages = slippages if slippages is not None else [0.0, 0.01]
    rows = []
    for cadence, n_days, min_dte, label in RESTRIKE_SPECS:
        for mode in strike_modes:
            for slip in slippages:
                res = run(underlying, restrike_cadence=cadence, restrike_n_days=n_days,
                          strike_mode=mode, min_dte_for_restrike=min_dte, slippage=slip)
                cycles = _cycle_pnl(res["trades"])
                disc = cycles[cycles["entry_date"] < SPLIT_DATE]
                hold_ = cycles[cycles["entry_date"] >= SPLIT_DATE]
                d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
                rows.append({
                    "restrike_spec": label, "cadence": cadence, "min_dte_for_restrike": min_dte,
                    "strike_mode": mode, "slippage": slip,
                    "avg_restrikes": cycles["n_restrikes"].mean() if not cycles.empty else np.nan,
                    "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                    "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"],
                })

    grid = pd.DataFrame(rows)
    print(f"{underlying}: restrike sweep ({len(grid)} cells), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")
    for _, row in grid.iterrows():
        flag = "" if row["disc_n"] >= MIN_CELL_N else "  (< MIN_CELL_N, low confidence)"
        print(f"  {row['restrike_spec']:<42s} mode={row['strike_mode']:<4s} slip={row['slippage']:.2f}  "
              f"avg_restrikes/cycle={row['avg_restrikes']:.1f}  "
              f"disc: n={row['disc_n']:>3.0f} mean_r={row['disc_mean_r']:+7.2%}   "
              f"hold: n={row['hold_n']:>3.0f} mean_r={row['hold_mean_r']:+7.2%}{flag}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"quarterly_restrike_short_straddle_sweep_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"  Full grid saved to {grid_out}\n")
    return grid


# ------------------------------------------------------------------ main
def main(underlying: str, restrike_cadence: str, restrike_n_days: int | None,
        strike_mode: str, target_delta: float, min_dte_for_restrike: int,
        slippage: float, do_sweep: bool) -> None:
    if do_sweep:
        sweep(underlying)
        return
    res = run(underlying, restrike_cadence=restrike_cadence, restrike_n_days=restrike_n_days,
             strike_mode=strike_mode, target_delta=target_delta,
             min_dte_for_restrike=min_dte_for_restrike, slippage=slippage)
    baseline = qs.run(underlying) if restrike_cadence != "none" else None
    report(res, underlying, restrike_cadence, strike_mode, min_dte_for_restrike, baseline=baseline)
    out_path = OUT_DIR / f"quarterly_restrike_short_straddle_{underlying}.png"
    plot(res, underlying, out_path, restrike_cadence)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--restrike-cadence", choices=["calendar_monthly", "trading_days", "none"],
                    default="calendar_monthly", dest="restrike_cadence")
    ap.add_argument("--restrike-days", type=int, default=21, dest="restrike_n_days",
                    help="for --restrike-cadence trading_days: re-strike every N trading days")
    ap.add_argument("--strike-mode", choices=["atm", "delta"], default="atm", dest="strike_mode")
    ap.add_argument("--target-delta", type=float, default=0.30, dest="target_delta",
                    help="for --strike-mode delta: sell the strike closest to +/- this delta")
    ap.add_argument("--min-dte-for-restrike", type=int, default=10, dest="min_dte_for_restrike",
                    help="never re-strike once fewer than this many days remain to expiry")
    ap.add_argument("--slippage", type=float, default=0.0,
                    help="adverse half-spread per leg, index points (default 0, lab convention)")
    ap.add_argument("--sweep", action="store_true", dest="do_sweep",
                    help="grid over RESTRIKE_SPECS x strike_mode x slippage instead")
    args = ap.parse_args()
    main(args.underlying, args.restrike_cadence, args.restrike_n_days, args.strike_mode,
        args.target_delta, args.min_dte_for_restrike, args.slippage, args.do_sweep)
