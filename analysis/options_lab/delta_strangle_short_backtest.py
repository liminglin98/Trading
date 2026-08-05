"""Naked short STRANGLE (sell 1x OTM call + 1x OTM put, same expiry, strikes
picked independently by target Delta rather than both legs at the ATM
strike) -- same VRP entry trigger as vol_rich_short_straddle_backtest.py
(reused, not reimplemented), only the strike-selection mechanism changes.

Motivation: the DTE-bucket sweep on vol_rich_short_straddle_backtest.py found
no edge from harvesting more theta/vega via shorter DTE, plausibly because
the extra gamma/pin risk of an ATM (delta~0.5) short straddle near expiry
offsets the extra decay. Selling further OTM (lower |delta|) trades some
premium for materially less gamma exposure at the same DTE -- this script
tests whether that tradeoff nets out better, using Delta (not DTE) as the
lever this time.

Strike selection: for a target_delta in (0, 1), the call leg is whichever
listed strike has Delta closest to +target_delta; the put leg, closest to
-target_delta. Picked independently per side (a real skew can put them at
different moneyness distances from spot) -- this is standard delta-targeted
strangle construction (e.g. "sell the 20-delta strangle"), not a symmetric
same-strike straddle. target_delta=0.5 is the closest analog to the sibling
script's ATM straddle (not identical: ATM there means nearest-strike-to-spot,
here means nearest-delta-to-0.5 -- the two usually agree but can differ by
one strike rung when spot sits between two listed strikes).

Same conventions as vol_rich_short_straddle_backtest.py: yesterday's VRP
signal gates entry, sequential single-position tracking, stop-loss (2x
credit) / max_hold / expiry-intrinsic exits, margin-sized via margin.py
(handles differing call/put strikes as two independent naked legs, same as
it already does for the straddle case), discovery < 2024-01-01 / holdout
>= 2024-01-01 split, pnl normalized on margin.

Requires real per-leg Delta (only validated for 510050/510300's own exchange
Greeks and the other 3 SSE names' officially-replaced Greeks -- see
[[reference-data-sources]] equivalent memory; SZSE Greeks are
BS-derived/unvalidated).

Usage:
    python delta_strangle_short_backtest.py --underlying 510050 --target-delta 0.20
    python delta_strangle_short_backtest.py --underlying 510050 --delta-sweep
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
import vol_rich_short_straddle_backtest as vr

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = vr.SPLIT_DATE
UNDERLYINGS = vr.UNDERLYINGS

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"

TARGET_DELTAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
MIN_CELL_N = 15
STOP_LOSS_MULT = vr.STOP_LOSS_MULT


# ------------------------------------------------------------------ strikes
def _delta_leg(day: pd.DataFrame, expiry: pd.Timestamp, cp: str,
               target_delta: float) -> pd.Series | None:
    """Listed strike (for this expiry/cp) whose Delta is closest to
    +target_delta (calls) or -target_delta (puts). None if no strike in the
    chain has a non-null Delta for this expiry/cp."""
    want = target_delta if cp == "C" else -target_delta
    m = day[(day[ds.C_EXPIRY] == expiry) & (day[ds.C_CP] == cp) &
            (day[ds.C_DELTA].notna()) & (day[ds.C_CLOSE] > 0)]
    if m.empty:
        return None
    idx = (m[ds.C_DELTA] - want).abs().idxmin()
    return m.loc[idx]


# ------------------------------------------------------------------ engine
def run(underlying: str, threshold: float, max_hold_days: int, target_delta: float,
        stop_loss_mult: float | None = STOP_LOSS_MULT, dte_min: int = 10,
        dte_max: int = 45) -> dict:
    chain = vr._prepped_chain(underlying)
    dates = sorted(chain[ds.C_DATE].unique())
    by_date = dict(tuple(chain.groupby(ds.C_DATE)))
    sig = vr._signal_series(underlying, "vrp")

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
                call_px = qs._intrinsic("C", open_pos["call_strike"], spot)
                put_px = qs._intrinsic("P", open_pos["put_strike"], spot)
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
                    "call_strike": open_pos["call_strike"], "put_strike": open_pos["put_strike"],
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "signal_at_entry": open_pos["signal_at_entry"],
                    "call_delta_at_entry": open_pos["call_delta_at_entry"],
                    "put_delta_at_entry": open_pos["put_delta_at_entry"],
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": days_held, "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: yesterday's VRP >= threshold, near-month expiry in range
        prev_dt = pd.Timestamp(dates[i - 1]) if i > 0 else None
        p = sig.get(prev_dt, np.nan) if prev_dt is not None else np.nan
        ok = (p == p) and p >= threshold
        if ok:
            expiries = sorted(day[ds.C_EXPIRY].unique())
            target = [e for e in expiries if dte_min <= (e - dt).days <= dte_max]
            if target:
                expiry = pd.Timestamp(target[0])
                call_row = _delta_leg(day, expiry, "C", target_delta)
                put_row = _delta_leg(day, expiry, "P", target_delta)
                if call_row is not None and put_row is not None:
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
                        "entry_date": dt, "entry_idx": i, "entry_spot": spot,
                        "expiry": expiry, "call_strike": call_strike, "put_strike": put_strike,
                        "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
                        "last_call_px": call_px, "last_put_px": put_px,
                        "cost": -(call_px + put_px), "margin": margin_yuan,
                        "signal_at_entry": float(p),
                        "call_delta_at_entry": float(call_row[ds.C_DELTA]),
                        "put_delta_at_entry": float(put_row[ds.C_DELTA]),
                    }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, threshold: float, max_hold_days: int,
           target_delta: float) -> None:
    trades, equity = res["trades"], res["equity"]
    name = UNDERLYINGS.get(underlying, underlying)
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

    print(f"{underlying} ({name}): delta-strangle short (target_delta={target_delta:.2f}, "
          f"yesterday's vrp>={threshold:.2f}, max_hold={max_hold_days}d, "
          f"stop_loss={STOP_LOSS_MULT:.1f}x credit)")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  avg_days_held={trades['days_held'].mean():.1f}  "
          f"avg call/put delta at entry={trades['call_delta_at_entry'].mean():+.2f}/"
          f"{trades['put_delta_at_entry'].mean():+.2f}")
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
    trades_out = OUT_DIR / f"delta_strangle_short_trades_{underlying}_{target_delta:.2f}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"  Trades saved to {trades_out}\n")


def plot(res: dict, underlying: str, target_delta: float, out_path: Path) -> None:
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
    ax1.set_title(f"{underlying} ({name}): delta-strangle short (target_delta={target_delta:.2f})",
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


def delta_sweep(underlying: str, threshold: float, max_hold_days: int,
                 target_deltas: list | None = None) -> pd.DataFrame:
    """Holds the already-validated VRP entry signal/threshold/max_hold fixed
    and varies only target_delta -- isolates the strike-selection effect the
    same way dte_bucket_sweep isolated the DTE-window effect."""
    target_deltas = target_deltas or TARGET_DELTAS
    name = UNDERLYINGS.get(underlying, underlying)
    rows = []
    for td in target_deltas:
        res = run(underlying, threshold=threshold, max_hold_days=max_hold_days, target_delta=td)
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
    print(f"{underlying} ({name}) threshold={threshold:.2f} max_hold={max_hold_days}d: "
          f"delta sweep ({len(grid)} target_deltas), discovery < {SPLIT_DATE.date()} <= holdout, "
          f"MIN_CELL_N={MIN_CELL_N}")
    for _, row in grid.iterrows():
        flag = "" if row["disc_n"] >= MIN_CELL_N else "  (< MIN_CELL_N, low confidence)"
        print(f"  target_delta={row['target_delta']:.2f}  actual C/P delta="
              f"{row['avg_call_delta']:+.2f}/{row['avg_put_delta']:+.2f}  "
              f"disc: n={row['disc_n']:>3.0f} mean_r={row['disc_mean_r']:+7.2%}   "
              f"hold: n={row['hold_n']:>3.0f} mean_r={row['hold_mean_r']:+7.2%}{flag}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"delta_strangle_short_sweep_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"  Full grid saved to {grid_out}\n")
    return grid


def joint_sweep(underlying: str, target_deltas: list | None = None) -> pd.DataFrame:
    """Full (threshold, max_hold, target_delta) grid -- checks whether
    target_delta=0.30 (the winner at the one already-frozen threshold=0.06/
    max_hold=20 cell from delta_sweep) is a genuinely stable optimum across
    the signal's other (threshold, max_hold) combinations, or just happened
    to be the best delta at that one cell. Reuses vr.SIGNAL_MINS['vrp'] and
    vr.MAX_HOLDS -- same grids vol_rich_short_straddle_backtest.py sweeps,
    so results are directly comparable to that script's frozen-pick table."""
    target_deltas = target_deltas or TARGET_DELTAS
    name = UNDERLYINGS.get(underlying, underlying)
    thresholds = vr.SIGNAL_MINS["vrp"]
    holds = vr.MAX_HOLDS
    rows = []
    for threshold in thresholds:
        for hold in holds:
            for td in target_deltas:
                res = run(underlying, threshold=threshold, max_hold_days=hold, target_delta=td)
                trades = res["trades"]
                disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
                hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
                d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
                rows.append({"threshold": threshold, "max_hold": hold, "target_delta": td,
                             "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                             "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"delta_strangle_short_joint_sweep_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"{underlying} ({name}): joint sweep ({len(grid)} cells = {len(thresholds)} thresholds x "
          f"{len(holds)} max_holds x {len(target_deltas)} target_deltas), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")
    print(f"  Full grid saved to {grid_out}")

    eligible = grid[grid["disc_n"] >= MIN_CELL_N]
    if eligible.empty:
        print("  No cell has >= MIN_CELL_N discovery trades.\n")
        return grid
    both_pos = eligible[(eligible["disc_mean_r"] > 0) & (eligible["hold_mean_r"] > 0)]
    both_neg = eligible[(eligible["disc_mean_r"] < 0) & (eligible["hold_mean_r"] < 0)]
    print(f"  Of {len(eligible)} eligible cells: {len(both_pos)} agree-positive, "
          f"{len(both_neg)} agree-negative, {len(eligible) - len(both_pos) - len(both_neg)} mixed sign")

    best = eligible.loc[eligible["disc_mean_r"].idxmax()]
    print(f"  Best overall (discovery mean_r, n>={MIN_CELL_N}): threshold={best['threshold']:.2f}  "
          f"max_hold={best['max_hold']:.0f}d  target_delta={best['target_delta']:.2f}")
    print(f"    discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.2%}")
    print(f"    holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.2%}")

    # per (threshold, max_hold): which target_delta wins on discovery, and does
    # it cluster (stable optimum) or scatter (0.30 was a one-cell coincidence)?
    print("\n  Best target_delta per (threshold, max_hold) cell (eligible only, n>=MIN_CELL_N):")
    winners = []
    for (threshold, hold), g in eligible.groupby(["threshold", "max_hold"]):
        w = g.loc[g["disc_mean_r"].idxmax()]
        winners.append(w["target_delta"])
        print(f"    threshold={threshold:.2f} max_hold={hold:>2.0f}d: "
              f"best target_delta={w['target_delta']:.2f}  "
              f"disc n={w['disc_n']:.0f} mean_r={w['disc_mean_r']:+.2%}   "
              f"hold n={w['hold_n']:.0f} mean_r={w['hold_mean_r']:+.2%}")
    winners = pd.Series(winners)
    print(f"\n  Winning target_delta distribution across {len(winners)} (threshold, max_hold) "
          f"cells: {winners.value_counts().sort_index().to_dict()}")
    print(f"  median={winners.median():.2f}  mean={winners.mean():.2f}\n")
    return grid


# ------------------------------------------------------------------ main
def main(underlying: str, threshold: float, max_hold_days: int, target_delta: float,
         do_sweep: bool, do_joint_sweep: bool = False) -> None:
    targets = list(UNDERLYINGS) if underlying == "all" else [underlying]
    for u in targets:
        if do_joint_sweep:
            joint_sweep(u)
            continue
        if do_sweep:
            delta_sweep(u, threshold, max_hold_days)
            continue
        res = run(u, threshold=threshold, max_hold_days=max_hold_days, target_delta=target_delta)
        report(res, u, threshold, max_hold_days, target_delta)
        out_path = OUT_DIR / f"delta_strangle_short_{u}_{target_delta:.2f}.png"
        plot(res, u, target_delta, out_path)
        print(f"  Chart saved to {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050", choices=list(UNDERLYINGS) + ["all"])
    ap.add_argument("--threshold", type=float, default=0.06,
                    help="entry when yesterday's vrp >= this (default 0.06, the frozen "
                         "510050 pick from vol_rich_short_straddle_backtest.py)")
    ap.add_argument("--max-hold", type=int, default=20, dest="max_hold_days")
    ap.add_argument("--target-delta", type=float, default=0.20, dest="target_delta",
                    help="sell the strike closest to +/- this delta on each side")
    ap.add_argument("--delta-sweep", action="store_true", dest="do_sweep",
                    help="fix threshold/max_hold, sweep TARGET_DELTAS instead")
    ap.add_argument("--joint-sweep", action="store_true", dest="do_joint_sweep",
                    help="full (threshold, max_hold, target_delta) grid -- checks whether "
                         "0.30 is a stable optimum across cells, not just the one frozen one")
    args = ap.parse_args()
    main(args.underlying, args.threshold, args.max_hold_days, args.target_delta, args.do_sweep,
         args.do_joint_sweep)
