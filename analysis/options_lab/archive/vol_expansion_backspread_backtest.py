"""Ratio backspread (1x short ATM + 2x long OTM, same expiry, same side)
triggered by the same rv_pct "cheap/quiet now, bet on expansion" signal as
long_straddle_rv_contraction_backtest.py, but direction-committed instead of
direction-agnostic: a trailing-momentum filter picks call backspread (bet
big up move) vs put backspread (bet big down move) at each signal date.

Payoff shape differs fundamentally from the straddle: bounded max loss at
strike K2 (the long/OTM strike) of (width + net_debit), ~flat/small near K1,
unbounded (bounded by 0) profit beyond K2 in the bet direction. Worst outcome
is a MODERATE move that stalls between the strikes, not a move in the wrong
direction -- opposite of the straddle's "any big move wins" shape. Because
net_debit can be ~0 or negative (net credit), P&L is normalized by strike
WIDTH, not by debit (debit-based % returns blow up near zero/negative
denominators -- same trap flagged in ma_signal_spread_backtest.py's cost
sensitivity study for credit spreads).

Momentum filter (untested before this script): sign of the underlying's own
trailing N-day return at the signal date. This is a new, unvalidated
directional signal -- the discovery/holdout split below is that validation,
not a formality.

Follows this repo's discovery/holdout discipline (condor_delta_investigation_summary.md):
discovery < 2024-01-01 <= holdout, split fixed once.

Usage:
    python vol_expansion_backspread_backtest.py --underlying 510500
    python vol_expansion_backspread_backtest.py --underlying all --sweep
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # archived: options_lab/ modules live one level up

import data_store as ds
import quarterly_atm_short_straddle as qs
import long_straddle_rv_contraction_backtest as ls

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = ls.SPLIT_DATE
UNDERLYINGS = ls.UNDERLYINGS

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"

MOMENTUM_WINDOW = 20  # trailing trading days for the direction filter
WING_PCTS = [0.05, 0.10]
MAX_HOLDS = [5, 10, 20, 30]
PROFIT_TARGETS = [0.25, 0.5, 1.0, None]  # multiple of strike WIDTH; None = hold to max_hold/expiry
MIN_CELL_N = 15
DEFAULT_THRESHOLD = 0.20  # rv_pct
DEFAULT_WING_PCT = 0.05


@lru_cache(maxsize=8)
def momentum_series(underlying: str, window: int = MOMENTUM_WINDOW) -> pd.Series:
    """Trailing N-day simple return of the underlying's own close. Sign only
    is used (>=0 -> call backspread / bet up, <0 -> put backspread / bet down)."""
    etf = ds.load_etf_ohlc(underlying)
    if etf is None:
        raise FileNotFoundError(f"no cached ETF OHLC for {underlying}")
    s = etf.set_index("date")["close"].sort_index()
    return s.pct_change(window)


def _pick_wing(day: pd.DataFrame, expiry: pd.Timestamp, cp: str, atm_strike: float,
                target_strike: float) -> pd.Series | None:
    """Nearest-strike contract at the same expiry/side as the ATM leg,
    distinct from the ATM strike itself."""
    same = day[(day[ds.C_EXPIRY] == expiry) & (day[ds.C_CP] == cp) &
               (day[ds.C_STRIKE] != atm_strike) & (day[ds.C_CLOSE] > 0)]
    if same.empty:
        return None
    return same.iloc[(same[ds.C_STRIKE] - target_strike).abs().argsort()].iloc[0]


# ------------------------------------------------------------------ engine
def run(underlying: str, threshold: float, wing_pct: float, max_hold_days: int,
        profit_target: float | None, dte_min: int = 10, dte_max: int = 45) -> dict:
    chain = ls._prepped_chain(underlying)
    sig = ls.rv_pct_series(underlying)
    mom = momentum_series(underlying)

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
            short_row = day[day[ds.C_CODE] == open_pos["short_code"]]
            long_row = day[day[ds.C_CODE] == open_pos["long_code"]]
            short_px = float(short_row[ds.C_CLOSE].iloc[0]) if not short_row.empty else open_pos["last_short_px"]
            long_px = float(long_row[ds.C_CLOSE].iloc[0]) if not long_row.empty else open_pos["last_long_px"]
            open_pos["last_short_px"], open_pos["last_long_px"] = short_px, long_px

            expired = pd.Timestamp(dt) >= open_pos["expiry"]
            days_held = i - open_pos["entry_idx"]
            reason = None
            if expired:
                short_px = qs._intrinsic(open_pos["cp"], open_pos["short_strike"], spot)
                long_px = qs._intrinsic(open_pos["cp"], open_pos["long_strike"], spot)
                reason = "expiry"

            value = 2 * long_px - short_px  # 1x short + 2x long
            pnl = value - open_pos["cost"]
            width = open_pos["width"]

            if reason is None:
                if profit_target is not None and pnl >= profit_target * width:
                    reason = "profit_target"
                elif days_held >= max_hold_days:
                    reason = "max_hold"

            if reason:
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "exit_date": dt,
                    "expiry": open_pos["expiry"], "exit_reason": reason,
                    "direction": open_pos["cp"], "short_strike": open_pos["short_strike"],
                    "long_strike": open_pos["long_strike"], "width": width,
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "signal_at_entry": open_pos["signal_at_entry"],
                    "momentum_at_entry": open_pos["momentum_at_entry"],
                    "net_debit": open_pos["cost"], "pnl": pnl, "days_held": days_held,
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: rv_pct cheap/quiet + momentum picks direction
        p = sig.get(dt, np.nan)
        m = mom.get(dt, np.nan)
        ok = (p == p) and p <= threshold and (m == m)
        if ok:
            cp = "C" if m >= 0 else "P"
            expiries = sorted(day[ds.C_EXPIRY].unique())
            target = [e for e in expiries if dte_min <= (e - pd.Timestamp(dt)).days <= dte_max]
            if target:
                expiry = pd.Timestamp(target[0])
                atm_strike = qs._atm_strike(day, expiry, spot)
                if atm_strike is not None:
                    target_wing = spot * (1 + wing_pct) if cp == "C" else spot * (1 - wing_pct)
                    wing_row = _pick_wing(day, expiry, cp, atm_strike, target_wing)
                    short_row = qs._leg_row(day, expiry, cp, atm_strike)
                    if wing_row is not None and short_row is not None:
                        long_strike = float(wing_row[ds.C_STRIKE])
                        long_px, short_px = float(wing_row[ds.C_CLOSE]), float(short_row[ds.C_CLOSE])
                        open_pos = {
                            "entry_date": pd.Timestamp(dt), "entry_idx": i, "entry_spot": spot,
                            "expiry": expiry, "cp": cp,
                            "short_strike": atm_strike, "long_strike": long_strike,
                            "width": abs(long_strike - atm_strike),
                            "short_code": short_row[ds.C_CODE], "long_code": wing_row[ds.C_CODE],
                            "last_short_px": short_px, "last_long_px": long_px,
                            "cost": 2 * long_px - short_px,
                            "signal_at_entry": float(p), "momentum_at_entry": float(m),
                        }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, threshold: float, wing_pct: float,
           max_hold_days: int, profit_target: float | None) -> None:
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
    r = trades["pnl"] / trades["width"]  # normalized by strike width, not debit (debit can be ~0/negative)

    pt_label = "none (hold to max_hold/expiry)" if profit_target is None else f"{profit_target:+.0%} of width"
    print(f"{underlying} ({name}): ratio backspread (rv_pct<={threshold:.0%}, "
          f"wing={wing_pct:.0%}, max_hold={max_hold_days}d, profit_target={pt_label})")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  call%={ (trades['direction']=='C').mean():.0%}  "
          f"avg_days_held={trades['days_held'].mean():.1f}")
    print(f"  total_pnl={pnl_yuan.sum():,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={r.mean():+.2%} (pnl/width per trade)  median_r={r.median():+.2%}")
    print(f"  mean net_debit/width={ (trades['net_debit']/trades['width']).mean():+.2%} "
          f"(negative = net credit received on average)")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    trades["year"] = trades["entry_date"].dt.year
    print("  by year:")
    for yr, g in trades.groupby("year"):
        gr = g["pnl"] / g["width"]
        print(f"    {yr}: n={len(g)}  mean_r={gr.mean():+.2%}  win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"backspread_trades_{underlying}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"  Trades saved to {trades_out}\n")


def plot(res: dict, underlying: str, out_path: Path) -> None:
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
    ax1.set_title(f"{underlying} ({name}): rv_pct ratio backspread", color=C_INK,
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
    r = trades["pnl"] / trades["width"]
    return {"n": len(trades), "mean_r": r.mean()}


def sweep(underlying: str) -> None:
    thresholds = ls.SIGNAL_MAXES["rv_pct"]
    rows = []
    for threshold, wing_pct, hold, pt in product(thresholds, WING_PCTS, MAX_HOLDS, PROFIT_TARGETS):
        res = run(underlying, threshold=threshold, wing_pct=wing_pct,
                  max_hold_days=hold, profit_target=pt)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        rows.append({"threshold": threshold, "wing_pct": wing_pct, "hold": hold,
                     "profit_target": "none" if pt is None else pt,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    name = UNDERLYINGS.get(underlying, underlying)
    print(f"{underlying} ({name}): sweep grid ({len(grid)} cells), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"backspread_sweep_{underlying}.csv"
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
          f"threshold={best['threshold']:.2f}  wing_pct={best['wing_pct']:.2f}  "
          f"max_hold={best['hold']:.0f}d  profit_target={best['profit_target']}")
    print(f"  discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.2%}")
    print(f"  holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.2%}\n")


# ------------------------------------------------------------------ main
def main(underlying: str, threshold: float, wing_pct: float, max_hold_days: int,
         profit_target: float | None, do_sweep: bool) -> None:
    targets = list(UNDERLYINGS) if underlying == "all" else [underlying]
    for u in targets:
        if do_sweep:
            sweep(u)
            continue
        res = run(u, threshold=threshold, wing_pct=wing_pct,
                  max_hold_days=max_hold_days, profit_target=profit_target)
        report(res, u, threshold, wing_pct, max_hold_days, profit_target)
        out_path = OUT_DIR / f"backspread_{u}.png"
        plot(res, u, out_path)
        print(f"  Chart saved to {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="all", choices=list(UNDERLYINGS) + ["all"])
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="enter when rv_pct <= this (default 0.20)")
    ap.add_argument("--wing-pct", type=float, default=DEFAULT_WING_PCT, dest="wing_pct",
                    help="OTM distance for the 2x long leg, as %% of spot (default 0.05)")
    ap.add_argument("--max-hold", type=int, default=20, dest="max_hold_days",
                    help="max trading days to hold before closing regardless of expiry")
    ap.add_argument("--profit-target", type=float, default=0.5,
                    help="close when position value gains this multiple of strike WIDTH "
                         "(not debit -- debit can be ~0/negative); pass --no-profit-target for none")
    ap.add_argument("--no-profit-target", action="store_true",
                    help="disable the profit target (hold to max_hold/expiry only)")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search (threshold, wing_pct, max_hold, profit_target) on discovery, "
                         "score the frozen pick on holdout")
    args = ap.parse_args()
    pt = None if args.no_profit_target else args.profit_target
    main(args.underlying, args.threshold, args.wing_pct, args.max_hold_days, pt, args.sweep)
