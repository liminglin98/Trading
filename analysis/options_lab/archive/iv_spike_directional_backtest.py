"""IV-spike-triggered directional premium sale: sell ONE side (call or put,
whichever side's own ATM IV just spiked more) after its near-month ATM IV
jumps >= --spike-pct (relative) over a trailing --spike-window trading days,
comparing a naked single leg against a defined-risk credit vertical spread
(short the ATM-side strike, buy protection --width listed strikes further
OTM). Exits after a fixed short holding period (trading days), not held to
expiry, with a stop-loss safety net -- same short-holding-period mechanism
as iv_spike_short_straddle_backtest.py, which is what kept that strategy's
worst trade to -35% of margin instead of the -90%+ blowups the monthly
condor investigation found (condor_delta_investigation_summary.md).

Side selection: compute call-side and put-side near-month ATM IV
separately. A day triggers if either side's IV jumped >= threshold; if both
did, sell the side with the larger jump (that side is "more mispriced").

Grid search: (spike_window, spike_pct, max_hold_days, width) is a 4-D grid,
216 cells at the default resolution. Brute force per cell (a full
day-by-day walk) profiled at ~3.3s/cell dominated by two things that are
identical across every (hold, width) at a fixed (window, pct): reloading
the chain CSV and recomputing the call/put ATM IV series. Both are cached
by `underlying` (_prepped_chain, _iv_sides via lru_cache), cutting the full
216-cell sweep from ~720s to a few minutes. The remaining per-cell cost is
the day-by-day position walk itself, which does legitimately differ across
(hold, width) and hasn't been vectorized -- if the grid grows much further,
the next win is precomputing each day's full sorted strike ladder once per
(underlying, side) instead of re-deriving it inside the entry branch on
every triggered day.

Usage:
    python iv_spike_directional_backtest.py --underlying 510050
    python iv_spike_directional_backtest.py --underlying 510050 --width 3
    python iv_spike_directional_backtest.py --underlying 510050 --sweep
"""
from __future__ import annotations

import argparse
import sys
import time
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
WIDTHS = [0, 1, 2, 3, 4, 6]  # 0 = naked single leg
STOP_LOSS_MULT = 2.0  # fixed by convention, not swept (see straddle script's docstring)
MIN_CELL_N = 15
MIN_STRIKE_STEP = 0.03  # below the real ~0.05 grid but above observed sub-0.02
                        # artifact gaps between listed strikes on a handful of days


# ------------------------------------------------------------------ signal
def atm_iv_side_series(chain: pd.DataFrame, cp: str) -> pd.Series:
    """Daily near-month ATM IV for one side (cp='C' or 'P') -- same
    definition as backtest.py's entry gate, generalized to either side."""
    d = chain[(chain[ds.C_CP] == cp) & (chain[ds.C_IV] > 0)]
    d = d.loc[d.groupby(ds.C_DATE)[ds.C_DTE].transform(
        lambda s: s == s[s >= 7].min() if (s >= 7).any() else s == s.min())]
    idx = (d[ds.C_STRIKE] - d[ds.C_UND_CLOSE]).abs().groupby(d[ds.C_DATE]).idxmin()
    return d.loc[idx].set_index(ds.C_DATE)[ds.C_IV].sort_index()


def side_signal(iv_call: pd.Series, iv_put: pd.Series, window: int,
                pct_threshold: float) -> pd.DataFrame:
    """Per day: which side (if any) triggers, and by how much it jumped.
    Both sides' IV are reindexed onto the union of trading days so a gap in
    one side's near-month listing doesn't silently drop the other."""
    idx = iv_call.index.union(iv_put.index)
    call_jump = (iv_call.reindex(idx) / iv_call.reindex(idx).shift(window) - 1.0)
    put_jump = (iv_put.reindex(idx) / iv_put.reindex(idx).shift(window) - 1.0)
    call_ok = call_jump >= pct_threshold
    put_ok = put_jump >= pct_threshold
    side = np.where(call_ok & put_ok, np.where(call_jump >= put_jump, "C", "P"),
                    np.where(call_ok, "C", np.where(put_ok, "P", None)))
    jump = np.where(side == "C", call_jump, np.where(side == "P", put_jump, np.nan))
    return pd.DataFrame({"side": side, "jump": jump}, index=idx)


# ------------------------------------------------------------------ engine
@lru_cache(maxsize=4)
def _prepped_chain(underlying: str) -> pd.DataFrame:
    chain = ds.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    return chain[chain[ds.C_CLOSE] > 0]


@lru_cache(maxsize=4)
def _iv_sides(underlying: str) -> tuple[pd.Series, pd.Series]:
    """(call, put) near-month ATM IV series -- independent of hold/width, so
    cached once per underlying instead of recomputed on every sweep cell."""
    chain = _prepped_chain(underlying)
    return atm_iv_side_series(chain, "C"), atm_iv_side_series(chain, "P")


def run(underlying: str, spike_window: int, spike_pct: float, max_hold_days: int,
       width: int, stop_loss_mult: float | None = STOP_LOSS_MULT,
       dte_min: int = 10, dte_max: int = 45) -> dict:
    chain = _prepped_chain(underlying)
    iv_call, iv_put = _iv_sides(underlying)
    signal = side_signal(iv_call, iv_put, spike_window, spike_pct)

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
            short = day[day[ds.C_CODE] == open_pos["short_code"]]
            short_px = float(short[ds.C_CLOSE].iloc[0]) if not short.empty else open_pos["last_short_px"]
            if open_pos["long_code"] is not None:
                long_ = day[day[ds.C_CODE] == open_pos["long_code"]]
                long_px = float(long_[ds.C_CLOSE].iloc[0]) if not long_.empty else open_pos["last_long_px"]
            else:
                long_px = 0.0
            open_pos["last_short_px"], open_pos["last_long_px"] = short_px, long_px

            expired = pd.Timestamp(dt) >= open_pos["expiry"]
            days_held = i - open_pos["entry_idx"]
            reason = None
            if expired:
                short_px = qs._intrinsic(open_pos["cp"], open_pos["short_strike"], spot)
                long_px = (qs._intrinsic(open_pos["cp"], open_pos["long_strike"], spot)
                          if open_pos["long_strike"] is not None else 0.0)
                reason = "expiry"

            mark = -short_px + long_px
            pnl = mark - open_pos["cost"]
            basis = abs(open_pos["cost"])

            if reason is None:
                if stop_loss_mult is not None and basis > 0 and pnl <= -stop_loss_mult * basis:
                    reason = "stop_loss"
                elif days_held >= max_hold_days:
                    reason = "max_hold"

            if reason:
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "exit_date": dt,
                    "expiry": open_pos["expiry"], "exit_reason": reason,
                    "side": open_pos["cp"], "short_strike": open_pos["short_strike"],
                    "long_strike": open_pos["long_strike"],
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "iv_jump_at_entry": open_pos["iv_jump_at_entry"],
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": days_held, "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: only when a side just triggered and a near-month expiry is in range
        row = signal.loc[dt] if dt in signal.index else None
        side = row["side"] if row is not None else None
        if side in ("C", "P"):
            expiries = sorted(day[ds.C_EXPIRY].unique())
            target = [e for e in expiries if dte_min <= (e - pd.Timestamp(dt)).days <= dte_max]
            if target:
                expiry = pd.Timestamp(target[0])
                strikes = np.sort(day.loc[(day[ds.C_EXPIRY] == expiry) &
                                          (day[ds.C_CP] == side), ds.C_STRIKE].unique())
                if len(strikes):
                    atm_idx = int(np.argmin(np.abs(strikes - spot)))
                    short_strike = float(strikes[atm_idx])
                    long_strike = None
                    if width > 0:
                        j = atm_idx + width if side == "C" else atm_idx - width
                        if 0 <= j < len(strikes) and abs(strikes[j] - short_strike) >= MIN_STRIKE_STEP:
                            long_strike = float(strikes[j])
                        else:
                            long_strike = "skip"  # can't construct at this width today,
                            # or the listed strikes that day include a stray
                            # sub-0.03 duplicate (adjustment-contract residue,
                            # not a real strike step -- see MIN_STRIKE_STEP)
                    if long_strike != "skip":
                        short_row = qs._leg_row(day, expiry, side, short_strike)
                        long_row = (qs._leg_row(day, expiry, side, long_strike)
                                   if long_strike is not None else None)
                        ok = short_row is not None and (long_strike is None or long_row is not None)
                        if ok:
                            short_px = float(short_row[ds.C_CLOSE])
                            long_px = float(long_row[ds.C_CLOSE]) if long_row is not None else 0.0
                            legs = [{"expiry": expiry, "cp": side, "strike": short_strike,
                                    "direction": -1, "qty": 1, "entry_price": short_px}]
                            if long_row is not None:
                                legs.append({"expiry": expiry, "cp": side, "strike": long_strike,
                                            "direction": 1, "qty": 1, "entry_price": long_px})
                            margin_yuan, _ = mg.compute_deposit(pd.DataFrame(legs), spot)
                            open_pos = {
                                "entry_date": pd.Timestamp(dt), "entry_idx": i, "entry_spot": spot,
                                "expiry": expiry, "cp": side,
                                "short_strike": short_strike, "long_strike": long_strike,
                                "short_code": short_row[ds.C_CODE],
                                "long_code": long_row[ds.C_CODE] if long_row is not None else None,
                                "last_short_px": short_px, "last_long_px": long_px,
                                "cost": -short_px + long_px, "margin": margin_yuan,
                                "iv_jump_at_entry": float(row["jump"]),
                            }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, spike_window: int, spike_pct: float,
          max_hold_days: int, width: int) -> None:
    trades, equity = res["trades"], res["equity"]
    label = "naked" if width == 0 else f"credit spread (width={width})"
    if trades.empty:
        print(f"{underlying} [{label}]: no trades.")
        return
    n_degenerate = (trades["margin_yuan"] <= 0).sum()
    if n_degenerate:
        print(f"  (dropping {n_degenerate} zero-margin trade(s) -- near-zero-width "
              f"strike artifact, see _cell_stats docstring)")
        trades = trades[trades["margin_yuan"] > 0]
    pnl_yuan = trades["pnl"] * ds.MULTIPLIER
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan
    avg_margin = trades["margin_yuan"].mean()
    total_pnl = pnl_yuan.sum()
    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    cagr_on_margin = (total_pnl / avg_margin) / span_years if span_years > 0 else np.nan

    print(f"{underlying}: IV-spike directional {label} "
          f"(window={spike_window}d, spike>={spike_pct:.0%}, max_hold={max_hold_days}d)")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  side_split={trades['side'].value_counts().to_dict()}  "
          f"avg_days_held={trades['days_held'].mean():.1f}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={(pnl_yuan/trades['margin_yuan']).mean():+.2%} (pnl/margin per trade)")
    print(f"  avg_win={pnl_yuan[pnl_yuan>0].mean():,.0f}  "
          f"avg_loss={pnl_yuan[pnl_yuan<=0].mean():,.0f} yuan")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin={avg_margin:,.0f} yuan  return_on_margin/yr~={cagr_on_margin:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    OUT_DIR.mkdir(exist_ok=True)
    suffix = "naked" if width == 0 else f"spread{width}"
    trades_out = OUT_DIR / f"iv_spike_directional_trades_{underlying}_{suffix}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"  Trades saved to {trades_out}")


def plot(res: dict, underlying: str, width: int, out_path: Path) -> None:
    equity = res["equity"]
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
    label = "naked" if width == 0 else f"credit spread (width={width})"
    ax1.set_title(f"{underlying}: IV-spike directional {label}", color=C_INK,
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
    """Drops zero-margin trades before computing mean_r -- a handful of
    listed strikes sit ~0.001 apart (adjustment-contract artifacts in the
    chain), producing a near-zero-width "spread" whose exchange margin
    floors to 0 in margin.py's formula. Real brokers never actually offer
    zero margin; including these divides by zero and fabricates +inf
    returns on trades that moved ~1 yuan of real P&L."""
    trades = trades[trades["margin_yuan"] > 0]
    if trades.empty:
        return {"n": 0, "mean_r": np.nan}
    r = trades["pnl"] * ds.MULTIPLIER / trades["margin_yuan"]
    return {"n": len(trades), "mean_r": r.mean()}


def sweep(underlying: str) -> None:
    t0 = time.time()
    rows = []
    for window, pct, hold, width in product(SPIKE_WINDOWS, SPIKE_PCTS, MAX_HOLDS, WIDTHS):
        res = run(underlying, spike_window=window, spike_pct=pct,
                 max_hold_days=hold, width=width)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        rows.append({"window": window, "pct": pct, "hold": hold, "width": width,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})
    elapsed = time.time() - t0

    grid = pd.DataFrame(rows)
    n_cells = len(grid)
    print(f"{underlying}: sweep grid ({n_cells} cells) in {elapsed:.0f}s "
          f"({elapsed/n_cells*1000:.0f}ms/cell), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"iv_spike_directional_sweep_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"Full grid saved to {grid_out}")

    eligible = grid[grid["disc_n"] >= MIN_CELL_N]
    if eligible.empty:
        print("No cell has >= MIN_CELL_N discovery trades -- cannot freeze a pick.")
        return
    best = eligible.loc[eligible["disc_mean_r"].idxmax()]
    print(f"\nFrozen pick (best discovery mean_r, n>={MIN_CELL_N}): "
          f"window={best['window']:.0f}d  spike>={best['pct']:.0%}  "
          f"max_hold={best['hold']:.0f}d  width={best['width']:.0f}"
          f"{' (naked)' if best['width'] == 0 else ' (spread)'}")
    print(f"  discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.2%}")
    print(f"  holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.2%}")

    # naked vs. best-spread-width comparison at the frozen (window, pct, hold)
    same = grid[(grid["window"] == best["window"]) & (grid["pct"] == best["pct"]) &
               (grid["hold"] == best["hold"])].sort_values("width")
    print(f"\nNaked vs. spread width, holding (window,pct,hold) fixed at the frozen pick:")
    print(same.to_string(index=False, formatters={
        "disc_mean_r": lambda v: f"{v:+.2%}" if v == v else "n/a",
        "hold_mean_r": lambda v: f"{v:+.2%}" if v == v else "n/a"}))


# ------------------------------------------------------------------ main
def main(underlying: str, spike_window: int, spike_pct: float, max_hold_days: int,
        width: int, do_sweep: bool) -> None:
    if do_sweep:
        sweep(underlying)
        return
    res = run(underlying, spike_window=spike_window, spike_pct=spike_pct,
             max_hold_days=max_hold_days, width=width)
    report(res, underlying, spike_window, spike_pct, max_hold_days, width)
    suffix = "naked" if width == 0 else f"spread{width}"
    out_path = OUT_DIR / f"iv_spike_directional_{underlying}_{suffix}.png"
    plot(res, underlying, width, out_path)
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
    ap.add_argument("--width", type=int, default=0,
                    help="protective long leg, listed strikes past the short "
                         "(0 = naked single leg)")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search (window, pct, max_hold, width) on discovery, "
                         "score the frozen pick on holdout")
    args = ap.parse_args()
    main(args.underlying, args.spike_window, args.spike_pct, args.max_hold_days,
        args.width, args.sweep)
