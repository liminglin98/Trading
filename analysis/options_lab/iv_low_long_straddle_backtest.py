"""Long straddle triggered by "IV is cheap," with two alternative entry
signals -- betting IV normalizes upward (or a big move happens) rather than
staying depressed. The mirror image of iv_spike_short_straddle_backtest.py's
mean-reversion thesis, but NOT a mirror of its trigger: volatility spikes up
abruptly but decays down gradually (it "takes the stairs down"), so there's
rarely a clean "sudden IV drop" to trigger on. Both signals below gate on
level, not on a change.

--signal iv_pct (original): rolling 252-day percentile rank of near-month
ATM call IV, same definition as backtest.py's entry gate and
iv_percentile_filter.py. "Cheap relative to its OWN history."

--signal vrp (new): near-month ATM call IV minus a TRAILING realized vol
(past --rv-window trading days). Deliberately NOT vrp_analysis.py's VRP
definition -- that one uses forward-looking realized vol and is only valid
for after-the-fact measurement of whether selling paid off, never as a live
trigger (using it here would be lookahead bias). This trailing version asks
"cheap relative to what's actually been happening," a value condition
rather than a purely historical-percentile one -- e.g. it can fire even
when IV's own percentile rank isn't at an extreme, if realized vol has
quietly risen faster than IV has.

--signal gex_pct / gex_mag_pct: rolling percentile of net (or unsigned
magnitude) dealer gamma exposure (gamma_exposure.py) -- low/negative
percentile = dealers structurally short gamma, the textbook backdrop for
buying premium ahead of an expansion. Same signal already tested on the
other 5 underlyings via long_straddle_rv_contraction_backtest.py; added
here too so 510050/510300 get the same coverage on the long side that they
already have on the short side (vol_rich_short_straddle_backtest.py).

--signal flip_dist: raw signed distance of spot above the gamma flip point
(gamma_flip.flip_distance_series) -- low/negative = spot below the flip
point, the negative-gamma/vol-amplifying regime, the textbook backdrop for
buying. 510050/510300 ONLY by construction (gamma_flip.py needs
per-contract IV, only available for these two).

--signal pc_oi_pct / pc_vol_pct: rolling percentile of the put/call
open-interest or volume ratio (put_call_ratio.py) -- a positioning/
sentiment signal, no a priori direction assumption.

Unlike the naked short straddle, this needs NO margin -- long options are
fully paid for at entry, so max loss is capped at the debit paid by
construction (margin.py's compute_deposit correctly returns 0 for an
all-long leg set). Returns below are on debit paid (the true capital at
risk), not on an exchange-margin proxy.

Exit: profit target (position value reaches a multiple of the debit paid),
OR a max holding period (long options bleed theta every day nothing
happens, so this caps that bleed the same way the short version's max-hold
caps naked gamma exposure), OR expiry (settled at intrinsic). No stop-loss
is needed for capital preservation (the debit already bounds the loss) --
whether cutting losses early anyway helps or hurts is an empirical
question, not assumed away, so it isn't built in by default.

Follows this repo's established discovery/holdout discipline
(condor_delta_investigation_summary.md): discovery 2015-02-09 -> 2023-12-31,
holdout 2024-01-01 -> 2026-07-08, touched once.

Look-ahead robustness note: default entry gating (lag_signal_days=0) uses the
SAME trading date's signal value while pricing the trade at that same date's
close -- not a true look-ahead (iv_pct/vrp are rolling/trailing functions of
data through that date, never later dates), but a same-bar execution
idealization applied less strictly than this repo's "rich/extreme" trigger
family (calendar_spread_entry_timing_backtest.py,
skew_vrp_combo_short_straddle_backtest.py), which gate on the PRIOR trading
date's value. --lag-signal-days N reruns under that stricter convention as a
robustness check.

Usage:
    python iv_low_long_straddle_backtest.py --underlying 510300 --signal vrp
    python iv_low_long_straddle_backtest.py --underlying 510300 --signal vrp --sweep
    python iv_low_long_straddle_backtest.py --underlying 510300 --signal vrp --lag-signal-days 1
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import data_store as ds
import gamma_exposure as gx
import gamma_flip as gf
import put_call_ratio as pc
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

# sweep grids (discovery-only search, frozen pick scored once on holdout)
SIGNAL_MAXES = {
    "iv_pct": [0.05, 0.10, 0.20, 0.30],
    "vrp": [-0.06, -0.03, 0.0, 0.03],  # near-month ATM IV minus trailing RV, vol points
    "gex_pct": [0.05, 0.10, 0.20, 0.30],  # rolling percentile of net dealer GEX, mirror of iv_pct's grid
    "gex_mag_pct": [0.05, 0.10, 0.20, 0.30],  # rolling percentile of |net dealer GEX|, unsigned fallback
    "flip_dist": [-0.05, -0.03, -0.02, -0.01],  # raw signed distance above the gamma flip point
    "pc_oi_pct": [0.05, 0.10, 0.20, 0.30],  # rolling percentile of put/call OI ratio
    "pc_vol_pct": [0.05, 0.10, 0.20, 0.30],  # rolling percentile of put/call volume ratio
}
DEFAULT_THRESHOLD = {"iv_pct": 0.20, "vrp": 0.0, "gex_pct": 0.20, "gex_mag_pct": 0.20,
                    "flip_dist": -0.02, "pc_oi_pct": 0.20, "pc_vol_pct": 0.20}
MAX_HOLDS = [5, 10, 20, 30]
PROFIT_TARGETS = [0.5, 1.0, 2.0, None]  # multiple of debit gained; None = no target, hold to max_hold/expiry
RV_WINDOW = 20  # trailing trading days for the vrp signal's realized-vol side
MIN_CELL_N = 15


# ------------------------------------------------------------------ signal
@lru_cache(maxsize=4)
def _prepped_chain(underlying: str) -> pd.DataFrame:
    chain = ds.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    return chain[chain[ds.C_CLOSE] > 0]


@lru_cache(maxsize=4)
def iv_pct_series(underlying: str) -> pd.Series:
    """Rolling 252-trading-day percentile rank of near-month ATM call IV --
    same definition as backtest.Rules.iv_pct_min/max and iv_percentile_filter.py.
    'Cheap relative to its own history.'"""
    chain = _prepped_chain(underlying)
    iv_atm = bt._atm_iv_series(chain)
    return iv_atm.rolling(252, min_periods=60).rank(pct=True)


@lru_cache(maxsize=4)
def _trailing_rv_series(underlying: str, window: int = RV_WINDOW) -> pd.Series:
    etf = ds.load_underlying_ohlc(underlying)
    if etf is None:
        raise FileNotFoundError(
            f"no cached underlying OHLC for {underlying} -- run updater.py at least once")
    s = etf.set_index("date")["close"].sort_index()
    logret = np.log(s).diff()
    return logret.rolling(window, min_periods=15).std() * np.sqrt(252)


@lru_cache(maxsize=4)
def vrp_trailing_series(underlying: str, rv_window: int = RV_WINDOW) -> pd.Series:
    """Causal VRP proxy: near-month ATM call IV minus TRAILING realized vol.
    Deliberately not vrp_analysis.py's definition (that one looks forward,
    fine for measuring after the fact, unusable as a live trigger). 'Cheap
    relative to what's actually been happening,' not relative to IV's own
    history -- can fire even when IV's percentile rank isn't extreme, if
    realized vol has quietly outrun it."""
    iv_atm = bt._atm_iv_series(_prepped_chain(underlying))
    rv = _trailing_rv_series(underlying, rv_window)
    idx = iv_atm.index.union(rv.index)
    return (iv_atm.reindex(idx) - rv.reindex(idx)).dropna()


def _signal_series(underlying: str, signal: str) -> pd.Series:
    if signal == "iv_pct":
        return iv_pct_series(underlying)
    if signal == "vrp":
        return vrp_trailing_series(underlying)
    if signal == "gex_pct":
        return gx.gex_pct_series(underlying)
    if signal == "gex_mag_pct":
        return gx.gex_magnitude_pct_series(underlying)  # unsigned fallback, see gamma_exposure.py
    if signal == "flip_dist":
        return gf.flip_distance_series(underlying)  # 510050/510300 only, see gamma_flip.py
    if signal == "pc_oi_pct":
        return pc.pc_oi_ratio_pct_series(underlying)
    if signal == "pc_vol_pct":
        return pc.pc_vol_ratio_pct_series(underlying)
    raise ValueError(f"unknown signal {signal!r}")


# ------------------------------------------------------------------ engine
def run(underlying: str, signal: str, threshold: float, max_hold_days: int,
       profit_target: float | None, dte_min: int = 10, dte_max: int = 45,
       lag_signal_days: int = 0) -> dict:
    chain = _prepped_chain(underlying)
    sig = _signal_series(underlying, signal)

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

            mark = call_px + put_px  # value of a long straddle: always >= 0
            pnl = mark - open_pos["cost"]  # cost = debit paid, positive

            if reason is None:
                if profit_target is not None and pnl >= profit_target * open_pos["cost"]:
                    reason = "profit_target"
                elif days_held >= max_hold_days:
                    reason = "max_hold"

            if reason:
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "exit_date": dt,
                    "expiry": open_pos["expiry"], "exit_reason": reason,
                    "strike": open_pos["strike"], "entry_spot": open_pos["entry_spot"],
                    "exit_spot": spot, "signal_at_entry": open_pos["signal_at_entry"],
                    "debit": open_pos["cost"], "pnl": pnl,
                    "days_held": days_held,
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: only when the signal is below threshold ("cheap") and a
        # near-month expiry is in range. lag_signal_days=0 (default) uses
        # today's own signal value, same as always -- see the look-ahead
        # robustness note in the module docstring.
        if lag_signal_days and i < lag_signal_days:
            p = np.nan
        else:
            sig_date = pd.Timestamp(dates[i - lag_signal_days]) if lag_signal_days else dt
            p = sig.get(sig_date, np.nan)
        ok = (p == p) and p <= threshold
        if ok:
            expiries = sorted(day[ds.C_EXPIRY].unique())
            target = [e for e in expiries if dte_min <= (e - pd.Timestamp(dt)).days <= dte_max]
            if target:
                expiry = pd.Timestamp(target[0])
                strike = qs._atm_strike(day, expiry, spot)
                call_row = qs._leg_row(day, expiry, "C", strike) if strike is not None else None
                put_row = qs._leg_row(day, expiry, "P", strike) if strike is not None else None
                if call_row is not None and put_row is not None:
                    call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
                    open_pos = {
                        "entry_date": pd.Timestamp(dt), "entry_idx": i, "entry_spot": spot,
                        "expiry": expiry, "strike": strike,
                        "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
                        "last_call_px": call_px, "last_put_px": put_px,
                        "cost": call_px + put_px, "signal_at_entry": float(p),
                    }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ report
def _fmt_threshold(signal: str, threshold: float) -> str:
    return f"iv_pct<={threshold:.0%}" if signal == "iv_pct" else f"vrp<={threshold:+.1%}"


def report(res: dict, underlying: str, signal: str, threshold: float,
          max_hold_days: int, profit_target: float | None) -> None:
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        print("No trades.")
        return
    pnl_yuan = trades["pnl"] * ds.MULTIPLIER
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan
    avg_debit = trades["debit"].mean() * ds.MULTIPLIER
    total_pnl = pnl_yuan.sum()
    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    cagr_on_debit = (total_pnl / avg_debit) / span_years if span_years > 0 else np.nan

    pt_label = "none (hold to max_hold/expiry)" if profit_target is None else f"{profit_target:+.0%} of debit"
    print(f"{underlying}: cheap-vol long straddle "
          f"(signal={signal}, {_fmt_threshold(signal, threshold)}, "
          f"max_hold={max_hold_days}d, profit_target={pt_label})")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  avg_days_held={trades['days_held'].mean():.1f}  "
          f"mean signal at entry={trades['signal_at_entry'].mean():+.1%}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={(pnl_yuan/(trades['debit']*ds.MULTIPLIER)).mean():+.2%} (pnl/debit per trade)")
    print(f"  avg_win={pnl_yuan[pnl_yuan>0].mean():,.0f}  "
          f"avg_loss={pnl_yuan[pnl_yuan<=0].mean():,.0f} yuan")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg debit paid={avg_debit:,.0f} yuan  return_on_debit/yr~={cagr_on_debit:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    trades["year"] = trades["entry_date"].dt.year
    print("\n  By year:")
    for yr, g in trades.groupby("year"):
        print(f"    {yr}: n={len(g)}  pnl={g['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"iv_low_long_straddle_trades_{underlying}_{signal}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\n  Trades saved to {trades_out}")


def plot(res: dict, underlying: str, signal: str, out_path: Path) -> None:
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
    ax1.set_title(f"{underlying}: cheap-vol long straddle ({signal})", color=C_INK,
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
    r = trades["pnl"] / trades["debit"]
    return {"n": len(trades), "mean_r": r.mean()}


def sweep(underlying: str, signal: str, lag_signal_days: int = 0) -> None:
    thresholds = SIGNAL_MAXES[signal]
    rows = []
    for threshold, hold, pt in product(thresholds, MAX_HOLDS, PROFIT_TARGETS):
        res = run(underlying, signal=signal, threshold=threshold,
                 max_hold_days=hold, profit_target=pt, lag_signal_days=lag_signal_days)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        rows.append({"threshold": threshold, "hold": hold,
                     "profit_target": "none" if pt is None else pt,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    lag_suffix = f"_lag{lag_signal_days}" if lag_signal_days else ""
    print(f"{underlying} [signal={signal}]{lag_suffix}: sweep grid ({len(grid)} cells), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"iv_low_long_straddle_sweep_{underlying}_{signal}{lag_suffix}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"Full grid saved to {grid_out}")

    eligible = grid[grid["disc_n"] >= MIN_CELL_N]
    if eligible.empty:
        print("No cell has >= MIN_CELL_N discovery trades -- cannot freeze a pick.")
        return
    both_pos = eligible[(eligible["disc_mean_r"] > 0) & (eligible["hold_mean_r"] > 0)]
    both_neg = eligible[(eligible["disc_mean_r"] < 0) & (eligible["hold_mean_r"] < 0)]
    print(f"\nOf {len(eligible)} eligible cells: {len(both_pos)} agree-positive, "
          f"{len(both_neg)} agree-negative, {len(eligible) - len(both_pos) - len(both_neg)} mixed sign")

    best = eligible.loc[eligible["disc_mean_r"].idxmax()]
    print(f"\nFrozen pick (best discovery mean_r, n>={MIN_CELL_N}): "
          f"threshold={best['threshold']:+.2f}  max_hold={best['hold']:.0f}d  "
          f"profit_target={best['profit_target']}")
    print(f"  discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.2%}")
    print(f"  holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.2%}")


# ------------------------------------------------------------------ main
def main(underlying: str, signal: str, threshold: float, max_hold_days: int,
        profit_target: float | None, do_sweep: bool, lag_signal_days: int = 0) -> None:
    if do_sweep:
        sweep(underlying, signal, lag_signal_days=lag_signal_days)
        return
    res = run(underlying, signal=signal, threshold=threshold,
             max_hold_days=max_hold_days, profit_target=profit_target, lag_signal_days=lag_signal_days)
    report(res, underlying, signal, threshold, max_hold_days, profit_target)
    out_path = OUT_DIR / f"iv_low_long_straddle_{underlying}_{signal}.png"
    plot(res, underlying, signal, out_path)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510300")
    ap.add_argument("--signal",
                    choices=["iv_pct", "vrp", "gex_pct", "gex_mag_pct", "flip_dist",
                             "pc_oi_pct", "pc_vol_pct"],
                    default="vrp",
                    help="iv_pct: cheap relative to IV's own history. "
                         "vrp: cheap relative to trailing realized vol. "
                         "gex_pct/gex_mag_pct: dealer net gamma exposure low vs its own "
                         "rolling history. "
                         "flip_dist: spot's raw distance above the gamma flip point. "
                         "pc_oi_pct/pc_vol_pct: put/call OI or volume ratio low vs its "
                         "own rolling history.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="entry when signal <= this (default: see DEFAULT_THRESHOLD)")
    ap.add_argument("--max-hold", type=int, default=20, dest="max_hold_days",
                    help="max trading days to hold before closing regardless of expiry")
    ap.add_argument("--profit-target", type=float, default=1.0,
                    help="close when position value reaches (1+this)x the debit paid; "
                         "pass --no-profit-target for none")
    ap.add_argument("--no-profit-target", action="store_true",
                    help="disable the profit target (hold to max_hold/expiry only)")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search (threshold, max_hold, profit_target) on discovery, "
                         "score the frozen pick on holdout")
    ap.add_argument("--lag-signal-days", type=int, default=0, dest="lag_signal_days",
                    help="gate entry on the signal value from N trading days before the "
                         "entry date instead of the entry date's own same-day value "
                         "(default 0 = today's exact current behavior, unchanged). "
                         "See module docstring's look-ahead robustness note.")
    args = ap.parse_args()
    pt = None if args.no_profit_target else args.profit_target
    threshold = args.threshold
    if threshold is None:
        threshold = DEFAULT_THRESHOLD[args.signal]
    main(args.underlying, args.signal, threshold, args.max_hold_days, pt, args.sweep,
         lag_signal_days=args.lag_signal_days)
