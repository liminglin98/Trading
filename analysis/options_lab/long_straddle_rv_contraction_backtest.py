"""Long straddle on 500ETF / STAR50 ETF(x2) / ChiNext ETF, triggered by a
"cheap/quiet now, bet on expansion" signal -- three alternatives, selected
via --signal:

  rv_pct (default): trailing realized vol is at a low percentile of its own
      252-day history. Needs only the underlying's own price history.
  iv_pct: near-month ATM call IV is at a low percentile of its own 252-day
      history -- same definition as iv_low_long_straddle_backtest.py's
      --signal iv_pct, but IV is backed out via Black-Scholes inversion
      (bs_iv.py, consolidated_store.atm_iv_series) since this dataset has
      Greeks but no IV column (see consolidated_store.py's module docstring).
  vrp: near-month ATM call IV minus trailing realized vol -- "cheap relative
      to what's actually been happening," the same vrp definition
      iv_low_long_straddle_backtest.py uses (deliberately NOT vrp_analysis.py's
      forward-looking definition, which would be lookahead bias as a trigger).

Underlying spot (needed for ATM strike selection, IV inversion, and expiry
settlement) is fetched via akshare fund_etf_hist_sina and cached at
Data/options/etf_daily_{code}.csv -- also missing from the consolidated
dataset, which has no spot price column at all.

Exit: same three rules as iv_low_long_straddle_backtest.py -- profit target
(multiple of debit), max holding period, or expiry (settled at intrinsic
using the spot price on the expiry date). No stop-loss -- the debit already
bounds the loss.

Follows this repo's discovery/holdout discipline (condor_delta_investigation_summary.md):
discovery < 2024-01-01 <= holdout, split fixed once. Flag explicitly: 588000/
588080 only start 2023-06-05, so their discovery window is ~7 months -- treat
any "win" there as thin-sample until more history accumulates, don't read a
strong ratio as confidence.

Usage:
    python long_straddle_rv_contraction_backtest.py --underlying 510500 --signal iv_pct
    python long_straddle_rv_contraction_backtest.py --underlying all --signal vrp --sweep
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

import consolidated_store as cs
import data_store as ds
import quarterly_atm_short_straddle as qs

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = pd.Timestamp("2024-01-01")  # same split as condor_delta_investigation_summary.md

UNDERLYINGS = {
    "510500": "500ETF", "588000": "科创50ETF", "588080": "科创板50ETF",
    "159915": "创业板ETF", "159901": "深证100ETF",
}

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"

# sweep grids (discovery-only search, frozen pick scored once on holdout)
SIGNAL_MAXES = {
    "rv_pct": [0.05, 0.10, 0.20, 0.30],
    "iv_pct": [0.05, 0.10, 0.20, 0.30],
    "vrp": [-0.06, -0.03, 0.0, 0.03],  # ATM IV minus trailing RV, vol points
}
MAX_HOLDS = [5, 10, 20, 30]
PROFIT_TARGETS = [0.5, 1.0, 2.0, None]  # multiple of debit gained; None = hold to max_hold/expiry
RV_WINDOW = 20  # trailing trading days for the realized-vol side of rv_pct/vrp
MIN_CELL_N = 15
DEFAULT_THRESHOLD = {"rv_pct": 0.20, "iv_pct": 0.20, "vrp": 0.0}


# ------------------------------------------------------------------ signal
@lru_cache(maxsize=8)
def _prepped_chain(underlying: str) -> pd.DataFrame:
    chain = cs.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    return chain[chain[ds.C_CLOSE] > 0]


@lru_cache(maxsize=8)
def _trailing_rv_series(underlying: str, window: int = RV_WINDOW) -> pd.Series:
    etf = ds.load_etf_ohlc(underlying)
    if etf is None:
        raise FileNotFoundError(
            f"no cached ETF OHLC for {underlying} -- see consolidated_store.py")
    s = etf.set_index("date")["close"].sort_index()
    logret = np.log(s).diff()
    return logret.rolling(window, min_periods=15).std() * np.sqrt(252)


@lru_cache(maxsize=8)
def rv_pct_series(underlying: str, window: int = RV_WINDOW) -> pd.Series:
    """Rolling 252-trading-day percentile rank of trailing `window`-day
    realized vol (annualized) of the underlying's own close price. 'Quiet
    relative to its own history' -- the RV analog of iv_pct_series in
    iv_low_long_straddle_backtest.py."""
    return _trailing_rv_series(underlying, window).rolling(252, min_periods=60).rank(pct=True)


@lru_cache(maxsize=8)
def iv_pct_series(underlying: str) -> pd.Series:
    """Rolling 252-trading-day percentile rank of near-month ATM call IV,
    backed out via Black-Scholes inversion (consolidated_store.atm_iv_series)."""
    iv_atm = cs.atm_iv_series(underlying)
    return iv_atm.rolling(252, min_periods=60).rank(pct=True)


@lru_cache(maxsize=8)
def vrp_trailing_series(underlying: str, rv_window: int = RV_WINDOW) -> pd.Series:
    """Causal VRP proxy: near-month ATM call IV minus TRAILING realized vol
    -- same definition iv_low_long_straddle_backtest.py uses, deliberately
    not vrp_analysis.py's forward-looking one (lookahead bias as a trigger)."""
    iv_atm = cs.atm_iv_series(underlying)
    rv = _trailing_rv_series(underlying, rv_window)
    idx = iv_atm.index.union(rv.index)
    return (iv_atm.reindex(idx) - rv.reindex(idx)).dropna()


def _signal_series(underlying: str, signal: str) -> pd.Series:
    if signal == "rv_pct":
        return rv_pct_series(underlying)
    if signal == "iv_pct":
        return iv_pct_series(underlying)
    if signal == "vrp":
        return vrp_trailing_series(underlying)
    raise ValueError(f"unknown signal {signal!r}")


# ------------------------------------------------------------------ engine
def run(underlying: str, signal: str, threshold: float, max_hold_days: int,
       profit_target: float | None, dte_min: int = 10, dte_max: int = 45) -> dict:
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

        # ---- entry: only when the signal is below threshold ("cheap"/"quiet")
        # and a near-month expiry is in range
        p = sig.get(dt, np.nan)
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
    return f"{signal}<={threshold:.0%}" if signal != "vrp" else f"vrp<={threshold:+.1%}"


def report(res: dict, underlying: str, signal: str, threshold: float,
          max_hold_days: int, profit_target: float | None) -> None:
    trades, equity = res["trades"], res["equity"]
    name = UNDERLYINGS.get(underlying, underlying)
    if trades.empty:
        print(f"{underlying} ({name}) [{signal}]: no trades.")
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
    print(f"{underlying} ({name}) [{signal}]: long straddle "
          f"({_fmt_threshold(signal, threshold)}, max_hold={max_hold_days}d, profit_target={pt_label})")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  avg_days_held={trades['days_held'].mean():.1f}  "
          f"mean signal at entry={trades['signal_at_entry'].mean():.1%}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={(pnl_yuan/(trades['debit']*ds.MULTIPLIER)).mean():+.2%} (pnl/debit per trade)")
    print(f"  avg_win={pnl_yuan[pnl_yuan>0].mean():,.0f}  "
          f"avg_loss={pnl_yuan[pnl_yuan<=0].mean():,.0f} yuan")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg debit paid={avg_debit:,.0f} yuan  return_on_debit/yr~={cagr_on_debit:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    trades["year"] = trades["entry_date"].dt.year
    print("  by year:")
    for yr, g in trades.groupby("year"):
        print(f"    {yr}: n={len(g)}  pnl={g['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"long_straddle_trades_{underlying}_{signal}.csv"
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
    ax1.set_title(f"{underlying} ({name}): long straddle [{signal}]", color=C_INK,
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


def sweep(underlying: str, signal: str) -> None:
    thresholds = SIGNAL_MAXES[signal]
    rows = []
    for threshold, hold, pt in product(thresholds, MAX_HOLDS, PROFIT_TARGETS):
        res = run(underlying, signal=signal, threshold=threshold,
                 max_hold_days=hold, profit_target=pt)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        rows.append({"threshold": threshold, "hold": hold,
                     "profit_target": "none" if pt is None else pt,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    name = UNDERLYINGS.get(underlying, underlying)
    print(f"{underlying} ({name}) [{signal}]: sweep grid ({len(grid)} cells), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"long_straddle_sweep_{underlying}_{signal}.csv"
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
          f"threshold={best['threshold']:.2f}  max_hold={best['hold']:.0f}d  "
          f"profit_target={best['profit_target']}")
    print(f"  discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.2%}")
    print(f"  holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.2%}\n")


# ------------------------------------------------------------------ main
def main(underlying: str, signal: str, threshold: float, max_hold_days: int,
        profit_target: float | None, do_sweep: bool) -> None:
    targets = list(UNDERLYINGS) if underlying == "all" else [underlying]
    for u in targets:
        if do_sweep:
            sweep(u, signal)
            continue
        res = run(u, signal=signal, threshold=threshold, max_hold_days=max_hold_days,
                 profit_target=profit_target)
        report(res, u, signal, threshold, max_hold_days, profit_target)
        out_path = OUT_DIR / f"long_straddle_{u}_{signal}.png"
        plot(res, u, signal, out_path)
        print(f"  Chart saved to {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="all",
                    choices=list(UNDERLYINGS) + ["all"])
    ap.add_argument("--signal", choices=["rv_pct", "iv_pct", "vrp"], default="rv_pct",
                    help="rv_pct: trailing RV cheap vs its own history. "
                         "iv_pct: ATM IV (Black-Scholes-inverted) cheap vs its own history. "
                         "vrp: ATM IV minus trailing RV.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="entry when signal <= this (default: 0.20 for rv_pct/iv_pct, "
                         "0.0 for vrp)")
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
    args = ap.parse_args()
    pt = None if args.no_profit_target else args.profit_target
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD[args.signal]
    main(args.underlying, args.signal, threshold, args.max_hold_days, pt, args.sweep)
