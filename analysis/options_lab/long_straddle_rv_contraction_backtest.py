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
  gex_pct / gex_mag_pct: rolling percentile of net (or unsigned magnitude)
      dealer gamma exposure (gamma_exposure.py) -- low/negative percentile
      = dealers structurally short gamma, whose delta-hedging mechanically
      amplifies moves, the textbook backdrop for buying premium ahead of an
      expansion. Sign convention (dealers long calls, short puts) is an
      industry-standard heuristic UNVERIFIED for this market -- see
      gamma_exposure.py's module docstring. Summed across the FULL chain
      (every strike/expiry), not just near-month ATM like rv_pct/iv_pct/vrp.
      No flip_dist here -- gamma_flip.py only covers 510050/510300, tested
      via iv_low_long_straddle_backtest.py instead.
  pc_oi_pct / pc_vol_pct: rolling percentile of the put/call open-interest
      or volume ratio (put_call_ratio.py) -- a positioning/sentiment
      signal, no a priori direction assumption.

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

Look-ahead robustness note: default entry gating (lag_signal_days=0) uses the
SAME trading date's signal value while pricing the trade at that same date's
close -- not a true look-ahead (rv_pct/iv_pct/vrp are all rolling/trailing
functions of data through that date, never later dates), but a same-bar
execution idealization applied less strictly than this repo's "rich/extreme"
trigger family (calendar_spread_entry_timing_backtest.py,
skew_vrp_combo_short_straddle_backtest.py), which gate on the PRIOR trading
date's value. --lag-signal-days N reruns under that stricter convention as a
robustness check on the frozen picks feeding the live VOL_SIGNAL_ALERTS
(alerts_core.py never calls run() directly, only _signal_series(), so this
flag cannot affect the live alert regardless of default).

Usage:
    python long_straddle_rv_contraction_backtest.py --underlying 510500 --signal iv_pct
    python long_straddle_rv_contraction_backtest.py --underlying all --signal vrp --sweep
    python long_straddle_rv_contraction_backtest.py --underlying 510500 --signal rv_pct --lag-signal-days 1
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

import cffex_margin as cm
import consolidated_store as cs
import data_store as ds
import gamma_exposure as gx
import put_call_ratio as pc
import quarterly_atm_short_straddle as qs

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = pd.Timestamp("2024-01-01")  # same split as condor_delta_investigation_summary.md

ETF_UNDERLYINGS = {
    "510500": "500ETF", "588000": "科创50ETF", "588080": "科创板50ETF",
    "159915": "创业板ETF", "159901": "深证100ETF",
}
# CFFEX index options (added 2026-08-17). They ride the same consolidated_store
# chain and the same engine; what differs is the contract multiplier (100
# yuan/index point vs. the ETF options' 10000) and which signals are
# computable -- see _multiplier() and CFFEX_UNSUPPORTED_SIGNALS below.
CFFEX_UNDERLYINGS = {
    "000300": "沪深300 (IO)", "000852": "中证1000 (MO)", "000016": "上证50 (HO)",
}
UNDERLYINGS = {**ETF_UNDERLYINGS, **CFFEX_UNDERLYINGS}

# ASCII names for matplotlib PNG titles: this environment's matplotlib has no
# CJK-capable font (checked: 21 families, none with CJK coverage), so a
# Chinese name in a title renders as tofu boxes. Only the CFFEX names are
# mapped here -- the Chinese ETF names (创业板ETF etc.) have the same problem,
# but that is pre-existing behaviour in the ETF charts and fixing it would
# change output this change has no business touching.
PLOT_NAMES = {"000300": "IO CSI300", "000852": "MO CSI1000", "000016": "HO SSE50"}

# Signals that cannot be computed for a CFFEX index option, with the reason.
# These are BLOCKED rather than left to fail on their own because both fail
# SILENTLY on this dataset, which is far worse than an exception:
#   gex_pct/gex_mag_pct -- ds.C_GAMMA is 0% populated for IO/MO/HO, and
#     gamma_exposure.net_gex_series does chain[ds.C_GAMMA].fillna(0), so it
#     returns an all-zero series whose rolling percentile rank is a CONSTANT
#     ~0.508. A sweep over it produces a full grid of plausible-looking,
#     entirely meaningless numbers.
# (flip_dist is not in this module's dispatch at all; gamma_flip.py raises
# on its own for anything outside 510050/510300.)
CFFEX_UNSUPPORTED_SIGNALS = {
    "gex_pct": "ds.C_GAMMA is 0% populated for CFFEX index options -- "
               "net GEX would be identically zero and its percentile a constant",
    "gex_mag_pct": "same as gex_pct: no per-contract gamma in this dataset",
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
    "gex_pct": [0.05, 0.10, 0.20, 0.30],  # rolling percentile of net dealer GEX, mirror of rv_pct's grid
    "gex_mag_pct": [0.05, 0.10, 0.20, 0.30],  # rolling percentile of |net dealer GEX|, unsigned fallback
    "pc_oi_pct": [0.05, 0.10, 0.20, 0.30],  # rolling percentile of put/call OI ratio
    "pc_vol_pct": [0.05, 0.10, 0.20, 0.30],  # rolling percentile of put/call volume ratio
}
MAX_HOLDS = [5, 10, 20, 30]
PROFIT_TARGETS = [0.5, 1.0, 2.0, None]  # multiple of debit gained; None = hold to max_hold/expiry
RV_WINDOW = 20  # trailing trading days for the realized-vol side of rv_pct/vrp
MIN_CELL_N = 15
DEFAULT_THRESHOLD = {"rv_pct": 0.20, "iv_pct": 0.20, "vrp": 0.0, "gex_pct": 0.20,
                    "gex_mag_pct": 0.20, "pc_oi_pct": 0.20, "pc_vol_pct": 0.20}


# ------------------------------------------------------------------ signal
@lru_cache(maxsize=8)
def _prepped_chain(underlying: str) -> pd.DataFrame:
    chain = cs.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    return chain[chain[ds.C_CLOSE] > 0]


@lru_cache(maxsize=8)
def _trailing_rv_series(underlying: str, window: int = RV_WINDOW) -> pd.Series:
    etf = ds.load_underlying_ohlc(underlying)
    if etf is None:
        raise FileNotFoundError(
            f"no cached underlying OHLC for {underlying} -- run updater.py (ETFs) "
            f"or index_price_updater.py (000300/000852/000016) at least once")
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


def _multiplier(underlying: str) -> int:
    """Contract multiplier in yuan per unit of underlying price. ETF options
    are 10000 (ds.MULTIPLIER); CFFEX index options are 100 yuan per index
    point (cffex_margin.MULTIPLIER, same for IO/MO/HO). Return-on-debit is a
    ratio and so is unaffected, but every yuan-denominated figure in report()
    and plot() is wrong by 100x if this is not dispatched."""
    return cm.MULTIPLIER if underlying in CFFEX_UNDERLYINGS else ds.MULTIPLIER


def _check_signal_supported(underlying: str, signal: str) -> None:
    """Raise on a (underlying, signal) pair that would silently produce a
    degenerate series -- see CFFEX_UNSUPPORTED_SIGNALS."""
    if underlying in CFFEX_UNDERLYINGS and signal in CFFEX_UNSUPPORTED_SIGNALS:
        raise ValueError(
            f"signal {signal!r} is not available for CFFEX index option "
            f"{underlying}: {CFFEX_UNSUPPORTED_SIGNALS[signal]}")


def _signal_series(underlying: str, signal: str) -> pd.Series:
    _check_signal_supported(underlying, signal)
    if signal == "rv_pct":
        return rv_pct_series(underlying)
    if signal == "iv_pct":
        return iv_pct_series(underlying)
    if signal == "vrp":
        return vrp_trailing_series(underlying)
    if signal == "gex_pct":
        return gx.gex_pct_series(underlying)
    if signal == "gex_mag_pct":
        return gx.gex_magnitude_pct_series(underlying)  # unsigned fallback, see gamma_exposure.py
    if signal == "pc_oi_pct":
        return pc.pc_oi_ratio_pct_series(underlying)
    if signal == "pc_vol_pct":
        return pc.pc_vol_ratio_pct_series(underlying)
    raise ValueError(f"unknown signal {signal!r}")


# ------------------------------------------------------------------ engine
def run(underlying: str, signal: str, threshold: float, max_hold_days: int,
       profit_target: float | None, dte_min: int = 10, dte_max: int = 45,
       lag_signal_days: int = 0, slippage: float = 0.0) -> dict:
    """slippage (default 0.0, this lab's no-fees/no-slippage convention,
    unchanged): adverse half-spread per leg, in underlying-price units
    (index points for CFFEX, yuan for ETF options), charged on BOTH legs at
    entry and at a close-based exit (profit_target/max_hold). NOT charged at
    expiry -- cash settlement has no spread to cross, same convention
    expiry_week_atm_straddle_backtest.py uses. profit_target/max_hold
    triggers are still evaluated on the raw (unslipped) mark -- slippage
    only affects what is actually realized once an exit fires, not whether
    one fires, so timing stays comparable across the slippage grid."""
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
                # Realized proceeds net of exit friction. At slippage=0 both
                # branches reduce to exactly `mark` (call_px/put_px are real
                # prices, always >= 0, so max(x-0,0) == x) -- byte-identical
                # to the pre-slippage behavior.
                if reason == "expiry":
                    proceeds = mark
                else:
                    proceeds = max(call_px - slippage, 0.0) + max(put_px - slippage, 0.0)
                realized_pnl = proceeds - open_pos["cost"]
                realized += realized_pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "exit_date": dt,
                    "expiry": open_pos["expiry"], "exit_reason": reason,
                    "strike": open_pos["strike"], "entry_spot": open_pos["entry_spot"],
                    "exit_spot": spot, "signal_at_entry": open_pos["signal_at_entry"],
                    "debit": open_pos["cost"], "pnl": realized_pnl,
                    "days_held": days_held,
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: only when the signal is below threshold ("cheap"/"quiet")
        # and a near-month expiry is in range. lag_signal_days=0 (default) uses
        # today's own signal value, same as always -- see the look-ahead
        # robustness note in the module docstring for why lag_signal_days>0
        # exists (mirrors the dates[i-N] lookup pattern used for "rich/extreme"
        # triggers elsewhere in this repo, e.g. calendar_spread_entry_timing_
        # backtest.py, skew_vrp_combo_short_straddle_backtest.py).
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
                        # entry friction on both legs (buying at the offer);
                        # 0 at slippage=0, unchanged from before this param existed
                        "cost": call_px + put_px + 2 * slippage, "signal_at_entry": float(p),
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
    mult = _multiplier(underlying)
    if trades.empty:
        print(f"{underlying} ({name}) [{signal}]: no trades.")
        return
    pnl_yuan = trades["pnl"] * mult
    eq = equity.set_index("date")["equity"] * mult
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan
    avg_debit = trades["debit"].mean() * mult
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
          f"mean_r={(pnl_yuan/(trades['debit']*mult)).mean():+.2%} (pnl/debit per trade)")
    print(f"  avg_win={pnl_yuan[pnl_yuan>0].mean():,.0f}  "
          f"avg_loss={pnl_yuan[pnl_yuan<=0].mean():,.0f} yuan")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg debit paid={avg_debit:,.0f} yuan  return_on_debit/yr~={cagr_on_debit:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    trades["year"] = trades["entry_date"].dt.year
    print("  by year:")
    for yr, g in trades.groupby("year"):
        print(f"    {yr}: n={len(g)}  pnl={g['pnl'].sum()*mult:>10,.0f}  win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"long_straddle_trades_{underlying}_{signal}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"  Trades saved to {trades_out}\n")


def plot(res: dict, underlying: str, signal: str, out_path: Path) -> None:
    equity = res["equity"]
    name = PLOT_NAMES.get(underlying, UNDERLYINGS.get(underlying, underlying))
    mult = _multiplier(underlying)
    eq = equity.set_index("date")["equity"] * mult
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
    name = UNDERLYINGS.get(underlying, underlying)
    lag_suffix = f"_lag{lag_signal_days}" if lag_signal_days else ""
    print(f"{underlying} ({name}) [{signal}]{lag_suffix}: sweep grid ({len(grid)} cells), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"long_straddle_sweep_{underlying}_{signal}{lag_suffix}.csv"
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
        profit_target: float | None, do_sweep: bool, lag_signal_days: int = 0) -> None:
    targets = list(UNDERLYINGS) if underlying == "all" else [underlying]
    for u in targets:
        # "all" means "this signal, across every underlying that can compute
        # it" -- an underlying the signal is unavailable for is skipped with
        # the reason, not allowed to abort the whole batch. An explicitly
        # named underlying still raises, so a typo is never silently ignored.
        if underlying == "all" and u in CFFEX_UNDERLYINGS and signal in CFFEX_UNSUPPORTED_SIGNALS:
            print(f"{u} ({UNDERLYINGS[u]}) [{signal}]: skipped -- "
                  f"{CFFEX_UNSUPPORTED_SIGNALS[signal]}\n")
            continue
        if do_sweep:
            sweep(u, signal, lag_signal_days=lag_signal_days)
            continue
        res = run(u, signal=signal, threshold=threshold, max_hold_days=max_hold_days,
                 profit_target=profit_target, lag_signal_days=lag_signal_days)
        report(res, u, signal, threshold, max_hold_days, profit_target)
        out_path = OUT_DIR / f"long_straddle_{u}_{signal}.png"
        plot(res, u, signal, out_path)
        print(f"  Chart saved to {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="all",
                    choices=list(UNDERLYINGS) + ["all"])
    ap.add_argument("--signal",
                    choices=["rv_pct", "iv_pct", "vrp", "gex_pct", "gex_mag_pct",
                             "pc_oi_pct", "pc_vol_pct"],
                    default="rv_pct",
                    help="rv_pct: trailing RV cheap vs its own history. "
                         "iv_pct: ATM IV (Black-Scholes-inverted) cheap vs its own history. "
                         "vrp: ATM IV minus trailing RV. "
                         "gex_pct/gex_mag_pct: dealer net gamma exposure LOW vs its own "
                         "rolling history -- short-gamma dealers amplify vol (see "
                         "gamma_exposure.py's module docstring for the unverified sign "
                         "convention). "
                         "pc_oi_pct/pc_vol_pct: put/call OI or volume ratio LOW vs its "
                         "own rolling history.")
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
    ap.add_argument("--lag-signal-days", type=int, default=0, dest="lag_signal_days",
                    help="gate entry on the signal value from N trading days before the "
                         "entry date instead of the entry date's own same-day value "
                         "(default 0 = today's exact current behavior, unchanged). "
                         "See module docstring's look-ahead robustness note.")
    args = ap.parse_args()
    pt = None if args.no_profit_target else args.profit_target
    threshold = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD[args.signal]
    main(args.underlying, args.signal, threshold, args.max_hold_days, pt, args.sweep,
         lag_signal_days=args.lag_signal_days)
