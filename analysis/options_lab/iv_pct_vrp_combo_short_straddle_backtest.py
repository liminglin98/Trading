"""Naked short straddle gated on BOTH conditions at once: yesterday's iv_pct
(vol_rich_short_straddle_backtest.py's "raw ATM IV high vs its own rolling
history" trigger) AND yesterday's vrp (that same script's "ATM IV rich vs
trailing realized vol" trigger) >= threshold, simultaneously. Both signals
passed discovery/holdout independently but only overlap on ~7% of entry
dates (see project_short_straddle_vol_signals memory) -- their UNION was
suggested as a way to get more trades; this tests the opposite motivation:
does requiring BOTH at once (a stricter, higher-conviction filter) produce
cleaner/safer entries than either alone, at the cost of fewer trades? Useful
specifically as a caution lever in choppy/pre-event markets, since it should
avoid entries where only one of the two rich-vol conditions is true (e.g. IV
elevated vs its own history but not actually rich vs realized vol yet --
plausibly the "catching the front edge of a vol event" case).

Same engine as both parents: naked ATM straddle, margin.py-sized, exits via
2x-credit stop-loss / max_hold / expiry. Only 510050 and 510300 supported
(real exchange IV -- both underlying signals need it).

Follows this repo's discovery/holdout discipline: discovery < 2024-01-01,
holdout >= 2024-01-01, split fixed once.

Usage:
    python iv_pct_vrp_combo_short_straddle_backtest.py --underlying 510050
    python iv_pct_vrp_combo_short_straddle_backtest.py --underlying 510050 --sweep
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qs
import iv_low_long_straddle_backtest as il

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = pd.Timestamp("2024-01-01")
UNDERLYINGS = {"510050": "50ETF", "510300": "300ETF"}

IV_PCT_THRESHOLDS = [0.70, 0.80, 0.90, 0.95]
VRP_THRESHOLDS = [0.0, 0.03, 0.06]
MAX_HOLDS = [10, 20]
STOP_LOSS_MULT = 2.0
MIN_CELL_N = 15


def run(underlying: str, iv_pct_threshold: float, vrp_threshold: float,
       max_hold_days: int, stop_loss_mult: float | None = STOP_LOSS_MULT,
       dte_min: int = 10, dte_max: int = 45) -> dict:
    chain = ds.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    chain = chain[chain[ds.C_CLOSE] > 0]

    iv_pct = il.iv_pct_series(underlying)
    vrp = il.vrp_trailing_series(underlying)

    dates = sorted(chain[ds.C_DATE].unique())
    by_date = dict(tuple(chain.groupby(ds.C_DATE)))

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

            expired = dt >= open_pos["expiry"]
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
                    "exit_spot": spot, "iv_pct_at_entry": open_pos["iv_pct_at_entry"],
                    "vrp_at_entry": open_pos["vrp_at_entry"],
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": days_held, "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: yesterday's iv_pct AND yesterday's vrp both >= threshold
        prev_dt = pd.Timestamp(dates[i - 1]) if i > 0 else None
        iv_pct_prev = iv_pct.get(prev_dt, np.nan) if prev_dt is not None else np.nan
        vrp_prev = vrp.get(prev_dt, np.nan) if prev_dt is not None else np.nan
        ok_iv_pct = (iv_pct_prev == iv_pct_prev) and iv_pct_prev >= iv_pct_threshold
        ok_vrp = (vrp_prev == vrp_prev) and vrp_prev >= vrp_threshold
        if ok_iv_pct and ok_vrp:
            expiries = sorted(day[ds.C_EXPIRY].unique())
            target = [e for e in expiries if dte_min <= (e - dt).days <= dte_max]
            if target:
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
                        "entry_date": dt, "entry_idx": i, "entry_spot": spot,
                        "expiry": expiry, "strike": strike,
                        "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
                        "last_call_px": call_px, "last_put_px": put_px,
                        "cost": -(call_px + put_px), "margin": margin_yuan,
                        "iv_pct_at_entry": float(iv_pct_prev),
                        "vrp_at_entry": float(vrp_prev),
                    }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, iv_pct_threshold: float,
          vrp_threshold: float, max_hold_days: int) -> None:
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

    print(f"{underlying}: iv_pct+vrp combo short straddle "
          f"(iv_pct>={iv_pct_threshold:.2f}, vrp>={vrp_threshold:+.0%}, "
          f"max_hold={max_hold_days}d)")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  avg_days_held={trades['days_held'].mean():.1f}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={(pnl_yuan/(trades['margin_yuan'])).mean():+.2%} (pnl/margin per trade)")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin (naked)={avg_margin:,.0f} yuan  return_on_margin/yr~={cagr_on_margin:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    trades["year"] = trades["entry_date"].dt.year
    print("\n  By year:")
    for yr, g in trades.groupby("year"):
        print(f"    {yr}: n={len(g)}  pnl={g['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"iv_pct_vrp_combo_short_straddle_trades_{underlying}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\n  Trades saved to {trades_out}")


# ------------------------------------------------------------------ sweep
def _cell_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "mean_r": np.nan}
    r = trades["pnl"] * ds.MULTIPLIER / trades["margin_yuan"]
    return {"n": len(trades), "mean_r": r.mean()}


def sweep(underlying: str) -> None:
    rows = []
    for iv_thr, vrp_thr, hold in product(IV_PCT_THRESHOLDS, VRP_THRESHOLDS, MAX_HOLDS):
        res = run(underlying, iv_pct_threshold=iv_thr, vrp_threshold=vrp_thr, max_hold_days=hold)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= SPLIT_DATE] if not trades.empty else trades
        d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
        rows.append({"iv_pct_thr": iv_thr, "vrp_thr": vrp_thr, "hold": hold,
                     "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                     "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    print(f"{underlying}: sweep grid ({len(grid)} cells), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")
    print(grid.to_string(index=False, formatters={
        "disc_mean_r": lambda v: f"{v:+.2%}" if v == v else "n/a",
        "hold_mean_r": lambda v: f"{v:+.2%}" if v == v else "n/a"}))

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"iv_pct_vrp_combo_sweep_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"\nFull grid saved to {grid_out}")

    eligible = grid[grid["disc_n"] >= MIN_CELL_N]
    if eligible.empty:
        print(f"\nNo cell has >= MIN_CELL_N discovery trades -- cannot freeze a pick "
              f"(max disc_n in grid = {grid['disc_n'].max():.0f}).")
        return
    both_pos = eligible[(eligible["disc_mean_r"] > 0) & (eligible["hold_mean_r"] > 0)]
    both_neg = eligible[(eligible["disc_mean_r"] < 0) & (eligible["hold_mean_r"] < 0)]
    print(f"\nOf {len(eligible)} eligible cells: {len(both_pos)} agree-positive, "
          f"{len(both_neg)} agree-negative, {len(eligible) - len(both_pos) - len(both_neg)} mixed sign")

    best = eligible.loc[eligible["disc_mean_r"].idxmax()]
    print(f"\nFrozen pick (best discovery mean_r, n>={MIN_CELL_N}): "
          f"iv_pct>={best['iv_pct_thr']:.2f}  vrp>={best['vrp_thr']:+.0%}  "
          f"max_hold={best['hold']:.0f}d")
    print(f"  discovery: n={best['disc_n']:.0f}  mean_r={best['disc_mean_r']:+.2%}")
    print(f"  holdout:   n={best['hold_n']:.0f}  mean_r={best['hold_mean_r']:+.2%}")


# ------------------------------------------------------------------ main
def main(underlying: str, iv_pct_threshold: float, vrp_threshold: float,
        max_hold_days: int, do_sweep: bool) -> None:
    if do_sweep:
        sweep(underlying)
        return
    res = run(underlying, iv_pct_threshold=iv_pct_threshold, vrp_threshold=vrp_threshold,
             max_hold_days=max_hold_days)
    report(res, underlying, iv_pct_threshold, vrp_threshold, max_hold_days)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050", choices=list(UNDERLYINGS))
    ap.add_argument("--iv-pct-threshold", type=float, default=0.80, dest="iv_pct_threshold")
    ap.add_argument("--vrp-threshold", type=float, default=0.03, dest="vrp_threshold")
    ap.add_argument("--max-hold", type=int, default=20, dest="max_hold_days")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    main(args.underlying, args.iv_pct_threshold, args.vrp_threshold,
        args.max_hold_days, args.sweep)
