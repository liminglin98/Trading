"""Tests whether allowing multiple CONCURRENT short straddles -- opening a new
one every day the vrp/iv_pct signal is active, instead of vol_rich_short_
straddle_backtest.py's sequential single-position rule (wait for the current
trade to exit before a new one can enter) -- changes results by more than
noise. Reuses the same signal series, expiry/strike selection, and margin
sizing (each position priced/margined independently, no portfolio-margin
netting modeled) -- only the "one position at a time" gate is removed.

Read-only research script, does not touch the live single-position modules
or app.py.

Usage:
    python vol_rich_short_straddle_stacked_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # archived: options_lab/ modules live one level up

import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qs
import vol_rich_short_straddle_backtest as vr

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ engine
def run_stacked(underlying: str, signal: str, threshold: float, max_hold_days: int,
                stop_loss_mult: float | None = vr.STOP_LOSS_MULT, dte_min: int = 10,
                dte_max: int = 45, dte_select: str = "nearest",
                max_concurrent: int | None = None) -> dict:
    chain = vr._prepped_chain(underlying)
    dates = sorted(chain[ds.C_DATE].unique())
    by_date = dict(tuple(chain.groupby(ds.C_DATE)))
    sig = vr._signal_series(underlying, signal)

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_positions: list[dict] = []
    realized = 0.0

    for i, dt in enumerate(dates):
        dt = pd.Timestamp(dt)
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        open_pnl_today = 0.0
        still_open = []
        for pos in open_positions:
            call = day[day[ds.C_CODE] == pos["call_code"]]
            put = day[day[ds.C_CODE] == pos["put_code"]]
            call_px = float(call[ds.C_CLOSE].iloc[0]) if not call.empty else pos["last_call_px"]
            put_px = float(put[ds.C_CLOSE].iloc[0]) if not put.empty else pos["last_put_px"]
            pos["last_call_px"], pos["last_put_px"] = call_px, put_px

            expired = dt >= pos["expiry"]
            days_held = i - pos["entry_idx"]
            reason = None
            if expired:
                call_px = qs._intrinsic("C", pos["strike"], spot)
                put_px = qs._intrinsic("P", pos["strike"], spot)
                reason = "expiry"

            mark = -(call_px + put_px)
            pnl = mark - pos["cost"]
            basis = abs(pos["cost"])

            if reason is None:
                if stop_loss_mult is not None and pnl <= -stop_loss_mult * basis:
                    reason = "stop_loss"
                elif days_held >= max_hold_days:
                    reason = "max_hold"

            if reason:
                realized += pnl
                trades.append({
                    "entry_date": pos["entry_date"], "exit_date": dt,
                    "expiry": pos["expiry"], "exit_reason": reason,
                    "strike": pos["strike"], "entry_spot": pos["entry_spot"],
                    "exit_spot": spot, "signal_at_entry": pos["signal_at_entry"],
                    "credit": -pos["cost"], "pnl": pnl,
                    "days_held": days_held, "margin_yuan": pos["margin"],
                })
            else:
                open_pnl_today += pnl
                still_open.append(pos)
        open_positions = still_open

        # ---- entry: yesterday's signal >= threshold -- opens a NEW position
        # regardless of how many are already open (the thing being tested;
        # the only difference vs. vr.run()'s sequential gate)
        prev_dt = pd.Timestamp(dates[i - 1]) if i > 0 else None
        p = sig.get(prev_dt, np.nan) if prev_dt is not None else np.nan
        ok = (p == p) and p >= threshold
        if ok and (max_concurrent is None or len(open_positions) < max_concurrent):
            expiries = sorted(day[ds.C_EXPIRY].unique())
            target = [e for e in expiries if dte_min <= (e - dt).days <= dte_max]
            if target:
                picked = vr._select_expiry(day, target, spot, dte_select)
                if picked is not None:
                    expiry, strike, call_row, put_row = picked
                    call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
                    legs_df = pd.DataFrame([
                        {"expiry": expiry, "cp": "C", "strike": strike, "direction": -1,
                         "qty": 1, "entry_price": call_px},
                        {"expiry": expiry, "cp": "P", "strike": strike, "direction": -1,
                         "qty": 1, "entry_price": put_px},
                    ])
                    margin_yuan, _ = mg.compute_deposit(legs_df, spot)
                    open_positions.append({
                        "entry_date": dt, "entry_idx": i, "entry_spot": spot,
                        "expiry": expiry, "strike": strike,
                        "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
                        "last_call_px": call_px, "last_put_px": put_px,
                        "cost": -(call_px + put_px), "margin": margin_yuan,
                        "signal_at_entry": float(p),
                    })

        total_margin = sum(pos["margin"] for pos in open_positions)
        equity_rows.append({"date": dt, "equity": realized + open_pnl_today,
                            "open_pnl": open_pnl_today, "spot": spot,
                            "n_open": len(open_positions), "open_margin": total_margin})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ stats
def _stats(trades: pd.DataFrame, equity: pd.DataFrame, capital_col: str | None) -> dict:
    if trades.empty:
        return {}
    pnl_yuan = trades["pnl"] * ds.MULTIPLIER
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = (eq - eq.cummax()).min()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan
    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    total_pnl = pnl_yuan.sum()
    if capital_col:
        # open_margin is already yuan-denominated (mg.compute_deposit's own
        # units), unlike "equity"/"open_pnl" which are raw per-contract price
        # units needing the *MULTIPLIER conversion -- do NOT scale again here
        cap_series = equity[capital_col]
        avg_cap = cap_series[cap_series > 0].mean()
        peak_cap = cap_series.max()
    else:
        avg_cap = trades["margin_yuan"].mean()
        peak_cap = avg_cap
    ret_avg_cap_yr = (total_pnl / avg_cap) / span_years if span_years > 0 and avg_cap else np.nan
    return dict(n=len(trades), win_rate=(trades["pnl"] > 0).mean(),
               mean_r=(pnl_yuan / trades["margin_yuan"]).mean(), total_pnl=total_pnl,
               max_dd=dd, sharpe=sharpe, avg_cap=avg_cap, peak_cap=peak_cap,
               ret_avg_cap_yr=ret_avg_cap_yr, span_years=span_years)


def _block_bootstrap_diff(base_eq: pd.Series, stack_eq: pd.Series,
                          n_boot: int = 2000, block_days: int = 20) -> dict:
    """Moving-block bootstrap on the daily PnL DIFFERENCE (stacked - baseline)
    series, preserving within-block time-correlation. Returns the bootstrap
    distribution's percentiles for the full-sample cumulative difference."""
    base_daily = base_eq.diff().dropna()
    stack_daily = stack_eq.diff().dropna()
    idx = base_daily.index.intersection(stack_daily.index)
    diff = (stack_daily.loc[idx] - base_daily.loc[idx]).to_numpy()
    n = len(diff)
    n_blocks_needed = int(np.ceil(n / block_days))
    starts = np.arange(0, n - block_days + 1)
    boot_totals = np.empty(n_boot)
    for b in range(n_boot):
        picks = RNG.choice(starts, size=n_blocks_needed, replace=True)
        sample = np.concatenate([diff[s:s + block_days] for s in picks])[:n]
        boot_totals[b] = sample.sum()
    return dict(actual_total_diff=diff.sum(),
               boot_mean=boot_totals.mean(),
               ci_lo=np.percentile(boot_totals, 2.5),
               ci_hi=np.percentile(boot_totals, 97.5),
               p_le_zero=(boot_totals <= 0).mean())


# ------------------------------------------------------------------ max_hold re-sweep
def _cell_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "mean_r": np.nan}
    r = trades["pnl"] * ds.MULTIPLIER / trades["margin_yuan"]
    return {"n": len(trades), "mean_r": r.mean()}


def sweep_max_hold(underlying: str, signal: str, threshold: float,
                   max_concurrent: int = 3, max_holds: list[int] | None = None) -> pd.DataFrame:
    """Re-optimize max_hold under the capped-at-max_concurrent stacking regime
    -- the frozen 20d pick was chosen under the single-position rule and isn't
    guaranteed to still be best once the exit rule interacts with a concurrency
    cap (faster turnover could free up slots sooner, or a longer hold could
    matter less now that positions layer instead of block each other)."""
    max_holds = max_holds or vr.MAX_HOLDS
    rows = []
    for hold in max_holds:
        res = run_stacked(underlying, signal, threshold, hold, max_concurrent=max_concurrent)
        trades = res["trades"]
        disc = trades[trades["entry_date"] < vr.SPLIT_DATE] if not trades.empty else trades
        hold_ = trades[trades["entry_date"] >= vr.SPLIT_DATE] if not trades.empty else trades
        d, h = _cell_stats(disc), _cell_stats(hold_)
        full = _stats(trades, res["equity"], capital_col="open_margin")
        rows.append({"max_hold": hold, "disc_n": d["n"], "disc_mean_r": d["mean_r"],
                    "hold_n": h["n"], "hold_mean_r": h["mean_r"],
                    "full_n": full.get("n", 0), "win_rate": full.get("win_rate", np.nan),
                    "sharpe": full.get("sharpe", np.nan), "max_dd": full.get("max_dd", np.nan),
                    "ret_avg_cap_yr": full.get("ret_avg_cap_yr", np.nan)})
    grid = pd.DataFrame(rows)
    print(f"\n=== {underlying} [{signal}>={threshold}] max_hold re-sweep, "
         f"max_concurrent={max_concurrent} (discovery<{vr.SPLIT_DATE.date()}<=holdout, MIN_CELL_N={vr.MIN_CELL_N}) ===")
    for _, row in grid.iterrows():
        flag = "" if row["disc_n"] >= vr.MIN_CELL_N else "  [disc_n<15, too thin]"
        agree = ("agree-POS" if row["disc_mean_r"] > 0 and row["hold_mean_r"] > 0
                else "agree-neg" if row["disc_mean_r"] < 0 and row["hold_mean_r"] < 0
                else "mixed")
        print(f"  hold={row['max_hold']:>2.0f}d  disc n={row['disc_n']:>3.0f} mean_r={row['disc_mean_r']:+.2%}  "
             f"hold n={row['hold_n']:>3.0f} mean_r={row['hold_mean_r']:+.2%}  ({agree})  "
             f"| full: win={row['win_rate']:.0%} sharpe={row['sharpe']:.2f} "
             f"max_dd={row['max_dd']:,.0f} ret/cap/yr={row['ret_avg_cap_yr']:.1%}{flag}")
    return grid


def compare(underlying: str, signal: str, threshold: float, max_hold_days: int,
           max_concurrent: int | None = None) -> None:
    base = vr.run(underlying, signal, threshold, max_hold_days)
    stack = run_stacked(underlying, signal, threshold, max_hold_days,
                        max_concurrent=max_concurrent)

    bs = _stats(base["trades"], base["equity"], capital_col=None)
    ss = _stats(stack["trades"], stack["equity"], capital_col="open_margin")

    cap_label = f"max_concurrent={max_concurrent}" if max_concurrent else "uncapped"
    print(f"\n=== {underlying} [{signal}>={threshold}, hold={max_hold_days}d] -- baseline vs stacked ({cap_label}) ===")
    print(f"  baseline: n={bs['n']}  win_rate={bs['win_rate']:.0%}  mean_r={bs['mean_r']:+.2%}  "
         f"total_pnl={bs['total_pnl']:,.0f}  max_dd={bs['max_dd']:,.0f}  sharpe={bs['sharpe']:.2f}  "
         f"avg_cap={bs['avg_cap']:,.0f}  ret/cap/yr={bs['ret_avg_cap_yr']:.1%}")
    print(f"  stacked : n={ss['n']}  win_rate={ss['win_rate']:.0%}  mean_r={ss['mean_r']:+.2%}  "
         f"total_pnl={ss['total_pnl']:,.0f}  max_dd={ss['max_dd']:,.0f}  sharpe={ss['sharpe']:.2f}  "
         f"avg_cap={ss['avg_cap']:,.0f}  peak_cap={ss['peak_cap']:,.0f}  ret/avg_cap/yr={ss['ret_avg_cap_yr']:.1%}")
    n_open = stack["equity"]["n_open"]
    print(f"  concurrent positions: mean={n_open[n_open>0].mean():.1f}  "
         f"median={n_open[n_open>0].median():.0f}  max={n_open.max()}  "
         f"days with >=2 open={100*(n_open>=2).mean():.0f}% of sample")

    boot = _block_bootstrap_diff(base["equity"].set_index("date")["equity"] * ds.MULTIPLIER,
                                 stack["equity"].set_index("date")["equity"] * ds.MULTIPLIER)
    sig_flag = "SIGNIFICANT" if (boot["ci_lo"] > 0 or boot["ci_hi"] < 0) else "not significant"
    print(f"  block-bootstrap on daily PnL diff (stacked-baseline), n=2000, block=20d:")
    print(f"    actual total diff={boot['actual_total_diff']:,.0f} yuan  "
         f"95% CI=[{boot['ci_lo']:,.0f}, {boot['ci_hi']:,.0f}]  "
         f"P(diff<=0)={boot['p_le_zero']:.0%}  -> {sig_flag}")


def main() -> None:
    configs = [
        ("510050", "vrp", 0.06, 20),
        ("510300", "vrp", 0.03, 20),
        ("510050", "iv_pct", 0.90, 20),
        ("510300", "iv_pct", 0.80, 20),
    ]
    for underlying, signal, threshold, max_hold in configs:
        compare(underlying, signal, threshold, max_hold, max_concurrent=None)
        compare(underlying, signal, threshold, max_hold, max_concurrent=3)


if __name__ == "__main__":
    main()
