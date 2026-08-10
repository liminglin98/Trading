"""Backtest a long ATM straddle on SHFE crude oil (SC) options, rolled
across contract-months, as a candidate overlay for the 510050 quarterly
short straddle (`quarterly_atm_short_straddle.py`).

Why crude oil, and why a straddle (not a directional position): the ETF
strategy's real trade history shows its worst quarters are symmetric in
direction -- big rallies (2019 Q1, 2020 Q3) hurt exactly as much as big
crashes (2015, 2018, 2022) -- so a directional commodity position can't
help across all of them. A magnitude-correlation scan across the ~45
commodity option varieties in Data/options/ found SC's realized move size
(not direction) lined up with several of those bad quarters, pointing to a
long-volatility (straddle) overlay instead, which profits from magnitude
rather than direction -- matching how the base strategy actually loses.

Data caveats (real quotes only, no synthetic/Black-Scholes pricing):
  * Data/options/option_daily_SC.csv has real option quotes only from
    2021-06-21 onward -- nothing earlier exists in this repo. This
    backtest is scoped to that window; it cannot say anything about the
    pre-2021 episodes (2015 crash, 2018, 2019 Q1, 2020 COVID
    crash/rebound).
  * No single SC contract spans a full ETF-strategy quarter -- SHFE crude
    options trade ~7-10 weeks, expiring roughly a month before their
    delivery month (empirically: SC2109->2021-08-13, SC2110->2021-09-03,
    SC2111->2021-10-13, SC2201->2021-12-07, SC2204->2022-03-15). So this
    rolls through 2-3 contract-months per ETF quarter, unlike the ETF
    leg's clean single-contract quarterly hold.
  * expire_date isn't reliably populated for the post-2026-05-29 rows (the
    live-fetched/backfilled period) -- rolls trigger on the held
    contract's call/put codes disappearing from the day's chain instead
    of a stored expiry date, same fallback quarterly_atm_short_straddle.py's
    run() already uses.
  * Multiplier: 1,000 barrels/lot -- published INE/SHFE crude-oil-option
    spec, not found anywhere in this repo's own files. Verify against
    exchange docs before sizing real capital.

Usage:
    python commodity_straddle_overlay_backtest.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_DIR = REPO / "Data" / "options"
OUT_DIR = HERE / "out"

MULTIPLIER = 1000  # barrels/lot -- see module docstring
ETF_MULTIPLIER = 10000  # data_store.MULTIPLIER, for the 510050 leg

CODE_RE = re.compile(r"^SC(\d{4})(C|P)")


def load_sc_chain() -> pd.DataFrame:
    """SC option chain with each row's own contract-month's futures settle
    price merged in as the underlying reference (the option chain itself
    has no underlying-close column, unlike the SSE ETF schema's
    C_UND_CLOSE) -- each option contract-month is priced off the matching
    futures contract-month, not a generic front-month proxy."""
    opt = pd.read_csv(DATA_DIR / "option_daily_SC.csv", dtype={"code": str})
    opt["date"] = pd.to_datetime(opt["date"])
    opt = opt[opt["close"] > 0].copy()
    parsed = opt["code"].str.extract(CODE_RE)
    opt["contract_month"] = parsed[0]
    opt["futures_contract"] = "SC" + parsed[0]

    fut = pd.read_csv(DATA_DIR / "futures_daily_SC.csv")
    fut["date"] = pd.to_datetime(fut["date"])
    fut = fut[["date", "contract", "settle"]].rename(
        columns={"contract": "futures_contract", "settle": "und_settle"})
    opt = opt.merge(fut, on=["date", "futures_contract"], how="left")
    return opt


def _atm_strike(day: pd.DataFrame, month: str, spot: float) -> float | None:
    strikes = day.loc[day["contract_month"] == month, "strike"].unique()
    if len(strikes) == 0:
        return None
    return float(strikes[np.argmin(np.abs(strikes - spot))])


def _leg_row(day: pd.DataFrame, month: str, cp: str, strike: float) -> pd.Series | None:
    m = day[(day["contract_month"] == month) & (day["contract_type"] == cp) &
            (day["strike"] == strike)]
    if m.empty or m["close"].iloc[0] <= 0:
        return None
    return m.iloc[0]


def _intrinsic(cp: str, strike: float, spot: float) -> float:
    return max(spot - strike, 0.0) if cp == "CO" else max(strike - spot, 0.0)


def run(start_date: str | None = None) -> dict:
    """Long 1x ATM call + 1x ATM put on SC's front contract-month, marked
    to close daily, rolling into the new front month the day after the
    held contract's legs disappear from the chain (no same-day re-entry,
    same convention as quarterly_atm_short_straddle.py's run())."""
    df = load_sc_chain()
    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    df = df.dropna(subset=["und_settle"])
    dates = sorted(df["date"].unique())
    by_date = dict(tuple(df.groupby("date")))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_pos: dict | None = None
    realized = 0.0

    for dt in dates:
        day = by_date[dt]

        if open_pos is not None:
            call = day[day["code"] == open_pos["call_code"]]
            put = day[day["code"] == open_pos["put_code"]]
            spot_rows = day.loc[day["contract_month"] == open_pos["contract_month"], "und_settle"]
            spot = (float(spot_rows.iloc[0]) if not spot_rows.empty and pd.notna(spot_rows.iloc[0])
                   else open_pos["last_spot"])
            if call.empty or put.empty:
                call_px = _intrinsic("CO", open_pos["strike"], spot)
                put_px = _intrinsic("PO", open_pos["strike"], spot)
                mark = call_px + put_px
                pnl = mark - open_pos["cost"]
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "contract_month": open_pos["contract_month"],
                    "roll_date": dt, "strike": open_pos["strike"],
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "call_entry_px": open_pos["call_px"], "put_entry_px": open_pos["put_px"],
                    "cost": open_pos["cost"], "pnl": pnl,
                    "days_held": (pd.Timestamp(dt) - open_pos["entry_date"]).days,
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
                continue
            call_px, put_px = float(call["close"].iloc[0]), float(put["close"].iloc[0])
            mark = call_px + put_px
            pnl = mark - open_pos["cost"]
            open_pos["last_spot"] = spot
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl})
            continue

        months = sorted(day["contract_month"].dropna().unique())
        if not months:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
            continue
        month = months[0]
        spot_rows = day.loc[day["contract_month"] == month, "und_settle"]
        if spot_rows.empty or pd.isna(spot_rows.iloc[0]):
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
            continue
        spot = float(spot_rows.iloc[0])
        strike = _atm_strike(day, month, spot)
        call_row = _leg_row(day, month, "CO", strike) if strike is not None else None
        put_row = _leg_row(day, month, "PO", strike) if strike is not None else None
        if call_row is None or put_row is None:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
            continue
        call_px, put_px = float(call_row["close"]), float(put_row["close"])
        open_pos = {
            "entry_date": pd.Timestamp(dt), "entry_spot": spot, "contract_month": month,
            "strike": strike, "call_code": call_row["code"], "put_code": put_row["code"],
            "call_px": call_px, "put_px": put_px, "cost": call_px + put_px, "last_spot": spot,
        }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})

    return {"trades": pd.DataFrame(trades), "equity": pd.DataFrame(equity_rows)}


def load_etf_trades(underlying: str = "510050") -> pd.DataFrame:
    path = OUT_DIR / f"quarterly_short_straddle_trades_{underlying}.csv"
    df = pd.read_csv(path)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["pnl_yuan"] = df["pnl"] * ETF_MULTIPLIER
    return df


def combine(etf_trades: pd.DataFrame, sc_equity: pd.DataFrame, sc_start: pd.Timestamp) -> pd.DataFrame:
    """Per-ETF-quarter comparison, using the SC equity curve (mark-to-
    market, already includes any still-open position) rather than only
    realized SC rolls -- a position that opens near one ETF quarter's end
    and doesn't roll off until into the next quarter is handled correctly
    via equity differencing instead of being dropped or double-counted."""
    eq = sc_equity.set_index("date")["equity"] * MULTIPLIER
    rows = []
    for _, tr in etf_trades.iterrows():
        if tr["exit_date"] < sc_start:
            continue
        window_start = max(tr["entry_date"], sc_start)
        entry_eq = eq[eq.index <= window_start]
        exit_eq = eq[eq.index <= tr["exit_date"]]
        if entry_eq.empty or exit_eq.empty:
            continue
        overlay_pnl = exit_eq.iloc[-1] - entry_eq.iloc[-1]
        rows.append({
            "entry_date": tr["entry_date"].date(), "exit_date": tr["exit_date"].date(),
            "etf_pnl": tr["pnl_yuan"], "sc_overlay_pnl": overlay_pnl,
            "combined": tr["pnl_yuan"] + overlay_pnl,
        })
    return pd.DataFrame(rows)


def _mdd(pnl_series: pd.Series) -> float:
    eq = pnl_series.cumsum()
    return (eq - eq.cummax()).min()


def report(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        print("No overlapping quarters -- nothing to report.")
        return
    pd.set_option("display.width", 140)
    disp = comparison.copy()
    for c in ("etf_pnl", "sc_overlay_pnl", "combined"):
        disp[c] = disp[c].round(0)
    print("--- 1 SC straddle lot per 1 ETF straddle combo ---")
    print(disp.to_string(index=False))

    print(f"\n510050 alone:          total_pnl={comparison['etf_pnl'].sum():>12,.0f} yuan  "
          f"worst_quarter={comparison['etf_pnl'].min():>12,.0f}  max_drawdown={_mdd(comparison['etf_pnl']):>12,.0f}")
    print(f"+ SC straddle (1 lot): total_pnl={comparison['combined'].sum():>12,.0f} yuan  "
          f"worst_quarter={comparison['combined'].min():>12,.0f}  max_drawdown={_mdd(comparison['combined']):>12,.0f}")
    print(f"SC overlay alone:      total_pnl={comparison['sc_overlay_pnl'].sum():>12,.0f} yuan  "
          f"worst_quarter={comparison['sc_overlay_pnl'].min():>12,.0f}  "
          f"win_rate={(comparison['sc_overlay_pnl']>0).mean():.0%}")

    # Risk-matched sizing: scale SC lots so its P&L volatility matches the
    # ETF leg's, rather than comparing at an arbitrary 1:1 lot ratio (SC
    # notional per lot is far larger than one ETF straddle combo's risk).
    scale = comparison["etf_pnl"].std() / comparison["sc_overlay_pnl"].std()
    scaled_overlay = comparison["sc_overlay_pnl"] * scale
    scaled_combined = comparison["etf_pnl"] + scaled_overlay
    print(f"\n--- risk-matched sizing: {scale:.4f} SC lots per 1 ETF straddle combo "
          f"(matches P&L std dev -- not a whole number, flagging the granularity mismatch "
          f"rather than rounding it away) ---")
    print(f"510050 alone:          total_pnl={comparison['etf_pnl'].sum():>12,.0f} yuan  "
          f"worst_quarter={comparison['etf_pnl'].min():>12,.0f}  max_drawdown={_mdd(comparison['etf_pnl']):>12,.0f}")
    print(f"+ SC straddle (scaled): total_pnl={scaled_combined.sum():>12,.0f} yuan  "
          f"worst_quarter={scaled_combined.min():>12,.0f}  max_drawdown={_mdd(scaled_combined):>12,.0f}")

    print("\n(Reporting cumulative yuan P&L and drawdown rather than CAGR/%MDD: the ETF leg's "
          "margin-based capital base and the SC leg's premium-based capital base aren't the same "
          "denominator, so a blended %-return figure would be misleading without an explicit "
          "sizing/capital-allocation assumption.)")


if __name__ == "__main__":
    sc = run(start_date="2021-06-21")
    OUT_DIR.mkdir(exist_ok=True)
    sc["trades"].to_csv(OUT_DIR / "sc_long_straddle_trades.csv", index=False)
    print(f"SC long-straddle trades: {len(sc['trades'])} rolls, "
          f"{sc['equity']['date'].min().date()} -> {sc['equity']['date'].max().date()}")
    print(f"Saved to {OUT_DIR / 'sc_long_straddle_trades.csv'}\n")

    etf_trades = load_etf_trades("510050")
    comparison = combine(etf_trades, sc["equity"], sc["equity"]["date"].min())
    report(comparison)
