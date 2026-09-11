"""Intraday short-straddle backtest: sell 1 ATM call + 1 ATM put at the
session open, buy both back at the session close, same day (no overnight
hold) -- the direct trade implied by intraday_iv_seasonality_check.py's
finding that ATM IV runs high at the open and decays through the session.

This backtest exists specifically to answer "does the bid/ask spread and
commission eat the edge": unlike the daily-EOD CFFEX backtests in this repo
(expiry_week_atm_straddle_backtest.py etc.), which only have close-to-close
marks and so *assume* a slippage-in-index-points sensitivity grid, the
minute chain in Data/options/minute/ has REAL polled bid/ask quotes -- so
entry/exit fills here are priced on the actual observed spread, not an
assumption. Commission has no established per-contract convention anywhere
in this repo (grep confirms), so it's swept as a sensitivity grid instead of
picked arbitrarily.

Method
------
Per underlying x trading day: take the front-month (min dte) contracts. At
the FIRST polled minute of the day, find the strike nearest spot and lock in
that call+put pair for the whole day (no re-selection intraday -- this is
one held position, not a moving-strike series). Sell both legs at their bid
(what a market sell actually receives). At the LAST polled minute of the day
(<=15:00), buy both legs back at their ask (what a market buy actually
pays). Compare against the frictionless mid-to-mid P&L to isolate the
spread's cost, then layer a commission-per-leg-per-trade grid on top.

P&L convention: positive = short-straddle-seller profit (premium decayed).
Reported in price units, in yuan (x each underlying's real contract
multiplier -- 10000 for SSE ETF options, 100 for CFFEX index options), and
as a return on the straddle premium collected (comparable across underlyings
with very different price levels/multipliers).

Usage: python intraday_short_straddle_backtest.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import data_store as ds

MINUTE_DIR = Path("/mnt/data/Trading/Data/options/minute")
FILE_RE = re.compile(r"^(index_)?option_minute_(\d+)_(\d{8})\.csv$")

ETF_UNDERLYINGS = {"510050", "510300", "510500", "588000"}
MULTIPLIER = {u: ds.MULTIPLIER for u in ETF_UNDERLYINGS} | \
             {u: 100 for u in ("000300", "000852", "000016")}  # CFFEX spec, cffex_margin.MULTIPLIER

COMMISSION_GRID_YUAN = [0.0, 3.0, 10.0, 20.0]  # per leg, per side (entry or exit), round-trip = x4


def _front_month_pair(day_df: pd.DataFrame) -> tuple[str, str, float, int, pd.Timestamp] | None:
    """Pick front-month ATM call+put codes locked in at the day's first snapshot."""
    front_dte = day_df["dte"].min()
    fm = day_df[day_df["dte"] == front_dte]
    t0 = fm["timestamp"].min()
    snap0 = fm[fm["timestamp"] == t0]
    spot0 = snap0["spot"].iloc[0]
    if not (spot0 > 0):
        return None
    idx = (snap0["strike"] - spot0).abs().idxmin()
    atm_strike = snap0.loc[idx, "strike"]
    leg = snap0[snap0["strike"] == atm_strike]
    calls = leg[leg["cp"] == "C"]
    puts = leg[leg["cp"] == "P"]
    if calls.empty or puts.empty:
        return None
    return calls["code"].iloc[0], puts["code"].iloc[0], atm_strike, int(front_dte), t0


def day_trade(path: Path, underlying: str) -> dict | None:
    df = pd.read_csv(path)
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    picked = _front_month_pair(df)
    if picked is None:
        return None
    call_code, put_code, atm_strike, dte, t_first = picked

    call_rows = df[df["code"] == call_code]
    put_rows = df[df["code"] == put_code]
    if call_rows.empty or put_rows.empty:
        return None

    entry_call = call_rows.iloc[0]
    entry_put = put_rows.iloc[0]
    exit_call = call_rows.iloc[-1]
    exit_put = put_rows.iloc[-1]

    def valid_quote(r):
        return r["bid"] > 0 and r["ask"] > 0 and r["ask"] >= r["bid"]

    if not all(valid_quote(r) for r in (entry_call, entry_put, exit_call, exit_put)):
        return None

    entry_call_mid = (entry_call["bid"] + entry_call["ask"]) / 2
    entry_put_mid = (entry_put["bid"] + entry_put["ask"]) / 2
    exit_call_mid = (exit_call["bid"] + exit_call["ask"]) / 2
    exit_put_mid = (exit_put["bid"] + exit_put["ask"]) / 2

    # Sell at bid, buy back at ask -- the real fill a market order gets.
    friction_pnl = (entry_call["bid"] + entry_put["bid"]) - (exit_call["ask"] + exit_put["ask"])
    mid_pnl = (entry_call_mid + entry_put_mid) - (exit_call_mid + exit_put_mid)
    premium_mid = entry_call_mid + entry_put_mid

    return {
        "underlying": underlying,
        "file": path.name,
        "date": t_first.date(),
        "dte": dte,
        "atm_strike": atm_strike,
        "entry_ts": entry_call["timestamp"],
        "exit_ts": exit_call["timestamp"],
        "premium_mid": premium_mid,
        "mid_pnl": mid_pnl,
        "friction_pnl": friction_pnl,
        "spread_cost": mid_pnl - friction_pnl,
    }


def main():
    files = sorted(MINUTE_DIR.glob("*.csv"))
    matched = [(f, m) for f in files if (m := FILE_RE.match(f.name))]
    print(f"{len(matched)} minute files found under {MINUTE_DIR}")

    trades = []
    for f, m in matched:
        underlying = m.group(2)
        t = day_trade(f, underlying)
        if t is not None:
            trades.append(t)

    if not trades:
        print("No usable trades extracted.")
        return

    trades = pd.DataFrame(trades)
    trades["multiplier"] = trades["underlying"].map(MULTIPLIER)
    trades["mid_pnl_yuan"] = trades["mid_pnl"] * trades["multiplier"]
    trades["friction_pnl_yuan"] = trades["friction_pnl"] * trades["multiplier"]
    trades["mid_ret"] = trades["mid_pnl"] / trades["premium_mid"]
    trades["friction_ret"] = trades["friction_pnl"] / trades["premium_mid"]

    print(f"\n{len(trades)} trades across {trades['underlying'].nunique()} underlyings")
    print(trades.groupby("underlying")["file"].nunique().to_string())

    n_0dte = int((trades["dte"] == 0).sum())
    print(f"\n{n_0dte} trades are on the front-month's OWN expiry day (dte=0) -- straddle premium "
          f"collapses toward zero by the close almost by construction there (extreme gamma/pin risk, "
          f"not a normal representative day). Excluded from the headline below and reported separately.")
    zero_dte = trades[trades["dte"] == 0]
    core = trades[trades["dte"] > 0].copy()

    print("\n=== Frictionless (mid-to-mid) vs bid/ask-crossed, by underlying (dte>0 only) ===")
    print("Return on premium collected (positive = short-straddle profit)\n")
    summary = core.groupby("underlying").agg(
        n=("file", "nunique"),
        mid_ret_mean=("mid_ret", "mean"),
        mid_win=("mid_pnl", lambda s: (s > 0).mean()),
        friction_ret_mean=("friction_ret", "mean"),
        friction_win=("friction_pnl", lambda s: (s > 0).mean()),
        spread_cost_ret=("spread_cost", "mean"),
    )
    summary["spread_cost_ret"] = summary["spread_cost_ret"] / core.groupby("underlying")["premium_mid"].mean()
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== Pooled across all underlyings/days (dte>0 only) ===")
    print(f"n = {len(core)}")
    print(f"Frictionless: mean return/trade = {core['mid_ret'].mean():+.2%}, "
          f"win rate = {(core['mid_pnl'] > 0).mean():.1%}, "
          f"mean P&L = {core['mid_pnl_yuan'].mean():+.1f} yuan/trade")
    print(f"Bid/ask-crossed (spread only, no commission): mean return/trade = {core['friction_ret'].mean():+.2%}, "
          f"win rate = {(core['friction_pnl'] > 0).mean():.1%}, "
          f"mean P&L = {core['friction_pnl_yuan'].mean():+.1f} yuan/trade")
    print(f"Spread cost alone = {(core['mid_ret'].mean() - core['friction_ret'].mean()):.2%} of premium/trade "
          f"({core['spread_cost'].mean() * core['multiplier'].mean():.1f} yuan avg, blended multiplier)")

    if n_0dte:
        print(f"\nFor reference, the {n_0dte} excluded dte=0 trades: mean friction return = "
              f"{zero_dte['friction_ret'].mean():+.2%}, win rate = {(zero_dte['friction_pnl'] > 0).mean():.1%} "
              f"-- do not extrapolate from this, n={n_0dte}.")

    print("\n=== Days-to-expiry sensitivity (bid/ask-crossed, dte>0) ===")
    bins = [(1, 3, "1-3 dte"), (4, 7, "4-7 dte"), (8, 14, "8-14 dte"), (15, 999, "15+ dte")]
    for lo, hi, label in bins:
        sub = core[(core["dte"] >= lo) & (core["dte"] <= hi)]
        if sub.empty:
            continue
        print(f"{label:<10} n={len(sub):>4}  friction_ret={sub['friction_ret'].mean():+.2%}  "
              f"win={(sub['friction_pnl'] > 0).mean():.1%}")

    print("\n=== Commission sensitivity (added on top of the bid/ask-crossed fill, dte>0) ===")
    print("Commission grid = per leg, per side (so round-trip straddle = 4x the figure)\n")
    header = "  ".join(f"{c:>8.1f}" for c in COMMISSION_GRID_YUAN)
    print(f"{'underlying':<12}{header}")
    for u, g in core.groupby("underlying"):
        cells = []
        for c in COMMISSION_GRID_YUAN:
            net_yuan = g["friction_pnl_yuan"] - 4 * c
            cells.append(f"{net_yuan.mean():>+8.1f}")
        print(f"{u:<12}" + "  ".join(cells))
    cells = []
    for c in COMMISSION_GRID_YUAN:
        net_yuan = core["friction_pnl_yuan"] - 4 * c
        cells.append(f"{net_yuan.mean():>+8.1f}")
    print(f"{'pooled':<12}" + "  ".join(cells) + "   (mean net P&L/trade, yuan, blended multiplier)")

    win_header = "  ".join(f"{c:>8.1f}" for c in COMMISSION_GRID_YUAN)
    print(f"\n{'underlying':<12}{win_header}   (win rate after commission)")
    for u, g in core.groupby("underlying"):
        cells = []
        for c in COMMISSION_GRID_YUAN:
            net_yuan = g["friction_pnl_yuan"] - 4 * c
            cells.append(f"{(net_yuan > 0).mean():>8.1%}")
        print(f"{u:<12}" + "  ".join(cells))

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "intraday_short_straddle_trades.csv"
    trades.to_csv(out_path, index=False)
    print(f"\nSaved trade-level table to {out_path}")


if __name__ == "__main__":
    main()
