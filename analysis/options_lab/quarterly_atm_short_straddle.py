"""Rolling ATM short straddle on the exchange's quarterly (季月) contract.

Strategy: sell 1x ATM call + 1x ATM put (same strike, nearest to spot) on
the nearest listed "quarterly" contract (expiry month in {3,6,9,12} -- SSE's
当季/下季 series), hold to expiry with no profit target / stop loss, settle
at intrinsic value, and immediately roll into the new nearest quarterly
contract the next trading day. Fills at daily close; no fees/slippage
(matches the options_lab backtest engine's conventions).

Usage:
    python quarterly_atm_short_straddle.py --underlying 510050
    python quarterly_atm_short_straddle.py --underlying 510300
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

OUT_DIR = Path(__file__).resolve().parent / "out"

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"


def _target_expiries(day: pd.DataFrame, roll_freq: str) -> list[pd.Timestamp]:
    exps = sorted(day[ds.C_EXPIRY].unique())
    if roll_freq == "monthly":
        return exps  # nearest listed expiry = front month (当月/次月)
    return [e for e in exps if pd.Timestamp(e).month in (3, 6, 9, 12)]


def _atm_strike(day: pd.DataFrame, expiry: pd.Timestamp, spot: float) -> float | None:
    strikes = day.loc[day[ds.C_EXPIRY] == expiry, ds.C_STRIKE].unique()
    if len(strikes) == 0:
        return None
    return float(strikes[np.argmin(np.abs(strikes - spot))])


def _leg_row(day: pd.DataFrame, expiry: pd.Timestamp, cp: str,
            strike: float) -> pd.Series | None:
    m = day[(day[ds.C_EXPIRY] == expiry) & (day[ds.C_CP] == cp) &
            (day[ds.C_STRIKE] == strike)]
    if m.empty or m[ds.C_CLOSE].iloc[0] <= 0:
        return None
    return m.iloc[0]


def _intrinsic(cp: str, strike: float, spot: float) -> float:
    return max(spot - strike, 0.0) if cp == "C" else max(strike - spot, 0.0)


def run(underlying: str, fixed_strike: bool = False, roll_freq: str = "quarterly",
       start_date: str | None = None, strike_target: float | None = None) -> dict:
    """fixed_strike=True: freeze the strike and reuse it (clamped to the
    nearest strike actually listed) on every roll, instead of re-centering
    on spot each roll. strike_target: explicit level to freeze at (e.g. 3.0);
    if omitted, freezes at the very first entry's ATM level instead.
    roll_freq: "quarterly" (季月 contract, ~90d) or "monthly" (front month,
    当月/次月, ~20-45d depending on where in the cycle the roll lands).
    start_date: optional 'YYYY-MM-DD' to restrict the backtest window."""
    df = ds.load_chain(underlying)
    df = df[~df[ds.C_NAME].astype(str).str.endswith("A")]  # exclude adjusted
    df = df[df[ds.C_CLOSE] > 0]
    if start_date is not None:
        df = df[df[ds.C_DATE] >= pd.Timestamp(start_date)]
    dates = sorted(df[ds.C_DATE].unique())
    by_date = dict(tuple(df.groupby(ds.C_DATE)))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_pos: dict | None = None
    realized = 0.0
    frozen_strike: float | None = strike_target

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            call = day[day[ds.C_CODE] == open_pos["call_code"]]
            put = day[day[ds.C_CODE] == open_pos["put_code"]]
            expired = pd.Timestamp(dt) >= open_pos["expiry"]
            if expired or call.empty or put.empty:
                call_px = _intrinsic("C", open_pos["strike"], spot)
                put_px = _intrinsic("P", open_pos["strike"], spot)
                mark = -(call_px + put_px)
                pnl = mark - open_pos["cost"]
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "expiry": open_pos["expiry"],
                    "exit_date": dt, "strike": open_pos["strike"],
                    "target_strike": open_pos.get("target_strike"),
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "call_entry_px": open_pos["call_px"], "put_entry_px": open_pos["put_px"],
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": (pd.Timestamp(dt) - open_pos["entry_date"]).days,
                    "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            call_px = float(call[ds.C_CLOSE].iloc[0])
            put_px = float(put[ds.C_CLOSE].iloc[0])
            mark = -(call_px + put_px)
            pnl = mark - open_pos["cost"]
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: roll into nearest target-frequency contract
        qexp = _target_expiries(day, roll_freq)
        if not qexp:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue
        expiry = pd.Timestamp(qexp[0])
        target = frozen_strike if (fixed_strike and frozen_strike is not None) else spot
        strike = _atm_strike(day, expiry, target)
        if fixed_strike and frozen_strike is None and strike is not None:
            frozen_strike = strike
            target = strike  # first entry defines the frozen reference
        call_row = _leg_row(day, expiry, "C", strike) if strike is not None else None
        put_row = _leg_row(day, expiry, "P", strike) if strike is not None else None
        if call_row is None or put_row is None:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue

        call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
        legs_df = pd.DataFrame([
            {"expiry": expiry, "cp": "C", "strike": strike, "direction": -1,
             "qty": 1, "entry_price": call_px},
            {"expiry": expiry, "cp": "P", "strike": strike, "direction": -1,
             "qty": 1, "entry_price": put_px},
        ])
        margin_yuan, _ = mg.compute_deposit(legs_df, spot)

        open_pos = {
            "entry_date": pd.Timestamp(dt), "entry_spot": spot, "expiry": expiry,
            "strike": strike, "target_strike": target, "call_code": call_row[ds.C_CODE],
            "put_code": put_row[ds.C_CODE],
            "call_px": call_px, "put_px": put_px, "cost": -(call_px + put_px),
            "margin": margin_yuan,
        }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


def report(res: dict, underlying: str, fixed_strike: bool = False,
          roll_freq: str = "quarterly") -> None:
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        print("No trades.")
        return
    pnl_yuan = trades["pnl"] * ds.MULTIPLIER
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan

    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    avg_margin = trades["margin_yuan"].mean()
    total_pnl = pnl_yuan.sum()
    cagr_on_margin = (total_pnl / avg_margin) / span_years if span_years > 0 else np.nan

    label = "fixed-strike" if fixed_strike else "re-centered ATM"
    unit = "quarters" if roll_freq == "quarterly" else "months"
    print(f"{underlying}: {roll_freq} short straddle ({label}), "
          f"{equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades({unit})={len(trades)}  avg_days_held={trades['days_held'].mean():.0f}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}")
    print(f"  avg_win={pnl_yuan[pnl_yuan>0].mean():,.0f}  "
          f"avg_loss={pnl_yuan[pnl_yuan<=0].mean():,.0f} yuan")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin (naked, per combo)={avg_margin:,.0f} yuan  "
          f"return_on_margin/yr~={cagr_on_margin:.1%}")
    if fixed_strike:
        moneyness = (trades["entry_spot"] - trades["strike"]) / trades["strike"]
        clamped = (trades["strike"] != trades["target_strike"]).sum()
        print(f"  frozen strike={trades['strike'].iloc[0]:g}  "
              f"clamped-to-nearest-listed on {clamped}/{len(trades)} rolls  "
              f"final moneyness={moneyness.iloc[-1]:+.1%} (spot vs strike)")

    trades["year"] = trades["entry_date"].dt.year
    print("\nBy year:")
    for yr, g in trades.groupby("year"):
        print(f"  {yr}: n={len(g)}  pnl={g['pnl'].sum()*ds.MULTIPLIER:>10,.0f}  "
              f"win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    suffix = ("_fixedstrike" if fixed_strike else "") + \
              ("_monthly" if roll_freq == "monthly" else "")
    trades_out = OUT_DIR / f"quarterly_short_straddle_trades_{underlying}{suffix}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\nTrades saved to {trades_out}")


def plot(res: dict, underlying: str, out_path: Path, fixed_strike: bool = False,
        roll_freq: str = "quarterly") -> None:
    equity, trades = res["equity"], res["trades"]
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
    for _, t in trades.iterrows():
        ax1.axvline(t["entry_date"], color=C_AXIS, linewidth=0.6, alpha=0.5)
    ax1.set_ylabel("cumulative P&L (yuan)", color=C_INK2, fontsize=10)
    label = "fixed-strike" if fixed_strike else "re-centered ATM"
    ax1.set_title(f"{underlying}: rolling short straddle ({label}, {roll_freq}, hold-to-expiry)",
                  color=C_INK, fontsize=12, loc="left", pad=10)

    ax2.fill_between(dd.index, 0, dd.values, color=C_RED, alpha=0.55, linewidth=0)
    ax2.set_ylabel("drawdown (yuan)", color=C_INK2, fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def main(underlying: str, fixed_strike: bool, roll_freq: str,
        start_date: str | None, strike_target: float | None) -> None:
    res = run(underlying, fixed_strike=fixed_strike, roll_freq=roll_freq,
             start_date=start_date, strike_target=strike_target)
    report(res, underlying, fixed_strike=fixed_strike, roll_freq=roll_freq)
    suffix = ("_fixedstrike" if fixed_strike else "") + \
              ("_monthly" if roll_freq == "monthly" else "")
    out_path = OUT_DIR / f"quarterly_short_straddle_{underlying}{suffix}.png"
    plot(res, underlying, out_path, fixed_strike=fixed_strike, roll_freq=roll_freq)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--fixed-strike", action="store_true",
                    help="freeze the strike instead of re-centering on spot "
                         "each roll (level set by --strike, or the first "
                         "entry's ATM if --strike is omitted)")
    ap.add_argument("--strike", type=float, default=None, dest="strike_target",
                    help="explicit fixed strike level, e.g. 3.0; implies --fixed-strike")
    ap.add_argument("--roll", choices=["quarterly", "monthly"], default="quarterly",
                    dest="roll_freq",
                    help="quarterly=季月 contract (~90d); monthly=front month (~20-45d)")
    ap.add_argument("--start-date", default=None,
                    help="restrict backtest to dates on/after YYYY-MM-DD")
    args = ap.parse_args()
    fixed = args.fixed_strike or args.strike_target is not None
    main(args.underlying, fixed, args.roll_freq, args.start_date, args.strike_target)
