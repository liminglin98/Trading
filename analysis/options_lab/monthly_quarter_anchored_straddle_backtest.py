"""Rolling ATM short straddle on the front-month (当月) contract, but the
strike is only re-centered on spot once per CALENDAR quarter -- the first
monthly roll of each quarter picks fresh ATM (nearest to spot) and that same
strike (clamped to whatever's actually listed each month) is reused,
unadjusted, for the remaining monthly rolls inside that quarter.

Motivation: `quarterly_atm_short_straddle.py --roll monthly` re-centers ATM
on every single monthly roll (~12x/year). This variant asks whether decoupling
the roll cadence (monthly, so each leg only carries ~20-45 DTE of gamma/theta
risk) from the strike-selection cadence (quarterly, so you're not re-paying
the bid/ask + re-picking a fresh strike as spot chops around inside a quarter)
captures monthly's faster theta without monthly's more frequent re-centering.

Same engine conventions as quarterly_atm_short_straddle.py: fills at daily
close, no fees/slippage, hold each leg to its own expiry, cash-settle at
intrinsic value, immediately roll into the next front-month contract.

Usage:
    python monthly_quarter_anchored_straddle_backtest.py --underlying 510050
    python monthly_quarter_anchored_straddle_backtest.py --underlying 510300
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
import quarterly_atm_short_straddle as qs

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = pd.Timestamp("2024-01-01")

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"


def _quarter_key(dt: pd.Timestamp) -> tuple[int, int]:
    return (dt.year, (dt.month - 1) // 3 + 1)


def run(underlying: str, start_date: str | None = None) -> dict:
    """Monthly roll cadence, strike re-centered on spot only at the first
    monthly entry of each calendar quarter; every other monthly roll inside
    that quarter reuses the quarter's anchor strike (clamped to the nearest
    strike actually listed for that month's expiry)."""
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
    anchor_quarter: tuple[int, int] | None = None
    anchor_strike: float | None = None

    for dt in dates:
        dt = pd.Timestamp(dt)
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            call = day[day[ds.C_CODE] == open_pos["call_code"]]
            put = day[day[ds.C_CODE] == open_pos["put_code"]]
            expired = dt >= open_pos["expiry"]
            if expired or call.empty or put.empty:
                call_px = qs._intrinsic("C", open_pos["strike"], spot)
                put_px = qs._intrinsic("P", open_pos["strike"], spot)
                mark = -(call_px + put_px)
                pnl = mark - open_pos["cost"]
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "expiry": open_pos["expiry"],
                    "exit_date": dt, "strike": open_pos["strike"],
                    "anchor_strike": open_pos["anchor_strike"], "quarter": open_pos["quarter"],
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "call_entry_px": open_pos["call_px"], "put_entry_px": open_pos["put_px"],
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": (dt - open_pos["entry_date"]).days,
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

        # ---- entry: roll into nearest front-month contract
        mexp = qs._target_expiries(day, "monthly")
        if not mexp:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue
        expiry = pd.Timestamp(mexp[0])

        qkey = _quarter_key(dt)
        if qkey != anchor_quarter:
            anchor_quarter = qkey
            anchor_strike = None  # first monthly roll of a new quarter re-centers

        target = anchor_strike if anchor_strike is not None else spot
        strike = qs._atm_strike(day, expiry, target)
        if anchor_strike is None and strike is not None:
            anchor_strike = strike

        call_row = qs._leg_row(day, expiry, "C", strike) if strike is not None else None
        put_row = qs._leg_row(day, expiry, "P", strike) if strike is not None else None
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
            "entry_date": dt, "entry_spot": spot, "expiry": expiry,
            "strike": strike, "anchor_strike": anchor_strike, "quarter": qkey,
            "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
            "call_px": call_px, "put_px": put_px, "cost": -(call_px + put_px),
            "margin": margin_yuan,
        }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["quarter_leg_index"] = trades_df.groupby("quarter").cumcount()
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


def _stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "total_pnl": np.nan, "mean_r": np.nan}
    pnl_yuan = trades["pnl"] * ds.MULTIPLIER
    r = pnl_yuan / trades["margin_yuan"]
    return {"n": len(trades), "total_pnl": pnl_yuan.sum(), "mean_r": r.mean()}


def report(res: dict, underlying: str, baselines: dict[str, dict]) -> None:
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        print(f"{underlying}: no trades.")
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

    print(f"{underlying}: monthly short straddle, quarter-anchored strike, "
          f"{equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades(months)={len(trades)}  avg_days_held={trades['days_held'].mean():.0f}  "
          f"re-strikes/quarter=1 (vs. ~3 for plain monthly re-centering)")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin={avg_margin:,.0f} yuan  return_on_margin/yr~={cagr_on_margin:.1%}")

    print("\n  By quarter-leg position (0=fresh ATM at quarter start, 1/2=reused anchor strike):")
    for idx, g in trades.groupby("quarter_leg_index"):
        moneyness = ((g["entry_spot"] - g["strike"]) / g["strike"]).abs()
        print(f"    leg {idx}: n={len(g):>3d}  avg_credit={g['credit'].mean()*ds.MULTIPLIER:>7,.0f} yuan  "
              f"avg_pnl={g['pnl'].mean()*ds.MULTIPLIER:>7,.0f} yuan  "
              f"avg_|moneyness|={moneyness.mean():.1%}  win%={(g['pnl']>0).mean():.0%}")

    disc = trades[trades["entry_date"] < SPLIT_DATE]
    hold_ = trades[trades["entry_date"] >= SPLIT_DATE]
    d_s, h_s = _stats(disc), _stats(hold_)
    print(f"\n  discovery (<{SPLIT_DATE.date()}): n={d_s['n']}  total_pnl={d_s['total_pnl']:,.0f}  mean_r={d_s['mean_r']:+.2%}")
    print(f"  holdout   (>={SPLIT_DATE.date()}): n={h_s['n']}  total_pnl={h_s['total_pnl']:,.0f}  mean_r={h_s['mean_r']:+.2%}")

    print(f"\n  vs. baselines, common window from {trades['entry_date'].min().date()}:")
    d_this, h_this = _stats(trades[trades["entry_date"] < SPLIT_DATE]), \
                     _stats(trades[trades["entry_date"] >= SPLIT_DATE])
    print(f"    quarter-anchored monthly (this):  disc n={d_this['n']:>3d} mean_r={d_this['mean_r']:+7.2%}   "
          f"hold n={h_this['n']:>3d} mean_r={h_this['mean_r']:+7.2%}   total_pnl={trades['pnl'].sum()*ds.MULTIPLIER:>10,.0f}")
    for name, bres in baselines.items():
        bt = bres["trades"]
        if bt.empty:
            continue
        common_start = max(trades["entry_date"].min(), bt["entry_date"].min())
        b = bt[bt["entry_date"] >= common_start]
        b_d, b_h = _stats(b[b["entry_date"] < SPLIT_DATE]), _stats(b[b["entry_date"] >= SPLIT_DATE])
        print(f"    {name:<32s}: disc n={b_d['n']:>3d} mean_r={b_d['mean_r']:+7.2%}   "
              f"hold n={b_h['n']:>3d} mean_r={b_h['mean_r']:+7.2%}   total_pnl={b['pnl'].sum()*ds.MULTIPLIER:>10,.0f}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"monthly_quarter_anchored_straddle_trades_{underlying}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\n  Trades saved to {trades_out}")


def plot(res: dict, underlying: str, out_path: Path) -> None:
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
        is_anchor = t["quarter_leg_index"] == 0
        ax1.axvline(t["entry_date"], color=(C_INK if is_anchor else C_AXIS),
                    linewidth=(1.0 if is_anchor else 0.6), alpha=(0.6 if is_anchor else 0.4))
    ax1.set_ylabel("cumulative P&L (yuan)", color=C_INK2, fontsize=10)
    ax1.set_title(f"{underlying}: monthly short straddle, quarter-anchored strike",
                  color=C_INK, fontsize=12, loc="left", pad=10)

    ax2.fill_between(dd.index, 0, dd.values, color=C_RED, alpha=0.55, linewidth=0)
    ax2.set_ylabel("drawdown (yuan)", color=C_INK2, fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def main(underlying: str, start_date: str | None, make_chart: bool) -> None:
    res = run(underlying, start_date=start_date)
    baselines = {
        "monthly, re-centered ATM": qs.run(underlying, roll_freq="monthly", start_date=start_date),
        "quarterly baseline (validated)": qs.run(underlying, roll_freq="quarterly", start_date=start_date),
    }
    report(res, underlying, baselines)
    if make_chart:
        out_path = OUT_DIR / f"monthly_quarter_anchored_straddle_{underlying}.png"
        plot(res, underlying, out_path)
        print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--start-date", default=None,
                    help="restrict backtest to dates on/after YYYY-MM-DD")
    ap.add_argument("--chart", action="store_true",
                    help="also save the equity/drawdown PNG (only do this once a config is validated)")
    args = ap.parse_args()
    main(args.underlying, args.start_date, args.chart)
