"""Two cheap, low-risk tweaks to quarterly_atm_short_straddle.py's plain
hold-to-expiry cadence, tested independently (and, for completeness,
crossed):

  * profit target: close the whole position early (at that day's mark) once
    unrealized profit reaches profit_target_pct of the credit received,
    then immediately look to re-establish a fresh ATM straddle at the SAME
    expiry for the remainder of the cycle. This is the mirror image of the
    QVIX-spike stop-loss already tested and REJECTED for this book in
    vol_targeted_short_straddle_backtest.py -- closing a WINNER banks
    realized theta/vega decay instead of locking in a loss on a spike that
    would likely have mean-reverted, so there's no reason to assume it
    fails the same way.

  * laddered entry: instead of opening the full position on a single day,
    spread it over `ladder_days` consecutive trading days (re-selecting
    the ATM strike each day -- generally a different strike as spot
    drifts), so the whole quarter's entry isn't riding on one day's
    price/IV print. All tranches settle together at the shared expiry.

Reuses quarterly_atm_short_straddle._target_expiries/_atm_strike/_leg_row/
_intrinsic unmodified. ladder_days=1 and profit_target_pct=None reproduces
quarterly_atm_short_straddle.py's baseline exactly (sanity check, same
convention as quarterly_restrike_short_straddle.py's --restrike-cadence
none).

A profit-target exit is a pre-committed threshold on CURRENT mark-to-market
P&L, evaluated causally day by day -- not conditioned on the future price
path, so it's not the hindsight-gated overlay pattern flagged elsewhere;
it's the same category of rule as the stop_loss_mult already used in
delta_strangle_short_backtest.py, just the mirror threshold.

Usage:
    python quarterly_atm_straddle_management_backtest.py --underlying 510050 --profit-target 0.5
    python quarterly_atm_straddle_management_backtest.py --underlying 510050 --ladder-days 3
    python quarterly_atm_straddle_management_backtest.py --underlying 510050 --sweep
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
import delta_strangle_short_backtest as dss
import quarterly_restrike_short_straddle as qrs

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = dss.SPLIT_DATE
MIN_CELL_N = 8

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"

LADDER_DAYS_GRID = [1, 2, 3, 5]
PROFIT_TARGETS = [None, 0.25, 0.5, 0.75]


# ------------------------------------------------------------------ engine
def run(underlying: str, roll_freq: str = "quarterly", ladder_days: int = 1,
       profit_target_pct: float | None = None, slippage: float = 0.0,
       start_date: str | None = None) -> dict:
    df = ds.load_chain(underlying)
    df = df[~df[ds.C_NAME].astype(str).str.endswith("A")]  # exclude adjusted
    df = df[df[ds.C_CLOSE] > 0]
    if start_date is not None:
        df = df[df[ds.C_DATE] >= pd.Timestamp(start_date)]
    dates = sorted(df[ds.C_DATE].unique())
    by_date = dict(tuple(df.groupby(ds.C_DATE)))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    cycle: dict | None = None  # {"expiry", "tranches": [...], "pending": int, "margin": float|None}
    realized = 0.0
    # Each tranche below is a full 1-lot straddle -- size_scale rescales the
    # combined position (cost/mark/margin) down to the SAME total capital as
    # a single-day, ladder_days=1 entry, so ladder_days changes only how
    # entry timing is spread, not how much capital/premium is deployed.
    size_scale = 1.0 / ladder_days

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if cycle is not None:
            expiry = cycle["expiry"]
            expired = pd.Timestamp(dt) >= expiry
            tranche_px = []  # (call_px_raw, put_px_raw) per tranche, this day's mark
            total_cost, any_missing = 0.0, False
            for tr in cycle["tranches"]:
                call = day[day[ds.C_CODE] == tr["call_code"]]
                put = day[day[ds.C_CODE] == tr["put_code"]]
                missing = call.empty or put.empty
                any_missing = any_missing or missing
                if expired or missing:
                    call_px = qs._intrinsic("C", tr["strike"], spot)
                    put_px = qs._intrinsic("P", tr["strike"], spot)
                else:
                    call_px = float(call[ds.C_CLOSE].iloc[0])
                    put_px = float(put[ds.C_CLOSE].iloc[0])
                tranche_px.append((call_px, put_px))
                total_cost += tr["cost"]
            total_cost *= size_scale
            total_mark = sum(-(c + p) for c, p in tranche_px) * size_scale

            pnl = total_mark - total_cost
            settle_now = expired or any_missing
            hit_profit = (not settle_now and cycle["pending"] == 0 and
                         profit_target_pct is not None and
                         pnl >= profit_target_pct * abs(total_cost))

            if settle_now or hit_profit:
                if hit_profit:
                    # Active close before expiry -- pay exit slippage on each
                    # leg (buy back at the offer). Settlement (expiry, or a
                    # contract that delisted early) uses the raw/intrinsic
                    # mark already computed above -- no spread to cross.
                    exec_mark = sum(
                        -(qrs._exit_price(c, slippage, at_settlement=False) +
                          qrs._exit_price(p, slippage, at_settlement=False))
                        for c, p in tranche_px
                    ) * size_scale
                else:
                    exec_mark = total_mark
                exec_pnl = exec_mark - total_cost
                realized += exec_pnl
                trades.append({
                    "entry_date": cycle["tranches"][0]["entry_date"], "expiry": expiry,
                    "exit_date": dt, "exit_reason": "expiry" if settle_now else "profit_target",
                    "n_tranches": len(cycle["tranches"]),
                    "entry_spot": cycle["tranches"][0]["entry_spot"], "exit_spot": spot,
                    "credit": -total_cost, "pnl": exec_pnl,
                    "days_held": (pd.Timestamp(dt) - cycle["tranches"][0]["entry_date"]).days,
                    "margin_yuan": cycle["margin"],
                })
                cycle = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue

            if cycle["pending"] > 0:
                strike = qs._atm_strike(day, expiry, spot)
                call_row = qs._leg_row(day, expiry, "C", strike) if strike is not None else None
                put_row = qs._leg_row(day, expiry, "P", strike) if strike is not None else None
                if call_row is not None and put_row is not None:
                    call_px = qrs._entry_price(float(call_row[ds.C_CLOSE]), slippage)
                    put_px = qrs._entry_price(float(put_row[ds.C_CLOSE]), slippage)
                    cycle["tranches"].append({
                        "entry_date": pd.Timestamp(dt), "entry_spot": spot, "strike": strike,
                        "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
                        "call_px": call_px, "put_px": put_px, "cost": -(call_px + put_px),
                    })
                    cycle["pending"] -= 1
                    if cycle["pending"] == 0:
                        legs_df = pd.DataFrame([
                            row for tr in cycle["tranches"] for row in (
                                {"expiry": expiry, "cp": "C", "strike": tr["strike"],
                                 "direction": -1, "qty": 1, "entry_price": tr["call_px"]},
                                {"expiry": expiry, "cp": "P", "strike": tr["strike"],
                                 "direction": -1, "qty": 1, "entry_price": tr["put_px"]},
                            )
                        ])
                        raw_margin, _ = mg.compute_deposit(legs_df, spot)
                        cycle["margin"] = raw_margin * size_scale

            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- no open cycle: start one on the nearest quarterly contract
        qexp = qs._target_expiries(day, roll_freq)
        if not qexp:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue
        expiry = pd.Timestamp(qexp[0])
        strike = qs._atm_strike(day, expiry, spot)
        call_row = qs._leg_row(day, expiry, "C", strike) if strike is not None else None
        put_row = qs._leg_row(day, expiry, "P", strike) if strike is not None else None
        if call_row is None or put_row is None:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue
        call_px = qrs._entry_price(float(call_row[ds.C_CLOSE]), slippage)
        put_px = qrs._entry_price(float(put_row[ds.C_CLOSE]), slippage)
        tranche = {
            "entry_date": pd.Timestamp(dt), "entry_spot": spot, "strike": strike,
            "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
            "call_px": call_px, "put_px": put_px, "cost": -(call_px + put_px),
        }
        pending = max(ladder_days - 1, 0)
        margin_yuan = None
        if pending == 0:
            legs_df = pd.DataFrame([
                {"expiry": expiry, "cp": "C", "strike": strike, "direction": -1,
                 "qty": 1, "entry_price": call_px},
                {"expiry": expiry, "cp": "P", "strike": strike, "direction": -1,
                 "qty": 1, "entry_price": put_px},
            ])
            margin_yuan, _ = mg.compute_deposit(legs_df, spot)
            margin_yuan *= size_scale
        cycle = {"expiry": expiry, "tranches": [tranche], "pending": pending, "margin": margin_yuan}
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


def _cycle_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["expiry", "entry_date", "exit_date", "pnl", "margin_yuan"])
    g = trades.groupby("expiry")
    return g.agg(entry_date=("entry_date", "min"), exit_date=("exit_date", "max"),
                pnl=("pnl", "sum"), margin_yuan=("margin_yuan", "mean")).reset_index()


# ------------------------------------------------------------------ report
def report(res: dict, underlying: str, ladder_days: int, profit_target_pct: float | None) -> None:
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        print(f"{underlying}: no trades.")
        return
    cycles = _cycle_pnl(trades)
    pnl_yuan = cycles["pnl"] * ds.MULTIPLIER
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan
    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    avg_margin = cycles["margin_yuan"].mean()
    total_pnl = pnl_yuan.sum()
    cagr_on_margin = (total_pnl / avg_margin) / span_years if span_years > 0 else np.nan
    avg_reloads = trades.groupby("expiry").size().mean() - 1

    pt_label = "none" if profit_target_pct is None else f"{profit_target_pct:.0%}"
    print(f"{underlying}: quarterly ATM straddle, ladder_days={ladder_days}, profit_target={pt_label}")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_cycles={len(cycles)}  n_trade_rows={len(trades)}  avg_reloads/cycle={avg_reloads:.2f}  "
          f"avg_tranches/entry={trades['n_tranches'].mean():.1f}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate(cycle)={(cycles['pnl']>0).mean():.0%}")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin={avg_margin:,.0f} yuan  return_on_margin/yr~={cagr_on_margin:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"quarterly_atm_straddle_mgmt_trades_{underlying}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\n  Trades saved to {trades_out}")


def plot(res: dict, underlying: str, out_path: Path, ladder_days: int,
        profit_target_pct: float | None) -> None:
    equity, trades = res["equity"], res["trades"]
    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    dd = eq - eq.cummax()
    cycles = _cycle_pnl(trades)

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
    for _, c in cycles.iterrows():
        ax1.axvline(c["entry_date"], color=C_AXIS, linewidth=0.6, alpha=0.5)
    ax1.set_ylabel("cumulative P&L (yuan)", color=C_INK2, fontsize=10)
    pt_label = "none" if profit_target_pct is None else f"{profit_target_pct:.0%}"
    ax1.set_title(f"{underlying}: quarterly ATM straddle, ladder={ladder_days}, profit_target={pt_label}",
                  color=C_INK, fontsize=12, loc="left", pad=10)

    ax2.fill_between(dd.index, 0, dd.values, color=C_RED, alpha=0.55, linewidth=0)
    ax2.set_ylabel("drawdown (yuan)", color=C_INK2, fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ sweep
def _cell_stats(cycles: pd.DataFrame) -> dict:
    if cycles.empty:
        return {"n": 0, "mean_r": np.nan}
    r = cycles["pnl"] * ds.MULTIPLIER / cycles["margin_yuan"]
    return {"n": len(cycles), "mean_r": r.mean()}


def sweep(underlying: str, ladder_days_grid: list[int] | None = None,
         profit_targets: list[float | None] | None = None,
         slippage: float = 0.0) -> pd.DataFrame:
    ladder_days_grid = ladder_days_grid or LADDER_DAYS_GRID
    profit_targets = profit_targets if profit_targets is not None else PROFIT_TARGETS
    rows = []
    for ld in ladder_days_grid:
        for pt in profit_targets:
            res = run(underlying, ladder_days=ld, profit_target_pct=pt, slippage=slippage)
            cycles = _cycle_pnl(res["trades"])
            disc = cycles[cycles["entry_date"] < SPLIT_DATE]
            hold_ = cycles[cycles["entry_date"] >= SPLIT_DATE]
            d_stats, h_stats = _cell_stats(disc), _cell_stats(hold_)
            pt_label = "none" if pt is None else f"{pt:.0%}"
            rows.append({"ladder_days": ld, "profit_target": pt_label,
                        "disc_n": d_stats["n"], "disc_mean_r": d_stats["mean_r"],
                        "hold_n": h_stats["n"], "hold_mean_r": h_stats["mean_r"]})

    grid = pd.DataFrame(rows)
    print(f"{underlying}: ladder x profit-target sweep ({len(grid)} cells, slippage={slippage:.2f}), "
          f"discovery < {SPLIT_DATE.date()} <= holdout, MIN_CELL_N={MIN_CELL_N}")
    for _, row in grid.iterrows():
        flag = "" if row["disc_n"] >= MIN_CELL_N else "  (< MIN_CELL_N, low confidence)"
        print(f"  ladder_days={row['ladder_days']:<2d} profit_target={row['profit_target']:<5s}  "
              f"disc: n={row['disc_n']:>3.0f} mean_r={row['disc_mean_r']:+7.2%}   "
              f"hold: n={row['hold_n']:>3.0f} mean_r={row['hold_mean_r']:+7.2%}{flag}")

    OUT_DIR.mkdir(exist_ok=True)
    slip_suffix = "" if slippage == 0.0 else f"_slip{slippage:.2f}"
    grid_out = OUT_DIR / f"quarterly_atm_straddle_mgmt_sweep_{underlying}{slip_suffix}.csv"
    grid.to_csv(grid_out, index=False)
    print(f"  Full grid saved to {grid_out}\n")
    return grid


# ------------------------------------------------------------------ main
def main(underlying: str, ladder_days: int, profit_target_pct: float | None,
        slippage: float, do_sweep: bool) -> None:
    if do_sweep:
        sweep(underlying, slippage=slippage)
        return
    res = run(underlying, ladder_days=ladder_days, profit_target_pct=profit_target_pct,
             slippage=slippage)
    report(res, underlying, ladder_days, profit_target_pct)
    out_path = OUT_DIR / f"quarterly_atm_straddle_mgmt_{underlying}.png"
    plot(res, underlying, out_path, ladder_days, profit_target_pct)
    print(f"Chart saved to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--ladder-days", type=int, default=1, dest="ladder_days",
                    help="spread entry over this many consecutive trading days (1=lump sum)")
    ap.add_argument("--profit-target", type=float, default=None, dest="profit_target_pct",
                    help="close early once profit reaches this fraction of credit received")
    ap.add_argument("--slippage", type=float, default=0.0,
                    help="adverse half-spread per leg, index points, charged on entries and "
                         "on any profit-target exit (not at expiry settlement)")
    ap.add_argument("--sweep", action="store_true", dest="do_sweep",
                    help="grid over LADDER_DAYS_GRID x PROFIT_TARGETS instead")
    args = ap.parse_args()
    main(args.underlying, args.ladder_days, args.profit_target_pct, args.slippage, args.do_sweep)
