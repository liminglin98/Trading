"""Quarterly ATM short straddle (unchanged: short call+put at ATM on the
nearest quarterly/季月 expiry, held to expiry, settled at intrinsic) paired
with OTM long wings that roll on their OWN monthly cadence instead of
sharing the shorts' quarterly expiry, as in quarterly_iron_butterfly.py.

Motivation: quarterly_iron_butterfly.py buys wings on the same quarterly
contract as the short straddle -- those are ~90d OTM options bought once and
held the whole quarter. Rolling the wings monthly instead means each wing
leg is cheaper per fill (shorter-dated OTM premium decays faster) and
re-centers on spot up to ~3x more often within a quarter. The mechanical
cost: margin.compute_deposit only pairs a short and a long leg that share an
expiry into a cheaper vertical-spread margin. Once the wings sit on a
different (monthly) expiry than the shorts, that pairing is gone for most of
the quarter -- the short legs price as fully naked margin, only getting
spread relief in whichever final month the wing's expiry happens to
coincide with the quarterly one. That's computed correctly here (not
special-cased) because compute_deposit already groups by (expiry, cp).

Wing width is grid-searched the same way as quarterly_iron_butterfly.py
(listed-strike steps from each month's own ATM), in-sample vs out-of-sample
split fixed in advance, then compared against both the naked short straddle
and the original same-expiry iron butterfly.

Usage:
    python quarterly_iron_butterfly_monthly_wings.py --underlying 510050 --split-date 2021-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # archived: options_lab/ modules live one level up

import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qas
import quarterly_iron_butterfly as qib

OUT_DIR = Path(__file__).resolve().parent / "out"


def _mark_leg(day: pd.DataFrame, code: str, cp: str, strike: float,
             spot: float, force: bool) -> float:
    row = day[day[ds.C_CODE] == code]
    if force or row.empty:
        return qas._intrinsic(cp, strike, spot)
    return float(row[ds.C_CLOSE].iloc[0])


def _open_wing(day: pd.DataFrame, dt, spot: float, wing_width: int) -> dict | None:
    mexp = qas._target_expiries(day, "monthly")
    if not mexp:
        return None
    expiry = pd.Timestamp(mexp[0])
    atm_strike = qas._atm_strike(day, expiry, spot)
    if atm_strike is None:
        return None
    strikes = np.array(sorted(day.loc[day[ds.C_EXPIRY] == expiry, ds.C_STRIKE].unique()))
    atm_idx = int(np.argmin(np.abs(strikes - atm_strike)))
    call_wing = qib._wing_strike(strikes, atm_idx, wing_width, "C")
    put_wing = qib._wing_strike(strikes, atm_idx, wing_width, "P")
    cl_row = qas._leg_row(day, expiry, "C", call_wing)
    pl_row = qas._leg_row(day, expiry, "P", put_wing)
    if cl_row is None or pl_row is None:
        return None
    cl_px, pl_px = float(cl_row[ds.C_CLOSE]), float(pl_row[ds.C_CLOSE])
    return {
        "entry_date": pd.Timestamp(dt), "expiry": expiry,
        "cl_strike": call_wing, "pl_strike": put_wing,
        "cl_code": cl_row[ds.C_CODE], "pl_code": pl_row[ds.C_CODE],
        "cl_px": cl_px, "pl_px": pl_px, "cost": cl_px + pl_px,
    }


def run(underlying: str, wing_width: int, start_date: str | None = None) -> dict:
    df = ds.load_chain(underlying)
    df = df[~df[ds.C_NAME].astype(str).str.endswith("A")]
    df = df[df[ds.C_CLOSE] > 0]
    if start_date is not None:
        df = df[df[ds.C_DATE] >= pd.Timestamp(start_date)]
    dates = sorted(df[ds.C_DATE].unique())
    by_date = dict(tuple(df.groupby(ds.C_DATE)))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_short: dict | None = None
    open_wing: dict | None = None
    realized = 0.0

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_short is None:
            qexp = qas._target_expiries(day, "quarterly")
            if not qexp:
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            expiry = pd.Timestamp(qexp[0])
            atm_strike = qas._atm_strike(day, expiry, spot)
            if atm_strike is None:
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            cs_row = qas._leg_row(day, expiry, "C", atm_strike)
            ps_row = qas._leg_row(day, expiry, "P", atm_strike)
            wing = _open_wing(day, dt, spot, wing_width)
            if cs_row is None or ps_row is None or wing is None:
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            cs_px, ps_px = float(cs_row[ds.C_CLOSE]), float(ps_row[ds.C_CLOSE])
            legs_df = pd.DataFrame([
                {"expiry": expiry, "cp": "C", "strike": atm_strike, "direction": -1, "qty": 1, "entry_price": cs_px},
                {"expiry": expiry, "cp": "P", "strike": atm_strike, "direction": -1, "qty": 1, "entry_price": ps_px},
                {"expiry": wing["expiry"], "cp": "C", "strike": wing["cl_strike"], "direction": 1, "qty": 1, "entry_price": wing["cl_px"]},
                {"expiry": wing["expiry"], "cp": "P", "strike": wing["pl_strike"], "direction": 1, "qty": 1, "entry_price": wing["pl_px"]},
            ])
            margin_yuan, _ = mg.compute_deposit(legs_df, spot)
            open_short = {
                "entry_date": pd.Timestamp(dt), "entry_spot": spot, "expiry": expiry,
                "cs_strike": atm_strike, "ps_strike": atm_strike,
                "cs_code": cs_row[ds.C_CODE], "ps_code": ps_row[ds.C_CODE],
                "cs_px": cs_px, "ps_px": ps_px, "cost": -(cs_px + ps_px),
                "margin": margin_yuan, "wing_realized": 0.0, "n_wing_rolls": 1,
            }
            open_wing = wing
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue

        short_codes = [open_short["cs_code"], open_short["ps_code"]]
        short_force = (pd.Timestamp(dt) >= open_short["expiry"] or
                       any(day[day[ds.C_CODE] == c].empty for c in short_codes))
        if short_force:
            short_mark = -(_mark_leg(day, open_short["cs_code"], "C", open_short["cs_strike"], spot, True) +
                          _mark_leg(day, open_short["ps_code"], "P", open_short["ps_strike"], spot, True))
            short_pnl = short_mark - open_short["cost"]
            wing_final_pnl = 0.0
            if open_wing is not None:
                wing_mark = (_mark_leg(day, open_wing["cl_code"], "C", open_wing["cl_strike"], spot, True) +
                           _mark_leg(day, open_wing["pl_code"], "P", open_wing["pl_strike"], spot, True))
                wing_final_pnl = wing_mark - open_wing["cost"]
            wing_total_pnl = open_short["wing_realized"] + wing_final_pnl
            combo_pnl = short_pnl + wing_total_pnl
            realized += combo_pnl
            trades.append({
                "entry_date": open_short["entry_date"], "expiry": open_short["expiry"],
                "exit_date": dt, "atm_strike": open_short["cs_strike"],
                "entry_spot": open_short["entry_spot"], "exit_spot": spot,
                "credit": -open_short["cost"], "short_pnl": short_pnl,
                "wing_pnl": wing_total_pnl, "n_wing_rolls": open_short["n_wing_rolls"],
                "pnl": combo_pnl,
                "days_held": (pd.Timestamp(dt) - open_short["entry_date"]).days,
                "margin_yuan": open_short["margin"],
            })
            open_short = None
            open_wing = None
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue

        if open_wing is not None:
            wing_codes = [open_wing["cl_code"], open_wing["pl_code"]]
            wing_close = (pd.Timestamp(dt) >= open_wing["expiry"] or
                         any(day[day[ds.C_CODE] == c].empty for c in wing_codes))
            if wing_close:
                wing_mark = (_mark_leg(day, open_wing["cl_code"], "C", open_wing["cl_strike"], spot, True) +
                           _mark_leg(day, open_wing["pl_code"], "P", open_wing["pl_strike"], spot, True))
                open_short["wing_realized"] += wing_mark - open_wing["cost"]
                open_wing = None

        if open_wing is None:
            wing = _open_wing(day, dt, spot, wing_width)
            if wing is not None:
                open_wing = wing
                open_short["n_wing_rolls"] += 1

        short_mark = -(_mark_leg(day, open_short["cs_code"], "C", open_short["cs_strike"], spot, False) +
                      _mark_leg(day, open_short["ps_code"], "P", open_short["ps_strike"], spot, False))
        short_open_pnl = short_mark - open_short["cost"]
        wing_open_pnl = 0.0
        if open_wing is not None:
            wing_mark = (_mark_leg(day, open_wing["cl_code"], "C", open_wing["cl_strike"], spot, False) +
                       _mark_leg(day, open_wing["pl_code"], "P", open_wing["pl_strike"], spot, False))
            wing_open_pnl = wing_mark - open_wing["cost"]
        total_open_pnl = short_open_pnl + open_short["wing_realized"] + wing_open_pnl
        equity_rows.append({"date": dt, "equity": realized + total_open_pnl,
                           "open_pnl": total_open_pnl, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


def grid_search(underlying: str, widths: list[int], split_date: str) -> pd.DataFrame:
    rows = []
    for w in widths:
        res = run(underlying, wing_width=w)
        trades = res["trades"]
        ins = qib._subset_stats(trades, None, split_date)
        oos = qib._subset_stats(trades, split_date, None)
        avg_rolls = trades["n_wing_rolls"].mean() if not trades.empty else np.nan
        rows.append({
            "wing_width_steps": w, "avg_wing_rolls/quarter": round(avg_rolls, 2) if avg_rolls == avg_rolls else None,
            "in_n": ins["n"], "in_pnl": ins.get("total_pnl"), "in_win%": ins.get("win_rate"),
            "in_avg_margin": ins.get("avg_margin"), "in_RoM/yr": ins.get("return_on_margin_yr"),
            "in_sharpe": ins.get("sharpe"), "in_maxdd": ins.get("max_drawdown"),
            "oos_n": oos["n"], "oos_pnl": oos.get("total_pnl"), "oos_win%": oos.get("win_rate"),
            "oos_avg_margin": oos.get("avg_margin"), "oos_RoM/yr": oos.get("return_on_margin_yr"),
            "oos_sharpe": oos.get("sharpe"), "oos_maxdd": oos.get("max_drawdown"),
        })
    return pd.DataFrame(rows)


def main(underlying: str, widths: list[int], split_date: str) -> None:
    grid = grid_search(underlying, widths, split_date)
    pd.set_option("display.width", 220)
    print(f"{underlying}: iron butterfly grid search, quarterly short / MONTHLY-rolling wings, "
         f"in-sample < {split_date} <= out-of-sample\n")
    print(grid.to_string(index=False))

    best_row = grid.sort_values("in_RoM/yr", ascending=False).iloc[0]
    best_w = int(best_row["wing_width_steps"])
    print(f"\nBest width by in-sample return-on-margin/yr: {best_w} steps")

    res_best = run(underlying, wing_width=best_w)
    trades_best = res_best["trades"]
    ins_best = qib._subset_stats(trades_best, None, split_date)
    oos_best = qib._subset_stats(trades_best, split_date, None)

    res_naked = qas.run(underlying, fixed_strike=False, roll_freq="quarterly")
    trades_naked = res_naked["trades"]
    ins_naked = qib._subset_stats(trades_naked, None, split_date)
    oos_naked = qib._subset_stats(trades_naked, split_date, None)

    res_qq = qib.run(underlying, wing_width=best_w)
    trades_qq = res_qq["trades"]
    ins_qq = qib._subset_stats(trades_qq, None, split_date)
    oos_qq = qib._subset_stats(trades_qq, split_date, None)

    print(f"\n=== naked short straddle vs quarterly/quarterly IB vs quarterly/MONTHLY-wing IB (width={best_w}) ===")
    print(f"{'':26}{'in-sample':>16}{'out-of-sample':>16}")
    for label, ins_v, oos_v in [
        ("naked      total_pnl", ins_naked["total_pnl"], oos_naked["total_pnl"]),
        ("IB q/q     total_pnl", ins_qq["total_pnl"], oos_qq["total_pnl"]),
        ("IB q/monthly total_pnl", ins_best["total_pnl"], oos_best["total_pnl"]),
        ("naked      avg_margin", ins_naked["avg_margin"], oos_naked["avg_margin"]),
        ("IB q/q     avg_margin", ins_qq["avg_margin"], oos_qq["avg_margin"]),
        ("IB q/monthly avg_margin", ins_best["avg_margin"], oos_best["avg_margin"]),
        ("naked      RoM/yr", ins_naked["return_on_margin_yr"], oos_naked["return_on_margin_yr"]),
        ("IB q/q     RoM/yr", ins_qq["return_on_margin_yr"], oos_qq["return_on_margin_yr"]),
        ("IB q/monthly RoM/yr", ins_best["return_on_margin_yr"], oos_best["return_on_margin_yr"]),
        ("naked      max_dd", ins_naked["max_drawdown"], oos_naked["max_drawdown"]),
        ("IB q/q     max_dd", ins_qq["max_drawdown"], oos_qq["max_drawdown"]),
        ("IB q/monthly max_dd", ins_best["max_drawdown"], oos_best["max_drawdown"]),
    ]:
        print(f"{label:26}{ins_v:>16}{oos_v:>16}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"iron_butterfly_monthlywings_gridsearch_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    trades_out = OUT_DIR / f"iron_butterfly_monthlywings_trades_{underlying}_w{best_w}.csv"
    trades_best.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\nGrid saved to {grid_out}\nBest-width trades saved to {trades_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--widths", default="1,2,3,4,5,6,7,8",
                    help="comma-separated wing widths in listed-strike steps from each month's own ATM")
    ap.add_argument("--split-date", default="2021-01-01",
                    help="in-sample = entries before this date, out-of-sample = on/after")
    args = ap.parse_args()
    widths = [int(x) for x in args.widths.split(",")]
    main(args.underlying, widths, args.split_date)
