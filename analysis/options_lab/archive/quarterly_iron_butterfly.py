"""Rolling ATM iron butterfly (short ATM straddle + OTM long strangle as
insurance) on the exchange's quarterly (季月) contract, 50ETF.

Same roll mechanics as quarterly_atm_short_straddle.py (ATM re-centered
every roll, hold to expiry, settle at intrinsic, roll next day into the new
nearest quarterly contract) but each combo also buys 1x OTM call and 1x OTM
put as protection, turning the undefined-risk straddle into a defined-risk
iron butterfly. Wing width is expressed in "listed strikes away from ATM"
(SSE strike spacing is ~0.05 near the money) since that's what's actually
tradeable, not a fixed yuan distance.

Grid-searches wing width in-sample, evaluates the winning width
out-of-sample, and compares P&L / margin usage against the naked short
straddle baseline over the same two windows.

Usage:
    python quarterly_iron_butterfly.py --underlying 510050 --split-date 2021-01-01
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

OUT_DIR = Path(__file__).resolve().parent / "out"


def _wing_strike(strikes: np.ndarray, atm_idx: int, width: int, side: str) -> float:
    if side == "C":
        idx = min(atm_idx + width, len(strikes) - 1)
    else:
        idx = max(atm_idx - width, 0)
    return float(strikes[idx])


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
    open_pos: dict | None = None
    realized = 0.0

    def _mark(day, pos, spot, force_intrinsic):
        codes = [pos["cs_code"], pos["ps_code"], pos["cl_code"], pos["pl_code"]]
        rows = [day[day[ds.C_CODE] == c] for c in codes]
        alive = all(not r.empty for r in rows)
        if force_intrinsic or not alive:
            cs = qas._intrinsic("C", pos["cs_strike"], spot)
            ps = qas._intrinsic("P", pos["ps_strike"], spot)
            cl = qas._intrinsic("C", pos["cl_strike"], spot)
            pl = qas._intrinsic("P", pos["pl_strike"], spot)
        else:
            cs, ps, cl, pl = (float(r[ds.C_CLOSE].iloc[0]) for r in rows)
        mark = -(cs + ps) + (cl + pl)
        return mark

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            expired = pd.Timestamp(dt) >= open_pos["expiry"]
            codes = [open_pos["cs_code"], open_pos["ps_code"],
                    open_pos["cl_code"], open_pos["pl_code"]]
            missing = any(day[day[ds.C_CODE] == c].empty for c in codes)
            if expired or missing:
                mark = _mark(day, open_pos, spot, force_intrinsic=True)
                pnl = mark - open_pos["cost"]
                realized += pnl
                trades.append({
                    "entry_date": open_pos["entry_date"], "expiry": open_pos["expiry"],
                    "exit_date": dt, "atm_strike": open_pos["cs_strike"],
                    "call_wing": open_pos["cl_strike"], "put_wing": open_pos["pl_strike"],
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "credit": -open_pos["cost"], "pnl": pnl,
                    "days_held": (pd.Timestamp(dt) - open_pos["entry_date"]).days,
                    "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            mark = _mark(day, open_pos, spot, force_intrinsic=False)
            pnl = mark - open_pos["cost"]
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        qexp = qas._target_expiries(day, "quarterly")
        if not qexp:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue
        expiry = pd.Timestamp(qexp[0])
        atm_strike = qas._atm_strike(day, expiry, spot)
        if atm_strike is None:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue
        strikes = np.array(sorted(day.loc[day[ds.C_EXPIRY] == expiry, ds.C_STRIKE].unique()))
        atm_idx = int(np.argmin(np.abs(strikes - atm_strike)))
        call_wing = _wing_strike(strikes, atm_idx, wing_width, "C")
        put_wing = _wing_strike(strikes, atm_idx, wing_width, "P")

        cs_row = qas._leg_row(day, expiry, "C", atm_strike)
        ps_row = qas._leg_row(day, expiry, "P", atm_strike)
        cl_row = qas._leg_row(day, expiry, "C", call_wing)
        pl_row = qas._leg_row(day, expiry, "P", put_wing)
        if any(r is None for r in (cs_row, ps_row, cl_row, pl_row)):
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
            continue

        cs_px, ps_px = float(cs_row[ds.C_CLOSE]), float(ps_row[ds.C_CLOSE])
        cl_px, pl_px = float(cl_row[ds.C_CLOSE]), float(pl_row[ds.C_CLOSE])
        legs_df = pd.DataFrame([
            {"expiry": expiry, "cp": "C", "strike": atm_strike, "direction": -1, "qty": 1, "entry_price": cs_px},
            {"expiry": expiry, "cp": "P", "strike": atm_strike, "direction": -1, "qty": 1, "entry_price": ps_px},
            {"expiry": expiry, "cp": "C", "strike": call_wing, "direction": 1, "qty": 1, "entry_price": cl_px},
            {"expiry": expiry, "cp": "P", "strike": put_wing, "direction": 1, "qty": 1, "entry_price": pl_px},
        ])
        margin_yuan, _ = mg.compute_deposit(legs_df, spot)

        open_pos = {
            "entry_date": pd.Timestamp(dt), "entry_spot": spot, "expiry": expiry,
            "cs_strike": atm_strike, "ps_strike": atm_strike,
            "cl_strike": call_wing, "pl_strike": put_wing,
            "cs_code": cs_row[ds.C_CODE], "ps_code": ps_row[ds.C_CODE],
            "cl_code": cl_row[ds.C_CODE], "pl_code": pl_row[ds.C_CODE],
            "cost": -(cs_px + ps_px) + (cl_px + pl_px),
            "margin": margin_yuan,
        }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ stats
def _subset_stats(trades: pd.DataFrame, lo: str | None, hi: str | None) -> dict:
    t = trades.copy()
    if lo is not None:
        t = t[t["entry_date"] >= pd.Timestamp(lo)]
    if hi is not None:
        t = t[t["entry_date"] < pd.Timestamp(hi)]
    if t.empty:
        return {"n": 0}
    pnl_yuan = t["pnl"] * ds.MULTIPLIER
    span_years = (t["entry_date"].max() - t["entry_date"].min()).days / 365.25
    span_years = max(span_years, 0.25)
    avg_margin = t["margin_yuan"].mean()
    total = pnl_yuan.sum()
    cum = pnl_yuan.cumsum()
    dd = (cum - cum.cummax()).min()
    ann_factor = np.sqrt(4)  # ~4 quarterly trades/year
    sharpe = (pnl_yuan.mean() / pnl_yuan.std() * ann_factor
             if pnl_yuan.std() > 0 else np.nan)
    return {
        "n": len(t), "total_pnl": round(total, 0), "win_rate": round((pnl_yuan > 0).mean(), 3),
        "avg_margin": round(avg_margin, 0),
        "return_on_margin_yr": round((total / avg_margin) / span_years, 4) if avg_margin else np.nan,
        "max_drawdown": round(dd, 0), "sharpe": round(sharpe, 2) if sharpe == sharpe else None,
    }


def grid_search(underlying: str, widths: list[int], split_date: str) -> pd.DataFrame:
    rows = []
    for w in widths:
        res = run(underlying, wing_width=w)
        trades = res["trades"]
        ins = _subset_stats(trades, None, split_date)
        oos = _subset_stats(trades, split_date, None)
        rows.append({
            "wing_width_steps": w,
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
    pd.set_option("display.width", 200)
    print(f"{underlying}: iron butterfly grid search, quarterly roll, "
         f"in-sample < {split_date} <= out-of-sample\n")
    print(grid.to_string(index=False))

    best_row = grid.sort_values("in_RoM/yr", ascending=False).iloc[0]
    best_w = int(best_row["wing_width_steps"])
    print(f"\nBest width by in-sample return-on-margin/yr: {best_w} steps")

    res_best = run(underlying, wing_width=best_w)
    trades_best = res_best["trades"]
    ins_best = _subset_stats(trades_best, None, split_date)
    oos_best = _subset_stats(trades_best, split_date, None)

    res_naked = qas.run(underlying, fixed_strike=False, roll_freq="quarterly")
    trades_naked = res_naked["trades"]
    ins_naked = _subset_stats(trades_naked, None, split_date)
    oos_naked = _subset_stats(trades_naked, split_date, None)

    print(f"\n=== Naked short straddle vs iron butterfly (width={best_w}) ===")
    print(f"{'':22}{'in-sample':>16}{'out-of-sample':>16}")
    for label, naked_v, ib_v in [
        ("naked  total_pnl", ins_naked["total_pnl"], oos_naked["total_pnl"]),
        ("ib     total_pnl", ins_best["total_pnl"], oos_best["total_pnl"]),
        ("naked  avg_margin", ins_naked["avg_margin"], oos_naked["avg_margin"]),
        ("ib     avg_margin", ins_best["avg_margin"], oos_best["avg_margin"]),
        ("naked  RoM/yr", ins_naked["return_on_margin_yr"], oos_naked["return_on_margin_yr"]),
        ("ib     RoM/yr", ins_best["return_on_margin_yr"], oos_best["return_on_margin_yr"]),
        ("naked  max_dd", ins_naked["max_drawdown"], oos_naked["max_drawdown"]),
        ("ib     max_dd", ins_best["max_drawdown"], oos_best["max_drawdown"]),
    ]:
        print(f"{label:22}{naked_v:>16}{ib_v:>16}")

    OUT_DIR.mkdir(exist_ok=True)
    grid_out = OUT_DIR / f"iron_butterfly_gridsearch_{underlying}.csv"
    grid.to_csv(grid_out, index=False)
    trades_out = OUT_DIR / f"iron_butterfly_trades_{underlying}_w{best_w}.csv"
    trades_best.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"\nGrid saved to {grid_out}\nBest-width trades saved to {trades_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--widths", default="1,2,3,4,5,6,7,8",
                    help="comma-separated wing widths in listed-strike steps from ATM")
    ap.add_argument("--split-date", default="2021-01-01",
                    help="in-sample = entries before this date, out-of-sample = on/after")
    args = ap.parse_args()
    widths = [int(x) for x in args.widths.split(",")]
    main(args.underlying, widths, args.split_date)
