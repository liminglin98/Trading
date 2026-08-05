"""Iron butterfly with per-day equity AND per-day margin (constant while a
combo is open) so we can build an account-level risk-ratio time series."""
import numpy as np
import pandas as pd
import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qas

def _wing_strike(strikes, atm_idx, width, side):
    if side == "C":
        idx = min(atm_idx + width, len(strikes) - 1)
    else:
        idx = max(atm_idx - width, 0)
    return float(strikes[idx])

def run(underlying, wing_width):
    df = ds.load_chain(underlying)
    df = df[~df[ds.C_NAME].astype(str).str.endswith("A")]
    df = df[df[ds.C_CLOSE] > 0]
    dates = sorted(df[ds.C_DATE].unique())
    by_date = dict(tuple(df.groupby(ds.C_DATE)))

    equity_rows = []
    open_pos = None
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
        return -(cs + ps) + (cl + pl)

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            expired = pd.Timestamp(dt) >= open_pos["expiry"]
            codes = [open_pos["cs_code"], open_pos["ps_code"], open_pos["cl_code"], open_pos["pl_code"]]
            missing = any(day[day[ds.C_CODE] == c].empty for c in codes)
            if expired or missing:
                mark = _mark(day, open_pos, spot, True)
                pnl = mark - open_pos["cost"]
                realized += pnl
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "margin": 0.0})
                open_pos = None
                continue
            mark = _mark(day, open_pos, spot, False)
            pnl = mark - open_pos["cost"]
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "margin": open_pos["margin"]})
            continue

        qexp = qas._target_expiries(day, "quarterly")
        if not qexp:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "margin": 0.0})
            continue
        expiry = pd.Timestamp(qexp[0])
        atm_strike = qas._atm_strike(day, expiry, spot)
        if atm_strike is None:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "margin": 0.0})
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
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "margin": 0.0})
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
            "entry_date": pd.Timestamp(dt), "expiry": expiry,
            "cs_strike": atm_strike, "ps_strike": atm_strike,
            "cl_strike": call_wing, "pl_strike": put_wing,
            "cs_code": cs_row[ds.C_CODE], "ps_code": ps_row[ds.C_CODE],
            "cl_code": cl_row[ds.C_CODE], "pl_code": pl_row[ds.C_CODE],
            "cost": -(cs_px + ps_px) + (cl_px + pl_px),
            "margin": margin_yuan,
        }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "margin": margin_yuan})

    return pd.DataFrame(equity_rows)

if __name__ == "__main__":
    eq = run("510050", wing_width=2)
    eq.to_csv("out/ib_w2_daily_equity_margin_510050.csv", index=False)
    print(eq.tail())
