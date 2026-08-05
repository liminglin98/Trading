"""Iron butterfly with the CORRECT combo margin formula:
margin = max(call_wing_width, put_wing_width)*mult - total_net_credit*mult
(the call-side and put-side max losses are mutually exclusive at expiry --
you never pay both, so they must NOT be summed like two independent
vertical spreads.)"""
import numpy as np
import pandas as pd
import data_store as ds
import quarterly_atm_short_straddle as qas

MULT = ds.MULTIPLIER

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

    trades = []
    open_pos = None

    def _mark(day, pos, spot, force_intrinsic):
        codes = [pos["cs_code"], pos["ps_code"], pos["cl_code"], pos["pl_code"]]
        rs = [day[day[ds.C_CODE] == c] for c in codes]
        alive = all(not r.empty for r in rs)
        if force_intrinsic or not alive:
            cs = qas._intrinsic("C", pos["cs_strike"], spot)
            ps = qas._intrinsic("P", pos["ps_strike"], spot)
            cl = qas._intrinsic("C", pos["cl_strike"], spot)
            pl = qas._intrinsic("P", pos["pl_strike"], spot)
        else:
            cs, ps, cl, pl = (float(r[ds.C_CLOSE].iloc[0]) for r in rs)
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
                trades.append({
                    "entry_date": open_pos["entry_date"], "expiry": open_pos["expiry"],
                    "exit_date": dt, "pnl": pnl, "credit": -open_pos["cost"],
                    "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                continue
            continue

        qexp = qas._target_expiries(day, "quarterly")
        if not qexp:
            continue
        expiry = pd.Timestamp(qexp[0])
        atm_strike = qas._atm_strike(day, expiry, spot)
        if atm_strike is None:
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
            continue

        cs_px, ps_px = float(cs_row[ds.C_CLOSE]), float(ps_row[ds.C_CLOSE])
        cl_px, pl_px = float(cl_row[ds.C_CLOSE]), float(pl_row[ds.C_CLOSE])

        w_c = (call_wing - atm_strike) * MULT
        w_p = (atm_strike - put_wing) * MULT
        net_credit_total = (cs_px + ps_px - cl_px - pl_px) * MULT
        margin_yuan = max(max(w_c, w_p) - net_credit_total, 0.0)

        open_pos = {
            "entry_date": pd.Timestamp(dt), "expiry": expiry,
            "cs_strike": atm_strike, "ps_strike": atm_strike,
            "cl_strike": call_wing, "pl_strike": put_wing,
            "cs_code": cs_row[ds.C_CODE], "ps_code": ps_row[ds.C_CODE],
            "cl_code": cl_row[ds.C_CODE], "pl_code": pl_row[ds.C_CODE],
            "cost": -(cs_px + ps_px) + (cl_px + pl_px),
            "margin": margin_yuan,
        }

    return pd.DataFrame(trades)

if __name__ == "__main__":
    for w in [1,2,3,4,5,6]:
        t = run("510050", wing_width=w)
        pnl_yuan = t["pnl"]*MULT
        span_years = (t["entry_date"].max()-t["entry_date"].min()).days/365.25
        avg_m = t["margin_yuan"].mean()
        max_m = t["margin_yuan"].max()
        total = pnl_yuan.sum()
        cum = pnl_yuan.cumsum(); dd=(cum-cum.cummax()).min()
        rom = (total/avg_m)/span_years
        print(f"w={w}: n={len(t)} total_pnl={total:,.0f} avg_margin={avg_m:,.0f} max_margin={max_m:,.0f} RoM/yr={rom:.1%} maxdd={dd:,.0f}")
