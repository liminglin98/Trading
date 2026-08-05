"""Naked ATM straddle: daily mark-to-market P&L AND daily real combo margin
in one pass, so we can build an account-level equity/margin/risk-ratio time
series (not just per-trade summaries)."""
import pandas as pd
import numpy as np
import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qas

def combo_margin_now(strike, spot, call_mark, put_mark, mult=ds.MULTIPLIER):
    call_m = mg.naked_margin_per_unit('C', strike, spot, call_mark) * mult
    put_m = mg.naked_margin_per_unit('P', strike, spot, put_mark) * mult
    if call_m >= put_m:
        return call_m + put_mark * mult
    return put_m + call_mark * mult

def run(underlying='510050'):
    df = ds.load_chain(underlying)
    df = df[~df[ds.C_NAME].astype(str).str.endswith('A')]
    df = df[df[ds.C_CLOSE] > 0]
    dates = sorted(df[ds.C_DATE].unique())
    by_date = dict(tuple(df.groupby(ds.C_DATE)))

    rows = []
    realized = 0.0
    open_pos = None

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            call = day[day[ds.C_CODE] == open_pos['call_code']]
            put = day[day[ds.C_CODE] == open_pos['put_code']]
            expired = pd.Timestamp(dt) >= open_pos['expiry']
            if expired or call.empty or put.empty:
                call_px = qas._intrinsic('C', open_pos['strike'], spot)
                put_px = qas._intrinsic('P', open_pos['strike'], spot)
                mark = -(call_px + put_px)
                pnl = mark - open_pos['cost']
                realized += pnl
                rows.append({'date': dt, 'equity': realized, 'open_pnl': 0.0, 'margin': 0.0})
                open_pos = None
                continue
            call_px = float(call[ds.C_CLOSE].iloc[0])
            put_px = float(put[ds.C_CLOSE].iloc[0])
            mark = -(call_px + put_px)
            pnl = mark - open_pos['cost']
            m_today = combo_margin_now(open_pos['strike'], spot, call_px, put_px)
            rows.append({'date': dt, 'equity': realized + pnl, 'open_pnl': pnl, 'margin': m_today})
            continue

        qexp = qas._target_expiries(day, 'quarterly')
        if not qexp:
            rows.append({'date': dt, 'equity': realized, 'open_pnl': 0.0, 'margin': 0.0})
            continue
        expiry = pd.Timestamp(qexp[0])
        strike = qas._atm_strike(day, expiry, spot)
        call_row = qas._leg_row(day, expiry, 'C', strike) if strike is not None else None
        put_row = qas._leg_row(day, expiry, 'P', strike) if strike is not None else None
        if call_row is None or put_row is None:
            rows.append({'date': dt, 'equity': realized, 'open_pnl': 0.0, 'margin': 0.0})
            continue
        call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
        entry_margin = combo_margin_now(strike, spot, call_px, put_px)
        open_pos = {
            'entry_date': pd.Timestamp(dt), 'expiry': expiry, 'strike': strike,
            'call_code': call_row[ds.C_CODE], 'put_code': put_row[ds.C_CODE],
            'cost': -(call_px + put_px),
        }
        rows.append({'date': dt, 'equity': realized, 'open_pnl': 0.0, 'margin': entry_margin})

    return pd.DataFrame(rows)

if __name__ == '__main__':
    eq = run('510050')
    eq.to_csv('out/naked_straddle_daily_equity_margin_510050.csv', index=False)
    print(eq.tail())
