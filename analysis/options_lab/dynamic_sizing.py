"""Compounding backtest: re-size lot count at every quarterly roll based on
CURRENT account equity (realized + unrealized), instead of a fixed N for
the whole 11-year history. Target: keep entry-day margin at roughly
target_frac of current equity (with a safety multiplier for naked, since
its margin can spike well above the entry-day snapshot during the trade)."""
import numpy as np
import pandas as pd
import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qas

CAP0 = 500_000
MULT = ds.MULTIPLIER

def combo_margin_now(strike, spot, call_mark, put_mark):
    call_m = mg.naked_margin_per_unit('C', strike, spot, call_mark) * MULT
    put_m = mg.naked_margin_per_unit('P', strike, spot, put_mark) * MULT
    if call_m >= put_m:
        return call_m + put_mark * MULT
    return put_m + call_mark * MULT

def run_naked_dynamic(underlying='510050', target_frac=0.5, safety_mult=2.0, min_lots=1):
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
                pnl_per_lot = (mark - open_pos['cost']) * MULT
                N = open_pos['N']
                realized += pnl_per_lot * N
                rows.append({'date': dt, 'event': 'close', 'N': N,
                            'trade_pnl': pnl_per_lot * N, 'acct_equity': CAP0 + realized,
                            'margin_total': 0.0, 'risk_ratio': 0.0})
                open_pos = None
                continue
            call_px = float(call[ds.C_CLOSE].iloc[0])
            put_px = float(put[ds.C_CLOSE].iloc[0])
            mark = -(call_px + put_px)
            pnl_per_lot = (mark - open_pos['cost']) * MULT
            N = open_pos['N']
            equity_now = CAP0 + realized + pnl_per_lot * N
            m_today = combo_margin_now(open_pos['strike'], spot, call_px, put_px) * N
            rows.append({'date': dt, 'event': 'hold', 'N': N,
                        'trade_pnl': np.nan, 'acct_equity': equity_now,
                        'margin_total': m_today,
                        'risk_ratio': m_today / equity_now if equity_now > 0 else np.nan})
            continue

        qexp = qas._target_expiries(day, 'quarterly')
        if not qexp:
            rows.append({'date': dt, 'event': 'flat', 'N': 0, 'trade_pnl': np.nan,
                        'acct_equity': CAP0 + realized, 'margin_total': 0.0, 'risk_ratio': 0.0})
            continue
        expiry = pd.Timestamp(qexp[0])
        strike = qas._atm_strike(day, expiry, spot)
        call_row = qas._leg_row(day, expiry, 'C', strike) if strike is not None else None
        put_row = qas._leg_row(day, expiry, 'P', strike) if strike is not None else None
        if call_row is None or put_row is None:
            rows.append({'date': dt, 'event': 'flat', 'N': 0, 'trade_pnl': np.nan,
                        'acct_equity': CAP0 + realized, 'margin_total': 0.0, 'risk_ratio': 0.0})
            continue
        call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
        entry_margin = combo_margin_now(strike, spot, call_px, put_px)
        current_equity = CAP0 + realized
        N = max(min_lots, int((current_equity * target_frac) / (entry_margin * safety_mult)))
        open_pos = {
            'entry_date': pd.Timestamp(dt), 'expiry': expiry, 'strike': strike,
            'call_code': call_row[ds.C_CODE], 'put_code': put_row[ds.C_CODE],
            'cost': -(call_px + put_px), 'N': N,
        }
        rows.append({'date': dt, 'event': 'open', 'N': N, 'trade_pnl': np.nan,
                    'acct_equity': current_equity, 'margin_total': entry_margin * N,
                    'risk_ratio': (entry_margin * N) / current_equity})

    return pd.DataFrame(rows)

if __name__ == '__main__':
    res = run_naked_dynamic()
    res.to_csv('out/naked_dynamic_sizing_510050.csv', index=False)
    opens = res[res['event'] == 'open']
    print(opens[['date','N','risk_ratio']].to_string())
    print('\nfinal equity:', CAP0 + res['acct_equity'].iloc[-1])
