"""Alert configs + pure evaluation logic shared by app.py's Alerts tab and
push_alerts.py (the unattended WeChat/PushPlus push script). Deliberately
streamlit-free -- app.py wraps the "now"/trigger functions in st.cache_data
itself for UI responsiveness, push_alerts.py calls them straight since it
only runs once per invocation.

All configs and thresholds here are copied verbatim from app.py's former
inline definitions -- see git history / the module docstrings on the
individual backtest scripts (long_straddle_rv_contraction_backtest.py,
iv_low_long_straddle_backtest.py, vol_rich_short_straddle_backtest.py,
delta_strangle_short_backtest.py, iv_spike_short_straddle_backtest.py,
iv_pct_vrp_combo_short_straddle_backtest.py) for how each threshold was
derived. Do not edit thresholds here without also updating those sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import consolidated_store as cs
import data_store as ds
import margin as mg
import quarterly_atm_short_straddle as qs
import delta_strangle_short_backtest as dss
import vol_rich_short_straddle_backtest as _vrs_mod

# Canonical stop-loss multiple for every short-straddle-family alert (short
# straddle, delta strangle, both combo variants) -- imported rather than
# copied so this stays in sync if the backtest's own constant ever changes.
# The combo backtest files each define their own local copy of the same
# value (confirmed 2.0 in all of them) rather than importing this one.
STOP_LOSS_MULT = _vrs_mod.STOP_LOSS_MULT

# --------------------------------------------------- SSE Composite MA150
def ma150_state(s: pd.Series) -> dict | None:
    """Current status of the SSE Composite vs. its 150-day MA and how many
    trading days it's been on that side -- the signal from the MA-crossover
    investigation (analysis/macro_model/ma_signal_investigation_leaderboard.md)."""
    if len(s) < 150:
        return None
    ma = s.rolling(150).mean()
    valid = ma.dropna()
    state = s.loc[valid.index] > valid
    last_state = bool(state.iloc[-1])
    run = 0
    for v in state.iloc[::-1]:
        if bool(v) == last_state:
            run += 1
        else:
            break
    last_px, last_ma = float(s.iloc[-1]), float(valid.iloc[-1])
    return dict(below=not last_state, run_days=run, last_px=last_px, last_ma=last_ma,
                dist_pct=last_px / last_ma - 1, asof=s.index[-1])


def next_quarterly_listing_estimate(asof: pd.Timestamp) -> pd.Timestamp:
    """Rough estimate only. 50ETF/300ETF quarterly-cycle contracts (Mar/Jun/
    Sep/Dec expiries) are listed at the start of each calendar quarter
    (Jan/Apr/Jul/Oct, observed on the 22nd-29th across 12 historical
    listings checked 2023-2026) -- see the listing-cycle finding in
    ma_signal_investigation_leaderboard.md."""
    for m in (1, 4, 7, 10):
        candidate = pd.Timestamp(year=asof.year, month=m, day=25)
        if candidate > asof:
            return candidate
    return pd.Timestamp(year=asof.year + 1, month=1, day=25)


def suggest_put_debit_spread(chain: pd.DataFrame, wing_pct: float = 0.05,
                              target_dte: int = 170, min_acceptable_dte: int = 140) -> dict | None:
    """Live suggested combination from the leaderboard leader: ATM put + a
    wing_pct-OTM short put, same expiry, DTE targeted to the ~120-trading-day
    horizon. Reads the most recent chain snapshot only -- descriptive, not
    an order."""
    if chain.empty:
        return None
    last_date = chain[ds.C_DATE].max()
    day = chain[(chain[ds.C_DATE] == last_date) & (chain[ds.C_CP] == "P")]
    if day.empty:
        return None
    u = float(day[ds.C_UND_CLOSE].iloc[0])
    atm_strike = day.iloc[(day[ds.C_STRIKE] - u).abs().argsort()].iloc[0][ds.C_STRIKE]
    same_strike = day[day[ds.C_STRIKE] == atm_strike]
    ge = same_strike[same_strike[ds.C_DTE] >= target_dte]
    pool = ge if not ge.empty else same_strike
    atm = pool.iloc[(pool[ds.C_DTE] - target_dte).abs().argsort()].iloc[0]
    if atm[ds.C_DTE] < min_acceptable_dte:
        return dict(
            insufficient_dte=True, best_available_dte=int(atm[ds.C_DTE]), asof=last_date,
            available_dtes=sorted(int(x) for x in day[ds.C_DTE].unique()),
            next_quarterly_listing_est=next_quarterly_listing_estimate(last_date),
        )

    same_exp = day[day[ds.C_EXPIRY] == atm[ds.C_EXPIRY]]
    wing_target = u * (1 - wing_pct)
    wing = same_exp.iloc[(same_exp[ds.C_STRIKE] - wing_target).abs().argsort()].iloc[0]
    if wing[ds.C_STRIKE] == atm[ds.C_STRIKE]:
        return None
    net_debit = float(atm[ds.C_CLOSE]) - float(wing[ds.C_CLOSE])
    if net_debit <= 0:
        return None
    return dict(asof=last_date, underlying_close=u, expiry=atm[ds.C_EXPIRY], dte=int(atm[ds.C_DTE]),
                buy_strike=float(atm[ds.C_STRIKE]), buy_price=float(atm[ds.C_CLOSE]),
                sell_strike=float(wing[ds.C_STRIKE]), sell_price=float(wing[ds.C_CLOSE]),
                net_debit=net_debit, stop_gain_price=net_debit * 2)


# --------------------------------------------------- vol-signal alerts (long straddle)
# Frozen picks from the discovery(<2024-01-01)/holdout(>=2024-01-01) sweeps
# (long_straddle_rv_contraction_backtest.py --sweep / iv_low_long_straddle_backtest.py
# --sweep). vrp failed on every underlying tested and isn't shown.
#
# Look-ahead robustness check (2026-08-06): both scripts gate entry on the
# SAME trading date's signal value (see their --lag-signal-days flag / module
# docstrings). Reran all 6 picks below with --lag-signal-days 1 (yesterday's
# value instead, matching the stricter convention used elsewhere in this repo)
# -- 5/6 hold up (same sign, discovery AND holdout, under both conventions).
# The one exception: 159901/iv_pct's discovery mean_r flips sign (+5.2% at
# lag0 -> -2.6% at lag1) -- it was already the weakest/most tightly-exited
# pick of the six (max_hold=5, profit_target=1.0) even before this check;
# holdout stays positive and gets stronger under lag1 (+3.2% -> +9.4%), so
# not a clean reject, but treat this one specifically with more skepticism
# than the other 5 until it's re-swept under the lag=1 convention directly.
VOL_SIGNAL_ALERTS = [
    {"u": "510500", "name": "500ETF", "module": "rv", "signal": "rv_pct", "threshold": 0.30,
     "max_hold": 10, "profit_target": 2.0},
    {"u": "159915", "name": "创业板ETF", "module": "rv", "signal": "rv_pct", "threshold": 0.20,
     "max_hold": 10, "profit_target": 2.0},
    {"u": "159901", "name": "深证100ETF", "module": "rv", "signal": "rv_pct", "threshold": 0.30,
     "max_hold": 10, "profit_target": 2.0},
    {"u": "159915", "name": "创业板ETF", "module": "rv", "signal": "iv_pct", "threshold": 0.20,
     "max_hold": 10, "profit_target": 0.5},
    {"u": "159901", "name": "深证100ETF", "module": "rv", "signal": "iv_pct", "threshold": 0.30,
     "max_hold": 5, "profit_target": 1.0},  # <- weakest pick, see look-ahead note above
    {"u": "510300", "name": "300ETF", "module": "iv_low", "signal": "iv_pct", "threshold": 0.20,
     "max_hold": 30, "profit_target": 2.0},
]


def vol_signal_now(u: str, module: str, signal: str) -> dict | None:
    """Most recent value of one of the tested long-straddle entry signals."""
    if module == "rv":
        import long_straddle_rv_contraction_backtest as lsb
        s = lsb._signal_series(u, signal)
    else:
        import iv_low_long_straddle_backtest as ivb
        s = ivb._signal_series(u, signal)
    if s.empty:
        return None
    return {"value": float(s.iloc[-1]), "asof": pd.Timestamp(s.index[-1])}


def pct_trigger_level(raw: pd.Series, threshold_pct: float, window: int = 252) -> float | None:
    """Approximate the raw (non-percentile) signal level that would put
    today's rolling percentile rank at threshold_pct."""
    s = raw.dropna()
    if len(s) < 60:
        return None
    return float(s.iloc[-window:].quantile(threshold_pct))


def vol_trigger_level(u: str, module: str, signal: str, threshold: float) -> dict | None:
    """Current raw value and the approximate raw level needed to flip a
    percentile-based long-straddle alert (VOL_SIGNAL_ALERTS) active."""
    if module == "rv":
        import long_straddle_rv_contraction_backtest as lsb
        raw = lsb._trailing_rv_series(u) if signal == "rv_pct" else cs.atm_iv_series(u)
    else:
        import iv_low_long_straddle_backtest as ivb
        raw = bt._atm_iv_series(ivb._prepped_chain(u))
    raw = raw.dropna()
    if raw.empty:
        return None
    level = pct_trigger_level(raw, threshold)
    if level is None:
        return None
    return {"current": float(raw.iloc[-1]), "level": level}


def suggest_atm_straddle(chain: pd.DataFrame, dte_min: int = 10, dte_max: int = 45) -> dict | None:
    """Live ATM long straddle (buy call + buy put, same strike/expiry) from
    the most recent chain snapshot -- descriptive, not an order."""
    if chain.empty:
        return None
    last_date = chain[ds.C_DATE].max()
    day = chain[chain[ds.C_DATE] == last_date]
    if day.empty or day[ds.C_UND_CLOSE].isna().all():
        return None
    spot = float(day[ds.C_UND_CLOSE].iloc[0])
    expiries = sorted(day[ds.C_EXPIRY].unique())
    target = [e for e in expiries if dte_min <= (pd.Timestamp(e) - last_date).days <= dte_max]
    if not target:
        return None
    expiry = pd.Timestamp(target[0])
    strike = qs._atm_strike(day, expiry, spot)
    if strike is None:
        return None
    call_row = qs._leg_row(day, expiry, "C", strike)
    put_row = qs._leg_row(day, expiry, "P", strike)
    if call_row is None or put_row is None:
        return None
    call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
    return dict(asof=last_date, spot=spot, expiry=expiry, dte=int((expiry - last_date).days),
                strike=strike, call_price=call_px, put_price=put_px,
                debit=call_px + put_px,
                call_code=str(call_row[ds.C_CODE]), put_code=str(put_row[ds.C_CODE]))


# --------------------------------------------------- vol-signal alerts (short straddle)
# Frozen picks from vol_rich_short_straddle_backtest.py --sweep. rv_pct alone
# failed validation on both 50ETF and 300ETF and isn't shown.
SHORT_VOL_SIGNAL_ALERTS = [
    {"u": "510050", "name": "50ETF", "signal": "vrp", "threshold": 0.06, "max_hold": 20},
    {"u": "510300", "name": "300ETF", "signal": "vrp", "threshold": 0.03, "max_hold": 20},
]


def short_vol_signal_now(u: str, signal: str) -> dict | None:
    """Most recent value of one of the tested short-straddle entry signals
    (vol_rich_short_straddle_backtest.py)."""
    import vol_rich_short_straddle_backtest as vrs
    s = vrs._signal_series(u, signal)
    if s.empty:
        return None
    return {"value": float(s.iloc[-1]), "asof": pd.Timestamp(s.index[-1])}


def vrp_trigger_level(u: str, threshold: float) -> dict | None:
    """IV/RV decomposition behind the vrp short-straddle signal (vrp = IV -
    trailing RV), so we can show how much ATM IV would need to reach to
    flip the alert active."""
    import iv_low_long_straddle_backtest as ivb
    iv_atm = bt._atm_iv_series(ivb._prepped_chain(u))
    rv = ivb._trailing_rv_series(u)
    if iv_atm.empty or rv.empty:
        return None
    return {"iv_now": float(iv_atm.iloc[-1]), "rv_now": float(rv.iloc[-1]),
            "needed": threshold + float(rv.iloc[-1])}


def suggest_atm_short_straddle(chain: pd.DataFrame, dte_min: int = 10, dte_max: int = 45) -> dict | None:
    """Live ATM short straddle (sell call + sell put, same strike/expiry)
    from the most recent chain snapshot -- descriptive, not an order."""
    if chain.empty:
        return None
    last_date = chain[ds.C_DATE].max()
    day = chain[chain[ds.C_DATE] == last_date]
    if day.empty or day[ds.C_UND_CLOSE].isna().all():
        return None
    spot = float(day[ds.C_UND_CLOSE].iloc[0])
    expiries = sorted(day[ds.C_EXPIRY].unique())
    target = [e for e in expiries if dte_min <= (pd.Timestamp(e) - last_date).days <= dte_max]
    if not target:
        return None
    expiry = pd.Timestamp(target[0])
    strike = qs._atm_strike(day, expiry, spot)
    if strike is None:
        return None
    call_row = qs._leg_row(day, expiry, "C", strike)
    put_row = qs._leg_row(day, expiry, "P", strike)
    if call_row is None or put_row is None:
        return None
    call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
    legs_df = pd.DataFrame([
        {"expiry": expiry, "cp": "C", "strike": strike, "direction": -1, "qty": 1, "entry_price": call_px},
        {"expiry": expiry, "cp": "P", "strike": strike, "direction": -1, "qty": 1, "entry_price": put_px},
    ])
    margin_yuan, _ = mg.compute_deposit(legs_df, spot)
    return dict(asof=last_date, spot=spot, expiry=expiry, dte=int((expiry - last_date).days),
                strike=strike, call_price=call_px, put_price=put_px,
                credit=call_px + put_px, margin_yuan=margin_yuan,
                call_code=str(call_row[ds.C_CODE]), put_code=str(put_row[ds.C_CODE]))


# --------------------------------------------------- always-on quarterly-roll strategies
# Not signal-gated like everything else in this module: quarterly_atm_short_straddle.py
# and archive/quarterly_iron_butterfly.py always hold exactly one position per
# underlying (sell the current 季月/quarterly ATM straddle, +wings for the
# butterfly), rolling into the next quarterly contract the trading day after
# expiry, no stop-loss/profit-target. So there's no "active"/"near-trigger"
# concept here -- just "what's the current cycle's position, and its P&L."
def _current_quarterly_cycle(chain: pd.DataFrame) -> dict | None:
    """The not-yet-expired 季月 (quarterly, expiry month in {3,6,9,12))
    expiry nearest today, and the entry_date of the position on it -- the
    first trading day after the PREVIOUS quarterly expiry (or the start of
    the chain's history, if this is the first quarterly cycle on file),
    matching quarterly_atm_short_straddle.py's roll mechanics."""
    if chain.empty:
        return None
    last_date = chain[ds.C_DATE].max()
    all_expiries = sorted(pd.Timestamp(e) for e in chain[ds.C_EXPIRY].unique())
    q_expiries = [e for e in all_expiries if e.month in (3, 6, 9, 12)]
    future = [e for e in q_expiries if e > last_date]
    if not future:
        return None
    current_expiry = future[0]
    past = [e for e in q_expiries if e <= last_date]
    all_dates = sorted(pd.Timestamp(d) for d in chain[ds.C_DATE].unique())
    if past:
        prev_expiry = max(past)
        after = [d for d in all_dates if d > prev_expiry]
        entry_date = after[0] if after else all_dates[0]
    else:
        entry_date = all_dates[0]
    return dict(current_expiry=current_expiry, entry_date=entry_date, last_date=last_date)


def suggest_quarterly_atm_short_straddle(chain: pd.DataFrame) -> dict | None:
    """Live construction for the always-on quarterly_atm_short_straddle.py
    strategy: sell 1x ATM call + 1x ATM put on the current quarterly
    contract, entered per _current_quarterly_cycle."""
    cycle = _current_quarterly_cycle(chain)
    if cycle is None:
        return None
    entry_day = chain[chain[ds.C_DATE] == cycle["entry_date"]]
    if entry_day.empty or entry_day[ds.C_UND_CLOSE].isna().all():
        return None
    spot = float(entry_day[ds.C_UND_CLOSE].iloc[0])
    expiry = cycle["current_expiry"]
    strike = qs._atm_strike(entry_day, expiry, spot)
    if strike is None:
        return None
    call_row = qs._leg_row(entry_day, expiry, "C", strike)
    put_row = qs._leg_row(entry_day, expiry, "P", strike)
    if call_row is None or put_row is None:
        return None
    call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
    return dict(entry_date=cycle["entry_date"], expiry=expiry, strike=strike,
                call_price=call_px, put_price=put_px, credit=call_px + put_px)


def suggest_quarterly_iron_butterfly(chain: pd.DataFrame, wing_width: int = 2) -> dict | None:
    """Live construction for the always-on archive/quarterly_iron_butterfly.py
    strategy (archived but not itself rejected -- the sibling monthly-wings
    variant was; see project_iron_butterfly_monthly_wings_rejected memory):
    same roll mechanics as the naked quarterly straddle, plus 1x OTM long
    call + 1x OTM long put wing_width listed strikes away from ATM, same
    (quarterly) expiry as the shorts -- what gives it the margin relief the
    monthly-wing variant lost. wing_width=2 is the validated in-sample AND
    OOS optimum for 510050 (archive/out/iron_butterfly_gridsearch_510050.csv);
    not separately validated for any other underlying."""
    cycle = _current_quarterly_cycle(chain)
    if cycle is None:
        return None
    entry_day = chain[chain[ds.C_DATE] == cycle["entry_date"]]
    if entry_day.empty or entry_day[ds.C_UND_CLOSE].isna().all():
        return None
    spot = float(entry_day[ds.C_UND_CLOSE].iloc[0])
    expiry = cycle["current_expiry"]
    strikes = np.sort(entry_day.loc[entry_day[ds.C_EXPIRY] == expiry, ds.C_STRIKE].unique())
    if len(strikes) == 0:
        return None
    atm_strike = qs._atm_strike(entry_day, expiry, spot)
    if atm_strike is None:
        return None
    atm_idx = int(np.argmin(np.abs(strikes - atm_strike)))
    call_wing_strike = float(strikes[min(atm_idx + wing_width, len(strikes) - 1)])
    put_wing_strike = float(strikes[max(atm_idx - wing_width, 0)])
    cs_row = qs._leg_row(entry_day, expiry, "C", atm_strike)
    ps_row = qs._leg_row(entry_day, expiry, "P", atm_strike)
    cl_row = qs._leg_row(entry_day, expiry, "C", call_wing_strike)
    pl_row = qs._leg_row(entry_day, expiry, "P", put_wing_strike)
    if any(r is None for r in (cs_row, ps_row, cl_row, pl_row)):
        return None
    cs_px, ps_px, cl_px, pl_px = (float(r[ds.C_CLOSE]) for r in (cs_row, ps_row, cl_row, pl_row))
    return dict(entry_date=cycle["entry_date"], expiry=expiry, atm_strike=atm_strike,
                call_wing_strike=call_wing_strike, put_wing_strike=put_wing_strike,
                short_call_price=cs_px, short_put_price=ps_px,
                long_call_price=cl_px, long_put_price=pl_px,
                credit=(cs_px + ps_px) - (cl_px + pl_px))


def quarterly_straddle_position(u: str, name: str, chain_loader=None) -> dict | None:
    """Current open position for the always-on quarterly ATM short
    straddle -- exactly one per underlying, no signal gate, held to the
    current quarterly expiry with no stop-loss/profit-target (matches the
    strategy's own exit discipline: settle at intrinsic, roll next day)."""
    loader = chain_loader or ds.load_chain
    chain = loader(u)
    constr = suggest_quarterly_atm_short_straddle(chain)
    if constr is None:
        return None
    legs = [{"expiry": constr["expiry"], "strike": constr["strike"], "cp": "C"},
            {"expiry": constr["expiry"], "strike": constr["strike"], "cp": "P"}]
    prices = reprice_legs(chain, legs)
    if prices is None:
        return None
    pos = _finalize_position("short", -constr["credit"], -sum(prices), constr["entry_date"],
                              False, constr["expiry"])
    return {"name": name, **pos, "construction": constr}


def quarterly_iron_butterfly_position(u: str, name: str, wing_width: int = 2,
                                       chain_loader=None) -> dict | None:
    """Current open position for the always-on quarterly iron butterfly
    (wing_width validated for 510050 only -- see
    suggest_quarterly_iron_butterfly)."""
    loader = chain_loader or ds.load_chain
    chain = loader(u)
    constr = suggest_quarterly_iron_butterfly(chain, wing_width)
    if constr is None:
        return None
    legs = [{"expiry": constr["expiry"], "strike": constr["atm_strike"], "cp": "C"},
            {"expiry": constr["expiry"], "strike": constr["atm_strike"], "cp": "P"},
            {"expiry": constr["expiry"], "strike": constr["call_wing_strike"], "cp": "C"},
            {"expiry": constr["expiry"], "strike": constr["put_wing_strike"], "cp": "P"}]
    prices = reprice_legs(chain, legs)
    if prices is None:
        return None
    mark = -(prices[0] + prices[1]) + (prices[2] + prices[3])
    pos = _finalize_position("short", -constr["credit"], mark, constr["entry_date"],
                              False, constr["expiry"])
    return {"name": name, **pos, "construction": constr}


# --------------------------------------------------- vol-signal alerts (delta-targeted strangle, short)
# Preliminary/exploratory (delta_strangle_short_backtest.py, 2026-07-28).
# Same vrp entry signal as SHORT_VOL_SIGNAL_ALERTS above, only the strike
# selection differs (delta-targeted strangle instead of ATM straddle).
DELTA_STRANGLE_ALERTS = [
    {"u": "510050", "name": "50ETF", "threshold": 0.06, "max_hold": 20, "target_delta": 0.30},
    {"u": "510300", "name": "300ETF", "threshold": 0.03, "max_hold": 20, "target_delta": 0.30},
]

# --------------------------------------------------- vol-signal alerts (delta-targeted strangle, iv_pct)
# Preliminary/exploratory (delta_strangle_short_backtest.py --signal iv_pct, 2026-08-07).
# Same iv_pct entry signal as IV_PCT_SHORT_ALERTS below, only the strike
# selection differs (delta-targeted strangle instead of ATM straddle). Both
# thresholds inherited unchanged from IV_PCT_SHORT_ALERTS -- only target_delta
# was newly swept for this construction (0.30 won on both underlyings, same
# pick as the vrp delta-strangle above).
DELTA_STRANGLE_IV_PCT_ALERTS = [
    {"u": "510050", "name": "50ETF", "module": "iv_low", "threshold": 0.90, "max_hold": 20, "target_delta": 0.30},
    {"u": "510300", "name": "300ETF", "module": "iv_low", "threshold": 0.80, "max_hold": 20, "target_delta": 0.30},
]


def suggest_delta_strangle(chain: pd.DataFrame, target_delta: float,
                            dte_min: int = 10, dte_max: int = 45) -> dict | None:
    """Live delta-targeted short strangle (sell call near +target_delta, sell
    put near -target_delta, independently-chosen strikes) from the most
    recent chain snapshot -- descriptive, not an order."""
    if chain.empty:
        return None
    last_date = chain[ds.C_DATE].max()
    day = chain[chain[ds.C_DATE] == last_date]
    if day.empty or day[ds.C_UND_CLOSE].isna().all():
        return None
    spot = float(day[ds.C_UND_CLOSE].iloc[0])
    expiries = sorted(day[ds.C_EXPIRY].unique())
    target = [e for e in expiries if dte_min <= (pd.Timestamp(e) - last_date).days <= dte_max]
    if not target:
        return None
    expiry = pd.Timestamp(target[0])
    call_row = dss._delta_leg(day, expiry, "C", target_delta)
    put_row = dss._delta_leg(day, expiry, "P", target_delta)
    if call_row is None or put_row is None:
        return None
    call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
    call_strike, put_strike = float(call_row[ds.C_STRIKE]), float(put_row[ds.C_STRIKE])
    legs_df = pd.DataFrame([
        {"expiry": expiry, "cp": "C", "strike": call_strike, "direction": -1, "qty": 1, "entry_price": call_px},
        {"expiry": expiry, "cp": "P", "strike": put_strike, "direction": -1, "qty": 1, "entry_price": put_px},
    ])
    margin_yuan, _ = mg.compute_deposit(legs_df, spot)
    return dict(asof=last_date, spot=spot, expiry=expiry, dte=int((expiry - last_date).days),
                call_strike=call_strike, put_strike=put_strike,
                call_delta=float(call_row[ds.C_DELTA]), put_delta=float(put_row[ds.C_DELTA]),
                call_price=call_px, put_price=put_px,
                credit=call_px + put_px, margin_yuan=margin_yuan)


# --------------------------------------------------- vol-signal alerts (iv_pct, short straddle)
# rolling percentile of raw ATM IV vs its own history (high side) --
# distinct from vrp. Reuses vol_trigger_level (only the comparison
# direction differs: >= here, <= for the long-straddle side).
IV_PCT_SHORT_ALERTS = [
    {"u": "510050", "name": "50ETF", "module": "iv_low", "threshold": 0.90, "max_hold": 20},
    {"u": "510300", "name": "300ETF", "module": "iv_low", "threshold": 0.80, "max_hold": 20},
]


# --------------------------------------------------- vol-signal alerts (spike+vrp combo, short straddle)
# Frozen picks from iv_spike_vrp_combo_short_straddle_backtest.py --sweep.
COMBO_VOL_SIGNAL_ALERTS = [
    {"u": "510050", "name": "50ETF", "spike_window": 10, "spike_pct": 0.15, "vrp_threshold": 0.03, "max_hold": 20},
    {"u": "510300", "name": "300ETF", "spike_window": 10, "spike_pct": 0.15, "vrp_threshold": 0.03, "max_hold": 20},
]


# --------------------------------------------------- vol-signal alerts (iv_pct + vrp AND-combo, short straddle)
# Preliminary/exploratory (iv_pct_vrp_combo_short_straddle_backtest.py,
# 2026-08-04) -- requires BOTH the iv_pct-rich AND vrp-rich signals at once.
# 300ETF sample too thin to freeze a pick -- not included below.
IV_PCT_VRP_COMBO_ALERTS = [
    {"u": "510050", "name": "50ETF", "iv_pct_threshold": 0.90, "vrp_threshold": 0.03, "max_hold": 20},
]


# --------------------------------------------------- vol-signal alerts (gamma flip / put-call ratio, short straddle)
# Preliminary/exploratory (gamma_flip.py / put_call_ratio.py, 2026-08-07) --
# each signal only validated on ONE of its eligible underlyings; the other
# didn't corroborate (flip_dist tested 510050/510300 only; pc_vol_pct
# tested all 7, only 510300 was clean) -- see
# project_short_straddle_vol_signals.md. Reuses short_vol_signal_now (same
# _signal_series dispatch as SHORT_VOL_SIGNAL_ALERTS above) -- no new
# evaluation function needed, just a different signal name/threshold/
# underlying per entry.
GAMMA_FLIP_PC_RATIO_ALERTS = [
    {"u": "510050", "name": "50ETF", "signal": "flip_dist", "threshold": 0.03, "max_hold": 20},
    {"u": "510300", "name": "300ETF", "signal": "pc_vol_pct", "threshold": 0.95, "max_hold": 10},
]


def combo_signal_now(u: str, spike_window: int, spike_pct: float, vrp_threshold: float,
                      chain_loader=None) -> dict | None:
    """Most recent state of the combined IV-spike + VRP short-straddle
    trigger (iv_spike_vrp_combo_short_straddle_backtest.py): a same-day ATM
    IV spike (today's IV vs. spike_window trading days ago) AND yesterday's
    VRP level, both required.

    Anchors "today" on the latest date that actually HAS an IV reading, not
    the chain's raw latest date -- official SSE Greeks/IV routinely land a
    trading day (or more) behind price data right after the close."""
    import iv_spike_short_straddle_backtest as spk
    import iv_low_long_straddle_backtest as il
    chain = (chain_loader or ds.load_chain)(u)
    if chain.empty:
        return None
    dates = [pd.Timestamp(d) for d in sorted(chain[ds.C_DATE].unique())]
    if len(dates) <= spike_window + 1:
        return None

    iv_atm = spk.iv_atm_series(u)
    vrp = il.vrp_trailing_series(u)

    chain_latest = dates[-1]
    iv_valid = [d for d in dates if pd.notna(iv_atm.get(d))]
    if len(iv_valid) <= spike_window:
        return None
    today = iv_valid[-1]
    idx = dates.index(today)
    if idx < spike_window:
        return None
    base_dt = dates[idx - spike_window]
    yst = dates[idx - 1] if idx >= 1 else None

    iv_now, iv_base = iv_atm.get(today), iv_atm.get(base_dt)
    if iv_now is None or iv_base is None or pd.isna(iv_now) or pd.isna(iv_base):
        return None
    iv_now, iv_base = float(iv_now), float(iv_base)
    spike_needed = iv_base * (1 + spike_pct)
    ok_spike = iv_now >= spike_needed

    rv_series = il._trailing_rv_series(u)
    iv_yst_raw = iv_atm.get(yst) if yst is not None else None
    iv_yst = float(iv_yst_raw) if iv_yst_raw is not None and iv_yst_raw == iv_yst_raw else None
    rv_yst_raw = rv_series.get(yst) if yst is not None else None
    rv_yst = float(rv_yst_raw) if rv_yst_raw is not None and rv_yst_raw == rv_yst_raw else None
    vrp_yst = vrp.get(yst) if yst is not None else None
    vrp_yst = float(vrp_yst) if vrp_yst is not None and vrp_yst == vrp_yst else None
    ok_vrp = vrp_yst is not None and vrp_yst >= vrp_threshold
    iv_needed_yst = (vrp_threshold + rv_yst) if rv_yst is not None else None

    return dict(asof=today, yst=yst, stale=bool(today < chain_latest), chain_latest=chain_latest,
                iv_now=iv_now, iv_base=iv_base, spike_needed=spike_needed, ok_spike=ok_spike,
                iv_yst=iv_yst, rv_yst=rv_yst, vrp_yst=vrp_yst, iv_needed_yst=iv_needed_yst,
                ok_vrp=ok_vrp, active=bool(ok_spike and ok_vrp))


# --------------------------------------------------- Alerts tab: side/strategy menu constants
ALERTS_SIDE_BUY = "Buy side (pay debit / long premium)"
ALERTS_SIDE_SELL = "Sell side (collect credit / short premium)"
ALERTS_BUY_STRATEGIES = [
    "SSE Composite MA150 (single indicator)",
    "Long straddle — cheap vol: rv_pct / iv_pct (single indicator)",
]
ALERTS_SELL_STRATEGIES = [
    "Short straddle — VRP rich vol (single indicator)",
    "Short straddle — iv_pct rich vol (single indicator)",
    "Delta-targeted strangle — VRP (single indicator, exploratory construction)",
    "IV spike + VRP combo (multiple indicators)",
    "iv_pct + VRP AND-combo (multiple indicators, exploratory)",
    "Gamma flip / put-call ratio (single indicator, exploratory, single-underlying)",
    "Delta-targeted strangle — iv_pct (single indicator, exploratory construction)",
]


def evaluate_summary(idx_sh_series: pd.Series, chain_loader=None) -> list[dict]:
    """Every tracked alert's active/inactive state at a glance -- the exact
    logic app.py's Alerts tab Summary section renders, factored out so
    push_alerts.py can consume the same evaluation without going through
    Streamlit. Each row: strategy, instrument, active, status, side
    ("buy"/"sell"), strategy_idx (index into ALERTS_BUY/SELL_STRATEGIES)."""
    rows: list[dict] = []

    ma_state = ma150_state(idx_sh_series)
    if ma_state is not None:
        active = ma_state["below"]
        status = f"{ma_state['run_days']}d below MA150" if active else "above MA150"
    else:
        active, status = False, "not enough history"
    rows.append(dict(strategy="SSE Composite MA150", instrument="SSE Composite",
                      active=active, status=status, side="buy", strategy_idx=0))

    for cfg in VOL_SIGNAL_ALERTS:
        sig_now = vol_signal_now(cfg["u"], cfg["module"], cfg["signal"])
        active = bool(sig_now) and sig_now["value"] <= cfg["threshold"]
        status = f"{sig_now['value']:.0%} (fires <= {cfg['threshold']:.0%})" if sig_now else "n/a"
        rows.append(dict(strategy=f"Long straddle ({cfg['signal']})", instrument=f"{cfg['name']} ({cfg['u']})",
                          active=active, status=status, side="buy", strategy_idx=1, cfg=cfg))

    for cfg in SHORT_VOL_SIGNAL_ALERTS:
        sig_now = short_vol_signal_now(cfg["u"], cfg["signal"])
        active = bool(sig_now) and sig_now["value"] >= cfg["threshold"]
        status = f"{sig_now['value']:+.1%} (fires >= {cfg['threshold']:+.0%})" if sig_now else "n/a"
        rows.append(dict(strategy=f"Short straddle ({cfg['signal']})", instrument=f"{cfg['name']} ({cfg['u']})",
                          active=active, status=status, side="sell", strategy_idx=0, cfg=cfg))

    for cfg in IV_PCT_SHORT_ALERTS:
        sig_now = short_vol_signal_now(cfg["u"], "iv_pct")
        active = bool(sig_now) and sig_now["value"] >= cfg["threshold"]
        status = f"{sig_now['value']:.0%} (fires >= {cfg['threshold']:.0%})" if sig_now else "n/a"
        rows.append(dict(strategy="Short straddle (iv_pct)", instrument=f"{cfg['name']} ({cfg['u']})",
                          active=active, status=status, side="sell", strategy_idx=1, cfg=cfg))

    for cfg in DELTA_STRANGLE_ALERTS:
        sig_now = short_vol_signal_now(cfg["u"], "vrp")
        active = bool(sig_now) and sig_now["value"] >= cfg["threshold"]
        status = f"{sig_now['value']:+.1%} (fires >= {cfg['threshold']:+.0%})" if sig_now else "n/a"
        rows.append(dict(strategy=f"Delta strangle (Δ={cfg['target_delta']:.2f})",
                          instrument=f"{cfg['name']} ({cfg['u']})",
                          active=active, status=status, side="sell", strategy_idx=2, cfg=cfg))

    for cfg in COMBO_VOL_SIGNAL_ALERTS:
        c_now = combo_signal_now(cfg["u"], cfg["spike_window"], cfg["spike_pct"], cfg["vrp_threshold"],
                                  chain_loader=chain_loader)
        if c_now is None:
            active, status = False, "n/a"
        else:
            active = c_now["active"]
            status = ("both conditions met" if active else
                      "spike only" if c_now["ok_spike"] else
                      "vrp only" if c_now["ok_vrp"] else "neither met")
        rows.append(dict(strategy="IV spike + VRP combo", instrument=f"{cfg['name']} ({cfg['u']})",
                          active=active, status=status, side="sell", strategy_idx=3, cfg=cfg))

    for cfg in IV_PCT_VRP_COMBO_ALERTS:
        iv_now = short_vol_signal_now(cfg["u"], "iv_pct")
        vrp_now = short_vol_signal_now(cfg["u"], "vrp")
        ok_iv = bool(iv_now) and iv_now["value"] >= cfg["iv_pct_threshold"]
        ok_vrp = bool(vrp_now) and vrp_now["value"] >= cfg["vrp_threshold"]
        active = ok_iv and ok_vrp
        if not iv_now or not vrp_now:
            status = "n/a"
        else:
            status = ("both conditions met" if active else
                      "iv_pct only" if ok_iv else
                      "vrp only" if ok_vrp else "neither met")
        rows.append(dict(strategy="iv_pct + VRP AND-combo", instrument=f"{cfg['name']} ({cfg['u']})",
                          active=active, status=status, side="sell", strategy_idx=4, cfg=cfg))

    for cfg in GAMMA_FLIP_PC_RATIO_ALERTS:
        sig_now = short_vol_signal_now(cfg["u"], cfg["signal"])
        active = bool(sig_now) and sig_now["value"] >= cfg["threshold"]
        status = f"{sig_now['value']:+.1%} (fires >= {cfg['threshold']:+.0%})" if sig_now else "n/a"
        rows.append(dict(strategy=f"Gamma flip / PC ratio ({cfg['signal']})",
                          instrument=f"{cfg['name']} ({cfg['u']})",
                          active=active, status=status, side="sell", strategy_idx=5, cfg=cfg))

    for cfg in DELTA_STRANGLE_IV_PCT_ALERTS:
        sig_now = short_vol_signal_now(cfg["u"], "iv_pct")
        active = bool(sig_now) and sig_now["value"] >= cfg["threshold"]
        status = f"{sig_now['value']:.0%} (fires >= {cfg['threshold']:.0%})" if sig_now else "n/a"
        rows.append(dict(strategy=f"Delta strangle iv_pct (Δ={cfg['target_delta']:.2f})",
                          instrument=f"{cfg['name']} ({cfg['u']})",
                          active=active, status=status, side="sell", strategy_idx=6, cfg=cfg))

    return rows


# =============================================================== push_alerts.py support
# Everything below is used only by push_alerts.py (the unattended WeChat
# push) -- app.py's UI doesn't currently show a P&L reconstruction or a
# near-trigger proximity list, so none of this is wired into evaluate_summary
# above. Kept in this module anyway (not push_alerts.py) so it stays
# alongside the exit-rule/config source of truth it depends on.

ALERT_LABELS_ZH = {
    ("buy", 0): "认沽借方价差（买入平值认沽+卖出低行权价认沽）",
    ("buy", 1): "买入跨式组合（同行权价买入认购+认沽）",
    ("sell", 0): "卖出跨式组合（同行权价卖出认购+认沽）",
    ("sell", 1): "卖出跨式组合（同行权价卖出认购+认沽，iv_pct信号）",
    ("sell", 2): "卖出宽跨式组合（不同行权价卖出认购+认沽，Δ≈0.30）",
    ("sell", 3): "卖出跨式组合（IV跳升+VRP双重信号）",
    ("sell", 4): "卖出跨式组合（iv_pct+VRP双重信号）",
    ("sell", 5): "卖出跨式组合（Gamma翻转点/认沽认购比信号，探索性）",
}


# A strategy can pyramid into multiple concurrent positions if it
# re-triggers while an earlier one from the same row is still open (hasn't
# hit its own exit rule yet) -- capped so a signal that fires constantly
# doesn't accumulate an unbounded number of tracked positions.
MAX_CONCURRENT_POSITIONS = 3


def _all_runs_from_bool_series(bool_series: pd.Series) -> list[dict]:
    """Every contiguous True run in bool_series, oldest to newest. Each:
    {"entry_date":..., "end_date":..., "run_days":...}. A signal that
    triggered a position and has since reverted below threshold still has
    a real, trackable position -- it's the position's own exit rule that
    closes it, not the raw signal -- so callers must check each run's
    end_date against today themselves (position_detail does, via each
    branch's own end_date/last_date check)."""
    s = bool_series.dropna()
    vals = list(s.values)
    runs = []
    i, n = 0, len(vals)
    while i < n:
        if bool(vals[i]):
            start = i
            while i < n and bool(vals[i]):
                i += 1
            end = i - 1
            runs.append({"entry_date": s.index[start], "end_date": s.index[end],
                        "run_days": end - start + 1})
        else:
            i += 1
    return runs


def _capped_entries(bool_series: pd.Series, max_hold: int | None,
                     max_concurrent: int = MAX_CONCURRENT_POSITIONS) -> list[dict]:
    """The most recent up-to-max_concurrent runs from
    _all_runs_from_bool_series, each independently capped at max_hold
    trading days: a raw signal can stay true longer than a strategy's own
    max_hold (e.g. vrp staying rich for 40 days when max_hold=20) -- per
    the backtest's own exit discipline, a hypothetical position opened on
    day 1 of that run would already have been closed by day max_hold.
    Rather than reporting a stale P&L against a trade that would already be
    over, this re-anchors that run's entry_date at max_hold trading days
    before its END and flags max_hold_reached. Oldest to newest, same order
    as _all_runs_from_bool_series."""
    runs = _all_runs_from_bool_series(bool_series)
    if not runs:
        return []
    s = bool_series.dropna()
    out = []
    for run in runs[-max_concurrent:]:
        if max_hold is not None and run["run_days"] > max_hold:
            end_pos = s.index.get_loc(run["end_date"])
            out.append({"entry_date": s.index[end_pos - max_hold + 1], "run_days": max_hold,
                        "max_hold_reached": True})
        else:
            out.append({"entry_date": run["entry_date"], "run_days": run["run_days"],
                        "max_hold_reached": False})
    return out


def vol_signal_series(u: str, module: str, signal: str) -> pd.Series:
    """Full history behind vol_signal_now -- needed to find when the current
    active run started, not just today's value."""
    if module == "rv":
        import long_straddle_rv_contraction_backtest as lsb
        return lsb._signal_series(u, signal)
    import iv_low_long_straddle_backtest as ivb
    return ivb._signal_series(u, signal)


def short_vol_signal_series(u: str, signal: str) -> pd.Series:
    """Full history behind short_vol_signal_now."""
    import vol_rich_short_straddle_backtest as vrs
    return vrs._signal_series(u, signal)


def combo_signal_series(u: str, spike_window: int, spike_pct: float, vrp_threshold: float,
                         chain_loader=None) -> pd.Series:
    """Full boolean history of the combo_signal_now trigger (today's ATM IV
    spike AND yesterday's VRP) -- reuses the exact per-date logic
    combo_signal_now applies to just the latest date, looped over every date
    that has a valid IV reading instead of only the last one."""
    import iv_spike_short_straddle_backtest as spk
    import iv_low_long_straddle_backtest as il
    chain = (chain_loader or ds.load_chain)(u)
    if chain.empty:
        return pd.Series(dtype=bool)
    dates = [pd.Timestamp(d) for d in sorted(chain[ds.C_DATE].unique())]
    if len(dates) <= spike_window + 1:
        return pd.Series(dtype=bool)

    iv_atm = spk.iv_atm_series(u)
    vrp = il.vrp_trailing_series(u)
    iv_valid = [d for d in dates if pd.notna(iv_atm.get(d))]

    out = {}
    for today in iv_valid:
        idx = dates.index(today)
        if idx < spike_window or idx < 1:
            continue
        base_dt, yst = dates[idx - spike_window], dates[idx - 1]
        iv_now, iv_base = iv_atm.get(today), iv_atm.get(base_dt)
        if iv_now is None or iv_base is None or pd.isna(iv_now) or pd.isna(iv_base):
            continue
        ok_spike = float(iv_now) >= float(iv_base) * (1 + spike_pct)
        vrp_yst = vrp.get(yst)
        ok_vrp = vrp_yst is not None and pd.notna(vrp_yst) and float(vrp_yst) >= vrp_threshold
        out[today] = bool(ok_spike and ok_vrp)
    return pd.Series(out).sort_index()


def iv_pct_vrp_combo_series(u: str, iv_pct_threshold: float, vrp_threshold: float) -> pd.Series:
    """Full boolean history of the iv_pct+vrp AND-combo trigger."""
    iv_s = short_vol_signal_series(u, "iv_pct").rename("iv")
    vrp_s = short_vol_signal_series(u, "vrp").rename("vrp")
    df = pd.concat([iv_s, vrp_s], axis=1).dropna()
    return (df["iv"] >= iv_pct_threshold) & (df["vrp"] >= vrp_threshold)


def reprice_legs(chain: pd.DataFrame, legs: list[dict]) -> list[float] | None:
    """Look up each leg's (expiry, strike, cp) close price on chain's latest
    date. legs: list of {"expiry":..., "strike":..., "cp": "C"|"P"}. Falls
    back to intrinsic value (qs._intrinsic) if the chain's latest date is at
    or past that leg's expiry -- same settlement convention every backtest
    run() uses for expired contracts.

    Also FLOORS any live quote at intrinsic value: far-dated/low-volume
    contracts occasionally carry a stale/missing close of 0.0 in the raw
    data even while deep ITM (confirmed on a real 300ETF put debit spread
    leg this session -- 0.0 print on a strike sitting well past intrinsic).
    An option can't rationally trade below intrinsic, so this floor is a
    correctness fix, not a smoothing choice."""
    if chain.empty:
        return None
    last_date = chain[ds.C_DATE].max()
    day = chain[chain[ds.C_DATE] == last_date]
    if day.empty:
        return None
    prices = []
    for leg in legs:
        if last_date >= leg["expiry"]:
            spot_col = day[ds.C_UND_CLOSE].dropna()
            if spot_col.empty:
                return None
            prices.append(qs._intrinsic(leg["cp"], leg["strike"], float(spot_col.iloc[0])))
            continue
        row = day[(day[ds.C_EXPIRY] == leg["expiry"]) & (day[ds.C_STRIKE] == leg["strike"])
                   & (day[ds.C_CP] == leg["cp"])]
        if row.empty:
            return None
        px = float(row.iloc[0][ds.C_CLOSE])
        spot_val = row.iloc[0][ds.C_UND_CLOSE]
        if pd.notna(spot_val):
            px = max(px, qs._intrinsic(leg["cp"], leg["strike"], float(spot_val)))
        prices.append(px)
    return prices


def _finalize_position(kind: str, cost: float, mark: float, entry_date: pd.Timestamp,
                        max_hold_reached: bool, end_date: pd.Timestamp,
                        stop_loss_mult: float | None = None,
                        profit_target_mult: float | None = None) -> dict:
    """cost/mark follow the same sign convention the backtests themselves use
    (and positions.py already displays): long = cost positive (debit paid),
    mark = current value (>=0). Short = cost negative (credit received),
    mark = negative cost-to-close. pnl = mark - cost in both cases; pnl_pct
    is pnl over abs(cost) (the debit for long, the credit for short) -- the
    exact basis each backtest's own profit_target/stop_loss check uses."""
    basis = abs(cost)
    pnl = mark - cost
    pnl_pct = pnl / basis if basis else 0.0
    out = dict(kind=kind, entry_date=entry_date, cost=cost, mark=mark,
               pnl_pct=pnl_pct, pnl_yuan=pnl * ds.MULTIPLIER,
               max_hold_reached=max_hold_reached, end_date=end_date)
    if stop_loss_mult is not None:
        out["stop_loss_pct"] = -stop_loss_mult
        out["stop_loss_yuan"] = -stop_loss_mult * basis * ds.MULTIPLIER
    if profit_target_mult is not None:
        out["profit_target_pct"] = profit_target_mult
        out["profit_target_yuan"] = profit_target_mult * basis * ds.MULTIPLIER
    return out


def _finalize_short_atm_straddle(cfg: dict, bool_s: pd.Series, chain_loader) -> list[dict] | None:
    """Shared tail for every alert family whose suggested construction is a
    plain ATM short straddle (SHORT_VOL_SIGNAL_ALERTS, IV_PCT_SHORT_ALERTS,
    COMBO_VOL_SIGNAL_ALERTS, IV_PCT_VRP_COMBO_ALERTS) -- only the entry
    condition (bool_s) differs between them. Up to MAX_CONCURRENT_POSITIONS
    positions if the signal re-triggered while an earlier one was still
    open (see _capped_entries)."""
    runs = _capped_entries(bool_s, cfg["max_hold"])
    if not runs:
        return None
    chain = chain_loader(cfg["u"])
    chain_last = chain[ds.C_DATE].max()
    out = []
    for run in runs:
        constr = suggest_atm_short_straddle(chain[chain[ds.C_DATE] <= run["entry_date"]])
        if constr is None:
            continue
        legs = [{"expiry": constr["expiry"], "strike": constr["strike"], "cp": "C"},
                {"expiry": constr["expiry"], "strike": constr["strike"], "cp": "P"}]
        prices = reprice_legs(chain, legs)
        if prices is None:
            continue
        end_date = min(constr["expiry"], run["entry_date"] + pd.tseries.offsets.BDay(cfg["max_hold"]))
        if chain_last >= end_date:
            continue  # already past its own exit rule -- closed, nothing to track
        pos = _finalize_position("short", -constr["credit"], -sum(prices), run["entry_date"],
                                  run["max_hold_reached"], end_date, stop_loss_mult=STOP_LOSS_MULT)
        out.append({"name": f"{cfg['name']} ({cfg['u']})", **pos, "construction": constr})
    return out or None


def position_detail(row: dict, chain_loader=None, idx_sh_series: pd.Series | None = None) -> list[dict] | None:
    """Reconstruct the hypothetical position from the alert row's most
    recent trigger (from evaluate_summary): entry construction (via the
    matching suggest_* fn, evaluated on the chain as of that date), reprice
    the SAME legs at today's chain, and report P&L against the strategy's
    own real exit rule (see the exit-rule table in the project plan /
    docstrings on the individual backtest modules -- not invented here).
    Works whether the row is currently active (today is inside the trigger
    run) or not -- a signal that triggered a position and has since
    reverted below threshold still has a real open position until its own
    exit rule (stop-loss/profit-target/max-hold/expiry), not the raw
    signal, closes it; each branch below checks the reconstructed end_date
    against today and drops it once that's passed. Up to
    MAX_CONCURRENT_POSITIONS positions per underlying if the signal
    re-triggered while an earlier one was still open (see
    _capped_entries) -- so MA150 (two underlyings) can return up to
    2*MAX_CONCURRENT_POSITIONS, everything else up to
    MAX_CONCURRENT_POSITIONS. None if there's no open position to report
    (never triggered, already closed, or couldn't be built -- e.g. chain
    unavailable, no listed contract at entry)."""
    loader = chain_loader or ds.load_chain
    side, idx, cfg = row["side"], row["strategy_idx"], row.get("cfg")

    if side == "buy" and idx == 0:  # SSE Composite MA150 -> put debit spread
        if idx_sh_series is None:
            return None
        ma = idx_sh_series.rolling(150).mean().dropna()
        below = ~(idx_sh_series.loc[ma.index] > ma)
        runs = _capped_entries(below, max_hold=None)
        if not runs:
            return None
        out = []
        for u, name in (("510050", "50ETF"), ("510300", "300ETF")):
            chain = loader(u)
            chain_last = chain[ds.C_DATE].max()
            for run in runs:
                constr = suggest_put_debit_spread(chain[chain[ds.C_DATE] <= run["entry_date"]])
                if constr is None or constr.get("insufficient_dte"):
                    continue
                legs = [{"expiry": constr["expiry"], "strike": constr["buy_strike"], "cp": "P"},
                        {"expiry": constr["expiry"], "strike": constr["sell_strike"], "cp": "P"}]
                prices = reprice_legs(chain, legs)
                if prices is None:
                    continue
                if chain_last >= constr["expiry"]:
                    continue  # already past expiry -- closed, nothing to track
                mark = prices[0] - prices[1]
                # leaderboard's own stop-gain is +100% of debit (stop_gain_price
                # = net_debit * 2 in suggest_put_debit_spread) -> profit_target_mult=1.0
                pos = _finalize_position("long", constr["net_debit"], mark, run["entry_date"],
                                          run["max_hold_reached"], constr["expiry"], profit_target_mult=1.0)
                out.append({"name": name, **pos, "construction": constr})
        return out or None

    if side == "buy" and idx == 1:  # long straddle (rv_pct/iv_pct)
        series = vol_signal_series(cfg["u"], cfg["module"], cfg["signal"])
        runs = _capped_entries(series <= cfg["threshold"], cfg["max_hold"])
        if not runs:
            return None
        chain = loader(cfg["u"])
        chain_last = chain[ds.C_DATE].max()
        out = []
        for run in runs:
            constr = suggest_atm_straddle(chain[chain[ds.C_DATE] <= run["entry_date"]])
            if constr is None:
                continue
            legs = [{"expiry": constr["expiry"], "strike": constr["strike"], "cp": "C"},
                    {"expiry": constr["expiry"], "strike": constr["strike"], "cp": "P"}]
            prices = reprice_legs(chain, legs)
            if prices is None:
                continue
            end_date = min(constr["expiry"], run["entry_date"] + pd.tseries.offsets.BDay(cfg["max_hold"]))
            if chain_last >= end_date:
                continue  # already past its own exit rule -- closed, nothing to track
            pos = _finalize_position("long", constr["debit"], sum(prices), run["entry_date"],
                                      run["max_hold_reached"], end_date, profit_target_mult=cfg["profit_target"])
            out.append({"name": f"{cfg['name']} ({cfg['u']})", **pos, "construction": constr})
        return out or None

    if side == "sell" and idx in (0, 1):  # short straddle: vrp / iv_pct
        signal = cfg["signal"] if idx == 0 else "iv_pct"
        bool_s = short_vol_signal_series(cfg["u"], signal) >= cfg["threshold"]
        return _finalize_short_atm_straddle(cfg, bool_s, loader)

    if side == "sell" and idx == 2:  # delta-targeted short strangle
        bool_s = short_vol_signal_series(cfg["u"], "vrp") >= cfg["threshold"]
        runs = _capped_entries(bool_s, cfg["max_hold"])
        if not runs:
            return None
        chain = loader(cfg["u"])
        chain_last = chain[ds.C_DATE].max()
        out = []
        for run in runs:
            constr = suggest_delta_strangle(chain[chain[ds.C_DATE] <= run["entry_date"]], cfg["target_delta"])
            if constr is None:
                continue
            legs = [{"expiry": constr["expiry"], "strike": constr["call_strike"], "cp": "C"},
                    {"expiry": constr["expiry"], "strike": constr["put_strike"], "cp": "P"}]
            prices = reprice_legs(chain, legs)
            if prices is None:
                continue
            end_date = min(constr["expiry"], run["entry_date"] + pd.tseries.offsets.BDay(cfg["max_hold"]))
            if chain_last >= end_date:
                continue  # already past its own exit rule -- closed, nothing to track
            pos = _finalize_position("short", -constr["credit"], -sum(prices), run["entry_date"],
                                      run["max_hold_reached"], end_date, stop_loss_mult=STOP_LOSS_MULT)
            out.append({"name": f"{cfg['name']} ({cfg['u']})", **pos, "construction": constr})
        return out or None

    if side == "sell" and idx == 3:  # IV spike + VRP combo
        bool_s = combo_signal_series(cfg["u"], cfg["spike_window"], cfg["spike_pct"],
                                      cfg["vrp_threshold"], chain_loader=loader)
        return _finalize_short_atm_straddle(cfg, bool_s, loader)

    if side == "sell" and idx == 4:  # iv_pct + VRP AND-combo
        bool_s = iv_pct_vrp_combo_series(cfg["u"], cfg["iv_pct_threshold"], cfg["vrp_threshold"])
        return _finalize_short_atm_straddle(cfg, bool_s, loader)

    return None


# Proximity bands for "close to triggering" -- purely a notification
# threshold to decide what's worth including in the daily push, not a
# backtested/validated parameter. Percentile-based signals (rv_pct/iv_pct)
# are compared in percentile-point terms; vrp (a raw vol-point spread) in
# vol-point terms.
NEAR_BAND_PCT = 0.10
NEAR_BAND_VOL = 0.02


def _trigger_math(row: dict) -> dict | None:
    """For an INACTIVE alert row, return current/threshold/raw-level detail
    unconditionally (no proximity filtering) if the underlying signal data
    is available, else None. MA150 is skipped (a binary above/below state
    has no natural 'distance' the way a percentile or vol-point spread
    does). The sell/idx==3 branch's ok_spike/ok_vrp check is NOT a
    proximity band -- it's a logical necessity (the VRP leg is frozen from
    yesterday's close and cannot be satisfied by intraday IV movement), so
    it stays enforced here rather than in the near-band wrapper below.

    Callers: near_trigger_detail() (below) re-applies NEAR_BAND_* on top of
    this for the daily-digest "close to triggering" list; intraday_monitor's
    build_watchlist() uses full_trigger_detail() to skip that filtering
    entirely, since it wants to track every live-trackable alert, not just
    ones already close at the prior close."""
    side, idx, cfg = row["side"], row["strategy_idx"], row.get("cfg")

    if side == "buy" and idx == 0:
        return None

    if side == "buy" and idx == 1:
        sig_now = vol_signal_now(cfg["u"], cfg["module"], cfg["signal"])
        trig = vol_trigger_level(cfg["u"], cfg["module"], cfg["signal"], cfg["threshold"])
        if not sig_now or not trig:
            return None
        return dict(current_pct=sig_now["value"], threshold_pct=cfg["threshold"],
                    current_raw=trig["current"], trigger_raw=trig["level"])

    if side == "sell" and idx in (0, 1, 2):
        signal = "iv_pct" if idx == 1 else "vrp"
        sig_now = short_vol_signal_now(cfg["u"], signal)
        if not sig_now:
            return None
        if signal == "vrp":
            trig = vrp_trigger_level(cfg["u"], cfg["threshold"])
            if not trig:
                return None
            return dict(current_vrp=sig_now["value"], threshold_vrp=cfg["threshold"],
                        current_raw=trig["iv_now"], trigger_raw=trig["needed"])
        trig = vol_trigger_level(cfg["u"], "iv_low", "iv_pct", cfg["threshold"])
        if not trig:
            return None
        return dict(current_pct=sig_now["value"], threshold_pct=cfg["threshold"],
                    current_raw=trig["current"], trigger_raw=trig["level"])

    if side == "sell" and idx == 3:
        c_now = combo_signal_now(cfg["u"], cfg["spike_window"], cfg["spike_pct"], cfg["vrp_threshold"])
        if not c_now or (not c_now["ok_spike"] and not c_now["ok_vrp"]):
            return None
        return dict(ok_spike=c_now["ok_spike"], ok_vrp=c_now["ok_vrp"],
                    iv_now=c_now["iv_now"], spike_needed=c_now["spike_needed"],
                    vrp_yst=c_now["vrp_yst"], iv_needed_yst=c_now["iv_needed_yst"])

    if side == "sell" and idx == 4:
        iv_now = short_vol_signal_now(cfg["u"], "iv_pct")
        vrp_now = short_vol_signal_now(cfg["u"], "vrp")
        if not iv_now or not vrp_now:
            return None
        return dict(iv_pct_now=iv_now["value"], iv_pct_threshold=cfg["iv_pct_threshold"],
                    vrp_now=vrp_now["value"], vrp_threshold=cfg["vrp_threshold"])

    return None


def full_trigger_detail(row: dict) -> dict | None:
    """Same as near_trigger_detail(), but without the NEAR_BAND_* proximity
    filter -- every alert with a valid computed trigger level is returned,
    however far from threshold it currently sits. Used by
    intraday_monitor.build_watchlist() so a signal that moves from far to
    triggered entirely within one session (as 510300's iv_pct alert did on
    2026-08-19) still gets tracked live, instead of only being caught by the
    next morning's daily digest. See NEAR_BAND_PCT/NEAR_BAND_VOL comment
    above: that band exists purely to keep the daily push concise, not to
    decide what's worth tracking intraday."""
    return _trigger_math(row)


def near_trigger_detail(row: dict) -> dict | None:
    """For an INACTIVE alert row, return current/threshold/raw-level detail
    if it's within NEAR_BAND_* of triggering, else None. Used by
    push_alerts.py's daily digest to keep the "close to triggering" section
    short -- see full_trigger_detail() for the unfiltered version."""
    side, idx = row["side"], row["strategy_idx"]
    detail = _trigger_math(row)
    if detail is None:
        return None

    if side == "buy" and idx == 1:
        if detail["current_pct"] > detail["threshold_pct"] + NEAR_BAND_PCT:
            return None
        return detail

    if side == "sell" and idx in (0, 1, 2):
        if "current_vrp" in detail:
            if detail["current_vrp"] < detail["threshold_vrp"] - NEAR_BAND_VOL:
                return None
            return detail
        if detail["current_pct"] < detail["threshold_pct"] - NEAR_BAND_PCT:
            return None
        return detail

    if side == "sell" and idx == 3:
        return detail

    if side == "sell" and idx == 4:
        ok_iv = detail["iv_pct_now"] >= detail["iv_pct_threshold"] - NEAR_BAND_PCT
        ok_vrp = detail["vrp_now"] >= detail["vrp_threshold"] - NEAR_BAND_VOL
        if not (ok_iv or ok_vrp):
            return None
        return detail

    return None
