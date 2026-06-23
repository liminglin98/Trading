# -*- coding: utf-8 -*-
"""
app.py -- 50ETF butterfly wing-sizing dashboard (Streamlit)
===========================================================
Local browser dashboard that wraps the existing model + optimizer. ANALYTICS &
SUGGESTIONS ONLY -- it never places or moves trades; you execute in your broker.

Panels:
  1. Suggestion   -- current-month butterfly spec (forward/live context) + 80%
                     band + payoff chart.
  2. Holdings P&L -- enter your option legs by hand; see payoff at expiry,
                     current mark-to-market, breakevens, max profit/loss.
  3. Grid explorer-- wing x profit-target x stop x exit-DTE walk-forward grid,
                     ranked by Calmar, with a wing x exit heatmap.
  4. Data         -- panel freshness + refresh / rebuild controls.

Run (from inside macro_model\):
    pip install streamlit
    streamlit run app.py

Notes: first load runs the model fit + pulls the SSE IV chain (a few minutes,
needs internet); results are cached until you hit a refresh button.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import butterfly_timing_backtest as B
import butterfly_param_search as PS
import butterfly_wing_exit_optimizer as WX
import iron_condor_backtest as IC
import pipeline

st.set_page_config(page_title="50ETF Butterfly Dashboard", layout="wide")
R_FREE = B.R_FREE

# =============================================================================
# Cached heavy builds (cleared by the refresh buttons)
# =============================================================================
@st.cache_data(show_spinner="Building contexts (model fit + SSE IV chain)...")
def load_contexts():
    contexts, etf, last_daily, X, y, df = PS.build_contexts()
    return contexts, etf, last_daily, X, y, df

@st.cache_data(show_spinner="Fetching the latest 50ETF close...")
def load_etf_fresh(stamp):
    # The daily ETF close series, re-pulled on every new day (sina is always
    # fresh + fast). Kept SEPARATE from the heavy load_contexts() cache so the
    # live spot/tracker no longer freeze at the price from the first cache build.
    return TK.load_etf()

@st.cache_data(show_spinner="Fetching the most recent chain for the live spec...")
def load_live_context(stamp):
    # `stamp` (a date string) is part of the cache key so the live chain AND the
    # daily ETF close are re-fetched on a new day; refresh buttons also clear it.
    contexts, etf, last_daily, X, y, df = load_contexts()
    etf_fresh = load_etf_fresh(stamp)            # fresh spot, not the frozen one
    ctx = WX.build_live_context(etf_fresh, X, y)
    live = ctx is not None
    if ctx is None:
        ctx = contexts[-1]
    return ctx, live

@st.cache_data(show_spinner="Running the wing x exit grid...")
def run_grid(wings, pts, stops, edtes):
    contexts, etf, last_daily, X, y, df = load_contexts()
    rows = []
    for w in wings:
        for pt in pts:
            for stp in stops:
                for ed in edtes:
                    bt = WX.evaluate_fly(contexts, etf, last_daily, w, pt, stp, ed)
                    m = WX._metrics(bt, df, y)
                    rows.append(dict(wing=w, pt=pt, stop=stp, exit_dte=ed,
                                     trades=m["trades"], cagr=m["rule_cagr"],
                                     mdd=m["rule_mdd"], calmar=m["calmar"],
                                     winrate=m["trade_winrate"],
                                     worst_month=m["worst_month"], total=m["rule_total"],
                                     bh_cagr=m["bh_cagr"], bh_mdd=m["bh_mdd"]))
    return pd.DataFrame(rows)

# =============================================================================
# Option / payoff helpers  (store + valuation live in tracking.py, shared with
# the headless EOD snapshot so both mark positions with identical code)
# =============================================================================
import tracking as TK
from tracking import (leg_value_now, load_saved, save_position, delete_position,
                      load_tracking, record_tracking, delete_tracking,
                      legs_from_saved, mark_position, latest_chain_for_expiry,
                      position_rules, position_status, set_position_rules,
                      position_max_profit, evaluate_rules, build_roll_suggestion,
                      record_roll, roll_lineage)

def leg_intrinsic_vec(cp, K, S):
    return np.maximum(S - K, 0.0) if cp == "C" else np.maximum(K - S, 0.0)

def position_pnl_expiry(legs, S):
    """legs: list of dict(type,strike,qty,entry). Signed P&L at expiry over S."""
    S = np.asarray(S, float)
    pnl = np.zeros_like(S)
    for lg in legs:
        pnl += lg["qty"] * (leg_intrinsic_vec(lg["type"], lg["strike"], S) - lg["entry"])
    return pnl

def breakevens(S, pnl):
    out = []
    for i in range(1, len(S)):
        if np.isfinite(pnl[i-1]) and np.isfinite(pnl[i]) and pnl[i-1] == pnl[i]:
            continue
        if (pnl[i-1] < 0 <= pnl[i]) or (pnl[i-1] > 0 >= pnl[i]):
            x = S[i-1] + (S[i]-S[i-1]) * (0 - pnl[i-1]) / (pnl[i] - pnl[i-1])
            out.append(float(x))
    return out

def leg_table(legs_df):
    """Pretty per-leg holdings table for display (action / type / strike / price)."""
    rows = []
    for _, r in legs_df.iterrows():
        rows.append(dict(action=f"{r['side']} {int(abs(float(r['qty'])))}x",
                         type=("call" if str(r["type"]).upper().startswith("C") else "put"),
                         strike=round(float(r["strike"]), 3),
                         **{"price/unit": round(float(r["entry"]), 4)}))
    return pd.DataFrame(rows)

def live_condor_spec(ctx, m, p):
    """Current-month iron condor (short put spread + short call spread) priced on
    the live smile. Net-credit structure; max loss = wing width - credit."""
    smile, spot, T = ctx["smile"], ctx["spot"], ctx["T"]
    legs_df = TK.build_condor_legs(smile, spot, T, m, p)
    if legs_df is None:
        return None
    credit = 0.0
    for _, r in legs_df.iterrows():
        sgn = 1.0 if str(r["side"]).lower().startswith("s") else -1.0
        credit += sgn * float(r["entry"]) * abs(float(r["qty"]))
    Kp_l, Kp_s, Kc_s, Kc_l = [float(x) for x in legs_df["strike"].tolist()]
    wing = max(Kp_s - Kp_l, Kc_l - Kc_s)
    max_loss = wing - credit
    return dict(legs_df=legs_df, credit=credit, max_loss=max_loss, wing=wing,
                strikes=(Kp_l, Kp_s, Kc_s, Kc_l), spot=spot,
                dte=int(round(T * 365)), kp_long=Kp_l, kp_short=Kp_s,
                kc_short=Kc_s, kc_long=Kc_l, m=m, p=p)

# =============================================================================
# Sidebar: freshness + refresh + grid params + bankroll
# =============================================================================
st.sidebar.title("Controls")
last, stale, msg = pipeline.freshness()
(st.sidebar.warning if stale else st.sidebar.success)(msg)

if st.sidebar.button("Reload model (clear cache)"):
    st.cache_data.clear(); st.rerun()

with st.sidebar.expander("Rebuild factor panel (slow)"):
    st.caption("Runs build_model_panel.py to refresh the macro/market predictors. "
               "Some sources need the proxy. Only the dispersion band uses these.")
    if st.button("Rebuild panel now"):
        with st.spinner("Rebuilding panel..."):
            ok, out = pipeline.rebuild_panel()
        st.text(out[-3000:] if out else "(no output)")
        (st.success if ok else st.error)("rebuild ok" if ok else "rebuild had problems")
        st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.subheader("Grid ranges")
wings = st.sidebar.multiselect("wing steps", [1,2,3,4,5,6], default=[1,2,3,4,5])
pts = st.sidebar.multiselect("profit target (frac of max)", [0.25,0.5,0.75,1.0],
                             default=[0.5,0.75,1.0])
use_no_pt = st.sidebar.checkbox("include no profit-target", value=True)
stops = st.sidebar.multiselect("stop (frac of debit)", [0.5,0.75], default=[0.5])
use_no_stop = st.sidebar.checkbox("include no stop", value=True)
edtes = st.sidebar.multiselect("exit DTE", [7,14,21], default=[7,14])
use_hold = st.sidebar.checkbox("include hold-to-expiry", value=True)
pt_list = pts + ([None] if use_no_pt else [])
stop_list = stops + ([None] if use_no_stop else [])
edte_list = edtes + ([None] if use_hold else [])

st.sidebar.markdown("---")
bankroll = st.sidebar.number_input("bankroll (for sizing)", value=100000.0, step=10000.0)
target_risk = st.sidebar.slider("risk per trade", 0.02, 0.50, float(WX.TARGET_RISK), 0.01)

# =============================================================================
# Load core data once
# =============================================================================
st.title("50ETF Butterfly -- wing sizing & risk dashboard")
st.caption("Analytics and suggestions only. Not investment advice; no orders are placed.")

try:
    contexts, etf, last_daily, X, y, df = load_contexts()
except Exception as e:
    st.error(f"Failed to build contexts: {e}")
    st.stop()

# pick a robust operational config from the grid (band wing + best exit there)
def _as_key(vals):
    """Stable, hashable tuple: sorted non-None values, with None last if present."""
    nn = sorted(v for v in vals if v is not None)
    return tuple(nn + ([None] if None in vals else []))

if wings and pt_list and stop_list and edte_list:
    res = run_grid(tuple(sorted(wings)), _as_key(pt_list), _as_key(stop_list), _as_key(edte_list))
else:
    res = pd.DataFrame()

band_steps = np.nanmedian([WX.band_wing_steps(c) for c in contexts])
grid_ok = len(res) > 0
if grid_ok and np.isfinite(band_steps):
    band_wing = int(np.clip(round(band_steps), min(wings), max(wings)))
    at = res[res["wing"] == band_wing].sort_values("calmar", ascending=False)
    best = at.iloc[0] if len(at) else res.sort_values("calmar", ascending=False).iloc[0]
elif grid_ok:
    best = res.sort_values("calmar", ascending=False).iloc[0]
    band_wing = int(best["wing"])
else:
    best = None; band_wing = int(round(band_steps)) if np.isfinite(band_steps) else 3

tab1, tab2, tab3, tab_track, tab4 = st.tabs(
    ["Suggestion", "Holdings P&L", "Grid explorer", "Tracker", "Data"])

# =============================================================================
# TAB 1 -- current-month suggestion
# =============================================================================
with tab1:
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    ctx, live = load_live_context(stamp)
    src = "LIVE (most recent chain)" if live else "FALLBACK (last settled month)"
    if best is not None:
        pt_b, stop_b, ed_b = best["pt"], best["stop"], best["exit_dte"]
    else:
        pt_b, stop_b, ed_b = 1.0, 0.5, 14
    condor_m = int(TK.STRAT_DEFAULTS["condor_m"])
    condor_p = int(TK.STRAT_DEFAULTS["condor_p"])

    st.caption(f"Chain source: **{src}**, as of "
               f"{pd.Timestamp(ctx['entry_date']).date()} "
               f"(expiry {ctx['expiry']} = {pd.Timestamp(ctx['exp_date']).date()}). "
               "Full strategy structure -- parameters + the exact legs to enter. "
               "Suggestions only; no orders are placed.")

    # -------------------------------------------------------------------------
    # STRATEGY 1 -- long call butterfly (sell dispersion; profits if pinned)
    # -------------------------------------------------------------------------
    st.subheader("1. Long call butterfly  --  short dispersion (profits near spot)")
    spec = WX.live_spec(ctx, int(band_wing), pt_b, stop_b, ed_b)
    if spec is None:
        st.warning("Could not build the butterfly (wing runs off the listed strike grid).")
    else:
        st.markdown("**Parameters**")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("wing width", f"{band_wing} steps",
                  help=f"model {int(WX.BAND_Q*100)}% band-implied {band_steps:.2f} steps")
        p2.metric("take profit", f"{pt_b} of max" if pt_b is not None else "off")
        p3.metric("stop loss", f"{stop_b} of debit" if stop_b is not None else "off")
        p4.metric("roll / exit DTE", f"{ed_b}" if ed_b is not None else "hold")

        st.markdown("**Current holdings suggestion**")
        bf_lots = int(target_risk * bankroll / spec["cost"]) if spec["cost"] > 0 else 0
        st.dataframe(leg_table(TK.build_butterfly_legs(
            ctx["smile"], ctx["spot"], ctx["T"], int(band_wing))),
            use_container_width=True, hide_index=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("spot", f"{spec['spot']:.3f}")
        c2.metric("net debit (max loss)", f"{spec['cost']:.4f}")
        c3.metric("max profit", f"{spec['max_profit']:.4f}")
        c4.metric("DTE", f"{spec['dte']}")
        c1.metric("lower / upper wing", f"{spec['wing_low']:.3f} / {spec['wing_high']:.3f}")
        c2.metric("body (sell 2x)", f"{spec['body']:.3f}")
        c3.metric("model 80% band", f"[{spec['band_lo']:.3f}, {spec['band_hi']:.3f}]")
        c4.metric(f"size @ {target_risk:.0%} risk", f"{bf_lots} lots")

        strikes = (spec["wing_low"], spec["body"], spec["wing_high"])
        S = np.linspace(spec["spot"]*0.80, spec["spot"]*1.20, 241)
        payoff = B.payoff_butterfly(S, strikes) - spec["cost"]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(S, payoff, color="#1f4e79", lw=1.8)
        ax.axhline(0, color="grey", lw=0.7)
        ax.axvline(spec["spot"], color="#c0392b", lw=1, ls="--", label="spot")
        ax.axvspan(spec["band_lo"], spec["band_hi"], color="#2e7d32", alpha=0.10, label="80% band")
        ax.set_xlabel("50ETF at expiry"); ax.set_ylabel("P&L per 1-lot"); ax.legend()
        ax.set_title("Butterfly payoff at expiry")
        st.pyplot(fig)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # STRATEGY 2 -- iron condor (short vol, defined risk; net credit)
    # -------------------------------------------------------------------------
    st.subheader("2. Iron condor  --  harvest the variance risk premium (net credit)")
    csp = live_condor_spec(ctx, condor_m, condor_p)
    if csp is None:
        st.warning("Could not build the condor (protective wings run off the listed grid).")
    else:
        st.markdown("**Parameters**")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("short strikes", f"{condor_m} steps OTM")
        q2.metric("protective wings", f"{condor_p} steps further")
        q3.metric("sell-vol gate", f"implied/model >= {IC.RATIO_GATE:.2f}")
        q4.metric("sizing", f"{IC.KELLY_FRAC:.0%} Kelly, cap {IC.KELLY_CAP:.0%}")

        st.markdown("**Current holdings suggestion**")
        cd_lots = int(target_risk * bankroll / csp["max_loss"]) if csp["max_loss"] > 0 else 0
        st.dataframe(leg_table(csp["legs_df"]),
                     use_container_width=True, hide_index=True)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("spot", f"{csp['spot']:.3f}")
        d2.metric("net credit (max profit)", f"{csp['credit']:.4f}")
        d3.metric("max loss", f"{csp['max_loss']:.4f}")
        d4.metric("DTE", f"{csp['dte']}")
        d1.metric("short put / call", f"{csp['kp_short']:.3f} / {csp['kc_short']:.3f}")
        d2.metric("long put / call", f"{csp['kp_long']:.3f} / {csp['kc_long']:.3f}")
        d3.metric("profit zone", f"[{csp['kp_short']:.3f}, {csp['kc_short']:.3f}]")
        d4.metric(f"size @ {target_risk:.0%} risk", f"{cd_lots} lots")

        Sc = np.linspace(csp["spot"]*0.80, csp["spot"]*1.20, 241)
        payoff_c = IC.payoff_condor(Sc, csp["strikes"], csp["credit"])
        fig2, ax2 = plt.subplots(figsize=(8, 3.6))
        ax2.plot(Sc, payoff_c, color="#6a1b9a", lw=1.8)
        ax2.axhline(0, color="grey", lw=0.7)
        ax2.axvline(csp["spot"], color="#c0392b", lw=1, ls="--", label="spot")
        ax2.axvspan(csp["kp_short"], csp["kc_short"], color="#2e7d32", alpha=0.10,
                    label="profit zone (short strikes)")
        ax2.set_xlabel("50ETF at expiry"); ax2.set_ylabel("P&L per 1-lot"); ax2.legend()
        ax2.set_title("Iron condor payoff at expiry")
        st.pyplot(fig2)
        st.caption("The condor is the structural mirror of the butterfly: both are "
                   "short realized vol, but the condor collects a credit and pays out "
                   "if 50ETF stays inside the short strikes, with the long wings "
                   "capping the tail. Friction (~bid/ask + commission) is not in these "
                   "marks; the backtest module charges it explicitly.")

# =============================================================================
# TAB 2 -- holdings P&L (manual leg entry)
# =============================================================================
with tab2:
    st.subheader("Enter your option legs")
    st.caption("side: Buy (long) or Sell (short). type: C (call) or P (put). "
               "qty: number of contracts (positive). entry: price paid (Buy) or "
               "received (Sell) per unit.")
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    ctx, live = load_live_context(stamp)
    smile = ctx["smile"]; spot_default = float(ctx["spot"]); T_default = float(ctx["T"])

    # prefill with the suggested butterfly if available
    if 'holdings' not in st.session_state:
        if best is not None:
            sp = WX.live_spec(ctx, int(band_wing), best["pt"], best["stop"], best["exit_dte"])
        else:
            sp = None
        if sp is not None:
            st.session_state.holdings = pd.DataFrame([
                dict(side="Buy",  type="C", strike=round(sp["wing_low"],3),  qty=1, entry=round(leg_value_now("C", sp["wing_low"],  spot_default, T_default, smile)[0], 4)),
                dict(side="Sell", type="C", strike=round(sp["body"],3),      qty=2, entry=round(leg_value_now("C", sp["body"],      spot_default, T_default, smile)[0], 4)),
                dict(side="Buy",  type="C", strike=round(sp["wing_high"],3), qty=1, entry=round(leg_value_now("C", sp["wing_high"], spot_default, T_default, smile)[0], 4)),
            ])
        else:
            st.session_state.holdings = pd.DataFrame(
                [dict(side="Buy", type="C", strike=spot_default, qty=1, entry=0.05)])

    edited = st.data_editor(
        st.session_state.holdings, num_rows="dynamic", use_container_width=True,
        key="holdings_editor",
        column_config={
            "side": st.column_config.SelectboxColumn("side", options=["Buy", "Sell"], required=True),
            "type": st.column_config.SelectboxColumn("type", options=["C", "P"], required=True),
            "strike": st.column_config.NumberColumn("strike", format="%.3f", step=0.05),
            "qty": st.column_config.NumberColumn("qty", min_value=0, step=1,
                                                 help="number of contracts (positive)"),
            "entry": st.column_config.NumberColumn("entry price", format="%.4f", step=0.0001),
        })

    colA, colB, colC = st.columns(3)
    spot_now = colA.number_input("current spot", value=spot_default, step=0.001, format="%.3f")
    dte_now = colB.number_input("days to expiry", value=int(round(T_default*365)), step=1)
    use_smile = colC.checkbox("mark IV from live smile", value=True)
    T_now = max(dte_now, 1) / 365.0

    legs = []
    for _, r in edited.iterrows():
        try:
            cp = str(r["type"]).strip().upper()[0]
            if cp not in ("C", "P"):
                continue
            sign = -1.0 if str(r["side"]).strip().lower().startswith("s") else 1.0
            qty = sign * abs(float(r["qty"]))         # Buy -> +, Sell -> -
            if qty == 0:
                continue
            legs.append(dict(type=cp, strike=float(r["strike"]),
                             qty=qty, entry=float(r["entry"])))
        except Exception:
            continue

    if legs:
        # current mark-to-market
        mark = 0.0; cost_basis = 0.0
        for lg in legs:
            val, _iv = leg_value_now(lg["type"], lg["strike"], spot_now, T_now,
                                     smile if use_smile else None)
            mark += lg["qty"] * val
            cost_basis += lg["qty"] * lg["entry"]
        pnl_now = mark - cost_basis

        S = np.linspace(spot_now*0.75, spot_now*1.25, 301)
        pnl_exp = position_pnl_expiry(legs, S)
        bes = breakevens(S, pnl_exp)
        maxp = float(np.nanmax(pnl_exp)); minp = float(np.nanmin(pnl_exp))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("cost basis (net debit)", f"{cost_basis:+.4f}")
        m2.metric("current mark", f"{mark:+.4f}")
        m3.metric("open P&L (now)", f"{pnl_now:+.4f}")
        m4.metric("breakevens", ", ".join(f"{b:.3f}" for b in bes) if bes else "none in range")
        m1.metric("max profit (expiry, in range)", f"{maxp:+.4f}")
        m2.metric("max loss (expiry, in range)", f"{minp:+.4f}")
        lots = int(target_risk * bankroll / abs(cost_basis)) if abs(cost_basis) > 1e-9 else 0
        m3.metric("size at risk budget (lots)", f"{lots}")

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(S, pnl_exp, color="#1f4e79", lw=1.8, label="P&L at expiry")
        ax.axhline(0, color="grey", lw=0.7)
        ax.axvline(spot_now, color="#c0392b", lw=1, ls="--", label="spot now")
        ax.scatter([spot_now], [pnl_now], color="#2e7d32", zorder=5, label="mark now")
        for b in bes:
            ax.axvline(b, color="#888", lw=0.7, ls=":")
        ax.set_xlabel("50ETF underlying"); ax.set_ylabel("position P&L"); ax.legend()
        ax.set_title("Holdings payoff at expiry (+ current mark)")
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("**Save this combination**")
        sc1, sc2, sc3 = st.columns([2, 2, 1])
        save_name = sc1.text_input("name", value="my butterfly", key="save_name",
                                   label_visibility="collapsed")
        default_exp = pd.Timestamp(ctx["exp_date"]).date() if ctx.get("exp_date") is not None \
            else (pd.Timestamp.now() + pd.Timedelta(days=int(dte_now))).date()
        save_exp = sc2.date_input("expiry", value=default_exp, key="save_exp",
                                  label_visibility="collapsed")
        if sc3.button("Save", use_container_width=True):
            snap = dict(expiry=str(save_exp), spot_at_save=round(spot_now, 4),
                        dte_at_save=int(dte_now), cost_basis=round(cost_basis, 4),
                        mark_at_save=round(mark, 4), pnl_at_save=round(pnl_now, 4))
            pid = save_position(save_name, edited, snap)
            st.success(f"Saved: {pid}  (expiry {save_exp})")
    else:
        st.info("Add at least one valid leg (side Buy/Sell, type C/P, numeric strike/qty/entry).")

    # --- saved combinations: load / delete / export (always visible) ---
    st.markdown("---")
    st.markdown("### Saved combinations")
    saved = load_saved()
    if not len(saved):
        st.caption("None saved yet. Build a position above and click 'Save combination'.")
    else:
        ids = saved["position_id"].dropna().unique().tolist()[::-1]   # newest first
        pick = st.selectbox("pick a saved combination", ids, key="saved_pick")
        b1, b2, b3 = st.columns(3)
        if b1.button("Load into editor", use_container_width=True):
            sub = (saved[saved["position_id"] == pick]
                   [["side", "type", "strike", "qty", "entry"]].reset_index(drop=True))
            st.session_state.holdings = sub
            st.session_state.pop("holdings_editor", None)   # reset the editor base
            st.rerun()
        if b2.button("Delete", use_container_width=True):
            delete_position(pick); st.rerun()
        b3.download_button("Download all saved (CSV)", saved.to_csv(index=False),
                           file_name="saved_positions.csv", use_container_width=True)

        snap_cols = ["position_id", "saved_at", "spot_at_save", "dte_at_save",
                     "cost_basis", "mark_at_save", "pnl_at_save"]
        summ = (saved.drop_duplicates("position_id")[snap_cols]
                .set_index("position_id").iloc[::-1])
        st.caption("One row per saved combination (snapshot at save time):")
        st.dataframe(summ, use_container_width=True)
        with st.expander("all saved legs (full table for analysis)"):
            st.dataframe(saved, use_container_width=True)

# =============================================================================
# TAB 3 -- grid explorer
# =============================================================================
with tab3:
    st.subheader("Wing x exit walk-forward grid")
    if not grid_ok:
        st.info("Pick at least one value in each grid range (sidebar).")
    else:
        bh_cagr = res["bh_cagr"].iloc[0]; bh_mdd = res["bh_mdd"].iloc[0]
        bh_cal = bh_cagr/abs(bh_mdd) if (bh_mdd and np.isfinite(bh_mdd) and bh_mdd != 0) else np.nan
        st.caption(f"buy & hold: CAGR {bh_cagr:+.3f}, MDD {bh_mdd:+.3f}, Calmar {bh_cal:+.2f}. "
                   "Long fly is ~breakeven on the real chain -> read as a RISK map, prefer a "
                   "stable region over the peak; widest wing = leverage artifact (worst-month tail).")
        show = ["wing","pt","stop","exit_dte","trades","cagr","mdd","calmar","winrate","worst_month"]
        st.dataframe(res.sort_values("calmar", ascending=False)[show]
                     .style.format({c: "{:+.3f}" for c in ["cagr","mdd","calmar","winrate","worst_month"]}),
                     use_container_width=True, height=360)

        st.markdown("**Calmar by wing width** (stability across PT/STOP/EXIT)")
        st.dataframe(res.groupby("wing")["calmar"].agg(["mean","median","max"])
                     .style.format("{:+.3f}"), use_container_width=True)

        piv = res.copy()
        piv["exit_dte"] = piv["exit_dte"].astype("object").where(piv["exit_dte"].notna(), "hold")
        pv = piv.groupby(["wing","exit_dte"])["calmar"].mean().unstack()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn", origin="lower")
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        ax.set_xlabel("exit DTE"); ax.set_ylabel("wing steps")
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="mean Calmar")
        ax.set_title("mean Calmar: wing x exit-DTE")
        st.pyplot(fig)

# =============================================================================
# TAB Tracker -- daily forward P&L of the two model strategies
# =============================================================================
with tab_track:
    st.subheader("Strategy tracker -- daily movement of the two model strategies")
    st.caption("Each snapshot locks the model's current butterfly + iron condor for "
               "the front expiry (once per month) and re-marks those exact legs on the "
               "latest real chain, logging the daily P&L of each. This is a forward "
               "paper-trade of the strategies themselves -- not a hand-entered "
               "combination. Suggestions only; no orders are placed. For a hands-off "
               "daily log, point Windows Task Scheduler at  python pipeline.py --snapshot.")

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    etf_fresh = load_etf_fresh(stamp)
    track_params = dict(bfly_wing=int(band_wing),
                        condor_m=int(TK.STRAT_DEFAULTS["condor_m"]),
                        condor_p=int(TK.STRAT_DEFAULTS["condor_p"]))

    cbtn, cinfo = st.columns([1, 2])
    if cbtn.button("Record snapshot now (both strategies)", use_container_width=True):
        results = TK.snapshot_strategies(etf=etf_fresh, params=track_params)
        for strat, msg in results:
            (st.success if "logged" in msg else st.info)(f"{strat}: {msg}")
        st.rerun()
    cinfo.caption(f"butterfly wing = {band_wing} steps (model band)  |  "
                  f"condor = {track_params['condor_m']} steps OTM, "
                  f"{track_params['condor_p']}-step wings")

    STRAT_COLOR = {"butterfly": "#1f4e79", "condor": "#6a1b9a"}
    track = TK.load_strategy_tracking()

    if not len(track):
        st.info("No snapshots logged yet. Click 'Record snapshot now' to start the "
                "forward track of both strategies.")
    else:
        track = track.copy()
        track["asof_date"] = pd.to_datetime(track["asof_date"])
        track = track.sort_values("asof_date")

        # --- combined P&L chart (both strategies, per-1-lot) ---
        fig, ax = plt.subplots(figsize=(9, 4))
        for strat in TK.STRATEGIES:
            h = track[track["strategy"] == strat]
            if len(h):
                ax.plot(h["asof_date"], h["pnl"], marker="o", lw=1.6,
                        color=STRAT_COLOR.get(strat, None), label=TK.STRAT_LABEL[strat])
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_ylabel("P&L per 1-lot since entry"); ax.set_xlabel("snapshot date")
        ax.set_title("Forward P&L -- butterfly vs iron condor")
        ax.legend(); fig.autofmt_xdate()
        st.pyplot(fig)

        # --- per-strategy detail ---
        for strat in TK.STRATEGIES:
            h = track[track["strategy"] == strat].copy()
            if not len(h):
                continue
            st.markdown(f"#### {TK.STRAT_LABEL[strat]}")
            cur_exp = str(h["expiry"].iloc[-1])
            locked = TK.get_locked_legs(strat, cur_exp)
            entry_date = TK.get_entry_date(strat, cur_exp) or str(h["entry_date"].iloc[-1])
            entry_cost = float(h["cost_basis"].iloc[-1])   # locked at entry, constant
            st.caption(f"Entered **{entry_date}** at cost basis {entry_cost:+.4f} "
                       f"(locked from that day's EOD chain; marked daily, never re-entered). "
                       f"A new entry is only opened when the front expiry rolls to a new month.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("entry date", entry_date)
            m2.metric("latest P&L", f"{h['pnl'].iloc[-1]:+.4f}")
            m3.metric("peak / trough", f"{h['pnl'].max():+.3f} / {h['pnl'].min():+.3f}")
            m4.metric("snapshots / DTE", f"{len(h)} / {int(h['dte'].iloc[-1])}")
            if locked is not None:
                with st.expander(f"locked entry legs for the current expiry ({cur_exp})"):
                    st.dataframe(leg_table(locked), use_container_width=True, hide_index=True)
            cols = [c for c in ["asof_date", "expiry", "spot", "dte",
                                "cost_basis", "mark", "pnl"] if c in h.columns]
            st.dataframe(h[cols].reset_index(drop=True), use_container_width=True,
                         hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button("Download strategy track (CSV)",
                           track.to_csv(index=False),
                           file_name="strategy_tracking.csv", use_container_width=True)
        if d2.button("Clear all strategy history", use_container_width=True):
            TK.delete_strategy_tracking(); st.rerun()

# =============================================================================
# TAB 4 -- data / freshness
# =============================================================================
with tab4:
    st.subheader("Data freshness")
    (st.warning if stale else st.success)(msg)
    st.write(f"backtest contexts: {len(contexts)} months "
             f"({contexts[0]['t'].date()} .. {contexts[-1]['t'].date()})")
    st.write(f"daily ETF (model/backtest series) through: {pd.Timestamp(last_daily).date()}")
    # live spot freshness -- the daily ETF series is re-pulled each new day so the
    # suggestion + tracker mark on TODAY's close, not the price at first cache build.
    etf_fresh = load_etf_fresh(pd.Timestamp.now().strftime("%Y-%m-%d"))
    st.write(f"live 50ETF close: {float(etf_fresh.iloc[-1]):.3f} "
             f"as of {pd.Timestamp(etf_fresh.index[-1]).date()}  (re-pulled daily)")
    st.caption("ETF prices + SSE IV chain auto-fetch each (re)build; the factor panel is "
               "static -- use 'Rebuild factor panel' in the sidebar when it goes stale. "
               "The live spot/tracker use a separately-cached daily ETF pull, so they "
               "refresh each new day without needing the 'Reload model' button.")
    if grid_ok:
        st.download_button("download grid CSV", res.to_csv(index=False),
                           file_name="wing_exit_grid.csv")
