"""Options Lab — 50ETF/300ETF options dashboard.

Run:  streamlit run app.py   (from analysis/options_lab)

Tabs:
  1. Data     — daily chain recorder / CSV updater status
  2. Positions— enter custom combos, track daily P&L, close them
  3. Backtest — template+rules strategies over full history, market graphs
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import alerts_core as ac
import backtest as bt
import charts
import consolidated_store as cs
import data_store as ds
import margin as mg
import positions as pos
import quarterly_atm_short_straddle as qs
import dashboard_data as DD

st.set_page_config(page_title="Options Lab", layout="wide")


@st.cache_data(show_spinner="Loading option chain history...")
def get_chain(u: str) -> pd.DataFrame:
    return ds.load_chain(u)


@st.cache_data
def get_ohlc(u: str) -> pd.DataFrame | None:
    return ds.load_etf_ohlc(u)


# 50ETF/300ETF only -- Shanghai Composite has no sina hq quote counterpart
# in this codebase, and idx_sh's own source (stock_zh_index_daily) doesn't
# suffer the same-day lag that fund_etf_hist_sina does.
LIVE_QUOTE_UNDERLYING = {"etf50": "510050", "etf300": "510300"}


@st.cache_data(ttl=60, show_spinner=False)
def get_live_quote(code: str) -> tuple[float, str]:
    import updater
    spot, trade_date = updater.etf_quote(code)
    return spot, trade_date.isoformat()


def fmt_expiry_options(chain: pd.DataFrame):
    last_dt = chain[ds.C_DATE].max()
    return sorted(chain.loc[chain[ds.C_DATE] == last_dt, ds.C_EXPIRY].unique())


st.title("Options Lab")
tab_data, tab_pos, tab_sim, tab_bt, tab_macro, tab_alerts, tab_ta = st.tabs(
    ["📥 Data", "📌 Positions", "🧮 Simulate", "🧪 Backtest",
     "🌐 Macro Dashboard", "🔔 Alerts", "📈 Technical"])

# =============================================================== Data tab
# All 9 ETF underlyings now have a native Data/options CSV (data_store) --
# consolidated_store.py is kept only for ad-hoc CFFEX index-option access,
# out of this app's scope (see Data/买断日更/README.md "Scope").
ALL_CHAIN_NAMES = {**ds.UNDERLYINGS, **ds.EXT_LIVE_UNDERLYINGS, **ds.EXT_SZSE_UNDERLYINGS}


def _load_any_chain(u: str) -> pd.DataFrame:
    return get_chain(u)


with tab_data:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Chain status")
        st.caption("Live-updated by the button below, real exchange IV.")
        for u, name in ds.UNDERLYINGS.items():
            last = ds.last_chain_date(u)
            st.metric(f"{name} ({u})",
                      last.date().isoformat() if last is not None else "missing")
        st.caption("Also live-updated (same Sina feed) -- Data/Technical tabs only, "
                   "not yet wired into Positions/Backtest.")
        for u, name in ds.EXT_LIVE_UNDERLYINGS.items():
            last = ds.last_chain_date(u)
            st.metric(f"{name} ({u})",
                      last.date().isoformat() if last is not None else "missing")
        st.divider()
        if st.button("Update now (fetch today's chain)", type="primary"):
            with st.spinner("Fetching from sina/tencent..."):
                try:
                    import importlib
                    import updater
                    importlib.reload(updater)
                    updater.run(list(ds.UNDERLYINGS) + list(ds.EXT_LIVE_UNDERLYINGS))
                    st.cache_data.clear()
                    st.success("Updated. Reload the page to see new data.")
                except Exception as e:  # network blocked / market closed
                    import traceback
                    st.error(f"Update failed: {e}")
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc())
        st.caption("Tip: schedule `python updater.py` after 15:30 on trading "
                   "days (Task Scheduler / run_snapshot.bat) so the CSV stays "
                   "current without opening this app.")
        st.divider()
        st.caption("Vendor snapshot only -- SZSE options aren't in Sina's "
                   "feed at all, so these only refresh when a new purchased data "
                   "drop is consolidated and re-materialized "
                   "(Data/买断日更/consolidate_etf_index_options.py then "
                   "materialize_szse_full_csv.py), not by the button above. "
                   "IV/most Greeks are Black-Scholes-derived, not exchange-quoted "
                   "-- see data_store.EXT_SZSE_UNDERLYINGS.")
        for u, name in ds.EXT_SZSE_UNDERLYINGS.items():
            last = ds.last_chain_date(u)
            st.metric(f"{name} ({u})",
                      last.date().isoformat() if last is not None else "missing")
    with c2:
        st.subheader("Latest chain snapshot")
        u = st.selectbox("Underlying", list(ALL_CHAIN_NAMES),
                         format_func=lambda x: f"{ALL_CHAIN_NAMES[x]} ({x})",
                         key="data_u")
        try:
            chain = _load_any_chain(u)
        except FileNotFoundError:
            chain = None
        if chain is None or chain.empty:
            st.info("No data yet for this underlying -- click \"Update now\" "
                    "(live-updatable ones) or wait for the next vendor drop.")
            st.stop()
        if u in ds.EXT_SZSE_UNDERLYINGS:
            st.caption(":gray[vendor snapshot -- not live-updated, IV/Greeks "
                       "Black-Scholes-derived]")
        last_dt = chain[ds.C_DATE].max()
        day = chain[chain[ds.C_DATE] == last_dt]
        exp = st.selectbox("Expiry", sorted(day[ds.C_EXPIRY].unique()),
                           format_func=lambda e: pd.Timestamp(e).date().isoformat())
        snap = day[day[ds.C_EXPIRY] == exp]
        tbl = snap.pivot_table(index=ds.C_STRIKE, columns=ds.C_CP,
                               values=[ds.C_CLOSE, ds.C_IV, ds.C_DELTA,
                                       ds.C_OI])
        spot = float(day[ds.C_UND_CLOSE].iloc[0])
        st.caption(f"As of {pd.Timestamp(last_dt).date()} — spot {spot}")
        st.dataframe(tbl, width='stretch', height=420)

# ============================================================ Positions tab
with tab_pos:
    chains = {u: get_chain(u) for u in ds.UNDERLYINGS}
    summary = pos.position_summary(chains)

    st.subheader("Account asset tracking")
    baseline = ds.load_account_baseline()
    ba1, ba2 = st.columns([2, 1])
    with ba1:
        baseline_value = st.number_input(
            "Current account value (¥, cash + all positions marked)",
            value=float(baseline.get("value") or 0.0), step=100.0, format="%.2f",
            key="account_baseline_input")
    with ba2:
        st.write("")  # vertical align the button with the number input
        if st.button("Save as today's anchor", use_container_width=True):
            ds.save_account_baseline(baseline_value, pd.Timestamp.now().date().isoformat())
            st.success("Saved.")
    eq = pos.account_equity_curve(chains, baseline_value) if not pos.load().empty \
        else pd.DataFrame()
    if eq.empty:
        st.info("No positions recorded yet -- the equity curve needs at least one "
                "tracked combo to reconstruct history from.")
    else:
        st.plotly_chart(charts.account_equity_figure(eq), width='stretch')
        asof_note = f" (last anchored {baseline['asof']})" if baseline.get("asof") else ""
        st.caption(f"Anchored to ¥{baseline_value:,.2f} as of today{asof_note}; earlier points "
                   f"are backed out using the combined P&L history of every tracked position "
                   f"(open + closed). Assumes no deposits/withdrawals during the window -- "
                   f"re-save the anchor whenever you check your real broker balance.")

    st.divider()
    st.subheader("Tracked positions")
    if summary.empty:
        st.info("No positions yet — add one below.")
    else:
        st.dataframe(summary.drop(columns=["asof"]),
                     width='stretch', hide_index=True)
        open_ids = summary["position_id"].tolist()
        sel = st.selectbox(
            "Inspect position", open_ids,
            format_func=lambda p: f"{summary.set_index('position_id').loc[p, 'pos_name']} ({p})")
        if sel:
            legs = pos.load()
            legs = legs[legs["position_id"] == sel]
            u_pos = summary.set_index("position_id").loc[sel, "underlying"]
            ch = chains[u_pos]
            marks_now, asof_now = pos.latest_marks(ch)
            spot_last = float(ch.loc[ch[ds.C_DATE] == ch[ds.C_DATE].max(),
                                     ds.C_UND_CLOSE].iloc[0])

            legs_show = legs[["leg_code", "leg_name", "cp", "strike", "expiry",
                              "direction", "qty", "entry_price", "status",
                              "close_price", "close_date",
                              "leg_open_date"]].copy()
            legs_show["direction"] = legs_show["direction"].map(
                {1: "buy", -1: "sell"})
            st.dataframe(legs_show, width='stretch', hide_index=True)

            open_legs = legs[legs["status"] != "closed"]
            closed_legs = legs[legs["status"] == "closed"]
            realized = float(sum(
                l["direction"] * l["qty"] * (float(l["close_price"]) - l["entry_price"])
                for _, l in closed_legs.iterrows())) * ds.MULTIPLIER

            if not open_legs.empty:
                st.plotly_chart(charts.payoff_figure(
                    open_legs, spot_last,
                    "P&L at expiry vs underlying price (open legs)",
                    realized_pnl_yuan=realized),
                    width='stretch')
            else:
                st.info(f"All legs closed -- realized P&L {realized:+,.0f} yuan.")

            row = summary.set_index("position_id").loc[sel]

            if row["status"] == "open":
                st.markdown("**Close whole combo**")
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    if st.button("Close at latest marks", type="primary"):
                        prices = {l["leg_code"]: float(marks_now.get(l["leg_code"],
                                                                     l["entry_price"]))
                                  for _, l in open_legs.iterrows()}
                        pos.close_position(sel, prices)
                        st.rerun()
                with cc2:
                    if st.button("Close at T-1 marks"):
                        marks_t1, dt_t1 = pos.marks_at_offset(ch, offset=1)
                        prices = {l["leg_code"]: float(marks_t1.get(l["leg_code"],
                                                                    l["entry_price"]))
                                  for _, l in open_legs.iterrows()}
                        pos.close_position(sel, prices,
                                          close_date=pd.Timestamp(dt_t1).date().isoformat())
                        st.rerun()
                with cc3:
                    if st.button("Delete (remove from tracking)"):
                        pos.delete_position(sel)
                        st.rerun()

                st.markdown("**Adjust an individual leg**")
                leg_labels = {
                    f"{l['leg_name']} {'buy' if l['direction'] == 1 else 'sell'} "
                    f"{int(l['qty'])}x @ {l['entry_price']:.4f} ({l['leg_code']})":
                        l["leg_code"]
                    for _, l in open_legs.iterrows()}
                leg_pick_label = st.selectbox("Leg", list(leg_labels.keys()),
                                              key=f"leg_pick_{sel}")
                leg_code_sel = leg_labels[leg_pick_label]
                leg_row = open_legs[open_legs["leg_code"] == leg_code_sel].iloc[0]
                full_qty = int(leg_row["qty"])
                default_px = float(marks_now.get(leg_code_sel, leg_row["entry_price"]))

                with st.expander(f"Close / partial-close -- {leg_pick_label}"):
                    e1, e2, e3 = st.columns(3)
                    with e1:
                        close_qty = st.number_input(
                            "Qty to close", 1, full_qty, full_qty,
                            key=f"close_qty_{sel}_{leg_code_sel}")
                    with e2:
                        close_px = st.number_input(
                            "Close price", value=default_px, format="%.4f",
                            key=f"close_px_{sel}_{leg_code_sel}")
                    with e3:
                        close_dt = st.date_input(
                            "Close date", value=pd.Timestamp.now().date(),
                            key=f"close_dt_{sel}_{leg_code_sel}")
                    if st.button("Close this leg", key=f"close_leg_btn_{sel}_{leg_code_sel}"):
                        if close_qty >= full_qty:
                            pos.close_leg(sel, leg_code_sel, close_px,
                                         close_dt.isoformat())
                        else:
                            pos.partial_close_leg(sel, leg_code_sel, int(close_qty),
                                                  close_px, close_dt.isoformat())
                        st.rerun()

                with st.expander(f"Edit -- {leg_pick_label} (data-entry fix)"):
                    f1, f2, f3, f4 = st.columns(4)
                    with f1:
                        edit_qty = st.number_input(
                            "Qty", 1, 500, full_qty, key=f"edit_qty_{sel}_{leg_code_sel}")
                    with f2:
                        edit_strike = st.number_input(
                            "Strike", value=float(leg_row["strike"]), format="%.3f",
                            key=f"edit_strike_{sel}_{leg_code_sel}")
                    with f3:
                        edit_entry = st.number_input(
                            "Entry price", value=float(leg_row["entry_price"]),
                            format="%.4f", key=f"edit_entry_{sel}_{leg_code_sel}")
                    with f4:
                        edit_dir = st.selectbox(
                            "Direction", ["buy", "sell"],
                            index=0 if leg_row["direction"] == 1 else 1,
                            key=f"edit_dir_{sel}_{leg_code_sel}")
                    if st.button("Save edit", key=f"save_edit_btn_{sel}_{leg_code_sel}"):
                        pos.edit_leg(sel, leg_code_sel, {
                            "qty": int(edit_qty), "strike": float(edit_strike),
                            "entry_price": float(edit_entry),
                            "direction": 1 if edit_dir == "buy" else -1})
                        st.rerun()

                with st.expander("Add a leg to this position (roll / adjustment)"):
                    exps_add = fmt_expiry_options(ch)
                    a1, a2, a3, a4, a5 = st.columns([2, 1, 2, 1, 1])
                    with a1:
                        add_exp = st.selectbox(
                            "Expiry", exps_add, key=f"add_leg_exp_{sel}",
                            format_func=lambda e: pd.Timestamp(e).date().isoformat())
                    add_picker = pos.contract_picker(ch, expiry=add_exp)
                    with a2:
                        add_cp = st.selectbox(
                            "Call / Put", ["C", "P"], key=f"add_leg_cp_{sel}",
                            format_func=lambda x: "Call (购)" if x == "C" else "Put (沽)")
                    add_opts = add_picker[add_picker[ds.C_CP] == add_cp].sort_values(ds.C_STRIKE)
                    add_opts_by_strike = add_opts.set_index(ds.C_STRIKE)
                    with a3:
                        add_strike = st.selectbox(
                            "Strike", add_opts[ds.C_STRIKE].tolist(),
                            key=f"add_leg_strike_{sel}",
                            index=int((add_opts[ds.C_STRIKE] - spot_last).abs().argmin()))
                    add_row = add_opts_by_strike.loc[add_strike]
                    add_code = add_row[ds.C_CODE]
                    with a4:
                        add_dir = st.selectbox("Buy / Sell", ["buy", "sell"],
                                               key=f"add_leg_dir_{sel}")
                    with a5:
                        add_qty = st.number_input("Qty", 1, 500, 1,
                                                  key=f"add_leg_qty_{sel}")
                    add_px = st.number_input(
                        f"Entry price — latest close {float(add_row[ds.C_CLOSE]):.4f}",
                        value=float(add_row[ds.C_CLOSE]), format="%.4f",
                        key=f"add_leg_px_{sel}_{add_code}")
                    add_entry_date = st.date_input(
                        "Entry date", value=pd.Timestamp.now().date(),
                        key=f"add_leg_date_{sel}")
                    if st.button("Add leg to this position", key=f"add_leg_btn_{sel}"):
                        pos.add_leg_to_position(sel, {
                            "leg_code": add_code, "leg_name": add_row[ds.C_NAME],
                            "cp": add_cp, "strike": float(add_strike),
                            "expiry": pd.Timestamp(add_exp).date().isoformat(),
                            "direction": 1 if add_dir == "buy" else -1,
                            "qty": int(add_qty), "entry_price": float(add_px)},
                            entry_date=add_entry_date.isoformat())
                        st.rerun()
            else:
                st.caption(f"All legs closed. Realized P&L: {realized:+,.0f} yuan.")
                if st.button("Delete closed position"):
                    pos.delete_position(sel)
                    st.rerun()

    st.divider()
    st.subheader("New position")
    u = st.selectbox("Underlying", list(ds.UNDERLYINGS),
                     format_func=lambda x: f"{ds.UNDERLYINGS[x]} ({x})",
                     key="pos_u")
    chain = chains[u]
    exps = fmt_expiry_options(chain)
    spot_now = float(chain.loc[chain[ds.C_DATE] == chain[ds.C_DATE].max(),
                               ds.C_UND_CLOSE].iloc[0])
    st.caption(f"Latest spot: **{spot_now:.3f}** "
               f"(as of {pd.Timestamp(chain[ds.C_DATE].max()).date()})")
    c1, c2, c3, c4, c5 = st.columns([2, 1, 2, 1, 1])
    with c1:
        exp = st.selectbox("Expiry", exps, key="pos_exp",
                           format_func=lambda e: pd.Timestamp(e).date().isoformat())
    picker = pos.contract_picker(chain, expiry=exp)
    with c2:
        cp = st.selectbox("Call / Put", ["C", "P"], key="pos_cp",
                          format_func=lambda x: "Call (购)" if x == "C"
                          else "Put (沽)")
    opts = picker[picker[ds.C_CP] == cp].sort_values(ds.C_STRIKE)
    opts_by_strike = opts.set_index(ds.C_STRIKE)
    with c3:
        strike = st.selectbox(
            "Strike", opts[ds.C_STRIKE].tolist(), key="pos_strike",
            index=int((opts[ds.C_STRIKE] - spot_now).abs().argmin()),
            format_func=lambda k: (
                f"{k:g}  (last {opts_by_strike.loc[k, ds.C_CLOSE]:.4f}"
                f"{', ATM' if abs(k - spot_now) == (opts[ds.C_STRIKE] - spot_now).abs().min() else ''})"))
    row = opts_by_strike.loc[strike]
    code = row[ds.C_CODE]
    with c4:
        direction = st.selectbox("Buy / Sell", ["buy", "sell"], key="pos_dir")
    with c5:
        qty = st.number_input("Qty", 1, 500, 1, key="pos_qty")
    px = st.number_input(
        f"Entry price — latest close {float(row[ds.C_CLOSE]):.4f}",
        value=float(row[ds.C_CLOSE]), format="%.4f", key=f"pos_px_{code}")

    if "staged_legs" not in st.session_state:
        st.session_state.staged_legs = []
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Add leg"):
            st.session_state.staged_legs.append({
                "leg_code": code, "leg_name": row[ds.C_NAME], "cp": cp,
                "strike": float(strike),
                "expiry": pd.Timestamp(exp).date().isoformat(),
                "direction": 1 if direction == "buy" else -1,
                "qty": int(qty), "entry_price": float(px)})
    with b2:
        if st.button("Clear staged legs"):
            st.session_state.staged_legs = []

    if st.session_state.staged_legs:
        st.caption("Staged legs — edit direction/qty/price in the table; "
                   "select a row and press Delete (or the trash icon) to "
                   "remove it.")
        staged = pd.DataFrame(st.session_state.staged_legs)
        staged["direction"] = staged["direction"].map({1: "buy", -1: "sell"})
        edited = st.data_editor(
            staged, key=f"staged_editor_{len(staged)}", num_rows="dynamic",
            width='stretch', hide_index=True,
            disabled=["leg_code", "leg_name", "cp", "strike", "expiry"],
            column_config={
                "direction": st.column_config.SelectboxColumn(
                    "direction", options=["buy", "sell"], required=True),
                "qty": st.column_config.NumberColumn(
                    "qty", min_value=1, max_value=500, step=1),
                "entry_price": st.column_config.NumberColumn(
                    "entry_price", min_value=0.0, format="%.4f"),
            })
        edited = edited.dropna(subset=["leg_code"])  # ignore blank added rows
        st.session_state.staged_legs = [
            {"leg_code": r.leg_code, "leg_name": r.leg_name, "cp": r.cp,
             "strike": float(r.strike), "expiry": r.expiry,
             "direction": 1 if r.direction == "buy" else -1,
             "qty": int(r.qty), "entry_price": float(r.entry_price)}
            for r in edited.itertuples()]
        name = st.text_input("Position name",
                             value=f"combo {pd.Timestamp.now():%m-%d}")
        open_date = st.date_input("Open date", value=pd.Timestamp.now().date())
        if st.button("Open position", type="primary",
                     disabled=not st.session_state.staged_legs):
            pid = pos.add_position(name, u, open_date.isoformat(),
                                   st.session_state.staged_legs)
            st.session_state.staged_legs = []
            st.success(f"Tracking position {pid}")
            st.rerun()

# ============================================================== Simulate tab
with tab_sim:
    st.caption("Sandbox: build a hypothetical combo, see its payoff and "
               "estimated deposit required. Nothing here is saved -- it "
               "never touches tracked positions.")
    u_sim = st.selectbox("Underlying", list(ds.UNDERLYINGS),
                         format_func=lambda x: f"{ds.UNDERLYINGS[x]} ({x})",
                         key="sim_u")
    chain_sim = chains[u_sim]
    exps_sim = fmt_expiry_options(chain_sim)
    spot_sim = float(chain_sim.loc[chain_sim[ds.C_DATE] == chain_sim[ds.C_DATE].max(),
                                   ds.C_UND_CLOSE].iloc[0])
    st.caption(f"Latest spot: **{spot_sim:.3f}** "
               f"(as of {pd.Timestamp(chain_sim[ds.C_DATE].max()).date()})")
    s1, s2, s3, s4, s5 = st.columns([2, 1, 2, 1, 1])
    with s1:
        exp_sim = st.selectbox("Expiry", exps_sim, key="sim_exp",
                               format_func=lambda e: pd.Timestamp(e).date().isoformat())
    picker_sim = pos.contract_picker(chain_sim, expiry=exp_sim)
    with s2:
        cp_sim = st.selectbox("Call / Put", ["C", "P"], key="sim_cp",
                              format_func=lambda x: "Call (购)" if x == "C"
                              else "Put (沽)")
    opts_sim = picker_sim[picker_sim[ds.C_CP] == cp_sim].sort_values(ds.C_STRIKE)
    opts_sim_by_strike = opts_sim.set_index(ds.C_STRIKE)
    with s3:
        strike_sim = st.selectbox(
            "Strike", opts_sim[ds.C_STRIKE].tolist(), key="sim_strike",
            index=int((opts_sim[ds.C_STRIKE] - spot_sim).abs().argmin()),
            format_func=lambda k: (
                f"{k:g}  (last {opts_sim_by_strike.loc[k, ds.C_CLOSE]:.4f}"
                f"{', ATM' if abs(k - spot_sim) == (opts_sim[ds.C_STRIKE] - spot_sim).abs().min() else ''})"))
    row_sim = opts_sim_by_strike.loc[strike_sim]
    code_sim = row_sim[ds.C_CODE]
    with s4:
        direction_sim = st.selectbox("Buy / Sell", ["buy", "sell"], key="sim_dir")
    with s5:
        qty_sim = st.number_input("Qty", 1, 500, 1, key="sim_qty")
    px_sim = st.number_input(
        f"Price — latest close {float(row_sim[ds.C_CLOSE]):.4f}",
        value=float(row_sim[ds.C_CLOSE]), format="%.4f", key=f"sim_px_{code_sim}")

    if "sim_legs" not in st.session_state:
        st.session_state.sim_legs = []
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Add leg", key="sim_add_leg"):
            st.session_state.sim_legs.append({
                "leg_code": code_sim, "leg_name": row_sim[ds.C_NAME], "cp": cp_sim,
                "strike": float(strike_sim),
                "expiry": pd.Timestamp(exp_sim).date().isoformat(),
                "direction": 1 if direction_sim == "buy" else -1,
                "qty": int(qty_sim), "entry_price": float(px_sim)})
    with b2:
        if st.button("Clear staged legs", key="sim_clear_legs"):
            st.session_state.sim_legs = []

    if st.session_state.sim_legs:
        st.caption("Staged legs — edit direction/qty/price in the table; "
                   "select a row and press Delete (or the trash icon) to "
                   "remove it.")
        staged_sim = pd.DataFrame(st.session_state.sim_legs)
        staged_sim["direction"] = staged_sim["direction"].map({1: "buy", -1: "sell"})
        edited_sim = st.data_editor(
            staged_sim, key=f"sim_staged_editor_{len(staged_sim)}", num_rows="dynamic",
            width='stretch', hide_index=True,
            disabled=["leg_code", "leg_name", "cp", "strike", "expiry"],
            column_config={
                "direction": st.column_config.SelectboxColumn(
                    "direction", options=["buy", "sell"], required=True),
                "qty": st.column_config.NumberColumn(
                    "qty", min_value=1, max_value=500, step=1),
                "entry_price": st.column_config.NumberColumn(
                    "entry_price", min_value=0.0, format="%.4f"),
            })
        edited_sim = edited_sim.dropna(subset=["leg_code"])
        st.session_state.sim_legs = [
            {"leg_code": r.leg_code, "leg_name": r.leg_name, "cp": r.cp,
             "strike": float(r.strike), "expiry": r.expiry,
             "direction": 1 if r.direction == "buy" else -1,
             "qty": int(r.qty), "entry_price": float(r.entry_price)}
            for r in edited_sim.itertuples()]

        legs_df_sim = pd.DataFrame(st.session_state.sim_legs)
        st.plotly_chart(charts.payoff_figure(
            legs_df_sim, spot_sim, "Simulated P&L at expiry vs underlying price"),
            width='stretch')

        st.subheader("Estimated deposit required")
        buf1, buf2 = st.columns([1, 3])
        with buf1:
            buffer_mult = st.number_input(
                "Broker buffer ×", min_value=1.0, value=1.0, step=0.05,
                format="%.2f", key="sim_margin_buffer")
        total_deposit, breakdown = mg.compute_deposit(
            legs_df_sim, spot_sim, buffer=buffer_mult)
        with buf2:
            st.metric("Total deposit (¥)", f"{total_deposit:,.0f}")
        if not breakdown.empty:
            st.dataframe(breakdown, width='stretch', hide_index=True)
        st.caption(
            "Naked shorts use the exchange-minimum formula "
            "([mark + max(12%×spot − OTM, 7%×spot or strike)] × multiplier); "
            "a short+long pair at different strikes (same expiry/cp) is "
            "priced as a vertical spread (max(width − net credit, 0)); any "
            "other short leg (straddles, strangles, ratio spreads, unmatched "
            "legs) falls back to a conservative sum of naked margins. Marks "
            "are latest chain closes, not real settlement prices, and the "
            "broker buffer is a flat multiplier you set above — treat this "
            "as a sizing estimate, not your broker's actual margin call.")
    else:
        st.info("Add at least one leg to see the payoff chart and deposit estimate.")

# ============================================================= Backtest tab
with tab_bt:
    st.subheader("Strategy designer")
    c1, c2, c3 = st.columns(3)
    with c1:
        u = st.selectbox("Underlying", list(ds.UNDERLYINGS),
                         format_func=lambda x: f"{ds.UNDERLYINGS[x]} ({x})",
                         key="bt_u")
        template = st.selectbox("Template", bt.TEMPLATE_NAMES + ["custom"])
        width = st.number_input("Wing width (strike steps)", 1, 10, 2)
        short_delta = st.number_input("Short leg |delta|", 0.05, 0.5, 0.25,
                                      step=0.05)
        long_delta = st.number_input("Long/protect |delta|", 0.02, 0.4, 0.10,
                                     step=0.02)
    with c2:
        dte_min = st.number_input("Entry DTE min", 1, 120, 20)
        dte_max = st.number_input("Entry DTE max", 1, 180, 40)
        iv_lo = st.slider("IV percentile gate (min)", 0.0, 1.0, 0.0, 0.05)
        iv_hi = st.slider("IV percentile gate (max)", 0.0, 1.0, 1.0, 0.05)
        excl_adj = st.checkbox("Exclude adjusted contracts (name ends 'A')",
                               True)
    with c3:
        use_pt = st.checkbox("Profit target", True)
        pt = st.number_input("  target (× |entry cost|)", 0.1, 5.0, 0.5, 0.1)
        use_sl = st.checkbox("Stop loss", True)
        sl = st.number_input("  stop (× |entry cost|)", 0.1, 5.0, 1.0, 0.1)
        exit_dte = st.number_input("Exit at DTE ≤", 0, 30, 5)
        d0 = st.date_input("Backtest start", value=pd.Timestamp("2016-01-01"))

    legs = None
    if template == "custom":
        st.caption("Custom legs — offset = strike steps from ATM (calls: + is "
                   "higher strike); or select by |delta|. expiry_rank 0=entry "
                   "month, 1=next month.")
        default = pd.DataFrame([
            {"cp": "C", "direction": -1, "ratio": 1, "select": "offset",
             "value": 0.0, "expiry_rank": 0},
            {"cp": "P", "direction": -1, "ratio": 1, "select": "offset",
             "value": 0.0, "expiry_rank": 0}])
        edited = st.data_editor(default, num_rows="dynamic",
                                width='stretch')
        legs = [bt.LegSpec(r.cp, int(r.direction), int(r.ratio), r.select,
                           float(r.value), int(r.expiry_rank))
                for r in edited.itertuples()]
    else:
        legs = bt.template_legs(template, width=width,
                                short_delta=short_delta,
                                long_delta=long_delta)
        st.caption("Legs: " + " | ".join(
            f"{'+' if l.direction > 0 else '-'}{l.ratio} {l.cp} "
            f"({l.select}={l.value:g}, m{l.expiry_rank})" for l in legs))

    if st.button("Run backtest", type="primary"):
        rules = bt.Rules(dte_min=int(dte_min), dte_max=int(dte_max),
                         iv_pct_min=iv_lo, iv_pct_max=iv_hi,
                         profit_target=pt if use_pt else None,
                         stop_loss=sl if use_sl else None,
                         exit_dte=int(exit_dte), exclude_adjusted=excl_adj)
        chain = get_chain(u)
        chain = chain[chain[ds.C_DATE] >= pd.Timestamp(d0)]
        with st.spinner("Running backtest..."):
            res = bt.run_backtest(chain, legs, rules)
        st.session_state["bt_result"] = (res, u)

    if "bt_result" in st.session_state:
        res, bu = st.session_state["bt_result"]
        stats, trades, equity = res["stats"], res["trades"], res["equity"]
        if stats.get("n_trades", 0) == 0:
            st.warning("No trades generated — loosen the entry rules.")
        else:
            cols = st.columns(5)
            cols[0].metric("Trades", stats["n_trades"])
            cols[1].metric("Total P&L (¥, 1x)", f"{stats['total_pnl_yuan']:,.0f}")
            cols[2].metric("Win rate", f"{stats['win_rate']:.1%}")
            cols[3].metric("Max DD (¥)", f"{stats['max_drawdown_yuan']:,.0f}")
            cols[4].metric("Sharpe", stats["sharpe"])
            cols = st.columns(4)
            cols[0].metric("Avg win (¥)", f"{stats['avg_win_yuan']:,.0f}")
            cols[1].metric("Avg loss (¥)", f"{stats['avg_loss_yuan']:,.0f}")
            cols[2].metric("Profit factor", stats["profit_factor"])
            cols[3].metric("Avg days held", stats["avg_days_held"])

            st.plotly_chart(charts.backtest_figure(equity, trades,
                                                   get_ohlc(bu)),
                            width='stretch')
            st.plotly_chart(charts.pnl_histogram(trades),
                            width='stretch')
            show = trades.copy()
            show["pnl_yuan"] = (show["pnl"] * ds.MULTIPLIER).round(0)
            st.dataframe(show[["entry_date", "exit_date", "legs_desc",
                               "cost", "exit_reason", "days_held",
                               "pnl_yuan"]],
                         width='stretch', hide_index=True)
            st.download_button("Download trades CSV",
                               show.to_csv(index=False).encode(),
                               "backtest_trades.csv")

# ======================================================= Macro Dashboard tab
# "Market" (etf50/etf300/idx_sh) is intentionally not shown here -- it's
# covered by the Technical tab instead. Valuation and northbound flow were
# dropped entirely (see dashboard_data.py for why); real-economy fetchers
# were swapped to fresher substitute sources where the original stalled.
with tab_macro:
    st.subheader("Macro Dashboard")
    st.caption("Liquidity, rates, FX, vol and real-economy series, each kept at its own "
               "native release frequency (daily where the source actually publishes daily, "
               "monthly for official macro prints). Updated by a cron job 3x/day on this "
               "server (run_dashboard_update.sh); a series with no new release since the "
               "last run is a no-op, so this naturally handles differing release schedules "
               "with no per-series calendar to maintain.")

    stale_df = DD.all_staleness()
    n_stale = int(stale_df["is_stale"].sum())
    if n_stale:
        st.warning(f"{n_stale} of {len(stale_df)} series look stale (no fresh print within "
                   f"the expected lag) -- see flags below.")
    else:
        st.success(f"All {len(stale_df)} series up to date.")

    SKIP_GROUPS = {"Market"}  # shown on the Technical tab instead
    group_order = [g for g in dict.fromkeys(v[1] for v in DD.SERIES.values())
                  if g not in SKIP_GROUPS]
    for group in group_order:
        keys = [k for k, v in DD.SERIES.items() if v[1] == group]
        st.markdown(f"**{group}**")
        cols = st.columns(min(len(keys), 4))
        for i, key in enumerate(keys):
            info = DD.staleness(key)
            with cols[i % len(cols)]:
                if info["value"] is None:
                    st.metric(info["label"], "no data")
                    continue
                st.metric(info["label"], f"{info['value']:,.3f}{info['unit']}")
                if info["is_stale"]:
                    st.caption(f":orange[stale -- last {info['last_date'].date()} "
                               f"({info['days_old']}d ago)]")
                else:
                    st.caption(f"as of {info['last_date'].date()}")
                with st.expander("chart (1Y)"):
                    s_hist = DD.load_series(key)
                    cutoff = s_hist.index.max() - pd.DateOffset(years=1)
                    st.plotly_chart(
                        charts.macro_series_figure(s_hist[s_hist.index >= cutoff],
                                                   info["label"]),
                        width='stretch', key=f"macro_chart_{key}")

# ============================================================== Technical tab
# Pure logic lives in alerts_core.py (shared with push_alerts.py) -- these
# are thin aliases so the rest of this file doesn't need to change.
_ma150_state = ac.ma150_state
_next_quarterly_listing_estimate = ac.next_quarterly_listing_estimate
_suggest_put_debit_spread = ac.suggest_put_debit_spread


# --------------------------------------------------- vol-signal alerts (long straddle)
# Configs/logic live in alerts_core.py -- see its docstrings for the frozen-
# pick provenance. Cached here (not in alerts_core) since caching is a
# Streamlit-UI concern, not shared with push_alerts.py.
VOL_SIGNAL_ALERTS = ac.VOL_SIGNAL_ALERTS
_vol_signal_now = st.cache_data(ttl=86400, show_spinner=False)(ac.vol_signal_now)
_vol_trigger_level = st.cache_data(ttl=86400, show_spinner=False)(ac.vol_trigger_level)
_suggest_atm_straddle = ac.suggest_atm_straddle


# --------------------------------------------------- vol-signal alerts (short straddle)
SHORT_VOL_SIGNAL_ALERTS = ac.SHORT_VOL_SIGNAL_ALERTS
_short_vol_signal_now = st.cache_data(ttl=86400, show_spinner=False)(ac.short_vol_signal_now)
_vrp_trigger_level = st.cache_data(ttl=86400, show_spinner=False)(ac.vrp_trigger_level)
_suggest_atm_short_straddle = ac.suggest_atm_short_straddle


# --------------------------------------------------- vol-signal alerts (delta-targeted strangle, short)
DELTA_STRANGLE_ALERTS = ac.DELTA_STRANGLE_ALERTS

# --------------------------------------------------- vol-signal alerts (iv_pct, short straddle)
IV_PCT_SHORT_ALERTS = ac.IV_PCT_SHORT_ALERTS

# --------------------------------------------------- Alerts tab: side/strategy menu constants
# Shared between the selectbox options and the Summary section's jump
# buttons below, so the two can't silently drift out of sync. Defined in
# alerts_core.py so push_alerts.py can reuse the same labels.
ALERTS_SIDE_BUY = ac.ALERTS_SIDE_BUY
ALERTS_SIDE_SELL = ac.ALERTS_SIDE_SELL
ALERTS_BUY_STRATEGIES = ac.ALERTS_BUY_STRATEGIES
ALERTS_SELL_STRATEGIES = ac.ALERTS_SELL_STRATEGIES


def _jump_to_strategy(side_val: str, strategy_val: str, strategy_key: str) -> None:
    """Button on_click callback: set the Side/Strategy widget state directly
    so the next rerun lands on that section, instead of making the user
    manually pick both dropdowns after spotting an active row in Summary."""
    st.session_state["alerts_side"] = side_val
    st.session_state[strategy_key] = strategy_val


_suggest_delta_strangle = ac.suggest_delta_strangle


# --------------------------------------------------- vol-signal alerts (spike+vrp combo, short straddle)
COMBO_VOL_SIGNAL_ALERTS = ac.COMBO_VOL_SIGNAL_ALERTS

# --------------------------------------------------- vol-signal alerts (iv_pct + vrp AND-combo, short straddle)
IV_PCT_VRP_COMBO_ALERTS = ac.IV_PCT_VRP_COMBO_ALERTS

_combo_signal_now_uncached = ac.combo_signal_now


@st.cache_data(ttl=86400, show_spinner=False)
def _combo_signal_now(u: str, spike_window: int, spike_pct: float, vrp_threshold: float) -> dict | None:
    return _combo_signal_now_uncached(u, spike_window, spike_pct, vrp_threshold,
                                       chain_loader=_load_any_chain)


with tab_alerts:
    st.subheader("Alerts")
    st.caption("Entry-timing alerts for tested signals, grouped by directional bias. "
               "Descriptive only -- no orders are placed, size and confirm before trading.")

    st.markdown("#### Summary")
    st.caption("Every tracked alert at a glance -- click **View →** to jump straight to that "
               "strategy's detail section below (same as picking it from Side/Strategy manually).")

    # Evaluation logic lives in alerts_core.py (shared with push_alerts.py) --
    # this just maps each row's side/strategy_idx to the widget values the
    # jump buttons need.
    summary_rows = ac.evaluate_summary(DD.load_series("idx_sh"), chain_loader=_load_any_chain)

    for i, row in enumerate(summary_rows):
        side_val = ALERTS_SIDE_BUY if row["side"] == "buy" else ALERTS_SIDE_SELL
        strategies = ALERTS_BUY_STRATEGIES if row["side"] == "buy" else ALERTS_SELL_STRATEGIES
        strategy_key = "alerts_buy_strategy" if row["side"] == "buy" else "alerts_sell_strategy"
        strategy_val = strategies[row["strategy_idx"]]

        r1, r2, r3, r4 = st.columns([2.6, 1.8, 2.6, 1])
        r1.markdown(f"**{row['strategy']}**")
        r2.markdown(row["instrument"])
        r3.markdown(f"\U0001F514 **ACTIVE** · {row['status']}" if row["active"] else f"⚪ inactive · {row['status']}")
        r4.button("View →", key=f"alerts_jump_{i}", on_click=_jump_to_strategy,
                  args=(side_val, strategy_val, strategy_key))

    st.divider()

    side = st.radio("Side", [ALERTS_SIDE_BUY, ALERTS_SIDE_SELL],
                     horizontal=True, key="alerts_side")

    if side.startswith("Buy"):
        strategy = st.selectbox("Strategy", ALERTS_BUY_STRATEGIES, key="alerts_buy_strategy")
        st.divider()

        if strategy.startswith("SSE Composite"):
            st.markdown("#### Entry timing — SSE Composite MA150 signal")
            st.caption("From the MA-crossover investigation (analysis/macro_model/"
                       "ma_signal_investigation_leaderboard.md): SSE Composite closing below its "
                       "150-day MA preceded weak 90-120 trading day forward returns, validated with a "
                       "discovery/holdout split and backtested as 510050/510300 options. Descriptive "
                       "only -- size and confirm before trading, and note ~70% of MA crossings revert "
                       "within 10 trading days (the leaderboard's winning trades entered immediately "
                       "regardless, so this isn't a recommendation to wait, just context).")

            s_idx_for_signal = DD.load_series("idx_sh")
            ma_state = _ma150_state(s_idx_for_signal)
            if ma_state is None:
                st.caption("Not enough SSE Composite history loaded to compute a 150-day MA yet.")
            else:
                e1, e2, e3 = st.columns(3)
                e1.metric("SSE Composite vs. MA150", f"{ma_state['last_px']:,.1f}", f"{ma_state['dist_pct']:+.2%}")
                e2.metric("Status", "Below (signal)" if ma_state["below"] else "Above (no signal)")
                e3.metric("Days in this state", f"{ma_state['run_days']}", help=f"as of {ma_state['asof'].date()}")

                if ma_state["below"]:
                    st.success(f"Signal active — SSE Composite has closed below its 150-day MA for "
                               f"{ma_state['run_days']} trading day(s).")
                    st.markdown("**Suggested combination (put debit spread):**")
                    sc1, sc2 = st.columns(2)
                    for col, u_key, u_label in [(sc1, "510050", "50ETF"), (sc2, "510300", "300ETF")]:
                        with col:
                            st.markdown(f"*{u_label} ({u_key})*")
                            try:
                                chain = get_chain(u_key)
                                suggestion = _suggest_put_debit_spread(chain)
                            except Exception as e:
                                suggestion = None
                                st.caption(f":gray[chain unavailable: {e}]")
                            if suggestion is None:
                                st.caption("No suitable contract found in the last chain snapshot.")
                            elif suggestion.get("insufficient_dte"):
                                st.warning(
                                    f"No sufficiently long-dated contract listed (longest available: "
                                    f"{suggestion['best_available_dte']} DTE, need ~140-170). The "
                                    f"validated ~120-trading-day structure isn't tradeable on today's "
                                    f"chain — this matches roughly 76% of historical signal dates where "
                                    f"the same thing happened (see the rolling-vs-fixed-DTE finding in "
                                    f"the leaderboard). A shorter-dated substitute was tested separately "
                                    f"and found to underperform for this signal, not shown here.")
                                st.caption(
                                    f"Currently listed DTEs: {suggestion['available_dtes']}. Likely cause: "
                                    f"the quarterly-cycle contract (Mar/Jun/Sep/Dec expiries, listed "
                                    f"~8 months ahead, always at the start of Jan/Apr/Jul/Oct) that would "
                                    f"give enough runway hasn't been listed yet, or lacks this strike yet. "
                                    f"Next quarterly listing window: **~{suggestion['next_quarterly_listing_est']:%Y-%m-%d}** "
                                    f"(rough calendar estimate only — the listing itself may not immediately "
                                    f"have the needed strike; see ma_signal_investigation_leaderboard.md's "
                                    f"listing-cycle finding). This is a market-structure gap, not something "
                                    f"to wait out by delaying entry — delayed entry was tested separately and "
                                    f"did not improve results.")
                                st.caption(f"chain as of {suggestion['asof']}")
                            else:
                                st.markdown(
                                    f"- Buy put, strike **{suggestion['buy_strike']:.2f}** @ "
                                    f"{suggestion['buy_price']:.4f}\n"
                                    f"- Sell put, strike **{suggestion['sell_strike']:.2f}** @ "
                                    f"{suggestion['sell_price']:.4f}\n"
                                    f"- Expiry **{suggestion['expiry']}** ({suggestion['dte']} DTE)\n"
                                    f"- Net debit: **{suggestion['net_debit']:.4f}**\n"
                                    f"- Stop-gain target (per leaderboard, +100% of debit): "
                                    f"**{suggestion['stop_gain_price']:.4f}**"
                                )
                                st.caption(f"chain as of {suggestion['asof']}")
                else:
                    st.caption("No active signal — SSE Composite is currently above its 150-day MA.")


        else:
            st.markdown("#### Entry timing — volatility signals (long straddle)")
            st.caption("Discovery/holdout-validated (<2024-01-01 / >=2024-01-01 split): entry when the "
                       "signal is at a low percentile of its own recent history, betting a quiet/cheap "
                       "stretch precedes a move big enough to clear the debit paid. rv_pct = trailing "
                       "realized vol percentile; iv_pct = ATM implied vol percentile (real exchange IV "
                       "for 300ETF, Black-Scholes-inverted for 创业板ETF -- no native IV column there). "
                       "vrp is not shown -- it failed discovery/holdout validation on every underlying "
                       "tested. Descriptive only, same as the MA150 section above -- not an order.")

            for cfg in VOL_SIGNAL_ALERTS:
                sig_now = _vol_signal_now(cfg["u"], cfg["module"], cfg["signal"])
                active = bool(sig_now) and sig_now["value"] <= cfg["threshold"]
                trig = _vol_trigger_level(cfg["u"], cfg["module"], cfg["signal"], cfg["threshold"])
                gap = (trig["current"] - trig["level"]) if trig else None  # positive = not yet triggered

                label = f"{cfg['name']} ({cfg['u']}) {cfg['signal']}"
                if active:
                    st.success(f"🔔 **{label} — ACTIVE** as of {sig_now['asof'].date()}")
                else:
                    gap_txt = f"needs to drop {gap:.2%} more" if gap is not None and gap > 0 else "n/a"
                    st.info(f"{label} — inactive · {gap_txt}")

                v1, v2, v3, v4 = st.columns(4)
                v1.metric(f"{cfg['signal'].split('_')[0]} now (raw)", f"{trig['current']:.2%}" if trig else "n/a",
                          help="the actual realized/implied vol level -- what the percentile in the next "
                               "column is ranking")
                v2.metric(f"{cfg['signal']}", f"{sig_now['value']:.1%}" if sig_now else "n/a",
                          help=f"rolling percentile rank -- fires when <= {cfg['threshold']:.0%}")
                v3.metric("Trigger level (raw)", f"{trig['level']:.2%}" if trig else "n/a",
                          help="the raw vol level that would put today's percentile exactly at the "
                               f"{cfg['threshold']:.0%} threshold")
                if trig:
                    v4.metric("Distance to trigger", f"{-gap:+.2%}",
                              help="current raw value minus trigger level, in percentage (vol) points -- "
                                   "positive means already past trigger, negative means "
                                   "still needs to drop this much further")
                else:
                    v4.metric("Distance to trigger", "n/a")
                if active:
                    try:
                        chain = _load_any_chain(cfg["u"])
                        straddle = _suggest_atm_straddle(chain)
                    except Exception as e:
                        straddle = None
                        st.caption(f":gray[chain unavailable: {e}]")
                    if straddle:
                        st.markdown(
                            f"**ATM straddle**, strike {straddle['strike']:.3f}, "
                            f"expiry {straddle['expiry'].date()} ({straddle['dte']}d): "
                            f"call {straddle['call_price']:.4f} + put {straddle['put_price']:.4f} "
                            f"= debit **{straddle['debit']:.4f}**")
                    else:
                        st.caption("No 10-45 DTE contract found in the latest chain.")
                pt_label = "none (hold to max_hold/expiry)" if cfg["profit_target"] is None else f"{cfg['profit_target']:+.0%} of debit"
                st.caption(f":gray[Exit: profit target {pt_label}, OR max_hold={cfg['max_hold']}d, OR expiry "
                           f"(no stop-loss -- debit paid already caps the loss)]")

    else:
        strategy = st.selectbox("Strategy", ALERTS_SELL_STRATEGIES, key="alerts_sell_strategy")
        st.divider()

        if strategy.startswith("Short straddle — VRP"):
            st.markdown("#### Entry timing — volatility signals (short straddle)")
            st.caption("Discovery/holdout-validated (<2024-01-01 / >=2024-01-01 split): entry when ATM "
                       "implied vol is rich relative to trailing 20d realized vol (vrp = ATM IV minus "
                       "trailing 20d realized vol, vol points), betting on vol mean-reversion/crush -- "
                       "the mirror of the cheap-vol long straddle above. Naked (uncapped risk, sized to "
                       "exchange-minimum margin), meant to be held up to the listed number of trading "
                       "days with a 2x-credit stop-loss (not shown here -- see vol_rich_short_straddle_"
                       "backtest.py). Descriptive only, same as the sections above -- not an order, size "
                       "and manage risk before trading. rv_pct (realized vol percentile alone, no IV "
                       "comparison) is not shown -- it failed validation on both underlyings tested.")

            for cfg in SHORT_VOL_SIGNAL_ALERTS:
                sig_now = _short_vol_signal_now(cfg["u"], cfg["signal"])
                active = bool(sig_now) and sig_now["value"] >= cfg["threshold"]
                vtrig = _vrp_trigger_level(cfg["u"], cfg["threshold"])
                gap = (vtrig["needed"] - vtrig["iv_now"]) if vtrig else None  # positive = not yet triggered

                label = f"{cfg['name']} ({cfg['u']}) {cfg['signal']}"
                if active:
                    st.success(f"🔔 **{label} — ACTIVE** as of {sig_now['asof'].date()}")
                else:
                    gap_txt = f"IV needs {gap:.2%} more to trigger" if gap is not None and gap > 0 else "n/a"
                    st.info(f"{label} — inactive · {gap_txt}")

                w1, w2, w3, w4, w5 = st.columns(5)
                w1.metric("IV now (raw)", f"{vtrig['iv_now']:.2%}" if vtrig else "n/a",
                          help="ATM implied vol -- the rich side of the vrp spread")
                w2.metric("Trailing RV now (raw)", f"{vtrig['rv_now']:.2%}" if vtrig else "n/a",
                          help="trailing 20d realized vol -- the cheap side of the vrp spread")
                w3.metric(f"{cfg['signal']} now", f"{sig_now['value']:+.1%}" if sig_now else "n/a",
                          help=f"= IV minus trailing RV; fires when >= {cfg['threshold']:+.0%}")
                w4.metric("IV needed to trigger", f"{vtrig['needed']:.2%}" if vtrig else "n/a",
                          help="the IV level (holding trailing RV fixed) that would put vrp exactly at "
                               f"the {cfg['threshold']:+.0%} threshold")
                if vtrig:
                    w5.metric("Distance to trigger", f"{-gap:+.2%}",
                              help="current IV minus IV-needed, in percentage (vol) points -- positive "
                                   "means already past trigger")
                else:
                    w5.metric("Distance to trigger", "n/a")
                if active:
                    try:
                        chain = _load_any_chain(cfg["u"])
                        straddle = _suggest_atm_short_straddle(chain)
                    except Exception as e:
                        straddle = None
                        st.caption(f":gray[chain unavailable: {e}]")
                    if straddle:
                        st.markdown(
                            f"**Short ATM straddle**, strike {straddle['strike']:.3f}, "
                            f"expiry {straddle['expiry'].date()} ({straddle['dte']}d): "
                            f"call {straddle['call_price']:.4f} + put {straddle['put_price']:.4f} "
                            f"= credit **{straddle['credit']:.4f}**, "
                            f"margin ~**{straddle['margin_yuan']:,.0f}** yuan/contract, "
                            f"hold up to {cfg['max_hold']}d")
                    else:
                        st.caption("No 10-45 DTE contract found in the latest chain.")
                st.caption(f":gray[Exit: 2.0x-credit stop-loss, OR max_hold={cfg['max_hold']}d, OR expiry "
                           f"(naked -- uncapped risk needs the stop-loss, unlike the long straddle above)]")

        elif strategy.startswith("Short straddle — iv_pct"):
            st.caption("Discovery/holdout-validated (<2024-01-01 / >=2024-01-01 split): entry when "
                       "raw ATM IV itself is at a HIGH percentile of its own rolling history -- "
                       "distinct from the VRP signal (which compares IV to realized vol, not to IV's "
                       "own history). Only 2 of 28-38 entry dates overlap with the VRP-only signal on "
                       "50ETF -- a largely independent mechanism, not a re-discovery of the same "
                       "trades. 50ETF: 16/16 eligible sweep cells agree-positive, monotonic in both "
                       "threshold and hold length; 300ETF: 7/10 (weaker at the loosest threshold=0.70, "
                       "clean from 0.80 up). Same naked/margin-sized/stop-loss construction as the VRP "
                       "version. Descriptive only -- not an order.")

            for cfg in IV_PCT_SHORT_ALERTS:
                sig_now = _short_vol_signal_now(cfg["u"], "iv_pct")
                active = bool(sig_now) and sig_now["value"] >= cfg["threshold"]
                trig = _vol_trigger_level(cfg["u"], cfg["module"], "iv_pct", cfg["threshold"])
                gap = (trig["level"] - trig["current"]) if trig else None  # positive = not yet triggered

                label = f"{cfg['name']} ({cfg['u']}) iv_pct"
                if active:
                    st.success(f"🔔 **{label} — ACTIVE** as of {sig_now['asof'].date()}")
                else:
                    gap_txt = f"needs to rise {gap:.2%} more" if gap is not None and gap > 0 else "n/a"
                    st.info(f"{label} — inactive · {gap_txt}")

                v1, v2, v3, v4 = st.columns(4)
                v1.metric("IV now (raw)", f"{trig['current']:.2%}" if trig else "n/a",
                          help="ATM implied vol -- what the percentile in the next column is ranking")
                v2.metric("iv_pct now", f"{sig_now['value']:.1%}" if sig_now else "n/a",
                          help=f"rolling percentile rank -- fires when >= {cfg['threshold']:.0%}")
                v3.metric("Trigger level (raw)", f"{trig['level']:.2%}" if trig else "n/a",
                          help="the raw IV level that would put today's percentile exactly at the "
                               f"{cfg['threshold']:.0%} threshold")
                if trig:
                    v4.metric("Distance to trigger", f"{-gap:+.2%}",
                              help="current raw IV minus trigger level, in percentage (vol) points -- "
                                   "positive means already past trigger")
                else:
                    v4.metric("Distance to trigger", "n/a")
                if active:
                    try:
                        chain = _load_any_chain(cfg["u"])
                        straddle = _suggest_atm_short_straddle(chain)
                    except Exception as e:
                        straddle = None
                        st.caption(f":gray[chain unavailable: {e}]")
                    if straddle:
                        st.markdown(
                            f"**Short ATM straddle**, strike {straddle['strike']:.3f}, "
                            f"expiry {straddle['expiry'].date()} ({straddle['dte']}d): "
                            f"call {straddle['call_price']:.4f} + put {straddle['put_price']:.4f} "
                            f"= credit **{straddle['credit']:.4f}**, "
                            f"margin ~**{straddle['margin_yuan']:,.0f}** yuan/contract, "
                            f"hold up to {cfg['max_hold']}d")
                    else:
                        st.caption("No 10-45 DTE contract found in the latest chain.")
                st.caption(f":gray[Exit: 2.0x-credit stop-loss, OR max_hold={cfg['max_hold']}d, OR expiry]")

        elif strategy.startswith("Delta-targeted"):
            st.markdown("#### Entry timing — volatility signal, delta-targeted strangle (exploratory)")
            st.caption(":orange[**Preliminary, not yet held to the same validation bar as the sections "
                       "above** -- each underlying only tested at its own one already-frozen "
                       "(threshold, max_hold) cell. See project memory for the full joint sweeps.] Same "
                       "vrp entry trigger as the ATM short straddle above (yesterday's ATM IV minus "
                       "trailing 20d realized vol >= threshold, same thresholds too); the difference is "
                       "which strikes get sold: instead of the ATM call+put (same strike), sell whichever "
                       "listed call/put strikes have Delta closest to +/-target_delta -- a real strangle, "
                       "independently-chosen legs, not a straddle. Backtest found this roughly matches the "
                       "ATM straddle's return on less margin and materially less drawdown at "
                       "target_delta~0.30 on both 50ETF and 300ETF, though the exact optimal delta drifted "
                       "between 0.20-0.40 depending on threshold in testing (treat 0.30 as directional, not "
                       "precise) and 300ETF's holdout replication is notably weaker than 50ETF's (bigger "
                       "discovery-to-holdout falloff). Same naked/margin-sized/stop-loss construction as "
                       "the ATM version. Descriptive only -- not an order.")

            for cfg in DELTA_STRANGLE_ALERTS:
                sig_now = _short_vol_signal_now(cfg["u"], "vrp")
                active = bool(sig_now) and sig_now["value"] >= cfg["threshold"]
                vtrig = _vrp_trigger_level(cfg["u"], cfg["threshold"])
                gap = (vtrig["needed"] - vtrig["iv_now"]) if vtrig else None

                label = f"{cfg['name']} ({cfg['u']}) vrp, target_delta={cfg['target_delta']:.2f}"
                if active:
                    st.success(f"🔔 **{label} — ACTIVE** as of {sig_now['asof'].date()}")
                else:
                    gap_txt = f"IV needs {gap:.2%} more to trigger" if gap is not None and gap > 0 else "n/a"
                    st.info(f"{label} — inactive · {gap_txt}")

                if active:
                    try:
                        chain = _load_any_chain(cfg["u"])
                        strangle = _suggest_delta_strangle(chain, cfg["target_delta"])
                    except Exception as e:
                        strangle = None
                        st.caption(f":gray[chain unavailable: {e}]")
                    if strangle:
                        st.markdown(
                            f"**Short strangle**, expiry {strangle['expiry'].date()} ({strangle['dte']}d): "
                            f"call strike {strangle['call_strike']:.3f} (delta {strangle['call_delta']:+.2f}) "
                            f"@ {strangle['call_price']:.4f} + "
                            f"put strike {strangle['put_strike']:.3f} (delta {strangle['put_delta']:+.2f}) "
                            f"@ {strangle['put_price']:.4f} "
                            f"= credit **{strangle['credit']:.4f}**, "
                            f"margin ~**{strangle['margin_yuan']:,.0f}** yuan/contract, "
                            f"hold up to {cfg['max_hold']}d")
                    else:
                        st.caption("No 10-45 DTE contract with valid Delta found in the latest chain.")
                st.caption(f":gray[Exit: 2.0x-credit stop-loss, OR max_hold={cfg['max_hold']}d, OR expiry]")

        elif strategy.startswith("IV spike"):
            st.markdown("#### Entry timing — volatility signals (IV spike + VRP combo, short straddle)")
            st.caption("Discovery/holdout-validated combo (iv_spike_vrp_combo_short_straddle_backtest.py, "
                       "same split as above): requires BOTH a same-day ATM IV spike (>= spike_pct over a "
                       "trailing window) AND yesterday's VRP at or above threshold. Every eligible "
                       "discovery/holdout cell agreed in sign on both 50ETF (12/12) and 300ETF (6/6) -- "
                       "the strongest short-straddle filter tested so far. Same naked/margin-sized/"
                       "stop-loss construction as the vrp-only section above. Descriptive only -- not an "
                       "order.")

            for cfg in COMBO_VOL_SIGNAL_ALERTS:
                c_now = _combo_signal_now(cfg["u"], cfg["spike_window"], cfg["spike_pct"], cfg["vrp_threshold"])
                label = f"{cfg['name']} ({cfg['u']})"
                if c_now is None:
                    st.info(f"{label} — n/a (not enough IV history yet)")
                    continue

                status = ("both conditions met" if c_now["active"] else
                         "spike only (vrp not met)" if c_now["ok_spike"] else
                         "vrp only (no spike)" if c_now["ok_vrp"] else "neither condition met")
                stale_note = (f" — *today's official IV not posted yet, showing as of "
                              f"{c_now['asof'].date()} (chain through {c_now['chain_latest'].date()})*"
                              if c_now["stale"] else f" as of {c_now['asof'].date()}")
                if c_now["active"]:
                    st.success(f"🔔 **{label} — ACTIVE** ({status}){stale_note}")
                elif c_now["ok_spike"] or c_now["ok_vrp"]:
                    st.warning(f"{label} — one of two conditions met ({status}){stale_note}")
                else:
                    st.info(f"{label} — inactive ({status}){stale_note}")

                st.caption(f"Spike condition (same-day IV vs. {cfg['spike_window']}d ago, as of "
                           f"{c_now['asof'].date()}):")
                x1, x2, x3, x4 = st.columns(4)
                x1.metric("IV now (raw)", f"{c_now['iv_now']:.2%}",
                          help="ATM implied vol as of the spike-check date")
                x2.metric(f"IV {cfg['spike_window']}d ago (raw)", f"{c_now['iv_base']:.2%}",
                          help="base IV the spike is measured against")
                x3.metric("Spike trigger level (raw)", f"{c_now['spike_needed']:.2%}",
                          help=f"= base IV x {1 + cfg['spike_pct']:.2f} (spike_pct={cfg['spike_pct']:+.0%})")
                spike_gap = c_now["iv_now"] - c_now["spike_needed"]
                x4.metric("Distance to trigger", f"{spike_gap:+.2%}",
                          help="positive means the spike condition is met")

                st.caption(f"VRP condition (yesterday's IV/RV, "
                           f"{c_now['yst'].date() if c_now['yst'] is not None else 'n/a'}):")
                y1, y2, y3, y4, y5 = st.columns(5)
                y1.metric("IV yesterday (raw)", f"{c_now['iv_yst']:.2%}" if c_now["iv_yst"] is not None else "n/a")
                y2.metric("Trailing RV yesterday (raw)",
                          f"{c_now['rv_yst']:.2%}" if c_now["rv_yst"] is not None else "n/a")
                y3.metric("vrp yesterday", f"{c_now['vrp_yst']:+.2%}" if c_now["vrp_yst"] is not None else "n/a",
                          help=f"= IV minus trailing RV; fires when >= {cfg['vrp_threshold']:+.0%}")
                y4.metric("IV needed to trigger",
                          f"{c_now['iv_needed_yst']:.2%}" if c_now["iv_needed_yst"] is not None else "n/a",
                          help="the IV level (holding yesterday's trailing RV fixed) that would put vrp "
                               f"exactly at the {cfg['vrp_threshold']:+.0%} threshold")
                if c_now["iv_yst"] is not None and c_now["iv_needed_yst"] is not None:
                    vrp_gap = c_now["iv_yst"] - c_now["iv_needed_yst"]
                    y5.metric("Distance to trigger", f"{vrp_gap:+.2%}",
                              help="positive means the vrp condition is met")
                else:
                    y5.metric("Distance to trigger", "n/a")

                if c_now["active"]:
                    try:
                        chain = _load_any_chain(cfg["u"])
                        straddle = _suggest_atm_short_straddle(chain)
                    except Exception as e:
                        straddle = None
                        st.caption(f":gray[chain unavailable: {e}]")
                    if straddle:
                        st.markdown(
                            f"**Short ATM straddle**, strike {straddle['strike']:.3f}, "
                            f"expiry {straddle['expiry'].date()} ({straddle['dte']}d): "
                            f"call {straddle['call_price']:.4f} + put {straddle['put_price']:.4f} "
                            f"= credit **{straddle['credit']:.4f}**, "
                            f"margin ~**{straddle['margin_yuan']:,.0f}** yuan/contract, "
                            f"hold up to {cfg['max_hold']}d")
                    else:
                        st.caption("No 10-45 DTE contract found in the latest chain.")
                st.caption(f":gray[Exit: 2.0x-credit stop-loss, OR max_hold={cfg['max_hold']}d, OR expiry]")

        else:
            st.markdown("#### Entry timing — volatility signals (iv_pct + VRP AND-combo, short straddle)")
            st.caption(":orange[**Preliminary, not yet held to the same validation bar as the "
                       "single-signal sections above** -- 50ETF's frozen cell has only n=6 holdout "
                       "trades (below the usual n>=15 bar) and 300ETF's sample is too thin to freeze a "
                       "pick at all, so 300ETF isn't shown here.] Requires BOTH yesterday's iv_pct AND "
                       "yesterday's vrp at or above their own thresholds simultaneously -- the two "
                       "signals only overlap on ~7% of entry dates individually (project memory), so "
                       "ANDing them is a much stricter filter than either alone, not a re-labeling of "
                       "the same trades. The point of this section isn't higher average return (single-"
                       "signal sections already do fine on that) -- it's lower tail risk: full-sample "
                       "backtest max_drawdown is -2,323 yuan and the worst trade is -12.2%, vs. the "
                       "iv_pct-only signal's own worst trade of -28% (Dec 2017), which this AND filter "
                       "excludes outright because vrp wasn't rich that day. Same naked/margin-sized/"
                       "stop-loss construction as the sections above. Descriptive only -- not an order.")

            for cfg in IV_PCT_VRP_COMBO_ALERTS:
                iv_now = _short_vol_signal_now(cfg["u"], "iv_pct")
                vrp_now = _short_vol_signal_now(cfg["u"], "vrp")
                ok_iv = bool(iv_now) and iv_now["value"] >= cfg["iv_pct_threshold"]
                ok_vrp = bool(vrp_now) and vrp_now["value"] >= cfg["vrp_threshold"]
                active = ok_iv and ok_vrp
                label = f"{cfg['name']} ({cfg['u']}) iv_pct+vrp"

                if not iv_now or not vrp_now:
                    st.info(f"{label} — n/a (not enough history yet)")
                    continue

                status = ("both conditions met" if active else
                         "iv_pct only (vrp not met)" if ok_iv else
                         "vrp only (iv_pct not met)" if ok_vrp else "neither condition met")
                asof = max(iv_now["asof"], vrp_now["asof"])
                if active:
                    st.success(f"🔔 **{label} — ACTIVE** ({status}) as of {asof.date()}")
                elif ok_iv or ok_vrp:
                    st.warning(f"{label} — one of two conditions met ({status}) as of {asof.date()}")
                else:
                    st.info(f"{label} — inactive ({status}) as of {asof.date()}")

                z1, z2, z3, z4 = st.columns(4)
                z1.metric("iv_pct now", f"{iv_now['value']:.1%}",
                          help=f"rolling percentile of raw ATM IV vs its own history; fires when "
                               f">= {cfg['iv_pct_threshold']:.0%}")
                z2.metric("iv_pct threshold", f"{cfg['iv_pct_threshold']:.0%}")
                z3.metric("vrp now", f"{vrp_now['value']:+.1%}",
                          help=f"ATM IV minus trailing 20d realized vol; fires when "
                               f">= {cfg['vrp_threshold']:+.0%}")
                z4.metric("vrp threshold", f"{cfg['vrp_threshold']:+.0%}")

                if active:
                    try:
                        chain = _load_any_chain(cfg["u"])
                        straddle = _suggest_atm_short_straddle(chain)
                    except Exception as e:
                        straddle = None
                        st.caption(f":gray[chain unavailable: {e}]")
                    if straddle:
                        st.markdown(
                            f"**Short ATM straddle**, strike {straddle['strike']:.3f}, "
                            f"expiry {straddle['expiry'].date()} ({straddle['dte']}d): "
                            f"call {straddle['call_price']:.4f} + put {straddle['put_price']:.4f} "
                            f"= credit **{straddle['credit']:.4f}**, "
                            f"margin ~**{straddle['margin_yuan']:,.0f}** yuan/contract, "
                            f"hold up to {cfg['max_hold']}d")
                    else:
                        st.caption("No 10-45 DTE contract found in the latest chain.")
                st.caption(f":gray[Exit: 2.0x-credit stop-loss, OR max_hold={cfg['max_hold']}d, OR expiry]")


with tab_ta:
    st.subheader("Technical Analysis")
    st.caption("Price with moving averages and rolling swing-based support/resistance. "
               "Analytics only -- not a signal, no orders are placed.")

    TA_MAP = {"Shanghai Composite": "idx_sh", "50ETF": "etf50", "300ETF": "etf300"}
    ta_choice = st.selectbox("Instrument", list(TA_MAP.keys()))
    ta_key = TA_MAP[ta_choice]
    s_ta = DD.load_series(ta_key)

    live_asof = None
    if ta_key in LIVE_QUOTE_UNDERLYING:
        try:
            live_spot, live_date = get_live_quote(LIVE_QUOTE_UNDERLYING[ta_key])
            live_ts = pd.Timestamp(live_date)
            if len(s_ta) == 0 or live_ts >= s_ta.index.max():
                s_ta = pd.concat(
                    [s_ta[s_ta.index < live_ts], pd.Series([live_spot], index=[live_ts])]
                ).sort_index()
                live_asof = live_ts
        except Exception as e:  # market closed / feed blocked -- fall back silently
            st.caption(f":gray[live quote unavailable: {e}]")

    if not len(s_ta):
        st.warning("No data stored for this instrument yet -- run run_dashboard_update.sh.")
    else:
        ta_lookback = st.radio("lookback", ["6M", "1Y", "3Y", "5Y", "All"], horizontal=True,
                               index=2, key="ta_lookback")
        if ta_lookback != "All":
            if ta_lookback.endswith("M"):
                cutoff = s_ta.index.max() - pd.DateOffset(months=int(ta_lookback[:-1]))
            else:
                cutoff = s_ta.index.max() - pd.DateOffset(years=int(ta_lookback[:-1]))
        else:
            cutoff = s_ta.index.min()

        ma_windows = st.multiselect("Moving averages (bars)", [10, 20, 50, 100, 150],
                                    default=[20, 50, 150])
        sr_window = st.slider("Support/resistance swing window (bars)", 5, 60, 20)
        vp_col1, vp_col2 = st.columns([1, 2])
        show_vp = vp_col1.checkbox("Show volume profile", value=False)
        vp_bins = vp_col2.slider("Volume profile bins", 10, 40, 24,
                                 disabled=not show_vp) if show_vp else 24

        last_val = float(s_ta.iloc[-1])
        prev_val = float(s_ta.iloc[-2]) if len(s_ta) > 1 else last_val
        t1, t2, t3 = st.columns(3)
        t1.metric(f"{ta_choice} last", f"{last_val:,.3f}", f"{(last_val - prev_val):+,.3f}")
        t2.metric("as of", f"{s_ta.index.max().date()}")
        t3.metric("bars loaded", f"{len(s_ta)}")
        if live_asof is not None and live_asof == s_ta.index.max():
            st.caption(":green[latest bar is a live sina quote — the EOD history source "
                       "for this instrument lags by up to a day, so this point isn't in "
                       "the persisted CSV yet and may be revised once it settles]")

        s_plot = s_ta[s_ta.index >= cutoff]
        vol_plot = None
        if show_vp:
            s_vol = DD.load_series(f"{ta_key}_vol")
            vol_plot = s_vol[s_vol.index >= cutoff]
        st.plotly_chart(
            charts.technical_figure(s_plot, ta_choice, ma_windows, sr_window,
                                    volume=vol_plot, vp_bins=vp_bins),
            width='stretch')

        if show_vp and vol_plot is not None:
            vp = charts.volume_profile(s_plot, vol_plot, n_bins=vp_bins)
            if vp["poc"] is not None:
                st.caption(
                    f"Volume profile ({vp_bins} bins, plotted window): "
                    f"POC (point of control) **{vp['poc']:,.3f}** "
                    f"({(vp['poc'] / last_val - 1) * 100:+.1f}%) — the price level with the "
                    f"most volume traded. Value area **{vp['va_low']:,.3f} – {vp['va_high']:,.3f}** "
                    f"holds ~70% of traded volume in this window. Volume-weighted, unlike the "
                    f"swing-based S/R below.")
            else:
                st.caption("Volume profile: no volume data available for this window.")

        res_levels, sup_levels = charts.swing_levels(s_plot, sr_window)
        r_col, s_col = st.columns(2)
        with r_col:
            st.markdown("**Resistance (above last)**")
            if res_levels:
                for lvl in res_levels:
                    st.markdown(f"- {lvl:,.3f} :red[({(lvl / last_val - 1) * 100:+.1f}%)]")
            else:
                st.caption("none found in this window")
        with s_col:
            st.markdown("**Support (below last)**")
            if sup_levels:
                for lvl in sup_levels:
                    st.markdown(f"- {lvl:,.3f} :green[({(lvl / last_val - 1) * 100:+.1f}%)]")
            else:
                st.caption("none found in this window")

        st.caption(f"Support/resistance = the most recent distinct swing highs/lows within "
                   f"the plotted window, using a {sr_window}-bar centered window to define a "
                   f"local extremum. Descriptive only -- not a validated trading signal.")
