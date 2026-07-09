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

import backtest as bt
import charts
import data_store as ds
import positions as pos
import dashboard_data as DD

st.set_page_config(page_title="Options Lab", layout="wide")


@st.cache_data(show_spinner="Loading option chain history...")
def get_chain(u: str) -> pd.DataFrame:
    return ds.load_chain(u)


@st.cache_data
def get_ohlc(u: str) -> pd.DataFrame | None:
    return ds.load_etf_ohlc(u)


def fmt_expiry_options(chain: pd.DataFrame):
    last_dt = chain[ds.C_DATE].max()
    return sorted(chain.loc[chain[ds.C_DATE] == last_dt, ds.C_EXPIRY].unique())


st.title("Options Lab")
tab_data, tab_pos, tab_bt, tab_macro, tab_ta = st.tabs(
    ["📥 Data", "📌 Positions", "🧪 Backtest", "🌐 Macro Dashboard", "📈 Technical"])

# =============================================================== Data tab
with tab_data:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Chain status")
        for u, name in ds.UNDERLYINGS.items():
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
                    updater.run(list(ds.UNDERLYINGS))
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
    with c2:
        st.subheader("Latest chain snapshot")
        u = st.selectbox("Underlying", list(ds.UNDERLYINGS),
                         format_func=lambda x: f"{ds.UNDERLYINGS[x]} ({x})",
                         key="data_u")
        chain = get_chain(u)
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
with tab_ta:
    st.subheader("Technical Analysis")
    st.caption("Price with moving averages and rolling swing-based support/resistance. "
               "Analytics only -- not a signal, no orders are placed.")

    TA_MAP = {"Shanghai Composite": "idx_sh", "50ETF": "etf50", "300ETF": "etf300"}
    ta_choice = st.selectbox("Instrument", list(TA_MAP.keys()))
    ta_key = TA_MAP[ta_choice]
    s_ta = DD.load_series(ta_key)

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

        ma_windows = st.multiselect("Moving averages (bars)", [10, 20, 50, 100, 200],
                                    default=[20, 50, 200])
        sr_window = st.slider("Support/resistance swing window (bars)", 5, 60, 20)

        last_val = float(s_ta.iloc[-1])
        prev_val = float(s_ta.iloc[-2]) if len(s_ta) > 1 else last_val
        t1, t2, t3 = st.columns(3)
        t1.metric(f"{ta_choice} last", f"{last_val:,.3f}", f"{(last_val - prev_val):+,.3f}")
        t2.metric("as of", f"{s_ta.index.max().date()}")
        t3.metric("bars loaded", f"{len(s_ta)}")

        s_plot = s_ta[s_ta.index >= cutoff]
        st.plotly_chart(
            charts.technical_figure(s_plot, ta_choice, ma_windows, sr_window),
            width='stretch')
        st.caption(f"Support/resistance = the most recent distinct swing highs/lows within "
                   f"the plotted window, using a {sr_window}-bar centered window to define a "
                   f"local extremum. Descriptive only -- not a validated trading signal.")
