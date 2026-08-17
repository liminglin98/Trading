# -*- coding: utf-8 -*-
"""
chart_common.py -- shared plumbing for the 3-panel strategy charts
====================================================================
Factored out of straddle_quarterly_chart.py once a second strategy
(straddle_iv_pct_chart.py) needed the identical treatment: every script in
this folder that follows the "sequential single-position tracking" backtest
convention (quarterly_atm_short_straddle.py, vol_rich_short_straddle_backtest.py,
iv_spike_short_straddle_backtest.py, ...) returns run() -> {"trades": df,
"equity": df} with the same shapes (equity: date/equity/open_pnl/spot;
trades: entry_date/exit_date/pnl/margin_yuan among others), so the
daily-P&L -> monthly-bucketing -> 3-panel-chart pipeline is genuinely
reusable across them, not just superficially similar.

Per-strategy scripts stay responsible for: calling their own run(), knowing
their own MULTIPLIER, and picking a chart title. Everything downstream of
that (monthly bucketing, the live interactive stats badge, the 3-panel
figure) lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STATS_WINDOWS = [("1y", 1), ("3y", 3), ("5y", 5), ("all", None)]


def attach_daily_pnl_margin(trades: pd.DataFrame, equity: pd.DataFrame, multiplier: int) -> pd.DataFrame:
    """From a backtest's raw equity_df (date/equity/open_pnl/spot) and
    trades_df (needs entry_date/exit_date/pnl/margin_yuan), build a daily
    frame with pnl_yuan (mark-to-market P&L each trading day) and
    margin_yuan (that day's open position's margin, NaN when flat)."""
    equity = equity.copy()
    equity["date"] = pd.to_datetime(equity["date"])
    equity = equity.sort_values("date").reset_index(drop=True)
    equity["pnl_yuan"] = equity["equity"].diff().fillna(0.0) * multiplier

    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    equity["margin_yuan"] = np.nan
    for _, t in trades.iterrows():
        mask = (equity["date"] >= t["entry_date"]) & (equity["date"] <= t["exit_date"])
        equity.loc[mask, "margin_yuan"] = t["margin_yuan"]
    return equity


def monthly_return_series(equity: pd.DataFrame) -> pd.DataFrame:
    """Bucket a daily equity curve (date/pnl_yuan/margin_yuan columns, as
    produced by attach_daily_pnl_margin) into calendar months: return% =
    that month's total P&L / that month's average margin while a position
    was open. Works regardless of the strategy's own holding period --
    calendar months are a cadence every daily equity curve can be resampled
    to, whether the underlying trades hold for a quarter or a few days."""
    e = equity.copy()
    e["month"] = e["date"].dt.to_period("M")
    g = e.groupby("month").agg(pnl_yuan=("pnl_yuan", "sum"), margin_yuan=("margin_yuan", "mean"))
    g = g.dropna(subset=["margin_yuan"]).reset_index()  # months with no open position at all
    g["return_pct"] = 100.0 * g["pnl_yuan"] / g["margin_yuan"]
    g["month_start"] = g["month"].dt.to_timestamp()
    g["month_end"] = (g["month"] + 1).dt.to_timestamp()
    return g


def window_stats(monthly: pd.DataFrame, n_years: int | None, compounding: bool = True) -> dict:
    """Total return and win rate (fraction of positive months) over the
    trailing n_years window ending at the last month in the data (n_years=None
    -> the whole series).

    compounding=True multiplies monthly return_pct together (prod(1+r/100)-1)
    -- a reasonable model when the strategy redeploys a growing capital pool
    month over month, as the margin-based short-straddle scripts do (margin
    is large relative to premium, so monthly swings are small). compounding=
    False sums the monthly return_pct instead -- required for debit-basis
    long-premium strategies, where the debit at risk is small and flat per
    trade (not a growing pool), so single months can show +100-300% "return"
    on debit; compounding dozens of those together fabricates exponential
    growth that was never actually realized. See long_straddle_chart.py."""
    last_start, last_end = monthly["month_start"].max(), monthly["month_end"].max()
    start = monthly["month_start"].min() if n_years is None else last_start - pd.DateOffset(years=n_years)
    w = monthly[monthly["month_start"] >= start]
    if len(w):
        if compounding:
            total_return = (np.prod(1.0 + w["return_pct"] / 100.0) - 1.0) * 100.0
        else:
            total_return = w["return_pct"].sum()
    else:
        total_return = 0.0
    win_rate = 100.0 * (w["return_pct"] > 0).mean() if len(w) else 0.0
    return {"start": start, "end": last_end, "total_return": total_return, "win_rate": win_rate, "n": len(w)}


def slice_to_window(
    monthly: pd.DataFrame, underlying: pd.DataFrame, sse: pd.DataFrame, window: str,
    compounding: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice all three frames to one of STATS_WINDOWS' labels (e.g. '1y',
    'all') -- the terminal-image equivalent of clicking a preset button in
    the browser chart, since a raster image can't be zoomed after the fact."""
    n_years = dict(STATS_WINDOWS)[window]
    st = window_stats(monthly, n_years, compounding=compounding)
    m = monthly[monthly["month_start"] >= st["start"]].reset_index(drop=True)
    u = underlying[(underlying["date"] >= st["start"]) & (underlying["date"] <= st["end"])].reset_index(drop=True)
    s = sse[(sse["date"] >= st["start"]) & (sse["date"] <= st["end"])].reset_index(drop=True)
    return m, u, s


def format_stats_html(total_return: float, win_rate: float, n: int) -> str:
    color = "#2e7d32" if total_return >= 0 else "#c0392b"
    sign = "+" if total_return >= 0 else ""
    return (
        f"<span style='font-size:20px'>"
        f"<b>total return</b> <span style='color:{color}'>{sign}{total_return:.1f}%</span>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;<b>win rate</b> {win_rate:.0f}%"
        f"</span><br><span style='font-size:11px;color:#898781'>n={n} months in view</span>"
    )


def load_price_series(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    s = pd.read_csv(path)
    s["date"] = pd.to_datetime(s["date"])
    return s[(s["date"] >= start) & (s["date"] <= end)].sort_values("date").reset_index(drop=True)


def build_three_panel_figure(
    monthly: pd.DataFrame,
    underlying: pd.DataFrame,
    sse: pd.DataFrame,
    chart_title: str,
    panel1_title: str,
    panel2_label: str,
    panel3_label: str = "SSE Composite (000001)",
    compounding: bool = True,
) -> tuple[go.Figure, int]:
    """Returns (fig, stats_annotation_index) -- the caller needs the index
    to wire up the matching live-stats JS via build_live_stats_script().
    compounding is forwarded to window_stats() -- see its docstring."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.28, 0.36, 0.36],
        vertical_spacing=0.04,
        subplot_titles=(panel1_title, panel2_label, panel3_label),
    )

    colors = ["#c0392b" if r < 0 else "#2e7d32" for r in monthly["return_pct"]]
    fig.add_trace(
        go.Bar(
            x=monthly["month_start"],
            y=monthly["return_pct"],
            marker_color=colors,
            name="monthly return on margin",
            hovertemplate="%{x|%Y-%m}<br>monthly return: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=underlying["date"],
            y=underlying["close"],
            name=panel2_label,
            line=dict(color="#4c8bf5", width=1.3),
            hovertemplate="%{x|%Y-%m-%d}<br>" + panel2_label + ": %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sse["date"],
            y=sse["close"],
            name=panel3_label,
            line=dict(color="#8a8f98", width=1.3),
            hovertemplate="%{x|%Y-%m-%d}<br>" + panel3_label + ": %{y:.0f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    losers = monthly[monthly["return_pct"] < 0]
    for _, row in losers.iterrows():
        fig.add_vrect(
            x0=row["month_start"], x1=row["month_end"],
            fillcolor="rgba(192,57,43,0.12)", line_width=0, row="all", col=1,
        )

    buttons = []
    for label, n_years in STATS_WINDOWS:
        st = window_stats(monthly, n_years, compounding=compounding)
        buttons.append(
            dict(
                label=label,
                method="relayout",
                args=[{
                    "xaxis.range": [st["start"], st["end"]],
                    "xaxis2.range": [st["start"], st["end"]],
                    "xaxis3.range": [st["start"], st["end"]],
                }],
            )
        )

    all_stats = window_stats(monthly, None, compounding=compounding)
    stats_annotation_idx = len(fig.layout.annotations)
    fig.add_annotation(
        text=format_stats_html(all_stats["total_return"], all_stats["win_rate"], all_stats["n"]),
        xref="paper", yref="paper", x=1.0, y=1.08, xanchor="right", yanchor="bottom",
        showarrow=False, align="right",
        bgcolor="rgba(255,255,255,0.92)", bordercolor="#c3c2b7", borderwidth=1, borderpad=8,
    )

    fig.update_layout(
        title=f"{chart_title} -- {len(monthly)} months, {len(losers)} losing months shaded",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=30, t=130, b=40),
        dragmode="zoom",
        updatemenus=[dict(
            type="buttons", direction="right", buttons=buttons, showactive=True,
            x=0.0, y=1.22, xanchor="left", yanchor="bottom", pad=dict(r=4, l=4),
        )],
    )
    fig.update_yaxes(title_text="monthly return on margin %", fixedrange=False, row=1, col=1)
    fig.update_yaxes(title_text="price", fixedrange=False, row=2, col=1)
    fig.update_yaxes(title_text="index", fixedrange=False, row=3, col=1)
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=3, col=1)
    fig.update_xaxes(title_text="date", row=3, col=1)
    return fig, stats_annotation_idx


# Row weights + gap for the 4-panel figure -- module-level so
# build_four_panel_figure() and build_panel_toggle_script() (which computes
# the SAME domain math in JS, for the collapse-on-toggle behavior) can never
# drift out of sync with each other.
FOUR_PANEL_ROW_HEIGHTS = [0.18, 0.18, 0.32, 0.32]
FOUR_PANEL_VSPACING = 0.045


def build_four_panel_figure(
    monthly: pd.DataFrame,
    underlying: pd.DataFrame,
    sse: pd.DataFrame,
    chart_title: str,
    panel1_title: str,
    panel2_title: str,
    panel3_label: str,
    panel4_label: str = "SSE Composite (000001)",
    compounding: bool = True,
) -> tuple[go.Figure, int]:
    """Four-row variant of build_three_panel_figure(): monthly return % /
    monthly P&L in yuan (NEW -- `monthly` already carries a pnl_yuan column,
    every caller of monthly_return_series()/monthly_on_deployed_debit()
    produces one, so no new data plumbing needed) / underlying spot / SSE
    Composite. Each row also gets a show/hide button that COLLAPSES the
    hidden row's space rather than leaving it blank -- see
    build_panel_toggle_script(), which the caller must additionally pass to
    write_interactive_html() as `extra_script`.

    A standalone function, not a parameterized build_three_panel_figure(),
    so the 7 existing 3-panel callers are provably untouched -- this
    function is new code, not a new code path through old code.

    Returns (fig, stats_annotation_idx) exactly like the 3-panel version;
    the caller also needs `FOUR_PANEL_ROW_HEIGHTS`/`FOUR_PANEL_VSPACING` and
    the four subplot-title annotation indices (always 0..3, since
    make_subplots adds subplot_titles' annotations before anything else) to
    wire up build_panel_toggle_script()."""
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=FOUR_PANEL_ROW_HEIGHTS,
        vertical_spacing=FOUR_PANEL_VSPACING,
        subplot_titles=(panel1_title, panel2_title, panel3_label, panel4_label),
    )

    colors = ["#c0392b" if r < 0 else "#2e7d32" for r in monthly["return_pct"]]
    fig.add_trace(
        go.Bar(
            x=monthly["month_start"], y=monthly["return_pct"], marker_color=colors,
            name="monthly return %",
            hovertemplate="%{x|%Y-%m}<br>monthly return: %{y:.1f}%<extra></extra>",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=monthly["month_start"], y=monthly["pnl_yuan"], marker_color=colors,
            name="monthly P&L (yuan)",
            # plain .0f, not +,.0f -- Plotly's hovertemplate format validator
            # (stricter than d3-format itself) warns "bad format" on the
            # combined +/,/0f flags and spams that warning on every hover
            # mousemove; every other hovertemplate in this file already uses
            # the plain form, this just brings row2 in line with them.
            hovertemplate="%{x|%Y-%m}<br>monthly P&L: %{y:.0f} yuan<extra></extra>",
        ), row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=underlying["date"], y=underlying["close"], name=panel3_label,
            line=dict(color="#4c8bf5", width=1.3),
            hovertemplate="%{x|%Y-%m-%d}<br>" + panel3_label + ": %{y:.3f}<extra></extra>",
        ), row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sse["date"], y=sse["close"], name=panel4_label,
            line=dict(color="#8a8f98", width=1.3),
            hovertemplate="%{x|%Y-%m-%d}<br>" + panel4_label + ": %{y:.0f}<extra></extra>",
        ), row=4, col=1,
    )

    losers = monthly[monthly["return_pct"] < 0]
    for _, row in losers.iterrows():
        fig.add_vrect(
            x0=row["month_start"], x1=row["month_end"],
            fillcolor="rgba(192,57,43,0.12)", line_width=0, row="all", col=1,
        )

    window_buttons = []
    for label, n_years in STATS_WINDOWS:
        st = window_stats(monthly, n_years, compounding=compounding)
        window_buttons.append(dict(
            label=label, method="relayout",
            args=[{
                "xaxis.range": [st["start"], st["end"]], "xaxis2.range": [st["start"], st["end"]],
                "xaxis3.range": [st["start"], st["end"]], "xaxis4.range": [st["start"], st["end"]],
            }],
        ))

    all_stats = window_stats(monthly, None, compounding=compounding)
    stats_annotation_idx = len(fig.layout.annotations)
    fig.add_annotation(
        text=format_stats_html(all_stats["total_return"], all_stats["win_rate"], all_stats["n"]),
        xref="paper", yref="paper", x=1.0, y=1.10, xanchor="right", yanchor="bottom",
        showarrow=False, align="right",
        bgcolor="rgba(255,255,255,0.92)", bordercolor="#c3c2b7", borderwidth=1, borderpad=8,
    )

    # Panel show/hide is plain HTML buttons injected by build_panel_toggle_script(),
    # not a second Plotly updatemenus row -- see that function's docstring for why
    # (a real bug found in testing: Plotly's own updatemenus click event misattributed
    # which button fired once panels were hidden out of left-to-right order).
    fig.update_layout(
        title=f"{chart_title} -- {len(monthly)} months, {len(losers)} losing months shaded",
        template="plotly_white", hovermode="x unified", showlegend=False,
        # Explicit height, unlike build_three_panel_figure -- that one relies on
        # whatever height the container CSS/Plotly default resolves to, which
        # was fine for 3 rows but left 4 rows fighting over the same (or an
        # ambiguous, container-dependent) budget: individual rows could render
        # at a near-zero pixel height where bars/lines are practically
        # invisible even though the row's own domain/axes are still correctly
        # laid out -- matches the "rows are fine, only the marks are missing"
        # symptom exactly. Pinning height removes the ambiguity outright
        # rather than depending on the page's CSS chain.
        height=1000,
        margin=dict(l=50, r=30, t=130, b=40), dragmode="zoom",
        updatemenus=[
            dict(type="buttons", direction="right", buttons=window_buttons, showactive=True,
                x=0.0, y=1.16, xanchor="left", yanchor="bottom", pad=dict(r=4, l=4)),
        ],
    )
    fig.update_yaxes(title_text="return %", fixedrange=False, row=1, col=1)
    fig.update_yaxes(title_text="P&L (yuan)", fixedrange=False, row=2, col=1)
    fig.update_yaxes(title_text="price", fixedrange=False, row=3, col=1)
    fig.update_yaxes(title_text="index", fixedrange=False, row=4, col=1)
    fig.update_xaxes(showticklabels=True)  # every row keeps its own date axis -- see module note
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=4, col=1)
    fig.update_xaxes(title_text="date", row=4, col=1)
    return fig, stats_annotation_idx


def build_panel_toggle_script(panel_labels: list[str], div_id: str) -> str:
    """JS for the 4-panel figure's per-row show/hide buttons.

    These are PLAIN HTML <button> elements injected just above the chart,
    each with its own directly-bound click listener -- not a second Plotly
    updatemenus row. That was the first implementation, and it had a real
    bug found in manual testing: hiding a panel out of left-to-right order
    (e.g. clicking the 3rd button while all 4 were still visible) hid every
    panel before it too, not just the clicked one -- Plotly's own
    plotly_buttonclicked event was misattributing which button index had
    actually fired once the button row had been relayout-ed (this function
    relabels the clicked button's text on every toggle, which forces Plotly
    to redraw that same updatemenus row it's currently handling a click
    from). The domain-collapse math itself was verified correct in
    isolation (see the module's dev notes) -- only the click-to-index
    attribution was wrong. Plain DOM buttons have no such ambiguity: each
    listener closes over its own fixed index, independent of Plotly
    entirely, so this sidesteps the bug rather than chasing it inside
    Plotly's internals.

    Domain math (computeDomains) recomputes every row's yaxis.domain so a
    hidden row's vertical space is reclaimed by the others (the "real
    collapse" option), using the exact same weight/spacing
    build_four_panel_figure() used to lay the rows out initially --
    FOUR_PANEL_ROW_HEIGHTS/FOUR_PANEL_VSPACING, hardcoded into the JS below
    rather than left to drift from the Python constants. Refuses to hide
    the last visible panel."""
    return f"""
    (function() {{
        var gd = document.getElementById({json.dumps(div_id)});
        var WEIGHTS = {json.dumps(FOUR_PANEL_ROW_HEIGHTS)};
        var VSPACE = {FOUR_PANEL_VSPACING};
        var LABELS = {json.dumps(panel_labels)};
        var N = WEIGHTS.length;
        var visible = WEIGHTS.map(function() {{ return true; }});

        function computeDomains() {{
            var visIdx = [];
            for (var i = 0; i < N; i++) if (visible[i]) visIdx.push(i);
            var nVis = visIdx.length;
            var avail = 1 - VSPACE * Math.max(nVis - 1, 0);
            var sumW = 0;
            visIdx.forEach(function(i) {{ sumW += WEIGHTS[i]; }});
            var domains = [], titleY = [];
            var top = 1.0;
            for (var i = 0; i < N; i++) {{
                if (visible[i]) {{
                    var h = sumW > 0 ? (WEIGHTS[i] / sumW) * avail : 0;
                    var y1 = top, y0 = Math.max(top - h, 0);
                    domains.push([y0, y1]);
                    titleY.push(y1 + 0.015);
                    top = y0 - VSPACE;
                }} else {{
                    var flat = Math.max(top, 0);
                    domains.push([flat, flat]);
                    titleY.push(null);
                }}
            }}
            return {{domains: domains, titleY: titleY}};
        }}

        var btns = [];  // filled in below, one <button> per panel, index-matched
        var busy = false;    // true while a Plotly.react() call is in flight
        var pending = false; // another click arrived while busy -- redraw once more when free

        var axisNames = ['yaxis', 'yaxis2', 'yaxis3', 'yaxis4'];
        var pendingIdx = null;

        function apply(changedIdx) {{
            if (busy) {{
                pending = true;
                pendingIdx = changedIdx;
                console.log('[panel-toggle] apply() called while a redraw is in flight -- queued, not sent yet');
                return;
            }}
            busy = true;

            for (var i = 0; i < N; i++) {{
                gd.data[i].visible = visible[i];
                if (btns[i]) {{
                    btns[i].textContent = (visible[i] ? "\\u2713 " : "\\u2715 ") + LABELS[i];
                    btns[i].style.opacity = visible[i] ? "1" : "0.5";
                }}
            }}

            // Confirmed (by the user, precisely): sequential/adjacent toggles
            // -- shrinking or growing the visible set from either end -- work
            // every time, however fast. A toggle that leaves a HIDDEN row
            // sandwiched between two still-visible ones (a "gap" in the
            // visible set, e.g. hiding row 2 while 0/1/3 stay visible) fails
            // every time, regardless of click history -- ruling out an
            // async-timing or cold-start explanation, both tested and ruled
            // out above. That is a very specific reproduction signature, and
            // Plotly.react() with ALL FOUR axes' domains changed in one call
            // (verified correct data going in, still mispainted coming out)
            // hasn't dodged it. This splits every toggle into two react()
            // calls: phase 1 changes ONLY the clicked row's own domain
            // (collapsing it in place) and its trace visibility -- every
            // OTHER row's domain is left untouched, so this call can never
            // itself create a "gap" pattern in one shot. Phase 2, once that
            // settles, applies the full correct redistribution across all
            // rows. This costs one extra redraw but turns the single
            // gap-creating jump into two smaller, boundary-shaped changes --
            // closer to what naturally-sequential clicking already does
            // without issue.
            var current = gd.layout[axisNames[changedIdx]].domain.slice();
            var phase1Domain = visible[changedIdx] ? current : [current[0], current[0]];
            gd.layout[axisNames[changedIdx]].domain = phase1Domain;
            console.log('[panel-toggle] PHASE 1 (isolated single-row change) idx=', changedIdx,
                        'domain=', JSON.stringify(phase1Domain));

            Plotly.react(gd, gd.data, gd.layout).then(function() {{
                var r = computeDomains();
                for (var i = 0; i < N; i++) {{
                    gd.layout[axisNames[i]].domain = r.domains[i];
                    gd.layout.annotations[i].opacity = visible[i] ? 1 : 0;
                    if (visible[i]) gd.layout.annotations[i].y = r.titleY[i];
                }}
                console.log('[panel-toggle] PHASE 2 (full redistribution) visible=', JSON.stringify(visible),
                            'domains=', JSON.stringify(r.domains));
                return Plotly.react(gd, gd.data, gd.layout);
            }}).then(function() {{
                // What Plotly itself actually holds AFTER applying the update --
                // if this doesn't match what was sent above, something inside
                // Plotly is overriding it (auto-margin, another resize pass,
                // etc.), not this script's own logic.
                console.log('[panel-toggle] apply() AFTER update, gd.layout actually holds:',
                    JSON.stringify({{
                        yaxis: gd.layout.yaxis.domain, yaxis2: gd.layout.yaxis2.domain,
                        yaxis3: gd.layout.yaxis3.domain, yaxis4: gd.layout.yaxis4.domain,
                        height: gd.layout.height,
                        traces_visible: gd.data.map(function(t) {{ return t.visible; }}),
                    }}));
                var svg = gd.querySelector('svg.main-svg');
                var rect = svg ? svg.getBoundingClientRect() : null;
                console.log('[panel-toggle] rendered svg size:', rect ? JSON.stringify({{w: rect.width, h: rect.height}}) : 'NO SVG FOUND');
                var barCount = gd.querySelectorAll('.barlayer path').length;
                var lineCount = gd.querySelectorAll('.scatterlayer .lines path').length;
                console.log('[panel-toggle] rendered mark counts: bars=', barCount, 'lines=', lineCount);

                // Is there more than one chart on the page? (would mean this
                // script is reading/writing a DIFFERENT node than the one on
                // screen)
                var allPlots = document.querySelectorAll('.js-plotly-plot');
                console.log('[panel-toggle] total .js-plotly-plot elements on page:', allPlots.length);

                // ACTUAL screen pixel box of each subplot's <g>, independent of
                // what gd.layout claims -- if this disagrees with the domain
                // numbers above, Plotly computed the right thing but painted it
                // somewhere else (or didn't repaint at all).
                var subplotClasses = ['xy', 'x2y2', 'x3y3', 'x4y4'];
                var pixelBoxes = subplotClasses.map(function(cls) {{
                    var g = gd.querySelector('g.subplot.' + cls);
                    if (!g) return {{cls: cls, found: false}};
                    var r = g.getBoundingClientRect();
                    var style = window.getComputedStyle(g);
                    return {{cls: cls, found: true, top: r.top, bottom: r.bottom, height: r.height,
                            display: style.display, visibility: style.visibility, opacity: style.opacity}};
                }});
                console.log('[panel-toggle] ACTUAL on-screen pixel position per subplot:',
                    JSON.stringify(pixelBoxes, null, 1));

                var pageBg = window.getComputedStyle(document.body).backgroundColor;
                var gdRect = gd.getBoundingClientRect();
                console.log('[panel-toggle] gd on-screen box:', JSON.stringify({{
                    top: gdRect.top, bottom: gdRect.bottom, left: gdRect.left, right: gdRect.right,
                }}), 'body background-color:', pageBg, 'devicePixelRatio:', window.devicePixelRatio);

                busy = false;
                if (pending) {{
                    pending = false;
                    var idx = pendingIdx;
                    pendingIdx = null;
                    console.log('[panel-toggle] redraw finished, a click arrived while busy -- redrawing again now, idx=', idx);
                    apply(idx);
                }}
            }});
        }}

        function toggle(i) {{
            console.log('[panel-toggle] toggle(' + i + ') called, visible BEFORE=', JSON.stringify(visible));
            if (visible[i] && visible.filter(Boolean).length === 1) {{
                console.log('[panel-toggle] toggle(' + i + ') refused -- last visible panel');
                return;
            }}
            visible[i] = !visible[i];
            console.log('[panel-toggle] toggle(' + i + ') visible AFTER=', JSON.stringify(visible));
            apply(i);
        }}

        var bar = document.createElement("div");
        bar.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;padding:2px 0 10px;" +
            "font-family:system-ui,-apple-system,'Segoe UI',sans-serif;";
        for (var i = 0; i < N; i++) {{
            (function(i) {{
                var b = document.createElement("button");
                b.type = "button";
                b.textContent = "\\u2713 " + LABELS[i];
                b.style.cssText = "font-size:11px;padding:3px 8px;border-radius:4px;" +
                    "border:1px solid #c3c2b7;background:#fcfcfb;color:#52514e;cursor:pointer;";
                b.addEventListener("click", function() {{ toggle(i); }});
                bar.appendChild(b);
                btns[i] = b;
            }})(i);
        }}
        gd.parentNode.insertBefore(bar, gd);
        // NOTE: previously called Plotly.Plots.resize(gd) here (added when the
        // figure had no explicit height and depended on the container's own
        // size). Removed: the figure now always has an explicit layout.height
        // (see build_four_panel_figure), so nothing here needs Plotly to
        // re-measure the container -- and resize() is a heavy, non-domain-
        // aware operation that may have been clobbering the custom per-row
        // yaxis.domain layout this script depends on, which would only show
        // up later, on the first toggle() call that layers a domain change on
        // top of whatever resize() left behind. Left as a comment rather than
        // silently deleted since this is a real suspect, not a cosmetic edit.
    }})();
    """


def build_live_stats_script(monthly: pd.DataFrame, annotation_idx: int, div_id: str,
                            compounding: bool = True, axis_names: list[str] | None = None) -> str:
    """JS that recomputes the total-return/win-rate badge for whatever date
    range is actually visible, on every relayout event (preset buttons,
    box-zoom, scroll-zoom, or dragging the range slider). compounding mirrors
    window_stats()'s flag -- see its docstring -- so live zoom/pan recompute
    stays consistent with however the initial badge was computed.

    axis_names: the shared-x-axis ids to watch for a range change, in row
    order (default ["xaxis","xaxis2","xaxis3"], the plain 3-panel figure's
    ids -- unchanged from before this parameter existed, so every existing
    caller's output is byte-identical). build_four_panel_figure passes a
    4-element list."""
    if axis_names is None:
        axis_names = ["xaxis", "xaxis2", "xaxis3"]
    points = list(zip(monthly["month_start"].dt.strftime("%Y-%m-%d"), monthly["return_pct"].round(4)))
    return f"""
    (function() {{
        var gd = document.getElementById({json.dumps(div_id)});
        var DATA = {json.dumps(points)};
        var ANNOT_IDX = {annotation_idx};
        var COMPOUNDING = {json.dumps(compounding)};
        var AXES = {json.dumps(axis_names)};
        var allTimes = DATA.map(function(d) {{ return new Date(d[0]).getTime(); }});
        var minT = Math.min.apply(null, allTimes), maxT = Math.max.apply(null, allTimes);

        function computeStats(t0, t1) {{
            var sel = DATA.filter(function(d) {{
                var t = new Date(d[0]).getTime();
                return t >= t0 && t <= t1;
            }});
            if (sel.length === 0) return null;
            var compound = 1.0, sum = 0.0, wins = 0;
            sel.forEach(function(d) {{
                compound *= (1.0 + d[1] / 100.0);
                sum += d[1];
                if (d[1] > 0) wins++;
            }});
            return {{
                totalReturn: COMPOUNDING ? (compound - 1.0) * 100.0 : sum,
                winRate: 100.0 * wins / sel.length,
                n: sel.length,
            }};
        }}

        function fmt(stats) {{
            if (!stats) return "<span style='font-size:13px;color:#898781'>no months in range</span>";
            var color = stats.totalReturn >= 0 ? '#2e7d32' : '#c0392b';
            var sign = stats.totalReturn >= 0 ? '+' : '';
            return "<span style='font-size:20px'><b>total return</b> " +
                "<span style='color:" + color + "'>" + sign + stats.totalReturn.toFixed(1) + "%</span>" +
                "&nbsp;&nbsp;|&nbsp;&nbsp;<b>win rate</b> " + stats.winRate.toFixed(0) + "%</span>" +
                "<br><span style='font-size:11px;color:#898781'>n=" + stats.n + " months in view</span>";
        }}

        function extractRange(ev) {{
            var axes = AXES;
            for (var i = 0; i < axes.length; i++) {{
                var a = axes[i];
                if (ev[a + '.range[0]'] !== undefined) {{
                    return [new Date(ev[a + '.range[0]']).getTime(), new Date(ev[a + '.range[1]']).getTime()];
                }}
                if (ev[a + '.range'] !== undefined) {{
                    return [new Date(ev[a + '.range'][0]).getTime(), new Date(ev[a + '.range'][1]).getTime()];
                }}
                if (ev[a + '.autorange']) {{
                    return [minT, maxT];
                }}
            }}
            return null;
        }}

        gd.on('plotly_relayout', function(ev) {{
            var range = extractRange(ev);
            if (!range) return;  // e.g. our own annotation-only relayout below -- no xaxis keys, ignored
            var text = fmt(computeStats(range[0], range[1]));
            var upd = {{}};
            upd['annotations[' + ANNOT_IDX + '].text'] = text;
            Plotly.relayout(gd, upd);
        }});
    }})();
    """


def write_interactive_html(fig: go.Figure, monthly: pd.DataFrame, annotation_idx: int,
                            out_path: Path, div_id: str, compounding: bool = True,
                            axis_names: list[str] | None = None, extra_script: str = "",
                            responsive: bool = True) -> None:
    """axis_names/extra_script: both new, both no-op by default -- see
    build_live_stats_script()'s axis_names docstring. extra_script is JS
    appended after the live-stats script (e.g. build_panel_toggle_script()'s
    output); "" reproduces the exact prior post_script for every caller that
    doesn't pass it.

    responsive (default True, unchanged from before this parameter existed):
    Plotly's own write_html always passes {"responsive": true} to
    Plotly.newPlot when this isn't overridden. That's the right behavior for
    the plain 3-panel charts -- they have no explicit layout.height, so they
    NEED responsive/fill mode to size themselves against the container's
    height:100% CSS. But responsive mode is also what makes Plotly set up a
    dynamic resize watcher, and on a page opened as a local file:// URL that
    watcher's own internal resize-detection (an iframe-based technique)
    trips the browser's file-URL isolation policy -- "Unsafe attempt to load
    URL file://...html from frame with URL file://...html. 'file:' URLs are
    treated as unique security origins" -- which throws and can leave the
    chart in a half-initialized state (found via a real user report, not
    reproduced locally: this sandbox's headless Chrome doesn't enforce the
    same file:// isolation a normal desktop browser does, so this couldn't be
    caught by automated testing here). build_four_panel_figure() always sets
    an explicit layout.height, so it doesn't need fill/responsive behavior at
    all -- its caller passes responsive=False to route around the watcher
    entirely rather than trying to make it behave under file:// isolation."""
    script = build_live_stats_script(monthly, annotation_idx, div_id, compounding=compounding,
                                     axis_names=axis_names)
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True, div_id=div_id,
                   post_script=script + extra_script,
                   config={"responsive": responsive})

    # Plotly.write_html(full_html=True) never emits <!DOCTYPE html> -- its own
    # docstring says the output "starts with an <html> tag", by design, no
    # option to change it. Every browser without a doctype renders in Quirks
    # Mode, which changes CSS box-model behavior; the chart div's own
    # `height:100%` sizing (needed for the responsive layout) is exactly the
    # rule that's most likely to compute inconsistently under quirks rules,
    # and Plotly's internal layout math reads that computed size back via the
    # DOM. Confirmed present on every chart this module has ever written, not
    # something the 4-panel/toggle feature introduced -- fixed here so it's
    # fixed everywhere at once, not just for the caller that happened to hit it.
    html = out_path.read_text(encoding="utf-8")
    if not html.lstrip().lower().startswith("<!doctype"):
        out_path.write_text("<!DOCTYPE html>\n" + html, encoding="utf-8")


# --------------------------------------------------------------- terminal
# Static-image fallback for showing a chart directly inside an SSH terminal
# (no browser): renders the same 3 panels with matplotlib, then pushes the
# PNG over the Kitty graphics protocol. No live zoom/hover here -- a
# terminal image is a fixed raster with no event loop to hook a JS listener
# into -- but the caller can pre-slice monthly/underlying/sse to whatever
# window it wants (see slice_to_window below) and pass a matching
# window_label, so re-running with a different flag is the terminal
# equivalent of clicking a different preset button.

def render_matplotlib_png(
    monthly: pd.DataFrame,
    underlying: pd.DataFrame,
    sse: pd.DataFrame,
    chart_title: str,
    panel1_title: str,
    panel2_label: str,
    panel3_label: str = "SSE Composite (000001)",
    window_label: str = "all",
    figsize: tuple[float, float] = (11, 8),
    dpi: int = 130,
    compounding: bool = True,
) -> bytes:
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_SURFACE, C_INK, C_INK2, C_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    C_GRID, C_AXIS, C_BLUE, C_RED, C_GREEN = "#e1e0d9", "#c3c2b7", "#4c8bf5", "#c0392b", "#2e7d32"

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3, 1.3]}, facecolor=C_SURFACE,
    )
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(C_SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C_AXIS)
        ax.spines["bottom"].set_color(C_AXIS)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.grid(axis="y", color=C_GRID, linewidth=0.7)
        ax.set_axisbelow(True)

    colors = [C_RED if r < 0 else C_GREEN for r in monthly["return_pct"]]
    ax1.bar(monthly["month_start"], monthly["return_pct"], color=colors, width=20)
    ax1.set_ylabel("monthly return\non margin %", color=C_INK2, fontsize=9)

    ax2.plot(underlying["date"], underlying["close"], color=C_BLUE, linewidth=1.1)
    ax2.set_ylabel(panel2_label, color=C_INK2, fontsize=9)

    ax3.plot(sse["date"], sse["close"], color=C_MUTED, linewidth=1.1)
    ax3.set_ylabel(panel3_label, color=C_INK2, fontsize=9)

    for _, row in monthly[monthly["return_pct"] < 0].iterrows():
        for ax in (ax1, ax2, ax3):
            ax.axvspan(row["month_start"], row["month_end"], color=C_RED, alpha=0.10, linewidth=0)

    st = window_stats(monthly, None, compounding=compounding)
    sign = "+" if st["total_return"] >= 0 else ""
    stats_color = C_GREEN if st["total_return"] >= 0 else C_RED
    ax1.set_title(
        f"{chart_title}\n{panel1_title}",
        color=C_INK, fontsize=11, loc="left", pad=8,
    )
    fig.text(
        0.99, 0.99,
        f"total return {sign}{st['total_return']:.1f}%  |  win rate {st['win_rate']:.0f}%  "
        f"(n={st['n']} mo, {window_label})",
        ha="right", va="top", fontsize=10, color=stats_color,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_AXIS, lw=0.8),
    )

    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=C_SURFACE)
    plt.close(fig)
    return buf.getvalue()


def write_kitty_image(png_bytes: bytes, chunk_size: int = 4096) -> None:
    """Push a PNG to stdout via the Kitty graphics protocol (chunked base64
    APC escape sequences) -- renders inline in any terminal that implements
    the protocol (kitty, Ghostty/libghostty-vt, WezTerm, ...). No-op-safe in
    terminals that don't: they'll just ignore or show nothing, no crash."""
    import base64
    import sys

    b64 = base64.b64encode(png_bytes)
    chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]
    out = sys.stdout.buffer
    for i, chunk in enumerate(chunks):
        more = 1 if i < len(chunks) - 1 else 0
        control = f"a=T,f=100,m={more}" if i == 0 else f"m={more}"
        out.write(b"\x1b_G" + control.encode("ascii") + b";" + chunk + b"\x1b\\")
    out.write(b"\n")
    out.flush()
