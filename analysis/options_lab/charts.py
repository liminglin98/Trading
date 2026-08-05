"""Plotly figures: candlestick with trade markers, equity, drawdown, P&L."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import data_store as ds


def backtest_figure(equity: pd.DataFrame, trades: pd.DataFrame,
                    ohlc: pd.DataFrame | None) -> go.Figure:
    """Candles (or close line) + entry/exit markers, equity, drawdown."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.3, 0.2],
        subplot_titles=("Underlying & trades", "Equity (yuan, 1x combo)",
                        "Drawdown (yuan)"))

    if ohlc is not None and not ohlc.empty:
        o = ohlc[(ohlc["date"] >= equity["date"].min()) &
                 (ohlc["date"] <= equity["date"].max())]
        fig.add_trace(go.Candlestick(
            x=o["date"], open=o["open"], high=o["high"], low=o["low"],
            close=o["close"], name="ETF",
            increasing_line_color="#d33", decreasing_line_color="#2a2"),
            row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["spot"],
                                 mode="lines", name="ETF close",
                                 line=dict(color="#888")), row=1, col=1)

    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=trades["entry_date"], y=trades["entry_spot"], mode="markers",
            name="entry", marker=dict(symbol="triangle-up", size=11,
                                      color="#1f77b4"),
            text=trades["legs_desc"], hovertemplate="%{x|%Y-%m-%d} entry<br>"
            "%{text}<extra></extra>"), row=1, col=1)
        win = trades["pnl"] > 0
        fig.add_trace(go.Scatter(
            x=trades["exit_date"], y=trades["exit_spot"], mode="markers",
            name="exit",
            marker=dict(symbol="triangle-down", size=11,
                        color=["#2a2" if w else "#d33" for w in win]),
            text=[f"{r.exit_reason} pnl={r.pnl * ds.MULTIPLIER:,.0f}"
                  for r in trades.itertuples()],
            hovertemplate="%{x|%Y-%m-%d} exit<br>%{text}<extra></extra>"),
            row=1, col=1)

    eq = equity.set_index("date")["equity"] * ds.MULTIPLIER
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="equity",
                             line=dict(color="#1f77b4")), row=2, col=1)
    dd = eq - eq.cummax()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="drawdown",
                             fill="tozeroy", line=dict(color="#d33")),
                  row=3, col=1)

    fig.update_layout(height=820, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.05),
                      margin=dict(l=40, r=20, t=60, b=30))
    return fig


def pnl_histogram(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=trades["pnl"] * ds.MULTIPLIER, nbinsx=40,
        marker_color=["#2a2"] * 1))
    fig.update_layout(title="Trade P&L distribution (yuan)", height=320,
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def payoff_figure(legs: pd.DataFrame, spot: float | None,
                  title: str = "P&L at expiry",
                  realized_pnl_yuan: float = 0.0) -> go.Figure:
    """Expiry payoff across underlying prices for the STILL-OPEN legs of a
    combo, plus a flat offset for realized_pnl_yuan (P&L already locked in
    from legs that have been closed -- fixed, not path-dependent, so it just
    shifts the curve rather than bending it).

    legs: rows with cp, strike, direction (+1/-1), qty, entry_price. Pass
    only open legs; closed legs belong in realized_pnl_yuan instead.
    All legs valued at intrinsic on expiry (for mixed expiries this is the
    usual approximation).
    """
    import numpy as np

    if legs.empty:
        strikes = pd.Series([spot or 1.0])
    else:
        strikes = legs["strike"].astype(float)
    lo = min(strikes.min(), spot or strikes.min()) * 0.85
    hi = max(strikes.max(), spot or strikes.max()) * 1.15
    prices = np.linspace(lo, hi, 400)
    pnl = np.zeros_like(prices)
    for _, l in legs.iterrows():
        k = float(l["strike"])
        intrinsic = np.maximum(prices - k, 0) if l["cp"] == "C" \
            else np.maximum(k - prices, 0)
        pnl += l["direction"] * l["qty"] * (intrinsic - float(l["entry_price"]))
    pnl *= ds.MULTIPLIER
    pnl += realized_pnl_yuan

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices, y=pnl, name="P&L at expiry", line=dict(color="#1f77b4"),
        fill="tozeroy"))
    fig.add_hline(y=0, line_color="#999", line_width=1)
    if spot:
        fig.add_vline(x=spot, line_dash="dot", line_color="#d33",
                      annotation_text=f"spot {spot:.3f}")
    # breakevens
    sign = np.sign(pnl)
    be = prices[:-1][sign[:-1] * sign[1:] < 0]
    for b in be[:6]:
        fig.add_vline(x=b, line_dash="dash", line_color="#2a2",
                      annotation_text=f"BE {b:.3f}",
                      annotation_position="bottom")
    fig.update_layout(title=title, height=380,
                      xaxis_title="underlying price",
                      yaxis_title="P&L (yuan)",
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig


def position_pnl_figure(hist: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["pnl_yuan"],
                             name="P&L (yuan)", fill="tozeroy",
                             line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["spot"], name="spot",
                             line=dict(color="#999", dash="dot")),
                  secondary_y=True)
    fig.update_layout(title=title, height=380,
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def account_equity_figure(eq: pd.DataFrame) -> go.Figure:
    """Whole-account asset-tracking curve (positions.account_equity_curve).

    No fill-to-zero here: the account value never gets near zero, and a
    tozeroy fill forces Plotly's y-axis autorange to include 0 -- which
    squashes real day-to-day variation (a few thousand yuan against a
    baseline of hundreds of thousands) into a visually flat line pinned
    near the top of the chart.
    """
    y = eq["equity_yuan"]
    pad = max((y.max() - y.min()) * 0.1, 1.0)
    # Category x-axis: one equal-width slot per row, so weekends/holidays
    # with no data don't stretch the line across a gap -- a true date axis
    # would otherwise space points by calendar days.
    x_labels = pd.to_datetime(eq["date"]).dt.strftime("%Y-%m-%d")
    fig = go.Figure(go.Scatter(x=x_labels, y=y, mode="lines",
                               line=dict(color="#1f77b4")))
    fig.update_layout(title="Account asset tracking (¥, cash + all positions)",
                      height=420, xaxis_title="date", xaxis_type="category",
                      yaxis_title="account value (¥)",
                      yaxis_range=[y.min() - pad, y.max() + pad],
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def macro_series_figure(s: pd.Series, title: str) -> go.Figure:
    """Single macro/market series over time (dashboard_data.py series)."""
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines",
                               line=dict(color="#1f77b4")))
    fig.update_layout(title=title, height=340,
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def swing_levels(s: pd.Series, sr_window: int) -> tuple[list[float], list[float]]:
    """Most recent distinct swing-high/low levels within a centered window.

    Returns (resistance_levels, support_levels), each sorted with the
    level closest to the last price first -- highest of the resistance
    levels last, lowest of the support levels last.
    """
    span = sr_window * 2 + 1
    roll_max = s.rolling(span, center=True).max()
    roll_min = s.rolling(span, center=True).min()
    swing_highs = s[s == roll_max]
    swing_lows = s[s == roll_min]
    res_levels = sorted(swing_highs.tail(8).unique(), reverse=True)[:4]
    sup_levels = sorted(swing_lows.tail(8).unique())[:4]
    return res_levels, sup_levels


def _smooth_profile(a: np.ndarray, window: int = 3) -> np.ndarray:
    """Edge-padded moving-average smoothing, same length as input."""
    if window <= 1 or len(a) < window:
        return a.copy()
    pad = window // 2
    padded = np.pad(a, pad, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[:len(a)]


def _profile_runs(vals: np.ndarray) -> list[tuple[float, int, int]]:
    """Collapse consecutive near-equal values into (value, start, end)
    runs (indices into `vals`)."""
    runs = []
    i, n = 0, len(vals)
    while i < n:
        j = i
        while j + 1 < n and np.isclose(vals[j + 1], vals[i]):
            j += 1
        runs.append((vals[i], i, j))
        i = j + 1
    return runs


def volume_profile(price: pd.Series, volume: pd.Series, n_bins: int = 24,
                   value_area_pct: float = 0.70, cluster_frac: float = 0.5) -> dict:
    """Volume-at-price histogram over the dates common to `price` and `volume`.

    Bins price into `n_bins` equal-width buckets and sums the volume traded
    on each date into the bucket holding that date's close. Returns the bin
    midpoints/volumes plus:
      * poc / va_high / va_low -- point of control (highest-volume bin) and
        the value area (smallest contiguous run of bins, grown outward from
        the POC, whose volume share reaches `value_area_pct`).
      * lvns -- up to 3 low-volume-node price levels: local minima (strictly
        below both neighboring bins) within the traded range, weakest first.
        Price tends to move through these quickly rather than linger, so
        they often act as support/resistance vacuums.
      * n_clusters / shape -- contiguous runs of bins at or above
        `cluster_frac` of the profile's own peak volume, within the traded
        range. One run reads as a balanced, single-peaked (D-shaped)
        distribution -- consolidation around a fair price. Two or more
        disjoint runs read as multi-modal -- value has migrated, often seen
        across trend days/periods. A coarse heuristic (bin-count based), not
        a rigorous mode test.
    """
    idx = price.index.intersection(volume.index)
    p = price.reindex(idx).astype(float)
    v = volume.reindex(idx).astype(float).fillna(0.0)
    empty = dict(bin_mids=[], bin_vols=[], poc=None, va_high=None, va_low=None,
                lvns=[], n_clusters=0, shape=None)
    if len(p) == 0 or v.sum() <= 0:
        return empty

    lo, hi = float(p.min()), float(p.max())
    if lo == hi:
        return dict(bin_mids=[lo], bin_vols=[float(v.sum())], poc=lo, va_high=lo,
                    va_low=lo, lvns=[], n_clusters=1, shape="balanced (single peak)")

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(p.values, edges[1:-1]), 0, n_bins - 1)
    bin_vols = np.zeros(n_bins)
    np.add.at(bin_vols, bin_idx, v.values)
    bin_mids = (edges[:-1] + edges[1:]) / 2

    poc_i = int(np.argmax(bin_vols))
    target = bin_vols.sum() * value_area_pct
    lo_i = hi_i = poc_i
    captured = bin_vols[poc_i]
    while captured < target and (lo_i > 0 or hi_i < n_bins - 1):
        left = bin_vols[lo_i - 1] if lo_i > 0 else -1.0
        right = bin_vols[hi_i + 1] if hi_i < n_bins - 1 else -1.0
        if left >= right:
            lo_i -= 1
            captured += bin_vols[lo_i]
        else:
            hi_i += 1
            captured += bin_vols[hi_i]

    occupied = np.nonzero(bin_vols)[0]
    occ_lo, occ_hi = int(occupied.min()), int(occupied.max())
    seg = bin_vols[occ_lo:occ_hi + 1]

    # Light smoothing before shape/LVN detection: with only 10-40 price bins
    # over a real trading history, per-bin volume is noisy enough that the
    # raw histogram alone throws up spurious single-bin peaks/troughs.
    smooth = _smooth_profile(seg, window=5)
    # Collapse consecutive near-equal smoothed values into runs so a WIDE
    # flat stretch (e.g. a true zero-volume gap) reads as one trough/peak
    # candidate instead of failing a strict "<" test against its own tied
    # neighbor.
    runs = _profile_runs(smooth)

    lvn_idx = []
    for k in range(1, len(runs) - 1):
        v_here, s, e = runs[k]
        if v_here < runs[k - 1][0] and v_here < runs[k + 1][0]:
            lvn_idx.append(occ_lo + (s + e) // 2)
    lvn_idx.sort(key=lambda i: bin_vols[i])
    lvns = [float(bin_mids[i]) for i in lvn_idx[:3]]

    above = smooth >= smooth.max() * cluster_frac
    n_clusters = int(np.sum(above & ~np.r_[False, above[:-1]]))
    shape = ("balanced (single peak)" if n_clusters <= 1
            else f"multi-modal ({n_clusters} clusters)")

    return dict(bin_mids=bin_mids.tolist(), bin_vols=bin_vols.tolist(),
               poc=float(bin_mids[poc_i]), va_high=float(bin_mids[hi_i]),
               va_low=float(bin_mids[lo_i]), lvns=lvns,
               n_clusters=n_clusters, shape=shape)


def technical_figure(s: pd.Series, title: str, ma_windows: list[int],
                     sr_window: int, volume: pd.Series | None = None,
                     vp_bins: int = 24) -> go.Figure:
    """Price + moving averages + rolling swing support/resistance levels.

    If `volume` is given, also overlays a volume profile (volume-at-price
    histogram) with POC / value-area lines in the right margin.
    """
    # Category x-axis: one equal-width slot per trading day, so weekends/
    # holidays with no bar don't stretch the line across a gap -- a true
    # date axis would otherwise space points by calendar days.
    x_labels = pd.to_datetime(s.index).strftime("%Y-%m-%d")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_labels, y=s.values, mode="lines", name=title,
                             line=dict(color="#ffc107", width=2.2)))

    ma_colors = ["#1f77b4", "#e65100", "#6a1b9a", "#00897b", "#d33"]
    for i, w in enumerate(sorted(ma_windows)):
        ma = s.rolling(w).mean()
        fig.add_trace(go.Scatter(x=x_labels, y=ma.values, mode="lines",
                                 name=f"MA{w}",
                                 line=dict(color=ma_colors[i % len(ma_colors)],
                                           width=1.2)))

    res_levels, sup_levels = swing_levels(s, sr_window)
    for j, lvl in enumerate(res_levels):
        fig.add_hline(y=lvl, line_dash="dash", line_color="#d33", opacity=0.6,
                      annotation_text=f"R {lvl:,.3f}", annotation_position="right",
                      annotation_font_color="#d33", annotation_font_size=11)
    for j, lvl in enumerate(sup_levels):
        fig.add_hline(y=lvl, line_dash="dash", line_color="#2a2", opacity=0.6,
                      annotation_text=f"S {lvl:,.3f}", annotation_position="right",
                      annotation_font_color="#2a2", annotation_font_size=11)

    right_margin = 70
    if volume is not None and len(volume):
        vp = volume_profile(s, volume, n_bins=vp_bins)
        if vp["bin_vols"]:
            fig.update_layout(xaxis=dict(domain=[0.0, 0.82]))
            fig.add_trace(go.Bar(
                x=vp["bin_vols"], y=vp["bin_mids"], orientation="h",
                xaxis="x2", marker_color="rgba(130,130,130,0.35)",
                name="volume profile", showlegend=False, hoverinfo="skip"))
            fig.update_layout(xaxis2=dict(
                domain=[0.83, 1.0], anchor="y", showticklabels=False,
                showgrid=False, zeroline=False))
            for y, label in [(vp["poc"], "POC"), (vp["va_high"], "VAH"),
                             (vp["va_low"], "VAL")]:
                dash = "solid" if label == "POC" else "dot"
                fig.add_shape(type="line", xref="paper", x0=0, x1=0.82, yref="y",
                             y0=y, y1=y, line=dict(color="#8e44ad", width=1.4, dash=dash))
                fig.add_annotation(xref="paper", x=1.0, y=y, yref="y", xanchor="left",
                                  text=f"{label} {y:,.3f}", showarrow=False,
                                  font=dict(color="#8e44ad", size=10))
            for y in vp["lvns"]:
                fig.add_shape(type="line", xref="paper", x0=0, x1=0.82, yref="y",
                             y0=y, y1=y, line=dict(color="#888", width=1.0, dash="dashdot"))
                fig.add_annotation(xref="paper", x=1.0, y=y, yref="y", xanchor="left",
                                  text=f"LVN {y:,.3f}", showarrow=False,
                                  font=dict(color="#888", size=9))
            right_margin = 90

    fig.update_layout(title=title, height=460, xaxis_type="category",
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=40, r=right_margin, t=60, b=30))
    return fig
