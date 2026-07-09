"""Plotly figures: candlestick with trade markers, equity, drawdown, P&L."""
from __future__ import annotations

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
    """Whole-account asset-tracking curve (positions.account_equity_curve)."""
    fig = go.Figure(go.Scatter(x=eq["date"], y=eq["equity_yuan"], mode="lines",
                               fill="tozeroy", line=dict(color="#1f77b4")))
    fig.update_layout(title="Account asset tracking (¥, cash + all positions)",
                      height=420, xaxis_title="date",
                      yaxis_title="account value (¥)",
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def macro_series_figure(s: pd.Series, title: str) -> go.Figure:
    """Single macro/market series over time (dashboard_data.py series)."""
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines",
                               line=dict(color="#1f77b4")))
    fig.update_layout(title=title, height=340,
                      margin=dict(l=40, r=20, t=50, b=30))
    return fig


def technical_figure(s: pd.Series, title: str, ma_windows: list[int],
                     sr_window: int) -> go.Figure:
    """Price + moving averages + rolling swing support/resistance levels."""
    import numpy as np

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=title,
                             line=dict(color="#ffc107", width=2.2)))

    ma_colors = ["#1f77b4", "#e65100", "#6a1b9a", "#00897b", "#d33"]
    for i, w in enumerate(sorted(ma_windows)):
        ma = s.rolling(w).mean()
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode="lines",
                                 name=f"MA{w}",
                                 line=dict(color=ma_colors[i % len(ma_colors)],
                                           width=1.2)))

    span = sr_window * 2 + 1
    roll_max = s.rolling(span, center=True).max()
    roll_min = s.rolling(span, center=True).min()
    swing_highs = s[s == roll_max]
    swing_lows = s[s == roll_min]
    res_levels = sorted(swing_highs.tail(8).unique(), reverse=True)[:4]
    sup_levels = sorted(swing_lows.tail(8).unique())[:4]
    for j, lvl in enumerate(res_levels):
        fig.add_hline(y=lvl, line_dash="dash", line_color="#d33", opacity=0.6,
                      annotation_text="resistance" if j == 0 else None)
    for j, lvl in enumerate(sup_levels):
        fig.add_hline(y=lvl, line_dash="dash", line_color="#2a2", opacity=0.6,
                      annotation_text="support" if j == 0 else None)

    fig.update_layout(title=title, height=460,
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(l=40, r=20, t=60, b=30))
    return fig
