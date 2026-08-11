# -*- coding: utf-8 -*-
"""
csi1000_skew_chart.py -- CSI1000 skew risk-reversal vs. context
=================================================================
3-panel interactive chart (monthly return on margin / CSI1000 spot / SSE
Composite) for csi1000_skew_risk_reversal_backtest.py's risk_reversal
results. Strategy-specific glue only -- panel/badge/live-recompute
machinery lives in chart_common.py.

No CSI1000 OHLC CSV exists in Data/index/ (unlike the SSE ETF underlyings
straddle_quarterly_chart.py reads via cc.load_price_series), so panel 2 uses
csi1000_skew_risk_reversal_backtest.build_underlying_price_series() instead,
which rebuilds the near-month parity-forward spot directly from the option
chain.

Run:
    .venv/bin/python analysis/options_lab/csi1000_skew_chart.py
Output:
    analysis/options_lab/out/csi1000_skew_risk_reversal_vs_underlying_sse.html
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cffex_margin as cm  # noqa: E402
import chart_common as cc  # noqa: E402
import csi1000_skew_risk_reversal_backtest as sk  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parent.parent

SSE_CSV = ROOT / "Data" / "index" / "index_daily_000001.csv"
OUT_HTML = OUT / "csi1000_skew_risk_reversal_vs_underlying_sse.html"

MULTIPLIER = cm.MULTIPLIER
DIV_ID = "csi1000-skew-risk-reversal-chart"
CHART_TITLE = "CSI1000 (MO) skew risk reversal: short 20-delta put / long 20-delta call"
PANEL1_TITLE = "skew risk reversal: monthly return on margin (mark-to-market, CFFEX margin APPROX)"
PANEL2_LABEL = "CSI1000 near-month parity-forward spot"


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    res = sk.run(structure="risk_reversal")
    trades, equity = res["trades"], res["equity"]
    equity = cc.attach_daily_pnl_margin(trades, equity, MULTIPLIER)
    monthly = cc.monthly_return_series(equity)
    start, end = equity["date"].iloc[0], equity["date"].iloc[-1]
    underlying = sk.build_underlying_price_series()
    underlying = underlying[(underlying["date"] >= start) & (underlying["date"] <= end)].reset_index(drop=True)
    sse = cc.load_price_series(SSE_CSV, start, end)
    return monthly, underlying, sse


def main() -> None:
    monthly, underlying, sse = load()

    fig, annotation_idx = cc.build_three_panel_figure(
        monthly, underlying, sse,
        chart_title=CHART_TITLE,
        panel1_title=PANEL1_TITLE,
        panel2_label=PANEL2_LABEL,
    )

    OUT.mkdir(exist_ok=True)
    cc.write_interactive_html(fig, monthly, annotation_idx, OUT_HTML, DIV_ID)
    print(f"wrote {OUT_HTML} ({len(monthly)} months, "
          f"{len(monthly[monthly['return_pct'] < 0])} losing months)")


if __name__ == "__main__":
    main()
