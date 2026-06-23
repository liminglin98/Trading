# -*- coding: utf-8 -*-
"""
akshare_data_availability_check.py
==================================
Index-level data-availability probe for the Butterfly next-month forecast model,
using the free akshare API instead of QMT.

WHY akshare here: QMT's quote service / multi-factor library would not connect,
and its BARRA factors are per-stock (cross-sectional). The butterfly model needs
INDEX-level time series (CSI300 / SSE50 / their ETFs), which akshare exposes
directly -- index OHLC, index P/E, margin balance, China & US 10Y yields, social
financing, FX, and a few cross-asset series.

RUN THIS IN A NORMAL PYTHON ENV (not the QMT editor):
    pip install akshare --upgrade
    python akshare_data_availability_check.py

(akshare needs internet. A few calls require Chinese arguments -- that's fine in a
normal interpreter; it is only the QMT editor that cannot handle Chinese.)

It walks the checklist categories, tries each series, and reports:
  AVAILABLE / EMPTY / ERROR, history start, #rows, columns.
Successful pulls are saved as CSV under  Data/akshare/  so they're ready to use.
A summary is written to  analysis/akshare_data_availability_report.csv .
"""

import os
import csv
import datetime as dt

try:
    import akshare as ak
    import pandas as pd
except Exception as e:  # noqa
    print("!! Need akshare + pandas:  pip install akshare pandas --upgrade")
    print("   import error:", e)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _find_trading_root(start):
    """Walk up from `start` to find the Trading root (has Data/ + analysis/).
    Works whether run as a file (HERE=...\\analysis) or pasted (cwd=...\\Trading)."""
    p = os.path.abspath(start)
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "Data")) and os.path.isdir(os.path.join(p, "analysis")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.path.abspath(start)


TRADING   = _find_trading_root(HERE)
SAVE_DIR  = os.path.join(TRADING, "Data", "akshare")
OUT_CSV   = os.path.join(TRADING, "analysis", "akshare_data_availability_report.csv")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Trading root: {TRADING}\nSaving series to: {SAVE_DIR}\n")

START     = "20100101"
TODAY     = dt.date.today().strftime("%Y%m%d")

results = []


def _find_date_col(df):
    for c in df.columns:
        lc = str(c).lower()
        cs = str(c)
        if lc in ("date", "datetime", "trade_date", "day", "time") or "date" in lc:
            return c
        # Chinese date columns, e.g. 日期 / 月份 / 时间 / 信用交易日期 (substring match)
        if any(tok in cs for tok in ("日期", "月份", "时间")):
            return c
    return None


def probe(category, item, fn, save_as=None, note=""):
    """fn: zero-arg callable returning a DataFrame. Reports + optionally saves."""
    try:
        df = fn()
        if df is None or len(df) == 0:
            _record(category, item, "EMPTY", note=note or "returned no rows")
            return None
        dc = _find_date_col(df)
        if dc is not None:
            try:
                ser = pd.to_datetime(df[dc].astype(str), errors="coerce").dropna()
                start = ser.min().strftime("%Y-%m-%d") if len(ser) else "?"
            except Exception:
                start = str(df[dc].iloc[0])[:10]
        else:
            start = "(no date col)"
        cols = ",".join(map(str, list(df.columns)[:8]))
        _record(category, item, "AVAILABLE", start=start, rows=len(df), cols=cols, note=note)
        if save_as:
            path = os.path.join(SAVE_DIR, save_as)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return df
    except Exception as e:  # noqa
        _record(category, item, "ERROR", note=f"{note} :: {type(e).__name__}: {e}".strip())
        return None


def _record(category, item, status, start="", rows="", cols="", note=""):
    row = dict(category=category, item=item, status=status,
               start=start, rows=rows, cols=cols, note=note)
    results.append(row)
    print(f"[{status:9}] {category:12} | {item:34} | start={start:>12} "
          f"rows={str(rows):>6} | {note}")


# ===========================================================================
# 1. MOMENTUM / TREND  -- index & ETF daily OHLC (long history)
# ===========================================================================
print("\n=== 1. Momentum / Trend (index + ETF OHLC) ===")
# Index OHLC via Sina (full history). Symbols: sh000300 (CSI300), sh000016 (SSE50).
probe("Momentum", "CSI300 index (sina)",
      lambda: ak.stock_zh_index_daily(symbol="sh000300"),
      save_as="csi300_index_daily.csv", note="000300 OHLC, long history")
probe("Momentum", "SSE50 index (sina)",
      lambda: ak.stock_zh_index_daily(symbol="sh000016"),
      save_as="sse50_index_daily.csv", note="000016 OHLC, long history")
# ETF OHLC via SINA (works direct on this network; eastmoney needs proxy).
# fund_etf_hist_sina returns date/open/high/low/close/volume (UNADJUSTED).
probe("Momentum", "50ETF 510050 (sina)",
      lambda: ak.fund_etf_hist_sina(symbol="sh510050"),
      save_as="etf_510050_daily.csv", note="sina, unadjusted; eastmoney blocked")
probe("Momentum", "300ETF 510300 (sina)",
      lambda: ak.fund_etf_hist_sina(symbol="sh510300"),
      save_as="etf_510300_daily.csv", note="sina, unadjusted; eastmoney blocked")


# ===========================================================================
# 2. VALUATION  -- index P/E, P/B (+ 10Y CGB handled in cross-asset block)
# ===========================================================================
print("\n=== 2. Valuation (index PE / PB) ===")
# legulegu index valuation. NOTE: Chinese symbol arg required by akshare.
probe("Valuation", "CSI300 PE (legulegu)",
      lambda: ak.stock_index_pe_lg(symbol="沪深300"),   # 沪深300
      save_as="csi300_pe_lg.csv", note="index PE TTM history")
probe("Valuation", "SSE50 PE (legulegu)",
      lambda: ak.stock_index_pe_lg(symbol="上证50"),    # 上证50
      save_as="sse50_pe_lg.csv", note="index PE TTM history")
probe("Valuation", "CSI300 PB (legulegu)",
      lambda: ak.stock_index_pb_lg(symbol="沪深300"),   # 沪深300
      save_as="csi300_pb_lg.csv", note="index PB history")


# ===========================================================================
# 3. LIQUIDITY / MONETARY  -- margin balance, northbound, social financing
# ===========================================================================
print("\n=== 3. Liquidity / Monetary ===")
# Margin financing (RZRQ) totals, by exchange.
probe("Liquidity", "SSE margin (RZRQ)",
      lambda: ak.stock_margin_sse(start_date=START, end_date=TODAY),
      save_as="margin_sse.csv", note="Shanghai financing+lending balance")
# SZSE margin is a per-date SNAPSHOT; today often unpublished (empty -> ValueError).
# Use a settled recent trading date instead.
_SZSE_DATE = (dt.date.today() - dt.timedelta(days=5)).strftime("%Y%m%d")
probe("Liquidity", "SZSE margin (RZRQ)",
      lambda: ak.stock_margin_szse(date=_SZSE_DATE),
      save_as="margin_szse_snapshot.csv",
      note=f"Shenzhen snapshot @ {_SZSE_DATE}; SSE series is primary")
# Northbound (Stock Connect) -- net-buy disclosure curtailed since Aug-2024.
probe("Liquidity", "Northbound flow (em)",
      lambda: ak.stock_hsgt_hist_em(symbol="北向资金"),  # 北向资金
      save_as="northbound_hist.csv", note="WARN curtailed since 2024-08")
# Social financing (TSF) increment / stock.
probe("Liquidity", "Social financing (TSF)",
      lambda: ak.macro_china_shrzgm(),
      save_as="social_financing.csv", note="monthly aggregate financing")
# Short rates (repo / SHIBOR) -- SHIBOR all tenors (FR007 already held locally).
probe("Liquidity", "SHIBOR all tenors",
      lambda: ak.macro_china_shibor_all(),
      save_as="shibor_all.csv", note="O/N..1Y SHIBOR; secondary (have FR007)")


# ===========================================================================
# 4. SENTIMENT  -- option PCR / IV (akshare option coverage is thin; probe)
# ===========================================================================
print("\n=== 4. Sentiment (options) ===")
probe("Sentiment", "50ETF QVIX",
      lambda: ak.index_option_50etf_qvix(),
      save_as="qvix_50etf.csv", note="50ETF QVIX (renamed index_option_*)")
probe("Sentiment", "300ETF QVIX",
      lambda: ak.index_option_300etf_qvix(),
      save_as="qvix_300etf.csv", note="300ETF QVIX (renamed index_option_*)")


# ===========================================================================
# 5. CROSS-ASSET / GLOBAL  -- China & US 10Y yields, FX, US equity/VIX, gold
# ===========================================================================
print("\n=== 5. Cross-Asset / Global ===")
# China + US treasury yields (2/5/10/30Y) in one frame -- gives 10Y CGB AND US10Y.
probe("CrossAsset", "China+US treasury yields",
      lambda: ak.bond_zh_us_rate(start_date=START),
      save_as="bond_cn_us_yields.csv", note="has China 10Y (-> ERP) and US 10Y")
# USD/CNY central parity.
probe("CrossAsset", "USDCNY central parity",
      lambda: ak.currency_boc_safe(),
      save_as="usdcny_boc.csv", note="SAFE central parity history")
# S&P500 via sina (works) as global risk proxy.
probe("CrossAsset", "S&P500 (sina)",
      lambda: ak.index_us_stock_sina(symbol=".INX"),
      save_as="sp500.csv", note="global risk proxy")
# Gold (Shanghai Gold Exchange spot).
probe("CrossAsset", "Gold SGE spot",
      lambda: ak.spot_hist_sge(symbol="Au99.99"),
      save_as="gold_sge.csv", note="gold proxy")


# ===========================================================================
# Write CSV summary + roll-up
# ===========================================================================
try:
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["category", "item", "status",
                                          "start", "rows", "cols", "note"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nWrote summary -> {OUT_CSV}")
    print(f"Saved series  -> {SAVE_DIR}")
except Exception as e:  # noqa
    print("Could not write CSV:", e)

from collections import Counter
c = Counter(r["status"] for r in results)
print("\n=== SUMMARY ===")
for k, v in c.most_common():
    print(f"  {k:10} {v}")
print("\nLegend: AVAILABLE=fetched & saved | EMPTY=returned nothing | ERROR=call failed")
print("ERRORs are usually a renamed akshare function or arg -- check the note and")
print("the akshare docs (akshare.akfamily.xyz) for the current signature.")
