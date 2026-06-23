# -*- coding: utf-8 -*-
"""
qmt_data_availability_check.py
================================
Run this INSIDE the QMT (miniQMT / xtquant) Python editor to probe what market
data is actually available for the Butterfly next-month forecast model.

It walks the factor categories from `butterfly_factor_data_checklist.md` and, for
each candidate series, tries to:
  - download history (so it's cached locally),
  - fetch a sample,
  - report:  AVAILABLE / PARTIAL / MISSING, history start, #bars, frequency,
             and the field names returned.

Nothing here trades or places orders. It is read-only diagnostics.

ASCII ONLY: the QMT editor does not read Chinese characters, so all sector names,
            field labels and comments below use English / pinyin only.

Output: prints a report to the console AND writes a CSV summary next to this file
        (qmt_data_availability_report.csv) so you can paste results back into the
        checklist's "Notes to fill in from QMT" section.

NOTE: field/sector names differ slightly across QMT builds & brokers. Where a
      probe fails, the error is captured (not raised) so the whole report still
      runs. Treat MISSING as "not under this name" -> try the alt names noted.
"""

import os
import csv
import datetime as dt

# ---------------------------------------------------------------------------
# xtquant import
# ---------------------------------------------------------------------------
try:
    from xtquant import xtdata
except Exception as e:  # noqa
    print("!! Could not import xtquant.xtdata. Run this inside the QMT python "
          "environment (miniQMT must be running). Error:", e)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TODAY      = dt.date.today().strftime("%Y%m%d")
START_FAR  = "20100101"          # ask for a long history to find the true start
OUT_CSV    = os.path.join(os.path.dirname(os.path.abspath(__file__))
                          if "__file__" in globals() else ".",
                          "qmt_data_availability_report.csv")

results = []   # rows: dict(category, item, code, status, start, bars, freq, fields, note)


# ---------------------------------------------------------------------------
# Connection preflight
# ---------------------------------------------------------------------------
# Every probe needs a live quote-service link. If QMT is not logged in / the
# market-data link is down, xtdata raises "cannot connect to quote service"
# on every call. Check once up front and abort early with a clear message
# instead of producing dozens of identical errors.
def preflight(retries=6, wait_s=3):
    import time
    # Some builds expose connect(); call it if present (no-op if already up).
    try:
        if hasattr(xtdata, "connect"):
            xtdata.connect()
    except Exception:
        pass
    for i in range(retries):
        try:
            # A cheap call that requires the data link: trading calendar.
            cal = xtdata.get_trading_dates("SH", start_time="20240101",
                                           end_time=TODAY)
            if cal:
                print(f"Quote service OK (calendar returned {len(cal)} dates).")
                return True
        except Exception as e:  # noqa
            print(f"  preflight attempt {i+1}/{retries} failed: {e}")
        time.sleep(wait_s)
    print("\n!! Quote service NOT connected (OHLC/quote probes will ERROR).\n"
          "   This is EXPECTED if only the research terminal is open -- the\n"
          "   quote feed comes from the separate miniQMT trading client.\n"
          "   To get OHLC: (1) launch the miniQMT TRADING client too,\n"
          "                (2) confirm its market-data link shows CONNECTED,\n"
          "                (3) make sure you are NOT in simulation mode,\n"
          "                (4) wait a few seconds, then re-run.\n"
          "   Factor-library probes (research terminal) still run below.\n")
    return False


def _record(category, item, code, status, start="", bars="", freq="", fields="", note=""):
    row = dict(category=category, item=item, code=code, status=status,
               start=start, bars=bars, freq=freq, fields=fields, note=note)
    results.append(row)
    print(f"[{status:8}] {category:18} | {item:32} | {code:14} "
          f"| start={start:>8} bars={str(bars):>6} {freq:6} {note}")


# ---------------------------------------------------------------------------
# Core probe: download + fetch one instrument, report what came back
# ---------------------------------------------------------------------------
def probe_market(category, item, code, period="1d",
                 fields=("open", "high", "low", "close", "volume", "amount"),
                 note=""):
    """Try to download + read OHLC-style market data for `code`."""
    try:
        # cache to local store first (safe to call repeatedly)
        try:
            xtdata.download_history_data(code, period=period,
                                         start_time=START_FAR, end_time=TODAY)
        except Exception:
            pass  # some builds need the data already subscribed; keep going

        data = xtdata.get_market_data_ex(
            field_list=list(fields), stock_list=[code], period=period,
            start_time=START_FAR, end_time=TODAY, count=-1,
        )

        df = None
        if isinstance(data, dict):
            df = data.get(code)
        if df is None or len(df) == 0:
            _record(category, item, code, "MISSING", note=note or "no rows returned")
            return None

        idx = list(df.index)
        start = str(idx[0])[:8]
        bars = len(df)
        got_fields = ",".join(map(str, list(df.columns)[:8]))
        status = "AVAILABLE" if bars > 50 else "PARTIAL"
        _record(category, item, code, status, start=start, bars=bars,
                freq=period, fields=got_fields, note=note)
        return df
    except Exception as e:  # noqa
        _record(category, item, code, "ERROR", note=f"{note} :: {e}".strip())
        return None


def probe_fields(category, item, code, fields, period="1d", note=""):
    """Probe specific non-OHLC fields (e.g. pe, pb, turnover) on a code."""
    return probe_market(category, item, code, period=period,
                         fields=fields, note=note or "field probe")


# ---------------------------------------------------------------------------
# Factor-library probe (BARRA / style factors in the multi-factor library)
# ---------------------------------------------------------------------------
# These are NOT quote data. Per xtdata docs, factor data is materialized with
# generate_index_data(formula_name, ...) -> writes a .feather file, and it
# REQUIRES the research terminal (touyan) to be connected, with the
# formula_name existing there. Values are per-stock & cross-sectional.
def probe_factor(category, item, formula_name, stock_list, note=""):
    try:
        import pandas as pd
    except Exception as e:  # noqa
        _record(category, item, formula_name, "ERROR", note=f"pandas missing: {e}")
        return None
    out_path = os.path.join(os.path.dirname(OUT_CSV), f"factor_{formula_name}.feather")
    try:
        xtdata.generate_index_data(
            formula_name, formula_param={}, stock_list=stock_list,
            period="1d", dividend_type="none",
            start_time=START_FAR, end_time=TODAY,
            fill_mode="fixed", fill_value=float("nan"),
            result_path=out_path,
        )
        if not os.path.exists(out_path):
            _record(category, item, formula_name, "MISSING",
                    note=note or "generate_index_data produced no file")
            return None
        df = pd.read_feather(out_path)
        bars = len(df)
        cols = ",".join(map(str, list(df.columns)[:8]))
        status = "AVAILABLE" if bars > 0 else "MISSING"
        _record(category, item, formula_name, status, bars=bars, freq="1d",
                fields=cols, note=note or "via generate_index_data")
        return df
    except Exception as e:  # noqa
        _record(category, item, formula_name, "ERROR",
                note=f"needs research terminal (touyan) connected :: {e}")
        return None


# ===========================================================================
# 0. PREFLIGHT -- check the quote service (warn-only; do NOT abort)
# ===========================================================================
# The quote service and the research-terminal factor library are SEPARATE
# services. A quote-service failure should not stop the factor-library probes,
# so we only warn here and let each probe report its own status.
QUOTE_OK = preflight()
if not QUOTE_OK:
    print(">> Quote service down: OHLC/quote probes will ERROR, but factor-"
          "library probes (research terminal) will still be attempted.\n")


# ===========================================================================
# 1. MOMENTUM / TREND  -- price-only, derivable from any OHLC series
# ===========================================================================
print("\n=== 1. Momentum / Trend (need daily OHLC) ===")
# ETFs and their underlying indices. Index history usually >> ETF history.
momentum_codes = [
    ("50ETF",        "510050.SH"),
    ("300ETF",       "510300.SH"),
    ("SSE50 index",  "000016.SH"),
    ("CSI300 index", "000300.SH"),
    ("ChiNext ETF",  "159915.SZ"),   # bonus, in case you widen the universe
]
for name, code in momentum_codes:
    probe_market("Momentum", name, code, period="1d",
                 fields=("open", "high", "low", "close", "volume", "amount"),
                 note="OHLC -> 3/6/12m return, MA50/200, realized vol")


# ===========================================================================
# 1b. FACTOR LIBRARY (BARRA / style factors) -- research-terminal data
# ===========================================================================
# Factor names taken from the multi-factor library tree. These map onto the
# checklist buckets (momentum/liquidity/value/vol) but are PER-STOCK and
# cross-sectional, so they describe constituents, not the index time series.
# Probe on the index codes + a couple of large constituents as a connectivity
# and availability test.
print("\n=== 1b. Factor library (BARRA/style) -- needs research terminal ===")
FACTOR_UNIVERSE = ["000300.SH", "000016.SH", "600519.SH", "601318.SH"]
barra_factors = [
    ("rstr   (momentum)",    "rstr"),
    ("STOM   (liquidity)",   "STOM"),
    ("STOQ   (liquidity)",   "STOQ"),
    ("STOA   (liquidity)",   "STOA"),
    ("btop   (value)",       "btop"),
    ("Beta",                 "Beta"),
    ("HSIGMA (vol)",         "HSIGMA"),
    ("DASTD  (vol)",         "DASTD"),
    ("CMRA",                 "CMRA"),
    ("lncap  (size)",        "lncap"),
    ("mlev   (leverage)",    "mlev"),
    ("dtoa   (leverage)",    "dtoa"),
]
for name, formula in barra_factors:
    probe_factor("FactorLib", name, formula, FACTOR_UNIVERSE,
                 note="per-stock cross-sectional; aggregate for index model")


# ===========================================================================
# 2. VALUATION / MEAN-REVERSION  -- index-level PE / PB / div yield
# ===========================================================================
print("\n=== 2. Valuation (index PE/PB/div yield + 10Y CGB) ===")
# In many QMT builds these come through get_market_data_ex with extended fields
# on the INDEX code. Field names vary: try the common ones.
for name, code in [("CSI300 index", "000300.SH"), ("SSE50 index", "000016.SH")]:
    probe_fields("Valuation", f"{name} PE/PB", code,
                 fields=("pe", "pb", "pe_ttm", "pb_mrq", "dividend_yield"),
                 note="try also: peTTM/pbMRQ via financial data")

# 10-year China Government Bond yield -- code varies by data vendor.
# Common candidates seen in QMT/Wind-style feeds:
for name, code in [
    ("5Y T-bond fut (proxy)", "TF00.CFE"),    # price proxy, NOT a yield
    ("SSE T-bond index",      "010010.SH"),    # price index, NOT a yield
    ("China 10Y yield",       "CGB10Y.WI"),    # vendor yield code, often absent
]:
    probe_market("Valuation", name, code, period="1d",
                 fields=("close",),
                 note="10Y govt yield series rarely in QMT; may need external source")


# ===========================================================================
# 3. LIQUIDITY / MONETARY  -- margin balance, northbound, turnover, social fin
# ===========================================================================
print("\n=== 3. Liquidity / Monetary ===")

# 3a. Turnover / volume -- derivable from the amount/volume fields above,
#     but also try a market-aggregate turnover index if present.
for name, code in [("SH composite (amount)", "000001.SH"),
                   ("SZ component (amount)", "399001.SZ")]:
    probe_market("Liquidity", name, code, period="1d",
                 fields=("close", "volume", "amount"),
                 note="aggregate turnover vs trailing avg")

# 3b. Margin financing balance (RZRQ = rongzi rongquan). QMT exposes this for
#     some builds via get_market_data_ex with special fields. Probe them.
print("  -- margin balance (RZRQ) --")
margin_ok = False
try:
    # Some builds: fields rzye=financing bal, rqye=securities-lending bal,
    #              rzrqye=combined balance.
    df = xtdata.get_market_data_ex(
        field_list=["rzye", "rqye", "rzrqye"],
        stock_list=["000300.SH"], period="1d",
        start_time=START_FAR, end_time=TODAY, count=-1)
    sub = df.get("000300.SH") if isinstance(df, dict) else None
    if sub is not None and len(sub) > 0:
        _record("Liquidity", "Margin balance (RZRQ)", "000300.SH", "AVAILABLE",
                start=str(list(sub.index)[0])[:8], bars=len(sub), freq="1d",
                fields=",".join(map(str, sub.columns)),
                note="rzye/rqye fields present")
        margin_ok = True
except Exception as e:  # noqa
    _record("Liquidity", "Margin balance (RZRQ)", "000300.SH", "ERROR", note=str(e))
if not margin_ok:
    _record("Liquidity", "Margin balance (RZRQ)", "-", "CHECK",
            note="not under rzye/rqye; check sector/financial export or external source")

# 3c. Northbound flows (Stock Connect). Disclosure curtailed since Aug-2024.
print("  -- northbound flows (Stock Connect) --")
nb_ok = False
for name, code in [("HK->SH connect", "HGT.SH"), ("HK->SZ connect", "SGT.SZ")]:
    df = probe_market("Liquidity", f"Northbound {name}", code, period="1d",
                       fields=("close",),
                       note="WARN: real-time net-buy curtailed since 2024-08")
    if df is not None:
        nb_ok = True
if not nb_ok:
    _record("Liquidity", "Northbound flows", "-", "CHECK",
            note="likely needs fund-flow export; daily net-buy discontinued 2024")

# 3d. Social financing / credit impulse (TSF) + repo rates.
_record("Liquidity", "Social financing (TSF)", "-", "MANUAL",
        note="macro release, not in QMT tick store; pull from PBoC/Wind/external")
for name, code in [("DR007", "DR007.IB"), ("FR007", "FR007.IR"),
                   ("SHIBOR 1W", "SHIBOR1W.IR")]:
    probe_market("Liquidity", name, code, period="1d", fields=("close",),
                 note="repo/short rate; FR007 already held locally per checklist")


# ===========================================================================
# 4. SENTIMENT / POSITIONING  -- options: vol, OI, put/call, IV term
# ===========================================================================
print("\n=== 4. Sentiment / Options ===")

# 4a. Option universe -- can we even list the option chain?
print("  -- option chain discovery --")
opt_codes = []
try:
    # Common option sector names (ASCII forms used by various QMT builds).
    for sector in ["SHO", "SZO", "CFFEX Option", "Option", "ETF Option"]:
        try:
            lst = xtdata.get_stock_list_in_sector(sector)
            if lst:
                opt_codes.extend(lst)
                _record("Sentiment", f"Option list [{sector}]", "-", "AVAILABLE",
                        bars=len(lst), note=f"sample={lst[:3]}")
        except Exception:
            pass
    opt_codes = sorted(set(opt_codes))
    if not opt_codes:
        _record("Sentiment", "Option chain", "-", "CHECK",
                note="no option sector under common names; run xtdata.get_sector_list()")
except Exception as e:  # noqa
    _record("Sentiment", "Option chain", "-", "ERROR", note=str(e))

# 4b. Per-contract volume & open interest (needed for put/call ratio).
if opt_codes:
    sample = opt_codes[0]
    probe_market("Sentiment", "Option vol/OI per-contract", sample, period="1d",
                 fields=("close", "volume", "openInterest", "open_interest"),
                 note="need volume + OI for put/call ratio")
else:
    _record("Sentiment", "Option vol/OI per-contract", "-", "SKIP",
            note="no option codes discovered to probe")

# 4c. QVIX -- checklist says already held locally; flag the IV term-structure todo.
for name, code in [("50ETF", "510050.SH"), ("300ETF", "510300.SH")]:
    _record("Sentiment", f"{name} implied vol", code, "LOCAL",
            note="QVIX held in folder per checklist; verify near/next-term IV slope")


# ===========================================================================
# 5. CROSS-ASSET / GLOBAL  -- FX, US rates, US equity/VIX, commodities, gold
# ===========================================================================
print("\n=== 5. Cross-Asset / Global ===")
cross_codes = [
    ("USDCNY spot",   "USDCNY.FX"),
    ("USDCNH",        "USDCNH.FX"),
    ("US 10Y yield",  "US10Y.WI"),
    ("VIX",           "VIX.GI"),
    ("S&P500",        "SPX.GI"),
    ("Gold USD",      "XAU.FX"),
    ("Brent oil",     "OIL.WI"),
    ("Copper (SHFE)", "cu00.SF"),
]
for name, code in cross_codes:
    probe_market("CrossAsset", name, code, period="1d", fields=("close",),
                 note="vendor codes vary; CNY/QVIX already local per checklist")


# ===========================================================================
# Write CSV summary
# ===========================================================================
try:
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["category", "item", "code", "status",
                                          "start", "bars", "freq", "fields", "note"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nWrote summary -> {OUT_CSV}")
except Exception as e:  # noqa
    print("Could not write CSV:", e)

# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------
from collections import Counter
c = Counter(r["status"] for r in results)
print("\n=== SUMMARY ===")
for k, v in c.most_common():
    print(f"  {k:10} {v}")
print("\nLegend: AVAILABLE=usable history | PARTIAL=short/thin | LOCAL=already in folder")
print("        CHECK/MANUAL/SKIP=needs a different name or an external source")
print("        ERROR/MISSING=not under the probed code/field on this build")
print("\nNext: paste qmt_data_availability_report.csv back into the checklist's")
print("      'Notes to fill in from QMT' section, fixing any vendor-specific codes.")
