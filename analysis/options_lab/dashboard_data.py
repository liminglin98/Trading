# -*- coding: utf-8 -*-
"""
dashboard_data.py
=================
Data layer for the Streamlit app's "Macro Dashboard" tab. Each series is kept
at its OWN NATIVE frequency (daily where the source actually publishes daily,
monthly where that's the real release cadence) and stored as its own CSV under
Data/akshare/dashboard/<key>.csv -- unlike build_model_panel.py's single
monthly panel (built for the forecasting model, which needs everything
collapsed to month-end), this is for DISPLAY: what's the latest print, and
how stale is it.

Reuses the same akshare sources build_model_panel.py already confirmed
reachable from this network (sina / legulegu / SAFE / SGE), plus real-economy
releases (official PMI, Caixin PMI, CPI, PPI, industrial production YoY,
retail sales) -- all confirmed reachable from this box with the proxy
stripped (see run_snapshot.sh's rationale: mainland feeds need a direct
connection, not the local Clash/V2Ray proxy).

Northbound flow was tried and dropped: even Eastmoney's live same-day snapshot
(stock_hsgt_fund_flow_summary_em) reports 0.0 for northbound net-buy now, so
this isn't a stale-column problem to route around -- the metric itself
stopped being published (~2024-08) and no working substitute exists.

Update model: no per-series release-calendar logic. Every run just re-fetches
each source's recent history and appends whatever dates aren't already
stored (keeping the freshest value on a revised date) -- the same
"safe to run many times, no-ops if nothing new" idiom as
options_lab/updater.py and run_snapshot.sh. This is what actually handles
"different release schedules" correctly: a series that hasn't published
a new print yet just contributes 0 new rows that run.

RUN IN A NORMAL PYTHON ENV (needs internet; proxy should be unset -- see
run_dashboard_update.sh):
    python dashboard_data.py
"""
from __future__ import annotations

import re
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import akshare as ak

HERE = Path(__file__).resolve().parent
TRADING_ROOT = HERE.parents[1]
DASH_DIR = TRADING_ROOT / "Data" / "akshare" / "dashboard"
DASH_DIR.mkdir(parents=True, exist_ok=True)

DAILY_MAX_LAG = 6      # calendar days; covers weekends + a holiday or two
MONTHLY_MAX_LAG = 50   # calendar days past the reference month-end before flagging stale
DISCONTINUED: dict[str, str] = {}


# ------------------------------------------------------------------ parsing helpers
def _parse_cn_month(s: str):
    """'2026年06月份' -> month-end Timestamp('2026-06-30')."""
    m = re.match(r"(\d{4})\D+(\d{1,2})\D", str(s) + "月")
    if not m:
        return pd.NaT
    return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1) + pd.offsets.MonthEnd(0)


def _parse_dates(col: pd.Series) -> pd.Series:
    ss = col.astype(str).str.strip()
    if ss.str.contains("年").any():        # '2026年06月份' -> Chinese Y/M format
        return ss.map(_parse_cn_month)
    if ss.str.fullmatch(r"\d{6}").all():   # '202604' -> plain YYYYMM
        return pd.to_datetime(ss, format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)
    return pd.to_datetime(ss, errors="coerce")


def _pick_col(df, candidates):
    cols = list(df.columns)
    low = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in str(c).lower():
                return c
    return None


def _as_series(df: pd.DataFrame, date_candidates: list[str], val_candidates: list[str]) -> pd.Series:
    dc = _pick_col(df, date_candidates) or df.columns[0]
    vc = _pick_col(df, val_candidates)
    if vc is None:
        num = df.select_dtypes("number").columns
        vc = num[-1] if len(num) else df.columns[-1]
    idx = _parse_dates(df[dc])
    val = pd.to_numeric(df[vc], errors="coerce")
    s = pd.Series(val.values, index=idx).dropna()
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


# ------------------------------------------------------------------ fetchers (one akshare call each)
# etf50/etf300/idx_sh are kept (not shown on the Macro Dashboard tab, which
# skips the "Market" group -- but the Technical tab charts these three
# directly by key, so they still need to be fetched/stored here).
def _fetch_etf50():      return _as_series(ak.fund_etf_hist_sina(symbol="sh510050"), ["date"], ["close"])
def _fetch_etf300():     return _as_series(ak.fund_etf_hist_sina(symbol="sh510300"), ["date"], ["close"])
def _fetch_idx_sh():     return _as_series(ak.stock_zh_index_daily(symbol="sh000001"), ["date"], ["close"])
def _fetch_margin():
    return _as_series(ak.stock_margin_sse(start_date="20100101", end_date=dt.date.today().strftime("%Y%m%d")),
                       ["信用交易日期", "日期", "date"],
                       ["融资余额", "融资融券余额"])
def _fetch_shibor():     return _as_series(ak.macro_china_shibor_all(), ["日期", "date"], ["1W", "1周", "O/N", "隔夜"])
def _fetch_cn10y():      return _as_series(ak.bond_zh_us_rate(start_date="20100101"), ["日期", "date"], ["中国国债收益率10年"])
def _fetch_us10y():      return _as_series(ak.bond_zh_us_rate(start_date="20100101"), ["日期", "date"], ["美国国债收益率10年"])
def _fetch_usdcny():     return _as_series(ak.currency_boc_safe(), ["日期", "date"], ["美元", "usd"])
def _fetch_sp500():      return _as_series(ak.index_us_stock_sina(symbol=".INX"), ["date"], ["close"])
def _fetch_gold():       return _as_series(ak.spot_hist_sge(symbol="Au99.99"), ["date"], ["close"])
def _fetch_qvix50():     return _as_series(ak.index_option_50etf_qvix(), ["date"], ["close"])
def _fetch_qvix300():    return _as_series(ak.index_option_300etf_qvix(), ["date"], ["close"])
def _fetch_tsf():        return _as_series(ak.macro_china_shrzgm(), ["月份", "日期"], ["社会融资规模增量", "增量", "规模"])
def _fetch_pmi_mfg():    return _as_series(ak.macro_china_pmi(), ["月份"], ["制造业-指数"])
# Official non-manufacturing PMI's source stalled (stuck ~11 months behind);
# Caixin services PMI is the standard substitute and stays current.
def _fetch_pmi_nonmfg(): return _as_series(ak.index_pmi_ser_cx(), ["日期"], ["服务业PMI"])
# Caixin manufacturing PMI via a fresher source (macro_china_cx_pmi_yearly
# stalled ~10 months behind; index_pmi_man_cx is the same series, current).
def _fetch_cx_pmi():     return _as_series(ak.index_pmi_man_cx(), ["日期"], ["制造业PMI"])
def _fetch_cpi():        return _as_series(ak.macro_china_cpi(), ["月份"], ["全国-同比增长"])
def _fetch_ppi():        return _as_series(ak.macro_china_ppi(), ["月份"], ["当月同比增长"])
# Jin10's industrial-production source stalled (~11 months behind); Eastmoney's
# 工业增加值增长 series covers the same concept and stays current.
def _fetch_ind_prod():   return _as_series(ak.macro_china_gyzjz(), ["月份"], ["同比增长"])
def _fetch_retail():     return _as_series(ak.macro_china_consumer_goods_retail(), ["月份"], ["同比增长"])


# key -> (label, group, unit, freq, fetch_fn)
SERIES: dict[str, tuple] = {
    "etf50":      ("50ETF close", "Market", "¥", "daily", _fetch_etf50),
    "etf300":     ("300ETF close", "Market", "¥", "daily", _fetch_etf300),
    "idx_sh":     ("Shanghai Composite", "Market", "", "daily", _fetch_idx_sh),
    "margin_bal": ("SSE margin financing balance", "Liquidity", "¥", "daily", _fetch_margin),
    "shibor_1w":  ("1W SHIBOR", "Liquidity", "%", "daily", _fetch_shibor),
    "cn10y":      ("China 10Y govt yield", "Rates", "%", "daily", _fetch_cn10y),
    "us10y":      ("US 10Y treasury yield", "Rates", "%", "daily", _fetch_us10y),
    "usdcny":     ("USDCNY central parity", "FX", "", "daily", _fetch_usdcny),
    "sp500":      ("S&P 500", "Global", "", "daily", _fetch_sp500),
    "gold":       ("Gold (SGE Au99.99)", "Global", "¥/g", "daily", _fetch_gold),
    "qvix_50":    ("50ETF QVIX", "Vol", "%", "daily", _fetch_qvix50),
    "qvix_300":   ("300ETF QVIX", "Vol", "%", "daily", _fetch_qvix300),
    "tsf":        ("Social financing (TSF) increment", "Real economy", "¥bn", "monthly", _fetch_tsf),
    "pmi_mfg":    ("Official manufacturing PMI", "Real economy", "", "monthly", _fetch_pmi_mfg),
    "pmi_nonmfg": ("Caixin services PMI", "Real economy", "", "monthly", _fetch_pmi_nonmfg),
    "cx_pmi":     ("Caixin manufacturing PMI", "Real economy", "", "monthly", _fetch_cx_pmi),
    "cpi":        ("CPI YoY", "Real economy", "%", "monthly", _fetch_cpi),
    "ppi":        ("PPI YoY", "Real economy", "%", "monthly", _fetch_ppi),
    "ind_prod":   ("Industrial production YoY", "Real economy", "%", "monthly", _fetch_ind_prod),
    "retail":     ("Retail sales YoY", "Real economy", "%", "monthly", _fetch_retail),
}


# ------------------------------------------------------------------ storage
def csv_path(key: str) -> Path:
    return DASH_DIR / f"{key}.csv"


def load_series(key: str) -> pd.Series:
    p = csv_path(key)
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(p, parse_dates=["date"])
    return df.set_index("date")["value"].sort_index()


def save_series(key: str, s: pd.Series) -> None:
    out = s.sort_index()
    out.index.name = "date"
    out.rename("value").to_frame().to_csv(csv_path(key))


def update_one(key: str) -> dict:
    _, _, _, _, fetch_fn = SERIES[key]
    try:
        fresh = fetch_fn()
    except Exception as e:
        return dict(key=key, ok=False, msg=f"{type(e).__name__}: {e}")
    if fresh is None or len(fresh) == 0:
        return dict(key=key, ok=False, msg="empty fetch")
    existing = load_series(key)
    combined = pd.concat([existing, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()  # fresh wins on revision
    n_new = len(combined) - len(existing)
    save_series(key, combined)
    return dict(key=key, ok=True,
                msg=f"{len(fresh)} fetched, {n_new} new, latest={combined.index.max().date()}")


def update_all() -> list[dict]:
    return [update_one(k) for k in SERIES]


def staleness(key: str) -> dict:
    label, group, unit, freq, _ = SERIES[key]
    s = load_series(key)
    if s.empty:
        return dict(key=key, label=label, group=group, unit=unit, freq=freq,
                    last_date=None, days_old=None, is_stale=True, value=None)
    last_date = s.index.max()
    days_old = (pd.Timestamp.now().normalize() - last_date).days
    max_lag = MONTHLY_MAX_LAG if freq == "monthly" else DAILY_MAX_LAG
    is_stale = (days_old > max_lag) and key not in DISCONTINUED
    return dict(key=key, label=label, group=group, unit=unit, freq=freq,
                last_date=last_date, days_old=days_old, is_stale=is_stale,
                value=float(s.iloc[-1]))


def all_staleness() -> pd.DataFrame:
    return pd.DataFrame([staleness(k) for k in SERIES])


def main() -> None:
    print(f"Updating {len(SERIES)} dashboard series -> {DASH_DIR}")
    n_ok = n_err = 0
    for r in update_all():
        flag = "ok" if r["ok"] else "ERR"
        n_ok += r["ok"]; n_err += not r["ok"]
        print(f"  [{flag:>3}] {r['key']:<12} {r['msg']}")
    print(f"Done: {n_ok} ok, {n_err} failed.")


if __name__ == "__main__":
    main()
