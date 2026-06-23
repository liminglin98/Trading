# -*- coding: utf-8 -*-
"""
build_model_panel.py
====================
Fetch the working akshare series and assemble ONE clean, model-ready MONTHLY panel
for the Butterfly next-month forecast (50ETF / 300ETF).

Design / no look-ahead:
  - Everything is reduced to month-end ('ME') frequency.
  - A row dated month t holds FEATURES known at the close of month t, and the
    TARGET = next-month log return (t -> t+1). The most recent month therefore has
    a NaN target (that row is the live forecast input; it is kept in the *_latest
    file and dropped from the training file).

Sources (all confirmed reachable on this network -- sina / legulegu / SAFE / SGE;
no eastmoney except optional northbound, which is wrapped in try/except):

RUN IN A NORMAL PYTHON ENV (not the QMT editor); needs internet:
    pip install akshare pandas numpy --upgrade
    python build_model_panel.py

Outputs (under Trading/Data/akshare/):
  panel_monthly.csv          full monthly panel (features + targets, NaNs kept)
  panel_monthly_train.csv    rows with a non-null target (for fitting)
  panel_monthly_latest.csv   the single most-recent row (live forecast features)
  panel_data_dictionary.csv  column -> description / source
"""

import os
import warnings
import datetime as dt

warnings.simplefilter("ignore")  # akshare/pandas emit harmless SettingWithCopy / format warnings

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
except Exception as e:  # noqa
    print("!! Need akshare pandas numpy:  pip install akshare pandas numpy --upgrade")
    print("   import error:", e)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def _find_trading_root(start):
    p = os.path.abspath(start)
    for _ in range(6):
        if os.path.isdir(os.path.join(p, "Data")) and os.path.isdir(os.path.join(p, "analysis")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.path.abspath(start)


TRADING  = _find_trading_root(HERE)
SAVE_DIR = os.path.join(TRADING, "Data", "akshare")
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Trading root: {TRADING}\nWriting panel to: {SAVE_DIR}\n")

DICT = []   # data-dictionary rows


def note(col, desc, src):
    DICT.append({"column": col, "description": desc, "source": src})


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def parse_dates(s):
    """Robustly parse a date column (handles 'YYYY-MM-DD', 'YYYYMMDD', date objs,
    and 'YYYYMM' month strings used by the social-financing series)."""
    ss = s.astype(str).str.strip()
    if ss.str.fullmatch(r"\d{6}").all():
        return pd.to_datetime(ss, format="%Y%m", errors="coerce")
    return pd.to_datetime(ss, errors="coerce")


def pick_col(df, candidates):
    """Return best-matching column. Prefer an EXACT (ci) name match across all
    candidates first (so '滚动市盈率' is not shadowed by '等权滚动市盈率'); fall
    back to substring match only if no exact match exists."""
    cols = list(df.columns)
    low = {str(c).lower(): c for c in cols}
    for cand in candidates:                      # exact pass
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:                      # substring pass
        for c in cols:
            if cand.lower() in str(c).lower():
                return c
    return None


def as_series(df, date_candidates, val_candidates, name):
    """Reduce an akshare DataFrame to a clean float Series indexed by datetime."""
    dc = pick_col(df, date_candidates) or df.columns[0]
    vc = pick_col(df, val_candidates)
    if vc is None:
        # fall back to last numeric column
        num = df.select_dtypes("number").columns
        vc = num[-1] if len(num) else df.columns[-1]
    idx = parse_dates(df[dc])
    val = pd.to_numeric(df[vc], errors="coerce")
    s = pd.Series(val.values, index=idx, name=name).dropna()
    s = s[~s.index.isna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def me_last(s):
    """Month-end last observation."""
    return s.sort_index().resample("ME").last()


def safe(fetch, label):
    """Run a fetch lambda; return Series or None and log."""
    try:
        s = fetch()
        if s is None or len(s) == 0:
            print(f"  [skip ] {label}: empty")
            return None
        print(f"  [ ok  ] {label}: {len(s)} obs, {s.index.min().date()} -> {s.index.max().date()}")
        return s
    except Exception as e:  # noqa
        print(f"  [ERR  ] {label}: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def price_features(daily, prefix):
    """Momentum / trend / realized-vol features from a daily close series, plus the
    next-month log-return TARGET. All snapshots taken at month-end (known at t)."""
    daily = daily.sort_index().dropna()
    ret = np.log(daily).diff()
    px_ma50 = daily / daily.rolling(50).mean() - 1.0
    px_ma200 = daily / daily.rolling(200).mean() - 1.0
    rv20 = ret.rolling(20).std() * np.sqrt(252)
    rv60 = ret.rolling(60).std() * np.sqrt(252)

    mc = me_last(daily)
    f = pd.DataFrame(index=mc.index)
    f[f"{prefix}_close"]    = mc
    f[f"{prefix}_mom_3m"]   = mc / mc.shift(3) - 1
    f[f"{prefix}_mom_6m"]   = mc / mc.shift(6) - 1
    f[f"{prefix}_mom_12m"]  = mc / mc.shift(12) - 1
    f[f"{prefix}_px_ma50"]  = me_last(px_ma50)
    f[f"{prefix}_px_ma200"] = me_last(px_ma200)
    f[f"{prefix}_rv20"]     = me_last(rv20)
    f[f"{prefix}_rv60"]     = me_last(rv60)
    # TARGET: next-month log return (t -> t+1)
    f[f"{prefix}_ret_next"] = np.log(mc.shift(-1) / mc)

    note(f"{prefix}_close",   "month-end close (level)", "sina")
    for h in (3, 6, 12):
        note(f"{prefix}_mom_{h}m", f"{h}-month trailing return", "derived")
    note(f"{prefix}_px_ma50",  "price / 50d MA - 1 (fast trend)", "derived")
    note(f"{prefix}_px_ma200", "price / 200d MA - 1 (trend filter)", "derived")
    note(f"{prefix}_rv20",     "annualized realized vol, 20d", "derived")
    note(f"{prefix}_rv60",     "annualized realized vol, 60d", "derived")
    note(f"{prefix}_ret_next", "TARGET: next-month log return", "derived")
    return f


def roll_pct(s, window=36):
    """Trailing percentile rank of the latest value within a rolling window."""
    return s.rolling(window, min_periods=12).apply(
        lambda x: (x <= x[-1]).mean(), raw=True)


# ===========================================================================
# 1) FETCH (reduce each source to a clean Series)
# ===========================================================================
print("Fetching sources...")
idx300 = safe(lambda: as_series(ak.stock_zh_index_daily(symbol="sh000300"),
                                ["date"], ["close"], "idx300"), "CSI300 index (sina)")
idx50  = safe(lambda: as_series(ak.stock_zh_index_daily(symbol="sh000016"),
                                ["date"], ["close"], "idx50"), "SSE50 index (sina)")
etf50  = safe(lambda: as_series(ak.fund_etf_hist_sina(symbol="sh510050"),
                                ["date"], ["close"], "etf50"), "50ETF (sina)")
etf300 = safe(lambda: as_series(ak.fund_etf_hist_sina(symbol="sh510300"),
                                ["date"], ["close"], "etf300"), "300ETF (sina)")

pe300 = safe(lambda: as_series(ak.stock_index_pe_lg(symbol="沪深300"),
                               ["日期", "date"], ["滚动市盈率", "市盈率", "pe"], "pe300"), "CSI300 PE")
pe50  = safe(lambda: as_series(ak.stock_index_pe_lg(symbol="上证50"),
                               ["日期", "date"], ["滚动市盈率", "市盈率", "pe"], "pe50"), "SSE50 PE")
pb300 = safe(lambda: as_series(ak.stock_index_pb_lg(symbol="沪深300"),
                               ["日期", "date"], ["市净率", "pb"], "pb300"), "CSI300 PB")

margin = safe(lambda: as_series(ak.stock_margin_sse(start_date="20100101",
                                                    end_date=dt.date.today().strftime("%Y%m%d")),
                                ["信用交易日期", "日期", "date"],
                                ["融资余额", "融资融券余额"], "margin"), "SSE margin (RZRQ)")
shibor = safe(lambda: as_series(ak.macro_china_shibor_all(),
                                ["日期", "date"], ["1W", "1周", "O/N", "隔夜"], "shibor"), "SHIBOR")

ylds_raw = None
try:
    ylds_raw = ak.bond_zh_us_rate(start_date="20100101")
    print(f"  [ ok  ] CN+US yields: {len(ylds_raw)} rows")
except Exception as e:  # noqa
    print(f"  [ERR  ] CN+US yields: {e}")
cn10 = us10 = None
if ylds_raw is not None:
    cn10 = as_series(ylds_raw, ["日期", "date"], ["中国国债收益率10年"], "cn10y")
    us10 = as_series(ylds_raw, ["日期", "date"], ["美国国债收益率10年"], "us10y")

usdcny = safe(lambda: as_series(ak.currency_boc_safe(),
                                ["日期", "date"], ["美元", "usd"], "usdcny"), "USDCNY parity")
sp500  = safe(lambda: as_series(ak.index_us_stock_sina(symbol=".INX"),
                                ["date"], ["close"], "sp500"), "S&P500 (sina)")
gold   = safe(lambda: as_series(ak.spot_hist_sge(symbol="Au99.99"),
                                ["date"], ["close"], "gold"), "Gold SGE")
qvix50 = safe(lambda: as_series(ak.index_option_50etf_qvix(),
                                ["date"], ["close"], "qvix50"), "50ETF QVIX")
qvix300 = safe(lambda: as_series(ak.index_option_300etf_qvix(),
                                 ["date"], ["close"], "qvix300"), "300ETF QVIX")
tsf = safe(lambda: as_series(ak.macro_china_shrzgm(),
                             ["月份", "日期"], ["社会融资规模增量", "增量", "规模"], "tsf"),
           "Social financing (TSF)")
# Northbound is eastmoney-hosted (may need proxy); optional.
nbnd = safe(lambda: as_series(ak.stock_hsgt_hist_em(symbol="北向资金"),
                              ["日期", "date"], ["当日成交净买额", "净买额", "净流入"], "nbnd"),
            "Northbound (em, optional)")


# ===========================================================================
# 2) ASSEMBLE MONTHLY PANEL
# ===========================================================================
print("\nAssembling monthly panel...")
parts = []

# --- momentum / trend / vol + targets (per traded instrument) ---
if etf50 is not None:
    parts.append(price_features(etf50, "etf50"))
if etf300 is not None:
    parts.append(price_features(etf300, "etf300"))
# index momentum (longer history; useful regime context)
if idx300 is not None:
    p = price_features(idx300, "idx300")
    parts.append(p.drop(columns=["idx300_ret_next"]))  # target only from ETFs
if idx50 is not None:
    p = price_features(idx50, "idx50")
    parts.append(p.drop(columns=["idx50_ret_next"]))

panel = pd.concat(parts, axis=1) if parts else pd.DataFrame()


def add(col, series, desc, src, transform=None):
    """Add a month-end feature column to the panel."""
    global panel
    if series is None:
        return
    s = me_last(series)
    if transform == "mom":
        s = s.pct_change()
    elif transform == "diff":
        s = s.diff()
    elif transform == "logret":
        s = np.log(s / s.shift(1))
    panel = panel.join(s.rename(col), how="outer") if len(panel) else s.rename(col).to_frame()
    note(col, desc, src)


# --- valuation ---
add("pe_300", pe300, "CSI300 trailing P/E (TTM)", "legulegu")
add("pe_50",  pe50,  "SSE50 trailing P/E (TTM)", "legulegu")
add("pb_300", pb300, "CSI300 P/B", "legulegu")
if pe300 is not None:
    ey = 1.0 / me_last(pe300)
    panel = panel.join(ey.rename("ey_300"), how="outer")
    note("ey_300", "CSI300 earnings yield = 1/PE", "derived")
if pe50 is not None:
    panel = panel.join((1.0 / me_last(pe50)).rename("ey_50"), how="outer")
    note("ey_50", "SSE50 earnings yield = 1/PE", "derived")
# equity risk premium = earnings yield - 10Y CGB (Fed-model spread)
if pe300 is not None and cn10 is not None:
    erp = (1.0 / me_last(pe300)) - me_last(cn10) / 100.0
    panel = panel.join(erp.rename("erp_300"), how="outer")
    note("erp_300", "CSI300 ERP = 1/PE - China 10Y yield", "derived")

# --- liquidity / monetary ---
add("margin_bal", margin, "SSE margin financing balance (level)", "akshare/SSE")
add("margin_mom", margin, "SSE margin balance MoM growth", "derived", transform="mom")
add("shibor_1w", shibor, "1-week SHIBOR (level)", "akshare")
add("shibor_chg", shibor, "1-week SHIBOR MoM change", "derived", transform="diff")
# TSF for month t is released ~15th of t+1 -> lag 1 month to avoid look-ahead.
if tsf is not None:
    tsf_lag = me_last(tsf).shift(1)
    panel = panel.join(tsf_lag.rename("tsf_increment_lag1"), how="outer")
    note("tsf_increment_lag1", "social financing increment, lagged 1m (release delay)", "akshare")
if nbnd is not None:
    nm = nbnd.resample("ME").sum()  # monthly net northbound flow
    panel = panel.join(nm.rename("nbnd_net_m"), how="outer")
    note("nbnd_net_m", "monthly net northbound flow (curtailed after 2024-08)", "eastmoney")

# --- sentiment ---
add("qvix_50", qvix50, "50ETF implied-vol index (QVIX) level", "akshare")
add("qvix_300", qvix300, "300ETF implied-vol index (QVIX) level", "akshare")
if qvix50 is not None:
    panel = panel.join(roll_pct(me_last(qvix50)).rename("qvix_50_pct"), how="outer")
    note("qvix_50_pct", "QVIX 50 trailing 36m percentile", "derived")
if qvix300 is not None:
    panel = panel.join(roll_pct(me_last(qvix300)).rename("qvix_300_pct"), how="outer")
    note("qvix_300_pct", "QVIX 300 trailing 36m percentile", "derived")

# --- cross-asset / global ---
add("usdcny", usdcny, "USDCNY central parity (level)", "SAFE")
add("usdcny_mom", usdcny, "USDCNY MoM change", "derived", transform="mom")
add("cn10y", cn10, "China 10Y govt bond yield (%)", "akshare")
add("us10y", us10, "US 10Y treasury yield (%)", "akshare")
add("us10y_chg", us10, "US 10Y MoM change", "derived", transform="diff")
add("sp500_ret", sp500, "S&P500 monthly log return", "derived", transform="logret")
add("gold_ret", gold, "SGE gold monthly log return", "derived", transform="logret")

# ---------------------------------------------------------------------------
# 3) CLEAN + SAVE
# ---------------------------------------------------------------------------
panel = panel.sort_index()
panel.index.name = "month_end"

# Trim to the common modern window (margin/QVIX start ~2015-2018; pre-2015 has
# mostly price+valuation only). Keep full file, but flag a recommended start.
full_path  = os.path.join(SAVE_DIR, "panel_monthly.csv")
panel.to_csv(full_path, encoding="utf-8-sig")

target_cols = [c for c in panel.columns if c.endswith("_ret_next")]
train = panel.dropna(subset=target_cols, how="all")          # need at least one target
train_path = os.path.join(SAVE_DIR, "panel_monthly_train.csv")
train.to_csv(train_path, encoding="utf-8-sig")

latest = panel.iloc[[-1]]                                     # live forecast features
latest_path = os.path.join(SAVE_DIR, "panel_monthly_latest.csv")
latest.to_csv(latest_path, encoding="utf-8-sig")

dict_df = pd.DataFrame(DICT).drop_duplicates("column")
dict_path = os.path.join(SAVE_DIR, "panel_data_dictionary.csv")
dict_df.to_csv(dict_path, index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 4) COVERAGE REPORT
# ---------------------------------------------------------------------------
print("\n=== PANEL COVERAGE ===")
print(f"shape: {panel.shape[0]} months x {panel.shape[1]} cols | "
      f"{panel.index.min().date()} -> {panel.index.max().date()}")
cov = panel.notna().sum().sort_values(ascending=False)
firstvalid = {c: (panel[c].first_valid_index().date() if panel[c].first_valid_index() is not None else None)
              for c in panel.columns}
print(f"{'column':22} {'n':>5}  first_valid")
for c in panel.columns:
    print(f"{c:22} {cov[c]:>5}  {firstvalid[c]}")

print("\nSaved:")
print(f"  full   -> {full_path}  ({panel.shape[0]} rows)")
print(f"  train  -> {train_path}  ({train.shape[0]} rows with target)")
print(f"  latest -> {latest_path}  (1 row, live forecast features; target is NaN)")
print(f"  dict   -> {dict_path}  ({len(dict_df)} columns documented)")
print("\nModeling notes:")
print("  - Predict *_ret_next (next-month log return). Features are month-end, no look-ahead.")
print("  - Margin/QVIX/SHIBOR start ~2015-2018; for the full feature set use rows from ~2018 on.")
print("  - Northbound (nbnd_net_m) is unreliable after 2024-08 -- treat as optional.")
