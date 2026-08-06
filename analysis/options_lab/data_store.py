"""Shared data access for the options lab.

Historical option chain CSVs live in Data/options/option_daily_{code}.csv
with the original (Chinese) column schema. This module centralizes column
names, loading, caching helpers and appending.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_DIR = REPO / "Data" / "options"
POSITIONS_CSV = HERE / "positions.csv"

UNDERLYINGS = {"510050": "50ETF", "510300": "300ETF"}

# Auto-updatable via updater.py (Sina's live quote feed covers these too --
# verified 2026-07-24) but deliberately kept OUT of UNDERLYINGS: that dict is
# consumed directly by the Positions tab (margin, real position tracking) and
# Backtest tab (template engine), which have only been exercised against
# 50ETF/300ETF. Shown in the Data tab and Technical tab only, until
# Positions/Backtest get their own validation pass for these products.
EXT_LIVE_UNDERLYINGS = {"510500": "500ETF", "588000": "科创50ETF", "588080": "科创板50ETF"}

# Vendor-snapshot-backed, not Sina-live (Sina's category list has no SZSE
# options at all -- verified 2026-07-24/28), but same full Data/options CSV
# format as the two groups above. Materialized 2026-07-28 by
# Data/买断日更/materialize_szse_full_csv.py from the consolidated 买断日更
# parquet: IV and most Greeks (all of vendor B's June-2026-onward period,
# plus IV throughout since neither vendor ever carries it) are Black-Scholes
# derived, NOT independently validated against an authoritative source --
# unlike the SSE underlyings above, no official SZSE per-contract
# risk-indicator feed exists (checked; see reference_data_sources memory).
# Data/Technical tab only, same as EXT_LIVE_UNDERLYINGS -- Positions/Backtest
# haven't been exercised against these either.
EXT_SZSE_UNDERLYINGS = {"159901": "深证100ETF", "159915": "创业板ETF",
                       "159919": "300ETF(深)", "159922": "500ETF(深)"}

MULTIPLIER = 10000  # contract multiplier for SSE ETF options
ACCOUNT_CONFIG = HERE / "account_config.json"


def load_account_baseline() -> dict:
    """{'value': <yuan>, 'asof': <iso date>} -- the last known TRUE account
    value (cash + all positions marked), used to anchor the reconstructed
    equity curve. Defaults to a placeholder if never set."""
    import json
    if not ACCOUNT_CONFIG.exists():
        return {"value": 0.0, "asof": None}
    with open(ACCOUNT_CONFIG) as f:
        return json.load(f)


def save_account_baseline(value: float, asof: str) -> None:
    import json
    with open(ACCOUNT_CONFIG, "w") as f:
        json.dump({"value": value, "asof": asof}, f)

# ---------------------------------------------------------------- schema
C_CODE = "合约代码"
C_NAME = "合约简称"
C_MKT = "交易市场"
C_CP = "认购(C)认沽(P)类型"
C_STRIKE = "行权价(元)"
C_LIST = "上市日"
C_EXPIRY = "到期日"
C_DTE = "距到期日剩余天数"
C_DATE = "交易日期"
C_PREV_CLOSE = "前收盘价(元)"
C_PREV_SETTLE = "前结算价(元)"
C_OPEN = "开盘价(元)"
C_HIGH = "最高价(元)"
C_LOW = "最低价(元)"
C_CLOSE = "收盘价(元)"
C_SETTLE = "结算价(元)"
C_CHG = "涨跌额"
C_PCT = "涨跌幅(%)"
C_VOL = "成交量(手)"
C_AMT = "成交额(万元)"
C_OI = "持仓量(手)"
C_DELTA = "Delta"
C_THETA = "Theta"
C_GAMMA = "Gamma"
C_VEGA = "Vega"
C_RHO = "Rho"
C_IV = "隐含波动率"
C_UND = "期权标的代码"
C_UND_CLOSE = "期权标的收盘价"

COLUMNS = [C_CODE, C_NAME, C_MKT, C_CP, C_STRIKE, C_LIST, C_EXPIRY, C_DTE,
           C_DATE, C_PREV_CLOSE, C_PREV_SETTLE, C_OPEN, C_HIGH, C_LOW,
           C_CLOSE, C_SETTLE, C_CHG, C_PCT, C_VOL, C_AMT, C_OI,
           C_DELTA, C_THETA, C_GAMMA, C_VEGA, C_RHO, C_IV, C_UND, C_UND_CLOSE]


def chain_path(underlying: str) -> Path:
    return DATA_DIR / f"option_daily_{underlying}.csv"


def etf_path(underlying: str) -> Path:
    return DATA_DIR / f"etf_daily_{underlying}.csv"


def load_chain(underlying: str) -> pd.DataFrame:
    """Full option chain history for one underlying, typed and sorted."""
    path = chain_path(underlying)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={C_CODE: str, C_UND: str})
    df[C_DATE] = pd.to_datetime(df[C_DATE])
    df[C_EXPIRY] = pd.to_datetime(df[C_EXPIRY])
    for col in (C_STRIKE, C_CLOSE, C_SETTLE, C_OPEN, C_HIGH, C_LOW, C_VOL,
                C_OI, C_DELTA, C_THETA, C_GAMMA, C_VEGA, C_IV, C_UND_CLOSE,
                C_DTE, C_PREV_SETTLE, C_PREV_CLOSE):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_values([C_DATE, C_EXPIRY, C_CP, C_STRIKE], inplace=True)
    return df


def load_etf_ohlc(underlying: str) -> pd.DataFrame | None:
    """ETF daily OHLC if the updater has cached it; None otherwise."""
    path = etf_path(underlying)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def last_chain_date(underlying: str) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(chain_path(underlying), encoding="utf-8-sig",
                         usecols=[C_DATE])
    except (FileNotFoundError, ValueError):
        return None
    if df.empty:
        return None
    return pd.to_datetime(df[C_DATE]).max()


def append_chain(underlying: str, rows: pd.DataFrame) -> int:
    """Append new rows (already in schema order, all sharing one trade date
    -- see updater.fetch_chain_snapshot). If that date is already on file,
    REPLACES the existing rows for it rather than silently skipping them:
    app.py's "Update now" button can be clicked mid-trading-day, writing an
    intraday snapshot for "today"; a later same-day re-fetch (e.g. the
    scheduled EOD job) needs to be able to correct it, not have its real
    closing data discarded. This is what happened to 2026-08-05's data
    across all 5 underlyings before this fix -- see backfill_aug5.py repair
    / project_options_lab_architecture memory. Refuses dates older than
    what's on file (out-of-order write), same as before."""
    rows = rows[COLUMNS]
    path = chain_path(underlying)
    if not path.exists():
        rows.to_csv(path, index=False, encoding="utf-8-sig")
        return len(rows)
    new_dates = pd.to_datetime(rows[C_DATE]).unique()
    if len(new_dates) != 1:
        raise ValueError(f"append_chain expects rows for a single date, got {len(new_dates)}")
    new_date = pd.Timestamp(new_dates[0])
    existing_last = last_chain_date(underlying)
    if existing_last is not None and new_date < existing_last:
        return 0
    if existing_last is not None and new_date == existing_last:
        existing = pd.read_csv(path, encoding="utf-8-sig", dtype={C_CODE: str, C_UND: str})
        kept = existing[pd.to_datetime(existing[C_DATE]) != new_date]
        pd.concat([kept, rows], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
        return len(rows)
    rows.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    return len(rows)


# ---------------------------------------------------------- fund flow / sentiment
FUND_FLOW_COLUMNS = ["date", "underlying", "main_net", "small_net", "mid_net",
                     "large_net", "super_net"]
SENTIMENT_COLUMNS = ["date", "zt_count", "zb_count", "dt_count", "break_rate",
                     "max_height"]


def fund_flow_path() -> Path:
    return DATA_DIR / "fund_flow_daily.csv"


def sentiment_path() -> Path:
    return DATA_DIR / "market_sentiment_daily.csv"


def append_fund_flow(rows: pd.DataFrame) -> int:
    """Append EOD ETF fund-flow rows; refuses (date, underlying) pairs on file."""
    rows = rows[FUND_FLOW_COLUMNS]
    path = fund_flow_path()
    if path.exists():
        existing = pd.read_csv(path, usecols=["date", "underlying"],
                               dtype={"underlying": str})
        seen = set(zip(existing["date"], existing["underlying"]))
        rows = rows[[(d, u) not in seen for d, u in zip(rows["date"], rows["underlying"])]]
    if rows.empty:
        return 0
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(rows)


def append_sentiment(row: dict) -> int:
    """Append one market-wide limit-up/down sentiment row; refuses a date on file."""
    path = sentiment_path()
    if path.exists():
        existing = pd.read_csv(path, usecols=["date"])
        if str(row["date"]) in set(existing["date"].astype(str)):
            return 0
    pd.DataFrame([row], columns=SENTIMENT_COLUMNS).to_csv(
        path, mode="a", header=not path.exists(), index=False)
    return 1
