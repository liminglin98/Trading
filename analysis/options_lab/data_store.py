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
    """Append new rows (already in schema order); refuses duplicate dates."""
    rows = rows[COLUMNS]
    existing = last_chain_date(underlying)
    new_dates = pd.to_datetime(rows[C_DATE])
    if existing is not None:
        rows = rows[new_dates > existing]
    if rows.empty:
        return 0
    path = chain_path(underlying)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False,
                encoding="utf-8-sig")
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
