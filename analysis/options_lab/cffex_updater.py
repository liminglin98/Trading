"""EOD + backfill updater for CFFEX index options (上证50/沪深300/中证1000
股指期权), appending into the same Data/买断日更/consolidated/{underlying}.parquet
files analysis/options_lab/consolidated_store.py reads for these three
underlyings -- CFFEX index-option data never lives in Data/options/ CSVs the
way SSE/SZSE ETF options do (checked: consolidated_store.consolidated_path
maps 000300/000016/000852 straight to this parquet directory; nothing reads
a Data/options/option_daily_000300.csv equivalent for them).

Distinct from analysis/options_lab/minute_updater.py (unmerged branch
worktree-minute-option-fetcher): that polls live per-minute quotes; this is
a once-daily EOD append plus a one-off historical backfill, meant to run
once after close.

Sources (akshare, Sina's CFFEX quote/history feed):
  * ak.option_cffex_{hs300,zz1000,sz50}_list_sina() -- today's listed
    contract-months for each underlying.
  * ak.option_cffex_{hs300,zz1000,sz50}_spot_sina(month_symbol) -- one
    contract-month's full strike ladder, current snapshot (bid/ask/last/OI
    per call+put leg, no OHLC/volume/date). Used for the daily EOD append --
    call once after close (~15:05 China time) and treat "last" as that
    day's close, same convention updater.py uses for SSE (settle isn't
    published, so close is stored in both places).
  * ak.option_cffex_{hs300,zz1000,sz50}_daily_sina(contract_symbol) -- one
    specific contract's full OHLCV history by date. Used for backfill.
    Needs a real, currently-listed contract code, so this can only backfill
    contract-months still on today's list_sina roster -- fine for a
    multi-week gap (no monthly contract expires and rolls off that fast)
    but would miss anything that already expired before the backfill runs.

Field notes vs UNIFIED_COLS (same schema consolidate_etf_index_options.py
writes): contract_type/strike/expire_date parsed from the code (3rd-Friday-
of-month convention -- Sina's getRemainderDay has no CFFEX support, same
issue noted in minute_updater.py and updater.py). No settle price is
reported by either endpoint -> close is stored there too. delta/theta/
gamma/vega/rho left blank, matching the existing vendor_b rows already in
these files -- consolidated_store.py derives IV/spot/Greeks on demand
(_parity_forward/_cp_avg_iv/atm_iv_series) and never reads them from the
parquet, so leaving them blank here doesn't remove anything downstream
depends on.

Usage:
    python cffex_updater.py                    # today, all 3 underlyings
    python cffex_updater.py --backfill 20260728 20260810
    python cffex_updater.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import time
from datetime import date, datetime, timedelta

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CONSOLIDATED_DIR = os.path.join(BASE, "..", "..", "Data", "买断日更", "consolidated")

UNIFIED_COLS = [
    "date", "code", "underlying_symbol", "name", "contract_type", "strike",
    "expire_date", "list_date", "delist_date", "prev_settle", "prev_close",
    "open", "high", "low", "close", "settle", "volume", "turnover",
    "open_interest", "delta", "theta", "gamma", "vega", "rho", "source",
]

# akshare function-name key -> (underlying_symbol, parquet filename stem)
UNDERLYINGS = {
    "sz50": "000016.XSHG",
    "hs300": "000300.XSHG",
    "zz1000": "000852.XSHG",
}

CODE_RE = re.compile(r"^([a-zA-Z]+)(\d{4})([CP])(\d+(?:\.\d+)?)$")


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)  # first Friday
    return d + timedelta(days=14)


def _parse_code(code: str):
    """'io2608C4050' -> ('IO2608C4050', 'CO', 4050.0, date(2026,8,21))."""
    m = CODE_RE.match(code)
    if not m:
        return None
    prefix, yymm, cp, strike = m.groups()
    year, month = 2000 + int(yymm[:2]), int(yymm[2:])
    expiry = _third_friday(year, month)
    return code.upper(), ("CO" if cp == "C" else "PO"), float(strike), expiry


def contract_months(key: str) -> list[str]:
    import akshare as ak
    fn = getattr(ak, f"option_cffex_{key}_list_sina")
    data = fn()
    return next(iter(data.values())) if data else []


def spot_snapshot(key: str, month_symbol: str) -> pd.DataFrame:
    import akshare as ak
    fn = getattr(ak, f"option_cffex_{key}_spot_sina")
    return fn(symbol=month_symbol)


def daily_history(key: str, contract_symbol: str) -> pd.DataFrame:
    import akshare as ak
    fn = getattr(ak, f"option_cffex_{key}_daily_sina")
    return fn(symbol=contract_symbol)


def parquet_path(key: str) -> str:
    return os.path.join(CONSOLIDATED_DIR, f"{UNDERLYINGS[key]}.parquet")


def _rows_from_snapshot(key: str, underlying_symbol: str, month_symbol: str,
                        date_str: str) -> list[dict]:
    snap = spot_snapshot(key, month_symbol)
    if snap.empty:
        return []
    rows = []
    for _, r in snap.iterrows():
        for price_col, oi_col, code_col in (
            ("看涨合约-最新价", "看涨合约-持仓量", "看涨合约-标识"),
            ("看跌合约-最新价", "看跌合约-持仓量", "看跌合约-标识"),
        ):
            code = r.get(code_col)
            if not isinstance(code, str):
                continue
            parsed = _parse_code(code)
            if parsed is None:
                continue
            norm_code, cp, strike, expiry = parsed
            close = r.get(price_col)
            rows.append({
                "date": date_str, "code": norm_code,
                "underlying_symbol": underlying_symbol, "name": pd.NA,
                "contract_type": cp, "strike": strike,
                "expire_date": expiry.isoformat(), "list_date": pd.NA,
                "delist_date": pd.NA, "prev_settle": pd.NA, "prev_close": pd.NA,
                "open": pd.NA, "high": pd.NA, "low": pd.NA,
                "close": close, "settle": close, "volume": pd.NA,
                "turnover": pd.NA, "open_interest": r.get(oi_col),
                "delta": pd.NA, "theta": pd.NA, "gamma": pd.NA, "vega": pd.NA,
                "rho": pd.NA, "source": "akshare_cffex_spot",
            })
    return rows


def fetch_eod_snapshot(key: str, trade_date: date | None = None) -> pd.DataFrame:
    underlying_symbol = UNDERLYINGS[key]
    date_str = (trade_date or date.today()).isoformat()
    all_rows: list[dict] = []
    for month_symbol in contract_months(key):
        all_rows.extend(_rows_from_snapshot(key, underlying_symbol, month_symbol, date_str))
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame(columns=UNIFIED_COLS)
    return pd.DataFrame(all_rows)[UNIFIED_COLS]


def fetch_backfill(key: str, start: str, end: str) -> pd.DataFrame:
    """Full OHLCV history for every currently-listed contract of `key`,
    filtered to [start, end] (YYYYMMDD, inclusive)."""
    underlying_symbol = UNDERLYINGS[key]
    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    codes: set[str] = set()
    for month_symbol in contract_months(key):
        snap = spot_snapshot(key, month_symbol)
        time.sleep(0.2)
        if snap.empty:
            continue
        for col in ("看涨合约-标识", "看跌合约-标识"):
            codes.update(c for c in snap[col].dropna().tolist() if isinstance(c, str))

    frames = []
    for code in sorted(codes):
        parsed = _parse_code(code)
        if parsed is None:
            continue
        norm_code, cp, strike, expiry = parsed
        try:
            hist = daily_history(key, code)
        except Exception as e:
            print(f"  [{key}] {code}: daily_sina failed: {e}")
            continue
        time.sleep(0.2)
        if hist.empty:
            continue
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist[(hist["date"] >= pd.Timestamp(d0)) & (hist["date"] <= pd.Timestamp(d1))]
        if hist.empty:
            continue
        frames.append(pd.DataFrame({
            "date": hist["date"].dt.strftime("%Y-%m-%d"),
            "code": norm_code, "underlying_symbol": underlying_symbol,
            "name": pd.NA, "contract_type": cp, "strike": strike,
            "expire_date": expiry.isoformat(), "list_date": pd.NA,
            "delist_date": pd.NA, "prev_settle": pd.NA, "prev_close": pd.NA,
            "open": hist["open"].values, "high": hist["high"].values,
            "low": hist["low"].values, "close": hist["close"].values,
            "settle": hist["close"].values, "volume": hist["volume"].values,
            "turnover": pd.NA, "open_interest": pd.NA,
            "delta": pd.NA, "theta": pd.NA, "gamma": pd.NA, "vega": pd.NA,
            "rho": pd.NA, "source": "akshare_cffex_daily",
        }))
    if not frames:
        return pd.DataFrame(columns=UNIFIED_COLS)
    return pd.concat(frames, ignore_index=True)[UNIFIED_COLS]


def merge_into_parquet(key: str, new_rows: pd.DataFrame) -> int:
    """Replace any (code, date) already on file with the new rows -- same
    de-dup preference (new wins) as consolidate_etf_index_options.py."""
    if new_rows.empty:
        return 0
    path = parquet_path(key)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
        key_pairs = set(zip(new_rows["code"], new_rows["date"]))
        existing = existing[~existing.apply(
            lambda r: (r["code"], r["date"]) in key_pairs, axis=1)]
        combined = pd.concat([existing, new_rows], ignore_index=True)[UNIFIED_COLS]
    else:
        combined = new_rows
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date", "code"])
    os.makedirs(CONSOLIDATED_DIR, exist_ok=True)
    combined.to_parquet(path, index=False)
    return len(new_rows)


def run_day(keys: list[str], trade_date: date | None = None, dry_run: bool = False) -> None:
    for key in keys:
        rows = fetch_eod_snapshot(key, trade_date)
        if rows.empty:
            print(f"[{key}] no rows fetched (non-trading day or feed empty)")
            continue
        if dry_run:
            print(f"[{key}] {rows['date'].iloc[0]}: {len(rows)} contracts (dry-run)")
        else:
            n = merge_into_parquet(key, rows)
            print(f"[{key}] {rows['date'].iloc[0]}: {n} contracts written")


def run_backfill(keys: list[str], start: str, end: str, dry_run: bool = False) -> None:
    for key in keys:
        print(f"=== {key} {start}-{end} ===")
        rows = fetch_backfill(key, start, end)
        if rows.empty:
            print(f"[{key}] no rows in range")
            continue
        n_dates = rows["date"].nunique()
        if dry_run:
            print(f"[{key}] {len(rows)} rows across {n_dates} dates (dry-run)")
        else:
            n = merge_into_parquet(key, rows)
            print(f"[{key}] {n} rows across {n_dates} dates written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default=",".join(UNDERLYINGS))
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                    help="YYYYMMDD YYYYMMDD, inclusive")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    keys = args.keys.split(",")
    if args.backfill:
        run_backfill(keys, args.backfill[0], args.backfill[1], dry_run=args.dry_run)
    else:
        run_day(keys, dry_run=args.dry_run)
