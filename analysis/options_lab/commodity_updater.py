"""EOD updater for SHFE/DCE/CZCE/GFEX commodity options.

Fills the same file Data/买断日更/consolidate_commodity_options.py writes
(Data/options/option_daily_{CODE}.csv, UNIFIED_COLS schema) but from a live,
per-exchange, per-trade-date source instead of the static paid vendor drops
that script reads -- so it works both as a one-off backfill for whatever gap
has opened up since the last vendor drop, and as an ongoing daily job.

Sources (each exchange's own official site, via akshare):
  * ak.option_hist_shfe(symbol, trade_date)  -- SHFE
  * ak.option_hist_dce(symbol, trade_date)   -- DCE
  * ak.option_hist_czce(symbol, trade_date)  -- CZCE
  * ak.option_hist_gfex(symbol, trade_date)  -- GFEX
All four internally validate trade_date against an exchange trading
calendar and return an empty frame (with a warning) for non-trading days,
so backfill can just walk every calendar day in the range without its own
holiday logic.

Coverage: the variety maps below cover every SHFE/DCE/CZCE/GFEX option
symbol these four akshare functions know about (53 of the ~64 product
codes seen under Data/options/option_daily_*.csv). The remainder (AD, BC,
BU, BZ, FU, JM, NR, OP, PD, PT, SP) have no known akshare mapping -- run()
skips them with a warning rather than guess.

DCE fallback: dce.com.cn itself returns HTTP 412 (a JS-challenge WAF) from
this network -- confirmed not specific to option_hist_dce, the bare
homepage 412s too. 9 of DCE's 17 varieties (M/C/I/PG/A/B/Y/EG/EB) are
separately reachable via Sina's commodity-option pages
(stock.finance.sina.com.cn/futures/view/optionsDP.php), a different host
entirely, so they get a Sina-backed fallback (DCE_SINA_VARIETIES /
dce_sina_* below) instead of being left unmapped like the other 8 DCE
varieties (L/V/PP/P/JD/CS/LH/LG, confirmed absent from Sina's page too --
no known free source for those). Sina's contract_table_sina gives a live
quote snapshot (no OHLC/settle/turnover, just last price + OI), used as
the daily EOD source; option_commodity_hist_sina gives real per-contract
daily history, used for backfill.

Field notes vs UNIFIED_COLS:
  * underlying_symbol/name/expire_date/list_date/delist_date aren't
    reported by these endpoints -- left blank, same as vendor_b rows in
    the existing CSVs.
  * contract_type/strike are parsed off the contract code's C/P<strike>
    suffix (same convention consolidate_etf_index_options.py uses).
  * prev_close isn't reported (only prev_settle) -- left blank rather than
    duplicating prev_settle into it.
  * theta/gamma/vega/rho aren't reported -- left blank. delta *is*
    reported by all four exchanges and filled in (an improvement over the
    existing vendor_b rows, which have no Greeks at all).
  * turnover: per the akshare module docstring, all four exchanges report
    this in 万元 (10k yuan); the existing vendor_b rows in these CSVs are in
    plain yuan (cross-checked CU2608C104000 2026-07-27: turnover/volume
    implies ~yuan, matching settle_price * 5-ton multiplier). Multiplied by
    1e4 below so both sources share one unit within a file.
  * code case: akshare returns lowercase product prefixes (e.g.
    "cu2609C76000"); uppercased to match the existing CU2608C100000 style.

Usage:
    python commodity_updater.py                        # today, all varieties
    python commodity_updater.py --varieties CU,AU,SR
    python commodity_updater.py --backfill 20260728 20260808
    python commodity_updater.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import time
from datetime import date, datetime, timedelta

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "..", "..", "Data", "options")

UNIFIED_COLS = [
    "date", "code", "underlying_symbol", "name", "contract_type", "strike",
    "expire_date", "list_date", "delist_date", "prev_settle", "prev_close",
    "open", "high", "low", "close", "settle", "volume", "turnover",
    "open_interest", "delta", "theta", "gamma", "vega", "rho", "source",
]

SHFE_MAP = {
    "SC": "原油期权", "CU": "铜期权", "AL": "铝期权", "ZN": "锌期权", "PB": "铅期权",
    "RB": "螺纹钢期权", "NI": "镍期权", "SN": "锡期权", "AO": "氧化铝期权",
    "AU": "黄金期权", "AG": "白银期权", "BR": "丁二烯橡胶期权", "RU": "天胶期权",
}

DCE_MAP = {
    "C": "玉米期权", "M": "豆粕期权", "I": "铁矿石期权", "PG": "液化石油气期权",
    "L": "聚乙烯期权", "V": "聚氯乙烯期权", "PP": "聚丙烯期权", "P": "棕榈油期权",
    "A": "黄大豆1号期权", "B": "黄大豆2号期权", "Y": "豆油期权", "EG": "乙二醇期权",
    "EB": "苯乙烯期权", "JD": "鸡蛋期权", "CS": "玉米淀粉期权", "LH": "生猪期权",
    "LG": "原木期权",
}

CZCE_MAP = {
    "SR": "白糖期权", "CF": "棉花期权", "MA": "甲醇期权", "TA": "PTA期权",
    "ZC": "动力煤期权", "RM": "菜籽粕期权", "OI": "菜籽油期权", "PK": "花生期权",
    "PX": "对二甲苯期权", "SH": "烧碱期权", "SA": "纯碱期权", "PF": "短纤期权",
    "SM": "锰硅期权", "SF": "硅铁期权", "UR": "尿素期权", "AP": "苹果期权",
    "CJ": "红枣期权", "FG": "玻璃期权", "PR": "瓶片期权", "PL": "丙烯期权",
}

GFEX_MAP = {"SI": "工业硅", "LC": "碳酸锂", "PS": "多晶硅"}

ALL_VARIETIES = {**SHFE_MAP, **DCE_MAP, **CZCE_MAP, **GFEX_MAP}

# Subset of DCE_MAP reachable via Sina instead of dce.com.cn directly (see
# module docstring). Chinese names match DCE_MAP's exactly, so no separate
# mapping dict is needed -- just a smaller key set.
DCE_SINA_VARIETIES = {"M", "C", "I", "PG", "A", "B", "Y", "EG", "EB"}

CP_STRIKE_RE = re.compile(r"(C|P)(\d+(?:\.\d+)?)$")


def _cp_strike(code: str) -> tuple[str | None, float | None]:
    m = CP_STRIKE_RE.search(code)
    if not m:
        return None, None
    return ("CO" if m.group(1) == "C" else "PO"), float(m.group(2))


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIFIED_COLS)


def _finish(df: pd.DataFrame, date_str: str, code_col: pd.Series,
           open_col, high_col, low_col, close_col, prev_settle_col,
           settle_col, volume_col, turnover_col, oi_col, delta_col,
           source: str) -> pd.DataFrame:
    codes = code_col.astype(str).str.upper()
    cp_strike = codes.map(_cp_strike)
    turnover = pd.to_numeric(turnover_col, errors="coerce") * 1e4
    out = pd.DataFrame({
        "date": date_str,
        "code": codes,
        "underlying_symbol": pd.NA,
        "name": pd.NA,
        "contract_type": [t for t, _ in cp_strike],
        "strike": [s for _, s in cp_strike],
        "expire_date": pd.NA,
        "list_date": pd.NA,
        "delist_date": pd.NA,
        "prev_settle": prev_settle_col.values,
        "prev_close": pd.NA,
        "open": open_col.values,
        "high": high_col.values,
        "low": low_col.values,
        "close": close_col.values,
        "settle": settle_col.values,
        "volume": volume_col.values,
        "turnover": turnover.values,
        "open_interest": oi_col.values,
        "delta": delta_col.values,
        "theta": pd.NA,
        "gamma": pd.NA,
        "vega": pd.NA,
        "rho": pd.NA,
        "source": source,
    })
    return out[UNIFIED_COLS]


def fetch_shfe(variety: str, trade_date: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.option_hist_shfe(symbol=SHFE_MAP[variety], trade_date=trade_date)
    if df.empty:
        return _empty()
    date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    return _finish(df, date_str, df["合约代码"], df["开盘价"], df["最高价"],
                  df["最低价"], df["收盘价"], df["前结算价"], df["结算价"],
                  df["成交量"], df["成交额"], df["持仓量"], df["德尔塔"],
                  "akshare_shfe")


def fetch_dce(variety: str, trade_date: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.option_hist_dce(symbol=DCE_MAP[variety], trade_date=trade_date)
    if df.empty:
        return _empty()
    date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    return _finish(df, date_str, df["合约"], df["开盘价"], df["最高价"],
                  df["最低价"], df["收盘价"], df["前结算价"], df["结算价"],
                  df["成交量"], df["成交额"], df["持仓量"], df["Delta"],
                  "akshare_dce")


def fetch_czce(variety: str, trade_date: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.option_hist_czce(symbol=CZCE_MAP[variety], trade_date=trade_date)
    if df.empty:
        return _empty()
    date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    return _finish(df, date_str, df["合约代码"], df["今开盘"], df["最高价"],
                  df["最低价"], df["今收盘"], df["昨结算"], df["今结算"],
                  df["成交量(手)"], df["成交额(万元)"], df["持仓量"], df["DELTA"],
                  "akshare_czce")


def fetch_gfex(variety: str, trade_date: str) -> pd.DataFrame:
    import akshare as ak
    df = ak.option_hist_gfex(symbol=GFEX_MAP[variety], trade_date=trade_date)
    if df.empty:
        return _empty()
    date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    return _finish(df, date_str, df["合约名称"], df["开盘价"], df["最高价"],
                  df["最低价"], df["收盘价"], df["前结算价"], df["结算价"],
                  df["成交量"], df["成交额"], df["持仓量"], df["Delta"],
                  "akshare_gfex")


def dce_sina_contract_months(variety: str) -> list[str]:
    import akshare as ak
    df = ak.option_commodity_contract_sina(symbol=DCE_MAP[variety])
    return df["合约"].tolist() if not df.empty else []


def dce_sina_contract_table(variety: str, month: str) -> pd.DataFrame:
    import akshare as ak
    return ak.option_commodity_contract_table_sina(symbol=DCE_MAP[variety], contract=month)


def fetch_dce_sina_day(variety: str, date_str: str) -> pd.DataFrame:
    """EOD snapshot for a Sina-covered DCE variety: one live quote call per
    contract-month (no trade_date parameter on this endpoint -- meant to be
    called once after close, treating 最新价 as that day's close/settle,
    same convention as cffex_updater.py's spot-snapshot path)."""
    rows = []
    for month in dce_sina_contract_months(variety):
        table = dce_sina_contract_table(variety, month)
        time.sleep(0.3)
        if table.empty:
            continue
        for _, r in table.iterrows():
            for price_col, oi_col, code_col in (
                ("看涨合约-最新价", "看涨合约-持仓量", "看涨合约-看涨期权合约"),
                ("看跌合约-最新价", "看跌合约-持仓量", "看跌合约-看跌期权合约"),
            ):
                code = r.get(code_col)
                if not isinstance(code, str) or not code:
                    continue
                cp, strike = _cp_strike(code.upper())
                rows.append({
                    "date": date_str, "code": code.upper(),
                    "underlying_symbol": pd.NA, "name": pd.NA,
                    "contract_type": cp,
                    "strike": strike if strike is not None else r.get("行权价"),
                    "expire_date": pd.NA, "list_date": pd.NA, "delist_date": pd.NA,
                    "prev_settle": pd.NA, "prev_close": pd.NA,
                    "open": pd.NA, "high": pd.NA, "low": pd.NA,
                    "close": r.get(price_col), "settle": r.get(price_col),
                    "volume": pd.NA, "turnover": pd.NA,
                    "open_interest": r.get(oi_col),
                    "delta": pd.NA, "theta": pd.NA, "gamma": pd.NA,
                    "vega": pd.NA, "rho": pd.NA, "source": "akshare_dce_sina_spot",
                })
    if not rows:
        return _empty()
    return pd.DataFrame(rows)[UNIFIED_COLS]


def fetch_dce_sina_backfill(variety: str, start: str, end: str) -> pd.DataFrame:
    """Real per-contract daily OHLCV history (option_commodity_hist_sina),
    filtered to [start, end] -- one call per contract discovered off the
    variety's currently-listed contract-months, same limitation as
    cffex_updater.py's backfill (can't reach contracts that already expired
    off the roster)."""
    import akshare as ak
    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    codes: set[str] = set()
    for month in dce_sina_contract_months(variety):
        table = dce_sina_contract_table(variety, month)
        time.sleep(0.3)
        if table.empty:
            continue
        codes.update(c for c in table["看涨合约-看涨期权合约"].dropna().tolist() if c)
        codes.update(c for c in table["看跌合约-看跌期权合约"].dropna().tolist() if c)

    frames = []
    for code in sorted(codes):
        cp, strike = _cp_strike(code.upper())
        try:
            hist = ak.option_commodity_hist_sina(symbol=code)
        except Exception as e:
            print(f"  [{variety}] {code}: hist_sina failed: {e}")
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
            "code": code.upper(), "underlying_symbol": pd.NA, "name": pd.NA,
            "contract_type": cp, "strike": strike,
            "expire_date": pd.NA, "list_date": pd.NA, "delist_date": pd.NA,
            "prev_settle": pd.NA, "prev_close": pd.NA,
            "open": hist["open"].values, "high": hist["high"].values,
            "low": hist["low"].values, "close": hist["close"].values,
            "settle": hist["close"].values, "volume": hist["volume"].values,
            "turnover": pd.NA, "open_interest": pd.NA,
            "delta": pd.NA, "theta": pd.NA, "gamma": pd.NA, "vega": pd.NA,
            "rho": pd.NA, "source": "akshare_dce_sina_hist",
        }))
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True)[UNIFIED_COLS]


_FETCHERS = ({v: fetch_shfe for v in SHFE_MAP}
            | {v: fetch_dce for v in DCE_MAP}
            | {v: fetch_czce for v in CZCE_MAP}
            | {v: fetch_gfex for v in GFEX_MAP})


def fetch_variety(variety: str, trade_date: str) -> pd.DataFrame:
    fn = _FETCHERS.get(variety)
    if fn is None:
        raise KeyError(variety)
    return fn(variety, trade_date)


def out_path(variety: str) -> str:
    return os.path.join(OUT_DIR, f"option_daily_{variety}.csv")


def append_day(variety: str, rows: pd.DataFrame) -> int:
    """Merge rows into the variety's CSV, replacing any existing rows for
    every date present in `rows` (same semantics as data_store.append_chain
    -- a rerun corrects rather than duplicates). Rows may span a single date
    (the normal daily-EOD case) or many (the Sina backfill path)."""
    if rows.empty:
        return 0
    path = out_path(variety)
    dates = set(rows["date"])
    if os.path.exists(path):
        existing = pd.read_csv(path, dtype={"code": str})
        existing = existing[~existing["date"].isin(dates)]
        combined = pd.concat([existing, rows], ignore_index=True)[UNIFIED_COLS]
    else:
        combined = rows
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["date", "code"])
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    combined.to_csv(path, index=False)
    return len(rows)


def run_day(varieties: list[str], trade_date: str, dry_run: bool = False,
           delay: float = 0.3, use_sina_fallback: bool = True) -> None:
    """use_sina_fallback must stay False for any trade_date that isn't
    "today" -- fetch_dce_sina_day has no trade_date parameter (it's a live
    quote snapshot), so calling it for a past date would silently mislabel
    today's prices as that past date's. run_backfill below handles DCE's
    Sina-covered varieties through fetch_dce_sina_backfill instead, which
    is date-aware."""
    date_str = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    for v in varieties:
        if v not in _FETCHERS:
            print(f"[{v}] no akshare mapping known -- skipped")
            continue
        try:
            rows = fetch_variety(v, trade_date)
        except Exception as e:
            rows = _empty()
            fallback_note = (" -- trying Sina fallback"
                            if use_sina_fallback and v in DCE_SINA_VARIETIES else "")
            print(f"[{v}] {trade_date}: primary fetch failed ({e}){fallback_note}")
        if rows.empty and use_sina_fallback and v in DCE_SINA_VARIETIES:
            try:
                rows = fetch_dce_sina_day(v, date_str)
            except Exception as e:
                print(f"[{v}] {trade_date}: Sina fallback also failed: {e}")
        if rows.empty:
            print(f"[{v}] {trade_date}: no data (non-trading day or not "
                  f"yet published)")
            continue
        if dry_run:
            print(f"[{v}] {trade_date}: {len(rows)} contracts (dry-run, not written)")
        else:
            n = append_day(v, rows)
            print(f"[{v}] {trade_date}: {n} contracts written")
        time.sleep(delay)


def run_backfill(varieties: list[str], start: str, end: str,
                 dry_run: bool = False) -> None:
    dce_sina = [v for v in varieties if v in DCE_SINA_VARIETIES]
    others = [v for v in varieties if v not in DCE_SINA_VARIETIES]

    d0 = datetime.strptime(start, "%Y%m%d").date()
    d1 = datetime.strptime(end, "%Y%m%d").date()
    d = d0
    while d <= d1:
        if d.weekday() < 5:  # Mon-Fri; akshare's own calendar check catches holidays
            print(f"=== {d.isoformat()} ===")
            run_day(others, d.strftime("%Y%m%d"), dry_run=dry_run,
                   use_sina_fallback=False)
        d += timedelta(days=1)

    for v in dce_sina:
        print(f"=== {v} (Sina, {start}-{end}) ===")
        rows = fetch_dce_sina_backfill(v, start, end)
        if rows.empty:
            print(f"[{v}] no rows in range")
            continue
        n_dates = rows["date"].nunique()
        if dry_run:
            print(f"[{v}] {len(rows)} rows across {n_dates} dates (dry-run)")
        else:
            n = append_day(v, rows)
            print(f"[{v}] {n} rows across {n_dates} dates written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--varieties", default=",".join(sorted(ALL_VARIETIES)))
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                    help="YYYYMMDD YYYYMMDD, inclusive")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    varieties = args.varieties.split(",")
    if args.backfill:
        run_backfill(varieties, args.backfill[0], args.backfill[1],
                    dry_run=args.dry_run)
    else:
        run_day(varieties, date.today().strftime("%Y%m%d"), dry_run=args.dry_run)
