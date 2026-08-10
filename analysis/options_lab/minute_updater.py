"""Live per-minute option quote fetcher: SSE ETF options + CFFEX index options.

Foundation for a real-time intraday alerting system (the alerting logic
itself -- thresholds, push wiring -- is a separate follow-up; this module
only builds the data feed it would run on). `updater.py` already fetches a
full option chain once a day after close; this module instead polls live
quotes roughly once a minute throughout both trading sessions and appends
them to per-underlying, per-day CSVs.

Deliberately stores BASIC data only (price/spot/strike/time-to-expiry/C-P),
not precomputed IV/Greeks: SSE's official risk-indicator feed
(option_risk_indicator_sse, what updater.py uses for daily Greeks) is
EOD-only, and Sina's own live Greeks feed (CON_SO_) was already found wrong
for the call side of 3 of 5 SSE underlyings -- see updater.py's docstring --
so there is no trustworthy *live* Greeks source to poll here. IV/Greeks get
computed on demand from the logged basics instead, via bs_iv.py (see
iv_for_row() below for the pattern) -- same convention updater.py itself
already uses for the SZSE underlyings that have no official feed either.

Sources (free, live, no key required):
  * sina hq.sinajs.cn CON_OP_{code}   -- SSE ETF option T-quotes (batched):
                                         bid/ask/last/OI/strike. Same field
                                         layout updater.py's _hq_batch
                                         already parses (partially) --
                                         reused here via `import updater`,
                                         reading two more fields (bid/ask)
                                         out of the same response instead of
                                         making extra calls.
  * sina hq.sinajs.cn sh{code}        -- ETF spot price (updater.etf_quote)
                                         and index spot price (000300/300852/
                                         000016 all resolve on the "sh"
                                         prefix -- verified live 2026-08-10,
                                         sz000852 resolves to an unrelated
                                         stock, not the CSI 1000 index).
  * akshare option_cffex_{hs300,zz1000,sz50}_spot_sina -- CFFEX index-option
    live quotes. One call per contract-month returns the WHOLE strike chain
    (call+put columns side by side) for that month -- confirmed live
    2026-08-10. No trade-volume column in this feed (only bid/ask size and
    OI) -- `volume` is left blank for CFFEX rows, same "blank when the
    source doesn't have it" convention as everywhere else in this project.
  * akshare option_cffex_{hs300,zz1000,sz50}_list_sina -- CFFEX contract
    month discovery, resolved once at startup (contracts don't change
    intraday).

CFFEX expiry dates: Sina's StockOptionService.getRemainderDay (what
updater.expiry_day uses for SSE ETF options) does not support CFFEX's
`exchange=cffex` cate values (returns expireDay: null -- checked live
2026-08-10). Falls back to the documented CFFEX convention (3rd Friday of
the contract month) via _third_friday() -- a formula, not exchange-verified,
same caveat updater.py itself already carries for _fourth_wednesday().

Usage:
    python minute_updater.py                  # run the live polling loop
    python minute_updater.py --dry-run --once  # one snapshot cycle, print only
    python minute_updater.py --etf-underlyings 510050 --index-underlyings ""
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as dtime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

import bs_iv
import data_store as ds
import updater as U

DATA_DIR = ds.DATA_DIR / "minute"

ETF_UNDERLYINGS = ["510050", "510300", "510500", "588000"]  # 588080 dropped: redundant STAR-50 twin of 588000
INDEX_UNDERLYINGS = ["000300", "000852", "000016"]  # 沪深300 / 中证1000 / 上证50

INDEX_SINA_SYM = {"000300": "sh000300", "000852": "sh000852", "000016": "sh000016"}
INDEX_LIST_FN = {"000300": "option_cffex_hs300_list_sina",
                  "000852": "option_cffex_zz1000_list_sina",
                  "000016": "option_cffex_sz50_list_sina"}
INDEX_SPOT_FN = {"000300": "option_cffex_hs300_spot_sina",
                  "000852": "option_cffex_zz1000_spot_sina",
                  "000016": "option_cffex_sz50_spot_sina"}

SESSION_WINDOWS = [(dtime(9, 30), dtime(11, 30)), (dtime(13, 0), dtime(15, 0))]
CLOSE_CUTOFF = dtime(15, 1)  # small buffer past the 15:00 window to catch the last minute
DEFAULT_INTERVAL = 60

COLUMNS = ["timestamp", "code", "cp", "strike", "dte", "spot", "bid", "ask", "last", "volume", "oi"]


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def in_session(now: datetime) -> bool:
    t = now.time()
    return any(lo <= t <= hi for lo, hi in SESSION_WINDOWS)


def past_close(now: datetime) -> bool:
    return now.time() > CLOSE_CUTOFF


@lru_cache(maxsize=1)
def _trading_days() -> set:
    import akshare as ak
    cal = ak.tool_trade_date_hist_sina()
    return set(pd.to_datetime(cal["trade_date"]).dt.date)


def is_trading_day(d: date) -> bool:
    try:
        return d in _trading_days()
    except Exception as e:
        print(f"[WARN] trade calendar fetch failed ({e}); falling back to weekday check")
        return d.weekday() < 5


def _third_friday(year: int, month: int) -> date:
    """CFFEX index-option contract-month expiry convention (formula, not
    exchange-verified -- see module docstring)."""
    from datetime import timedelta
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)  # first Friday
    return d + timedelta(days=14)


def _cp_from_name(name: str) -> str:
    """Same call/put determination updater.py uses (name-text based, not
    which OP_UP_/OP_DOWN_ list the code came from -- option_codes() merges
    both into one flat list)."""
    if "购" in name:
        return "C"
    if "沽" in name:
        return "P"
    return "C" if "C" in name.upper()[:6] else "P"


# ------------------------------------------------------------------ ETF (SSE) contract universe
def resolve_etf_universe(underlyings: list[str]) -> dict[str, dict]:
    """Resolved ONCE at process start: {underlying: {"codes": [(code, yymm), ...],
    "expiry_by_month": {yymm: date}}}. Strike/cp come from the live quote each
    cycle (already in the response), so only the code list + expiry need
    caching -- redoing contract_months/option_codes/expiry_day every minute
    would be wasted network calls for data that doesn't change intraday."""
    universe = {}
    for u in underlyings:
        expiry_by_month, codes = {}, []
        for yymm in U.contract_months(u):
            expiry_by_month[yymm] = U.expiry_day(u, yymm)
            codes += [(c, yymm) for c in U.option_codes(u, yymm)]
        universe[u] = {"codes": codes, "expiry_by_month": expiry_by_month}
        print(f"[{u}] {len(codes)} contracts across {len(expiry_by_month)} expiry months")
    return universe


def snap_etf(underlying: str, meta: dict, dry_run: bool) -> int:
    spot, _ = U.etf_quote(underlying)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    quotes = U._hq_batch([f"CON_OP_{c}" for c, _ in meta["codes"]])
    today = date.today()
    rows = []
    for code, yymm in meta["codes"]:
        q = quotes.get(f"CON_OP_{code}")
        if not q or len(q) < 43:
            continue
        expiry = meta["expiry_by_month"][yymm]
        rows.append({
            "timestamp": ts, "code": code, "cp": _cp_from_name(q[37]),
            "strike": _f(q[7]), "dte": (expiry - today).days, "spot": spot,
            "bid": _f(q[1]), "ask": _f(q[3]), "last": _f(q[2]),
            "volume": _f(q[41]), "oi": _f(q[5]),
        })
    return _emit(underlying, rows, is_index=False, dry_run=dry_run)


# ------------------------------------------------------------------ CFFEX index-option universe
def resolve_index_universe(underlyings: list[str]) -> dict[str, list[str]]:
    """{underlying: [contract-month codes, e.g. "io2608", ...]} -- resolved
    once; option_cffex_*_spot_sina(month) doubles as both discovery (strike/
    contract-identifier list) and live quote, so no separate per-contract
    discovery step is needed the way the ETF path needs one."""
    import akshare as ak
    universe = {}
    for u in underlyings:
        fn = getattr(ak, INDEX_LIST_FN[u])
        d = fn()
        months = [m for v in d.values() for m in v]
        universe[u] = months
        print(f"[{u}] {len(months)} contract months resolved")
    return universe


def index_spot(underlying: str) -> float:
    sym = INDEX_SINA_SYM[underlying]
    got = U._hq_batch([sym])
    v = got.get(sym)
    if not v or len(v) < 4:
        raise RuntimeError(f"sina index quote for {sym} unavailable: {got}")
    return _f(v[3])


def snap_index(underlying: str, months: list[str], dry_run: bool) -> int:
    import akshare as ak
    spot = index_spot(underlying)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fn = getattr(ak, INDEX_SPOT_FN[underlying])
    today = date.today()
    rows = []
    for m in months:
        try:
            chain = fn(symbol=m)
        except Exception as e:
            print(f"[WARN] {underlying} {m} spot fetch failed: {e}")
            continue
        yymm = re.sub(r"[^0-9]", "", m)
        if len(yymm) < 4:
            continue
        expiry = _third_friday(2000 + int(yymm[:2]), int(yymm[2:4]))
        dte = (expiry - today).days
        for _, r in chain.iterrows():
            strike = _f(r["行权价"])
            rows.append({"timestamp": ts, "code": r["看涨合约-标识"], "cp": "C",
                        "strike": strike, "dte": dte, "spot": spot,
                        "bid": _f(r["看涨合约-买价"]), "ask": _f(r["看涨合约-卖价"]),
                        "last": _f(r["看涨合约-最新价"]), "volume": float("nan"),
                        "oi": _f(r["看涨合约-持仓量"])})
            rows.append({"timestamp": ts, "code": r["看跌合约-标识"], "cp": "P",
                        "strike": strike, "dte": dte, "spot": spot,
                        "bid": _f(r["看跌合约-买价"]), "ask": _f(r["看跌合约-卖价"]),
                        "last": _f(r["看跌合约-最新价"]), "volume": float("nan"),
                        "oi": _f(r["看跌合约-持仓量"])})
    return _emit(underlying, rows, is_index=True, dry_run=dry_run)


# ------------------------------------------------------------------ output
def minute_csv_path(underlying: str, is_index: bool, when: date | None = None) -> Path:
    when = when or date.today()
    prefix = "index_option_minute" if is_index else "option_minute"
    return DATA_DIR / f"{prefix}_{underlying}_{when.strftime('%Y%m%d')}.csv"


def _emit(underlying: str, rows: list[dict], is_index: bool, dry_run: bool) -> int:
    if not rows:
        print(f"[{underlying}] no rows this cycle")
        return 0
    df = pd.DataFrame(rows, columns=COLUMNS)
    if dry_run:
        print(f"[{underlying}] {len(df)} rows (dry-run):")
        print(df.head(3).to_string())
    else:
        path = minute_csv_path(underlying, is_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")
    return len(df)


def iv_for_row(row: pd.Series, r: float = bs_iv.RISK_FREE_DEFAULT) -> float:
    """Convenience: back out IV for one stored minute-row via bs_iv.py --
    the basic-data-in/IV-out pattern this module's schema exists for. Prefers
    the bid/ask mid (more robust intraday than a possibly-stale last trade),
    falling back to last if the quote is one-sided or crossed."""
    price = _f(row.get("last"))
    bid, ask = _f(row.get("bid")), _f(row.get("ask"))
    if bid > 0 and ask > 0:
        price = (bid + ask) / 2
    spot, strike, dte = _f(row.get("spot")), _f(row.get("strike")), _f(row.get("dte"))
    if not (price > 0 and spot > 0 and strike > 0 and dte > 0):
        return float("nan")
    return bs_iv.implied_vol(row["cp"], spot, strike, dte / 365.0, price, r=r)


# ------------------------------------------------------------------ main loop
def _safe_snap_etf(u: str, meta: dict, dry_run: bool) -> int:
    try:
        return snap_etf(u, meta, dry_run)
    except Exception as e:
        print(f"[WARN] {u} ETF snapshot failed: {e}")
        return 0


def _safe_snap_index(u: str, months: list[str], dry_run: bool) -> int:
    try:
        return snap_index(u, months, dry_run)
    except Exception as e:
        print(f"[WARN] {u} index snapshot failed: {e}")
        return 0


def run(etf_underlyings: list[str], index_underlyings: list[str],
        interval: int = DEFAULT_INTERVAL, dry_run: bool = False, once: bool = False) -> None:
    today = date.today()
    if not once and not is_trading_day(today):
        print(f"{today} is not a trading day -- exiting.")
        return

    print("Resolving contract universe...")
    etf_universe = resolve_etf_universe(etf_underlyings) if etf_underlyings else {}
    index_universe = resolve_index_universe(index_underlyings) if index_underlyings else {}

    print(f"Entering polling loop (interval={interval}s, dry_run={dry_run})...")
    while True:
        now = datetime.now()
        if now.date() != today:
            print("date rolled over -- exiting.")
            break
        if not once and past_close(now):
            print("past close -- exiting.")
            break
        if not once and not in_session(now):
            time.sleep(5)
            continue

        cycle_start = time.monotonic()
        n_underlyings = len(etf_underlyings) + len(index_underlyings)
        # Each underlying's fetch is a separate endpoint writing a separate
        # file -- no shared state -- so they run concurrently (I/O-bound:
        # network latency, not CPU, dominates; threads are enough, no need
        # for multiprocessing). Sequential summing of all 7 underlyings
        # measured ~61s, over the 60s target interval; concurrent measured
        # ~19s (bounded by the single slowest underlying instead of the sum).
        n = 0
        with ThreadPoolExecutor(max_workers=max(n_underlyings, 1)) as ex:
            futures = ([ex.submit(_safe_snap_etf, u, etf_universe[u], dry_run) for u in etf_underlyings]
                      + [ex.submit(_safe_snap_index, u, index_universe[u], dry_run) for u in index_underlyings])
            for fut in as_completed(futures):
                n += fut.result()
        print(f"{now.strftime('%H:%M:%S')} cycle done: {n} rows"
              f"{' (dry-run)' if dry_run else ''}")

        if once:
            break
        elapsed = time.monotonic() - cycle_start
        time.sleep(max(1.0, interval - elapsed))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--etf-underlyings", default=",".join(ETF_UNDERLYINGS))
    ap.add_argument("--index-underlyings", default=",".join(INDEX_UNDERLYINGS))
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true", help="fetch a single snapshot cycle and exit (testing)")
    args = ap.parse_args()
    etf_list = [u for u in args.etf_underlyings.split(",") if u]
    idx_list = [u for u in args.index_underlyings.split(",") if u]
    try:
        run(etf_list, idx_list, interval=args.interval, dry_run=args.dry_run, once=args.once)
    except requests.RequestException as e:
        print(f"network error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("interrupted, exiting.")
