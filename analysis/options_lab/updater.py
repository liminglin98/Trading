"""Daily EOD updater: appends today's option chain to Data/options CSVs.

Sources (free, run from your own machine — the sina/akshare endpoints need
a mainland-reachable network):
  * sina hq.sinajs.cn         — option T-quotes (batched): price/volume/OI/
                                 strike, everything except Greeks/IV
  * sina openapi              — contract months + expiry day per month
  * akshare option_risk_indicator_sse (query.sse.com.cn) — official SSE
                                 Greeks + IV, one call/run covering all 5
                                 underlyings. Replaced Sina's own CON_SO_
                                 Greeks feed 2026-07-28: cross-validated and
                                 found Sina's own call-side IV/Greeks for
                                 510500/588000/588080 were simply wrong
                                 (puts fine) -- see reference_data_sources
                                 memory / Data/买断日更/README.md for the
                                 full investigation. Empty if the exchange
                                 hasn't published today's file yet (normal
                                 right after close) -- Greeks/IV land as
                                 blank for that day. append_chain() is
                                 append-only and never revisits an
                                 already-recorded date, so a blank day would
                                 stay blank forever without help -- run()
                                 below calls backfill_missing_greeks() every
                                 run (cheap: only re-fetches dates it finds
                                 still blank) to self-heal once SSE catches
                                 up, so a plain rerun (button or scheduled
                                 script) is normally enough; no manual
                                 backfill script needed for this case anymore
                                 (2026-08-05, generalized from a one-off
                                 fix -- see
                                 project_options_lab_architecture memory).
  * tencent ifzq.gtimg.cn     — ETF daily OHLC (for candlestick charts)
  * eastmoney push2           — ETF EOD cumulative fund flow (main/large/mid/
                                 small net, yuan) -> Data/options/fund_flow_daily.csv
  * eastmoney push2ex         — market-wide limit-up/limit-down board counts,
                                 a risk-on/off breadth gauge ->
                                 Data/options/market_sentiment_daily.csv
    Eastmoney has anti-bot throttling (see _em_get); we only make a handful
    of calls per run so it is a non-issue in practice.

Field notes vs the historical schema:
  * 结算价 is not published by sina -> we store the close price there.
  * 前结算价 = sina prev_close (for options sina reports prev settle here).
  * 上市日 is carried over from the existing CSV; new contracts get the
    first date we see them.
  * 成交额(万元) = sina amount / 1e4 (sina reports yuan). Verify once with
    --dry-run against the exchange daily stats.

Covers ds.UNDERLYINGS (50ETF/300ETF, also position-tracked in the Positions/
Backtest tabs) plus ds.EXT_LIVE_UNDERLYINGS (500ETF/科创50ETF/科创板50ETF,
Data-tab-only for now -- Sina's live quote feed covers these the same way,
verified 2026-07-24, but Positions/Backtest haven't been validated against
them). Does NOT cover any SZSE underlying (创业板ETF 159915/深证100ETF 159901/
300ETF(深) 159919/500ETF(深) 159922) -- Sina has zero SZSE option coverage
(checked repeatedly) and neither does any other free source found so far;
see Data/买断日更/materialize_szse_full_csv.py for their (vendor-snapshot +
Black-Scholes-derived, unvalidated) data path instead.

Usage:
    python updater.py                 # append all of the above + ETF OHLC
    python updater.py --dry-run       # fetch and show, do not write
    python updater.py --underlyings 510050
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd
import requests

import data_store as ds

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
SINA_HDR = {"Referer": "https://stock.finance.sina.com.cn/", "User-Agent": UA}
CATE = {"510050": "50ETF", "510300": "300ETF", "510500": "500ETF",
        "588000": "科创50", "588080": "科创板50"}
BATCH = 40  # codes per hq.sinajs.cn request


# ------------------------------------------------------------------ sina raw
def _hq_batch(params: list[str]) -> dict[str, list[str]]:
    """Fetch many hq.sinajs.cn symbols in one request -> {symbol: fields}."""
    out: dict[str, list[str]] = {}
    for i in range(0, len(params), BATCH):
        chunk = params[i:i + BATCH]
        r = requests.get("https://hq.sinajs.cn/list=" + ",".join(chunk),
                         headers=SINA_HDR, timeout=15)
        r.encoding = "gbk"
        for line in r.text.splitlines():
            if '="' not in line:
                continue
            key = line.split("=")[0].replace("var hq_str_", "").strip()
            out[key] = line.split('"')[1].split(",")
        time.sleep(0.15)
    return out


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def contract_months(underlying: str) -> list[str]:
    url = ("https://stock.finance.sina.com.cn/futures/api/openapi.php/"
           f"StockOptionService.getStockName?exchange=null&cate={CATE[underlying]}")
    js = requests.get(url, headers=SINA_HDR, timeout=15).json()
    data = (js.get("result") or {}).get("data") if isinstance(js, dict) else None
    if not isinstance(data, dict) or "contractMonth" not in data:
        raise RuntimeError(f"sina getStockName unexpected response: {str(js)[:300]}")
    return sorted({m.replace("-", "")[2:] for m in data["contractMonth"]})  # YYMM


def expiry_day(underlying: str, yymm: str) -> date:
    url = ("https://stock.finance.sina.com.cn/futures/api/openapi.php/"
           f"StockOptionService.getRemainderDay?date={yymm}&exchange=null"
           f"&cate={CATE[underlying]}")
    try:
        d = requests.get(url, headers=SINA_HDR, timeout=15).json()[
            "result"]["data"]["expireDay"]
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return _fourth_wednesday(2000 + int(yymm[:2]), int(yymm[2:]))


def _fourth_wednesday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(2 - d.weekday()) % 7)  # first Wednesday
    return d + timedelta(days=21)


def option_codes(underlying: str, yymm: str) -> list[str]:
    codes: list[str] = []
    for flag in ("OP_UP_", "OP_DOWN_"):
        got = _hq_batch([f"{flag}{underlying}{yymm}"])
        for fields in got.values():
            codes += [c.replace("CON_OP_", "") for c in fields
                      if c.startswith("CON_OP_")]
    return codes


def etf_quote(underlying: str) -> tuple[float, date]:
    """ETF last price and the exchange trade date."""
    got = _hq_batch([f"sh{underlying}"])
    v = got.get(f"sh{underlying}")
    if not v or len(v) < 31:
        raise RuntimeError(f"sina hq quote for sh{underlying} unavailable "
                           f"(blocked or empty): {got}")
    return _f(v[3]), datetime.strptime(v[30], "%Y-%m-%d").date()


@lru_cache(maxsize=8)
def fetch_official_sse_greeks(trade_date_iso: str) -> dict[str, dict]:
    """{code: {'delta','gamma','theta','vega','rho','iv'}} from akshare's
    option_risk_indicator_sse -- official SSE data, covers all 5 SSE
    underlyings in one call (cached per date so fetch_chain_snapshot's
    per-underlying loop only hits the network once per run). Returns {} if
    the exchange hasn't published this date yet -- callers should leave
    Greeks/IV blank rather than guess, same convention as everywhere else
    in this project."""
    import akshare as ak
    ymd = trade_date_iso.replace("-", "")
    try:
        df = ak.option_risk_indicator_sse(date=ymd)
    except Exception as e:
        print(f"  [WARN] option_risk_indicator_sse({ymd}) failed "
              f"(not published yet?): {e}")
        return {}
    if df.empty:
        return {}
    return {
        str(r.SECURITY_ID): {
            "delta": r.DELTA_VALUE, "gamma": r.GAMMA_VALUE,
            "theta": r.THETA_VALUE, "vega": r.VEGA_VALUE,
            "rho": r.RHO_VALUE, "iv": r.IMPLC_VOLATLTY,
        }
        for r in df.itertuples()
    }


GREEKS_COLS = {"delta": ds.C_DELTA, "gamma": ds.C_GAMMA, "theta": ds.C_THETA,
               "vega": ds.C_VEGA, "rho": ds.C_RHO, "iv": ds.C_IV}


def backfill_missing_greeks(underlyings: list[str], lookback_days: int = 10) -> None:
    """General-purpose fix for the gap fetch_official_sse_greeks() warns
    about above: append_chain() never revisits an already-recorded date, so
    a date fetched before SSE published its risk-indicator file stays blank
    forever unless something goes back for it. Scans each underlying's last
    `lookback_days` trading dates for blank IV and re-fetches just those
    dates (fetch_official_sse_greeks is date-keyed and lru_cached, so dates
    shared across underlyings only hit the network once). Safe to call every
    run: no-ops instantly once nothing is blank, and dates SSE still hasn't
    published simply get skipped again next time. Only meaningful for the
    5 underlyings this feed covers (ds.UNDERLYINGS + ds.EXT_LIVE_UNDERLYINGS);
    passing others is harmless (just finds no blanks, since only-real-IV
    underlyings can ever have a blank IV column here)."""
    for u in underlyings:
        path = ds.chain_path(u)
        if not path.exists():
            continue
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype={ds.C_CODE: str})
        dates = sorted(pd.to_datetime(raw[ds.C_DATE]).unique())[-lookback_days:]
        blank_dates = [d for d in dates
                      if raw.loc[pd.to_datetime(raw[ds.C_DATE]) == d, ds.C_IV].isna().all()]
        if not blank_dates:
            continue
        changed = False
        for d in blank_dates:
            d_iso = pd.Timestamp(d).strftime("%Y-%m-%d")
            official = fetch_official_sse_greeks(d_iso)
            if not official:
                print(f"[{u}] {d_iso}: still not published by SSE, will retry next run")
                continue
            mask = pd.to_datetime(raw[ds.C_DATE]) == d
            n_filled = 0
            for idx in raw[mask].index:
                info = official.get(str(raw.at[idx, ds.C_CODE]))
                if info is None:
                    continue
                for key, col in GREEKS_COLS.items():
                    raw.at[idx, col] = info[key]
                n_filled += 1
            print(f"[{u}] {d_iso}: backfilled {n_filled}/{mask.sum()} contracts' Greeks/IV")
            changed = True
        if changed:
            raw.to_csv(path, index=False, encoding="utf-8-sig")


# ------------------------------------------------------------------ build
def fetch_chain_snapshot(underlying: str) -> pd.DataFrame:
    spot, trade_date = etf_quote(underlying)
    print(f"[{underlying}] spot={spot} trade_date={trade_date}")

    official = fetch_official_sse_greeks(trade_date.isoformat())
    if not official:
        print(f"  [WARN] no official SSE Greeks/IV for {trade_date} yet -- "
              f"Delta/Gamma/Theta/Vega/Rho/IV will be blank for this run "
              f"(rerun later, or backfill with "
              f"Data/买断日更/replace_greeks_from_sse_official.py)")

    listing_dates = _known_listing_dates(underlying)
    rows = []
    for yymm in contract_months(underlying):
        codes = option_codes(underlying, yymm)
        if not codes:
            continue
        exp = expiry_day(underlying, yymm)
        quotes = _hq_batch([f"CON_OP_{c}" for c in codes])
        for c in codes:
            q = quotes.get(f"CON_OP_{c}", [])
            if len(q) < 43:
                continue
            name = q[37]
            cp = "C" if ("购" in name or "C" in name.upper()[:6]) else "P"
            cp = "C" if "购" in name else ("P" if "沽" in name else cp)
            strike, last = _f(q[7]), _f(q[2])
            prev = _f(q[8])
            dte = (exp - trade_date).days
            g = official.get(c, {})
            rows.append({
                ds.C_CODE: c, ds.C_NAME: name, ds.C_MKT: "XSHG", ds.C_CP: cp,
                ds.C_STRIKE: strike,
                ds.C_LIST: listing_dates.get(c, trade_date.isoformat()),
                ds.C_EXPIRY: exp.isoformat(), ds.C_DTE: dte,
                ds.C_DATE: trade_date.isoformat(),
                ds.C_PREV_CLOSE: prev, ds.C_PREV_SETTLE: prev,
                ds.C_OPEN: _f(q[9]), ds.C_HIGH: _f(q[39]), ds.C_LOW: _f(q[40]),
                ds.C_CLOSE: last, ds.C_SETTLE: last,
                ds.C_CHG: round(last - prev, 4) if prev == prev else "",
                ds.C_PCT: _f(q[6]),
                ds.C_VOL: _f(q[41]), ds.C_AMT: round(_f(q[42]) / 1e4, 2),
                ds.C_OI: _f(q[5]),
                ds.C_DELTA: g.get("delta", float("nan")),
                ds.C_THETA: g.get("theta", float("nan")),
                ds.C_GAMMA: g.get("gamma", float("nan")),
                ds.C_VEGA: g.get("vega", float("nan")),
                ds.C_RHO: g.get("rho", float("nan")),
                ds.C_IV: g.get("iv", float("nan")),
                ds.C_UND: underlying, ds.C_UND_CLOSE: spot,
            })
    return pd.DataFrame(rows, columns=ds.COLUMNS)


def _known_listing_dates(underlying: str) -> dict[str, str]:
    try:
        df = pd.read_csv(ds.chain_path(underlying), encoding="utf-8-sig",
                         usecols=[ds.C_CODE, ds.C_LIST], dtype={ds.C_CODE: str})
        return df.drop_duplicates(ds.C_CODE, keep="last").set_index(
            ds.C_CODE)[ds.C_LIST].to_dict()
    except Exception:
        return {}


# ------------------------------------------------------------------ ETF OHLC
def _tencent_kline(underlying: str, end: str = "", count: int = 640
                   ) -> pd.DataFrame:
    """One page of daily OHLC. tencent rejects count > ~800 ('param error')."""
    url = ("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
           f"param=sh{underlying},day,,{end},{count},qfq")
    js = requests.get(url, headers={"User-Agent": UA}, timeout=20).json()
    data = js.get("data") if isinstance(js, dict) else None
    node = data.get(f"sh{underlying}") if isinstance(data, dict) else None
    if not isinstance(node, dict):
        raise RuntimeError(f"tencent kline unexpected response: {str(js)[:300]}")
    kl = node.get("qfqday") or node.get("day") or []
    df = pd.DataFrame([k[:6] for k in kl],
                      columns=["date", "open", "close", "high", "low", "volume"])
    for col in ("open", "close", "high", "low", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df


def update_etf_ohlc(underlying: str, dry_run: bool = False,
                    history_start: str = "2015-01-01") -> int:
    """Merge daily OHLC into etf_daily_{code}.csv, paging back as needed."""
    old = ds.load_etf_ohlc(underlying)
    have_min = old["date"].min() if old is not None and len(old) else None
    frames = [old] if old is not None else []
    end = ""
    for _ in range(12):  # safety cap (~12*640 bars)
        chunk = _tencent_kline(underlying, end=end)
        if chunk.empty:
            break
        frames.append(chunk)
        earliest = chunk["date"].min()
        if earliest <= pd.Timestamp(history_start):
            break
        if have_min is not None and have_min <= earliest:
            break  # older history already on file
        end = (earliest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.3)
    df = (pd.concat(frames).drop_duplicates("date", keep="last")
          .sort_values("date"))
    if not dry_run:
        df.to_csv(ds.etf_path(underlying), index=False)
    return len(df)


# ------------------------------------------------------------------ eastmoney (fund flow + limit board)
# Eastmoney rate-limits aggressive callers (429/403, IP-level). We only make
# a handful of calls per run, but still throttle + reuse the session so a
# manual re-run right after a scheduled one never looks like a burst.
EM_SESSION = requests.Session()
EM_SESSION.headers.update({"User-Agent": UA})
EM_MIN_INTERVAL = 1.0
_em_last_call = [0.0]


def _em_get(url: str, params: dict | None = None, headers: dict | None = None,
           timeout: int = 15):
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait)
    try:
        return EM_SESSION.get(url, params=params, headers=headers, timeout=timeout)
    finally:
        _em_last_call[0] = time.time()


def eastmoney_fund_flow_minute(code: str) -> list[dict]:
    """Minute-level cumulative ETF fund flow for today's session (yuan)."""
    # SSE-listed codes (ETFs 5xxxxx, A-shares 6xxxxx/9xxxxx) -> market 1;
    # SZSE-listed (ETFs/A-shares 0/1/2/3xxxxx) -> market 0. Both our ETFs
    # (510050/510300) are SSE-listed, unlike the stock-code convention this
    # was adapted from where only 6/9-prefixed codes are SSE.
    secid = f"1.{code}" if code.startswith(("5", "6", "9")) else f"0.{code}"
    url = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"
    params = {"secid": secid, "klt": 1, "fields1": "f1,f2,f3,f7",
              "fields2": "f51,f52,f53,f54,f55,f56,f57"}
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/",
               "Origin": "https://quote.eastmoney.com"}
    try:
        d = _em_get(url, params=params, headers=headers, timeout=10).json()
    except Exception as e:
        print(f"[WARN] eastmoney fund flow request failed for {code}: {e}")
        return []
    rows = []
    for line in (d.get("data") or {}).get("klines", []):
        parts = line.split(",")
        if len(parts) >= 6:
            rows.append({"time": parts[0], "main_net": _f(parts[1]),
                         "small_net": _f(parts[2]), "mid_net": _f(parts[3]),
                         "large_net": _f(parts[4]), "super_net": _f(parts[5])})
    return rows


def fetch_fund_flow_eod(underlying: str, trade_date: str) -> dict | None:
    """Day's closing cumulative net fund-flow snapshot for one ETF (yuan)."""
    rows = eastmoney_fund_flow_minute(underlying)
    if not rows:
        return None
    last = rows[-1]
    return {"date": trade_date, "underlying": underlying, "main_net": last["main_net"],
            "small_net": last["small_net"], "mid_net": last["mid_net"],
            "large_net": last["large_net"], "super_net": last["super_net"]}


ZTB_UT = "7eea3edcaed734bea9cbfc24409ed989"


def _em_zt_api(endpoint: str, sort: str, ymd: str) -> list[dict]:
    """endpoint: getTopicZTPool / getTopicZBPool / getTopicDTPool."""
    url = f"https://push2ex.eastmoney.com/{endpoint}"
    params = {"ut": ZTB_UT, "dpt": "wz.ztzt", "Pageindex": 0, "pagesize": 10000,
              "sort": sort, "date": ymd}
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = _em_get(url, params=params, headers=headers, timeout=10)
        return (r.json().get("data") or {}).get("pool") or []
    except Exception as e:
        print(f"[WARN] limit board {endpoint} request failed: {e}")
        return []


def fetch_market_sentiment(trade_date: str) -> dict | None:
    """Market-wide limit-up/limit-down breadth gauge for one trading day.
    None on non-trading days or if eastmoney returns nothing for all 3 pools."""
    ymd = trade_date.replace("-", "")
    zt = _em_zt_api("getTopicZTPool", "fbt:asc", ymd)
    zb = _em_zt_api("getTopicZBPool", "fbt:asc", ymd)
    dt = _em_zt_api("getTopicDTPool", "fund:asc", ymd)
    if not zt and not zb and not dt:
        return None
    zt_n, zb_n = len(zt), len(zb)
    return {"date": trade_date, "zt_count": zt_n, "zb_count": zb_n, "dt_count": len(dt),
            "break_rate": round(zb_n / (zt_n + zb_n) * 100, 1) if (zt_n + zb_n) else 0.0,
            "max_height": max((p["lbc"] for p in zt), default=0)}


# ------------------------------------------------------------------ main
def run(underlyings: list[str], dry_run: bool = False) -> None:
    sentiment_done = False
    for u in underlyings:
        snap = fetch_chain_snapshot(u)
        if snap.empty:
            print(f"[{u}] no rows fetched — market feed unavailable?")
            continue
        trade_date = snap[ds.C_DATE].iloc[0]
        print(f"[{u}] fetched {len(snap)} contracts for {trade_date}")
        if dry_run:
            print(snap.head(8).to_string())
            print("...dry-run: nothing written.")
        else:
            n = ds.append_chain(u, snap)
            print(f"[{u}] appended {n} rows "
                  f"({'skipped: date already recorded' if n == 0 else 'ok'})")
        n_ohlc = update_etf_ohlc(u, dry_run=dry_run)
        print(f"[{u}] ETF OHLC rows on file: {n_ohlc}")

        flow = fetch_fund_flow_eod(u, trade_date)
        if not flow:
            print(f"[{u}] fund flow unavailable — skipped.")
        elif dry_run:
            print(f"[{u}] fund flow (dry-run): {flow}")
        else:
            n = ds.append_fund_flow(pd.DataFrame([flow]))
            print(f"[{u}] fund flow appended "
                  f"({'ok' if n else 'skipped: already recorded'})")

        # market-wide, not per-underlying -- only needs to run once
        if not sentiment_done:
            sentiment = fetch_market_sentiment(trade_date)
            if not sentiment:
                print("market sentiment unavailable — skipped.")
            elif dry_run:
                print(f"market sentiment (dry-run): {sentiment}")
            else:
                n = ds.append_sentiment(sentiment)
                print(f"market sentiment appended "
                      f"({'ok' if n else 'skipped: already recorded'})")
            sentiment_done = True

    if not dry_run:
        sse_underlyings = [u for u in underlyings
                          if u in ds.UNDERLYINGS or u in ds.EXT_LIVE_UNDERLYINGS]
        if sse_underlyings:
            backfill_missing_greeks(sse_underlyings)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlyings", default=",".join(ds.UNDERLYINGS) + "," +
                    ",".join(ds.EXT_LIVE_UNDERLYINGS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    try:
        run(args.underlyings.split(","), dry_run=args.dry_run)
    except requests.RequestException as e:
        print(f"network error: {e}")
        sys.exit(1)
