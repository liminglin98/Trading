# -*- coding: utf-8 -*-
"""index_price_updater.py -- daily OHLCV cache for the three CFFEX index-option
underlyings: 000300 (沪深300/IO), 000852 (中证1000/MO), 000016 (上证50/HO).

Why this exists: the realized-vol signals in this lab
(`long_straddle_rv_contraction_backtest._trailing_rv_series` and its twin in
`iv_low_long_straddle_backtest`) need the underlying's own close series, and
they get it from `data_store.load_etf_ohlc` -> `Data/options/etf_daily_*.csv`.
No such file exists for an index -- so `rv_pct` and `vrp` raised
FileNotFoundError on IO/MO/HO and those signals were simply unavailable there.
This writes the missing series so `data_store.load_underlying_ohlc` can serve
them.

Output format is byte-compatible with the existing `Data/index/
index_daily_000001.csv` (SSE Composite, read by ~20 macro_model studies via
`index_trend_signal.INDEX_CSV`): columns `date,open,close,high,low,volume`,
one row per trading day, ascending. Note that is NOT akshare's column order
(it returns date,open,high,low,close,volume) -- the reorder is deliberate,
not incidental.

Why the real index rather than the option chain's own parity forward:
`consolidated_store`'s `C_UND_CLOSE` is a put-call-parity forward and is the
right spot for picking ATM strikes (it is self-consistent with the raw
strikes -- see that module's docstring on why the 前复权-adjusted ETF caches
are not). But it carries bid-ask noise, and measured against the real index
it inflates 20-day realized vol by 3.5-4.2% relative on all three
underlyings (IO 0.1758 -> 0.1819, MO 0.2270 -> 0.2355, HO 0.1462 -> 0.1523;
level correlation 0.9995+, mean absolute level gap 0.25-0.43%). Since `vrp =
IV - RV` is gated at thresholds of 0.03-0.06, a systematic ~0.6 vol-point
distortion in RV is not acceptable in the signal itself. Strike selection
still uses the parity forward; only RV uses this file.

Source: `akshare.stock_zh_index_daily(symbol="sh000300")` etc. -- sina's
realstock history endpoint, the same one dashboard_data.py's fetchers use.
History runs far earlier than each option's listing date (000300 from
2002-01, 000852 from 2014-10, 000016 from 2004-01, vs. options listed
2019-12 / 2022-07 / 2022-12), so RV's trailing window is fully warm before
the first tradeable option day -- no sample is lost to warm-up.

Rerunnable: merges on date and keeps the union, so a later run just extends
the tail. Network is only touched here, never inside a backtest.

Usage:
    python index_price_updater.py                    # all three
    python index_price_updater.py --underlying 000852
    python index_price_updater.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO / "Data" / "index"

# short code -> sina symbol, same mapping minute_updater.INDEX_SINA_SYM uses
SINA_SYMBOLS = {"000300": "sh000300", "000852": "sh000852", "000016": "sh000016"}
NAMES = {"000300": "沪深300 (IO)", "000852": "中证1000 (MO)", "000016": "上证50 (HO)"}

COLS = ["date", "open", "close", "high", "low", "volume"]  # index_daily_000001.csv order


def index_csv_path(underlying: str) -> Path:
    return INDEX_DIR / f"index_daily_{underlying}.csv"


def fetch(underlying: str) -> pd.DataFrame:
    """Full history from sina, normalized to this repo's index CSV schema."""
    import akshare as ak
    raw = ak.stock_zh_index_daily(symbol=SINA_SYMBOLS[underlying])
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "close", "high", "low", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[COLS].dropna(subset=["date", "close"]).sort_values("date")
    return df.drop_duplicates("date", keep="last").reset_index(drop=True)


def merge_into_csv(underlying: str, fresh: pd.DataFrame, dry_run: bool = False) -> dict:
    """Union `fresh` with whatever is already cached, newest row wins on a
    date collision. Returns a small stats dict for the caller to print."""
    path = index_csv_path(underlying)
    if path.exists():
        old = pd.read_csv(path)
        old["date"] = pd.to_datetime(old["date"])
        merged = pd.concat([old, fresh], ignore_index=True)
        merged = merged.drop_duplicates("date", keep="last").sort_values("date")
    else:
        old, merged = None, fresh

    merged = merged[COLS].reset_index(drop=True)
    stats = {"rows": len(merged), "added": len(merged) - (len(old) if old is not None else 0),
             "start": merged["date"].min(), "end": merged["date"].max(), "path": path}
    if not dry_run:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        out = merged.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
    return stats


def verify_against_chain(underlying: str) -> dict:
    """Cross-check the fetched index against the option chain's own parity
    forward. These are different quantities (index vs. forward), so they will
    not match exactly -- but a wrong symbol would show up immediately as a
    broken correlation or a level gap of several percent."""
    import consolidated_store as cs
    import data_store as ds
    chain = cs.load_chain(underlying).dropna(subset=[ds.C_UND_CLOSE])
    chain = chain[chain[ds.C_DTE] >= 0].sort_values([ds.C_DATE, ds.C_DTE])
    fwd = chain.groupby(ds.C_DATE)[ds.C_UND_CLOSE].first()

    idx = pd.read_csv(index_csv_path(underlying))
    idx["date"] = pd.to_datetime(idx["date"])
    px = idx.set_index("date")["close"]

    j = pd.concat({"idx": px, "fwd": fwd}, axis=1).dropna()
    gap = (j["fwd"] - j["idx"]).abs()
    return {"n": len(j), "corr": float(j["idx"].corr(j["fwd"])),
            "mean_abs_gap_pct": float(100 * gap.mean() / j["idx"].mean())}


def main(underlyings: list[str], dry_run: bool, do_verify: bool) -> None:
    for u in underlyings:
        fresh = fetch(u)
        st = merge_into_csv(u, fresh, dry_run=dry_run)
        tag = " (dry-run, not written)" if dry_run else ""
        print(f"{NAMES[u]:<18} {st['rows']:>6,} rows  "
              f"{st['start'].date()} -> {st['end'].date()}  "
              f"(+{st['added']}){tag}")
        if do_verify and not dry_run:
            v = verify_against_chain(u)
            ok = v["corr"] >= 0.999 and v["mean_abs_gap_pct"] < 0.5
            print(f"{'':18} vs chain parity forward: n={v['n']:,}  corr={v['corr']:.5f}  "
                  f"mean|gap|={v['mean_abs_gap_pct']:.2f}%  {'OK' if ok else 'CHECK SYMBOL'}")
        print(f"{'':18} -> {st['path']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default=None, choices=list(SINA_SYMBOLS),
                    help="default: all three")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--no-verify", action="store_true", dest="no_verify",
                    help="skip the cross-check against the option chain's parity forward")
    args = ap.parse_args()
    us = [args.underlying] if args.underlying else list(SINA_SYMBOLS)
    main(us, args.dry_run, not args.no_verify)
