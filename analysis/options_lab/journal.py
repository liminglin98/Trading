"""Snapshot the rows this machine just fetched, so a bad sync can't lose them.

After each update run, copies ONLY the newest trading day's rows out of the
main CSVs into:

    <journal>/<YYYY-MM-DD>/options/option_daily_510050.csv
    <journal>/<YYYY-MM-DD>/options/option_daily_510300.csv
    <journal>/<YYYY-MM-DD>/options/etf_daily_510050.csv
    ...

That layout is deliberately identical to Data/, so merge_data.py can consume a
journal day directly as its --other side:

    python merge_data.py --other "..\\..\\Data\\journal\\2026-07-20" --write

Roughly 200 KB/day rather than the 125 MB a full Data/ snapshot would cost.

Usage:
    python journal.py                       # journal the latest date on file
    python journal.py --date 2026-07-20     # journal a specific date
    python journal.py --journal "E:/backup" # write somewhere outside the sync tree
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import data_store as ds

DEFAULT_JOURNAL = ds.DATA_DIR.parent / "journal"

# rel path under Data/ -> (date column, read kwargs)
SOURCES: dict[str, tuple[str, dict]] = {
    "options/option_daily_510050.csv": (ds.C_DATE,
                                        {"encoding": "utf-8-sig",
                                         "dtype": {ds.C_CODE: str, ds.C_UND: str}}),
    "options/option_daily_510300.csv": (ds.C_DATE,
                                        {"encoding": "utf-8-sig",
                                         "dtype": {ds.C_CODE: str, ds.C_UND: str}}),
    "options/etf_daily_510050.csv": ("date", {}),
    "options/etf_daily_510300.csv": ("date", {}),
    "options/fund_flow_daily.csv": ("date", {"dtype": {"underlying": str}}),
    "options/market_sentiment_daily.csv": ("date", {}),
}


def _norm(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.strftime("%Y-%m-%d")


def latest_date(data_root: Path) -> str | None:
    """Newest date across the option chains - what the last run just wrote."""
    best = None
    for rel in ("options/option_daily_510050.csv", "options/option_daily_510300.csv"):
        p = data_root / rel
        if not p.exists():
            continue
        col = pd.read_csv(p, encoding="utf-8-sig", usecols=[ds.C_DATE])[ds.C_DATE]
        d = _norm(col).max()
        if d and (best is None or d > best):
            best = d
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ds.DATA_DIR.parent))
    ap.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD; default = newest date on file")
    args = ap.parse_args()

    data_root = Path(args.data)
    day = args.date or latest_date(data_root)
    if not day:
        print("[journal] no option chain data found - nothing to journal")
        return

    out_root = Path(args.journal) / day
    total = 0
    for rel, (date_col, read_kw) in SOURCES.items():
        src = data_root / rel
        if not src.exists():
            continue
        try:
            df = pd.read_csv(src, **read_kw)
            if date_col not in df.columns:
                print(f"[journal] {rel}: no '{date_col}' column - skipped")
                continue
            rows = df[_norm(df[date_col]) == day]
            if rows.empty:
                continue
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(dest, index=False, encoding=read_kw.get("encoding", "utf-8"))
            total += len(rows)
            print(f"[journal] {rel}: {len(rows)} rows -> {dest}")
        except Exception as e:
            print(f"[journal] {rel}: ERROR {type(e).__name__}: {e}")

    if total:
        print(f"[journal] {day}: {total} rows saved under {out_root}")
    else:
        print(f"[journal] {day}: nothing new to save")


if __name__ == "__main__":
    main()
