"""Union-merge two copies of Data/ that diverged while running on separate
machines (e.g. the home server and a laptop).

Why this exists: data_store.append_chain() only accepts dates strictly greater
than the max date already on file, so it can never fill a hole in the middle.
Once the laptop has appended 2026-07-20+, the server's 2026-07-11..17 rows can
only be brought in by a real union. That is what this does.

Usage:
    python merge_data.py --other "D:/from-server/Data"              # dry run
    python merge_data.py --other "D:/from-server/Data" --write      # apply
    python merge_data.py --other ... --prefer other --write

--prefer decides which side wins when the SAME key exists on both sides with
different values. Default 'mine' (this checkout). Rows whose key exists on only
one side are always kept, whichever side they came from.

Every written file is backed up to <name>.bak-YYYYmmdd-HHMMSS first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import data_store as ds

STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

# filename -> (key columns, read kwargs)
CHAIN_KEYS = [ds.C_CODE, ds.C_DATE]
PLANS: dict[str, tuple[list[str], dict]] = {
    "options/option_daily_510050.csv": (CHAIN_KEYS,
                                        {"encoding": "utf-8-sig",
                                         "dtype": {ds.C_CODE: str, ds.C_UND: str}}),
    "options/option_daily_510300.csv": (CHAIN_KEYS,
                                        {"encoding": "utf-8-sig",
                                         "dtype": {ds.C_CODE: str, ds.C_UND: str}}),
    "options/etf_daily_510050.csv": (["date"], {}),
    "options/etf_daily_510300.csv": (["date"], {}),
    "options/fund_flow_daily.csv": (["date", "underlying"],
                                    {"dtype": {"underlying": str}}),
    "options/market_sentiment_daily.csv": (["date"], {}),
}

DATE_COL = {"options/option_daily_510050.csv": ds.C_DATE,
            "options/option_daily_510300.csv": ds.C_DATE}


def _sort_cols(df: pd.DataFrame, keys: list[str]) -> list[str]:
    return [c for c in keys if c in df.columns]


def merge_one(rel: str, mine_root: Path, other_root: Path,
              keys: list[str], read_kw: dict, prefer: str,
              write: bool) -> str:
    mine_p, other_p = mine_root / rel, other_root / rel
    if not other_p.exists():
        return f"  {rel}: not present on other side - skipped"
    if not mine_p.exists():
        if write:
            mine_p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(other_p, mine_p)
        return f"  {rel}: missing locally - copied whole file from other"

    mine = pd.read_csv(mine_p, **read_kw)
    other = pd.read_csv(other_p, **read_kw)

    missing = [k for k in keys if k not in mine.columns or k not in other.columns]
    if missing:
        return f"  {rel}: SKIPPED - key column(s) {missing} not in both files"

    # normalise key dtypes so the dedupe actually matches
    for k in keys:
        if "date" in k.lower() or k == ds.C_DATE:
            mine[k] = pd.to_datetime(mine[k]).dt.strftime("%Y-%m-%d")
            other[k] = pd.to_datetime(other[k]).dt.strftime("%Y-%m-%d")
        else:
            mine[k] = mine[k].astype(str)
            other[k] = other[k].astype(str)

    first, second = (mine, other) if prefer == "mine" else (other, mine)
    combined = pd.concat([first, second], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="first")

    sortby = _sort_cols(combined, [DATE_COL.get(rel, keys[0])] + keys)
    combined = combined.sort_values(sortby).reset_index(drop=True)

    added = len(combined) - len(mine)
    dcol = DATE_COL.get(rel, "date" if "date" in combined.columns else keys[0])
    span = ""
    if dcol in combined.columns:
        span = f"  span {combined[dcol].min()} -> {combined[dcol].max()}"

    if write:
        shutil.copy2(mine_p, mine_p.with_suffix(mine_p.suffix + f".bak-{STAMP}"))
        combined.to_csv(mine_p, index=False,
                        encoding=read_kw.get("encoding", "utf-8"))

    return (f"  {rel}: mine={len(mine)} other={len(other)} "
            f"-> {len(combined)} (+{added}){span}")


def merge_dashboard(mine_root: Path, other_root: Path, prefer: str,
                    write: bool) -> list[str]:
    """Every Data/akshare/dashboard/*.csv is a plain date,value series."""
    out = []
    other_dir = other_root / "akshare" / "dashboard"
    if not other_dir.is_dir():
        return ["  akshare/dashboard: not present on other side - skipped"]
    for other_p in sorted(other_dir.glob("*.csv")):
        rel = f"akshare/dashboard/{other_p.name}"
        out.append(merge_one(rel, mine_root, other_root, ["date"], {},
                             prefer, write))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--other",
                     help="path to the OTHER Data/ directory (e.g. from the server)")
    src.add_argument("--journal",
                     help="path to a journal root; replays every YYYY-MM-DD "
                          "subdirectory in date order")
    ap.add_argument("--mine", default=str(ds.DATA_DIR.parent),
                    help="path to this machine's Data/ (default: repo Data/)")
    ap.add_argument("--prefer", choices=["mine", "other"], default="mine",
                    help="which side wins on conflicting duplicate keys")
    ap.add_argument("--write", action="store_true",
                    help="actually write; otherwise dry-run")
    args = ap.parse_args()

    mine_root = Path(args.mine)
    if not mine_root.is_dir():
        sys.exit(f"--mine is not a directory: {mine_root}")

    mode = "WRITING" if args.write else "DRY RUN (nothing written)"

    if args.journal:
        jroot = Path(args.journal)
        if not jroot.is_dir():
            sys.exit(f"--journal is not a directory: {jroot}")
        days = sorted(p for p in jroot.iterdir()
                      if p.is_dir() and len(p.name) == 10 and p.name[4] == "-")
        if not days:
            sys.exit(f"no YYYY-MM-DD subdirectories under {jroot}")
        print(f"{mode} | prefer={args.prefer} | replaying {len(days)} journal days")
        print(f"mine   : {mine_root}")
        print(f"journal: {jroot}\n")
        for day in days:
            print(f"[{day.name}]")
            for rel, (keys, read_kw) in PLANS.items():
                try:
                    print(merge_one(rel, mine_root, day, keys, read_kw,
                                    args.prefer, args.write))
                except Exception as e:
                    print(f"  {rel}: ERROR {type(e).__name__}: {e}")
        print("\nDry run only. Re-run with --write to apply." if not args.write
              else f"\nDone. Backups written as *.bak-{STAMP}")
        return

    other_root = Path(args.other)
    if not other_root.is_dir():
        sys.exit(f"--other is not a directory: {other_root}")

    print(f"{mode} | prefer={args.prefer}")
    print(f"mine : {mine_root}")
    print(f"other: {other_root}\n")

    lines = []
    for rel, (keys, read_kw) in PLANS.items():
        try:
            lines.append(merge_one(rel, mine_root, other_root, keys, read_kw,
                                   args.prefer, args.write))
        except Exception as e:
            lines.append(f"  {rel}: ERROR {type(e).__name__}: {e}")
    try:
        lines += merge_dashboard(mine_root, other_root, args.prefer, args.write)
    except Exception as e:
        lines.append(f"  akshare/dashboard: ERROR {type(e).__name__}: {e}")

    print("\n".join(lines))
    if not args.write:
        print("\nDry run only. Re-run with --write to apply.")
    else:
        print(f"\nDone. Backups written as *.bak-{STAMP}")


if __name__ == "__main__":
    main()
