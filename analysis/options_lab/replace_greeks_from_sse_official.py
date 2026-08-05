"""
Replace Delta/Gamma/Theta/Vega/Rho/IV in Data/options/option_daily_{code}.csv
for 2026-06-01 through the latest published date with official SSE
risk-indicator data (`akshare.option_risk_indicator_sse`, query.sse.com.cn --
independent of Sina and eastmoney, the two sources everything else in this
pipeline has used so far).

Why: cross-validated 2026-07-28 (see reference_data_sources memory)
against 165 matched contracts on 2026-07-27 -- feeding the *official* IV
into bs_iv.py's formulas reproduces the *official* Greeks almost exactly
(corr 0.99-1.00, both calls and puts), proving Sina's own reported
call-side IV/Greeks for 510500/588000/588080 specifically were the
unreliable ingredient, not the formulas. This source is authoritative
(the exchange itself), covers all 5 SSE underlyings together in one
request per date, and goes back to each underlying's actual listing date
(verified against 2015/2020/2022/2023 spot-checks, not just the docstring).

Scope: only replaces Delta/Gamma/Theta/Vega/Rho/IV -- this endpoint has no
price/OHLC/volume, so those columns are untouched. Only rows where the
official source actually has a matching (code, date) get overwritten;
everything else (dates outside the window, contracts the official feed
doesn't carry) is left as-is.

Paced at ~0.3s/request (~40 requests for the June-July window) -- a
different domain than eastmoney's push2 API that got anti-bot-blocked
earlier this session, but still worth not hammering.
"""
import time

import akshare as ak
import pandas as pd

import data_store as ds

SSE_CODES = ["510050", "510300", "510500", "588000", "588080"]
OFFICIAL_TO_LOCAL = {
    "DELTA_VALUE": ds.C_DELTA, "GAMMA_VALUE": ds.C_GAMMA,
    "THETA_VALUE": ds.C_THETA, "VEGA_VALUE": ds.C_VEGA,
    "RHO_VALUE": ds.C_RHO, "IMPLC_VOLATLTY": ds.C_IV,
}
START, END = "2026-06-01", "2026-07-27"


def fetch_official(start: str = START, end: str = END) -> pd.DataFrame:
    cal = ak.tool_trade_date_hist_sina()
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    dates = cal[(cal["trade_date"] >= start) & (cal["trade_date"] <= end)][
        "trade_date"].dt.strftime("%Y%m%d").tolist()
    frames = []
    for d in dates:
        df = ak.option_risk_indicator_sse(date=d)
        if len(df):
            frames.append(df)
        print(f"  fetched {d}: {len(df)} rows")
        time.sleep(0.3)
    official = pd.concat(frames, ignore_index=True)
    official["TRADE_DATE"] = pd.to_datetime(official["TRADE_DATE"]).dt.strftime("%Y-%m-%d")
    return official


def replace_for_code(code: str, official: pd.DataFrame) -> int:
    path = ds.chain_path(code)
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype={ds.C_CODE: str})

    off = official.rename(columns={"SECURITY_ID": ds.C_CODE, "TRADE_DATE": ds.C_DATE})
    off = off[[ds.C_CODE, ds.C_DATE, *OFFICIAL_TO_LOCAL]].drop_duplicates(
        subset=[ds.C_CODE, ds.C_DATE])

    merged = raw.merge(off, on=[ds.C_CODE, ds.C_DATE], how="left",
                       suffixes=("", "_official"))
    n_matched = merged["DELTA_VALUE"].notna().sum()
    for official_col, local_col in OFFICIAL_TO_LOCAL.items():
        mask = merged[official_col].notna()
        merged.loc[mask, local_col] = merged.loc[mask, official_col]
    merged = merged[ds.COLUMNS]
    merged.to_csv(path, index=False, encoding="utf-8-sig")
    return int(n_matched)


def main():
    print(f"Fetching option_risk_indicator_sse for {START} .. {END}...")
    official = fetch_official()
    print(f"Total official rows fetched: {len(official):,}")

    for code in SSE_CODES:
        n = replace_for_code(code, official)
        print(f"{code}: {n:,} rows replaced with official SSE Greeks/IV")


if __name__ == "__main__":
    main()
