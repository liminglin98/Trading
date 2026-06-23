# -*- coding: utf-8 -*-
"""
option_iv_history_probe.py
==========================
Confirm the SSE historical per-contract IV source works on this network and that
we can parse 50ETF strikes/IV from it -- the data QMT lacks, needed for the
butterfly/straddle timing backtest.

Source: ak.option_risk_indicator_sse(date="YYYYMMDD")
  Returns greeks + implied vol (IMPLC_VOLATLTY) for EVERY SSE option contract on a
  trade date, from 2015-02-09 onward. CONTRACT_SYMBOL encodes underlying/type/strike,
  e.g. '510050C2412M03000' -> underlying 510050, Call, expiry 2412, strike 3.000.

What it does: pulls a recent and an old month-end, filters 50ETF, parses the
contract symbol into (type, expiry, strike), and prints the IV smile + coverage so
we can confirm parsing before building the backtest.

RUN IN A NORMAL PYTHON ENV (needs internet). SSE-hosted (not eastmoney).
"""

import re
import warnings
import datetime as dt

warnings.simplefilter("ignore")

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
except Exception as e:  # noqa
    print("!! need akshare pandas numpy:", e)
    raise SystemExit(1)

UNDERLYING = "510050"   # 50ETF

# Candidate parsers for the contract identifier. We don't yet know which column
# (CONTRACT_ID vs CONTRACT_SYMBOL) holds a machine-parseable code, or whether it's
# a code ('510050C2412M03000') or a Chinese name ('50ETF购12月3000A'). Try several.
RE_CODE = re.compile(r"(\d{6})\s*([CP])\s*(\d{4})[MA](\d+)")          # code form
RE_CN   = re.compile(r"(50ETF|510050).*?(购|沽).*?(\d{3,5})\s*[A-Z]?$")  # Chinese name


def parse_any(s):
    s = str(s)
    m = RE_CODE.search(s)
    if m:
        und, cp, exp, k = m.groups()
        return dict(underlying=und, type=cp, expiry=exp, strike=int(k) / 1000.0)
    m = RE_CN.search(s)
    if m:
        und, cp, k = m.groups()
        cp = "C" if cp == "购" else "P"
        kv = int(k)
        strike = kv / 1000.0 if kv >= 1000 else kv / 100.0   # 3000->3.0 ; 300->3.0
        return dict(underlying="510050", type=cp, expiry="?", strike=strike)
    return None


def pull(date):
    print(f"\n=== option_risk_indicator_sse(date={date}) ===")
    try:
        df = ak.option_risk_indicator_sse(date=date)
    except Exception as e:  # noqa
        print(f"  [ERR] {type(e).__name__}: {e}")
        return None
    if df is None or len(df) == 0:
        print("  [empty] no rows (non-trading day?)")
        return None
    print(f"  rows={len(df)}  cols={list(df.columns)}")

    id_col  = next((c for c in df.columns if c.upper() == "CONTRACT_ID"), None)
    sym_col = next((c for c in df.columns if "SYMBOL" in c.upper()), None)
    iv_col  = next((c for c in df.columns if "VOLAT" in c.upper()), None)

    # show raw samples so we can see the real identifier format
    for c in [id_col, sym_col]:
        if c is not None:
            print(f"  sample {c}: {list(df[c].astype(str).head(8))}")

    # try parsing each candidate column; report which yields the most matches
    best_col, best_info = None, None
    for c in [id_col, sym_col]:
        if c is None:
            continue
        parsed = df[c].map(parse_any)
        ok = parsed.notna()
        n = int(ok.sum())
        print(f"  parse {c}: {n}/{len(df)} matched")
        if n and (best_info is None or n > len(best_info)):
            info = pd.DataFrame([p for p in parsed if p is not None])
            info["iv"] = pd.to_numeric(df.loc[ok.values, iv_col].values, errors="coerce")
            best_col, best_info = c, info

    if best_info is None:
        print("  [warn] no parser matched -- paste the samples above and I'll fix the parser")
        return df

    etf = best_info[best_info["underlying"].isin(["510050", UNDERLYING])].dropna(subset=["iv"])
    if etf.empty:
        print(f"  [warn] parsed via {best_col} but no {UNDERLYING} rows")
        return df
    exps = sorted(e for e in etf["expiry"].unique())
    print(f"  {UNDERLYING}: {len(etf)} contracts via {best_col} | expiries={exps}")
    front = exps[0]
    sm = (etf[etf["expiry"] == front].groupby(["type", "strike"])["iv"]
          .mean().unstack("type").sort_index())
    print(f"  front-expiry ({front}) IV smile (strike -> call/put IV):")
    print(sm.to_string(float_format=lambda v: f"{v:.3f}"))
    return df


def main():
    print("Probing SSE historical option IV (option_risk_indicator_sse)")
    # a recent trading day and a couple of historical month-ends to gauge depth
    today = dt.date.today()
    recent = (today - dt.timedelta(days=4)).strftime("%Y%m%d")
    for d in [recent, "20240628", "20200630", "20150302"]:
        pull(d)
    print("\nIf the smiles print with sensible IVs across strikes, we can build the")
    print("backtest: loop month-ends, reconstruct the real IV smile, BS-price the")
    print("structures vs the model's predictive draws, and test the timing rule.")


if __name__ == "__main__":
    main()
