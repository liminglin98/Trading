"""Calibrates intraday_monitor.IV_METHOD_BIAS -- the conversion from the
live IV the monitor measures to the IV units each alert's trigger level was
calibrated in.

Why this exists: intraday_monitor compares a LIVE ATM IV against a trigger
level derived from a DAILY historical series, and the two were not being
computed the same way (found 2026-08-17). There are two independent
mismatches, and this script measures both per underlying:

  1. Basis (call-only vs call+put average). This is the big one -- put IV
     sits well above call IV at the same strike, so averaging both legs
     reads high against a call-only trigger. NOT corrected numerically:
     intraday_monitor now just measures the same legs the trigger's series
     used (see IV_BASIS_BY_SOURCE there). Reported here for the record.

  2. Method (how IV is obtained), which remains even once the legs match:
       - "call_only" triggers come from backtest._atm_iv_series, which READS
         the exchange's published 隐含波动率 column. The monitor instead
         BS-inverts a bid/ask mid. The exchange's inversion uses its own
         rate/daycount conventions, so the two differ by a small, very
         stable offset.
       - "cp_avg" triggers come from consolidated_store.atm_iv_series, which
         BS-inverts with the SAME bs_iv.implied_vol call the monitor uses --
         only the price differs (daily close vs intraday mid). Expected
         ~zero; measured here to confirm.

  Component 2 IS corrected, as a per-underlying constant subtracted from the
  live reading. A constant offset is the right shape of correction here:
  the trigger levels are percentiles of the historical series, and a
  percentile shifts by exactly the constant you shift the whole series by --
  so "live minus a constant, vs. the existing level" is equivalent to
  re-deriving the level from a shifted series, without disturbing any
  backtested signal definition.

Usage:
    python tools/iv_basis_calibration.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

warnings.filterwarnings("ignore")

import bs_iv  # noqa: E402
import consolidated_store as cs  # noqa: E402
import data_store as ds  # noqa: E402
import minute_updater as MU  # noqa: E402

DTE_MIN = 7
LOOKBACK = 60  # trading days used for the reported constants


def _atm_pairs(u: str) -> pd.DataFrame:
    """One row per date: the same near-month ATM strike backtest._atm_iv_series
    and consolidated_store.atm_iv_series both select, with the exchange's IV
    and the BS-inverted close IV for each leg."""
    chain = ds.load_chain(u)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    chain = chain[chain[ds.C_CLOSE] > 0]

    rows = []
    for dt, day in chain.groupby(ds.C_DATE):
        if day[ds.C_UND_CLOSE].isna().all():
            continue
        spot = float(day[ds.C_UND_CLOSE].iloc[0])
        dtes = day[ds.C_DTE]
        tgt = dtes[dtes >= DTE_MIN].min() if (dtes >= DTE_MIN).any() else dtes.min()
        day = day[day[ds.C_DTE] == tgt]
        calls, puts = day[day[ds.C_CP] == "C"], day[day[ds.C_CP] == "P"]
        if calls.empty or puts.empty:
            continue
        ci = (calls[ds.C_STRIKE] - spot).abs().idxmin()
        cr = calls.loc[ci]
        K = float(cr[ds.C_STRIKE])
        pm = puts[puts[ds.C_STRIKE] == K]
        if pm.empty:
            continue
        pr = pm.iloc[0]
        t = float(cr[ds.C_DTE]) / 365.0
        rows.append(dict(
            date=dt, strike=K, spot=spot, dte=int(cr[ds.C_DTE]),
            exch_C=float(cr[ds.C_IV]) if cr[ds.C_IV] > 0 else np.nan,
            bs_C=bs_iv.implied_vol("C", spot, K, t, float(cr[ds.C_CLOSE])),
            bs_P=bs_iv.implied_vol("P", spot, K, t, float(pr[ds.C_CLOSE])),
        ))
    return pd.DataFrame(rows).set_index("date").sort_index()


def _mid_vs_close(u: str) -> tuple[float, int]:
    """BS(bid/ask mid) - BS(last) on the ATM legs, from the minute files we
    have on disk -- the only part of the live path that can't be measured
    from daily data. Expected ~0 for a liquid ATM contract."""
    diffs = []
    for path in sorted((ds.DATA_DIR / "minute").glob(f"option_minute_{u}_*.csv")):
        df = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
        if df.empty:
            continue
        last_ts = df["timestamp"].max()
        snap = df[df["timestamp"] == last_ts]
        snap = snap[(snap["dte"] >= DTE_MIN) & (snap["bid"] > 0) & (snap["ask"] > 0) & (snap["last"] > 0)]
        if snap.empty:
            continue
        near = snap[snap["dte"] == snap["dte"].min()]
        spot = float(near["spot"].iloc[0])
        atm_k = near.loc[(near["strike"] - spot).abs().idxmin(), "strike"]
        for _, r in near[near["strike"] == atm_k].iterrows():
            t = float(r["dte"]) / 365.0
            iv_mid = bs_iv.implied_vol(r["cp"], spot, float(r["strike"]), t,
                                       (float(r["bid"]) + float(r["ask"])) / 2)
            iv_last = bs_iv.implied_vol(r["cp"], spot, float(r["strike"]), t, float(r["last"]))
            if iv_mid == iv_mid and iv_last == iv_last:
                diffs.append(iv_mid - iv_last)
    return (float(np.mean(diffs)) if diffs else float("nan")), len(diffs)


def main() -> None:
    out = {}
    for u in MU.ETF_UNDERLYINGS:
        df = _atm_pairs(u)
        if df.empty:
            print(f"{u}: no usable chain rows")
            continue
        df["bs_avg"] = df[["bs_C", "bs_P"]].mean(axis=1)
        w = df.tail(LOOKBACK)

        # (1) basis: what averaging both legs adds vs. a call-only reading
        skew = (w["bs_avg"] - w["bs_C"]).dropna()
        # (2a) method, call_only basis: our BS inversion vs the exchange column
        m_call = (w["bs_C"] - w["exch_C"]).dropna()
        # (2b) method, cp_avg basis: same inversion both sides -> price source only
        mid_close, n_mid = _mid_vs_close(u)

        # Median, not mean: 510500/588000 populate the exchange IV column on
        # only a fraction of sessions (see n= below), so this estimator has to
        # survive a short, outlier-prone sample.
        out[u] = (float(m_call.median()), mid_close)
        print(f"\n=== {u} ===  ({len(w)} sessions, {w.index.min().date()} .. {w.index.max().date()})")
        print(f"  [1] basis   call+put avg - call only : mean {skew.mean():+.4f}  "
              f"median {skew.median():+.4f}  std {skew.std():.4f}  n={len(skew)}")
        print(f"  [2a] method call_only  BS(close) - exch : mean {m_call.mean():+.4f}  "
              f"median {m_call.median():+.4f}  std {m_call.std():.4f}  n={len(m_call)}")
        print(f"  [2b] method cp_avg     BS(mid) - BS(last): mean {mid_close:+.4f}  n={n_mid}")

    print("\n\n# paste into intraday_monitor.py")
    print("IV_METHOD_BIAS = {")
    for u, (call_bias, mid_bias) in out.items():
        mid = 0.0 if not (mid_bias == mid_bias) else mid_bias
        print(f'    "{u}": {{"call_only": {call_bias:+.4f}, "cp_avg": {mid:+.4f}}},')
    print("}")


if __name__ == "__main__":
    main()
