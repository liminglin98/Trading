# -*- coding: utf-8 -*-
"""
vrp_universe.py
===============
Part 2 of the plan: the 50ETF-monthly condor showed a REAL but WEAK signal
(corr(gap, r_trade) ~ +0.3, positive across all geometries but p~0.3 on only
~10 trades) sitting on a tail-dominated payoff. The fix for "weak + tiny
sample" is more data, NOT more tuning. This extends the same model-gated,
fixed-size, narrow iron condor to additional SSE ETF-option underlyings and
POOLS the trades for a properly powered correlation test.

  underlyings: 50ETF (510050), 300ETF (510300), 500ETF (510500)
  shared signal: model_ann from the macro factor model is treated as a common
                 market vol-REGIME forecast; each underlying is gated on its
                 own (implied - model_ann).

EFFICIENCY: one RAW SSE pull per date (option_risk_indicator_sse returns ALL
SSE options that day) is cached to disk and filtered per underlying in memory,
so adding underlyings costs ~no extra network. A date-aware, UNDERLYING-aware
filter is monkeypatched into the backtest module so entry build AND exit
re-fetch both resolve to the active underlying.

CAVEATS:
 * 500ETF options listed ~2022-09 -> short history, few months.
 * model_ann is a 50ETF-return-model forecast reused as a market vol proxy;
   the pooled corr test is exactly what checks whether that transfers.
 * pooled trades are NOT independent (underlyings correlate ~0.9), so the
   effective sample is smaller than the raw count -> read the pooled p as
   optimistic. Still far better than n=10.

RUN IN A NORMAL PYTHON ENV (first run fetches dates not already cached).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

import butterfly_timing_backtest as B
import butterfly_param_search as PS
import vrp_harvest_compare as VH

OUTDIR = B.OUTDIR

# underlying code -> sina daily ETF symbol
UNIVERSE = {"510050": "sh510050", "510300": "sh510300", "510500": "sh510500"}

# fixed strategy config (the well-sampled, non-artifact geometry)
VH.SIZING_MODE = "structural"
VH.WING_STEPS  = 2
VH.SHORT_STEPS = 1
VH.GAP_MIN     = 0.00
VH.PROFIT_TAKE = 0.50
VH.EXIT_DTE    = 7
VH.STOP_FRAC   = None
VH.TARGET_RISK = 0.10

# =============================================================================
# Raw-by-date SSE IV cache (one pull serves every underlying)
# =============================================================================
RAW_CACHE_PATH = OUTDIR / "iv_raw_by_date.pkl"
_RAW: dict[str, object] = {}
if RAW_CACHE_PATH.exists():
    try:
        _RAW = pickle.load(open(RAW_CACHE_PATH, "rb"))
        print(f"loaded raw IV cache: {len(_RAW)} dates")
    except Exception:
        _RAW = {}
_DIRTY = 0

def _raw_for_date(date):
    """Full SSE option_risk_indicator_sse frame for a date (cached, all underlyings)."""
    global _DIRTY
    key = pd.Timestamp(date).strftime("%Y%m%d")
    if key in _RAW:
        return _RAW[key]
    try:
        df = B.ak.option_risk_indicator_sse(date=pd.Timestamp(date).strftime("%Y%m%d"))
    except Exception:
        df = None
    _RAW[key] = df
    _DIRTY += 1
    if _DIRTY % 25 == 0:
        _flush()
    return df

def _flush():
    try:
        pickle.dump(_RAW, open(RAW_CACHE_PATH, "wb"))
    except Exception as e:
        print("raw cache flush failed:", e)

def fetch_iv_filtered(trade_date):
    """Mirror B.fetch_iv_table, but source from the raw cache and filter by the
    ACTIVE B.UNDERLYING -> [type, expiry, strike, iv] or None."""
    df = _raw_for_date(trade_date)
    if df is None or len(df) == 0:
        return None
    cid = next(c for c in df.columns if c.upper() == "CONTRACT_ID")
    ivc = next(c for c in df.columns if "VOLAT" in c.upper())
    recs = []
    for c, iv in zip(df[cid], pd.to_numeric(df[ivc], errors="coerce")):
        p = B.parse_cid(c)
        if p and p[0] == B.UNDERLYING and np.isfinite(iv) and iv > 0:
            recs.append((p[1], p[2], p[3], iv))
    if not recs:
        return None
    return pd.DataFrame(recs, columns=["type", "expiry", "strike", "iv"])

# route entry + exit fetches through the underlying-aware filter
B.fetch_iv_table = fetch_iv_filtered

# =============================================================================
# Per-underlying contexts (shared factor model; underlying-specific chain/ETF)
# =============================================================================
def build_contexts_for(etf_symbol, X, y):
    etf = B.ak.fund_etf_hist_sina(symbol=etf_symbol)
    etf["date"] = pd.to_datetime(etf["date"])
    etf = etf.set_index("date")["close"].sort_index()
    last_daily = etf.index.max()
    idx = X.index

    contexts = []
    for ti in range(B.MIN_TRAIN, len(X)):
        t = idx[ti]
        draws = B.fit_predict_conjugate(X.values[:ti], y.values[:ti], X.values[ti:ti + 1])[0]
        model_ann = float(draws.std()) * np.sqrt(12)

        iv_tbl, entry_date = None, None
        for back in range(0, 7):
            d = pd.Timestamp((t - pd.Timedelta(days=back)).date())
            tbl = B.fetch_iv_table(d)          # underlying-aware (B.UNDERLYING)
            if tbl is not None:
                iv_tbl, entry_date = tbl, d
                break
        if iv_tbl is None:
            continue
        spot = float(etf.asof(entry_date))
        if not np.isfinite(spot):
            continue
        expiry, exp_date = B.choose_expiry(iv_tbl, entry_date)
        if expiry is None or pd.Timestamp(exp_date) > last_daily:
            continue
        T = max((exp_date - entry_date.date()).days, 1) / 365.0
        smile = B.otm_smile(iv_tbl, expiry, spot)
        if len(smile) < 5:
            continue
        atm_iv = float(smile.iloc[(smile.index - spot).to_series().abs().values.argmin()])
        Tm = T * 12.0
        d_h = draws.mean() * Tm + (draws - draws.mean()) * np.sqrt(Tm)
        ST = spot * np.exp(d_h)
        contexts.append(dict(t=t, entry_date=entry_date, spot=spot, expiry=expiry,
                             exp_date=exp_date, T=T, smile=smile, ST=ST,
                             atm_iv=atm_iv, model_ann=model_ann))
    return contexts, etf, last_daily

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 72)
    print("VRP UNIVERSE -- model-gated narrow iron condor across SSE ETF options")
    print(f"  geometry short={VH.SHORT_STEPS} wing={VH.WING_STEPS} GAP_MIN={VH.GAP_MIN}")
    print("=" * 72)

    df = B.load_panel()
    X, y, _ = B.build_xy(df)

    per_rows, pooled = [], []
    for u, sym in UNIVERSE.items():
        B.UNDERLYING = u                       # drives the filter + exit fetch
        try:
            contexts, etf, last_daily = build_contexts_for(sym, X, y)
        except Exception as e:
            print(f"  {u}: build failed ({e})"); continue
        bt, funnel = VH.evaluate("iron_condor", contexts, etf, last_daily)
        _bt2, _bh, m = B.equity_and_metrics(bt, df, y)
        mdd = m["rule_mdd"]
        calmar = m["rule_cagr"] / abs(mdd) if (mdd and np.isfinite(mdd) and mdd != 0) else np.nan
        tr = bt[bt.get("traded", 0) == 1].copy()
        n = len(tr)
        r = p = worst = np.nan
        if n >= 4 and {"gap", "r_trade"}.issubset(tr.columns):
            gg = tr["gap"].to_numpy(float); rr = tr["r_trade"].to_numpy(float)
            ok = np.isfinite(gg) & np.isfinite(rr)
            if ok.sum() >= 4:
                r, p = pearsonr(gg[ok], rr[ok])
        if n >= 1:
            worst = float(tr["r_trade"].min())
        if n:
            tr["underlying"] = u
            pooled.append(tr[["underlying", "gap", "r_trade"]])
        per_rows.append(dict(underlying=u, months=funnel["months"], trades=n,
                             cagr=m["rule_cagr"], mdd=mdd, calmar=calmar,
                             winrate=m["trade_winrate"], corr_gap=r, corr_p=p,
                             worst_month=worst))
        bt.to_csv(OUTDIR / f"vrp_universe_{u}_trades.csv")
        print(f"  {u}: months={funnel['months']} no_gap={funnel['no_gap']} "
              f"no_build={funnel['no_build']} -> trades={n}  "
              f"calmar={calmar:+.2f} corr_gap={r:+.2f} (p={p:.2f}) worst={worst:+.2f}")
    _flush()

    print("\n--- per-underlying ---")
    pdf = pd.DataFrame(per_rows)
    pdf.to_csv(OUTDIR / "vrp_universe_summary.csv", index=False)
    print(pdf.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # ---- POOLED powered correlation test ----
    print("\n--- POOLED signal test (all underlyings) ---")
    if pooled:
        allt = pd.concat(pooled, ignore_index=True)
        allt.to_csv(OUTDIR / "vrp_universe_pooled_trades.csv", index=False)
        g = allt["gap"].to_numpy(float); rt = allt["r_trade"].to_numpy(float)
        ok = np.isfinite(g) & np.isfinite(rt)
        N = int(ok.sum())
        if N >= 4:
            r_all, p_all = pearsonr(g[ok], rt[ok])
            # Fisher CI for r
            z = np.arctanh(r_all); se = 1.0 / np.sqrt(N - 3)
            lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
            print(f"  pooled trades N={N}")
            print(f"  corr(gap, r_trade) = {r_all:+.3f}   p={p_all:.3f}   "
                  f"95% CI [{lo:+.2f}, {hi:+.2f}]")
            print(f"  mean r_trade={np.mean(rt[ok]):+.3f}  worst={np.min(rt[ok]):+.3f}  "
                  f"win%={(rt[ok] > 0).mean()*100:.0f}%")
            print("\n  Read: if the CI is comfortably above 0 with N~25-35, the model")
            print("  signal is CONFIRMED across underlyings (not 50ETF luck). If the CI")
            print("  straddles 0, the +0.3 was small-sample noise -> stop, model is a")
            print("  risk/sizing tool, not alpha. (Underlyings ~0.9 correlated, so the")
            print("  effective N is smaller than N -> treat p as optimistic.)")
        else:
            print("  too few pooled trades.")
    print(f"\nSaved: vrp_universe_summary.csv, vrp_universe_pooled_trades.csv, "
          f"vrp_universe_*_trades.csv  (in {OUTDIR})")


if __name__ == "__main__":
    main()
