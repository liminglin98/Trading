# -*- coding: utf-8 -*-
"""
butterfly_param_search.py
=========================
2-D grid search over the vol-timing strategy in butterfly_timing_backtest.py.

Design (mirrors the macro-model optimizer: cache the expensive kernel, iterate
cheaply on top):
  * The expensive part is the SSE IV fetches. We wrap fetch_iv_table in a
    DISK cache keyed by date and monkeypatch it into the backtest module, so
    every unique date is pulled from the network at most once -- across ALL
    grid cells AND across re-runs of this script. First pass warms the cache;
    later passes are basically instant.
  * Per-month, param-INDEPENDENT work (model fit, entry smile, chosen expiry,
    scaled draws) is built ONCE into `contexts`.
  * evaluate(...) is pure compute given the contexts + a parameter vector. It
    reuses the exact math helpers from the single-run script; only the
    structure-selection + first-to-fire exit orchestration is mirrored here,
    with every parameter passed explicitly (no module globals).

Searched axes (2-D, per the chosen scope):
    GATE_PARAM in {HI_RATIO, LO_RATIO}   x   PROFIT_TAKE
STOP_LOSS is fixed off (shown inert-to-harmful). KELLY_FRAC is NOT in the grid
-- it only scales sizing, so it is swept separately on the best cell from the
stored per-trade (r_trade, kelly_f), with no re-run.

Objective: ranked by Calmar (CAGR / |max drawdown|) by default; CAGR, MDD,
win-rate and trade count are reported for every cell regardless.

NOTE: this was written without a live run here (needs internet/SSE). Sanity
check: a single-cell grid at your last params should reproduce your last
single-run numbers -- see the regression note printed at the end.

RUN IN A NORMAL PYTHON ENV (needs internet on the FIRST pass to warm the cache).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import butterfly_timing_backtest as B   # reuse all the proven math helpers

# =============================================================================
# SEARCH CONFIG
# =============================================================================
GATE_PARAM = "LO_RATIO"          # which entry threshold to sweep on the gate axis.
                                 # straddles dominate this sample (they fire on
                                 # ratio <= LO_RATIO), so LO is the active gate;
                                 # set "HI_RATIO" to tune the butterfly side instead.
GATE_GRID  = [0.75, 0.80, 0.85, 0.90, 0.95]      # values for GATE_PARAM
PT_GRID    = [0.25, 0.40, 0.50, 0.65, 0.80, 1.00]  # PROFIT_TAKE values

# held fixed across the grid (the non-swept gate keeps its baseline)
FIXED_HI    = 1.20
FIXED_LO    = 0.85
FIXED_EXIT_DTE = 7
FIXED_STOP  = None               # stop-loss off (inert-to-harmful in tests)
KELLY_FRAC  = 0.33               # grid uses this; Kelly is swept separately below
KELLY_CAP   = B.KELLY_CAP

OBJECTIVE   = "calmar"           # "calmar" (CAGR/|MDD|) or "rule_cagr"
KELLY_GRID  = [0.10, 0.20, 0.33, 0.50, 0.75, 1.00]   # for the post-hoc sizing sweep

OUTDIR = B.OUTDIR
CACHE_PATH = OUTDIR / "iv_table_cache.pkl"

# =============================================================================
# Disk-cached IV fetch (monkeypatched into B so exit refetches hit it too)
# =============================================================================
_RAW_FETCH = B.fetch_iv_table          # keep a handle to the real network fetch
_IV_MEM: dict[str, object] = {}
if CACHE_PATH.exists():
    try:
        _IV_MEM = pickle.load(open(CACHE_PATH, "rb"))
        print(f"loaded IV cache: {len(_IV_MEM)} dates")
    except Exception:
        _IV_MEM = {}

_DIRTY = 0
def cached_fetch_iv_table(trade_date):
    """Date-keyed cache around the SSE fetch. Stores None for empty dates too,
    so we never re-hit the network for a known-empty date."""
    global _DIRTY
    key = pd.Timestamp(trade_date).strftime("%Y%m%d")
    if key in _IV_MEM:
        return _IV_MEM[key]
    try:
        tbl = _RAW_FETCH(pd.Timestamp(trade_date))
    except Exception:
        tbl = None
    # Do NOT cache an empty result for TODAY: the EOD chain may not be posted yet
    # when an early call runs, and we must be free to re-fetch it later the same
    # day (otherwise the live spec / tracker get stuck on a stale chain).
    is_today = (key == pd.Timestamp.now().strftime("%Y%m%d"))
    if not (tbl is None and is_today):
        _IV_MEM[key] = tbl
        _DIRTY += 1
        if _DIRTY % 25 == 0:           # periodic flush during a long warm-up
            _flush_cache()
    return tbl

def _flush_cache():
    try:
        pickle.dump(_IV_MEM, open(CACHE_PATH, "wb"))
    except Exception as e:
        print("cache flush failed:", e)

# route ALL fetches (entry here + exit refetch inside B.find_iv_on_or_before)
B.fetch_iv_table = cached_fetch_iv_table

# =============================================================================
# Phase A: build per-month, parameter-INDEPENDENT contexts (expensive, once)
# =============================================================================
def build_contexts():
    df = B.load_panel()
    X, y, _ = B.build_xy(df)
    idx = X.index

    etf = B.ak.fund_etf_hist_sina(symbol="sh510050")
    etf["date"] = pd.to_datetime(etf["date"])
    etf = etf.set_index("date")["close"].sort_index()
    last_daily = etf.index.max()

    contexts = []
    for ti in range(B.MIN_TRAIN, len(X)):
        t = idx[ti]
        draws = B.fit_predict_conjugate(X.values[:ti], y.values[:ti],
                                        X.values[ti:ti + 1])[0]
        model_ann = float(draws.std()) * np.sqrt(12)

        iv_tbl, entry_date = None, None
        for back in range(0, 7):
            d = pd.Timestamp((t - pd.Timedelta(days=back)).date())
            tbl = cached_fetch_iv_table(d)
            if tbl is not None:
                iv_tbl, entry_date = tbl, d
                break
        if iv_tbl is None:
            continue

        spot = float(etf.asof(entry_date))
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
    _flush_cache()
    print(f"built {len(contexts)} month contexts; IV cache now {len(_IV_MEM)} dates")
    return contexts, etf, last_daily, X, y, df

# =============================================================================
# Phase B: pure-compute evaluation for one parameter vector
# =============================================================================
def _target_value(side, cost, strikes, pt):
    if pt is None:
        return None
    if side == "butterfly":
        klow, kmid, _ = strikes
        mp = (kmid - klow) - cost
        return cost + pt * mp if mp > 0 else None
    return cost * (1.0 + pt)

def evaluate(contexts, etf, last_daily, hi, lo, pt,
             exit_dte=FIXED_EXIT_DTE, stop=FIXED_STOP,
             kelly_frac=KELLY_FRAC, kelly_cap=KELLY_CAP):
    rows = []
    for c in contexts:
        smile, spot, T, ST = c["smile"], c["spot"], c["T"], c["ST"]
        expiry, exp_date, entry_date = c["expiry"], c["exp_date"], c["entry_date"]
        t = c["t"]
        ratio = c["atm_iv"] / c["model_ann"] if c["model_ann"] > 0 else np.nan

        # --- structure selection (same gate + Kelly-width logic as single run)
        struct, side, kelly_f = None, "flat", 0.0
        if ratio >= hi:
            best = None
            for n in range(1, B.MAX_WING_STEPS + 1):
                pr = B.price_butterfly(smile, spot, T, n)
                if not pr or pr["cost"] <= 1e-6:
                    continue
                f_, g_ = B.kelly_opt(B.payoff_butterfly(ST, pr["strikes"]) / pr["cost"] - 1.0)
                if g_ > 0 and (best is None or g_ > best[1]):
                    best = (pr, g_, f_)
            if best:
                struct, side, kelly_f = best[0], "butterfly", best[2]
        elif ratio <= lo:
            pr = B.price_straddle(smile, spot, T)
            if pr["cost"] > 1e-6:
                f_, g_ = B.kelly_opt(B.payoff_straddle(ST, pr["strikes"]) / pr["cost"] - 1.0)
                if g_ > 0:
                    struct, side, kelly_f = pr, "straddle", f_

        if struct is None:
            rows.append(dict(date=t, traded=0))
            continue

        cost, strikes = struct["cost"], struct["strikes"]
        val_fn = B.value_butterfly if side == "butterfly" else B.value_straddle
        exp_ts = pd.Timestamp(exp_date)
        cap_ts = exp_ts - pd.Timedelta(days=exit_dte) if exit_dte is not None else exp_ts

        # first-to-fire daily scan at FIXED entry IV (cheap, no per-day SSE)
        tgt = _target_value(side, cost, strikes, pt)
        stop_val = cost * (1.0 - stop) if stop is not None else None
        hit_date, hit_kind = None, None
        if tgt is not None or stop_val is not None:
            sd = etf.index[(etf.index > entry_date) & (etf.index <= cap_ts)
                           & (etf.index <= last_daily)]
            for d in sd:
                spot_d = float(etf.loc[d])
                T_d = max((exp_date - d.date()).days, 1) / 365.0
                m = val_fn(smile, spot_d, T_d, strikes)
                if tgt is not None and m >= tgt:
                    hit_date, hit_kind = d, "target"; break
                if stop_val is not None and m <= stop_val:
                    hit_date, hit_kind = d, "stop"; break

        if hit_date is not None:
            exit_kind, want = hit_kind, hit_date
        elif exit_dte is not None and entry_date < cap_ts < exp_ts:
            exit_kind, want = "roll", cap_ts
        else:
            exit_kind, want = "expiry", exp_ts

        # book exit: target/stop/roll -> real-IV mark; expiry -> intrinsic
        realized = None
        if exit_kind in ("target", "stop", "roll"):
            exit_tbl, ex_date = B.find_iv_on_or_before(want)
            if (exit_tbl is not None and ex_date > entry_date
                    and ex_date < exp_ts and ex_date <= last_daily):
                spot_ex = float(etf.asof(ex_date))
                smile_ex = B.otm_smile(exit_tbl, expiry, spot_ex)
                T_ex = max((exp_date - ex_date.date()).days, 1) / 365.0
                if len(smile_ex) >= 3:
                    realized = float(val_fn(smile_ex, spot_ex, T_ex, strikes))
            if realized is None:
                exit_kind = "expiry"

        if exit_kind == "expiry":
            S = float(etf.asof(exp_ts))
            pay_fn = B.payoff_butterfly if side == "butterfly" else B.payoff_straddle
            realized = float(pay_fn(np.array([S]), strikes)[0])

        r_trade = realized / cost - 1.0
        f_used = min(kelly_frac * kelly_f, kelly_cap)
        rows.append(dict(date=t, traded=1, side=side, exit_kind=exit_kind,
                         r_trade=r_trade, kelly_f=kelly_f, f_used=f_used,
                         contrib=f_used * r_trade))
    return pd.DataFrame(rows).set_index("date")

# =============================================================================
# Metrics helpers
# =============================================================================
def metrics_for(bt, df, y):
    bt2, _bh, m = B.equity_and_metrics(bt, df, y)
    mdd = m.get("rule_mdd", np.nan)
    m["calmar"] = m["rule_cagr"] / abs(mdd) if (mdd and np.isfinite(mdd) and mdd != 0) else np.nan
    return m

def resize_kelly(bt, kf, cap=KELLY_CAP):
    """Re-derive contrib/equity for a different KELLY_FRAC from stored kelly_f."""
    out = bt.copy()
    kf_col = out.get("kelly_f")
    r_col = out.get("r_trade")
    if kf_col is None or r_col is None:
        return out
    f_used = np.minimum(kf * kf_col.fillna(0.0), cap)
    out["f_used"] = f_used
    out["contrib"] = (f_used * r_col.fillna(0.0))
    return out

# =============================================================================
# Main: grid + heatmap + separate Kelly sweep
# =============================================================================
def main():
    print("=" * 72)
    print(f"GRID  {GATE_PARAM} x PROFIT_TAKE   objective={OBJECTIVE}")
    print(f"  fixed: HI={FIXED_HI} LO={FIXED_LO} EXIT_DTE={FIXED_EXIT_DTE} "
          f"STOP={FIXED_STOP} KELLY_FRAC={KELLY_FRAC}")
    print("=" * 72)

    contexts, etf, last_daily, X, y, df = build_contexts()

    results = []
    bt_cache = {}                                  # (gate, pt) -> bt for re-use
    for gate in GATE_GRID:
        hi = gate if GATE_PARAM == "HI_RATIO" else FIXED_HI
        lo = gate if GATE_PARAM == "LO_RATIO" else FIXED_LO
        for pt in PT_GRID:
            bt = evaluate(contexts, etf, last_daily, hi, lo, pt)
            bt_cache[(gate, pt)] = bt
            m = metrics_for(bt, df, y)
            results.append(dict(gate=gate, pt=pt, trades=m["trades"],
                                cagr=m["rule_cagr"], mdd=m["rule_mdd"],
                                calmar=m["calmar"], winrate=m["trade_winrate"],
                                total=m["rule_total"]))
            print(f"  {GATE_PARAM}={gate:<5} PT={pt:<4}  trades={m['trades']:<3} "
                  f"cagr={m['rule_cagr']:+.3f} mdd={m['rule_mdd']:+.3f} "
                  f"calmar={m['calmar']:+.2f} win={m['trade_winrate']:.2f}")
    res = pd.DataFrame(results)
    res.to_csv(OUTDIR / "param_search_grid.csv", index=False)

    rank_col = "calmar" if OBJECTIVE == "calmar" else "cagr"
    res_ranked = res.sort_values(rank_col, ascending=False)
    print("\n--- top cells by", rank_col, "---")
    print(res_ranked.head(8).to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    # heatmap of the objective
    piv = res.pivot(index="pt", columns="gate", values=rank_col)
    fig, ax = plt.subplots(figsize=(1.4 * len(GATE_GRID) + 2, 1.0 * len(PT_GRID) + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
    ax.set_xlabel(GATE_PARAM); ax.set_ylabel("PROFIT_TAKE")
    ax.set_title(f"{rank_col} heatmap ({GATE_PARAM} x PROFIT_TAKE)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=rank_col)
    fig.tight_layout(); fig.savefig(OUTDIR / "param_search_heatmap.png", dpi=130)
    plt.close(fig)

    # ---- separate Kelly sweep on the best cell (no re-run) ----
    best = res_ranked.iloc[0]
    best_bt = bt_cache[(best["gate"], best["pt"])]
    print(f"\n--- Kelly sweep on best cell "
          f"({GATE_PARAM}={best['gate']}, PT={best['pt']}) ---")
    krows = []
    for kf in KELLY_GRID:
        m = metrics_for(resize_kelly(best_bt, kf), df, y)
        krows.append(dict(kelly_frac=kf, cagr=m["rule_cagr"], mdd=m["rule_mdd"],
                          calmar=m["calmar"], total=m["rule_total"]))
        print(f"  KELLY_FRAC={kf:<4} cagr={m['rule_cagr']:+.3f} "
              f"mdd={m['rule_mdd']:+.3f} calmar={m['calmar']:+.2f}")
    pd.DataFrame(krows).to_csv(OUTDIR / "param_search_kelly_sweep.csv", index=False)

    _flush_cache()
    print(f"\nSaved: param_search_grid.csv, param_search_heatmap.png, "
          f"param_search_kelly_sweep.csv  (in {OUTDIR})")
    print("\nREGRESSION CHECK: set GATE_GRID=[<your LO>], PT_GRID=[<your PROFIT_TAKE>],")
    print("FIXED_HI/LO/EXIT_DTE/STOP to your last single-run config; the printed")
    print("trades/cagr/mdd for that one cell should match your last backtest run.")
    print("\nCaveat: ~29 trades / 8 winners -> treat the heatmap as a robustness")
    print("map, not a single optimum. Prefer a broad stable region over a lone peak.")


if __name__ == "__main__":
    main()
