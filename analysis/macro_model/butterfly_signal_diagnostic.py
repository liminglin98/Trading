# -*- coding: utf-8 -*-
"""
butterfly_signal_diagnostic.py
==============================
Before rebuilding the model, MEASURE whether the signal has any edge. Three
decisive checks, all on data you already have (reuses the warm IV cache from
butterfly_param_search, so almost no new network calls):

  1. VOL-FORECAST HORSE RACE. The strategy compares ATM implied to `model_ann`
     (= std of the factor model's next-month RETURN draws, annualized). Does
     model_ann actually predict next-month REALIZED vol? Race it against a
     trailing-21d realized-vol baseline and against implied itself. If model_ann
     is ~uncorrelated while trailing-RV / implied are not, the vol forecast is
     the weak link.

  2. VARIANCE RISK PREMIUM. How often, and by how much, does implied exceed the
     realized vol that follows? A persistent implied>realized gap is a
     harvestable short-vol premium that needs NO model at all.

  3. SIGNAL -> P&L. Using a gate-free long ATM straddle each month (held to
     expiry), does the model ratio (implied/model_ann) sort trades? Compare it
     to a trailing-RV signal (implied - trailing_rv). A useful "buy vol when
     cheap" signal should be NEGATIVELY correlated with straddle P&L (low
     signal -> big straddle gain). ~0 means the signal is noise.

Outputs a printed report, a per-month panel CSV, and a 2x2 scatter figure.

RUN IN A NORMAL PYTHON ENV. First run of butterfly_param_search should have
warmed iv_table_cache.pkl; otherwise this will fetch (needs internet).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import butterfly_timing_backtest as B
import butterfly_param_search as PS      # importing reuses cache + monkeypatch

OUTDIR = B.OUTDIR

# =============================================================================
# Realized-vol helpers (annualized, from daily 50ETF log returns)
# =============================================================================
def hold_realized_vol(etf, entry_date, exp_ts):
    """Realized vol over the holding window (entry, expiry] -- the OUTCOME."""
    w = etf[(etf.index >= entry_date) & (etf.index <= exp_ts)]
    if len(w) < 6:
        return np.nan
    lr = np.log(w).diff().dropna()
    return float(lr.std(ddof=1) * np.sqrt(252))

def trailing_realized_vol(etf, entry_date, n=22):
    """Realized vol over the n trading days up to entry -- KNOWN at entry."""
    w = etf[etf.index <= entry_date].tail(n)
    if len(w) < 10:
        return np.nan
    lr = np.log(w).diff().dropna()
    return float(lr.std(ddof=1) * np.sqrt(252))

# =============================================================================
# Tiny OLS with t-stats (no statsmodels dependency)
# =============================================================================
def ols(y, cols: dict):
    """cols: {name: array}. Adds an intercept. Returns (DataFrame of beta/t, R2)."""
    names = list(cols.keys())
    Xmat = np.column_stack([np.ones(len(y))] + [np.asarray(cols[n], float) for n in names])
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
    resid = y - Xmat @ beta
    n, k = Xmat.shape
    dof = max(n - k, 1)
    s2 = float(resid @ resid) / dof
    XtX_inv = np.linalg.inv(Xmat.T @ Xmat)
    se = np.sqrt(np.diag(s2 * XtX_inv))
    tstat = beta / se
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    tab = pd.DataFrame({"beta": beta, "t": tstat}, index=["const"] + names)
    return tab, r2

def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])

# =============================================================================
# Build the per-month diagnostic panel
# =============================================================================
def build_panel():
    contexts, etf, last_daily, X, y, df = PS.build_contexts()
    rows = []
    for c in contexts:
        smile, spot, T = c["smile"], c["spot"], c["T"]
        entry_date, exp_date = c["entry_date"], c["exp_date"]
        exp_ts = pd.Timestamp(exp_date)
        implied = c["atm_iv"]
        model_ann = c["model_ann"]

        rv_hold = hold_realized_vol(etf, entry_date, exp_ts)
        rv_trail = trailing_realized_vol(etf, entry_date)

        # gate-free long ATM straddle, realized at expiry (intrinsic)
        r_straddle = np.nan
        pr = B.price_straddle(smile, spot, T)
        if pr["cost"] > 1e-6:
            S_exp = float(etf.asof(exp_ts))
            payoff = float(B.payoff_straddle(np.array([S_exp]), pr["strikes"])[0])
            r_straddle = payoff / pr["cost"] - 1.0

        rows.append(dict(
            date=c["t"], implied=implied, model_ann=model_ann,
            ratio=implied / model_ann if model_ann > 0 else np.nan,
            rv_hold=rv_hold, rv_trail=rv_trail,
            vrp_real=implied - rv_hold,            # premium actually earned (outcome)
            vrp_proxy=implied - rv_trail,          # tradable VRP estimate (entry-known)
            r_straddle=r_straddle,
        ))
    panel = pd.DataFrame(rows).set_index("date")
    panel.to_csv(OUTDIR / "signal_diagnostic_panel.csv")
    return panel

# =============================================================================
# Report
# =============================================================================
def main():
    print("=" * 72)
    print("SIGNAL DIAGNOSTIC  (is the edge in the forecast, the VRP, or nowhere?)")
    print("=" * 72)
    p = build_panel()
    n = len(p)
    print(f"\nmonths in panel: {n}")

    # ---------- 1) vol-forecast horse race ----------
    print("\n--- 1) FORECAST HORSE RACE: predict next-month realized vol ---")
    print("  Pearson corr with realized vol over the holding month (rv_hold):")
    print(f"    model_ann   : {corr(p['model_ann'], p['rv_hold']):+.3f}   (the factor-model forecast)")
    print(f"    trailing RV : {corr(p['rv_trail'], p['rv_hold']):+.3f}   (last 21d realized, free baseline)")
    print(f"    implied     : {corr(p['implied'],  p['rv_hold']):+.3f}   (ATM IV at entry)")
    sub = p.dropna(subset=["rv_hold", "model_ann", "rv_trail", "implied"])
    tab, r2 = ols(sub["rv_hold"], {"model_ann": sub["model_ann"],
                                   "trailing_rv": sub["rv_trail"],
                                   "implied": sub["implied"]})
    print(f"\n  Multiple OLS  rv_hold ~ model_ann + trailing_rv + implied   (R2={r2:.3f}):")
    print(tab.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("  -> if model_ann's t is small while trailing_rv/implied carry the R2,")
    print("     the macro forecast adds little over free baselines.")

    # ---------- 2) variance risk premium ----------
    print("\n--- 2) VARIANCE RISK PREMIUM (implied vs subsequent realized) ---")
    vrp = p["vrp_real"].dropna()
    print(f"    mean implied        : {p['implied'].mean():.3f}")
    print(f"    mean realized       : {p['rv_hold'].mean():.3f}")
    print(f"    mean (implied-real) : {vrp.mean():+.3f}")
    print(f"    pct months implied>real: {(vrp > 0).mean()*100:.0f}%")
    print("  -> a persistent positive gap is a short-vol premium harvestable")
    print("     without any model (sell vol structurally, size by the gap).")

    # ---------- 3) signal -> straddle P&L ----------
    print("\n--- 3) DOES THE ENTRY SIGNAL SORT TRADES? (gate-free long straddle) ---")
    print("  corr of entry-known signal with straddle return (want strong NEGATIVE):")
    print(f"    ratio = implied/model_ann   : {corr(p['ratio'],     p['r_straddle']):+.3f}   (current signal)")
    print(f"    implied - model_ann         : {corr(p['implied']-p['model_ann'], p['r_straddle']):+.3f}")
    print(f"    implied - trailing_rv (VRP) : {corr(p['vrp_proxy'], p['r_straddle']):+.3f}   (alternative)")
    s2 = p.dropna(subset=["r_straddle", "ratio", "vrp_proxy"])
    _, r2_ratio = ols(s2["r_straddle"], {"ratio": s2["ratio"]})
    _, r2_vrp = ols(s2["r_straddle"], {"vrp_proxy": s2["vrp_proxy"]})
    print(f"\n  single-signal R2 explaining straddle P&L:  ratio={r2_ratio:.3f}   vrp_proxy={r2_vrp:.3f}")
    print("  -> whichever R2 is higher is the better entry signal; if both ~0,")
    print("     neither sorts trades and the bet is essentially blind.")

    # ---------- plots ----------
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    def sc(a, xcol, ycol, xl, yl):
        d = p.dropna(subset=[xcol, ycol])
        a.scatter(d[xcol], d[ycol], s=18, alpha=0.7, color="#1f4e79")
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3)
    sc(ax[0, 0], "model_ann", "rv_hold", "model_ann (forecast)", "realized vol")
    ax[0, 0].set_title("forecast vs realized (want tight 45-line)")
    sc(ax[0, 1], "rv_trail", "rv_hold", "trailing 21d RV", "realized vol")
    ax[0, 1].set_title("free baseline vs realized")
    sc(ax[1, 0], "ratio", "r_straddle", "ratio = implied/model_ann", "straddle return")
    ax[1, 0].set_title("current signal vs trade P&L")
    sc(ax[1, 1], "vrp_proxy", "r_straddle", "implied - trailing_rv", "straddle return")
    ax[1, 1].set_title("VRP signal vs trade P&L")
    fig.tight_layout()
    fig.savefig(OUTDIR / "signal_diagnostic.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved: signal_diagnostic_panel.csv, signal_diagnostic.png  (in {OUTDIR})")
    print("\nReading it: weak model_ann corr + positive VRP + higher vrp_proxy R2")
    print("would say 'replace the forecast with a vol model / trade the VRP'.")


if __name__ == "__main__":
    main()
