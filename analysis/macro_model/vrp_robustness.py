# -*- coding: utf-8 -*-
"""
vrp_robustness.py
=================
Stress-test the validated-but-fragile result from vrp_harvest_compare.py:
the model-gated, fixed-size, NARROW iron condor showed Calmar ~0.22 with
corr(gap, r_trade) ~ +0.4 -- but on only ~10 trades, and the condor jumped
from ~0 to +0.22 purely by narrowing the strikes. Before trusting it (or
spending effort expanding the universe), check it isn't a knife-edge:

  1. GEOMETRY SWEEP  -- Calmar / worst-month / signal-corr across a grid of
     (SHORT_STEPS, WING_STEPS). A real edge is a broad green region, not one
     bright cell.
  2. GATE SWEEP      -- vary GAP_MIN; does the edge degrade gracefully or only
     exist at one threshold?
  3. STATS + TAIL    -- Pearson r WITH p-value/CI for corr(gap, r_trade), and
     the worst single-month r_trade (short-gamma tail).

Reuses vrp_harvest_compare.evaluate by overriding its module globals (same
trick as the param search). Reuses the warm IV cache -> little/no network.

Set STRUCT below to robustness-check a different structure.
RUN IN A NORMAL PYTHON ENV.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

import butterfly_timing_backtest as B
import butterfly_param_search as PS
import vrp_harvest_compare as VH

OUTDIR = B.OUTDIR
STRUCT = "iron_condor"          # structure to stress (iron_condor / iron_fly / butterfly)

# geometry grid (only SHORT < WING is valid for a condor)
WING_GRID  = [2, 3, 4, 5]
SHORT_GRID = [1, 2, 3, 4]
GAP_GRID   = [-0.04, -0.02, 0.00, 0.02, 0.04]   # vol points

# fixed during the sweeps (mirror the comparison run)
VH.SIZING_MODE = "structural"
VH.PROFIT_TAKE = 0.50
VH.EXIT_DTE    = 7
VH.STOP_FRAC   = None
VH.TARGET_RISK = 0.10

def _calmar(m):
    mdd = m["rule_mdd"]
    return m["rule_cagr"] / abs(mdd) if (mdd and np.isfinite(mdd) and mdd != 0) else np.nan

def _run(contexts, etf, last_daily, df, y):
    """One evaluate at the CURRENT VH globals -> metrics + signal stats + tail."""
    bt, funnel = VH.evaluate(STRUCT, contexts, etf, last_daily)
    _bt2, _bh, m = B.equity_and_metrics(bt, df, y)
    tr = bt[bt.get("traded", 0) == 1]
    n = len(tr)
    r = p = worst = np.nan
    if n >= 4 and {"gap", "r_trade"}.issubset(tr.columns):
        g = tr["gap"].to_numpy(float); rt = tr["r_trade"].to_numpy(float)
        ok = np.isfinite(g) & np.isfinite(rt)
        if ok.sum() >= 4:
            r, p = pearsonr(g[ok], rt[ok])
    if n >= 1 and "r_trade" in tr.columns:
        worst = float(tr["r_trade"].min())
    return dict(trades=n, calmar=_calmar(m), cagr=m["rule_cagr"],
                mdd=m["rule_mdd"], winrate=m["trade_winrate"],
                corr_gap=r, corr_p=p, worst_month=worst)

def main():
    print("=" * 72)
    print(f"ROBUSTNESS  struct={STRUCT}  (structural sizing, real IV)")
    print("=" * 72)
    contexts, etf, last_daily, X, y, df = PS.build_contexts()

    # ---- 1) geometry sweep ----
    print("\n--- 1) GEOMETRY SWEEP (SHORT_STEPS x WING_STEPS) ---")
    VH.GAP_MIN = 0.00
    geo_rows = []
    for ws in WING_GRID:
        for ss in SHORT_GRID:
            if ss >= ws:
                continue
            VH.WING_STEPS, VH.SHORT_STEPS = ws, ss
            res = _run(contexts, etf, last_daily, df, y)
            res.update(wing=ws, short=ss)
            geo_rows.append(res)
            print(f"  short={ss} wing={ws}: trades={res['trades']:<3} "
                  f"calmar={res['calmar']:+.2f} cagr={res['cagr']:+.3f} "
                  f"mdd={res['mdd']:+.3f} corr_gap={res['corr_gap']:+.2f} "
                  f"(p={res['corr_p']:.2f}) worst={res['worst_month']:+.2f}")
    geo = pd.DataFrame(geo_rows)
    geo.to_csv(OUTDIR / f"robust_{STRUCT}_geometry.csv", index=False)

    # heatmap of Calmar over geometry
    piv = geo.pivot(index="short", columns="wing", values="calmar")
    fig, ax = plt.subplots(figsize=(1.3 * len(WING_GRID) + 2, 1.0 * len(SHORT_GRID) + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
    ax.set_xlabel("WING_STEPS"); ax.set_ylabel("SHORT_STEPS")
    ax.set_title(f"{STRUCT} Calmar over geometry (want a broad green region)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Calmar")
    fig.tight_layout(); fig.savefig(OUTDIR / f"robust_{STRUCT}_geometry.png", dpi=130)
    plt.close(fig)

    # pick the most ROBUST geometry: best median-ish cell with >=8 trades
    cand = geo[(geo["trades"] >= 8) & np.isfinite(geo["calmar"])]
    base = cand.sort_values("calmar", ascending=False).iloc[0] if len(cand) else geo.iloc[0]
    VH.WING_STEPS, VH.SHORT_STEPS = int(base["wing"]), int(base["short"])
    print(f"\n  using geometry short={VH.SHORT_STEPS} wing={VH.WING_STEPS} for the gate sweep")

    # ---- 2) gate sweep ----
    print("\n--- 2) GATE SWEEP (GAP_MIN) ---")
    gap_rows = []
    for gm in GAP_GRID:
        VH.GAP_MIN = gm
        res = _run(contexts, etf, last_daily, df, y); res["gap_min"] = gm
        gap_rows.append(res)
        print(f"  GAP_MIN={gm:+.2f}: trades={res['trades']:<3} calmar={res['calmar']:+.2f} "
              f"cagr={res['cagr']:+.3f} corr_gap={res['corr_gap']:+.2f} "
              f"(p={res['corr_p']:.2f}) worst={res['worst_month']:+.2f}")
    pd.DataFrame(gap_rows).to_csv(OUTDIR / f"robust_{STRUCT}_gate.csv", index=False)

    print("\nHow to read it:")
    print(" * geometry heatmap GREEN over many cells -> edge is real, not a knife-edge;")
    print("   a lone bright cell surrounded by red -> overfit, distrust it.")
    print(" * corr_gap staying positive across geometry/gate -> the model signal is")
    print("   robust; p-values are large at n~10, so judge by CONSISTENCY not one p.")
    print(" * worst_month is the short-gamma tail you must survive; size for THAT.")
    print(f"\nSaved: robust_{STRUCT}_geometry.csv/.png, robust_{STRUCT}_gate.csv (in {OUTDIR})")


if __name__ == "__main__":
    main()
