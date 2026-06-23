# -*- coding: utf-8 -*-
"""
butterfly_wing_exit_optimizer.py
================================
OPERATIONAL wing-sizing + exit-management tool for the monthly 50ETF long
butterfly (the "normal-regime" position: body at spot, symmetric wings).

Context (settled): the factor/macro model has NO vol-timing alpha (level,
term-structure and skew all tested NULL). Its ONE validated, durable use is the
calibrated posterior-predictive DISPERSION BAND -> how wide to set the wings.
So this tool is NOT an alpha search. It does two things:

  1. WING x EXIT ROBUSTNESS MAP. Always-on long ATM butterfly, priced against
     the REAL listed SSE chain, walk-forward, with first-to-fire exits. Grid over
        wing width (grid-steps)  x  profit target  x  stop loss  x  exit-DTE.
     Ranked by Calmar (CAGR / |MDD|), but worst single-month r_trade and a
     buy & hold benchmark are printed for EVERY cell. Read it as a robustness
     map (prefer a broad stable region), NOT a single optimum -- the sample is
     ~49 months and the long fly is ~breakeven on the real chain.

  2. MODEL-BAND WING + LIVE SPEC. Computes the wing width implied by the model's
     own 80% posterior-predictive band (the principled, non-overfit choice) and
     compares it to the backtest-robust width. Then prints a concrete CURRENT-
     MONTH trade spec: body strike, wing strikes (mapped to nearest LISTED
     strikes), the chosen profit-target / stop / exit-DTE rule, and position size.

Discipline reused from the rest of the project:
  * real chain pricing (SSE IV via the warm cache; no headline-vol shortcuts),
  * no look-ahead (model trained <= t; entry at t; exits dated > t),
  * judge by Calmar + worst-month + vs buy & hold, not raw return.

RUN IN A NORMAL PYTHON ENV from inside macro_model\ (uses out/iv_table_cache.pkl;
uncached dates fetched from SSE -> needs internet).
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
import butterfly_param_search as PS      # build_contexts + warm disk IV cache

OUTDIR = B.OUTDIR

# =============================================================================
# GRIDS (edit freely; kept modest to stay a robustness map, not a peak hunt)
# =============================================================================
WING_GRID   = [1, 2, 3, 4, 5]            # body half-width in strike-grid steps
PT_GRID     = [0.50, 0.75, 1.00, None]   # take profit at frac of MAX profit (None=off)
STOP_GRID   = [None, 0.50, 0.75]         # stop when mark <= cost*(1-stop) (None=off)
EXIT_DTE_GRID = [7, 14, None]            # roll/close at this DTE (None=hold to expiry)

TARGET_RISK = 0.10                       # fixed bankroll fraction per trade (debit=CaR)
KELLY_CAP   = B.KELLY_CAP
BAND_Q      = 0.80                       # posterior band used for the principled wing
OBJECTIVE   = "calmar"                   # ranking metric

# =============================================================================
# Always-on long ATM butterfly, one (wing, pt, stop, exit_dte) cell, walk-forward
# =============================================================================
def evaluate_fly(contexts, etf, last_daily, wing, pt, stop, exit_dte):
    rows = []
    for c in contexts:
        smile, spot, T = c["smile"], c["spot"], c["T"]
        expiry, exp_date, entry_date = c["expiry"], c["exp_date"], c["entry_date"]
        t = c["t"]

        pr = B.price_butterfly(smile, spot, T, wing)
        if not pr or pr["cost"] <= 1e-6:
            rows.append(dict(date=t, traded=0)); continue
        strikes, cost = pr["strikes"], pr["cost"]
        klow, kmid, khigh = strikes
        max_profit = (kmid - klow) - cost            # long fly max payoff = wing width
        if max_profit <= 0:
            rows.append(dict(date=t, traded=0)); continue

        car = cost                                   # debit structure: risk = premium paid
        f_used = min(TARGET_RISK, KELLY_CAP)
        exp_ts = pd.Timestamp(exp_date)
        cap_ts = exp_ts - pd.Timedelta(days=exit_dte) if exit_dte is not None else exp_ts

        tgt_val = cost + pt * max_profit if pt is not None else None
        stop_val = cost * (1.0 - stop) if stop is not None else None

        # first-to-fire daily scan at FIXED entry IV (cheap, no per-day SSE)
        hit_date, hit_kind = None, None
        if tgt_val is not None or stop_val is not None:
            sd = etf.index[(etf.index > entry_date) & (etf.index <= cap_ts)
                           & (etf.index <= last_daily)]
            for d in sd:
                spot_d = float(etf.loc[d])
                T_d = max((exp_date - d.date()).days, 1) / 365.0
                mark = B.value_butterfly(smile, spot_d, T_d, strikes)
                if tgt_val is not None and mark >= tgt_val:
                    hit_date, hit_kind = d, "target"; break
                if stop_val is not None and mark <= stop_val:
                    hit_date, hit_kind = d, "stop"; break

        if hit_date is not None:
            exit_kind, want = hit_kind, hit_date
        elif exit_dte is not None and entry_date < cap_ts < exp_ts:
            exit_kind, want = "roll", cap_ts
        else:
            exit_kind, want = "expiry", exp_ts

        realized = None
        if exit_kind in ("target", "stop", "roll"):
            tbl, ex_date = B.find_iv_on_or_before(want)
            if (tbl is not None and ex_date > entry_date
                    and ex_date < exp_ts and ex_date <= last_daily):
                spot_ex = float(etf.asof(ex_date))
                smile_ex = B.otm_smile(tbl, expiry, spot_ex)
                T_ex = max((exp_date - ex_date.date()).days, 1) / 365.0
                if len(smile_ex) >= 3:
                    realized = float(B.value_butterfly(smile_ex, spot_ex, T_ex, strikes))
            if realized is None:
                exit_kind = "expiry"
        if exit_kind == "expiry":
            S = float(etf.asof(exp_ts))
            realized = float(B.payoff_butterfly(np.array([S]), strikes)[0])

        r_trade = realized / cost - 1.0
        rows.append(dict(date=t, traded=1, exit_kind=exit_kind, r_trade=r_trade,
                         f_used=f_used, contrib=f_used * r_trade))
    return pd.DataFrame(rows).set_index("date")

def _metrics(bt, df, y):
    bt2, _bh, m = B.equity_and_metrics(bt, df, y)
    mdd = m.get("rule_mdd", np.nan)
    m["calmar"] = m["rule_cagr"] / abs(mdd) if (mdd and np.isfinite(mdd) and mdd != 0) else np.nan
    tr = bt[bt.get("traded", 0) == 1]
    m["worst_month"] = float(tr["r_trade"].min()) if len(tr) else np.nan
    return m

# =============================================================================
# Model-band-implied wing width (the principled, non-overfit choice)
# =============================================================================
def band_wing_steps(c):
    """Wing width (in strike-grid steps) implied by the model's BAND_Q posterior
    band of the terminal price, for one month's context. body=spot."""
    ST = np.asarray(c["ST"], float)
    spot = c["spot"]
    half = np.quantile(np.abs(ST / spot - 1.0), BAND_Q)        # fractional half-width
    ks = c["smile"].index.values.astype(float)
    if len(ks) < 3:
        return np.nan
    step = float(np.median(np.diff(np.sort(ks))))             # listed strike spacing
    if step <= 0:
        return np.nan
    return float(spot * half / step)

def model_band_wing(contexts):
    """Median model-band wing across the backtest months, snapped to WING_GRID.
    The single 'principled' operational wing (shared by the app + the tracker)."""
    bw = np.array([band_wing_steps(c) for c in contexts], float)
    bw = bw[np.isfinite(bw)]
    steps = float(np.median(bw)) if len(bw) else np.nan
    if not np.isfinite(steps):
        return WING_GRID[len(WING_GRID) // 2]
    return int(np.clip(round(steps), min(WING_GRID), max(WING_GRID)))

# =============================================================================
# Forward LIVE context: fit model on all available data + pull the MOST RECENT
# real chain (future expiry, no realized P&L). This is the tradeable "today"
# context -- distinct from the backtest contexts, which require a SETTLED expiry
# and therefore stop at the last month whose option has already expired.
# =============================================================================
def build_live_context(etf, X, y, asof=None):
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp(pd.Timestamp.now().date())
    # most recent forward dispersion band from the latest known predictors
    N = len(X)
    draws = B.fit_predict_conjugate(X.values[:N], y.values[:N], X.values[N - 1:N])[0]
    model_ann = float(draws.std()) * np.sqrt(12)

    # most recent trading day with a real SSE chain for B.UNDERLYING
    iv_tbl, entry_date = None, None
    for back in range(0, 8):
        d = pd.Timestamp((asof - pd.Timedelta(days=back)).date())
        tbl = B.fetch_iv_table(d)
        if tbl is not None:
            iv_tbl, entry_date = tbl, d
            break
    if iv_tbl is None:
        return None
    spot = float(etf.asof(entry_date))
    if not np.isfinite(spot):
        return None
    expiry, exp_date = B.choose_expiry(iv_tbl, entry_date)   # future expiry OK for live
    if expiry is None:
        return None
    T = max((exp_date - entry_date.date()).days, 1) / 365.0
    smile = B.otm_smile(iv_tbl, expiry, spot)
    if len(smile) < 5:
        return None
    atm_iv = float(smile.iloc[(smile.index - spot).to_series().abs().values.argmin()])
    Tm = T * 12.0
    d_h = draws.mean() * Tm + (draws - draws.mean()) * np.sqrt(Tm)
    ST = spot * np.exp(d_h)
    return dict(t=entry_date, entry_date=entry_date, spot=spot, expiry=expiry,
                exp_date=exp_date, T=T, smile=smile, ST=ST,
                atm_iv=atm_iv, model_ann=model_ann)

# =============================================================================
# Live current-month trade spec
# =============================================================================
def live_spec(c, wing, pt, stop, exit_dte):
    smile, spot, T = c["smile"], c["spot"], c["T"]
    ks = smile.index.values.astype(float)
    i0 = int(np.argmin(np.abs(ks - spot)))
    lo, hi = i0 - wing, i0 + wing
    if lo < 0 or hi >= len(ks):
        return None
    klow, kmid, khigh = ks[lo], ks[i0], ks[hi]
    pr = B.price_butterfly(smile, spot, T, wing)
    cost = pr["cost"] if pr else np.nan
    max_profit = (kmid - klow) - cost if np.isfinite(cost) else np.nan
    band_lo = float(spot * np.quantile(c["ST"] / spot, (1 - BAND_Q) / 2))
    band_hi = float(spot * np.quantile(c["ST"] / spot, 1 - (1 - BAND_Q) / 2))
    return dict(entry_date=c["entry_date"], expiry=c["expiry"], exp_date=c["exp_date"],
                spot=spot, body=kmid, wing_low=klow, wing_high=khigh,
                cost=cost, max_profit=max_profit, dte=int(round(T * 365)),
                band_lo=band_lo, band_hi=band_hi, pt=pt, stop=stop, exit_dte=exit_dte)

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 78)
    print("BUTTERFLY WING + EXIT OPTIMIZER  (operational sizing tool, 50ETF long fly)")
    print(f"  grid: wing{WING_GRID} x PT{PT_GRID} x STOP{STOP_GRID} x EXIT_DTE{EXIT_DTE_GRID}")
    print(f"  always-on (body=spot), real chain, walk-forward, size={TARGET_RISK}, rank={OBJECTIVE}")
    print("=" * 78)

    contexts, etf, last_daily, X, y, df = PS.build_contexts()
    print(f"contexts: {len(contexts)} months "
          f"({contexts[0]['t'].date()} .. {contexts[-1]['t'].date()})")

    # buy & hold reference (computed once via any cell's metrics)
    rows = []
    for wing in WING_GRID:
        for pt in PT_GRID:
            for stop in STOP_GRID:
                for ed in EXIT_DTE_GRID:
                    bt = evaluate_fly(contexts, etf, last_daily, wing, pt, stop, ed)
                    m = _metrics(bt, df, y)
                    rows.append(dict(wing=wing, pt=pt, stop=stop, exit_dte=ed,
                                     trades=m["trades"], cagr=m["rule_cagr"],
                                     mdd=m["rule_mdd"], calmar=m["calmar"],
                                     winrate=m["trade_winrate"], worst_month=m["worst_month"],
                                     total=m["rule_total"],
                                     bh_cagr=m["bh_cagr"], bh_mdd=m["bh_mdd"]))
    res = pd.DataFrame(rows)
    res.to_csv(OUTDIR / "wing_exit_grid.csv", index=False)

    bh_cagr = res["bh_cagr"].iloc[0]; bh_mdd = res["bh_mdd"].iloc[0]
    bh_calmar = bh_cagr / abs(bh_mdd) if (bh_mdd and np.isfinite(bh_mdd) and bh_mdd != 0) else np.nan

    rank = "calmar" if OBJECTIVE == "calmar" else "cagr"
    top = res.sort_values(rank, ascending=False).head(12)
    print(f"\n--- top cells by {rank}  (buy & hold: cagr={bh_cagr:+.3f} mdd={bh_mdd:+.3f} "
          f"calmar={bh_calmar:+.2f}) ---")
    show = ["wing", "pt", "stop", "exit_dte", "trades", "cagr", "mdd", "calmar",
            "winrate", "worst_month"]
    print(top[show].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    beats = res[res["calmar"] > bh_calmar]
    print(f"\n  cells beating buy & hold Calmar: {len(beats)}/{len(res)}")
    print("  (long fly is ~breakeven on the real chain -- if few/none beat b&h, the")
    print("   wings are about RISK SHAPE, not return; pick a stable region, not the peak.)")

    # ---- robustness: average Calmar by wing width (which width is stably ok?) ----
    by_wing = res.groupby("wing")["calmar"].agg(["mean", "median", "max"])
    print("\n--- Calmar by wing width (stability across PT/STOP/EXIT) ---")
    print(by_wing.to_string(float_format=lambda v: f"{v:+.3f}"))

    # ---- model-band-implied wing (the principled choice) ----
    bw = np.array([band_wing_steps(c) for c in contexts], float)
    bw = bw[np.isfinite(bw)]
    band_steps = float(np.median(bw)) if len(bw) else np.nan
    print(f"\n--- model-band wing ({int(BAND_Q*100)}% posterior) ---")
    print(f"  implied wing width = {band_steps:.2f} strike-steps (median across months)")
    band_wing_int = int(np.clip(round(band_steps), min(WING_GRID), max(WING_GRID))) \
        if np.isfinite(band_steps) else WING_GRID[len(WING_GRID)//2]
    print(f"  -> nearest grid wing = {band_wing_int} steps  (principled, non-overfit width)")

    # ---- choose an operational config: band wing, best exit rule AT that wing ----
    at_band = res[res["wing"] == band_wing_int].sort_values(rank, ascending=False)
    best = at_band.iloc[0] if len(at_band) else res.sort_values(rank, ascending=False).iloc[0]
    print(f"\n--- chosen operational config (wing fixed to model band, best exit there) ---")
    print(f"  wing={int(best['wing'])}  PT={best['pt']}  STOP={best['stop']}  "
          f"EXIT_DTE={best['exit_dte']}  ->  calmar={best['calmar']:+.2f} "
          f"cagr={best['cagr']:+.3f} mdd={best['mdd']:+.3f} worst={best['worst_month']:+.3f}")

    # ---- live spec: build a FORWARD context from the most recent real chain ----
    live_ctx, live_src = None, "BACKTEST FALLBACK (last settled month)"
    try:
        live_ctx = build_live_context(etf, X, y)
        if live_ctx is not None:
            live_src = "LIVE (most recent chain as of run date)"
    except Exception as e:
        print(f"\n  [live context fetch failed: {e} -- falling back to last backtest month]")
    src = live_ctx if live_ctx is not None else contexts[-1]
    spec = live_spec(src, int(best["wing"]), best["pt"], best["stop"], best["exit_dte"])
    print("\n" + "=" * 78)
    print("CURRENT-MONTH TRADE SPEC (long call butterfly, body = spot)")
    print(f"  source: {live_src}")
    print("=" * 78)
    if spec is None:
        print("  could not build (wing runs off the listed strike grid this month).")
    else:
        print(f"  as of entry   : {pd.Timestamp(spec['entry_date']).date()}  "
              f"(expiry {spec['expiry']} = {spec['exp_date']}, ~{spec['dte']} DTE)")
        print(f"  spot          : {spec['spot']:.3f}")
        print(f"  BUY  1x call  @ {spec['wing_low']:.3f}   (lower wing)")
        print(f"  SELL 2x call  @ {spec['body']:.3f}   (body = ATM)")
        print(f"  BUY  1x call  @ {spec['wing_high']:.3f}   (upper wing)")
        print(f"  net debit     : {spec['cost']:.4f}  per 1-lot  (= capital at risk)")
        print(f"  max profit    : {spec['max_profit']:.4f}  at body if pinned")
        print(f"  model {int(BAND_Q*100)}% band : [{spec['band_lo']:.3f}, {spec['band_hi']:.3f}]  "
              f"(wings ~ band edges by construction)")
        print(f"  position size : {TARGET_RISK:.0%} of bankroll at risk")
        print(f"  EXIT RULE     : take profit at {spec['pt']} of max | "
              f"stop at {spec['stop']} of debit | roll at {spec['exit_dte']} DTE "
              f"(first to fire; else hold to expiry)")
    print("=" * 78)

    # ---- heatmap: Calmar over wing x exit-DTE, averaged over PT/STOP ----
    piv = res.groupby(["wing", "exit_dte"], dropna=False)["calmar"].mean().reset_index()
    piv["exit_dte"] = piv["exit_dte"].astype("object").where(piv["exit_dte"].notna(), "hold")
    pv = piv.pivot(index="wing", columns="exit_dte", values="calmar")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pv.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index)));   ax.set_yticklabels(pv.index)
    ax.set_xlabel("exit DTE"); ax.set_ylabel("wing steps")
    ax.set_title("mean Calmar by wing x exit-DTE (avg over PT/STOP)")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Calmar")
    fig.tight_layout(); fig.savefig(OUTDIR / "wing_exit_heatmap.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved: wing_exit_grid.csv, wing_exit_heatmap.png  (in {OUTDIR})")
    print("Caveats: ~49 months, BS-priced from SSE IV (no bid/ask/fees/slippage),")
    print("expiry realized from actual 50ETF close. Long fly ~breakeven on the real")
    print("chain -> treat the grid as a RISK/structure map; the model-band wing is the")
    print("principled default, the exit rule trims tails. Benchmark every cell vs b&h.")


if __name__ == "__main__":
    main()
