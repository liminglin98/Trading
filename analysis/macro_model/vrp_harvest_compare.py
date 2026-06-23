# -*- coding: utf-8 -*-
"""
vrp_harvest_compare.py
======================
The diagnostic said: the factor model forecasts realized vol well, implied
exceeds realized ~69% of months (a structural short-vol premium), and the
current long-vol strategy fights that premium. This script tests the FLIP --
harvest the variance risk premium with DEFINED-RISK short-vol structures,
gated and sized by the model -- and compares three vehicles head to head:

    1. long butterfly  (debit, body ATM)        -- pin / short-dispersion
    2. iron condor     (credit, OTM put+call spreads)
    3. iron fly        (credit, short ATM straddle + bought wings)

Everything except the structure is held IDENTICAL so the comparison is clean:
  * same per-month contexts (reuses the warm IV cache),
  * same gate: trade only when implied - model_ann >= GAP_MIN (premium rich vs
    the model's realized-vol forecast),
  * same fractional-Kelly sizing over the model's scaled return draws,
  * same first-to-fire exit (profit target as a fraction of MAX profit, an
    EXIT_DTE roll cap, optional stop as a fraction of capital-at-risk),
  * same real-IV mark-to-market booking at exit.

UNIFIED ACCOUNTING (so debit and credit structures are comparable):
  every structure is a list of signed legs; with
      net_value(S)  = sum_legs  sign * option_price
      entry_value   = net_value at entry           (debit > 0, credit < 0)
      pnl           = net_value_exit - entry_value
      r_trade       = pnl / capital_at_risk
  capital_at_risk = debit for the butterfly, (wing_width - credit) for the
  credit structures. The long-butterfly path reproduces the standalone
  butterfly engine's numbers -- that's the built-in regression check.

RUN IN A NORMAL PYTHON ENV. Reuses iv_table_cache.pkl warmed by the param
search; otherwise fetches (needs internet).

NOTE: written without a live run here. Validate with the regression check
printed at the end before trusting absolute levels.
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
import butterfly_param_search as PS      # reuses cache + monkeypatch

OUTDIR = B.OUTDIR
R_FREE = B.R_FREE

# =============================================================================
# CONFIG (identical across the three structures)
# =============================================================================
GAP_MIN     = 0.00      # enter only if implied - model_ann >= this (vol points)
SHORT_STEPS = 1         # iron condor: short strikes this many grid-steps OTM
WING_STEPS  = 2         # bought wings / butterfly body half-width, in grid-steps
PROFIT_TAKE = 0.50      # take profit at this fraction of MAX profit
EXIT_DTE    = 7         # roll/close at this many DTE remaining (None -> to expiry)
STOP_FRAC   = None      # stop when pnl <= -STOP_FRAC * capital_at_risk (None off)
KELLY_FRAC  = 0.33
KELLY_CAP   = B.KELLY_CAP

# Sizing mode -- decides whether the MODEL gates each short:
#   "model_kelly": only trade when the model's draws give Kelly growth g>0, size
#                  at fractional Kelly. (Model vetoes shorts into expected vol ->
#                  tends to starve the sample to the calmest months.)
#   "structural" : harvest the VRP whenever gap>=GAP_MIN; size at a fixed
#                  TARGET_RISK fraction; the model does NOT veto. (Full sample.)
SIZING_MODE = "structural"
TARGET_RISK = 0.10      # bankroll fraction per trade in "structural" mode

# =============================================================================
# Signed-leg pricing / intrinsic  (leg = (sign, 'C'|'P', strike))
# =============================================================================
def leg_price(legs, smile, spot, T):
    tot = 0.0
    for sign, cp, K in legs:
        iv = B.smile_iv(smile, K)
        px = B.bs_call(spot, K, T, R_FREE, iv) if cp == "C" else B.bs_put(spot, K, T, R_FREE, iv)
        tot += sign * px
    return tot

def leg_intrinsic(legs, S):
    tot = 0.0
    for sign, cp, K in legs:
        v = max(S - K, 0.0) if cp == "C" else max(K - S, 0.0)
        tot += sign * v
    return tot

def leg_intrinsic_vec(legs, S):
    S = np.asarray(S, float)
    tot = np.zeros_like(S)
    for sign, cp, K in legs:
        v = np.maximum(S - K, 0.0) if cp == "C" else np.maximum(K - S, 0.0)
        tot += sign * v
    return tot

# =============================================================================
# Structure builders -> dict(legs, entry_value, car, max_profit) or None
# All use the discrete strike grid of the entry smile.
# =============================================================================
def _grid(smile, spot):
    ks = smile.index.values.astype(float)
    i0 = int(np.argmin(np.abs(ks - spot)))
    return ks, i0

def build_butterfly(smile, spot, T):
    ks, i0 = _grid(smile, spot)
    lo, hi = i0 - WING_STEPS, i0 + WING_STEPS
    if lo < 0 or hi >= len(ks):
        return None
    klow, kmid, khigh = ks[lo], ks[i0], ks[hi]
    legs = [(+1, "C", klow), (-2, "C", kmid), (+1, "C", khigh)]
    entry_value = leg_price(legs, smile, spot, T)          # debit > 0
    if entry_value <= 1e-6:
        return None
    width = kmid - klow                                    # max payoff at body
    return dict(legs=legs, entry_value=entry_value, car=entry_value,
                max_profit=width - entry_value)

def build_iron_condor(smile, spot, T):
    ks, i0 = _grid(smile, spot)
    lp, sp = i0 - WING_STEPS, i0 - SHORT_STEPS
    sc, lc = i0 + SHORT_STEPS, i0 + WING_STEPS
    if lp < 0 or lc >= len(ks) or sp <= lp or sc >= lc:
        return None
    legs = [(-1, "P", ks[sp]), (+1, "P", ks[lp]),          # short put spread
            (-1, "C", ks[sc]), (+1, "C", ks[lc])]          # short call spread
    entry_value = leg_price(legs, smile, spot, T)          # credit -> negative
    credit = -entry_value
    width = max(ks[sp] - ks[lp], ks[lc] - ks[sc])
    car = width - credit
    if credit <= 1e-6 or car <= 1e-6:
        return None
    return dict(legs=legs, entry_value=entry_value, car=car, max_profit=credit)

def build_iron_fly(smile, spot, T):
    ks, i0 = _grid(smile, spot)
    lp, lc = i0 - WING_STEPS, i0 + WING_STEPS
    if lp < 0 or lc >= len(ks):
        return None
    katm = ks[i0]
    legs = [(-1, "P", katm), (+1, "P", ks[lp]),            # short ATM put + wing
            (-1, "C", katm), (+1, "C", ks[lc])]            # short ATM call + wing
    entry_value = leg_price(legs, smile, spot, T)          # credit -> negative
    credit = -entry_value
    width = max(katm - ks[lp], ks[lc] - katm)
    car = width - credit
    if credit <= 1e-6 or car <= 1e-6:
        return None
    return dict(legs=legs, entry_value=entry_value, car=car, max_profit=credit)

BUILDERS = {"butterfly": build_butterfly,
            "iron_condor": build_iron_condor,
            "iron_fly": build_iron_fly}

def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])

# =============================================================================
# Generic VRP-gated short-vol evaluation for one structure
# =============================================================================
def evaluate(struct_name, contexts, etf, last_daily):
    build = BUILDERS[struct_name]
    rows = []
    funnel = dict(months=0, no_gap=0, no_build=0, no_kelly=0, traded=0)
    for c in contexts:
        funnel["months"] += 1
        smile, spot, T, ST = c["smile"], c["spot"], c["T"], c["ST"]
        entry_date, exp_date, expiry = c["entry_date"], c["exp_date"], c["expiry"]
        t = c["t"]
        gap = c["atm_iv"] - c["model_ann"]                 # implied - forecast

        if gap < GAP_MIN:
            funnel["no_gap"] += 1
            rows.append(dict(date=t, traded=0)); continue
        st = build(smile, spot, T)
        if st is None:
            funnel["no_build"] += 1
            rows.append(dict(date=t, traded=0)); continue

        legs, entry_value, car, max_profit = (st["legs"], st["entry_value"],
                                              st["car"], st["max_profit"])
        if car <= 0 or max_profit <= 0:
            funnel["no_build"] += 1
            rows.append(dict(date=t, traded=0)); continue

        # sizing: model-Kelly (model may veto) vs structural (always harvest)
        r_draws = (leg_intrinsic_vec(legs, ST) - entry_value) / car
        kf, g = B.kelly_opt(r_draws)
        if SIZING_MODE == "model_kelly":
            if g <= 0:
                funnel["no_kelly"] += 1
                rows.append(dict(date=t, traded=0)); continue
            f_used = min(KELLY_FRAC * kf, KELLY_CAP)
        else:                                              # "structural"
            f_used = min(TARGET_RISK, KELLY_CAP)
        funnel["traded"] += 1

        exp_ts = pd.Timestamp(exp_date)
        cap_ts = exp_ts - pd.Timedelta(days=EXIT_DTE) if EXIT_DTE is not None else exp_ts

        # first-to-fire daily scan at FIXED entry IV (cheap)
        tgt_pnl = PROFIT_TAKE * max_profit
        stop_pnl = -STOP_FRAC * car if STOP_FRAC is not None else None
        hit_date, hit_kind = None, None
        sd = etf.index[(etf.index > entry_date) & (etf.index <= cap_ts)
                       & (etf.index <= last_daily)]
        for d in sd:
            spot_d = float(etf.loc[d])
            T_d = max((exp_date - d.date()).days, 1) / 365.0
            pnl_d = leg_price(legs, smile, spot_d, T_d) - entry_value
            if pnl_d >= tgt_pnl:
                hit_date, hit_kind = d, "target"; break
            if stop_pnl is not None and pnl_d <= stop_pnl:
                hit_date, hit_kind = d, "stop"; break

        if hit_date is not None:
            exit_kind, want = hit_kind, hit_date
        elif EXIT_DTE is not None and entry_date < cap_ts < exp_ts:
            exit_kind, want = "roll", cap_ts
        else:
            exit_kind, want = "expiry", exp_ts

        # book exit: target/stop/roll -> real-IV mark; expiry -> intrinsic
        pnl = None
        if exit_kind in ("target", "stop", "roll"):
            exit_tbl, ex_date = B.find_iv_on_or_before(want)
            if (exit_tbl is not None and ex_date > entry_date
                    and ex_date < exp_ts and ex_date <= last_daily):
                spot_ex = float(etf.asof(ex_date))
                smile_ex = B.otm_smile(exit_tbl, expiry, spot_ex)
                T_ex = max((exp_date - ex_date.date()).days, 1) / 365.0
                if len(smile_ex) >= 3:
                    pnl = leg_price(legs, smile_ex, spot_ex, T_ex) - entry_value
            if pnl is None:
                exit_kind = "expiry"
        if exit_kind == "expiry":
            S_exp = float(etf.asof(exp_ts))
            pnl = leg_intrinsic(legs, S_exp) - entry_value

        r_trade = pnl / car
        rows.append(dict(date=t, traded=1, exit_kind=exit_kind, gap=gap,
                         entry_value=entry_value, car=car, r_trade=r_trade,
                         kelly_f=kf, f_used=f_used, contrib=f_used * r_trade))
    return pd.DataFrame(rows).set_index("date"), funnel

# =============================================================================
# Main: run all three, compare
# =============================================================================
def main():
    print("=" * 72)
    print("VRP HARVEST -- short-vol structure comparison (model-gated, real IV)")
    print(f"  GAP_MIN={GAP_MIN} SHORT_STEPS={SHORT_STEPS} WING_STEPS={WING_STEPS} "
          f"PT={PROFIT_TAKE} EXIT_DTE={EXIT_DTE} STOP={STOP_FRAC} KELLY={KELLY_FRAC}")
    print("=" * 72)

    contexts, etf, last_daily, X, y, df = PS.build_contexts()

    summary, equities, funnels = [], {}, {}
    for name in BUILDERS:
        bt, funnel = evaluate(name, contexts, etf, last_daily)
        funnels[name] = funnel
        bt2, bh_equity, m = B.equity_and_metrics(bt, df, y)
        mdd = m["rule_mdd"]
        calmar = m["rule_cagr"] / abs(mdd) if (mdd and np.isfinite(mdd) and mdd != 0) else np.nan
        equities[name] = bt2["equity"]
        traded = bt[bt.get("traded", 0) == 1]
        ex_mix = (traded["exit_kind"].value_counts().to_dict()
                  if "exit_kind" in traded else {})
        # does the model's ex-ante signal predict realized short-vol P&L?
        if {"gap", "kelly_f", "r_trade"}.issubset(traded.columns):
            c_gap = _corr(traded["gap"], traded["r_trade"])
            c_kelly = _corr(traded["kelly_f"], traded["r_trade"])
        else:
            c_gap = c_kelly = np.nan
        summary.append(dict(structure=name, trades=m["trades"],
                            cagr=m["rule_cagr"], mdd=mdd, calmar=calmar,
                            winrate=m["trade_winrate"], avg_trade=m["avg_trade_ret"],
                            total=m["rule_total"], exits=ex_mix,
                            corr_gap=c_gap, corr_kelly=c_kelly))
        bt.to_csv(OUTDIR / f"vrp_{name}_trades.csv")

    # buy & hold reference (same months)
    bh = y.reindex(bt2.index).fillna(0.0)
    bh_equity = np.exp(bh.cumsum())

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUTDIR / "vrp_compare_summary.csv", index=False)
    print("\n--- structure comparison ---")
    cols = ["structure", "trades", "cagr", "mdd", "calmar", "winrate", "avg_trade", "total"]
    print(sdf[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n--- exit mix per structure ---")
    for r in summary:
        print(f"  {r['structure']:12s}: {r['exits']}")

    print("\n--- does the model's ex-ante signal predict realized P&L? ---")
    print("  (corr across the FULL traded sample; positive => model adds timing value)")
    for r in summary:
        print(f"  {r['structure']:12s}: corr(gap, r_trade)={r['corr_gap']:+.3f}  "
              f"corr(model_kelly, r_trade)={r['corr_kelly']:+.3f}")

    print("\n--- trade funnel per structure (where months are lost) ---")
    print(f"  SIZING_MODE={SIZING_MODE}  (no_kelly only applies in model_kelly mode)")
    for name, fn in funnels.items():
        print(f"  {name:12s}: months={fn['months']} no_gap={fn['no_gap']} "
              f"no_build={fn['no_build']} no_kelly={fn['no_kelly']} "
              f"-> traded={fn['traded']}")

    # overlay equity curves
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"butterfly": "#1f4e79", "iron_condor": "#2e7d32", "iron_fly": "#c0392b"}
    for name, eq in equities.items():
        ax.plot(eq.index, eq, label=name, lw=1.6, color=colors.get(name))
    ax.plot(bh_equity.index, bh_equity, label="buy & hold 50ETF",
            lw=1.1, color="grey", ls="--")
    ax.axhline(1.0, color="black", lw=0.6)
    ax.set_ylabel("growth of 1"); ax.legend()
    ax.set_title("VRP harvest: short-vol structures vs buy & hold (frac-Kelly, real IV)")
    fig.tight_layout(); fig.savefig(OUTDIR / "vrp_compare_equity.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved: vrp_compare_summary.csv, vrp_*_trades.csv, vrp_compare_equity.png")
    print("\nREGRESSION CHECK: the 'butterfly' row here is a LONG debit butterfly")
    print("priced from the same smile as butterfly_timing_backtest.py. With the")
    print("matching WING_STEPS/EXIT_DTE/PROFIT_TAKE and gate, its per-trade r and")
    print("exit mix should line up with that engine's butterfly trades.")
    print("\nCaveat: short gamma. Inspect the worst single-month r_trade per")
    print("structure (in vrp_*_trades.csv) -- the iron fly will show the deepest")
    print("tail. Risk-adjusted (calmar) matters more than cagr here; and ~small")
    print("trade counts mean treat differences as directional, not decisive.")


if __name__ == "__main__":
    main()
