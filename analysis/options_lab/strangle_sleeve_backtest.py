"""Stand-alone monthly short-strangle sleeve rolled before expiry, tested alone AND as an add-on to the quarterly
short straddle book. Follow-up to rolling_itm_call_short_strangle_backtest, whose strangle-only leg matched the
straddle per deposit (510050 2015+ 0.48 vs 0.31) but was never tested for what it adds to the live book.

Sleeve (one lot, no entry signal, no long leg): short call and short put at the listed strike nearest spot*(1 +/- d),
strictly OTM, nearest expiry with DTE >= 25 (~monthly), closed when DTE <= ROLL and re-entered the same day; or
"hold" (cash-settled at expiry, re-entered next trading day).
PRIMARY config (pre-registered, the only one the verdict uses): d = 3%, roll at DTE <= 7.
ROBUSTNESS grid (reported, not in the verdict): d in {3,5,7}% x {hold, roll 3, roll 7, roll 10} = 12 configs.

Costs / margin: measured half-spread tiers t0/med/p90 + flat 0.005 stress (ATM table, as the other scripts); broker naked
combo margin re-marked daily (broker_margin.combo_short_margin), no offset. Window default 2019-12-23 (also 510050 2015+).
Benchmark and book: quarterly ATM short straddle (short_vol_structures_backtest.run), same window/costs.
Sizing: lots = 15% x 1.0m / |max DD per lot|; ret_per_dep is scale-free.

PRE-REGISTERED READING (written before any result was seen) -- on the PRIMARY config, at BOTH tier med and p90:
  P1 stand-alone: discovery (<2024-01-01) > 0, holdout (>=2024-01-01) > 0, positive after dropping the 3 best months,
     and circular block-bootstrap 95% CI low of the monthly mean P&L > 0.
  P2 adds to the book: combine straddle + r x sleeve per lot (r = sleeve lots per straddle lot, r in {0.5, 1.0}),
     deposit = sum of deposits, re-sized to max DD 15%. For at least one r: combined ann return at 15% DD AND combined
     ret_per_dep both exceed the straddle alone, with disc > 0 and holdout > 0.
  P3 not a duplicate: monthly P&L correlation with the straddle < 0.7 (reported with the straddle's 10 worst months:
     sleeve P&L in them; not a pass/fail beyond the correlation threshold).
VALIDATED only if P1, P2 and P3 all pass on BOTH 510050 and 510300 (same config); a pass on one ETF only is reported as
"one-ETF support", not validated. Otherwise FAILED. Configs tried: 12 (primary pre-registered; 11 robustness).

Usage:  python strangle_sleeve_backtest.py --underlying 510050 [--start 2019-12-23]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

MAIN_OL = Path("/mnt/data/Trading/analysis/options_lab")
if MAIN_OL.exists() and MAIN_OL != Path(__file__).resolve().parent:   # helpers + Data/ live in the main checkout
    sys.path.insert(0, str(MAIN_OL))

import numpy as np
import pandas as pd

import data_store as ds
import short_vol_structures_backtest as sv
from rolling_itm_call_short_strangle_backtest import (ACCOUNT, MULT, OUT_DIR, STRADDLE_SLIP, TIERS, boot_lo, prep,
                                                      simulate, stats, tier_pnl)

warnings.filterwarnings("ignore")
PRIMARY = (0.03, "7dte", 7)
GRID = [(d, mode, r) for d in (0.03, 0.05, 0.07)
        for mode, r in (("hold", 7), ("7dte", 3), ("7dte", 7), ("7dte", 10))]
RATIOS = (0.5, 1.0)
CORR_MAX = 0.7


def label(c) -> str:
    return f"d{c[0]:.0%}_" + ("hold" if c[1] == "hold" else f"roll{c[2]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--start", default="2019-12-23")
    args = ap.parse_args()
    u, start = args.underlying, pd.Timestamp(args.start)
    pd.set_option("display.width", 250)
    df, dates, by_date = prep(u)
    idx = pd.DatetimeIndex(dates)

    book = {}
    for t in TIERS:
        r = sv.run(("straddle",), STRADDLE_SLIP[t], by_date, dates)["eq"]
        book[t] = (r["equity"].diff().fillna(0.0) * MULT, r["margin"])

    rows, comb_rows, extras = [], [], {}
    for cfg in GRID:
        lab = label(cfg)
        res = simulate(by_date, dates, cfg[0], cfg[1], 25, roll_dte=cfg[2], with_long=False)
        dep = pd.Series(res["mS"], index=idx)
        for t in TIERS:
            p = pd.Series(tier_pnl(res, "S", t, u), index=idx)
            s = stats(p, dep, start)
            if not s:
                continue
            m = p[p.index >= start].resample("ME").sum()
            rows.append({"cfg": lab, "tier": t, **s, "boot_lo_monthly": boot_lo((m / ACCOUNT).values),
                         "worst_month_yuan_per_lot": m.min()})
            ps, ds_ = book[t]
            sm = ps[ps.index >= start].resample("ME").sum()
            corr_m = float(np.corrcoef(m.values, sm.values)[0, 1])
            worst = sm.nsmallest(10).index
            extras[(lab, t)] = {"corr_monthly": corr_m,
                                "corr_daily": float(np.corrcoef(p[p.index >= start], ps[ps.index >= start])[0, 1]),
                                "sleeve_in_straddle_worst10_mean": float(m.loc[worst].mean()),
                                "sleeve_negative_in_worst10": int((m.loc[worst] < 0).sum())}
            base = stats(ps, ds_, start)
            for r_ in RATIOS:
                c = stats(ps + r_ * p, ds_ + r_ * dep, start)
                if c:
                    comb_rows.append({"cfg": lab, "tier": t, "ratio": r_, "ann_ret": c["ann_ret"],
                                      "ret_per_dep": c["ret_per_dep"], "disc": c["disc"], "hold": c["hold"],
                                      "worst20": c["worst20"], "base_ann_ret": base["ann_ret"],
                                      "base_ret_per_dep": base["ret_per_dep"], "base_worst20": base["worst20"]})
        print(f"{lab:<14} done", flush=True)
    out, comb = pd.DataFrame(rows), pd.DataFrame(comb_rows)
    ex = pd.DataFrame([{"cfg": k[0], "tier": k[1], **v} for k, v in extras.items()])
    OUT_DIR.mkdir(exist_ok=True)
    tag = f"{u}_{start:%Y%m%d}"
    out.to_csv(OUT_DIR / f"strangle_sleeve_{tag}.csv", index=False)
    comb.to_csv(OUT_DIR / f"strangle_sleeve_{tag}_combined.csv", index=False)
    ex.to_csv(OUT_DIR / f"strangle_sleeve_{tag}_corr.csv", index=False)

    cols = ["cfg", "lots", "ann_ret", "ret_per_dep", "sharpe", "disc", "hold", "drop3", "pos_yrs", "worst20",
            "avg_dep", "worst_month_yuan_per_lot", "boot_lo_monthly"]
    for t in ("med", "p90"):
        print(f"\n== {u} {start.date()}.. tier {t}: sleeve alone, sized to max DD 15% of 1.0m ==")
        print(out[out.tier == t][cols].round(3).to_string(index=False))
    sb = {t: stats(*book[t], start) for t in ("med", "p90")}
    print("\nquarterly straddle book:", {t: {k: round(v, 3) for k, v in s.items() if k in
          ("ann_ret", "ret_per_dep", "disc", "hold", "worst20")} for t, s in sb.items()})
    print("\n== combined (straddle + r x sleeve per lot), re-sized to 15% DD ==")
    print(comb[comb.tier.isin(["med", "p90"])].round(3).to_string(index=False))
    print("\n== correlation / tail overlap with the straddle ==")
    print(ex[ex.tier.isin(["med", "p90"])].round(3).to_string(index=False))

    pl = label(PRIMARY)
    p1 = p2 = p3 = True
    for t in ("med", "p90"):
        s = out[(out.cfg == pl) & (out.tier == t)]
        if s.empty:
            p1 = p2 = p3 = False
            break
        s = s.iloc[0]
        p1 &= bool(s.disc > 0 and s.hold > 0 and s.drop3 > 0 and s.boot_lo_monthly > 0)
        c = comb[(comb.cfg == pl) & (comb.tier == t)]
        p2 &= bool(((c.ann_ret > c.base_ann_ret) & (c.ret_per_dep > c.base_ret_per_dep) & (c.disc > 0)
                    & (c.hold > 0)).any())
        p3 &= bool(extras[(pl, t)]["corr_monthly"] < CORR_MAX)
    print(f"\nPRIMARY {pl} on {u}: P1 stand-alone {'PASS' if p1 else 'FAIL'} | P2 adds to book "
          f"{'PASS' if p2 else 'FAIL'} | P3 corr<{CORR_MAX} {'PASS' if p3 else 'FAIL'}")
    p1_grid = sum(
        bool(all((lambda s: (not s.empty) and s.iloc[0].disc > 0 and s.iloc[0].hold > 0 and s.iloc[0].drop3 > 0
                  and s.iloc[0].boot_lo_monthly > 0)(out[(out.cfg == label(c)) & (out.tier == t)])
                 for t in ("med", "p90"))) for c in GRID)
    print(f"robustness: {p1_grid}/{len(GRID)} grid configs pass P1 on {u}")


if __name__ == "__main__":
    main()
