"""Quarterly ATM short straddle: spread a 25-lot entry continuously from T90 down to T75 vs a 5-lots-x-5-days ladder.

Question: since T75 ~ T90 per cycle with better capital efficiency, and a 5-day entry ladder beat the lump entry, does
spreading the same 25 lots over every trading day with 75 <= DTE <= 90 (~16 days, ~1.5 lots/day) do better than 5 lots
on each of the first 5 days of T90?

Method: ONE ENTRY-DAY PANEL. For every quarterly expiry E and every eligible trading day d (after the previous
settlement, before E), a 1-lot ATM short straddle opened at d (strike = nearest raw spot that day, as the live book
re-strikes daily; qs._atm_strike / qs._leg_row, same loader and no-dedupe filter as the entry-DTE sweep) is carried to
E: daily mark = call + put close, settlement at intrinsic on the first date >= E or when a leg's row is missing
(the sweep's rule), daily broker combo margin (broker_margin.combo_short_margin, previous-day close and prev settle).
Every arm is a lot-weight vector over these tranches, all held to the same E, 25 lots per cycle:
  L90      25 lots on the first available day with DTE <= 90        (sweep T90)
  L75      25 lots on the first available day with DTE <= 75        (sweep T75)
  B_settle 5 lots/day x the first 5 available days after the previous settlement (existing --ladder-days 5 rule)
  B_T90    5 lots/day x 5 consecutive available days starting at the first day with DTE <= 90
  A        25 lots over every available day with 75 <= DTE <= 90, integer lots, evenly spread (cumulative rounding)
  A_frac   same days, equal fractional weights (sensitivity)
Costs: per-leg entry half-spread by entry DTE, measured tiers t0 / med / p90 (rolling_itm_call_short_strangle_backtest
ATM table: 510050 .00025 / .00045, 510300 .0004 / .00075 at DTE 60-100), flat 0.005 stress; no exit fill (cash
settlement). Window: all cycles for the reproduction check, FULL cycles only (first eligible DTE >= 88) for arms.
Sizing: the same 25 lots in every arm, so yuan P&L and DD are directly comparable; also reported scaled to a 15% DD cap
(1.0m account). Discovery < 2024-01-01 <= holdout.

PRE-REGISTERED READING (written before any result was seen) -- primary comparison A vs B_T90, per ETF, at BOTH
tier med and p90 (B_settle is a robustness check):
  P&L edge      mean paired per-cycle P&L (A - B_T90) > 0 AND circular block-bootstrap (block 2 cycles, 5000 draws,
                seed 0) 95% CI low > 0, AND both the discovery and holdout mean paired differences > 0
  capital edge  return per margin-day (annualised, sum P&L / sum margin x calendar days held) higher in A, AND daily
                Sharpe not lower, AND the margin-day ratio higher in both discovery and holdout
  and in either case max drawdown (yuan, same lots) of A not worse than B_T90's.
The verdict names which edge (if any) holds on BOTH ETFs. Expectation stated up front: the per-cycle P&L difference
will be not significant (T75 vs T90 was t = 1.0 / 0.5); a gain, if any, will be capital efficiency.
Arms tried: 6 (primary pre-registered A vs B_T90; others context/robustness).

Usage:  python quarterly_entry_ladder_window_backtest.py --underlying 510050
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
import broker_margin as bm
import quarterly_atm_short_straddle as qs
from rolling_itm_call_short_strangle_backtest import ACCOUNT, MULT, OUT_DIR, boot_lo, half_spread

warnings.filterwarnings("ignore")
SPLIT = pd.Timestamp("2024-01-01")
CAP = 0.15
LOTS = 25
TIERS = ["t0", "med", "p90", "flat"]
ARMS = ["L90", "L75", "B_settle", "B_T90", "A", "A_frac"]
FULL_CYCLE_DTE = 88


# ------------------------------------------------------------------ data
def load(u: str) -> dict:
    df = ds.load_chain(u)
    df = df[~df[ds.C_NAME].astype(str).str.endswith("A")]
    df = df[df[ds.C_CLOSE] > 0]
    by_date = dict(tuple(df.groupby(ds.C_DATE)))
    dates = [pd.Timestamp(d) for d in sorted(by_date)]
    close, prev, spot = {}, {}, {}
    for d in dates:
        f = by_date[d].drop_duplicates(ds.C_CODE, keep="first")
        close[d] = dict(zip(f[ds.C_CODE], f[ds.C_CLOSE].astype(float)))
        prev[d] = dict(zip(f[ds.C_CODE], f[ds.C_PREV_SETTLE].astype(float)))
        spot[d] = float(by_date[d][ds.C_UND_CLOSE].iloc[0])
    quarterly = sorted({pd.Timestamp(e) for e in df[ds.C_EXPIRY].unique() if pd.Timestamp(e).month in (3, 6, 9, 12)})
    return {"by_date": by_date, "dates": dates, "close": close, "prev": prev, "spot": spot, "quarterly": quarterly}


def build_tranche(D: dict, di: int, E: pd.Timestamp) -> dict | None:
    dates = D["dates"]
    dt = dates[di]
    day, s = D["by_date"][dt], D["spot"][dt]
    k = qs._atm_strike(day, E, s)
    if k is None:
        return None
    crow, prow = qs._leg_row(day, E, "C", k), qs._leg_row(day, E, "P", k)
    if crow is None or prow is None:
        return None
    cc, pc = crow[ds.C_CODE], prow[ds.C_CODE]
    c0, p0 = float(crow[ds.C_CLOSE]), float(prow[ds.C_CLOSE])
    marks, margins, idxs = [], [], []
    settle_val, settle_idx = None, None
    for j in range(di, len(dates)):
        d = dates[j]
        if j > di:
            cl = D["close"][d]
            if d >= E or cc not in cl or pc not in cl:
                sp = D["spot"][d]
                settle_val = qs._intrinsic("C", k, sp) + qs._intrinsic("P", k, sp)
                settle_idx = j
                break
            mark = cl[cc] + cl[pc]
        else:
            mark = c0 + p0
        ps = D["prev"][d]
        cps = ps.get(cc, np.nan)
        pps = ps.get(pc, np.nan)
        cps = cps if pd.notna(cps) else D["close"][d].get(cc, c0)
        pps = pps if pd.notna(pps) else D["close"][d].get(pc, p0)
        s_prev = D["spot"][dates[j - 1]] if j > 0 else s
        marks.append(mark)
        margins.append(bm.combo_short_margin(k, k, s_prev, cps, pps))
        idxs.append(j)
    if settle_idx is None:                     # data ends before expiry
        return None
    return {"di": di, "date": dt, "dte": (E - dt).days, "strike": k, "c0": c0, "p0": p0, "idx": np.array(idxs),
            "marks": np.array(marks), "margins": np.array(margins), "settle_val": settle_val,
            "settle_idx": settle_idx}


def cycles(D: dict) -> list[dict]:
    dates, q = D["dates"], D["quarterly"]
    out = []
    for n, E in enumerate(q):
        if E > dates[-1]:
            continue
        prev_E = q[n - 1] if n > 0 else None
        lo = 0
        if prev_E is not None and prev_E >= dates[0]:
            lo = int(np.searchsorted(dates, prev_E, side="left")) + 1       # day after the previous settlement
        hi = int(np.searchsorted(dates, E, side="left"))                    # strictly before E
        days = [i for i in range(lo, hi) if (prev_E is None or dates[i] > prev_E)]
        tr = []
        for i in days:
            if (E - dates[i]).days < 40:
                break
            t = build_tranche(D, i, E)
            if t is not None:
                tr.append(t)
        if tr:
            out.append({"E": E, "tr": tr, "full": tr[0]["dte"] >= FULL_CYCLE_DTE})
    return out


def allocate(cyc: dict, arm: str) -> list[tuple[dict, float]]:
    tr = cyc["tr"]
    if arm in ("L90", "L75"):
        T = 90 if arm == "L90" else 75
        t = next((x for x in tr if x["dte"] <= T), None)
        return [(t, LOTS)] if t else []
    if arm == "B_settle":
        return [(x, 5.0) for x in tr[:5]] if len(tr) >= 5 else []
    if arm == "B_T90":
        i = next((n for n, x in enumerate(tr) if x["dte"] <= 90), None)
        return [(x, 5.0) for x in tr[i:i + 5]] if i is not None and len(tr[i:i + 5]) == 5 else []
    days = [x for x in tr if 75 <= x["dte"] <= 90]
    if not days:
        return []
    n = len(days)
    if arm == "A":
        w = [round(LOTS * (i + 1) / n) - round(LOTS * i / n) for i in range(n)]
        return [(x, float(l)) for x, l in zip(days, w) if l > 0]
    return [(x, LOTS / n) for x in days]                                      # A_frac


def credit(t: dict, tier: str, u: str) -> float:
    s = half_spread(tier, "short", u, t["dte"])
    return max(t["c0"] - s, 0.0) + max(t["p0"] - s, 0.0)


# ------------------------------------------------------------------ metrics
def arm_series(D: dict, cycs: list[dict], arm: str, tier: str, u: str, full_only: bool = True) -> dict:
    n = len(D["dates"])
    cum = np.zeros(n)
    margin = np.zeros(n)
    rows = []
    for c in cycs:
        if full_only and not c["full"]:
            continue
        alloc = allocate(c, arm)
        if not alloc:
            continue
        assert abs(sum(w for _, w in alloc) - LOTS) < 1e-6, (c["E"], arm)
        cyc_cum = np.zeros(n)
        mdays = 0.0
        last_settle = 0
        for t, w in alloc:
            cr = credit(t, tier, u)
            eq = np.zeros(n)
            eq[t["idx"]] = w * (cr - t["marks"])
            eq[t["settle_idx"]:] = w * (cr - t["settle_val"])
            cyc_cum += eq * MULT
            margin[t["idx"]] += w * t["margins"]
            last_settle = max(last_settle, t["settle_idx"])
            gaps = np.array([(D["dates"][min(j + 1, n - 1)] - D["dates"][j]).days for j in t["idx"]])
            mdays += float((w * t["margins"] * gaps).sum())
        cum += cyc_cum
        rows.append({"E": c["E"], "pnl": cyc_cum[last_settle], "margin_days": mdays,
                     "entry_date": min(t["date"] for t, _ in alloc),
                     "avg_entry_dte": float(np.average([t["dte"] for t, _ in alloc], weights=[w for _, w in alloc]))})
    return {"daily": np.diff(np.concatenate([[0.0], cum])), "margin": margin, "cycles": pd.DataFrame(rows)}


def summarise(D: dict, s: dict) -> dict:
    idx = pd.DatetimeIndex(D["dates"])
    p = pd.Series(s["daily"], index=idx)
    m = pd.Series(s["margin"], index=idx)
    cum = p.cumsum()
    dd = float((cum - cum.cummax()).min())
    yrs = (idx[-1] - idx[0]).days / 365.25
    ann = p.sum() / ACCOUNT / yrs
    dep = m[m > 0].mean() / ACCOUNT
    cy = s["cycles"]
    rpd = cy["pnl"].sum() / cy["margin_days"].sum() * 365 if len(cy) else np.nan
    cd = cy[cy["entry_date"] < SPLIT]
    ch = cy[cy["entry_date"] >= SPLIT]
    return {"n": len(cy), "total": p.sum(), "ann_ret": ann, "ann_at_dd_cap": ann * CAP * ACCOUNT / abs(dd) if dd < 0 else np.nan,
            "max_dd": dd, "sharpe": p.mean() / p.std() * np.sqrt(244) if p.std() > 0 else np.nan,
            "avg_dep": dep, "ret_per_dep": ann / dep, "rpd_margin_day": rpd,
            "rpd_disc": cd["pnl"].sum() / cd["margin_days"].sum() * 365 if len(cd) else np.nan,
            "rpd_hold": ch["pnl"].sum() / ch["margin_days"].sum() * 365 if len(ch) else np.nan,
            "disc_pnl": cd["pnl"].sum(), "hold_pnl": ch["pnl"].sum(),
            "avg_entry_dte": cy["avg_entry_dte"].mean(), "mean_cycle_pnl": cy["pnl"].mean()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    args = ap.parse_args()
    u = args.underlying
    pd.set_option("display.width", 250)
    D = load(u)
    cycs = cycles(D)
    nfull = sum(c["full"] for c in cycs)
    print(f"{u}: {len(cycs)} cycles with tranches, {nfull} full cycles; "
          f"dates {D['dates'][0].date()}..{D['dates'][-1].date()}", flush=True)

    # reproduction of the entry-DTE sweep (1 lot, all cycles, flat slippage per leg) --------------
    print("\n== reproduction: single-lot totals vs quarterly_entry_dte_sweep (all cycles) ==")
    global LOTS
    keep, LOTS = LOTS, 1
    for T, arm in ((90, "L90"), (75, "L75")):
        for slip in (0.0, 0.005):
            tot, n = 0.0, 0
            for c in cycs:
                a = allocate(c, arm)
                if not a:
                    continue
                t, _ = a[0]
                tot += ((max(t["c0"] - slip, 0) + max(t["p0"] - slip, 0)) - t["settle_val"]) * MULT
                n += 1
            print(f"  {arm} slip {slip:.3f}: n={n} total {tot:,.0f}")
    LOTS = keep

    rows, cyc_tables = [], {}
    for t in TIERS:
        for arm in ARMS:
            s = arm_series(D, cycs, arm, t, u)
            r = summarise(D, s)
            rows.append({"tier": t, "arm": arm, **r})
            cyc_tables[(t, arm)] = s["cycles"].set_index("E")
    res = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    res.to_csv(OUT_DIR / f"quarterly_entry_ladder_window_{u}.csv", index=False)
    cols = ["arm", "n", "total", "ann_ret", "ann_at_dd_cap", "max_dd", "sharpe", "avg_dep", "ret_per_dep",
            "rpd_margin_day", "rpd_disc", "rpd_hold", "disc_pnl", "hold_pnl", "avg_entry_dte"]
    for t in TIERS:
        print(f"\n== {u} tier {t}: {LOTS} lots, full cycles ==")
        print(res[res.tier == t][cols].round(3).to_string(index=False))

    # paired per-cycle and the pre-registered rule -------------------------------------------------
    print("\n== paired per-cycle P&L (yuan, 25 lots): A vs comparators ==")
    verdict = {}
    for t in ("med", "p90"):
        for other in ("B_T90", "B_settle", "L90", "L75"):
            a, b = cyc_tables[(t, "A")]["pnl"], cyc_tables[(t, other)]["pnl"]
            j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
            d = j["a"] - j["b"]
            tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 2 and d.std() > 0 else np.nan
            entry = cyc_tables[(t, "A")]["entry_date"].reindex(d.index)
            dd_ = d[entry < SPLIT]
            dh = d[entry >= SPLIT]
            lo = boot_lo(d.values, block=2)
            print(f"  tier {t} A - {other:<9} n={len(d)} mean {d.mean():+8.0f} t {tstat:+.2f} CI-low {lo:+8.0f} "
                  f"better {int((d > 0).sum())}/{len(d)} median {d.median():+7.0f} | disc {dd_.mean():+.0f} "
                  f"hold {dh.mean():+.0f}")
            if other == "B_T90":
                ra = res[(res.tier == t) & (res.arm == "A")].iloc[0]
                rb = res[(res.tier == t) & (res.arm == "B_T90")].iloc[0]
                pnl_edge = bool(d.mean() > 0 and lo > 0 and dd_.mean() > 0 and dh.mean() > 0)
                cap_edge = bool(ra.rpd_margin_day > rb.rpd_margin_day and ra.sharpe >= rb.sharpe
                                and ra.rpd_disc > rb.rpd_disc and ra.rpd_hold > rb.rpd_hold)
                dd_ok = bool(ra.max_dd >= rb.max_dd)
                verdict[t] = {"pnl_edge": pnl_edge and dd_ok, "capital_edge": cap_edge and dd_ok, "dd_ok": dd_ok}
    print(f"\nPRE-REGISTERED (A vs B_T90) on {u}: {verdict}")
    both = lambda k: all(verdict[t][k] for t in ("med", "p90"))
    print(f"{u}: P&L edge {'YES' if both('pnl_edge') else 'NO'} | capital edge {'YES' if both('capital_edge') else 'NO'}"
          f"  (needs the same on the other ETF for the overall verdict)")

    # diagnostic: single-lot P&L by entry DTE ---------------------------------------------------------
    diag = []
    for c in cycs:
        if not c["full"]:
            continue
        for t in c["tr"]:
            for tier in ("t0", "med"):
                diag.append({"dte": t["dte"], "tier": tier, "pnl": (credit(t, tier, u) - t["settle_val"]) * MULT})
    g = pd.DataFrame(diag)
    g["bucket"] = (g["dte"] // 5) * 5
    pv = g.pivot_table(index="bucket", columns="tier", values="pnl", aggfunc=["mean", "count"])
    pv.to_csv(OUT_DIR / f"quarterly_entry_ladder_window_{u}_bydte.csv")
    print("\nmean single-lot P&L (yuan) by entry-DTE bucket (full cycles):")
    print(pv.round(0).to_string())


if __name__ == "__main__":
    main()
