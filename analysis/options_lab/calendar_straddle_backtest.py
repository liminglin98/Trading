"""Always-on ATM straddle CALENDAR: short near-month ATM straddle + long far-month ATM straddle, same strike.

Hypothesis: the long far leg (long vega, long gamma at distance) cuts the vol-spike / gap tail of the short near
straddle, so return per margin and drawdown-sized return hold up better than the near-only short straddle. The near
leg alone is a monthly short ATM straddle (already seen to lose as a monthly slot in quarterly_fill_cycle), so the
question is whether the far leg adds value net of its cost, and whether the whole beats the live quarterly straddle.

Structure (one lot, re-struck every cycle, no entry signal):
  strike K  listed strike nearest spot that exists for call and put in BOTH expiries
  near      expiry = first with DTE >= 25;  short K call + short K put
  far       expiry = first with DTE >= near_DTE + GAP;  long K call + long K put
  exit      "hold": near cash-settled at expiry (intrinsic), far sold at that day's close, re-enter next trading day
            "7dte": whole position closed at near DTE <= 7, re-entered same day
Grid (pre-registered, 4 configs per ETF): GAP in {50, 80} x exit in {hold, 7dte}.

Costs / margin conventions as rolling_itm_call_short_strangle_backtest (measured half-spread tiers t0/med/p90 + flat
0.005 stress, ATM table applied to all legs; spot = chain's C_UND_CLOSE; "A" contracts excluded; force-close counted).
Deposit (per lot, yuan): CONSERVATIVE = broker naked combo margin on the near straddle (broker_margin) + far legs' mark
(no calendar offset assumed); OPTIMISTIC = max(far mark - near mark, 0) i.e. net debit, if the broker recognises a
calendar combination (NOT verified -- reported only as an upper bound, never used for the pass rule).
Sizing: lots = 15% x 1.0m / |max DD per lot|; ret_per_dep is scale-free.
Benchmarks (same window, same costs): near-only short straddle at the same entries (the calendar minus its far leg),
and the live quarterly ATM short straddle (short_vol_structures_backtest.run).

PRE-REGISTERED READING (written before any result was seen)
A config passes on an underlying only if, at BOTH tier med and tier p90, ALL hold:
  (a) calendar ret_per_dep (conservative deposit) > near-only AND > quarterly straddle;
  (b) discovery (<2024-01-01) and holdout (>=2024-01-01) account return both > 0;
  (c) total monthly P&L > 0 after dropping the 3 best months;
  (d) the far leg adds P&L: far-leg monthly mean > 0 with circular block-bootstrap 95% CI low > 0.
Validated only if the same config passes on BOTH 510050 and 510300; otherwise FAILED. Reported but not part of the rule:
worst-20-day and worst-month P&L of calendar vs near-only (the tail-protection claim), optimistic-deposit return.
Configs tried: 4 per ETF.

Usage:  python calendar_straddle_backtest.py --underlying 510050 [--start 2019-12-23]
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
import short_vol_structures_backtest as sv
from rolling_itm_call_short_strangle_backtest import (ACCOUNT, MULT, OUT_DIR, STRADDLE_SLIP, TIERS, boot_lo,
                                                      half_spread, intrinsic, prep, stats)

warnings.filterwarnings("ignore")
GAPS = (50, 80)
EXITS = ("hold", "7dte")
NEAR_MIN_DTE = 25


def pick(day: pd.DataFrame, dt, spot: float, gap: int):
    exps = sorted(pd.Timestamp(e) for e in day[ds.C_EXPIRY].unique())
    near = next((e for e in exps if (e - dt).days >= NEAR_MIN_DTE), None)
    if near is None:
        return None
    nd = (near - dt).days
    far = next((e for e in exps if (e - dt).days >= nd + gap), None)
    if far is None:
        return None
    sets = []
    for e in (near, far):
        for cp in ("C", "P"):
            m = day[(day[ds.C_EXPIRY] == e) & (day[ds.C_CP] == cp)]
            sets.append(set(m[ds.C_STRIKE].round(6)))
    common = sorted(set.intersection(*sets))
    if not common:
        return None
    k = min(common, key=lambda x: abs(x - spot))
    legs = []
    for e, q, comp in ((near, -1, "N"), (far, 1, "F")):
        for cp in ("C", "P"):
            r = day[(day[ds.C_EXPIRY] == e) & (day[ds.C_CP] == cp) & (day[ds.C_STRIKE].round(6) == k)].iloc[0]
            legs.append({"code": r[ds.C_CODE], "cp": cp, "qty": q, "comp": comp, "expiry": e, "strike": float(k),
                         "px": float(r[ds.C_CLOSE]), "last": float(r[ds.C_CLOSE])})
    return legs


def simulate(by_date: dict, dates: list, gap: int, exit_rule: str) -> dict:
    n = len(dates)
    eq = {"N": np.zeros(n), "F": np.zeros(n)}
    cash = {"N": 0.0, "F": 0.0}
    mS, vF, vN, spots = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    fills: list[tuple[int, str, float]] = []
    legs: list[dict] | None = None
    forced, s_prev = 0, np.nan

    for i, dt in enumerate(dates):
        dt = pd.Timestamp(dt)
        day = by_date[dates[i]]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])
        spots[i] = spot
        rows = day.set_index(ds.C_CODE)
        just_settled = False

        if legs is not None:
            near_exp = legs[0]["expiry"]
            present = [l["code"] in rows.index for l in legs]
            ok = all(present) and all(abs(float(rows.at[l["code"], ds.C_STRIKE]) - l["strike"]) < 1e-9 for l in legs)
            if dt >= near_exp or not ok or (exit_rule == "7dte" and (near_exp - dt).days <= 7):
                if not ok and dt < near_exp:
                    forced += 1
                for l, p in zip(legs, present):
                    if l["comp"] == "N" and dt >= near_exp:
                        k = float(rows.at[l["code"], ds.C_STRIKE]) if p else l["strike"]
                        cash["N"] += l["qty"] * intrinsic(l["cp"], k, spot)         # cash settlement, no spread
                    else:
                        px = float(rows.at[l["code"], ds.C_CLOSE]) if p else l["last"]
                        cash[l["comp"]] += l["qty"] * px
                        fills.append((i, l["comp"], (l["expiry"] - dt).days))
                just_settled = dt >= near_exp
                legs = None
            else:
                for l in legs:
                    l["last"] = float(rows.at[l["code"], ds.C_CLOSE])

        if legs is None and not (just_settled and exit_rule == "hold"):
            sel = pick(day, dt, spot, gap)
            if sel:
                for l in sel:
                    cash[l["comp"]] -= l["qty"] * l["px"]
                    fills.append((i, l["comp"], (l["expiry"] - dt).days))
                legs = sel

        for c in ("N", "F"):
            eq[c][i] = cash[c] + sum(l["qty"] * l["last"] for l in (legs or []) if l["comp"] == c)
        if legs:
            vF[i] = sum(l["last"] for l in legs if l["comp"] == "F") * MULT
            vN[i] = sum(l["last"] for l in legs if l["comp"] == "N") * MULT
            if not np.isnan(s_prev):
                nc, npu = [l for l in legs if l["comp"] == "N"]

                def sp(l):
                    v = rows.at[l["code"], ds.C_PREV_SETTLE] if l["code"] in rows.index else np.nan
                    return float(v) if pd.notna(v) else l["last"]
                mS[i] = bm.combo_short_margin(nc["strike"], npu["strike"], s_prev, sp(nc), sp(npu))
        s_prev = spot
    return {"dates": pd.DatetimeIndex(dates), "eq": eq, "mS": mS, "vF": vF, "vN": vN, "spot": spots,
            "fills": fills, "forced": forced}


def tier_pnl(res: dict, comps: str, tier: str, u: str) -> np.ndarray:
    n = len(res["dates"])
    out = np.zeros(n)
    for c in comps:
        cost = np.zeros(n)
        for i, cc, dte in res["fills"]:
            if cc == c:
                cost[i] += half_spread(tier, "short", u, dte)
        out += np.diff(np.concatenate([[0.0], res["eq"][c] - np.cumsum(cost)]))
    return out * MULT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--start", default="2019-12-23")
    args = ap.parse_args()
    u, start = args.underlying, pd.Timestamp(args.start)
    pd.set_option("display.width", 250)
    df, dates, by_date = prep(u)
    idx = pd.DatetimeIndex(dates)

    bench = {}
    for t in TIERS:
        r = sv.run(("straddle",), STRADDLE_SLIP[t], by_date, dates)["eq"]
        bench[t] = stats(r["equity"].diff().fillna(0.0) * MULT, r["margin"], start)

    rows, monthly = [], {}
    forced = {}
    for gap in GAPS:
        for ex in EXITS:
            lab = f"gap{gap}_{ex}"
            res = simulate(by_date, dates, gap, ex)
            forced[lab] = res["forced"]
            dep_cons = pd.Series(res["mS"] + res["vF"], index=idx)
            dep_opt = pd.Series(np.maximum(res["vF"] - res["vN"], 0.0), index=idx)
            dep_near = pd.Series(res["mS"], index=idx)
            for t in TIERS:
                for comps, name, dep in (("NF", "calendar", dep_cons), ("NF", "calendar_opt", dep_opt),
                                         ("N", "near_only", dep_near), ("F", "far_leg", dep_cons)):
                    p = pd.Series(tier_pnl(res, comps, t, u), index=idx)
                    s = stats(p, dep, start)
                    if not s:
                        continue
                    m = (p[p.index >= start].resample("ME").sum() / ACCOUNT)
                    rows.append({"cfg": lab, "tier": t, "comp": name, **s, "boot_lo_monthly": boot_lo(m.values),
                                 "worst_month_yuan_per_lot": p[p.index >= start].resample("ME").sum().min()})
                    if t == "med":
                        monthly[(lab, name)] = p[p.index >= start].resample("ME").sum()
            print(f"{lab:<14} forced-closes {res['forced']}", flush=True)
    out = pd.DataFrame(rows)
    bdf = pd.DataFrame([{"cfg": "quarterly_straddle", "tier": t, **s} for t, s in bench.items() if s])
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / f"calendar_straddle_{u}_{start:%Y%m%d}.csv", index=False)
    bdf.to_csv(OUT_DIR / f"calendar_straddle_{u}_{start:%Y%m%d}_bench.csv", index=False)

    cols = ["cfg", "comp", "lots", "ann_ret", "ret_per_dep", "sharpe", "disc", "hold", "drop3", "pos_yrs", "worst20",
            "avg_dep", "worst_month_yuan_per_lot", "boot_lo_monthly"]
    for t in ("med", "p90"):
        print(f"\n== {u} {start.date()}.. tier {t}, sized to max DD 15% of 1.0m ==")
        print(out[out.tier == t].sort_values(["cfg", "comp"])[cols].round(3).to_string(index=False))
    print("\n== quarterly straddle benchmark ==")
    print(bdf[["tier", "lots", "ann_ret", "ret_per_dep", "sharpe", "disc", "hold", "drop3", "worst20"]].round(3).to_string(index=False))

    passes = []
    for lab in out["cfg"].unique():
        ok = True
        for t in ("med", "p90"):
            g = out[(out.cfg == lab) & (out.tier == t)].set_index("comp")
            b = bdf[bdf.tier == t]
            if not {"calendar", "near_only", "far_leg"} <= set(g.index) or b.empty:
                ok = False
                break
            c, nn, fl = g.loc["calendar"], g.loc["near_only"], g.loc["far_leg"]
            ok &= bool(c.ret_per_dep > nn.ret_per_dep and c.ret_per_dep > b.iloc[0].ret_per_dep and c.disc > 0
                       and c.hold > 0 and c.drop3 > 0 and fl.total > 0 and fl.boot_lo_monthly > 0)
        if ok:
            passes.append(lab)
    print(f"\nconfigs passing the pre-registered rule on {u}: {passes or 'NONE'}   (configs tried {len(forced)})")
    print("forced closes:", forced)


if __name__ == "__main__":
    main()
