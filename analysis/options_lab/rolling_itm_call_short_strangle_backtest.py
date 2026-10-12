"""Always-on rolling structure: long deep-ITM call + short OTM call + short OTM put (no entry signal).

Structure (one lot each, held continuously, rolled mechanically):
  long   call, vendor delta closest to 0.90 (>= 0.85), nearest expiry with DTE >= 60; rolled when DTE <= LROLL
  short  call at the listed strike nearest spot*(1+d), strictly OTM, nearest expiry with DTE >= 25
  short  put  at the listed strike nearest spot*(1-d), strictly OTM, same expiry
         rolled either at expiry ("hold": cash-settled at intrinsic, re-entered next trading day) or at DTE <= 7
Grid (pre-registered, 12 configs per ETF): d in {3%,5%,7%} x short roll in {hold, 7dte} x LROLL in {25,45}.

Because the legs are additive and independent, one simulation per config yields every component:
  full (L + strangle), long-only (L), strangle-only (S). Benchmarks on the same window / costs / margin basis:
  quarterly ATM short straddle (short_vol_structures_backtest.run, the live rule), ETF buy-and-hold (research only;
  stock and option accounts are funded separately, project_stock_option_account_separation).

Conventions: daily close fills, spot = the chain's own C_UND_CLOSE (raw), adjusted ("A") contracts excluded, a held
leg whose row vanishes or whose strike changes is force-closed at its last close (counted). Half-spread charged per
fill, adverse, none at cash settlement; tiers t0 / med / p90 (measured, see below) and flat 0.005 as stress only.
  long ITM call : rally_roll_measured_spread_backtest tiers (510300 measured, strike/spot 0.91-0.97; also used as a
                  PROXY for 510050, whose deep-ITM call spread was never measured; deltas >= 0.85 are deeper).
  short OTM legs: measured ATM table by DTE (reference_measured_spread_data) -- conservative, off-ATM was tighter.
Margin (per lot, yuan): shorts = broker combo rule (broker_margin.combo_short_margin, prev-day close and prev settle),
long call = its mark value (debit). NO offset between the long call and the short legs is assumed -- the broker's
diagonal treatment is not modelled (conservative).
Sizing: lots = 15% * 1.0m / |max DD per lot| on the common window, as short_vol_structures_backtest; ret_per_dep is
scale-free.

PRE-REGISTERED READING (written before any result was seen)
A config "passes" on an underlying only if, at BOTH tier med and tier p90, ALL hold:
  (a) full ret_per_dep > long-only ret_per_dep (same LROLL) AND > quarterly-straddle ret_per_dep;
  (b) discovery (<2024-01-01) and holdout (>=2024-01-01) account return both > 0;
  (c) total monthly P&L stays > 0 after dropping the 3 best months;
  (d) the strangle adds P&L: strangle-only mean monthly P&L > 0 with circular block-bootstrap 95% CI low > 0.
The structure is VALIDATED only if the same config passes on BOTH 510050 and 510300; otherwise the verdict is
FAILED (ETF beta plus a short-vol overlay that loses to its benchmarks). ETF buy-and-hold per notional and the
regime table (rally / drop / range months) are reported for interpretation, not for the pass rule.
Configs tried: 12 per ETF (multiple testing: ~0.6 false passes per ETF at p<0.05 for the CI condition).

Usage:  python rolling_itm_call_short_strangle_backtest.py --underlying 510050 [--start 2019-12-23]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_OL = Path("/mnt/data/Trading/analysis/options_lab")
if MAIN_OL.exists() and MAIN_OL != HERE:        # helpers + Data/ live in the main checkout (uncommitted there)
    sys.path.insert(0, str(MAIN_OL))

import numpy as np
import pandas as pd

import data_store as ds
import broker_margin as bm
import short_vol_structures_backtest as sv

warnings.filterwarnings("ignore")
OUT_DIR = HERE / "out"
MULT = ds.MULTIPLIER
ACCOUNT, CAP = 1_000_000.0, 0.15
SPLIT = pd.Timestamp("2024-01-01")
TIERS = ["t0", "med", "p90", "flat"]
STRADDLE_SLIP = {"t0": 0.0, "med": 0.0003, "p90": 0.0005, "flat": 0.005}

DISTS = (0.03, 0.05, 0.07)
SHORT_ROLLS = ("hold", "7dte")
LONG_ROLLS = (25, 45)
LONG_MIN_DTE, LONG_DELTA, LONG_DELTA_MIN = 60, 0.90, 0.85
SHORT_MIN_DTE = 25

ITM_EDGES, ITM_MED, ITM_P90 = [30, 60, 100, 150], [.0007, .0008, .0009, .0009, .0021], [.0011, .0012, .00138, .0013, .00295]
ATM_EDGES = [10, 30, 60, 100]
ATM = {"510050": {"med": [.0001, .0001, .00015, .00025, .0008], "p90": [.00015, .00015, .00025, .00045, .0014]},
       "510300": {"med": [.0001, .00015, .00025, .0004, .0011], "p90": [.00025, .00025, .0005, .00075, .0021]}}


def half_spread(tier: str, kind: str, u: str, dte: float) -> float:
    if tier == "t0":
        return 0.0
    if tier == "flat":
        return 0.005
    if kind == "long":
        tab, edges = (ITM_MED if tier == "med" else ITM_P90), ITM_EDGES
    else:
        tab, edges = ATM[u][tier], ATM_EDGES
    return tab[int(np.searchsorted(edges, dte, side="left"))]


# ------------------------------------------------------------------ data
def prep(u: str):
    df = ds.load_chain(u)
    df = df[~df[ds.C_NAME].astype(str).str.endswith("A")]
    df = df[df[ds.C_CLOSE] > 0].drop_duplicates([ds.C_CODE, ds.C_DATE], keep="last")
    df = df.assign(_oi=df[ds.C_OI].fillna(0), _vol=df[ds.C_VOL].fillna(0))
    df = (df.sort_values([ds.C_DATE, ds.C_EXPIRY, ds.C_CP, ds.C_STRIKE, "_oi", "_vol"])
            .drop_duplicates([ds.C_DATE, ds.C_EXPIRY, ds.C_CP, ds.C_STRIKE], keep="last"))
    dates = sorted(df[ds.C_DATE].unique())
    by_date = dict(tuple(df.groupby(ds.C_DATE)))
    return df, dates, by_date


def _nearest(strikes: pd.Series, target: float, side: str, ref: float):
    c = strikes[strikes > ref] if side == "up" else strikes[strikes < ref]
    return None if c.empty else c.index[int(np.argmin(np.abs(c.values - target)))]


def _expiry(day: pd.DataFrame, dt, min_dte: int):
    for e in sorted(day[ds.C_EXPIRY].unique()):
        if (pd.Timestamp(e) - pd.Timestamp(dt)).days >= min_dte:
            return pd.Timestamp(e)
    return None


def pick_long(day, dt):
    e = _expiry(day, dt, LONG_MIN_DTE)
    if e is None:
        return None
    m = day[(day[ds.C_EXPIRY] == e) & (day[ds.C_CP] == "C") & (day[ds.C_DELTA] >= LONG_DELTA_MIN)]
    if m.empty:
        return None
    r = m.loc[(m[ds.C_DELTA] - LONG_DELTA).abs().idxmin()]
    return {"code": r[ds.C_CODE], "cp": "C", "qty": 1, "expiry": e, "strike": float(r[ds.C_STRIKE]),
            "px": float(r[ds.C_CLOSE])}


def pick_shorts(day, dt, spot, d):
    e = _expiry(day, dt, SHORT_MIN_DTE)
    if e is None:
        return None
    out = []
    for cp, tgt, side in (("C", spot * (1 + d), "up"), ("P", spot * (1 - d), "dn")):
        m = day[(day[ds.C_EXPIRY] == e) & (day[ds.C_CP] == cp)]
        if m.empty:
            return None
        k = _nearest(m.set_index(ds.C_CODE)[ds.C_STRIKE], tgt, side, spot)
        if k is None:
            return None
        r = m[m[ds.C_CODE] == k].iloc[0]
        out.append({"code": r[ds.C_CODE], "cp": cp, "qty": -1, "expiry": e, "strike": float(r[ds.C_STRIKE]),
                    "px": float(r[ds.C_CLOSE])})
    return out


def intrinsic(cp: str, k: float, s: float) -> float:
    return max(s - k, 0.0) if cp == "C" else max(k - s, 0.0)


# ------------------------------------------------------------------ engine
def simulate(by_date: dict, dates: list, d: float, short_roll: str, long_roll: int,
             roll_dte: int = 7, with_long: bool = True) -> dict:
    n = len(dates)
    legs = {"L": None, "C": None, "P": None}
    cash = {"L": 0.0, "S": 0.0}
    eq = {"L": np.zeros(n), "S": np.zeros(n)}
    mS, vL, spots = np.zeros(n), np.zeros(n), np.zeros(n)
    fills: list[tuple[int, str, str, float]] = []          # (i, comp, kind, dte) priced per unit, one side each
    forced = 0
    s_prev = np.nan

    for i, dt in enumerate(dates):
        dt = pd.Timestamp(dt)
        day = by_date[dates[i]]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])
        spots[i] = spot
        rows = day.set_index(ds.C_CODE)
        settled_short = False

        def close(role, px, cost_dte=None):
            comp = "L" if role == "L" else "S"
            cash[comp] += legs[role]["qty"] * px
            if cost_dte is not None:
                fills.append((i, comp, "long" if role == "L" else "short", cost_dte))
            legs[role] = None

        for role in ("L", "C", "P"):
            leg = legs[role]
            if leg is None:
                continue
            present = leg["code"] in rows.index
            if dt >= leg["expiry"]:
                k = float(rows.at[leg["code"], ds.C_STRIKE]) if present else leg["strike"]
                close(role, intrinsic(leg["cp"], k, spot))               # cash settlement, no spread
                settled_short |= role != "L"
            elif not present or abs(float(rows.at[leg["code"], ds.C_STRIKE]) - leg["strike"]) > 1e-9:
                forced += 1
                close(role, leg["last"], (leg["expiry"] - dt).days)
            else:
                leg["last"] = float(rows.at[leg["code"], ds.C_CLOSE])
                dte = (leg["expiry"] - dt).days
                if (role == "L" and dte <= long_roll) or (role != "L" and short_roll == "7dte" and dte <= roll_dte):
                    close(role, leg["last"], dte)

        def open_(leg, role):
            leg["last"] = leg["px"]
            legs[role] = leg
            comp = "L" if role == "L" else "S"
            cash[comp] -= leg["qty"] * leg["px"]
            fills.append((i, comp, "long" if role == "L" else "short", (leg["expiry"] - dt).days))

        if legs["L"] is None and with_long:
            sel = pick_long(day, dt)
            if sel:
                open_(sel, "L")
        if legs["C"] is None and legs["P"] is None and not settled_short:
            sel = pick_shorts(day, dt, spot, d)
            if sel:
                open_(sel[0], "C")
                open_(sel[1], "P")

        for comp, roles in (("L", ("L",)), ("S", ("C", "P"))):
            eq[comp][i] = cash[comp] + sum(legs[r]["qty"] * legs[r]["last"] for r in roles if legs[r])
        if legs["L"]:
            vL[i] = legs["L"]["last"] * MULT
        if legs["C"] and legs["P"] and not np.isnan(s_prev):
            def sp(r):
                v = rows.at[legs[r]["code"], ds.C_PREV_SETTLE] if legs[r]["code"] in rows.index else np.nan
                return float(v) if pd.notna(v) else legs[r]["last"]
            mS[i] = bm.combo_short_margin(legs["C"]["strike"], legs["P"]["strike"], s_prev, sp("C"), sp("P"))
        s_prev = spot
    return {"dates": pd.DatetimeIndex(dates), "eq": eq, "mS": mS, "vL": vL, "spot": spots, "fills": fills,
            "forced": forced}


def tier_pnl(res: dict, comp: str, tier: str, u: str) -> np.ndarray:
    """daily P&L per lot in yuan for component 'L' | 'S' | 'LS' at a spread tier."""
    n = len(res["dates"])
    out = np.zeros(n)
    for c in comp:
        cost = np.zeros(n)
        for i, cc, kind, dte in res["fills"]:
            if cc == c:
                cost[i] += half_spread(tier, kind, u, dte)
        out += np.diff(np.concatenate([[0.0], res["eq"][c] - np.cumsum(cost)]))
    return out * MULT


# ------------------------------------------------------------------ metrics
def stats(p_unit: pd.Series, dep: pd.Series, start: pd.Timestamp) -> dict:
    p_unit, dep = p_unit[p_unit.index >= start], dep[dep.index >= start]
    cum = p_unit.cumsum()
    dd = (cum - cum.cummax()).min()
    if not dd < 0:
        return {}
    lots = CAP * ACCOUNT / abs(dd)
    yrs = (p_unit.index[-1] - p_unit.index[0]).days / 365.25
    p = p_unit * lots
    dp = dep * lots / ACCOUNT
    m = p.resample("ME").sum() / ACCOUNT
    top3 = m.nlargest(3).sum()
    yr = p.groupby(p.index.year).sum()
    return {"lots": lots, "ann_ret": p.sum() / ACCOUNT / yrs, "ret_per_dep": p.sum() / ACCOUNT / yrs / dp[dp > 0].mean(),
            "sharpe": p.mean() / p.std() * np.sqrt(244) if p.std() > 0 else np.nan,
            "disc": p[p.index < SPLIT].sum() / ACCOUNT, "hold": p[p.index >= SPLIT].sum() / ACCOUNT,
            "drop3": m.sum() - top3, "pos_yrs": float((yr > 0).mean()),
            "worst20": p.rolling(20).sum().min() / ACCOUNT, "avg_dep": dp[dp > 0].mean(), "peak_dep": dp.max(),
            "total": m.sum()}


def boot_lo(x: np.ndarray, block: int = 6, n: int = 5000, seed: int = 0) -> float:
    if len(x) < 2 * block:
        return np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(len(x) / block))
    idx = (rng.integers(0, len(x), size=(n, nb))[:, :, None] + np.arange(block)[None, None, :]) % len(x)
    return float(np.percentile(x[idx.reshape(n, -1)[:, :len(x)]].mean(axis=1), 2.5))


def cfg_label(d, sr, lr) -> str:
    return f"d{d:.0%}_{sr}_L{lr}"


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--start", default="2019-12-23")
    args = ap.parse_args()
    u, start = args.underlying, pd.Timestamp(args.start)
    pd.set_option("display.width", 250)
    df, dates, by_date = prep(u)
    idx = pd.DatetimeIndex(dates)
    sv.START = start

    # benchmarks ---------------------------------------------------------
    bench = {}
    for t in TIERS:
        r = sv.run(("straddle",), STRADDLE_SLIP[t], by_date, dates)["eq"]
        bench[t] = (r["equity"].diff().fillna(0.0) * MULT, r["margin"])
    etf = ds.load_etf_ohlc(u)
    spot_raw = pd.Series([float(by_date[d][ds.C_UND_CLOSE].iloc[0]) for d in dates], index=idx)
    adj = etf.set_index("date")["close"].reindex(idx).ffill()
    etf_p = (adj.pct_change().fillna(0.0) * spot_raw.shift(1).bfill() * MULT)
    etf_dep = spot_raw * MULT

    rows, monthly = [], {}
    n_forced = {}
    for d in DISTS:
        for sr in SHORT_ROLLS:
            for lr in LONG_ROLLS:
                lab = cfg_label(d, sr, lr)
                res = simulate(by_date, dates, d, sr, lr)
                n_forced[lab] = res["forced"]
                depL, depS = pd.Series(res["vL"], index=idx), pd.Series(res["mS"], index=idx)
                for t in TIERS:
                    for comp, name, dep in (("LS", "full", depL + depS), ("L", "long", depL), ("S", "strangle", depS)):
                        p = pd.Series(tier_pnl(res, comp, t, u), index=idx)
                        s = stats(p, dep, start)
                        if not s:
                            continue
                        rows.append({"cfg": lab, "d": d, "short_roll": sr, "long_roll": lr, "tier": t, "comp": name, **s})
                        if t == "med":
                            monthly[(lab, name)] = p[p.index >= start].resample("ME").sum()
                        if name == "strangle":
                            mm = (p[p.index >= start].resample("ME").sum() / ACCOUNT).values
                            rows[-1]["boot_lo_monthly"] = boot_lo(mm)
                print(f"{lab:<16} forced-closes {res['forced']}", flush=True)
    out = pd.DataFrame(rows)

    bm_rows = []
    for t in TIERS:
        s = stats(bench[t][0], bench[t][1], start)
        if s:
            bm_rows.append({"cfg": "quarterly_straddle", "tier": t, "comp": "bench", **s})
    se = stats(etf_p, etf_dep, start)
    bm_rows.append({"cfg": "ETF_buy_hold", "tier": "t0", "comp": "bench", **se})
    bench_df = pd.DataFrame(bm_rows)

    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / f"rolling_itm_call_short_strangle_{u}_{start:%Y%m%d}.csv", index=False)
    bench_df.to_csv(OUT_DIR / f"rolling_itm_call_short_strangle_{u}_{start:%Y%m%d}_bench.csv", index=False)

    cols = ["cfg", "comp", "lots", "ann_ret", "ret_per_dep", "sharpe", "disc", "hold", "drop3", "pos_yrs",
            "worst20", "avg_dep"]
    for t in ("med", "p90"):
        print(f"\n== {u} {start.date()}.. tier {t}: sized to max DD 15% of 1.0m; ret_per_dep = ann return / avg deposit ==")
        print(out[(out.tier == t) & (out.comp == "full")][cols].round(3).to_string(index=False))
        print(f"-- long-only / strangle-only (tier {t}) --")
        print(out[(out.tier == t) & (out.comp != "full")].sort_values(["comp", "cfg"])[cols + ["boot_lo_monthly"]]
              .round(3).to_string(index=False))
    print("\n== benchmarks (same window, sized to 15% DD) ==")
    print(bench_df[["cfg", "tier", "lots", "ann_ret", "ret_per_dep", "sharpe", "disc", "hold", "drop3", "worst20",
                    "avg_dep"]].round(3).to_string(index=False))

    # pre-registered pass rule ------------------------------------------
    passes = []
    for lab in out["cfg"].unique():
        ok = True
        for t in ("med", "p90"):
            g = out[(out.cfg == lab) & (out.tier == t)].set_index("comp")
            b = bench_df[(bench_df.cfg == "quarterly_straddle") & (bench_df.tier == t)]
            if not {"full", "long", "strangle"} <= set(g.index) or b.empty:
                ok = False
                break
            f, lg, sg = g.loc["full"], g.loc["long"], g.loc["strangle"]
            ok &= (f.ret_per_dep > lg.ret_per_dep and f.ret_per_dep > b.iloc[0].ret_per_dep and f.disc > 0
                   and f.hold > 0 and f.drop3 > 0 and sg.total > 0 and sg.boot_lo_monthly > 0)
        if ok:
            passes.append(lab)
    print(f"\nconfigs passing the pre-registered rule on {u}: {passes or 'NONE'}")
    print(f"ETF buy-hold at same sizing: ann {se['ann_ret']:+.3f} ret_per_dep {se['ret_per_dep']:.3f} "
          f"worst20 {se['worst20']:+.3f}  | configs tried: {len(n_forced)}")

    # regime table (months by underlying return), tier med, full vs long vs strangle ----------
    mret = spot_raw[spot_raw.index >= start].resample("ME").last().pct_change()
    reg = pd.cut(mret, [-1, -0.04, 0.04, 5], labels=["drop<-4%", "range", "rally>+4%"])
    rr = []
    for (lab, name), m in monthly.items():
        for r in reg.cat.categories:
            sel = m[m.index.isin(reg[reg == r].index)]
            rr.append({"cfg": lab, "comp": name, "regime": r, "n": len(sel), "mean_yuan_per_lot": sel.mean() * 1.0})
    reg_df = pd.DataFrame(rr)
    reg_df.to_csv(OUT_DIR / f"rolling_itm_call_short_strangle_{u}_{start:%Y%m%d}_regime.csv", index=False)
    pv = reg_df.pivot_table(index=["cfg", "comp"], columns="regime", values="mean_yuan_per_lot")
    print("\nmean monthly P&L per lot (yuan) by market regime, tier med:")
    print(pv.round(0).to_string())
    print("\nforced closes (row vanished / strike changed):", n_forced)


if __name__ == "__main__":
    main()
