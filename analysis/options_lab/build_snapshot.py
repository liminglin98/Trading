"""Build the phone-app snapshot (out/snapshot.json) from positions.csv + live data.

Everything here reuses existing code -- nothing is re-derived:
  * open legs ........ positions.load()            (positions.csv, one row per leg)
  * leg quotes ....... minute_updater CSV, last row per contract; mid price and
                       IV via minute_updater.mid_price_for_row / iv_for_row
  * greeks ........... bs_iv.bs_* at that IV
  * vol signals ...... alerts_core.short_vol_signal_now / vol_signal_now /
                       vrp_trigger_level / vol_trigger_level  (daily-close series,
                       same numbers as the daily push)
  * index/ETF quotes . updater._hq_batch (Sina)
  * margin ........... margin.compute_deposit      (entry-snapshot formula, as in
                       live_book_stress.py)

Units written to the JSON (see "units"): vega is per 1 vol point, theta per
calendar day, delta in ETF shares and yuan-of-underlying, P&L in yuan.

The P&L curves are per underlying (a 50ETF and a 300ETF book cannot share a
spot axis): one whole-book "today" curve (time held fixed, plus realized P&L of
closed legs inside still-open combos as a constant) and one SEPARATE curve per
open expiry E containing only the legs expiring on E, valued at intrinsic on E
(`pnl_today` = the same legs marked today, `legs`/`lots` = their count).
Realized P&L is not split by expiry, so it appears only in "today".

Usage:
    python build_snapshot.py                 # writes out/snapshot.json
    python build_snapshot.py --upload        # ...and publishes it
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import alerts_core as ac
import bs_iv
import data_store as ds
import margin as mg
import minute_updater as MU
import positions
import updater as U

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "snapshot.json"
CN = timezone(timedelta(hours=8))
R = bs_iv.RISK_FREE_DEFAULT
MULT = ds.MULTIPLIER

# (key, display name, Sina symbol)
MARKET = [
    ("000001", "上证指数", "sh000001"),
    ("510050", "50ETF", "sh510050"),
    ("510300", "300ETF", "sh510300"),
    ("159915", "创业板ETF", "sz159915"),
    ("588000", "科创50ETF", "sh588000"),
]
VOL_UNDERLYINGS = [("510050", "50ETF"), ("510300", "300ETF")]
CURVE_SPAN = 0.15
CURVE_POINTS = 61


def _num(x):
    """JSON-safe float (NaN/inf -> None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, 6) if math.isfinite(x) else None


# ------------------------------------------------------------------ market
def market_quotes() -> list[dict]:
    got = U._hq_batch([s for _, _, s in MARKET])
    out = []
    for key, name, sym in MARKET:
        v = got.get(sym)
        try:
            prev, last = float(v[2]), float(v[3])
            if not (prev > 0 and last > 0):
                raise ValueError
            out.append({"code": key, "name": name, "price": last, "prev_close": prev,
                        "change_pct": round(last / prev - 1, 6),
                        "quote_time": f"{v[30]} {v[31]}"})
        except (TypeError, ValueError, IndexError):
            out.append({"code": key, "name": name, "price": None, "prev_close": None,
                        "change_pct": None, "quote_time": None})
    return out


# ------------------------------------------------------------------ minute data
def latest_minute_df(u: str) -> tuple[pd.DataFrame | None, str | None]:
    """Most recent option_minute_<u>_*.csv (today's if present)."""
    files = sorted(MU.DATA_DIR.glob(f"option_minute_{u}_*.csv"))
    if not files:
        return None, None
    df = pd.read_csv(files[-1], dtype={"code": str}, encoding="utf-8-sig")
    return df, str(df["timestamp"].max()) if len(df) else None


def last_rows(df: pd.DataFrame | None) -> dict[str, pd.Series]:
    if df is None or df.empty:
        return {}
    return {c: g.iloc[-1] for c, g in df.groupby("code")}


# ------------------------------------------------------------------ legs
def build_legs(today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, float, list[dict]]:
    """(live per-contract table, stale rows, realized yuan, warnings)."""
    pos = positions.load()
    pos["expiry"] = pd.to_datetime(pos["expiry"])
    op = pos[pos["status"] != "closed"].copy()
    stale = op[op["expiry"].dt.normalize() < today]
    op = op[op["expiry"].dt.normalize() >= today].copy()
    op["signed"] = op["direction"] * op["qty"]
    op["entry_val"] = op["signed"] * op["entry_price"]

    # realized P&L of closed legs that belong to a combo that still has open legs
    open_pids = set(op["position_id"]) | set(stale["position_id"])
    cl = pos[(pos["status"] == "closed") & pos["position_id"].isin(open_pids)
             & pos["close_price"].notna()]
    realized = float((cl["direction"] * cl["qty"] * (cl["close_price"] - cl["entry_price"])).sum() * MULT)

    warns: list[dict] = []
    rows = []
    cache: dict[str, tuple] = {}
    for (u, code, cp, strike, expiry), g in op.groupby(
            ["underlying", "leg_code", "cp", "strike", "expiry"]):
        if u not in cache:
            df, ts = latest_minute_df(u)
            cache[u] = (last_rows(df), ts)
        q, ts = cache[u]
        lots = float(g["signed"].sum())
        if lots == 0:
            continue
        entry = float(g["entry_val"].sum() / lots)
        r = q.get(code)
        mark = spot = dte = iv = float("nan")
        src = "none"
        if r is not None:
            mark = MU.mid_price_for_row(r)
            spot, dte = float(r["spot"]), float(r["dte"])
            iv = MU.iv_for_row(r)
            src = "minute"
        if not (mark > 0):
            warns.append({"leg_code": code, "msg": "no usable minute quote; mark falls back to entry"})
            mark, src = entry, "entry_fallback"
        if not (dte > 0):
            dte = float((expiry.normalize() - today).days)
        rows.append({"und": u, "code": code, "name": str(g["leg_name"].iloc[-1]), "cp": cp,
                     "strike": float(strike), "expiry": expiry, "dte": dte, "lots": lots,
                     "entry": entry, "mark": mark, "spot": spot, "iv": iv,
                     "pos_names": sorted(set(g["pos_name"])), "src": src, "quote_ts": ts})
    return pd.DataFrame(rows), stale, realized, warns


def leg_greeks(r: pd.Series) -> dict:
    s, k, t, sig = r["spot"], r["strike"], r["dte"] / 365.0, r["iv"]
    ok = all(math.isfinite(x) for x in (s, k, t, sig)) and sig > 0
    if not ok:
        return dict(delta=None, gamma=None, vega=None, theta=None)
    return dict(delta=_num(bs_iv.bs_delta(r["cp"], s, k, t, sig, R)),
                gamma=_num(bs_iv.bs_gamma(s, k, t, sig, R)),
                vega=_num(bs_iv.bs_vega(s, k, t, sig, R) * 0.01),          # per 1 vol point
                theta=_num(bs_iv.bs_theta(r["cp"], s, k, t, sig, R) / 365.0))  # per day


# ------------------------------------------------------------------ curves
def _value(cp: str, s: float, k: float, t_years: float, iv: float, fallback: float) -> float:
    if t_years <= 0:
        return max(s - k, 0.0) if cp == "C" else max(k - s, 0.0)
    if not (math.isfinite(iv) and iv > 0):
        return fallback
    return bs_iv.bs_price(cp, s, k, t_years, iv, R)


def pnl_curves(book: pd.DataFrame, realized_by_und: dict[str, float], today: pd.Timestamp) -> dict:
    out = {}
    for u, g in book.groupby("und"):
        spot0 = float(g["spot"].dropna().iloc[0]) if g["spot"].notna().any() else None
        if spot0 is None:
            continue
        grid = np.linspace(spot0 * (1 - CURVE_SPAN), spot0 * (1 + CURVE_SPAN), CURVE_POINTS)
        real = realized_by_und.get(u, 0.0)
        curves = []

        def total(fn):
            ys = []
            for s in grid:
                pnl = real
                for _, r in g.iterrows():
                    pnl += r["lots"] * (fn(r, float(s)) - r["entry"]) * MULT
                ys.append(round(pnl, 0))
            return ys

        curves.append({"kind": "today", "label": "今天(时间不变)",
                       "pnl": total(lambda r, s: _value(r["cp"], s, r["strike"], r["dte"] / 365.0,
                                                        r["iv"], r["mark"]))})
        for e in sorted(g["expiry"].unique()):
            e = pd.Timestamp(e)
            ge = g[g["expiry"] == e]

            def sub_total(fn, ge=ge):
                return [round(sum(r["lots"] * (fn(r, float(s)) - r["entry"]) * MULT
                                  for _, r in ge.iterrows()), 0) for s in grid]

            curves.append({
                "kind": "expiry", "expiry": e.strftime("%Y-%m-%d"),
                "label": f"{e:%m-%d} 到期", "legs": int(len(ge)),
                "lots": float(ge["lots"].abs().sum()),
                # only the legs expiring on E, intrinsic value on E
                "pnl": sub_total(lambda r, s: _value(r["cp"], s, r["strike"], 0.0,
                                                     r["iv"], r["mark"])),
                # same legs marked today (time held fixed)
                "pnl_today": sub_total(lambda r, s: _value(r["cp"], s, r["strike"],
                                                           r["dte"] / 365.0, r["iv"], r["mark"])),
            })
        out[u] = {"spot": spot0, "spots": [round(float(s), 4) for s in grid], "curves": curves}
    return out


# ------------------------------------------------------------------ vol signals
def live_atm(u: str) -> dict | None:
    """Live ATM IV (call-only basis, no method bias) from the latest minute rows:
    nearest-strike call of the first expiry with dte >= 10 (same window
    alerts_core.suggest_atm_straddle uses)."""
    df, ts = latest_minute_df(u)
    q = last_rows(df)
    if not q:
        return None
    last = pd.DataFrame(q.values())
    last = last[(last["cp"] == "C") & (last["dte"] >= 10)]
    if last.empty:
        return None
    dte = last["dte"].min()
    c = last[last["dte"] == dte]
    c = c.iloc[(c["strike"] - c["spot"]).abs().argsort()[:1]].iloc[0]
    iv = MU.iv_for_row(c)
    return {"iv": _num(iv), "code": str(c["code"]), "strike": float(c["strike"]),
            "dte": int(c["dte"]), "asof": ts, "basis": "call_only_no_bias"}


_VOL_CACHE: dict = {"t": 0.0, "data": None}
VOL_TTL_SEC = 3600  # daily-close series only change after the EOD updates; live_atm below is always fresh


def vol_block(refresh: bool = True) -> list[dict]:
    """iv_pct/vrp/rv_pct block. The daily-series part costs ~10 s (reads the full
    chain history), so a loop passes refresh=False to reuse it for VOL_TTL_SEC."""
    import copy
    import time
    if refresh or _VOL_CACHE["data"] is None or time.monotonic() - _VOL_CACHE["t"] > VOL_TTL_SEC:
        _VOL_CACHE["data"], _VOL_CACHE["t"] = _vol_daily(), time.monotonic()
    out = copy.deepcopy(_VOL_CACHE["data"])
    for d in out:
        d["live_atm"] = live_atm(d["u"])
    return out


def _vol_daily() -> list[dict]:
    short = {a["u"]: a for a in ac.SHORT_VOL_SIGNAL_ALERTS}
    ivp = {a["u"]: a for a in ac.IV_PCT_SHORT_ALERTS}
    out = []
    for u, name in VOL_UNDERLYINGS:
        d: dict = {"u": u, "name": name}
        try:
            s = ac.short_vol_signal_now(u, "iv_pct")
            d["iv_pct"] = _num(s["value"]) if s else None
            d["iv_pct_asof"] = str(s["asof"].date()) if s else None
            d["iv_pct_threshold"] = ivp.get(u, {}).get("threshold")
            s = ac.short_vol_signal_now(u, "vrp")
            d["vrp"] = _num(s["value"]) if s else None
            d["vrp_asof"] = str(s["asof"].date()) if s else None
            d["vrp_threshold"] = short.get(u, {}).get("threshold")
            s = ac.vol_signal_now(u, "rv", "rv_pct")
            d["rv_pct"] = _num(s["value"]) if s else None
            d["rv_pct_asof"] = str(s["asof"].date()) if s else None
            thr = d["vrp_threshold"]
            t = ac.vrp_trigger_level(u, thr) if thr is not None else None
            if t:
                d.update(atm_iv_close=_num(t["iv_now"]), rv_20d=_num(t["rv_now"]),
                         iv_needed_for_vrp=_num(t["needed"]))
            t = ac.vol_trigger_level(u, "iv_low", "iv_pct", d["iv_pct_threshold"] or 0.9)
            if t:
                d["iv_level_for_iv_pct_threshold"] = _num(t["level"])
        except Exception as e:  # one broken signal must not kill the snapshot
            d["error"] = f"{type(e).__name__}: {e}"
        out.append(d)
    return out


# ------------------------------------------------------------------ end-of-day P&L baseline
EOD_CACHE = Path(__file__).resolve().parent / "out" / "eod_baseline.json"


def eod_baseline(today: pd.Timestamp) -> dict | None:
    """Total P&L of the current book as of the LAST OFFICIAL CLOSE before today (chain close prices),
    so the phone app can show today's P&L live as  (portfolio unrealized+realized) - eod.total_pnl
    + eod.closed_today_delta.

    Same position set as portfolio.*: every combo that still has an open leg, valued from its own
    pnl_history (legs open after the close contribute 0; closed legs frozen at close price).
    closed_today_delta: combos FULLY closed today leave the portfolio totals, so their day P&L
    (final realized - their P&L at the last close) is added back here.
    Cached on (last close date, positions.csv content) -- recomputed only when either changes."""
    import hashlib
    pos = positions.load()
    if pos.empty:
        return None
    key = f"{today.date()}|{hashlib.md5(ds.POSITIONS_CSV.read_bytes()).hexdigest()}"
    try:
        c = json.loads(EOD_CACHE.read_text())
        if c.get("key") == key:
            return c["eod"]
    except (OSError, ValueError, KeyError):
        pass
    live = set(pos[pos["status"] != "closed"]["position_id"])
    cd = pd.to_datetime(pos["close_date"], errors="coerce")
    fully_closed_today = set()
    for pid, g in pos.groupby("position_id"):
        if (g["status"] == "closed").all() and (cd[g.index] == today).any():
            fully_closed_today.add(pid)
    chains = {u: ds.load_chain(u) for u in pos[pos["position_id"].isin(live | fully_closed_today)]["underlying"].unique()}
    asof = None
    for ch in chains.values():
        d = ch[ch[ds.C_DATE] < today][ds.C_DATE].max()
        asof = d if asof is None or d > asof else asof
    if asof is None:
        return None
    total, by_und, closed_delta = 0.0, {}, 0.0
    for pid in live | fully_closed_today:
        u = str(pos[pos["position_id"] == pid]["underlying"].iloc[0])
        h = positions.pnl_history(pid, chains)
        h = h[h["date"] <= asof] if not h.empty else h
        at_close = float(h["pnl_yuan"].iloc[-1]) if len(h) else 0.0
        if pid in fully_closed_today:
            g = pos[pos["position_id"] == pid]
            final = float((g["direction"] * g["qty"] * (g["close_price"] - g["entry_price"])).sum() * MULT)
            closed_delta += final - at_close
        else:
            total += at_close
            by_und[u] = by_und.get(u, 0.0) + at_close
    eod = {"asof": str(asof.date()), "total_pnl": round(total, 0),
           "by_underlying": {u: round(v, 0) for u, v in by_und.items()},
           "closed_today_delta": round(closed_delta, 0),
           "formula": "today_pnl = (portfolio.unrealized_pnl + portfolio.realized_pnl) - eod.total_pnl + eod.closed_today_delta"}
    try:
        EOD_CACHE.write_text(json.dumps({"key": key, "eod": eod}, ensure_ascii=False))
    except OSError:
        pass
    return eod


# ------------------------------------------------------------------ main
def build(refresh_vol: bool = True) -> dict:
    now = datetime.now(CN)
    today = pd.Timestamp(now.date())
    book, stale, realized, warns = build_legs(today)

    legs, by_und = [], {}
    if not book.empty:
        book["pnl"] = book["lots"] * (book["mark"] - book["entry"]) * MULT
        for _, r in book.iterrows():
            g = leg_greeks(r)
            legs.append({"und": r["und"], "code": r["code"], "name": r["name"], "cp": r["cp"],
                         "strike": r["strike"], "expiry": r["expiry"].strftime("%Y-%m-%d"),
                         "dte": int(r["dte"]), "lots": r["lots"], "entry": _num(r["entry"]),
                         "mark": _num(r["mark"]), "iv": _num(r["iv"]), "pnl": round(r["pnl"], 0),
                         "mark_source": r["src"], "positions": r["pos_names"], **g})

    # realized split by underlying (needed for the per-underlying curves)
    pos = positions.load()
    cl = pos[(pos["status"] == "closed") & pos["close_price"].notna()]
    open_pids = set(pos[pos["status"] != "closed"]["position_id"])
    cl = cl[cl["position_id"].isin(open_pids)]
    realized_by_und = {u: float((g["direction"] * g["qty"] * (g["close_price"] - g["entry_price"])).sum() * MULT)
                       for u, g in cl.groupby("underlying")}

    portfolio = {"unrealized_pnl": 0.0, "realized_pnl": round(realized, 0), "by_underlying": {}}
    if not book.empty:
        for u, g in book.groupby("und"):
            spot = float(g["spot"].dropna().iloc[0]) if g["spot"].notna().any() else None
            gg = [x for x in legs if x["und"] == u]

            def tot(key, scale=1.0):
                return sum((x[key] or 0.0) * x["lots"] for x in gg) * scale

            ml = pd.DataFrame([{"expiry": r["expiry"], "cp": r["cp"], "strike": r["strike"],
                                "direction": int(np.sign(r["lots"])), "qty": int(abs(r["lots"])),
                                "entry_price": r["mark"]} for _, r in g.iterrows()])
            margin = mg.compute_deposit(ml, spot)[0] if spot else None
            portfolio["by_underlying"][u] = {
                "spot": spot,
                "unrealized_pnl": round(float(g["pnl"].sum()), 0),
                "realized_pnl": round(realized_by_und.get(u, 0.0), 0),
                "delta_shares": round(tot("delta", MULT), 0),
                "delta_yuan": round(tot("delta", MULT) * spot, 0) if spot else None,
                "gamma_shares_per_1pct": round(tot("gamma", MULT) * spot * spot * 0.01, 0) if spot else None,
                "vega_yuan_per_volpt": round(tot("vega", MULT), 0),
                "theta_yuan_per_day": round(tot("theta", MULT), 0),
                "margin_yuan": round(float(margin), 0) if margin is not None else None,
                "n_legs": len(gg), "net_lots": float(g["lots"].sum()),
            }
        portfolio["unrealized_pnl"] = round(float(book["pnl"].sum()), 0)
    try:
        portfolio["eod"] = eod_baseline(today)
    except Exception as e:  # the baseline must never kill the snapshot
        portfolio["eod"] = None
        warns.append({"leg_code": "", "msg": f"eod baseline failed: {type(e).__name__}: {e}"})

    snap = {
        "schema": 1,
        "generated_at": now.isoformat(timespec="seconds"),
        "units": {"pnl": "yuan", "delta_shares": "ETF shares (lots*10000*delta)",
                  "vega": "per 1 vol point (leg: yuan-per-share; portfolio: yuan)",
                  "theta": "per calendar day", "iv": "annualised, decimal",
                  "margin": "margin.compute_deposit, exchange formula without broker 1.15x"},
        "market": market_quotes(),
        "vol": vol_block(refresh_vol),
        "portfolio": portfolio,
        "legs": legs,
        "pnl_curves": pnl_curves(book, realized_by_und, today) if not book.empty else {},
        "warnings": warns + [{"leg_code": str(r["leg_code"]), "msg": f"open row already expired {r['expiry']:%Y-%m-%d}; excluded"}
                             for _, r in stale.iterrows()],
        "minute_data_asof": {u: latest_minute_df(u)[1] for u, _ in VOL_UNDERLYINGS},
    }
    return snap


def write_snapshot(snap: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snap, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, OUT)


def cycle(upload: bool, refresh_vol: bool) -> None:
    """One build (+ upload). Never raises: a failed cycle is logged and the next one retries."""
    stamp = datetime.now(CN).strftime("%H:%M:%S")
    if upload:
        try:  # apply any trade confirmations from the phone BEFORE building, so this snapshot has them
            import fills_inbox
            res = fills_inbox.run(rebuild=False)
            if res:
                print(f"{stamp} fills: " + ", ".join(r["status"] for r in res), flush=True)
        except Exception as e:
            print(f"{stamp} fills sync failed {type(e).__name__}: {e}", flush=True)
    try:
        snap = build(refresh_vol)
        write_snapshot(snap)
        msg = f"{stamp} built {len(snap['legs'])} legs"
        if upload:
            import upload_snapshot
            upload_snapshot.upload_and_verify(OUT, "snapshot.json")
            msg += ", uploaded+verified"
        print(msg, flush=True)
    except Exception as e:
        print(f"{stamp} CYCLE FAILED {type(e).__name__}: {e}", flush=True)
    if upload:
        try:  # re-publish alerts.json if an earlier upload from pushplus.send() failed
            import alerts_feed
            if alerts_feed.sync():
                print(f"{stamp} alerts.json re-published", flush=True)
        except Exception as e:
            print(f"{stamp} alerts sync failed {type(e).__name__}: {e}", flush=True)


SESSION_START = (9, 25)   # China time; first cycle just before the 09:30 open
SESSION_END = (15, 2)     # last cycle just after the 15:00 close


def loop(interval: int, upload: bool) -> None:
    """Cron starts this once per weekday; it exits after the post-close cycle.
    Same pattern as minute_updater.py / intraday_monitor.py."""
    import time
    today = MU.now_cn().date()
    if not MU.is_trading_day(today):
        print(f"{today} is not a trading day -- exiting.")
        return
    print(f"snapshot loop: every {interval}s from {SESSION_START} to {SESSION_END} China time", flush=True)
    first = True
    while True:
        now = MU.now_cn()
        if now.date() != today:
            print("date rolled over -- exiting.")
            return
        hm = (now.hour, now.minute)
        if hm < SESSION_START:
            time.sleep(20)
            continue
        if hm > SESSION_END:
            cycle(upload, refresh_vol=True)   # closing snapshot
            print("past close -- exiting.")
            return
        t0 = time.monotonic()
        cycle(upload, refresh_vol=first)
        first = False
        time.sleep(max(1.0, interval - (time.monotonic() - t0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="publish to the snapshot server afterwards")
    ap.add_argument("--loop", action="store_true",
                    help="trading-day loop: build every --interval s in session, one closing build, then exit")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    if args.loop:
        loop(args.interval, args.upload)
        return
    snap = build()
    write_snapshot(snap)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB), {len(snap['legs'])} legs")
    if args.upload:
        import upload_snapshot
        upload_snapshot.upload_and_verify(OUT, "snapshot.json")
        print("published and verified")


if __name__ == "__main__":
    main()
