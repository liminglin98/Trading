# -*- coding: utf-8 -*-
"""
butterfly_timing_backtest.py
============================
Does the "implied-vs-model" signal actually pay? Walk-forward backtest of a vol
timing rule on 50ETF options, priced with the REAL historical IV surface.

Each month-end t (using only data known by t):
  1. Train the factor model on history <= t; draw next-month return draws.
  2. Pull the real SSE IV smile at t (option_risk_indicator_sse) for the ~1-month
     expiry; build each strike's vol from the OTM side (puts<spot, calls>spot).
  3. ratio = ATM implied vol (annualized) / model annualized vol.
        ratio >= HI_RATIO  -> LONG BUTTERFLY  (implied rich -> sell dispersion)
        ratio <= LO_RATIO  -> LONG STRADDLE   (implied cheap -> buy vol)
        else               -> FLAT
     Trade only if the model also says EV>0; size at FRACTIONAL Kelly.
  4. Exit on whichever fires first:
       - PROFIT_TAKE: scan actual daily 50ETF closes, marking the position at
         FIXED entry IV (cheap, no per-day SSE calls); trigger when value hits
         the target (butterfly: frac of max profit; straddle: return on cost).
       - STOP_LOSS  : same daily scan; stop out when the mark has lost that
         fraction of the premium paid (value <= cost*(1-STOP_LOSS)).
       - EXIT_DTE   : roll/close when that many days to expiry remain.
       - else hold to the option's actual expiry (intrinsic P&L).
     For a target/roll exit, re-fetch the REAL SSE IV surface on the chosen date
     and mark the same strikes to market (BS); expiry exits use intrinsic.
Compare the timing rule's equity curve vs always-on butterfly vs buy & hold.

No look-ahead: model uses data <= t; entry priced at t; exit/outcome dated > t.

RUN IN A NORMAL PYTHON ENV (needs internet; ~100 SSE calls, give it a few min).
Needs: akshare pandas numpy scipy matplotlib + butterfly_factor_forecast.py.
"""

from __future__ import annotations

import re
import time
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

warnings.filterwarnings("ignore")

from butterfly_factor_forecast import load_panel, build_xy, fit_predict_conjugate, SPOT_COL  # noqa
import akshare as ak

# =============================================================================
# CONFIG
# =============================================================================
OUTDIR    = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

MIN_TRAIN = 48           # months before first OOS trade (matches the model)
HI_RATIO  = 1.10         # implied/model above this -> long butterfly
LO_RATIO  = 0.85         # below this -> long straddle
KELLY_FRAC = 0.33        # fraction of full Kelly actually deployed
KELLY_CAP  = 0.50        # hard cap on bankroll fraction per trade
R_FREE     = 0.02
MAX_WING_STEPS = 6
UNDERLYING = "510050"
TARGET_DTE = 30          # prefer the expiry closest to ~30 calendar days out
EXIT_DTE   = 7           # roll/early-exit cap: close when this many days to expiry remain.
                         # None -> no DTE cap (only the profit target / hold to expiry).
PROFIT_TAKE = 0.50       # take-profit, whichever fires first with EXIT_DTE.
                         # butterfly: fraction of MAX profit; straddle: return on cost.
                         # None -> no profit target (DTE-only / hold-to-expiry).
STOP_LOSS  = 0.50        # stop-out when the position has lost this fraction of the
                         # premium paid (mark <= cost*(1-STOP_LOSS)). None -> no stop.
EXIT_SEARCH = 6          # max days to step looking for a trading day w/ IV near exit
CONTRACT_RE = re.compile(r"(\d{6})([CP])(\d{4})[MA](\d+)")

# =============================================================================
# Black-Scholes
# =============================================================================
def bs_call(S, K, T, r, sig):
    if sig <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - sig * np.sqrt(T))

def bs_put(S, K, T, r, sig):
    return bs_call(S, K, T, r, sig) - S + K * np.exp(-r * T)        # parity

def kelly_opt(net_ret, fgrid=np.linspace(0.0, 0.99, 100)):
    best_f, best_g = 0.0, 0.0
    for f in fgrid:
        v = 1.0 + f * net_ret
        if np.any(v <= 0):
            continue
        g = np.mean(np.log(v))
        if g > best_g:
            best_g, best_f = g, f
    return best_f, best_g

def fourth_wednesday(yymm):
    yy, mm = int(yymm[:2]), int(yymm[2:])
    d = dt.date(2000 + yy, mm, 1)
    return d + dt.timedelta(days=(2 - d.weekday()) % 7 + 21)        # Wed=2; +3 weeks

# =============================================================================
# Historical IV smile
# =============================================================================
def parse_cid(cid):
    m = CONTRACT_RE.search(str(cid))
    if not m:
        return None
    und, cp, exp, k = m.groups()
    return und, cp, exp, int(k) / 1000.0

def fetch_iv_table(trade_date):
    """Return DataFrame [type, expiry, strike, iv] for 50ETF on trade_date, or None."""
    df = ak.option_risk_indicator_sse(date=trade_date.strftime("%Y%m%d"))
    if df is None or len(df) == 0:
        return None
    cid = next(c for c in df.columns if c.upper() == "CONTRACT_ID")
    ivc = next(c for c in df.columns if "VOLAT" in c.upper())
    recs = []
    for c, iv in zip(df[cid], pd.to_numeric(df[ivc], errors="coerce")):
        p = parse_cid(c)
        if p and p[0] == UNDERLYING and np.isfinite(iv) and iv > 0:
            recs.append((p[1], p[2], p[3], iv))
    if not recs:
        return None
    return pd.DataFrame(recs, columns=["type", "expiry", "strike", "iv"])

def otm_smile(iv_tbl, expiry, spot):
    """strike -> IV using the OTM side (puts below spot, calls above)."""
    sub = iv_tbl[iv_tbl["expiry"] == expiry]
    out = {}
    for k in sorted(sub["strike"].unique()):
        side = "P" if k < spot else "C"
        row = sub[(sub["strike"] == k) & (sub["type"] == side)]
        alt = sub[(sub["strike"] == k) & (sub["type"] != side)]
        iv = row["iv"].mean() if len(row) else (alt["iv"].mean() if len(alt) else np.nan)
        if np.isfinite(iv) and iv > 0:
            out[k] = iv
    return pd.Series(out).sort_index()

def choose_expiry(iv_tbl, t):
    """Pick the listed expiry whose 4th-Wed date is closest to t + TARGET_DTE,
    and strictly after t. Returns (expiry_code, expiry_date)."""
    best = None
    for e in sorted(iv_tbl["expiry"].unique()):
        ed = fourth_wednesday(e)
        dte = (ed - t.date()).days
        if dte <= 5:                      # skip about-to-expire / past
            continue
        score = abs(dte - TARGET_DTE)
        if best is None or score < best[0]:
            best = (score, e, ed)
    return (best[1], best[2]) if best else (None, None)

# =============================================================================
# Structure pricing / payoff (priced from the smile IVs at entry)
# =============================================================================
def price_butterfly(smile, spot, T, n):
    ks = smile.index.values
    i0 = int(np.argmin(np.abs(ks - spot)))
    lo, hi = i0 - n, i0 + n
    if lo < 0 or hi >= len(ks):
        return None
    klow, kmid, khigh = ks[lo], ks[i0], ks[hi]
    c = lambda K: bs_call(spot, K, T, R_FREE, smile[K])
    cost = c(klow) - 2 * c(kmid) + c(khigh)
    return dict(kind="butterfly", strikes=(klow, kmid, khigh), cost=cost)

def payoff_butterfly(S, strikes):
    klow, kmid, khigh = strikes
    return (np.maximum(S - klow, 0) - 2 * np.maximum(S - kmid, 0) + np.maximum(S - khigh, 0))

def price_straddle(smile, spot, T):
    ks = smile.index.values
    K = ks[int(np.argmin(np.abs(ks - spot)))]
    cost = bs_call(spot, K, T, R_FREE, smile[K]) + bs_put(spot, K, T, R_FREE, smile[K])
    return dict(kind="straddle", strikes=(K,), cost=cost)

def payoff_straddle(S, strikes):
    return np.abs(S - strikes[0])

# =============================================================================
# Mark-to-market valuation at an early-exit date (re-priced from exit smile)
# =============================================================================
def smile_iv(smile, K):
    """IV for an arbitrary strike, linearly interpolated across the exit smile."""
    ks = smile.index.values.astype(float)
    return float(np.interp(float(K), ks, smile.values.astype(float)))

def value_butterfly(smile, spot, T, strikes):
    """BS value of the SAME long call butterfly at exit (fixed strikes, new spot/T/IV)."""
    klow, kmid, khigh = strikes
    c = lambda K: bs_call(spot, K, T, R_FREE, smile_iv(smile, K))
    return c(klow) - 2 * c(kmid) + c(khigh)

def value_straddle(smile, spot, T, strikes):
    K = strikes[0]
    iv = smile_iv(smile, K)
    return bs_call(spot, K, T, R_FREE, iv) + bs_put(spot, K, T, R_FREE, iv)

def profit_target_value(side, cost, strikes):
    """Absolute position value at which the take-profit fires.
    butterfly: cost + PROFIT_TAKE * (max_profit - cost), where max_profit = wing width.
    straddle : cost * (1 + PROFIT_TAKE)  (unbounded payoff -> return-on-cost)."""
    if PROFIT_TAKE is None:
        return None
    if side == "butterfly":
        klow, kmid, _ = strikes
        max_profit = (kmid - klow) - cost          # long fly max payoff = wing width
        if max_profit <= 0:
            return None
        return cost + PROFIT_TAKE * max_profit
    return cost * (1.0 + PROFIT_TAKE)

def find_iv_on_or_before(target_date, max_back=EXIT_SEARCH):
    """Step back up to max_back days to find a trading day with an SSE IV table.
    Returns (iv_tbl, actual_date) or (None, None)."""
    for back in range(0, max_back + 1):
        d = pd.Timestamp((target_date - pd.Timedelta(days=back)).date())
        try:
            tbl = fetch_iv_table(d)
        except Exception:
            tbl = None
        if tbl is not None:
            return tbl, d
    return None, None

# =============================================================================
# Backtest
# =============================================================================
def run():
    df = load_panel()
    X, y, used = build_xy(df)
    idx = X.index
    spot_m = df[SPOT_COL]                       # month-end 50ETF close (for ratio ref)

    # daily 50ETF for entry spot + realized price at option expiry
    etf = ak.fund_etf_hist_sina(symbol="sh510050")
    etf["date"] = pd.to_datetime(etf["date"])
    etf = etf.set_index("date")["close"].sort_index()
    last_daily = etf.index.max()

    rows = []
    for ti in range(MIN_TRAIN, len(X)):
        t = idx[ti]
        draws = fit_predict_conjugate(X.values[:ti], y.values[:ti], X.values[ti:ti + 1])[0]
        model_ann = float(draws.std()) * np.sqrt(12)    # annualized model vol

        # entry trade date with option data (month-end, stepping back to a trading day)
        iv_tbl, entry_date = None, None
        for back in range(0, 7):
            d = (t - pd.Timedelta(days=back)).date()
            try:
                tbl = fetch_iv_table(pd.Timestamp(d))
            except Exception:
                tbl = None
            if tbl is not None:
                iv_tbl, entry_date = tbl, pd.Timestamp(d)
                break
        if iv_tbl is None:
            rows.append(dict(date=t, traded=0, note="no IV data"))
            continue

        spot = float(etf.asof(entry_date))
        expiry, exp_date = choose_expiry(iv_tbl, entry_date)
        if expiry is None or pd.Timestamp(exp_date) > last_daily:
            rows.append(dict(date=t, traded=0, note="no usable/!realized expiry"))
            continue
        T = max((exp_date - entry_date.date()).days, 1) / 365.0
        smile = otm_smile(iv_tbl, expiry, spot)
        if len(smile) < 5:
            rows.append(dict(date=t, traded=0, note="thin smile"))
            continue

        atm_iv = float(smile.iloc[(smile.index - spot).to_series().abs().values.argmin()])
        ratio = atm_iv / model_ann if model_ann > 0 else np.nan

        # scale model draws to the option horizon for EV/Kelly consistency
        Tm = T * 12.0
        d_h = draws.mean() * Tm + (draws - draws.mean()) * np.sqrt(Tm)
        ST = spot * np.exp(d_h)

        # regime gate -> structure
        struct, side = None, "flat"
        if ratio >= HI_RATIO:
            # choose butterfly width by best model Kelly growth (>0)
            best = None
            for n in range(1, MAX_WING_STEPS + 1):
                pr = price_butterfly(smile, spot, T, n)
                if not pr or pr["cost"] <= 1e-6:
                    continue
                pay = payoff_butterfly(ST, pr["strikes"])
                f_, g_ = kelly_opt(pay / pr["cost"] - 1.0)
                if g_ > 0 and (best is None or g_ > best[1]):
                    best = (pr, g_, f_, pay)
            if best:
                struct, side = best[0], "butterfly"
                kelly_f, model_pay = best[2], best[3]
        elif ratio <= LO_RATIO:
            pr = price_straddle(smile, spot, T)
            if pr["cost"] > 1e-6:
                pay = payoff_straddle(ST, pr["strikes"])
                f_, g_ = kelly_opt(pay / pr["cost"] - 1.0)
                if g_ > 0:
                    struct, side = pr, "straddle"
                    kelly_f, model_pay = f_, pay

        if struct is None:
            rows.append(dict(date=t, traded=0, ratio=ratio, side="flat",
                             spot=spot, atm_iv=atm_iv, model_ann=model_ann))
            continue

        # --- exit: first of {profit target, EXIT_DTE cap, expiry} to fire ---
        cost = struct["cost"]
        strikes = struct["strikes"]
        val_fn = value_butterfly if side == "butterfly" else value_straddle
        exp_ts = pd.Timestamp(exp_date)

        # DTE cap date (roll). If EXIT_DTE is None there is no cap (scan to expiry).
        cap_ts = exp_ts - pd.Timedelta(days=EXIT_DTE) if EXIT_DTE is not None else exp_ts

        # 1) scan ACTUAL daily closes for take-profit / stop-loss, marking at FIXED
        #    entry IV (cheap: no per-day SSE calls). Window = (entry, cap].
        #    First day to cross EITHER level wins.
        tgt_val = profit_target_value(side, cost, strikes)
        stop_val = cost * (1.0 - STOP_LOSS) if STOP_LOSS is not None else None
        hit_date, hit_kind = None, None
        if tgt_val is not None or stop_val is not None:
            scan_days = etf.index[(etf.index > entry_date) & (etf.index <= cap_ts)
                                  & (etf.index <= last_daily)]
            for d in scan_days:
                spot_d = float(etf.loc[d])
                T_d = max((exp_date - d.date()).days, 1) / 365.0
                mark_d = val_fn(smile, spot_d, T_d, strikes)
                if tgt_val is not None and mark_d >= tgt_val:
                    hit_date, hit_kind = d, "target"; break
                if stop_val is not None and mark_d <= stop_val:
                    hit_date, hit_kind = d, "stop"; break

        # 2) decide the exit date / kind (first to fire)
        if hit_date is not None:
            exit_kind, want_date = hit_kind, hit_date
        elif EXIT_DTE is not None and entry_date < cap_ts < exp_ts:
            exit_kind, want_date = "roll", cap_ts
        else:
            exit_kind, want_date = "expiry", exp_ts

        # 3) book the exit. target/roll -> re-fetch REAL IV at the date and mark;
        #    expiry (or any failure) -> intrinsic from ACTUAL underlying at expiry.
        realized, exit_date, S_used = None, None, None
        if exit_kind in ("target", "stop", "roll"):
            exit_tbl, ex_date = find_iv_on_or_before(want_date)
            if (exit_tbl is not None and ex_date > entry_date
                    and ex_date < exp_ts and ex_date <= last_daily):
                spot_ex = float(etf.asof(ex_date))
                smile_ex = otm_smile(exit_tbl, expiry, spot_ex)
                T_ex = max((exp_date - ex_date.date()).days, 1) / 365.0
                if len(smile_ex) >= 3:
                    realized = float(val_fn(smile_ex, spot_ex, T_ex, strikes))
                    exit_date, S_used = ex_date, spot_ex
            if realized is None:                      # couldn't book -> fall back
                exit_kind = "expiry"

        if exit_kind == "expiry":
            S_used = float(etf.asof(exp_ts))
            pay_fn = payoff_butterfly if side == "butterfly" else payoff_straddle
            realized = float(pay_fn(np.array([S_used]), strikes)[0])
            exit_date = exp_date

        r_trade = realized / cost - 1.0
        f_used = min(KELLY_FRAC * kelly_f, KELLY_CAP)
        rows.append(dict(date=t, traded=1, side=side, ratio=ratio, spot=spot,
                         atm_iv=atm_iv, model_ann=model_ann, expiry=expiry,
                         exp_date=exp_date, exit_kind=exit_kind, exit_date=exit_date,
                         S_exit=S_used, exit_value=realized, cost=cost,
                         strikes=struct["strikes"], r_trade=r_trade,
                         kelly_f=kelly_f, f_used=f_used,
                         contrib=f_used * r_trade))
        print(f"  {t.date()} {side:9} ratio={ratio:4.2f} cost={cost:.4f} "
              f"r={r_trade:+.2f} f={f_used:.2f} [{exit_kind}] -> {f_used*r_trade:+.3f}")
        time.sleep(0.2)        # be gentle on the SSE endpoint over ~100 calls

    bt = pd.DataFrame(rows).set_index("date")
    return bt, X, y, df


# =============================================================================
# Eval
# =============================================================================
def equity_and_metrics(bt, df, y):
    bt = bt.copy()
    contrib = bt["contrib"] if "contrib" in bt.columns else pd.Series(0.0, index=bt.index)
    bt["contrib"] = contrib.fillna(0.0)
    bt["growth"] = np.log1p(bt["contrib"].clip(lower=-0.999))
    bt["equity"] = np.exp(bt["growth"].cumsum())

    # buy & hold 50ETF over the same months (model target y = next-month log ret)
    bh = y.reindex(bt.index).fillna(0.0)
    bh_equity = np.exp(bh.cumsum())

    traded = bt[bt["traded"] == 1] if "traded" in bt else bt
    rt = traded["r_trade"].dropna() if "r_trade" in traded else pd.Series(dtype=float)
    n_tot = len(bt); n_tr = int((bt.get("traded", 0) == 1).sum())
    yrs = max((bt.index[-1] - bt.index[0]).days / 365.25, 1e-9)

    def cagr(eq): return eq.iloc[-1] ** (1 / yrs) - 1 if len(eq) else np.nan
    def mdd(eq):
        roll = eq.cummax(); return float((eq / roll - 1).min()) if len(eq) else np.nan

    m = dict(
        months=n_tot, trades=n_tr, pct_traded=n_tr / n_tot if n_tot else np.nan,
        rule_total=float(bt["equity"].iloc[-1]) if len(bt) else np.nan,
        rule_cagr=cagr(bt["equity"]), rule_mdd=mdd(bt["equity"]),
        bh_total=float(bh_equity.iloc[-1]) if len(bh_equity) else np.nan,
        bh_cagr=cagr(bh_equity), bh_mdd=mdd(bh_equity),
        trade_winrate=float((rt > 0).mean()) if len(rt) else np.nan,
        avg_trade_ret=float(rt.mean()) if len(rt) else np.nan,
    )
    return bt, bh_equity, m


def main():
    print("=" * 72)
    print("Vol-timing backtest: butterfly (rich) + long straddle (cheap), 50ETF")
    print(f"HI_RATIO={HI_RATIO} LO_RATIO={LO_RATIO} fracKelly={KELLY_FRAC}")
    print(f"EXIT_DTE={EXIT_DTE if EXIT_DTE is not None else 'none'}  "
          f"PROFIT_TAKE={PROFIT_TAKE if PROFIT_TAKE is not None else 'none'}  "
          f"STOP_LOSS={STOP_LOSS if STOP_LOSS is not None else 'none'} "
          f"(first to fire; else hold-to-expiry)")
    print("=" * 72)
    bt, X, y, df = run()
    bt, bh_equity, m = equity_and_metrics(bt, df, y)

    bt.to_csv(OUTDIR / "timing_backtest_trades.csv")
    print("\n--- metrics ---")
    for k, v in m.items():
        print(f"  {k:14s}: {v:.4f}" if isinstance(v, float) else f"  {k:14s}: {v}")

    traded_bt = bt[bt.get("traded", 0) == 1]
    by_side = traded_bt.groupby("side")["r_trade"].agg(["count", "mean"])
    print("\n--- by structure ---")
    print(by_side.to_string(float_format=lambda v: f"{v:.4f}"))

    if "exit_kind" in traded_bt.columns:
        by_exit = traded_bt.groupby("exit_kind")["r_trade"].agg(["count", "mean"])
        print("\n--- by exit type (roll vs held-to-expiry) ---")
        print(by_exit.to_string(float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt.index, bt["equity"], label="timing rule", color="#1f4e79", lw=1.6)
    ax.plot(bh_equity.index, bh_equity, label="buy & hold 50ETF", color="#c0392b", lw=1.2)
    ax.axhline(1.0, color="grey", lw=0.7)
    ax.set_ylabel("growth of 1"); ax.legend()
    ax.set_title("Vol-timing rule vs buy & hold (fractional-Kelly, real IV-priced)")
    fig.tight_layout(); fig.savefig(OUTDIR / "timing_backtest_equity.png", dpi=130); plt.close(fig)

    print(f"\nSaved: {OUTDIR/'timing_backtest_trades.csv'} and timing_backtest_equity.png")
    print("Caveats: entry priced from SSE IV via BS (no bid/ask, no fees/slippage);")
    print("realized at expiry from actual 50ETF close; model horizon ~ option DTE")
    print("(draws scaled to DTE). Not investment advice.")


if __name__ == "__main__":
    main()
