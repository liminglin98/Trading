# -*- coding: utf-8 -*-
"""
Butterfly WING-WIDTH optimizer - 50ETF
======================================
Turns the model's posterior-predictive band into a wing-width decision.

Logic chain:
  1. The factor model (butterfly_factor_forecast.py) gives a calibrated predictive
     distribution of next-month 50ETF return. Direction is ~uninformative, so we
     center the butterfly at spot (symmetric, market-neutral short-dispersion bet).
  2. For a GRID of wing half-widths w, price the long call butterfly
     [long K-w, short 2K, long K+w] and integrate its expiry payoff over the
     model's predictive draws -> E[return], P(profit), max loss, and E/sigma.
  3. The EDGE only exists if IMPLIED dispersion (option prices / QVIX) is WIDER
     than the MODEL band: a long fly is cheap when implied vol is high, and pays
     off if realized pins near the body. We report implied-vs-model dispersion so
     the vol-risk-premium that funds the trade is explicit, not assumed.
  4. Pick w* = argmax E[PnL]/sigma(PnL) (user choice), and also print the full
     frontier so you can choose per trade.

Pricing source (user choice = "both"):
  - Default: Black-Scholes priced off QVIX (no chain needed; ignores skew).
  - If CHAIN_FILE is set to a CSV with columns [strike, type(C/P), mid], real
    call debits are used instead (snap/interp to the 3 strikes).

NOT investment advice. A long butterfly is short gamma; it loses if the index
trends or jumps. The edge is regime-dependent and evaporates if implied ~ model.

Run:  python butterfly_wing_optimizer.py
Needs: pandas, numpy, scipy, matplotlib  (+ butterfly_factor_forecast.py alongside)
"""

from __future__ import annotations

import re
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

try:
    import akshare as ak
except Exception:                       # only needed for the real-chain path
    ak = None

warnings.filterwarnings("ignore")

# reuse the model: importing does NOT run its main() (guarded by __main__)
from butterfly_factor_forecast import (
    load_panel, build_xy, fit_predict_conjugate, SPOT_COL,
)

# =============================================================================
# CONFIG
# =============================================================================
OUTDIR     = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

T_YEARS    = 21 / 252.0      # ~1 trading month to expiry
R_FREE     = 0.02            # short rate (minor for a fly)
QVIX_COL   = "qvix_50"       # implied-vol index (annualized %, e.g. 18.0)
WIDTH_GRID = np.round(np.arange(0.010, 0.1201, 0.005), 4)   # wing half-width as % of spot
OBJECTIVE  = "kelly"   # "kelly" (interior, recommended) | "e_over_sigma" | "e_return"
PPROFIT_FLOOR = 0.50   # for the constrained reference pick

# --- pricing source ---------------------------------------------------------
# "real"  -> live 50ETF option chain via akshare option_finance_board (skew priced,
#            discrete listed strikes). Falls back to QVIX-BS if the fetch fails.
# "bs"    -> QVIX-implied Black-Scholes on a continuous width grid.
PRICE_SOURCE   = "real"
ETF_OPT_SYMBOL = "华夏上证50ETF期权"     # 50ETF (510050) options board
END_MONTH      = None                    # 'YYMM' e.g. '2607'; None -> auto next month
MAX_WING_STEPS = 12                      # widest fly = body +/- this many listed strikes

# Optional real option chain. CSV columns: strike, type ('C'/'P'), mid (price).
# Leave as None to price off QVIX-BS.
CHAIN_FILE = None   # e.g. Path(__file__).resolve().parents[1] / "Data" / "option_chain_50etf.csv"


# =============================================================================
# Black-Scholes call + butterfly pricing
# =============================================================================
def bs_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_implied_vol(price, S, K, T=T_YEARS, r=R_FREE):
    """Back out call implied vol by bisection (NaN if price below intrinsic)."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-9:
        return np.nan
    lo, hi = 1e-4, 3.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fly_cost_bs(spot, w, sigma, T=T_YEARS, r=R_FREE):
    """Net debit of a long call butterfly with wings +/- w*spot around the body."""
    klow, kmid, khigh = spot * (1 - w), spot, spot * (1 + w)
    return (bs_call(spot, klow, T, r, sigma)
            - 2 * bs_call(spot, kmid, T, r, sigma)
            + bs_call(spot, khigh, T, r, sigma))


def _chain_call_price(chain, S, K, T, r):
    """Interpolate a call mid price at strike K from a real chain (falls back to
    put-call parity via puts if calls are sparse)."""
    calls = chain[chain["type"].str.upper().str[0] == "C"].sort_values("strike")
    if len(calls) >= 2:
        return float(np.interp(K, calls["strike"], calls["mid"]))
    puts = chain[chain["type"].str.upper().str[0] == "P"].sort_values("strike")
    p = float(np.interp(K, puts["strike"], puts["mid"]))
    return p + S - K * np.exp(-r * T)          # parity: C = P + S - K e^{-rT}


def fly_cost_chain(chain, spot, w, T=T_YEARS, r=R_FREE):
    klow, kmid, khigh = spot * (1 - w), spot, spot * (1 + w)
    return (_chain_call_price(chain, spot, klow, T, r)
            - 2 * _chain_call_price(chain, spot, kmid, T, r)
            + _chain_call_price(chain, spot, khigh, T, r))


def fly_payoff(ST, spot, w):
    """Expiry payoff of the symmetric long call butterfly (peaks at body = w*spot)."""
    klow, kmid, khigh = spot * (1 - w), spot, spot * (1 + w)
    return fly_payoff_k(ST, klow, kmid, khigh)


def fly_payoff_k(ST, klow, kmid, khigh):
    """Long call butterfly payoff for explicit (possibly asymmetric) strikes."""
    return (np.maximum(ST - klow, 0)
            - 2 * np.maximum(ST - kmid, 0)
            + np.maximum(ST - khigh, 0))


# =============================================================================
# Real 50ETF option chain (akshare option_finance_board)
# =============================================================================
def _auto_end_month():
    """YYMM of next calendar month (the ~1-month-ahead expiry matching the model)."""
    today = dt.date.today()
    nxt = (today.replace(day=28) + dt.timedelta(days=10)).replace(day=1)
    return nxt.strftime("%y%m")


def fetch_50etf_chain(end_month=None):
    """Return (calls, puts, em): LIVE 50ETF call & put price tables [strike, mid]
    for one expiry. Uses last price (当前价), falling back to prior settle (前结价)."""
    if ak is None:
        raise RuntimeError("akshare not importable")
    em = end_month or END_MONTH or _auto_end_month()
    board = ak.option_finance_board(symbol=ETF_OPT_SYMBOL, end_month=em)
    code_col = "合约交易代码" if "合约交易代码" in board.columns else board.columns[1]
    k_col = "行权价"
    last_col = "当前价" if "当前价" in board.columns else None
    prev_col = "前结价" if "前结价" in board.columns else None

    def _type(code):
        m = re.search(r"([CP])\d{3,4}", str(code))
        return m.group(1) if m else None

    board = board.copy()
    board["type"] = board[code_col].map(_type)

    def _side(letter):
        sub = board[board["type"] == letter].copy()
        mid = pd.to_numeric(sub[last_col], errors="coerce") if last_col else pd.Series(np.nan, index=sub.index)
        if prev_col is not None:
            prev = pd.to_numeric(sub[prev_col], errors="coerce")
            mid = mid.where(mid.fillna(0) > 0, prev)
        out = pd.DataFrame({"strike": pd.to_numeric(sub[k_col], errors="coerce"), "mid": mid}).dropna()
        return out[out["mid"] > 0].sort_values("strike").drop_duplicates("strike").reset_index(drop=True)

    return _side("C"), _side("P"), em


# =============================================================================
# Model live predictive draws
# =============================================================================
def live_draws():
    """Train on all complete-case history; draw the predictive log-return
    distribution for the most recent month-end feature row (target unknown)."""
    df = load_panel()
    X, y, used = build_xy(df)
    full = df[used].dropna()
    asof = full.index[-1]
    Xte = full.iloc[[-1]].values
    draws = fit_predict_conjugate(X.values, y.values, Xte)[0]   # log returns
    spot = float(df[SPOT_COL].reindex([asof]).iloc[0])
    qvix = float(df[QVIX_COL].reindex([asof]).iloc[0]) if QVIX_COL in df else np.nan
    return draws, spot, qvix, asof, used


# =============================================================================
# Frontier + selection
# =============================================================================
def kelly_opt(net_ret, fgrid=np.linspace(0.0, 0.99, 100)):
    """Optimal Kelly fraction f* and expected log-growth g* for a per-trade net
    return distribution. log(1 + f*r): r=-1 (total debit loss) forces f<1, so the
    narrow-lottery flys self-limit; wide flys with no total-loss tail allow large f.
    Returns (f_star, g_star)."""
    best_f, best_g = 0.0, 0.0
    for f in fgrid:
        val = 1.0 + f * net_ret
        if np.any(val <= 0):
            continue
        g = np.mean(np.log(val))
        if g > best_g:
            best_g, best_f = g, f
    return best_f, best_g


def build_frontier(draws, spot, sigma_impl, chain=None):
    ST = spot * np.exp(draws)
    rows = []
    for w in WIDTH_GRID:
        cost = fly_cost_chain(chain, spot, w) if chain is not None else fly_cost_bs(spot, w, sigma_impl)
        if cost <= 1e-6:
            continue
        payoff = fly_payoff(ST, spot, w)
        pnl = payoff - cost
        e_pnl, s_pnl = pnl.mean(), pnl.std()
        net_ret = payoff / cost - 1.0
        f_star, g_star = kelly_opt(net_ret)
        rows.append(dict(
            width_pct=w * 100,
            klow=spot * (1 - w), kmid=spot, khigh=spot * (1 + w),
            cost=cost,
            e_return=e_pnl / cost,
            p_profit=float((payoff > cost).mean()),
            max_loss=cost,
            peak_return=(w * spot - cost) / cost,
            e_over_sigma=e_pnl / s_pnl if s_pnl > 0 else np.nan,
            kelly_f=f_star,
            kelly_growth=g_star,
            model_ev=e_pnl,                 # model expected $ PnL (the edge, in price)
        ))
    return pd.DataFrame(rows)


def build_frontier_chain(draws, spot, calls):
    """Discrete-strike frontier from a real call chain. Body = listed strike
    nearest spot; wings = body +/- n listed strikes (n = 1..MAX_WING_STEPS).
    Prices are the actual call mids, so skew is reflected; wings may be slightly
    asymmetric in price (broken-wing) when strike spacing is uneven."""
    ST = spot * np.exp(draws)
    ks = calls["strike"].values
    px = dict(zip(calls["strike"].values, calls["mid"].values))
    i0 = int(np.argmin(np.abs(ks - spot)))               # ATM body index
    kbody = ks[i0]
    rows = []
    for n in range(1, MAX_WING_STEPS + 1):
        lo, hi = i0 - n, i0 + n
        if lo < 0 or hi >= len(ks):
            break
        klow, khigh = ks[lo], ks[hi]
        cost = px[klow] - 2 * px[kbody] + px[khigh]
        if cost <= 1e-6:                                 # arbitrage/illiquid quote; skip
            continue
        payoff = fly_payoff_k(ST, klow, kbody, khigh)
        pnl = payoff - cost
        e_pnl, s_pnl = pnl.mean(), pnl.std()
        net_ret = payoff / cost - 1.0
        f_star, g_star = kelly_opt(net_ret)
        peak = min(kbody - klow, khigh - kbody)          # max payoff (min wing for broken-wing)
        rows.append(dict(
            width_pct=(khigh - klow) / 2 / spot * 100,
            n_steps=n, klow=klow, kmid=kbody, khigh=khigh,
            cost=cost, e_return=e_pnl / cost,
            p_profit=float((payoff > cost).mean()),
            max_loss=cost, peak_return=(peak - cost) / cost,
            e_over_sigma=e_pnl / s_pnl if s_pnl > 0 else np.nan,
            kelly_f=f_star, kelly_growth=g_star, model_ev=e_pnl,
        ))
    return pd.DataFrame(rows)


def vol_structures(draws, spot, calls, puts, n_strangle=2):
    """Model-EV scan of the ATM vol structures, so 'which structure, if any' is
    answered in one table. Long and short are exact P&L negatives (ex-fees), so a
    negative long-EV == a positive short-EV; the sign tells you which side.

    Returns a DataFrame; pnl/EV are per 1 spread in price units (same as spot).
    Short straddle/strangle have UNCAPPED loss -> Kelly not reported for them."""
    ST = spot * np.exp(draws)
    ck = calls["strike"].values; cpx = dict(zip(calls["strike"], calls["mid"]))
    pk = puts["strike"].values;  ppx = dict(zip(puts["strike"], puts["mid"]))
    rows = []

    def _emit(name, cost, payoff, capped):
        pnl = payoff - cost
        e = pnl.mean()
        pprofit = float((pnl > 0).mean())
        kelly = ""
        if capped and cost > 0:
            f_star, g_star = kelly_opt(payoff / cost - 1.0)
            kelly = f"{f_star*100:.0f}%/{g_star:.4f}"
        rows.append(dict(structure=name, net=cost, e_pnl=e,
                         e_return=(e / cost if (capped and cost > 0) else np.nan),
                         p_profit=pprofit, kelly_f_growth=kelly,
                         risk=("capped" if capped else "UNCAPPED")))

    # ATM straddle (nearest common strike to spot present on both sides)
    common = sorted(set(ck) & set(pk))
    if common:
        katm = min(common, key=lambda k: abs(k - spot))
        c0, p0 = cpx[katm], ppx[katm]
        straddle_payoff = np.abs(ST - katm)
        _emit(f"Long straddle  K={katm:.3f}", c0 + p0, straddle_payoff, capped=True)
        # short = negate: cost is the credit received; payoff to us is credit - |move|
        short_pnl_payoff = (c0 + p0) - straddle_payoff   # already net P&L; emit via cost=0 trick
        rows.append(dict(structure=f"Short straddle K={katm:.3f}", net=-(c0 + p0),
                         e_pnl=short_pnl_payoff.mean(), e_return=np.nan,
                         p_profit=float((short_pnl_payoff > 0).mean()),
                         kelly_f_growth="", risk="UNCAPPED"))

    # OTM strangle: put n steps below ATM, call n steps above
    if len(common) > 2 * n_strangle:
        i = common.index(katm)
        if i - n_strangle >= 0 and i + n_strangle < len(common):
            kp, kc = common[i - n_strangle], common[i + n_strangle]
            strangle_payoff = np.maximum(kp - ST, 0) + np.maximum(ST - kc, 0)
            cost = ppx[kp] + cpx[kc]
            _emit(f"Long strangle  {kp:.3f}/{kc:.3f}", cost, strangle_payoff, capped=True)
            sp = cost - strangle_payoff
            rows.append(dict(structure=f"Short strangle {kp:.3f}/{kc:.3f}", net=-cost,
                             e_pnl=sp.mean(), e_return=np.nan,
                             p_profit=float((sp > 0).mean()), kelly_f_growth="",
                             risk="UNCAPPED"))
    return pd.DataFrame(rows)


def pick(frontier):
    f = frontier.copy()
    pos = f[f["model_ev"] > 0]
    pool = pos if len(pos) else f
    key = {"e_return": "e_return", "e_over_sigma": "e_over_sigma", "kelly": "kelly_growth"}[OBJECTIVE]
    best = pool.loc[pool[key].idxmax()]
    # constrained reference: best risk-adjusted subject to a P(profit) floor
    con = pool[pool["p_profit"] >= PPROFIT_FLOOR]
    best_con = con.loc[con["kelly_growth"].idxmax()] if len(con) else None
    return best, best_con, len(pos) > 0


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 72)
    print("Butterfly wing-width optimizer - 50ETF (model band vs option-implied)")
    print("=" * 72)

    draws, spot, qvix, asof, used = live_draws()
    sig_model_m = float(draws.std())                  # model 1-month sd (log)
    qvix_sigma = (qvix / 100.0) if np.isfinite(qvix) else 0.20

    # ---- choose pricing source: real chain (preferred) or QVIX-BS ----
    calls, puts, src, em = None, None, "QVIX-BS", None
    if PRICE_SOURCE == "real":
        try:
            calls, puts, em = fetch_50etf_chain()
            if len(calls) >= 5:
                src = f"real 50ETF chain (exp {em}, {len(calls)}C/{len(puts)}P strikes)"
            else:
                print(f"  [warn] only {len(calls)} call strikes for {em}; falling back to QVIX-BS")
                calls = None
        except Exception as e:  # noqa
            print(f"  [warn] chain fetch failed ({type(e).__name__}: {e}); falling back to QVIX-BS")
            calls = None
    elif CHAIN_FILE and Path(CHAIN_FILE).exists():
        df_csv = pd.read_csv(CHAIN_FILE)
        calls = df_csv[df_csv["type"].str.upper().str[0] == "C"][["strike", "mid"]] \
            .sort_values("strike").reset_index(drop=True)
        puts = df_csv[df_csv["type"].str.upper().str[0] == "P"][["strike", "mid"]] \
            .sort_values("strike").reset_index(drop=True)
        src = f"chain file ({Path(CHAIN_FILE).name})"

    # implied vol used for the dispersion diagnostic: back out the ATM IV from the
    # REAL chain when available (consistent with how the frontier is priced);
    # otherwise fall back to QVIX. QVIX is a headline number and tends to overstate
    # the harvestable premium vs the actual priced surface.
    atm_iv = np.nan
    if calls is not None:
        katm = calls["strike"].iloc[(calls["strike"] - spot).abs().idxmin()]
        atm_iv = bs_implied_vol(float(calls.loc[calls["strike"] == katm, "mid"].iloc[0]), spot, katm)
    sigma_impl = atm_iv if np.isfinite(atm_iv) else qvix_sigma
    sig_impl_m = sigma_impl * np.sqrt(T_YEARS)

    print(f"as of {pd.Timestamp(asof).date()} | spot={spot:.4f} | QVIX={qvix:.1f}% "
          f"| pricing={src}")
    print(f"pred mean (log) = {draws.mean():+.4f}  P(up)={ (draws>0).mean():.2f}")
    print("\n--- dispersion: where the edge comes from ---")
    iv_src = f"chain ATM IV {sigma_impl*100:.1f}%" if np.isfinite(atm_iv) else f"QVIX {qvix:.1f}%"
    print(f"  implied 1m sd ({iv_src}): {sig_impl_m*100:5.2f}%")
    if np.isfinite(atm_iv):
        print(f"  (QVIX headline for reference: {qvix:.1f}% ann -> {qvix_sigma*np.sqrt(T_YEARS)*100:.2f}% 1m)")
    print(f"  model   1m sd (band)     : {sig_model_m*100:5.2f}%   (ann {sig_model_m/np.sqrt(T_YEARS)*100:.1f}%)")
    ratio = sig_impl_m / sig_model_m if sig_model_m > 0 else np.nan
    # 1.0-1.10x is within the band's own estimation noise -> not a real edge.
    if ratio > 1.20:
        lbl = "implied materially wider -> long fly may be favored (confirm on frontier)"
    elif ratio > 1.05:
        lbl = "marginal: implied ~ model within noise -> likely no harvestable premium"
    else:
        lbl = "none: implied <= model -> do not buy the fly"
    print(f"  implied / model          : {ratio:5.2f}x  ({lbl})")

    if calls is not None:
        frontier = build_frontier_chain(draws, spot, calls)
    else:
        frontier = build_frontier(draws, spot, sigma_impl)
    frontier.to_csv(OUTDIR / "butterfly_wing_frontier.csv", index=False)

    print("\n--- wing-width frontier (long call butterfly, body at spot) ---")
    show = frontier[["width_pct", "cost", "e_return", "p_profit", "peak_return",
                     "e_over_sigma", "kelly_f", "kelly_growth", "model_ev"]].copy()
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best, best_con, has_edge = pick(frontier)
    print(f"\n--- selected width (objective = {OBJECTIVE}) ---")
    print(f"  wing half-width : {best['width_pct']:.2f}% of spot")
    print(f"  strikes         : {best['klow']:.3f} / {best['kmid']:.3f} / {best['khigh']:.3f}")
    print(f"  net debit       : {best['cost']:.4f}  (= max loss per spread)")
    print(f"  E[return]       : {best['e_return']*100:+.1f}%   peak if pinned: {best['peak_return']*100:+.0f}%")
    print(f"  P(profit)       : {best['p_profit']*100:.0f}%   E/sigma: {best['e_over_sigma']:.3f}")
    print(f"  Kelly fraction  : {best['kelly_f']*100:.0f}% of bankroll   log-growth: {best['kelly_growth']:.4f}")
    print("  (Kelly is full-Kelly; trade 1/4-1/2 of it in practice.)")
    if best_con is not None:
        print(f"\n--- constrained reference (max Kelly s.t. P(profit) >= {PPROFIT_FLOOR:.0%}) ---")
        print(f"  width {best_con['width_pct']:.2f}% | strikes {best_con['klow']:.3f}/"
              f"{best_con['kmid']:.3f}/{best_con['khigh']:.3f} | debit {best_con['cost']:.4f} | "
              f"E[ret] {best_con['e_return']*100:+.1f}% | P(profit) {best_con['p_profit']*100:.0f}% | "
              f"Kelly {best_con['kelly_f']*100:.0f}%")
    # ---- VERDICT (driven by the frontier, not the implied/model ratio) ----
    # A whisker-positive EV (e.g. +0.0003) at Kelly~1% is noise, not a trade.
    KELLY_MIN, GROWTH_MIN = 0.05, 1e-3
    material = (best["kelly_f"] >= KELLY_MIN) and (best["kelly_growth"] >= GROWTH_MIN)
    print("\n--- VERDICT ---")
    if material:
        print(f"  TRADE-ABLE: real-priced edge clears the bar (Kelly {best['kelly_f']*100:.0f}%, "
              f"growth {best['kelly_growth']:.4f}). Use the width above, sized at 1/4-1/2 Kelly.")
    else:
        print(f"  STAND ASIDE: no material edge once priced on the real chain "
              f"(best Kelly {best['kelly_f']*100:.0f}%, growth {best['kelly_growth']:.4f}, "
              f"E[ret] {best['e_return']*100:+.1f}%). The implied/model gap is within noise.\n"
              f"  Revisit when chain ATM IV is materially above the model band (ratio > ~1.20).")

    # ---- four-corner vol structure scan (straddle / strangle, long & short) ----
    if calls is not None and puts is not None and len(puts) >= 3:
        vs = vol_structures(draws, spot, calls, puts)
        if len(vs):
            vs.to_csv(OUTDIR / "vol_structures.csv", index=False)
            print("\n--- vol structures: model-EV scan (per 1 spread, price units) ---")
            show2 = vs[["structure", "net", "e_pnl", "e_return", "p_profit",
                        "kelly_f_growth", "risk"]].copy()
            print(show2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            print("  net: debit(+)/credit(-).  e_pnl>0 favors that row; long & short are\n"
                  "  mirror signs, so pick the positive-EV side. SHORT = uncapped tail risk.\n"
                  "  At implied~model, all e_pnl are near zero -> no clear vol trade.")

    # plot
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(frontier["width_pct"], frontier["e_return"] * 100, "-o", color="#1f4e79",
             ms=3, label="E[return] %")
    ax1.axhline(0, color="grey", lw=0.7)
    ax1.set_xlabel("wing half-width (% of spot)")
    ax1.set_ylabel("E[return] (%)", color="#1f4e79")
    ax2 = ax1.twinx()
    ax2.plot(frontier["width_pct"], frontier["p_profit"] * 100, "-s", color="#c0392b",
             ms=3, label="P(profit) %")
    ax2.set_ylabel("P(profit) (%)", color="#c0392b")
    ax1.axvline(best["width_pct"], color="green", ls="--", lw=1, label=f"w* ({OBJECTIVE})")
    ax1.set_title(f"50ETF butterfly: width vs return/probability  (as of {pd.Timestamp(asof).date()})")
    fig.tight_layout(); fig.savefig(OUTDIR / "butterfly_wing_frontier.png", dpi=130); plt.close(fig)

    print(f"\nSaved: {OUTDIR/'butterfly_wing_frontier.csv'} and butterfly_wing_frontier.png")
    print("Reminder: long fly = short gamma; loses on trends/jumps. Not investment advice.")


if __name__ == "__main__":
    main()
