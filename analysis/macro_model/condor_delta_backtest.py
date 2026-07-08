# -*- coding: utf-8 -*-
"""
condor_delta_backtest.py
=========================
Self-contained iron condor backtest using ONLY the local Sina-sourced option
chain (Data/options/option_daily_510050.csv) -- no macro factor panel, no
Bayesian posterior draws, no akshare, no network at all.

Strike placement is by TARGET DELTA -- the option market's own implied
probability of finishing ITM, already computed and stored per contract in
the chain (Sina's Greeks feed, same one iron_condor_backtest.py's
local_iv_smile() reads). This is the standard, model-free way professional
options traders size an iron condor ("a 20-delta condor"), and a direct,
data-only substitute for the macro model's band_q-quantile-of-posterior-
draws approach used by condor10 (see condor10_recalibrate.py).

Why this exists: every macro-model-based backtest so far (iron_condor_
backtest.py's RATIO_GATE, condor10_recalibrate.py's band_q sweep) is bounded
to ~50 OOS months (2022-2026) by the factor panel's margin_mom column (first
available 2018-04) + MIN_TRAIN=48 -- see the "why does the backtest only go
back to 2022" thread. The option chain itself goes back to 2015. Dropping
the macro model dependency entirely unlocks ~11 years / ~130 monthly entries
instead of ~50, and removes a fragile dependency this project's own README
already calls "SETTLED: no tradeable vol-timing / VRP alpha" for the timing
half of that machinery.

Pricing: uses REAL quoted closing prices (收盘价) for each leg, not a
Black-Scholes reconstruction from IV -- one fewer assumption than every
other backtest in this project.

Method, walk-forward, ALWAYS enters (no timing gate -- matches condor10's
"enter every month" sleeve design, since the timing half was separately
settled as having no edge):
  1. One entry per calendar month (that month's last cached trading day).
  2. Pick the listed expiry closest to ~30 DTE (skip <=5 DTE).
  3. Short put/call at the strike nearest a TARGET |delta|; protective wings
     `p` listed strikes further OTM.
  4. Credit = real quoted close prices of the 4 legs, minus per-leg slippage.
  5. Realize P&L at the option's actual expiry, from the real ETF close.
Sweeps target delta x protective-wing-steps p; reports CAGR/MDD/win/Calmar
per cell (sized at each cell's own fractional-Kelly, damped the same way as
the rest of the project: KELLY_FRAC=0.33, KELLY_CAP=0.50) vs. buy & hold
over the identical full window.

RUN IN A NORMAL PYTHON ENV. No network calls.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import iron_condor_backtest as IC       # noqa: E402  local chain/ETF loaders
import butterfly_timing_backtest as B   # noqa: E402  kelly_opt, KELLY_FRAC/CAP

OUTDIR = IC.OUTDIR
SLIP_PER_LEG = IC.SLIP_PER_LEG
TARGET_DTE = IC.TARGET_DTE

DELTA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
P_GRID = [1, 2, 3, 4]

C_DATE, C_EXP, C_DTE = "交易日期", "到期日", "距到期日剩余天数"
C_STRIKE, C_CP, C_DELTA, C_CLOSE = "行权价(元)", "认购(C)认沽(P)类型", "Delta", "收盘价(元)"

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_MUTED = "#898781"


# ------------------------------------------------------------------ raw (unadjusted) spot
def chain_spot_series(chain: pd.DataFrame) -> pd.Series:
    """One RAW (unadjusted) underlying close per day, straight from the
    chain's own embedded 期权标的收盘价 (Sina's real-time quote at capture
    time). Option strikes are always nominal/unadjusted, so settlement MUST
    be priced off this, not off etf_daily_*.csv -- that file is Tencent's
    QFQ (dividend-forward-adjusted) series, correct for buy&hold returns but
    off by double digits % vs. raw price the further back in time you go
    (25-30% in 2015-16, ~0% today.) Mixing the two silently wrecks every
    payoff calc on older trades. etf_daily_*.csv / load_local_etf() is still
    the right, standard choice for the buy & hold comparison below."""
    return chain.groupby(C_DATE)["期权标的收盘价"].first().sort_index()


# ------------------------------------------------------------------ entries
def monthly_entry_dates(chain: pd.DataFrame) -> list[pd.Timestamp]:
    """One entry per calendar month: that month's last cached trading day."""
    dates = chain[C_DATE].drop_duplicates().sort_values()
    df = pd.DataFrame({"date": dates})
    df["month"] = df["date"].dt.to_period("M")
    return df.groupby("month")["date"].max().tolist()


def choose_expiry_for(day: pd.DataFrame, target_dte: int = TARGET_DTE):
    exp_info = day[[C_EXP, C_DTE]].drop_duplicates()
    exp_info = exp_info[exp_info[C_DTE] > 5]
    if exp_info.empty:
        return None
    exp_info = exp_info.assign(score=(exp_info[C_DTE] - target_dte).abs())
    return exp_info.sort_values("score").iloc[0][C_EXP]


# ------------------------------------------------------------------ strike selection
def pick_by_delta(sub: pd.DataFrame, cp: str, target: float):
    """Row of type `cp` whose Delta is closest to the target magnitude
    (sign-aware: calls +target, puts -target -- see the sign check we ran:
    Sina reports calls in [0,1], puts in [-1,0])."""
    rows = sub[(sub[C_CP] == cp) & sub[C_DELTA].notna()]
    if rows.empty:
        return None
    want = target if cp == "C" else -target
    i = (rows[C_DELTA] - want).abs().idxmin()
    return rows.loc[i]


def build_condor_by_delta(sub: pd.DataFrame, target_delta: float, p_steps: int):
    """Short put/call at target |delta|; protective wings p listed strikes
    further OTM, priced off REAL quoted closes. Returns dict(strikes, credit,
    max_loss) or None if it can't be built (grid edge, missing quotes, etc)."""
    strikes_sorted = np.sort(sub[C_STRIKE].unique())
    if len(strikes_sorted) < 5:
        return None
    short_p, short_c = pick_by_delta(sub, "P", target_delta), pick_by_delta(sub, "C", target_delta)
    if short_p is None or short_c is None:
        return None
    Kp_s, Kc_s = float(short_p[C_STRIKE]), float(short_c[C_STRIKE])
    if Kp_s >= Kc_s:                      # degenerate: shorts crossed
        return None
    ip_s = int(np.searchsorted(strikes_sorted, Kp_s))
    ic_s = int(np.searchsorted(strikes_sorted, Kc_s))
    ip_l, ic_l = ip_s - p_steps, ic_s + p_steps
    if ip_l < 0 or ic_l >= len(strikes_sorted):
        return None
    Kp_l, Kc_l = float(strikes_sorted[ip_l]), float(strikes_sorted[ic_l])

    def close_of(K, cp):
        r = sub[(sub[C_STRIKE] == K) & (sub[C_CP] == cp)]
        return float(r[C_CLOSE].iloc[0]) if len(r) and pd.notna(r[C_CLOSE].iloc[0]) else np.nan

    p_s_px, p_l_px = close_of(Kp_s, "P"), close_of(Kp_l, "P")
    c_s_px, c_l_px = close_of(Kc_s, "C"), close_of(Kc_l, "C")
    if any(np.isnan(x) for x in (p_s_px, p_l_px, c_s_px, c_l_px)):
        return None
    credit = (p_s_px - p_l_px) + (c_s_px - c_l_px) - 4 * SLIP_PER_LEG
    wing = max(Kp_s - Kp_l, Kc_l - Kc_s)
    max_loss = wing - credit
    if credit <= 0 or max_loss <= 1e-6:
        return None
    return dict(strikes=(Kp_l, Kp_s, Kc_s, Kc_l), credit=credit, max_loss=max_loss)


def realized_pnl(strikes, credit, S_exp):
    Kp_l, Kp_s, Kc_s, Kc_l = strikes
    payout = (max(Kp_s - S_exp, 0) - max(Kp_l - S_exp, 0)
              + max(S_exp - Kc_s, 0) - max(S_exp - Kc_l, 0))
    return credit - payout


# ------------------------------------------------------------------ sweep
def run_grid(chain: pd.DataFrame, spot_series: pd.Series):
    entry_dates = monthly_entry_dates(chain)
    trades: dict[tuple, list] = {}
    skipped_no_expiry = skipped_unrealized = 0
    for d in entry_dates:
        day = chain[chain[C_DATE] == d]
        if day.empty:
            continue
        exp_date = choose_expiry_for(day)
        if exp_date is None:
            skipped_no_expiry += 1
            continue
        exp_date = pd.Timestamp(exp_date)
        if exp_date not in spot_series.index:
            skipped_unrealized += 1     # not yet expired, or a rare data gap
            continue
        sub = day[day[C_EXP] == exp_date]
        S_exp = float(spot_series.loc[exp_date])
        for target_delta in DELTA_GRID:
            for p in P_GRID:
                built = build_condor_by_delta(sub, target_delta, p)
                if built is None:
                    continue
                pnl = realized_pnl(built["strikes"], built["credit"], S_exp)
                r = pnl / built["max_loss"]
                trades.setdefault((target_delta, p), []).append((d, r))
    print(f"entry months: {len(entry_dates)}  "
          f"(skipped: no valid expiry={skipped_no_expiry}, "
          f"not yet realized={skipped_unrealized})")
    return trades


def cell_stats(rows: list) -> dict | None:
    if len(rows) < 10:
        return None
    dates = [d for d, _ in rows]
    r = np.array([x for _, x in rows])
    f_kelly, _ = B.kelly_opt(r)
    f_used = min(B.KELLY_FRAC * f_kelly, B.KELLY_CAP) if f_kelly > 0 else 0.0
    eq = np.exp(np.log1p((f_used * r).clip(min=-0.999)).cumsum())
    yrs = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    cagr = eq[-1] ** (1 / yrs) - 1
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    win = float((r > 0).mean())
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return dict(n=len(r), f=f_used, cagr=cagr, mdd=mdd, win=win, calmar=calmar,
               worst=float(r.min()), start=dates[0], end=dates[-1])


def buy_and_hold(etf: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    s = etf[(etf.index >= start) & (etf.index <= end)]
    eq = s / s.iloc[0]
    yrs = max((end - start).days / 365.25, 1e-9)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    return dict(cagr=cagr, mdd=mdd, calmar=cagr / abs(mdd) if mdd != 0 else np.nan)


def plot_heatmap(grid: pd.DataFrame, out_path: Path) -> None:
    piv = grid.pivot(index="p", columns="target_delta", values="calmar")
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    im = ax.imshow(piv.values, aspect="auto", cmap="Blues", origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{q:.2f}" for q in piv.columns], color=C_MUTED, fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, color=C_MUTED, fontsize=9)
    ax.set_xlabel("target |delta| (short strike)", color=C_INK, fontsize=10)
    ax.set_ylabel("protective wing steps (p)", color=C_INK, fontsize=10)
    ax.set_title("Delta-targeted condor, Calmar (CAGR / |MDD|), full 2015-2026 "
                 "history", color=C_INK, fontsize=12, loc="left", pad=10)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                       color=C_INK)
    fig.colorbar(im, ax=ax, label="Calmar")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


def main() -> None:
    chain = IC.load_local_chain()
    etf = IC.load_local_etf()          # QFQ-adjusted -- buy & hold comparison only
    spot_series = chain_spot_series(chain)   # RAW -- option settlement
    print(f"chain: {chain[C_DATE].min().date()} -> {chain[C_DATE].max().date()}, "
          f"{chain[C_DATE].nunique()} trading days")

    trades = run_grid(chain, spot_series)

    rows = []
    for (target_delta, p), lst in trades.items():
        s = cell_stats(lst)
        if s is not None:
            rows.append(dict(target_delta=target_delta, p=p, **s))
    grid = pd.DataFrame(rows)
    grid.to_csv(OUTDIR / "condor_delta_grid.csv", index=False)

    print("\nTop 8 cells by Calmar:")
    top = grid.sort_values("calmar", ascending=False).head(8)
    for _, r in top.iterrows():
        print(f"  delta={r['target_delta']:.2f}  p={int(r['p'])}  n={int(r['n']):>3}  "
              f"f={r['f']:.2f}  CAGR={r['cagr']:+.2%}  MDD={r['mdd']:+.2%}  "
              f"win={r['win']:.0%}  Calmar={r['calmar']:.2f}  worst={r['worst']:+.2f}")

    start, end = grid["start"].min(), grid["end"].max()
    bh = buy_and_hold(etf, start, end)
    print(f"\nBuy & hold 50ETF over the same window ({start.date()} -> {end.date()}):")
    print(f"  CAGR={bh['cagr']:+.2%}  MDD={bh['mdd']:+.2%}  Calmar={bh['calmar']:.2f}")

    out_path = OUTDIR / "condor_delta_heatmap.png"
    plot_heatmap(grid, out_path)
    print(f"\nSaved: {OUTDIR/'condor_delta_grid.csv'} and {out_path.name}")
    print("Caveats: real quoted closes can be thin/stale on far-OTM strikes on "
          "some days (bid/ask wide even though a last-traded price exists); "
          "slippage is the same flat per-leg assumption used elsewhere, not "
          "size-dependent. ~130-month sample, still a backtest, not a live "
          "track record.")


if __name__ == "__main__":
    main()
