# -*- coding: utf-8 -*-
"""
shock_overlay_backtest.py -- does a DOMESTIC risk-off overlay help the short-vol book?
=====================================================================================
The butterfly + iron-condor model is short realized vol. The global-overlay test
showed the book's tail is DOMESTIC (China-specific), so this version watches the
domestic signal -- QVIX implied-vol spikes + 50ETF's own move (see shock.py) --
APPLIED THROUGH THE HOLDING PERIOD, and asks whether reacting earns its keep, by
re-running the monthly vol-timing backtest under four scenarios:

    base      : the model as-is (no overlay).
    derisk    : on a risk-off REGIME at month entry, cut the short-vol size
                (DERISK_MULT, 0 = skip the trade); AND close the short-vol
                position EARLY on the first risk-off day inside the holding
                window (a QVIX spike, a 50ETF crash, or a developing regime).
    hedge     : on a risk-off REGIME at entry (e.g. QVIX already elevated), ALSO
                buy a long ATM straddle (HEDGE_RISK of bankroll); the hedge is
                taken off at the first acute shock day (lock the vol pop) else
                held to expiry.
    both      : derisk + hedge together.

All marks reuse the exact primitives in butterfly_timing_backtest (B.*), so the
only thing that changes vs the live model is the overlay. CN signals are
same-session (decided + acted at the close, DOMESTIC_LAG=0): NO lookahead.

Run (from inside macro_model\, needs internet for the SSE + global feeds):
    python shock_overlay_backtest.py

Outputs (out/):
    shock_overlay_summary.csv     -- one row per scenario (CAGR/MDD/Calmar/...)
    shock_overlay_<scenario>.csv  -- per-month trades for each scenario
    shock_overlay_equity.png      -- equity curves, four scenarios overlaid

NOTE: thresholds in shock.py (ACUTE_1D, DD_THR, ...) and the two knobs below are
first-guesses. Read the summary, then tune. A useful overlay should LIFT Calmar
and (especially) worst_month / mdd without gutting total return.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import butterfly_timing_backtest as B        # core backtest primitives + constants
import butterfly_param_search as PS          # importing installs the IV fetch cache
import shock

OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

# --- overlay knobs (tune after the first run) --------------------------------
DERISK_MULT = 0.0        # short-vol size multiplier on a risk-off regime (0 = skip)
HEDGE_RISK  = 0.10       # bankroll fraction allocated to the long-vol hedge

SCENARIOS = {
    "base":   dict(derisk=False, derisk_mult=1.0,         shock_exit=False, hedge=False, hedge_risk=0.0),
    "derisk": dict(derisk=True,  derisk_mult=DERISK_MULT, shock_exit=True,  hedge=False, hedge_risk=0.0),
    "hedge":  dict(derisk=False, derisk_mult=1.0,         shock_exit=False, hedge=True,  hedge_risk=HEDGE_RISK),
    "both":   dict(derisk=True,  derisk_mult=DERISK_MULT, shock_exit=True,  hedge=True,  hedge_risk=HEDGE_RISK),
}


# =============================================================================
# Structure selection (identical gate/Kelly logic as B.run)
# =============================================================================
def select_structure(smile, spot, T, ST, ratio):
    if ratio >= B.HI_RATIO:
        best = None
        for n in range(1, B.MAX_WING_STEPS + 1):
            pr = B.price_butterfly(smile, spot, T, n)
            if not pr or pr["cost"] <= 1e-6:
                continue
            pay = B.payoff_butterfly(ST, pr["strikes"])
            f_, g_ = B.kelly_opt(pay / pr["cost"] - 1.0)
            if g_ > 0 and (best is None or g_ > best[1]):
                best = (pr, g_, f_)
        if best:
            return "butterfly", best[0], best[2]
    elif ratio <= B.LO_RATIO:
        pr = B.price_straddle(smile, spot, T)
        if pr["cost"] > 1e-6:
            pay = B.payoff_straddle(ST, pr["strikes"])
            f_, g_ = B.kelly_opt(pay / pr["cost"] - 1.0)
            if g_ > 0:
                return "straddle", pr, f_
    return "flat", None, 0.0


# =============================================================================
# Exit marking (faithful to B.run; adds an optional forced early-exit date)
# =============================================================================
def mark_position(side, struct, smile, entry_date, exp_date, expiry, etf, last_daily,
                  force_exit_date=None, skip_tp_sl=False):
    """Return (realized_value, exit_kind, exit_date) for one option structure.
    force_exit_date: if given (an acute-shock CN day), mark at REAL IV on/just
    before it. skip_tp_sl: bypass the model's TP/SL scan (used for the hedge,
    which only exits on a shock or at expiry)."""
    cost, strikes = struct["cost"], struct["strikes"]
    val_fn = B.value_butterfly if side == "butterfly" else B.value_straddle
    pay_fn = B.payoff_butterfly if side == "butterfly" else B.payoff_straddle
    exp_ts = pd.Timestamp(exp_date)
    cap_ts = exp_ts - pd.Timedelta(days=B.EXIT_DTE) if B.EXIT_DTE is not None else exp_ts

    # 0) forced early exit on an acute shock -> real IV mark at that date
    if force_exit_date is not None:
        exit_tbl, ex_date = B.find_iv_on_or_before(pd.Timestamp(force_exit_date))
        if (exit_tbl is not None and ex_date > entry_date and ex_date < exp_ts
                and ex_date <= last_daily):
            spot_ex = float(etf.asof(ex_date))
            smile_ex = B.otm_smile(exit_tbl, expiry, spot_ex)
            T_ex = max((exp_date - ex_date.date()).days, 1) / 365.0
            if len(smile_ex) >= 3:
                return float(val_fn(smile_ex, spot_ex, T_ex, strikes)), "shock_exit", ex_date
        # if it couldn't be booked, fall through to the normal path

    # 1) TP/SL scan at fixed entry IV (unless skipped), window (entry, cap]
    hit_date, hit_kind = None, None
    if not skip_tp_sl:
        tgt_val = B.profit_target_value(side, cost, strikes)
        stop_val = cost * (1.0 - B.STOP_LOSS) if B.STOP_LOSS is not None else None
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

    # 2) decide exit date/kind
    if hit_date is not None:
        exit_kind, want = hit_kind, hit_date
    elif (not skip_tp_sl) and B.EXIT_DTE is not None and entry_date < cap_ts < exp_ts:
        exit_kind, want = "roll", cap_ts
    else:
        exit_kind, want = "expiry", exp_ts

    # 3) book it
    realized, exit_date = None, None
    if exit_kind in ("target", "stop", "roll"):
        exit_tbl, ex_date = B.find_iv_on_or_before(want)
        if (exit_tbl is not None and ex_date > entry_date and ex_date < exp_ts
                and ex_date <= last_daily):
            spot_ex = float(etf.asof(ex_date))
            smile_ex = B.otm_smile(exit_tbl, expiry, spot_ex)
            T_ex = max((exp_date - ex_date.date()).days, 1) / 365.0
            if len(smile_ex) >= 3:
                realized = float(val_fn(smile_ex, spot_ex, T_ex, strikes))
                exit_date = ex_date
        if realized is None:
            exit_kind = "expiry"
    if exit_kind == "expiry":
        S_used = float(etf.asof(exp_ts))
        realized = float(pay_fn(np.array([S_used]), strikes)[0])
        exit_date = exp_date
    return realized, exit_kind, exit_date


# =============================================================================
# Build the per-month plans ONCE (structure + acute days), overlay-independent
# =============================================================================
def build_plans(shock_tbl, lag_days=0):
    df = B.load_panel()
    X, y, _ = B.build_xy(df)
    idx = X.index

    etf = B.ak.fund_etf_hist_sina(symbol="sh510050")
    etf["date"] = pd.to_datetime(etf["date"])
    etf = etf.set_index("date")["close"].sort_index()
    last_daily = etf.index.max()

    plans = []
    for ti in range(B.MIN_TRAIN, len(X)):
        t = idx[ti]
        draws = B.fit_predict_conjugate(X.values[:ti], y.values[:ti],
                                        X.values[ti:ti + 1])[0]
        model_ann = float(draws.std()) * np.sqrt(12)

        iv_tbl, entry_date = None, None
        for back in range(0, 7):
            d = pd.Timestamp((t - pd.Timedelta(days=back)).date())
            try:
                tbl = B.fetch_iv_table(d)
            except Exception:
                tbl = None
            if tbl is not None:
                iv_tbl, entry_date = tbl, d
                break
        if iv_tbl is None:
            plans.append(dict(t=t, side="flat", struct=None)); continue

        spot = float(etf.asof(entry_date))
        expiry, exp_date = B.choose_expiry(iv_tbl, entry_date)
        if expiry is None or pd.Timestamp(exp_date) > last_daily:
            plans.append(dict(t=t, side="flat", struct=None)); continue
        T = max((exp_date - entry_date.date()).days, 1) / 365.0
        smile = B.otm_smile(iv_tbl, expiry, spot)
        if len(smile) < 5:
            plans.append(dict(t=t, side="flat", struct=None)); continue

        atm_iv = float(smile.iloc[(smile.index - spot).to_series().abs().values.argmin()])
        ratio = atm_iv / model_ann if model_ann > 0 else np.nan
        Tm = T * 12.0
        d_h = draws.mean() * Tm + (draws - draws.mean()) * np.sqrt(Tm)
        ST = spot * np.exp(d_h)

        side, struct, kelly_f = select_structure(smile, spot, T, ST, ratio)

        # domestic risk-off CN sessions inside the holding window.
        #   acute_dates : QVIX spike / 50ETF crash  -> hedge take-profit timing
        #   exit_dates  : any risk_off (acute|regime) -> short-vol early-exit
        exp_ts = pd.Timestamp(exp_date)
        window = etf.index[(etf.index > entry_date) & (etf.index <= exp_ts)
                           & (etf.index <= last_daily)]
        acute_dates = [d for d in window
                       if shock.is_risk_off(shock_tbl, d, "acute", lag_days)]
        exit_dates = [d for d in window
                      if shock.is_risk_off(shock_tbl, d, "risk_off", lag_days)]

        # ATM straddle priced at entry for the hedge leg (reused across scenarios)
        hedge_struct = B.price_straddle(smile, spot, T) if struct is not None else None

        plans.append(dict(
            t=t, entry_date=entry_date, spot=spot, expiry=expiry, exp_date=exp_date,
            T=T, smile=smile, ratio=ratio, side=side, struct=struct, kelly_f=kelly_f,
            regime_entry=shock.is_risk_off(shock_tbl, entry_date, "regime", lag_days),
            acute_dates=acute_dates, exit_dates=exit_dates, hedge_struct=hedge_struct))
        print(f"  plan {t.date()} {side:9} ratio={ratio:4.2f} "
              f"regime_in={plans[-1]['regime_entry']} "
              f"n_riskoff={len(exit_dates)} n_acute={len(acute_dates)}")

    return plans, X, y, df, etf, last_daily


# =============================================================================
# Run one scenario from the shared plans
# =============================================================================
def run_scenario(plans, cfg, etf, last_daily):
    rows = []
    for p in plans:
        t = p["t"]
        if p.get("side", "flat") == "flat" or p.get("struct") is None:
            rows.append(dict(date=t, traded=0, side="flat", contrib=0.0,
                             r_trade=np.nan)); continue

        side, struct, kelly_f = p["side"], p["struct"], p["kelly_f"]
        f_used = min(B.KELLY_FRAC * kelly_f, B.KELLY_CAP)
        is_sv = (side == "butterfly")          # the short-vol months
        regime, acute = p["regime_entry"], p["acute_dates"]
        exitd = p.get("exit_dates", [])

        size_mult = cfg["derisk_mult"] if (cfg["derisk"] and is_sv and regime) else 1.0
        force = exitd[0] if (cfg["shock_exit"] and is_sv and len(exitd)) else None

        realized, kind, _ = mark_position(side, struct, p["smile"], p["entry_date"],
                                          p["exp_date"], p["expiry"], etf, last_daily,
                                          force_exit_date=force)
        r_trade = realized / struct["cost"] - 1.0
        contrib_sv = size_mult * f_used * r_trade

        contrib_h, r_hedge = 0.0, np.nan
        if (cfg["hedge"] and is_sv and regime and p["hedge_struct"] is not None
                and p["hedge_struct"]["cost"] > 1e-6):
            hforce = acute[0] if len(acute) else None
            hr, _, _ = mark_position("straddle", p["hedge_struct"], p["smile"],
                                     p["entry_date"], p["exp_date"], p["expiry"],
                                     etf, last_daily, force_exit_date=hforce,
                                     skip_tp_sl=True)
            r_hedge = hr / p["hedge_struct"]["cost"] - 1.0
            contrib_h = cfg["hedge_risk"] * r_hedge

        rows.append(dict(date=t, traded=1, side=side, regime=regime,
                         n_riskoff=len(exitd), n_acute=len(acute), exit_kind=kind,
                         r_trade=r_trade, size_mult=size_mult, r_hedge=r_hedge,
                         contrib_sv=contrib_sv, contrib_h=contrib_h,
                         contrib=contrib_sv + contrib_h))
    return pd.DataFrame(rows).set_index("date")


def summarize(bt, df, y, label):
    bt2, _bh, m = B.equity_and_metrics(bt, df, y)
    cal = (m["rule_cagr"] / abs(m["rule_mdd"])
           if m["rule_mdd"] and np.isfinite(m["rule_mdd"]) and m["rule_mdd"] != 0
           else np.nan)
    reg = bt[bt.get("regime", False) == True] if "regime" in bt else bt.iloc[0:0]
    row = dict(scenario=label, total=m["rule_total"], cagr=m["rule_cagr"],
               mdd=m["rule_mdd"], calmar=cal, worst_month=float(bt["contrib"].min()),
               winrate=m["trade_winrate"], trades=m["trades"],
               regime_months=int(len(reg)),
               regime_contrib=float(reg["contrib"].sum()) if len(reg) else 0.0)
    return row, bt2["equity"]


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 72)
    print("Domestic risk-off overlay backtest -- base vs derisk vs hedge vs both")
    print(f"DERISK_MULT={DERISK_MULT}  HEDGE_RISK={HEDGE_RISK}")
    print(f"shock.py domestic: ETF_ACUTE={shock.ETF_ACUTE_1D} QVIX_JUMP={shock.QVIX_JUMP} "
          f"ETF_DD_THR={shock.ETF_DD_THR} QVIX_PCTL_THR={shock.QVIX_PCTL_THR}")
    print("=" * 72)

    shock_tbl = shock.domestic_risk_table()
    print(f"domestic risk-off table: {len(shock_tbl)} sessions, "
          f"acute days={int(shock_tbl['acute'].sum())}, "
          f"regime days={int(shock_tbl['regime'].sum())}, "
          f"any={int(shock_tbl['risk_off'].sum())}")

    plans, X, y, df, etf, last_daily = build_plans(shock_tbl, lag_days=shock.DOMESTIC_LAG)

    summary, equities = [], {}
    for name, cfg in SCENARIOS.items():
        bt = run_scenario(plans, cfg, etf, last_daily)
        bt.to_csv(OUTDIR / f"shock_overlay_{name}.csv")
        row, eq = summarize(bt, df, y, name)
        summary.append(row); equities[name] = eq
        print(f"  [{name:6}] total={row['total']:.3f} cagr={row['cagr']:+.3f} "
              f"mdd={row['mdd']:+.3f} calmar={row['calmar']:+.2f} "
              f"worst={row['worst_month']:+.3f}")

    sm = pd.DataFrame(summary).set_index("scenario")
    sm.to_csv(OUTDIR / "shock_overlay_summary.csv")
    print("\n=== scenario comparison ===")
    print(sm.to_string(float_format=lambda v: f"{v:+.3f}"))

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, eq in equities.items():
        ax.plot(eq.index, eq.values, lw=1.7, label=name)
    ax.set_yscale("log"); ax.axhline(1.0, color="grey", lw=0.7)
    ax.set_ylabel("equity (log, start=1)"); ax.set_xlabel("month")
    ax.set_title("Short-vol book under a global risk-off overlay")
    ax.legend(); fig.autofmt_xdate()
    fig.savefig(OUTDIR / "shock_overlay_equity.png", dpi=130, bbox_inches="tight")
    print(f"\nsaved -> {OUTDIR / 'shock_overlay_summary.csv'}")
    print(f"saved -> {OUTDIR / 'shock_overlay_equity.png'}")


if __name__ == "__main__":
    main()
