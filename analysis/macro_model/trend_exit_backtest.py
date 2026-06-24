# -*- coding: utf-8 -*-
"""
trend_exit_backtest.py -- does cutting size on a one-sided move help the short-vol book?
=======================================================================================
A long butterfly is short realized vol: it wants 50ETF to CHOP near the body and
is hurt when spot trends away in one direction. The oneside_scan census showed
clean trends are real but rare (ER>=0.5 ~ 1 in 5 months; 90% never happens), so
the actionable bar is an efficiency ratio around 0.5-0.6, tracked in REAL TIME.

Rule under test (no lookahead): track the efficiency ratio of the path SINCE
ENTRY,  ER(entry->d) = |P_d - P_entry| / sum|daily change|.  On the first day d
(after MIN_DAYS) that ER >= threshold, REDUCE the short-vol position -- because
the sold legs tie up margin, you cannot always close cleanly, you scale down.
Keep a fraction K running to the model's normal exit; lock the closed (1-K) part
at that day's real-IV mark. K = 1.0 is the base model (no reduction); K = 0.0 is
a full close (shown for reference / when margin allows it).

    contrib(month) = f_used * [ (1-K) * r_at_trend_day  +  K * r_normal_exit ]

Everything else (structure selection, pricing, the normal exit) is identical to
butterfly_timing_backtest, reusing the shared helpers in shock_overlay_backtest.

Run (from inside macro_model\, needs internet for the SSE + 50ETF feeds):
    python trend_exit_backtest.py
Outputs (out/):
    trend_exit_summary.csv            -- one row per (threshold, keep) scenario
    trend_exit_trigger_diagnostic.csv -- per-threshold: did reducing actually help?
    trend_exit_equity.png             -- equity curves
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import butterfly_timing_backtest as B
import butterfly_param_search as PS                     # installs the IV cache
from shock_overlay_backtest import select_structure, mark_position

OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

# --- rule knobs --------------------------------------------------------------
THRESHOLDS = [0.50, 0.55, 0.60]    # efficiency-ratio trigger levels
KEEPS      = [0.5, 0.25, 0.0]      # fraction of the short-vol position KEPT after the cut
MIN_DAYS   = 5                     # min sessions since entry before the rule can fire


def er_trigger_day(path: pd.Series, thr: float, min_days: int):
    """First session (after min_days) where ER(entry->d) >= thr. No lookahead:
    ER at d uses only closes through d. Returns a Timestamp or None."""
    vals = path.values.astype(float)
    idx = path.index
    for k in range(min_days, len(vals)):
        seg = vals[:k + 1]
        gross = float(np.abs(np.diff(seg)).sum())
        if gross <= 0:
            continue
        er = abs(seg[-1] - seg[0]) / gross
        if er >= thr:
            return idx[k]
    return None


# =============================================================================
# Build per-month plans once: structure, normal exit, and per-threshold trend cut
# =============================================================================
def build_plans():
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
            plans.append(dict(t=t, side="flat")); continue

        spot = float(etf.asof(entry_date))
        expiry, exp_date = B.choose_expiry(iv_tbl, entry_date)
        if expiry is None or pd.Timestamp(exp_date) > last_daily:
            plans.append(dict(t=t, side="flat")); continue
        T = max((exp_date - entry_date.date()).days, 1) / 365.0
        smile = B.otm_smile(iv_tbl, expiry, spot)
        if len(smile) < 5:
            plans.append(dict(t=t, side="flat")); continue

        atm_iv = float(smile.iloc[(smile.index - spot).to_series().abs().values.argmin()])
        ratio = atm_iv / model_ann if model_ann > 0 else np.nan
        Tm = T * 12.0
        d_h = draws.mean() * Tm + (draws - draws.mean()) * np.sqrt(Tm)
        ST = spot * np.exp(d_h)
        side, struct, kelly_f = select_structure(smile, spot, T, ST, ratio)
        if struct is None:
            plans.append(dict(t=t, side="flat")); continue

        cost = struct["cost"]
        f_used = min(B.KELLY_FRAC * kelly_f, B.KELLY_CAP)

        # normal (model) exit -- the base outcome and the kept-fraction outcome
        realized_n, _kind, exit_n = mark_position(side, struct, smile, entry_date,
                                                  exp_date, expiry, etf, last_daily)
        r_normal = realized_n / cost - 1.0
        exit_n = pd.Timestamp(exit_n)

        is_sv = (side == "butterfly")
        # holding window for the trend rule: (entry, cap], cap = exp - EXIT_DTE
        exp_ts = pd.Timestamp(exp_date)
        cap_ts = (exp_ts - pd.Timedelta(days=B.EXIT_DTE)) if B.EXIT_DTE is not None else exp_ts
        path = etf.loc[entry_date:min(cap_ts, last_daily)]

        thr_map = {}
        for thr in THRESHOLDS:
            triggered, r_trend, tday = False, r_normal, None
            if is_sv:
                tday = er_trigger_day(path, thr, MIN_DAYS)
                # only meaningful if the trend fires BEFORE the model already exited
                if tday is not None and tday < exit_n:
                    rt, _, _ = mark_position(side, struct, smile, entry_date,
                                             exp_date, expiry, etf, last_daily,
                                             force_exit_date=tday)
                    r_trend, triggered = rt / cost - 1.0, True
            thr_map[thr] = dict(triggered=triggered, r_trend=r_trend, tday=tday)

        plans.append(dict(t=t, side=side, is_sv=is_sv, f_used=f_used,
                          r_normal=r_normal, thr_map=thr_map))
        fired = {thr: thr_map[thr]["triggered"] for thr in THRESHOLDS}
        print(f"  plan {t.date()} {side:9} f={f_used:.2f} r_norm={r_normal:+.2f} "
              f"trend_fired={fired}")

    return plans, X, y, df


# =============================================================================
# Scenario evaluation
# =============================================================================
def run_scenario(plans, thr, keep):
    """keep = fraction of the short-vol position kept to the normal exit.
    keep = 1.0 -> base (no reduction)."""
    rows = []
    for p in plans:
        if p.get("side", "flat") == "flat":
            rows.append(dict(date=p["t"], traded=0, contrib=0.0, r_trade=np.nan)); continue
        f_used, r_normal = p["f_used"], p["r_normal"]
        if p["is_sv"] and thr in p["thr_map"] and p["thr_map"][thr]["triggered"] and keep < 1.0:
            r_t = p["thr_map"][thr]["r_trend"]
            r_eff = (1.0 - keep) * r_t + keep * r_normal
        else:
            r_eff = r_normal
        rows.append(dict(date=p["t"], traded=1, side=p["side"],
                         r_trade=r_eff, contrib=f_used * r_eff))
    return pd.DataFrame(rows).set_index("date")


def summarize(bt, df, y, label):
    bt2, _bh, m = B.equity_and_metrics(bt, df, y)
    cal = (m["rule_cagr"] / abs(m["rule_mdd"])
           if m["rule_mdd"] and np.isfinite(m["rule_mdd"]) and m["rule_mdd"] != 0
           else np.nan)
    return dict(scenario=label, total=m["rule_total"], cagr=m["rule_cagr"],
                mdd=m["rule_mdd"], calmar=cal,
                worst_month=float(bt["contrib"].min()),
                winrate=m["trade_winrate"], trades=m["trades"]), bt2["equity"]


def trigger_diagnostic(plans):
    """For each threshold: on the months the trend FIRED, did closing at the
    trend day beat holding to the normal exit? (r_trend vs r_normal)."""
    rows = []
    for thr in THRESHOLDS:
        fired = [p for p in plans if p.get("is_sv")
                 and p["thr_map"][thr]["triggered"]]
        if not fired:
            rows.append(dict(threshold=thr, n_fired=0)); continue
        rt = np.array([p["thr_map"][thr]["r_trend"] for p in fired])
        rn = np.array([p["r_normal"] for p in fired])
        rows.append(dict(threshold=thr, n_fired=len(fired),
                         mean_r_trend=float(rt.mean()), mean_r_normal=float(rn.mean()),
                         reduce_helped_share=float((rt > rn).mean()),
                         avg_gain_from_cut=float((rt - rn).mean())))
    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("Trend (efficiency-ratio) partial-reduction backtest on the short-vol book")
    print(f"THRESHOLDS={THRESHOLDS}  KEEPS={KEEPS}  MIN_DAYS={MIN_DAYS}")
    print("=" * 72)

    plans, X, y, df = build_plans()

    summary, equities = [], {}
    base_bt = run_scenario(plans, THRESHOLDS[0], keep=1.0)
    row, eq = summarize(base_bt, df, y, "base")
    summary.append(row); equities["base"] = eq
    print(f"  [base] total={row['total']:.3f} cagr={row['cagr']:+.3f} "
          f"mdd={row['mdd']:+.3f} calmar={row['calmar']:+.2f} worst={row['worst_month']:+.3f}")

    for thr in THRESHOLDS:
        for keep in KEEPS:
            name = f"ER>={thr:.2f} keep{int(keep*100)}"
            bt = run_scenario(plans, thr, keep)
            bt.to_csv(OUTDIR / f"trend_exit_{thr:.2f}_keep{int(keep*100)}.csv")
            r, eq = summarize(bt, df, y, name)
            summary.append(r); equities[name] = eq
            print(f"  [{name:18}] total={r['total']:.3f} cagr={r['cagr']:+.3f} "
                  f"mdd={r['mdd']:+.3f} calmar={r['calmar']:+.2f} worst={r['worst_month']:+.3f}")

    sm = pd.DataFrame(summary).set_index("scenario")
    sm.to_csv(OUTDIR / "trend_exit_summary.csv")
    diag = trigger_diagnostic(plans)
    diag.to_csv(OUTDIR / "trend_exit_trigger_diagnostic.csv", index=False)

    print("\n=== scenario comparison ===")
    print(sm.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\n=== did the cut help on the months it fired? ===")
    print(diag.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, eq in equities.items():
        lw = 2.2 if name == "base" else 1.2
        ax.plot(eq.index, eq.values, lw=lw, label=name)
    ax.set_yscale("log"); ax.axhline(1.0, color="grey", lw=0.7)
    ax.set_ylabel("equity (log, start=1)"); ax.set_xlabel("month")
    ax.set_title("Short-vol book with a rolling-ER partial-reduction exit")
    ax.legend(fontsize=8, ncol=2); fig.autofmt_xdate()
    fig.savefig(OUTDIR / "trend_exit_equity.png", dpi=130, bbox_inches="tight")
    print(f"\nsaved -> {OUTDIR / 'trend_exit_summary.csv'}")
    print(f"saved -> {OUTDIR / 'trend_exit_trigger_diagnostic.csv'}")
    print(f"saved -> {OUTDIR / 'trend_exit_equity.png'}")


if __name__ == "__main__":
    main()
