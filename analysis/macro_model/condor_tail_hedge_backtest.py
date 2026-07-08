# -*- coding: utf-8 -*-
"""
condor_tail_hedge_backtest.py
================================
Quantifies the "just buy a cheap tail hedge" idea flagged as untested at the
end of condor_trend_exit_backtest.py's investigation (every REACTIVE
mid-trade rule tried so far -- PnL stop-loss, delta-breach, trend-exit --
fails OOS because they all cut into the position's own recovery process).
A hedge is mechanically different: a separate small position bought
alongside the condor at entry, sized to pay off exactly when the condor's
own wings get blown through, independent of any mid-trade reaction.

Each month, alongside the usual delta-targeted condor (target_delta=0.20,
p=3, unchanged from condor_delta_backtest.py), also BUY one far-OTM put and
one far-OTM call at a lower target |delta| (the hedge grid below), same
expiry, priced off the same real quoted closes + per-leg slippage. Held to
the SAME expiry as the condor (no separate management). Combined PnL =
condor PnL + hedge PnL; combined_r uses the condor's OWN max_loss as the
denominator (isolates the overlay's effect on the identical position) --
capital_r additionally divides by (max_loss + hedge cost), the more honest
number since the hedge premium is real capital committed every month, not
free.

This uses condor_delta_backtest.py's INDEPENDENT MONTHLY entries (not the
continuous roll chain) -- appropriate here since, unlike an exit/roll rule,
a hedge never changes trade boundaries or creates path-dependence between
consecutive months, so there's no reason to leave the simpler methodology
condor_delta_backtest.py itself already established.

OOS discipline (same fixed split used throughout this investigation):
  DISCOVERY  2015-02-09 -> 2023-12-31
  HOLDOUT    2024-01-01 -> 2026-07-08, touched once at the end.
Grid: hedge target |delta| in {0.03, 0.05, 0.08, 0.10, 0.15}.

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

import iron_condor_backtest as IC   # noqa: E402
import condor_delta_backtest as CD  # noqa: E402

OUTDIR = IC.OUTDIR
SPLIT_DATE = pd.Timestamp("2024-01-01")
PRIMARY = dict(target_delta=0.20, p=3)
SECOND = dict(target_delta=0.25, p=3)
HEDGE_DELTA_GRID = [0.03, 0.05, 0.08, 0.10, 0.15]

C_SURFACE, C_INK, C_MUTED = "#fcfcfb", "#0b0b0b", "#898781"
C_SERIES = ["#3b6ea5", "#c1502e", "#5a8f5a", "#8a7ab5"]


def window(chain: pd.DataFrame, spot_series: pd.Series, start, end):
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


# ============================================================== hedge leg pricing
def price_hedge(sub: pd.DataFrame, hedge_delta: float):
    """One far-OTM put + one far-OTM call, both bought (debit), priced off
    real quoted closes. Returns dict(Kp, Kc, cost, put_delta, call_delta) or
    None if either leg can't be priced that day."""
    put, call = CD.pick_by_delta(sub, "P", hedge_delta), CD.pick_by_delta(sub, "C", hedge_delta)
    if put is None or call is None:
        return None
    put_px, call_px = float(put[CD.C_CLOSE]), float(call[CD.C_CLOSE])
    if pd.isna(put_px) or pd.isna(call_px):
        return None
    cost = put_px + call_px + 2 * CD.SLIP_PER_LEG   # buying -- slippage adds to cost, not subtracts
    return dict(Kp=float(put[CD.C_STRIKE]), Kc=float(call[CD.C_STRIKE]), cost=cost,
                put_delta=float(put[CD.C_DELTA]), call_delta=float(call[CD.C_DELTA]))


def hedge_payout(hedge: dict, S_exp: float) -> float:
    return max(hedge["Kp"] - S_exp, 0.0) + max(S_exp - hedge["Kc"], 0.0)


# ============================================================== trades
def build_trades(chain: pd.DataFrame, spot_series: pd.Series, target_delta: float, p: int,
                  hedge_delta: float | None) -> list[dict]:
    trades = []
    for d in CD.monthly_entry_dates(chain):
        day = chain[chain[CD.C_DATE] == d]
        if day.empty:
            continue
        exp_date = CD.choose_expiry_for(day)
        if exp_date is None:
            continue
        exp_date = pd.Timestamp(exp_date)
        if exp_date not in spot_series.index:
            continue
        sub = day[day[CD.C_EXP] == exp_date]
        built = CD.build_condor_by_delta(sub, target_delta, p)
        if built is None:
            continue
        S_exp = float(spot_series.loc[exp_date])
        condor_pnl = CD.realized_pnl(built["strikes"], built["credit"], S_exp)

        hedge_cost, hedge_pnl = 0.0, 0.0
        realized_put_delta = realized_call_delta = np.nan
        if hedge_delta is not None:
            hedge = price_hedge(sub, hedge_delta)
            if hedge is None:
                continue   # skip months where the hedge itself can't be priced -- keeps comparison apples-to-apples
            hedge_cost = hedge["cost"]
            hedge_pnl = hedge_payout(hedge, S_exp) - hedge_cost
            realized_put_delta, realized_call_delta = hedge["put_delta"], hedge["call_delta"]

        combined_pnl = condor_pnl + hedge_pnl
        trades.append(dict(
            entry=d, exit=exp_date, condor_pnl=condor_pnl, hedge_pnl=hedge_pnl,
            hedge_cost=hedge_cost, max_loss=built["max_loss"], credit=built["credit"],
            combined_r=combined_pnl / built["max_loss"],
            capital_r=combined_pnl / (built["max_loss"] + hedge_cost),
            condor_r=condor_pnl / built["max_loss"],
            realized_put_delta=realized_put_delta, realized_call_delta=realized_call_delta,
        ))
    return trades


def rows_of(trades: list[dict], col: str) -> list[tuple]:
    return [(t["entry"], t[col]) for t in trades]


def raw_stats(rows: list) -> dict:
    r = np.array([x for _, x in rows])
    return dict(mean_r=float(r.mean()), std_r=float(r.std(ddof=1)) if len(r) > 1 else np.nan,
                win_raw=float((r > 0).mean()))


def equity_curve(rows: list, f_used: float) -> pd.Series:
    dates = [d for d, _ in rows]
    r = np.array([x for _, x in rows])
    eq = np.exp(np.log1p((f_used * r).clip(min=-0.999)).cumsum())
    return pd.Series(eq, index=dates)


# ============================================================== sweep
def sweep(chain: pd.DataFrame, spot: pd.Series, label: str, target_delta: float, p: int) -> pd.DataFrame:
    rows_out = []
    for hd in HEDGE_DELTA_GRID:
        trades = build_trades(chain, spot, target_delta, p, hd)
        rows = rows_of(trades, "combined_r")
        stats = CD.cell_stats(rows)
        if stats is None:
            continue
        rstats = raw_stats(rows)
        cap_rows = rows_of(trades, "capital_r")
        cap_stats = raw_stats(cap_rows)
        avg_cost = float(np.mean([t["hedge_cost"] for t in trades]))
        avg_credit = float(np.mean([t["credit"] for t in trades]))
        avg_put_d = float(np.nanmean([t["realized_put_delta"] for t in trades]))
        avg_call_d = float(np.nanmean([t["realized_call_delta"] for t in trades]))
        rows_out.append(dict(hedge_delta=hd, **stats, **rstats, capital_mean_r=cap_stats["mean_r"],
                              avg_hedge_cost=avg_cost, avg_credit=avg_credit,
                              hedge_cost_pct_credit=avg_cost / avg_credit,
                              realized_put_delta=avg_put_d, realized_call_delta=avg_call_d))
        print(f"  [{label}] hedge_delta={hd:.2f}  n={stats['n']:>3}  "
              f"cost={avg_cost:.4f} ({avg_cost/avg_credit:.0%} of credit={avg_credit:.4f})  "
              f"realized_delta(P/C)=({avg_put_d:+.3f}/{avg_call_d:+.3f})  "
              f"mean_r={rstats['mean_r']:+.4f}  capital_mean_r={cap_stats['mean_r']:+.4f}  "
              f"f={stats['f']:.2f}  CAGR={stats['cagr']:+.2%}  MDD={stats['mdd']:+.2%}  "
              f"win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}")
    return pd.DataFrame(rows_out)


def main() -> None:
    chain_full = IC.load_local_chain()
    spot_full = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()
    disc_end = SPLIT_DATE - pd.Timedelta(days=1)

    print(f"Option chain: {data_start.date()} -> {data_end.date()}")
    print(f"Split date: {SPLIT_DATE.date()}  Discovery: {data_start.date()}->{disc_end.date()}  "
          f"Holdout: {SPLIT_DATE.date()}->{data_end.date()}")
    print(f"Config: delta={PRIMARY['target_delta']} p={PRIMARY['p']} (independent monthly entries, "
          f"same universe as condor_delta_backtest.py)\n")

    disc_chain, disc_spot = window(chain_full, spot_full, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_full, SPLIT_DATE, data_end)

    no_hedge_disc = build_trades(disc_chain, disc_spot, *PRIMARY.values(), None)
    base_disc_rows = rows_of(no_hedge_disc, "condor_r")
    base_disc, base_disc_raw = CD.cell_stats(base_disc_rows), raw_stats(base_disc_rows)
    print(f"discovery baseline (no hedge): n={base_disc['n']}  mean_r={base_disc_raw['mean_r']:+.4f}  "
          f"CAGR={base_disc['cagr']:+.2%}  MDD={base_disc['mdd']:+.2%}  Calmar={base_disc['calmar']:.2f}")

    print(f"\n--- DISCOVERY sweep ({data_start.date()} -> {disc_end.date()}) ---")
    disc_grid = sweep(disc_chain, disc_spot, "discovery", **PRIMARY)
    disc_grid.to_csv(OUTDIR / "condor_tail_hedge_oos_discovery.csv", index=False)

    if disc_grid.empty:
        print("\nNo discovery cell had enough trades to score -- cannot proceed.")
        return
    frozen = disc_grid.sort_values("mean_r", ascending=False).iloc[0]
    frozen_hd = float(frozen["hedge_delta"])
    print(f"\nFROZEN pick from discovery (best mean_r): hedge_delta={frozen_hd:.2f} "
          f"(discovery mean_r={frozen['mean_r']:+.4f} vs no-hedge baseline {base_disc_raw['mean_r']:+.4f}, "
          f"cost={frozen['hedge_cost_pct_credit']:.0%} of credit, CAGR={frozen['cagr']:+.2%}, "
          f"Calmar={frozen['calmar']:.2f})")

    print(f"\n--- HOLDOUT sweep ({SPLIT_DATE.date()} -> {data_end.date()}, context only) ---")
    hold_grid = sweep(hold_chain, hold_spot, "holdout", **PRIMARY)
    hold_grid.to_csv(OUTDIR / "condor_tail_hedge_oos_holdout.csv", index=False)

    no_hedge_hold = build_trades(hold_chain, hold_spot, *PRIMARY.values(), None)
    base_hold_rows = rows_of(no_hedge_hold, "condor_r")
    base_hold, base_hold_raw = CD.cell_stats(base_hold_rows), raw_stats(base_hold_rows)

    print(f"\n--- OOS EVALUATION: frozen hedge_delta={frozen_hd:.2f} applied to holdout-only data ---")
    frozen_trades = build_trades(hold_chain, hold_spot, *PRIMARY.values(), frozen_hd)
    frozen_rows = rows_of(frozen_trades, "combined_r")
    frozen_oos, frozen_oos_raw = CD.cell_stats(frozen_rows), raw_stats(frozen_rows)
    if frozen_oos is None:
        print(f"  frozen hedge on holdout: too few trades to score (n={len(frozen_rows)})")
    else:
        print(f"  frozen hedge on holdout:   n={frozen_oos['n']:>3}  mean_r={frozen_oos_raw['mean_r']:+.4f}  "
              f"f={frozen_oos['f']:.2f}  CAGR={frozen_oos['cagr']:+.2%}  MDD={frozen_oos['mdd']:+.2%}  "
              f"Calmar={frozen_oos['calmar']:.2f}")
    print(f"  holdout baseline (no hedge):  n={base_hold['n']:>3}  mean_r={base_hold_raw['mean_r']:+.4f}  "
          f"f={base_hold['f']:.2f}  CAGR={base_hold['cagr']:+.2%}  MDD={base_hold['mdd']:+.2%}  "
          f"Calmar={base_hold['calmar']:.2f}")
    if not hold_grid.empty:
        hb = hold_grid.sort_values("mean_r", ascending=False).iloc[0]
        print(f"  holdout's OWN best (hedge_delta={hb['hedge_delta']:.2f}, in-sample on holdout, "
              f"context only): mean_r={hb['mean_r']:+.4f} CAGR={hb['cagr']:+.2%} Calmar={hb['calmar']:.2f}")

    print("\n--- Verdict ---")
    if frozen_oos is not None:
        beats = frozen_oos_raw["mean_r"] > base_hold_raw["mean_r"]
        print(f"  Frozen discovery pick on holdout: "
              f"{'beats' if beats else 'does not beat'} the unhedged baseline "
              f"(mean_r {frozen_oos_raw['mean_r']:+.4f} vs {base_hold_raw['mean_r']:+.4f}; "
              f"CAGR {frozen_oos['cagr']:+.2%} vs {base_hold['cagr']:+.2%}).")

    # ---- generalization check on a second entry config ----
    base2_trades = build_trades(hold_chain, hold_spot, *SECOND.values(), None)
    frozen2_trades = build_trades(hold_chain, hold_spot, *SECOND.values(), frozen_hd)
    base2_rows, frozen2_rows = rows_of(base2_trades, "condor_r"), rows_of(frozen2_trades, "combined_r")
    base2, frozen2 = CD.cell_stats(base2_rows), CD.cell_stats(frozen2_rows)
    base2_raw, frozen2_raw = raw_stats(base2_rows), raw_stats(frozen2_rows)
    print(f"\nGeneralization check on delta={SECOND['target_delta']} p={SECOND['p']}, holdout window:")
    if base2:
        print(f"  baseline (no hedge)  mean_r={base2_raw['mean_r']:+.4f} CAGR={base2['cagr']:+.2%} "
              f"MDD={base2['mdd']:+.2%} Calmar={base2['calmar']:.2f}")
    if frozen2:
        print(f"  frozen hedge         mean_r={frozen2_raw['mean_r']:+.4f} CAGR={frozen2['cagr']:+.2%} "
              f"MDD={frozen2['mdd']:+.2%} Calmar={frozen2['calmar']:.2f}")

    # ---- mechanistic check: the 5 known disaster trades ----
    known_disasters = ["2022-12-28", "2018-01-02", "2019-02-12", "2015-08-11", "2020-12-23"]
    all_base = build_trades(chain_full, spot_full, *PRIMARY.values(), None)
    all_frozen = build_trades(chain_full, spot_full, *PRIMARY.values(), frozen_hd)
    base_by_entry = {t["entry"].strftime("%Y-%m-%d"): t for t in all_base}
    frozen_by_entry = {t["entry"].strftime("%Y-%m-%d"): t for t in all_frozen}
    print(f"\n--- Mechanistic check: known disaster trades vs frozen hedge (delta={frozen_hd:.2f}) ---")
    for ds in known_disasters:
        b, f = base_by_entry.get(ds), frozen_by_entry.get(ds)
        if b is None:
            print(f"  {ds}: no matching trade"); continue
        line = f"  {ds}->{b['exit'].date()}  condor r={b['condor_r']:+.3f}"
        if f is not None:
            line += f"  |  combined r={f['combined_r']:+.3f}  (hedge_pnl={f['hedge_pnl']:+.4f}, hedge_cost={f['hedge_cost']:.4f})"
        print(line)

    # ---- equity plot: full-history baseline vs frozen hedge ----
    base_all_rows = rows_of(all_base, "condor_r")
    frozen_all_rows = rows_of(all_frozen, "combined_r")
    base_all, frozen_all = CD.cell_stats(base_all_rows), CD.cell_stats(frozen_all_rows)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    if base_all:
        curve = equity_curve(base_all_rows, base_all["f"])
        ax.plot(curve.index, curve.values, label="condor, no hedge (baseline)", color=C_SERIES[0], linewidth=1.6)
    if frozen_all:
        curve = equity_curve(frozen_all_rows, frozen_all["f"])
        ax.plot(curve.index, curve.values, label=f"condor + tail hedge (delta={frozen_hd:.2f})",
                 color=C_SERIES[1], linewidth=1.6)
    ax.axvline(SPLIT_DATE, color=C_MUTED, linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_title(f"Condor + tail hedge vs unhedged, full 2015-2026 history "
                 f"(delta={PRIMARY['target_delta']} p={PRIMARY['p']})",
                 color=C_INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=C_MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "condor_tail_hedge_equity.png", dpi=150, facecolor=C_SURFACE)
    plt.close(fig)

    print(f"\nSaved: {OUTDIR/'condor_tail_hedge_oos_discovery.csv'}, "
          f"{OUTDIR/'condor_tail_hedge_oos_holdout.csv'}, and condor_tail_hedge_equity.png")


if __name__ == "__main__":
    main()
