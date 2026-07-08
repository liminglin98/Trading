# -*- coding: utf-8 -*-
"""
condor_asymmetric_skew_backtest.py
=====================================
The one idea not yet tried in this investigation: instead of reacting to
the condor's own PnL (§3.1-3.2, §3.7 -- all reactive, all failed), buying a
separate hedge (§3.8 -- too expensive), or holding an independent trend
position (§3.9 -- crash-alpha thesis, didn't show up here), this changes
the condor's OWN CONSTRUCTION at entry: an unbalanced/skewed condor, sold
further from the money on whichever side the prevailing trend threatens,
same as real options desks do on a directional view ("broken wing" /
skewed condor). §3.4 found 4 of 5 disasters ran the CALL side over during
uptrends -- so on a bullish month, sell the call further OTM (safer, less
credit) and leave the put where it is; symmetric mirror on bearish months.

This is genuinely different from §3.7/§3.9: it's neither reactive (the
skew is fixed at entry, no mid-trade rule) nor a second position (it's
still ONE condor, just an unbalanced one) -- it directly targets the
known failure mode via the construction itself.

Design:
  TREND SIGNAL   sign of trailing 3-month SSE Composite momentum
                 (mom_3m, index_trend_signal.py) as of the entry date --
                 fixed by convention at 3m (the project's own established
                 default lookback, used already in panel_monthly's
                 etf/idx mom_3m columns), NOT tuned here, to keep this
                 test to one free axis.
  SKEW           if bullish (mom_3m > 0): call sold at (target_delta - skew)
                 -- further OTM, safer, less credit; put stays at
                 target_delta. Mirror for bearish. skew=0.00 exactly
                 reproduces the unskewed condor_delta_backtest.py baseline
                 (sanity-checked in main()).
  WING WIDTH     unchanged (p listed strikes past each short, same p for
                 both sides) -- wing width was already tested on its own
                 in §3.5; this isolates the skew lever specifically.

PARAMETERS:
  SWEPT: skew magnitude (delta reduction on the threatened side) in
         {0.00 (baseline), 0.03, 0.05, 0.08, 0.10}.
  FIXED BY CONVENTION: momentum lookback = 3m, deadband = 0% (always
         skews one way; no flat/neutral case).

Independent monthly entries (condor_delta_backtest.py's universe, not the
continuous roll chain -- a fixed-at-entry construction choice creates no
path-dependence between consecutive months).

OOS discipline (same fixed split used throughout this investigation):
  DISCOVERY  2015-02-09 -> 2023-12-31
  HOLDOUT    2024-01-01 -> 2026-07-08, touched once at the end.

RUN IN A NORMAL PYTHON ENV. No network calls.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import iron_condor_backtest as IC   # noqa: E402
import condor_delta_backtest as CD  # noqa: E402
import index_trend_signal as TS     # noqa: E402

OUTDIR = IC.OUTDIR
SPLIT_DATE = pd.Timestamp("2024-01-01")
PRIMARY = dict(target_delta=0.20, p=3)
SECOND = dict(target_delta=0.25, p=3)
SKEW_GRID = [0.00, 0.03, 0.05, 0.08, 0.10]
MOM_LOOKBACK = "3m"      # fixed by convention -- see docstring
MIN_DELTA = 0.02         # floor so a large skew can't push the threatened side to a degenerate delta

C_SURFACE, C_INK, C_MUTED = "#fcfcfb", "#0b0b0b", "#898781"
C_SERIES = ["#3b6ea5", "#c1502e", "#5a8f5a", "#8a7ab5"]


def window(chain: pd.DataFrame, spot_series: pd.Series, start, end):
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


# ============================================================== skewed condor
def build_skewed_condor(sub: pd.DataFrame, delta_put: float, delta_call: float, p_steps: int):
    """Same as CD.build_condor_by_delta but with independently chosen short
    put/call target deltas -- delta_put == delta_call reproduces it exactly."""
    strikes_sorted = np.sort(sub[CD.C_STRIKE].unique())
    if len(strikes_sorted) < 5:
        return None
    short_p = CD.pick_by_delta(sub, "P", delta_put)
    short_c = CD.pick_by_delta(sub, "C", delta_call)
    if short_p is None or short_c is None:
        return None
    Kp_s, Kc_s = float(short_p[CD.C_STRIKE]), float(short_c[CD.C_STRIKE])
    if Kp_s >= Kc_s:
        return None
    ip_s = int(np.searchsorted(strikes_sorted, Kp_s))
    ic_s = int(np.searchsorted(strikes_sorted, Kc_s))
    ip_l, ic_l = ip_s - p_steps, ic_s + p_steps
    if ip_l < 0 or ic_l >= len(strikes_sorted):
        return None
    Kp_l, Kc_l = float(strikes_sorted[ip_l]), float(strikes_sorted[ic_l])

    def close_of(K, cp):
        r = sub[(sub[CD.C_STRIKE] == K) & (sub[CD.C_CP] == cp)]
        return float(r[CD.C_CLOSE].iloc[0]) if len(r) and pd.notna(r[CD.C_CLOSE].iloc[0]) else np.nan

    p_s_px, p_l_px = close_of(Kp_s, "P"), close_of(Kp_l, "P")
    c_s_px, c_l_px = close_of(Kc_s, "C"), close_of(Kc_l, "C")
    if any(np.isnan(x) for x in (p_s_px, p_l_px, c_s_px, c_l_px)):
        return None
    credit = (p_s_px - p_l_px) + (c_s_px - c_l_px) - 4 * CD.SLIP_PER_LEG
    wing = max(Kp_s - Kp_l, Kc_l - Kc_s)
    max_loss = wing - credit
    if credit <= 0 or max_loss <= 1e-6:
        return None
    return dict(strikes=(Kp_l, Kp_s, Kc_s, Kc_l), credit=credit, max_loss=max_loss)


# ============================================================== trades
def build_trades(chain: pd.DataFrame, spot_series: pd.Series, trend_sig: pd.DataFrame,
                  target_delta: float, p: int, skew: float) -> list[dict]:
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

        sig_row = trend_sig.reindex([d], method="ffill").iloc[0]
        mom = sig_row[f"mom_{MOM_LOOKBACK}"]
        bullish = bool(pd.notna(mom) and mom > 0)   # deadband=0 -- always picks a side (fixed by convention)

        delta_put = target_delta if bullish else max(target_delta - skew, MIN_DELTA)
        delta_call = max(target_delta - skew, MIN_DELTA) if bullish else target_delta

        built = build_skewed_condor(sub, delta_put, delta_call, p)
        if built is None:
            continue
        S_exp = float(spot_series.loc[exp_date])
        pnl = CD.realized_pnl(built["strikes"], built["credit"], S_exp)
        trades.append(dict(entry=d, exit=exp_date, r=pnl / built["max_loss"],
                            bullish=bullish, delta_put=delta_put, delta_call=delta_call,
                            credit=built["credit"]))
    return trades


def rows_of(trades: list[dict]) -> list[tuple]:
    return [(t["entry"], t["r"]) for t in trades]


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
def sweep(chain: pd.DataFrame, spot: pd.Series, trend_sig: pd.DataFrame, label: str,
          target_delta: float, p: int) -> pd.DataFrame:
    rows_out = []
    for skew in SKEW_GRID:
        trades = build_trades(chain, spot, trend_sig, target_delta, p, skew)
        rows = rows_of(trades)
        stats = CD.cell_stats(rows)
        if stats is None:
            continue
        rstats = raw_stats(rows)
        avg_credit = float(np.mean([t["credit"] for t in trades]))
        rows_out.append(dict(skew=skew, **stats, **rstats, avg_credit=avg_credit))
        print(f"  [{label}] skew={skew:.2f}  n={stats['n']:>3}  avg_credit={avg_credit:.4f}  "
              f"mean_r={rstats['mean_r']:+.4f}  f={stats['f']:.2f}  CAGR={stats['cagr']:+.2%}  "
              f"MDD={stats['mdd']:+.2%}  win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}")
    return pd.DataFrame(rows_out)


def main() -> None:
    chain_full = IC.load_local_chain()
    spot_full = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()
    disc_end = SPLIT_DATE - pd.Timedelta(days=1)

    trend_sig = TS.trend_signal_frame(TS.load_index())
    print(f"Option chain: {data_start.date()} -> {data_end.date()}")
    print(f"Split date: {SPLIT_DATE.date()}  Discovery: {data_start.date()}->{disc_end.date()}  "
          f"Holdout: {SPLIT_DATE.date()}->{data_end.date()}")
    print(f"Config: target_delta={PRIMARY['target_delta']} p={PRIMARY['p']}, "
          f"momentum lookback={MOM_LOOKBACK} (fixed by convention)\n")

    # sanity check: skew=0.00 must reproduce condor_delta_backtest.py's own numbers exactly
    zero_skew_trades = build_trades(chain_full, spot_full, trend_sig, *PRIMARY.values(), 0.00)
    zero_skew_rows = rows_of(zero_skew_trades)
    orig_grid = CD.run_grid(chain_full, spot_full)
    orig_stats = CD.cell_stats(orig_grid.get((PRIMARY["target_delta"], PRIMARY["p"]), []))
    zero_stats = CD.cell_stats(zero_skew_rows)
    if orig_stats is not None and zero_stats is not None:
        print(f"Sanity check (skew=0.00 vs condor_delta_backtest.py directly): "
              f"CAGR {zero_stats['cagr']:+.2%} vs {orig_stats['cagr']:+.2%}, "
              f"n {zero_stats['n']} vs {orig_stats['n']}\n")

    disc_chain, disc_spot = window(chain_full, spot_full, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_full, SPLIT_DATE, data_end)

    print(f"--- DISCOVERY sweep ({data_start.date()} -> {disc_end.date()}) ---")
    disc_grid = sweep(disc_chain, disc_spot, trend_sig, "discovery", **PRIMARY)
    disc_grid.to_csv(OUTDIR / "condor_asymmetric_skew_oos_discovery.csv", index=False)

    if disc_grid.empty:
        print("\nNo discovery cell had enough trades to score -- cannot proceed.")
        return
    baseline_row = disc_grid[disc_grid["skew"] == 0.00].iloc[0]
    candidates = disc_grid[disc_grid["skew"] > 0.00]
    if candidates.empty:
        print("\nNo non-trivial skew cell scored -- cannot proceed.")
        return
    frozen = candidates.sort_values("mean_r", ascending=False).iloc[0]
    frozen_skew = float(frozen["skew"])
    print(f"\nFROZEN pick from discovery (best mean_r, skew>0): skew={frozen_skew:.2f} "
          f"(discovery mean_r={frozen['mean_r']:+.4f} vs skew=0 baseline {baseline_row['mean_r']:+.4f}, "
          f"CAGR={frozen['cagr']:+.2%}, Calmar={frozen['calmar']:.2f})")

    print(f"\n--- HOLDOUT sweep ({SPLIT_DATE.date()} -> {data_end.date()}, context only) ---")
    hold_grid = sweep(hold_chain, hold_spot, trend_sig, "holdout", **PRIMARY)
    hold_grid.to_csv(OUTDIR / "condor_asymmetric_skew_oos_holdout.csv", index=False)

    base_hold_row = hold_grid[hold_grid["skew"] == 0.00].iloc[0] if not hold_grid.empty else None
    print(f"\n--- OOS EVALUATION: frozen skew={frozen_skew:.2f} applied to holdout-only data ---")
    frozen_trades = build_trades(hold_chain, hold_spot, trend_sig, *PRIMARY.values(), frozen_skew)
    frozen_rows = rows_of(frozen_trades)
    frozen_oos, frozen_oos_raw = CD.cell_stats(frozen_rows), raw_stats(frozen_rows)
    if frozen_oos is None:
        print(f"  frozen skew on holdout: too few trades to score (n={len(frozen_rows)})")
    else:
        print(f"  frozen skew on holdout:      n={frozen_oos['n']:>3}  mean_r={frozen_oos_raw['mean_r']:+.4f}  "
              f"f={frozen_oos['f']:.2f}  CAGR={frozen_oos['cagr']:+.2%}  MDD={frozen_oos['mdd']:+.2%}  "
              f"Calmar={frozen_oos['calmar']:.2f}")
    if base_hold_row is not None:
        print(f"  holdout baseline (skew=0.00): n={int(base_hold_row['n']):>3}  "
              f"mean_r={base_hold_row['mean_r']:+.4f}  f={base_hold_row['f']:.2f}  "
              f"CAGR={base_hold_row['cagr']:+.2%}  MDD={base_hold_row['mdd']:+.2%}  "
              f"Calmar={base_hold_row['calmar']:.2f}")
    if not hold_grid.empty:
        hb = hold_grid[hold_grid["skew"] > 0.00].sort_values("mean_r", ascending=False).iloc[0]
        print(f"  holdout's OWN best (skew={hb['skew']:.2f}, in-sample on holdout, context only): "
              f"mean_r={hb['mean_r']:+.4f} CAGR={hb['cagr']:+.2%} Calmar={hb['calmar']:.2f}")

    print("\n--- Verdict ---")
    if frozen_oos is not None and base_hold_row is not None:
        beats = frozen_oos_raw["mean_r"] > base_hold_row["mean_r"]
        print(f"  Frozen discovery pick on holdout: "
              f"{'beats' if beats else 'does not beat'} the unskewed baseline "
              f"(mean_r {frozen_oos_raw['mean_r']:+.4f} vs {base_hold_row['mean_r']:+.4f}; "
              f"CAGR {frozen_oos['cagr']:+.2%} vs {base_hold_row['cagr']:+.2%}).")

    # ---- generalization check on a second entry config ----
    base2_trades = build_trades(hold_chain, hold_spot, trend_sig, *SECOND.values(), 0.00)
    frozen2_trades = build_trades(hold_chain, hold_spot, trend_sig, *SECOND.values(), frozen_skew)
    base2_rows, frozen2_rows = rows_of(base2_trades), rows_of(frozen2_trades)
    base2, frozen2 = CD.cell_stats(base2_rows), CD.cell_stats(frozen2_rows)
    base2_raw, frozen2_raw = raw_stats(base2_rows), raw_stats(frozen2_rows)
    print(f"\nGeneralization check on target_delta={SECOND['target_delta']} p={SECOND['p']}, holdout window:")
    if base2:
        print(f"  baseline (skew=0)   mean_r={base2_raw['mean_r']:+.4f} CAGR={base2['cagr']:+.2%} "
              f"MDD={base2['mdd']:+.2%} Calmar={base2['calmar']:.2f}")
    if frozen2:
        print(f"  frozen skew         mean_r={frozen2_raw['mean_r']:+.4f} CAGR={frozen2['cagr']:+.2%} "
              f"MDD={frozen2['mdd']:+.2%} Calmar={frozen2['calmar']:.2f}")

    # ---- mechanistic check: worst 5 trades in the unskewed (baseline) universe ----
    all_base = build_trades(chain_full, spot_full, trend_sig, *PRIMARY.values(), 0.00)
    all_frozen = build_trades(chain_full, spot_full, trend_sig, *PRIMARY.values(), frozen_skew)
    frozen_by_entry = {t["entry"]: t for t in all_frozen}
    worst5 = sorted(all_base, key=lambda t: t["r"])[:5]
    print(f"\n--- Mechanistic check: worst 5 (unskewed) trades vs frozen skew={frozen_skew:.2f} ---")
    for b in worst5:
        f = frozen_by_entry.get(b["entry"])
        side = "bullish(call skewed out)" if b["bullish"] else "bearish(put skewed out)"
        line = f"  {b['entry'].date()}->{b['exit'].date()}  {side}  unskewed r={b['r']:+.3f}"
        if f is not None:
            line += f"  |  skewed r={f['r']:+.3f}"
        print(line)

    # ---- equity plot ----
    base_all_rows = rows_of(all_base)
    frozen_all_rows = rows_of(all_frozen)
    base_all, frozen_all = CD.cell_stats(base_all_rows), CD.cell_stats(frozen_all_rows)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    if base_all:
        curve = equity_curve(base_all_rows, base_all["f"])
        ax.plot(curve.index, curve.values, label="symmetric condor (baseline)", color=C_SERIES[0], linewidth=1.6)
    if frozen_all:
        curve = equity_curve(frozen_all_rows, frozen_all["f"])
        ax.plot(curve.index, curve.values, label=f"trend-skewed condor (skew={frozen_skew:.2f})",
                 color=C_SERIES[1], linewidth=1.6)
    ax.axvline(SPLIT_DATE, color=C_MUTED, linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_title(f"Trend-skewed vs symmetric condor, full 2015-2026 history "
                 f"(target_delta={PRIMARY['target_delta']} p={PRIMARY['p']})",
                 color=C_INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=C_MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "condor_asymmetric_skew_equity.png", dpi=150, facecolor=C_SURFACE)
    plt.close(fig)

    print(f"\nSaved: {OUTDIR/'condor_asymmetric_skew_oos_discovery.csv'}, "
          f"{OUTDIR/'condor_asymmetric_skew_oos_holdout.csv'}, and condor_asymmetric_skew_equity.png")


if __name__ == "__main__":
    main()
