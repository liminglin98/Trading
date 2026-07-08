# -*- coding: utf-8 -*-
"""
condor_trend_sleeve_backtest.py
==================================
Tests pairing the condor with a genuinely DIFFERENT return stream, instead
of another rule layered onto the condor itself. Every mechanism tried so
far in this investigation -- exit rules, rolling, the trend-based mid-trade
exit (condor_trend_exit_backtest.py), the tail hedge (condor_tail_hedge_
backtest.py) -- reacts to or modifies the SAME position, and all of them
either hurt outright or failed OOS. A trend-following SLEEVE is different
in kind: a separate position on the SSE Composite (000001) that is expected
to profit exactly in the persistent-grind regimes that blow through the
condor's wings (4 of the 5 known disasters were violent rallies -- a
trend sleeh long into that rally profits from it directly, rather than
just avoiding/cutting the condor's loss).

Design (time-series momentum, same style as CTA/managed-futures overlays):
  REBALANCE   monthly, same dates as the condor's own entries (independent
              monthly universe -- condor_delta_backtest.py's convention,
              not the continuous roll chain, since there's no path-
              dependence to a separate parallel sleeve).
  SIGNAL      sign of trailing momentum on the SSE Composite as of the
              condor's entry date (causal -- no lookahead). Long if
              positive, short if negative, flat if exactly zero (deadband
              fixed at 0%, i.e. plain sign -- see PARAMETERS below).
  HOLDING     exactly the condor's own entry->exit window for that trade
              (so both sleeves' returns line up on identical dates for
              blending) -- unlevered 1x notional exposure to the index's
              raw price return over that window, minus a flat round-trip
              transaction cost.
  BLEND       portfolio_r = w * condor_r + (1-w) * trend_r, blended BEFORE
              Kelly-sizing (so CD.cell_stats sizes the COMBINED stream
              once, not each leg separately) -- w=1.0 reproduces the
              unhedged condor baseline exactly.

PARAMETERS -- what's actually being optimized here, and what's fixed by
convention (be honest about this since it's a bigger search than most
single-axis sweeps in this investigation):
  SWEPT (grid, OOS-disciplined below):
    - momentum lookback: {1m, 3m, 6m, 12m}
    - allocation weight w (condor share of blended risk): {1.0 (baseline,
      no trend sleeve), 0.9, 0.8, 0.7, 0.6, 0.5}
  FIXED BY CONVENTION (not tuned -- kept out of the grid on purpose to
  limit the multiple-comparisons problem a 2-parameter x holdout-freeze
  search already has):
    - deadband = 0% (plain sign of momentum, no dead zone)
    - sleeve leverage = 1x notional (no vol-targeting/scaling)
    - transaction cost = TREND_TC flat round-trip charge on the sleeve

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

LOOKBACK_GRID = list(TS.MOM_WINDOWS.items())          # [("1m",21), ("3m",63), ("6m",126), ("12m",252)]
WEIGHT_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]           # condor's share of blended risk; 1.0 = baseline
TREND_TC = 0.0010                                      # flat round-trip cost on the sleeve, fixed by convention

C_SURFACE, C_INK, C_MUTED = "#fcfcfb", "#0b0b0b", "#898781"
C_SERIES = ["#3b6ea5", "#c1502e", "#5a8f5a", "#8a7ab5"]


def window(chain: pd.DataFrame, spot_series: pd.Series, start, end):
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


# ============================================================== trades
def build_trades(chain: pd.DataFrame, spot_series: pd.Series, idx_close: pd.Series,
                  trend_sig: pd.DataFrame, target_delta: float, p: int,
                  lookback_label: str | None) -> list[dict]:
    """One row per condor trade: condor_r (unchanged), trend_r (sign of
    trailing momentum at entry x index's own return over the SAME
    entry->exit window, minus TREND_TC), or trend_r=0 if lookback_label is
    None (pure condor, no sleeve)."""
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
        condor_r = condor_pnl / built["max_loss"]

        trend_r, side, mom_val = 0.0, 0, np.nan
        if lookback_label is not None:
            sig_row = trend_sig.reindex([d], method="ffill").iloc[0]
            mom_val = sig_row[f"mom_{lookback_label}"]
            side = int(np.sign(mom_val)) if pd.notna(mom_val) and mom_val != 0 else 0
            if side != 0 and d in idx_close.index and exp_date in idx_close.index:
                idx_entry, idx_exit = float(idx_close.loc[d]), float(idx_close.loc[exp_date])
                trend_r = side * (idx_exit / idx_entry - 1) - TREND_TC

        trades.append(dict(entry=d, exit=exp_date, condor_r=condor_r, trend_r=trend_r,
                            side=side, mom=mom_val))
    return trades


def blend_rows(trades: list[dict], w: float) -> list[tuple]:
    return [(t["entry"], w * t["condor_r"] + (1 - w) * t["trend_r"]) for t in trades]


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
def sweep(chain: pd.DataFrame, spot: pd.Series, idx_close: pd.Series, trend_sig: pd.DataFrame,
          label: str, target_delta: float, p: int) -> pd.DataFrame:
    rows_out = []
    trades_cache = {lb: build_trades(chain, spot, idx_close, trend_sig, target_delta, p, lb)
                     for lb, _ in LOOKBACK_GRID}
    for lb, _days in LOOKBACK_GRID:
        trades = trades_cache[lb]
        for w in WEIGHT_GRID:
            rows = blend_rows(trades, w)
            stats = CD.cell_stats(rows)
            if stats is None:
                continue
            rstats = raw_stats(rows)
            long_frac = float(np.mean([t["side"] == 1 for t in trades]))
            short_frac = float(np.mean([t["side"] == -1 for t in trades]))
            rows_out.append(dict(lookback=lb, weight=w, **stats, **rstats,
                                  long_frac=long_frac, short_frac=short_frac))
            print(f"  [{label}] lookback={lb:>3} w={w:.1f}  n={stats['n']:>3}  "
                  f"long/short={long_frac:.0%}/{short_frac:.0%}  mean_r={rstats['mean_r']:+.4f}  "
                  f"f={stats['f']:.2f}  CAGR={stats['cagr']:+.2%}  MDD={stats['mdd']:+.2%}  "
                  f"win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}")
    return pd.DataFrame(rows_out)


def main() -> None:
    chain_full = IC.load_local_chain()
    spot_full = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()
    disc_end = SPLIT_DATE - pd.Timedelta(days=1)

    idx_close = TS.load_index()
    trend_sig = TS.trend_signal_frame(idx_close)
    print(f"SSE Composite: {idx_close.index.min().date()} -> {idx_close.index.max().date()}")
    print(f"Option chain: {data_start.date()} -> {data_end.date()}")
    print(f"Split date: {SPLIT_DATE.date()}  Discovery: {data_start.date()}->{disc_end.date()}  "
          f"Holdout: {SPLIT_DATE.date()}->{data_end.date()}")
    print(f"Config: delta={PRIMARY['target_delta']} p={PRIMARY['p']} (independent monthly entries)")
    print(f"Fixed by convention (not tuned): deadband=0%, sleeve leverage=1x, "
          f"transaction cost={TREND_TC:.2%} round-trip\n")

    disc_chain, disc_spot = window(chain_full, spot_full, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_full, SPLIT_DATE, data_end)

    base_disc_trades = build_trades(disc_chain, disc_spot, idx_close, trend_sig, *PRIMARY.values(), None)
    base_disc_rows = blend_rows(base_disc_trades, 1.0)
    base_disc, base_disc_raw = CD.cell_stats(base_disc_rows), raw_stats(base_disc_rows)
    print(f"discovery baseline (no trend sleeve, w=1.0): n={base_disc['n']}  "
          f"mean_r={base_disc_raw['mean_r']:+.4f}  CAGR={base_disc['cagr']:+.2%}  "
          f"MDD={base_disc['mdd']:+.2%}  Calmar={base_disc['calmar']:.2f}")

    print(f"\n--- DISCOVERY sweep ({data_start.date()} -> {disc_end.date()}) ---")
    disc_grid = sweep(disc_chain, disc_spot, idx_close, trend_sig, "discovery", **PRIMARY)
    disc_grid.to_csv(OUTDIR / "condor_trend_sleeve_oos_discovery.csv", index=False)

    if disc_grid.empty:
        print("\nNo discovery cell had enough trades to score -- cannot proceed.")
        return
    # exclude w=1.0 (that's just the baseline re-appearing in the grid) when picking the frozen NON-trivial pick
    candidates = disc_grid[disc_grid["weight"] < 1.0]
    frozen = candidates.sort_values("mean_r", ascending=False).iloc[0]
    frozen_lb, frozen_w = str(frozen["lookback"]), float(frozen["weight"])
    print(f"\nFROZEN pick from discovery (best mean_r, w<1.0): lookback={frozen_lb} weight={frozen_w:.1f} "
          f"(discovery mean_r={frozen['mean_r']:+.4f} vs baseline {base_disc_raw['mean_r']:+.4f}, "
          f"CAGR={frozen['cagr']:+.2%}, Calmar={frozen['calmar']:.2f})")

    print(f"\n--- HOLDOUT sweep ({SPLIT_DATE.date()} -> {data_end.date()}, context only) ---")
    hold_grid = sweep(hold_chain, hold_spot, idx_close, trend_sig, "holdout", **PRIMARY)
    hold_grid.to_csv(OUTDIR / "condor_trend_sleeve_oos_holdout.csv", index=False)

    base_hold_trades = build_trades(hold_chain, hold_spot, idx_close, trend_sig, *PRIMARY.values(), None)
    base_hold_rows = blend_rows(base_hold_trades, 1.0)
    base_hold, base_hold_raw = CD.cell_stats(base_hold_rows), raw_stats(base_hold_rows)

    print(f"\n--- OOS EVALUATION: frozen (lookback={frozen_lb}, w={frozen_w:.1f}) applied to holdout ---")
    frozen_trades = build_trades(hold_chain, hold_spot, idx_close, trend_sig, *PRIMARY.values(), frozen_lb)
    frozen_rows = blend_rows(frozen_trades, frozen_w)
    frozen_oos, frozen_oos_raw = CD.cell_stats(frozen_rows), raw_stats(frozen_rows)
    if frozen_oos is None:
        print(f"  frozen sleeve on holdout: too few trades to score (n={len(frozen_rows)})")
    else:
        print(f"  frozen sleeve on holdout:  n={frozen_oos['n']:>3}  mean_r={frozen_oos_raw['mean_r']:+.4f}  "
              f"f={frozen_oos['f']:.2f}  CAGR={frozen_oos['cagr']:+.2%}  MDD={frozen_oos['mdd']:+.2%}  "
              f"Calmar={frozen_oos['calmar']:.2f}")
    print(f"  holdout baseline (no sleeve): n={base_hold['n']:>3}  mean_r={base_hold_raw['mean_r']:+.4f}  "
          f"f={base_hold['f']:.2f}  CAGR={base_hold['cagr']:+.2%}  MDD={base_hold['mdd']:+.2%}  "
          f"Calmar={base_hold['calmar']:.2f}")
    if not hold_grid.empty:
        hb = hold_grid[hold_grid["weight"] < 1.0].sort_values("mean_r", ascending=False).iloc[0]
        print(f"  holdout's OWN best (lookback={hb['lookback']}, w={hb['weight']:.1f}, in-sample on "
              f"holdout, context only): mean_r={hb['mean_r']:+.4f} CAGR={hb['cagr']:+.2%} Calmar={hb['calmar']:.2f}")

    print("\n--- Verdict ---")
    if frozen_oos is not None:
        beats = frozen_oos_raw["mean_r"] > base_hold_raw["mean_r"]
        print(f"  Frozen discovery pick on holdout: "
              f"{'beats' if beats else 'does not beat'} the no-sleeve baseline "
              f"(mean_r {frozen_oos_raw['mean_r']:+.4f} vs {base_hold_raw['mean_r']:+.4f}; "
              f"CAGR {frozen_oos['cagr']:+.2%} vs {base_hold['cagr']:+.2%}).")

    # ---- generalization check on a second entry config ----
    base2_trades = build_trades(hold_chain, hold_spot, idx_close, trend_sig, *SECOND.values(), None)
    frozen2_trades = build_trades(hold_chain, hold_spot, idx_close, trend_sig, *SECOND.values(), frozen_lb)
    base2_rows, frozen2_rows = blend_rows(base2_trades, 1.0), blend_rows(frozen2_trades, frozen_w)
    base2, frozen2 = CD.cell_stats(base2_rows), CD.cell_stats(frozen2_rows)
    base2_raw, frozen2_raw = raw_stats(base2_rows), raw_stats(frozen2_rows)
    print(f"\nGeneralization check on delta={SECOND['target_delta']} p={SECOND['p']}, holdout window:")
    if base2:
        print(f"  baseline (no sleeve)  mean_r={base2_raw['mean_r']:+.4f} CAGR={base2['cagr']:+.2%} "
              f"MDD={base2['mdd']:+.2%} Calmar={base2['calmar']:.2f}")
    if frozen2:
        print(f"  frozen sleeve         mean_r={frozen2_raw['mean_r']:+.4f} CAGR={frozen2['cagr']:+.2%} "
              f"MDD={frozen2['mdd']:+.2%} Calmar={frozen2['calmar']:.2f}")

    # ---- mechanistic check: worst 5 trades in THIS (monthly) universe ----
    all_base = build_trades(chain_full, spot_full, idx_close, trend_sig, *PRIMARY.values(), None)
    all_frozen = build_trades(chain_full, spot_full, idx_close, trend_sig, *PRIMARY.values(), frozen_lb)
    frozen_by_entry = {t["entry"]: t for t in all_frozen}
    worst5 = sorted(all_base, key=lambda t: t["condor_r"])[:5]
    print(f"\n--- Mechanistic check: worst 5 condor trades vs frozen sleeve "
          f"(lookback={frozen_lb}, w={frozen_w:.1f}) ---")
    for b in worst5:
        f = frozen_by_entry.get(b["entry"])
        line = f"  {b['entry'].date()}->{b['exit'].date()}  condor r={b['condor_r']:+.3f}"
        if f is not None:
            blended = frozen_w * f["condor_r"] + (1 - frozen_w) * f["trend_r"]
            side_str = {1: "long", -1: "short", 0: "flat"}[f["side"]]
            line += (f"  |  sleeve was {side_str} (trend_r={f['trend_r']:+.3f})  "
                     f"blended r={blended:+.3f}")
        print(line)

    # ---- equity plot ----
    base_all_rows = blend_rows(all_base, 1.0)
    frozen_all_rows = blend_rows(all_frozen, frozen_w)
    base_all, frozen_all = CD.cell_stats(base_all_rows), CD.cell_stats(frozen_all_rows)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    if base_all:
        curve = equity_curve(base_all_rows, base_all["f"])
        ax.plot(curve.index, curve.values, label="condor only (baseline)", color=C_SERIES[0], linewidth=1.6)
    if frozen_all:
        curve = equity_curve(frozen_all_rows, frozen_all["f"])
        ax.plot(curve.index, curve.values,
                 label=f"condor + trend sleeve (lookback={frozen_lb}, w={frozen_w:.1f})",
                 color=C_SERIES[1], linewidth=1.6)
    ax.axvline(SPLIT_DATE, color=C_MUTED, linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_title(f"Condor + trend sleeve vs condor-only, full 2015-2026 history "
                 f"(delta={PRIMARY['target_delta']} p={PRIMARY['p']})",
                 color=C_INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=C_MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "condor_trend_sleeve_equity.png", dpi=150, facecolor=C_SURFACE)
    plt.close(fig)

    print(f"\nSaved: {OUTDIR/'condor_trend_sleeve_oos_discovery.csv'}, "
          f"{OUTDIR/'condor_trend_sleeve_oos_holdout.csv'}, and condor_trend_sleeve_equity.png")


if __name__ == "__main__":
    main()
