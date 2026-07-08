# -*- coding: utf-8 -*-
"""
condor_trend_exit_backtest.py
================================
The one untested idea flagged at the end of condor_delta_investigation_
summary.md: a trend/momentum filter aimed at the actual failure mode found
in condor_delta_regime_diagnostic.py (persistent directional grinds, not
vol spikes -- one disaster, Jan 2018, had realized vol BELOW what was
priced, so a vol screen alone wouldn't have caught it).

A quick pre-check (trailing efficiency-ratio / momentum AS OF each entry
date, on the SSE Composite) found NO reliable ability to predict which
trades go bad -- the known disasters don't stand out at entry time. So this
is built as a REACTIVE MID-TRADE monitor instead of a predictive entry
filter: hold the same blind delta-targeted condor, but watch the SSE
Composite's trend-persistence (Kaufman efficiency ratio, index_trend_
signal.py) every day of the trade's life, and exit early the first day the
index is grinding harder than a threshold -- i.e. react to a grind once it's
actually underway, rather than trying to guess it's coming.

Uses the CONTINUOUS-CHAIN methodology (condor_delta_roll_backtest.py /
condor_delta_dte_sweep.py), not condor_delta_exit_backtest.py's independent
monthly entries -- this is the same universe condor_delta_regime_diagnostic.
py used to name its 5 disaster trades, so a trigger built to react to those
specific failures needs to be tested against the exact same trades, not a
different (overlapping, monthly) trade universe. (An earlier draft of this
script used the monthly-entries universe and could not even locate 4 of the
5 named disasters in its own trade list -- different entry cadence produces
different actual expiry windows. Switched to the continuous chain to fix
that.)

This is mechanically the same "close early on a trigger" architecture as
condor_delta_exit_backtest.py (PnL-based triggers, all of which hurt
outright) and condor_delta_roll_backtest.py's stop-loss (looked good
in-sample, failed OOS twice). The trigger here is different in kind -- price
STRUCTURE on the broad index, not the position's own PnL -- so it gets its
own genuine discovery/holdout test rather than being assumed guilty by
association.

OOS discipline (same fixed split used throughout this investigation):
  DISCOVERY  2015-02-09 -> 2023-12-31
  HOLDOUT    2024-01-01 -> 2026-07-08, touched once at the end.
Grid: ER window in {10,20,40,60} x ER threshold in {0.40,0.50,0.60,0.70}
(2 free parameters -- a bigger search than this investigation's earlier
single-axis sweeps, so the OOS split matters even more here).

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

import iron_condor_backtest as IC        # noqa: E402
import condor_delta_backtest as CD       # noqa: E402
import condor_delta_exit_backtest as CX  # noqa: E402  build_life
import index_trend_signal as TS          # noqa: E402

OUTDIR = IC.OUTDIR
SPLIT_DATE = pd.Timestamp("2024-01-01")
PRIMARY = dict(target_delta=0.20, p=3)
SECOND = dict(target_delta=0.25, p=3)
NEVER = 1.01   # ER in [0,1] so a threshold > 1 never trips -- the "no rule" sentinel

ER_WINDOW_GRID = [10, 20, 40, 60]
ER_THRESH_GRID = [0.40, 0.50, 0.60, 0.70]
MAX_LEGS = 3000

C_SURFACE, C_INK, C_MUTED = "#fcfcfb", "#0b0b0b", "#898781"
C_SERIES = ["#3b6ea5", "#c1502e", "#5a8f5a", "#8a7ab5"]


def window(chain: pd.DataFrame, spot_series: pd.Series, start, end):
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


# ============================================================== trend exit rule
def find_trend_exit(life: pd.DataFrame, trend_sig: pd.DataFrame, er_window: int,
                     er_threshold: float):
    """First day of the trade's life the SSE Composite's trailing ER_{er_window}
    exceeds er_threshold (checked causally -- trend_sig at day t only uses
    trailing data). Returns (exit_date, cost_to_close) or (None, None)."""
    col = f"ER_{er_window}"
    for day, row in life.iterrows():
        have_value = pd.notna(row[["pl_close", "ps_close", "cs_close", "cl_close"]]).all()
        if not have_value:
            continue
        sig_row = trend_sig.reindex([day], method="ffill").iloc[0]
        er = sig_row[col]
        if pd.notna(er) and er >= er_threshold:
            value = (row["ps_close"] - row["pl_close"]) + (row["cs_close"] - row["cl_close"])
            return day, value
    return None, None


# ============================================================== continuous chain walk
def walk_rolling_chain_trend(chain: pd.DataFrame, spot_series: pd.Series,
                              target_delta: float, p: int, trend_sig: pd.DataFrame,
                              er_window: int, er_threshold: float,
                              target_dte: int = CD.TARGET_DTE) -> list[dict]:
    """Same continuous-chain walk as condor_delta_roll_backtest.walk_rolling_
    chain (whenever a leg ends -- trigger or its own expiry -- the next leg
    opens the same day), with find_trend_exit as the trigger instead of
    condor_delta_exit_backtest.find_exit's PnL-based rules."""
    trading_days = np.sort(chain[CD.C_DATE].unique())
    last_daily = spot_series.index.max()
    cur_date = pd.Timestamp(trading_days[0])
    trades = []

    def next_day(d):
        later = trading_days[trading_days > np.datetime64(d)]
        return pd.Timestamp(later[0]) if len(later) else None

    for _ in range(MAX_LEGS):
        day = chain[chain[CD.C_DATE] == cur_date]
        if day.empty:
            nd = next_day(cur_date)
            if nd is None:
                break
            cur_date = nd
            continue
        exp_date = CD.choose_expiry_for(day, target_dte)
        if exp_date is None:
            nd = next_day(cur_date)
            if nd is None:
                break
            cur_date = nd
            continue
        exp_date = pd.Timestamp(exp_date)
        sub = day[day[CD.C_EXP] == exp_date]
        built = CD.build_condor_by_delta(sub, target_delta, p)
        if built is None:
            nd = next_day(cur_date)
            if nd is None:
                break
            cur_date = nd
            continue
        life = CX.build_life(chain, exp_date, built["strikes"], cur_date)
        if life is None or life.empty:
            break
        trig_day, value = find_trend_exit(life, trend_sig, er_window, er_threshold)
        if trig_day is not None:
            pnl = built["credit"] - value - 4 * CD.SLIP_PER_LEG
            trades.append(dict(entry=cur_date, exit=trig_day, pnl=pnl,
                                max_loss=built["max_loss"], reason="trend"))
            cur_date = trig_day
            continue
        if exp_date > last_daily or exp_date not in spot_series.index:
            break
        pnl = CD.realized_pnl(built["strikes"], built["credit"], float(spot_series.loc[exp_date]))
        trades.append(dict(entry=cur_date, exit=exp_date, pnl=pnl,
                            max_loss=built["max_loss"], reason="expiry"))
        cur_date = exp_date
    return trades


def rows_from_trades(trades: list[dict]) -> list[tuple]:
    return [(t["entry"], t["pnl"] / t["max_loss"]) for t in trades]


def equity_curve(rows: list, f_used: float) -> pd.Series:
    dates = [d for d, _ in rows]
    r = np.array([x for _, x in rows])
    eq = np.exp(np.log1p((f_used * r).clip(min=-0.999)).cumsum())
    return pd.Series(eq, index=dates)


def raw_stats(rows: list) -> dict:
    """Un-levered mean/std/win -- unlike CD.cell_stats() this stays informative
    even when Kelly rejects the whole cell (f=0 collapses CAGR/MDD/Calmar to
    0/0/nan, which happens throughout this discovery window since the base
    construction already has ~zero/negative discovery-period expectancy --
    see condor_delta_investigation_summary.md 3.3). Selection for the frozen
    pick below uses mean_r, not Calmar, for exactly this reason."""
    r = np.array([x for _, x in rows])
    return dict(mean_r=float(r.mean()), std_r=float(r.std(ddof=1)) if len(r) > 1 else np.nan,
                win_raw=float((r > 0).mean()))


# ============================================================== sweep
def sweep(chain: pd.DataFrame, spot: pd.Series, trend_sig: pd.DataFrame, label: str,
          target_delta: float, p: int) -> pd.DataFrame:
    rows_out = []
    for er_window in ER_WINDOW_GRID:
        for er_threshold in ER_THRESH_GRID:
            trades = walk_rolling_chain_trend(chain, spot, target_delta, p, trend_sig,
                                               er_window, er_threshold)
            rows = rows_from_trades(trades)
            stats = CD.cell_stats(rows)
            if stats is None:
                continue
            rstats = raw_stats(rows)
            n_exits = sum(1 for t in trades if t["reason"] == "trend")
            rows_out.append(dict(er_window=er_window, er_threshold=er_threshold,
                                  **stats, **rstats, n_trend_exits=n_exits))
            print(f"  [{label}] window={er_window:>2} thresh={er_threshold:.2f}  "
                  f"n={stats['n']:>3}  exits={n_exits:>3}  mean_r={rstats['mean_r']:+.4f}  "
                  f"f={stats['f']:.2f}  CAGR={stats['cagr']:+.2%}  MDD={stats['mdd']:+.2%}  "
                  f"win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}")
    return pd.DataFrame(rows_out)


def main() -> None:
    chain_full = IC.load_local_chain()
    spot_full = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()
    disc_end = SPLIT_DATE - pd.Timedelta(days=1)

    idx = TS.load_index()
    trend_sig = TS.trend_signal_frame(idx)
    print(f"SSE Composite trend signal: {idx.index.min().date()} -> {idx.index.max().date()}")
    print(f"Option chain: {data_start.date()} -> {data_end.date()}")
    print(f"Split date: {SPLIT_DATE.date()}  Discovery: {data_start.date()}->{disc_end.date()}  "
          f"Holdout: {SPLIT_DATE.date()}->{data_end.date()}")
    print(f"Config: delta={PRIMARY['target_delta']} p={PRIMARY['p']}, continuous chain "
          f"(same universe as condor_delta_regime_diagnostic.py)\n")

    disc_chain, disc_spot = window(chain_full, spot_full, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_full, SPLIT_DATE, data_end)

    print(f"--- DISCOVERY sweep ({data_start.date()} -> {disc_end.date()}) ---")
    disc_grid = sweep(disc_chain, disc_spot, trend_sig, "discovery", **PRIMARY)
    disc_grid.to_csv(OUTDIR / "condor_trend_exit_oos_discovery.csv", index=False)

    base_disc_trades = walk_rolling_chain_trend(disc_chain, disc_spot, *PRIMARY.values(),
                                                 trend_sig, 60, NEVER)
    base_disc_rows = rows_from_trades(base_disc_trades)
    base_disc = CD.cell_stats(base_disc_rows)
    base_disc_raw = raw_stats(base_disc_rows)
    print(f"\n  discovery baseline (hold to expiry, no trend exit): n={base_disc['n']}  "
          f"mean_r={base_disc_raw['mean_r']:+.4f}  CAGR={base_disc['cagr']:+.2%}  "
          f"MDD={base_disc['mdd']:+.2%}  Calmar={base_disc['calmar']:.2f}")

    if disc_grid.empty:
        print("\nNo discovery cell had enough trades to score -- cannot proceed.")
        return
    # select by mean_r, NOT Calmar -- see raw_stats() docstring: Calmar collapses
    # to 0/0/nan whenever Kelly rejects, which is most of this grid, since the
    # discovery-period baseline itself already has ~zero/negative edge.
    frozen = disc_grid.sort_values("mean_r", ascending=False).iloc[0]
    frozen_window, frozen_thresh = int(frozen["er_window"]), float(frozen["er_threshold"])
    print(f"\nFROZEN pick from discovery (best mean_r): ER_window={frozen_window} threshold={frozen_thresh:.2f} "
          f"(discovery mean_r={frozen['mean_r']:+.4f} vs baseline {base_disc_raw['mean_r']:+.4f}, "
          f"CAGR={frozen['cagr']:+.2%}, Calmar={frozen['calmar']:.2f}, "
          f"n_exits={int(frozen['n_trend_exits'])}/{int(frozen['n'])})")

    print(f"\n--- HOLDOUT sweep ({SPLIT_DATE.date()} -> {data_end.date()}, context only) ---")
    hold_grid = sweep(hold_chain, hold_spot, trend_sig, "holdout", **PRIMARY)
    hold_grid.to_csv(OUTDIR / "condor_trend_exit_oos_holdout.csv", index=False)

    print(f"\n--- OOS EVALUATION: frozen (window={frozen_window}, thresh={frozen_thresh:.2f}) "
          f"applied to holdout-only data ---")
    frozen_trades = walk_rolling_chain_trend(hold_chain, hold_spot, *PRIMARY.values(),
                                              trend_sig, frozen_window, frozen_thresh)
    frozen_rows = rows_from_trades(frozen_trades)
    frozen_oos = CD.cell_stats(frozen_rows)
    frozen_oos_raw = raw_stats(frozen_rows)
    n_frozen_exits = sum(1 for t in frozen_trades if t["reason"] == "trend")

    base_hold_trades = walk_rolling_chain_trend(hold_chain, hold_spot, *PRIMARY.values(),
                                                 trend_sig, 60, NEVER)
    base_hold_rows = rows_from_trades(base_hold_trades)
    base_hold = CD.cell_stats(base_hold_rows)
    base_hold_raw = raw_stats(base_hold_rows)

    if frozen_oos is None:
        print(f"  frozen rule on holdout: too few trades to score (n={len(frozen_rows)})")
    else:
        print(f"  frozen rule on holdout:      n={frozen_oos['n']:>3}  exits={n_frozen_exits:>3}  "
              f"mean_r={frozen_oos_raw['mean_r']:+.4f}  f={frozen_oos['f']:.2f}  CAGR={frozen_oos['cagr']:+.2%}  "
              f"MDD={frozen_oos['mdd']:+.2%}  Calmar={frozen_oos['calmar']:.2f}")
    if base_hold is not None:
        print(f"  holdout baseline (unchanged, hold to expiry): n={base_hold['n']:>3}  "
              f"mean_r={base_hold_raw['mean_r']:+.4f}  f={base_hold['f']:.2f}  CAGR={base_hold['cagr']:+.2%}  "
              f"MDD={base_hold['mdd']:+.2%}  Calmar={base_hold['calmar']:.2f}")
    if not hold_grid.empty:
        hb = hold_grid.sort_values("mean_r", ascending=False).iloc[0]
        print(f"  holdout's OWN best (window={int(hb['er_window'])}, thresh={hb['er_threshold']:.2f}, "
              f"in-sample on holdout, context only): mean_r={hb['mean_r']:+.4f} CAGR={hb['cagr']:+.2%}  "
              f"Calmar={hb['calmar']:.2f}")

    print("\n--- Verdict ---")
    if frozen_oos is not None and base_hold is not None:
        beats = frozen_oos_raw["mean_r"] > base_hold_raw["mean_r"]
        print(f"  Frozen discovery pick on holdout: "
              f"{'beats' if beats else 'does not beat'} the unchanged hold-to-expiry baseline "
              f"(mean_r {frozen_oos_raw['mean_r']:+.4f} vs {base_hold_raw['mean_r']:+.4f}; "
              f"CAGR {frozen_oos['cagr']:+.2%} vs {base_hold['cagr']:+.2%}).")
    else:
        print("  Could not score cleanly on holdout -- inconclusive.")

    # ---- generalization check on a second entry config ----
    base2_trades = walk_rolling_chain_trend(hold_chain, hold_spot, *SECOND.values(),
                                             trend_sig, 60, NEVER)
    frozen2_trades = walk_rolling_chain_trend(hold_chain, hold_spot, *SECOND.values(),
                                               trend_sig, frozen_window, frozen_thresh)
    base2_rows, frozen2_rows = rows_from_trades(base2_trades), rows_from_trades(frozen2_trades)
    base2, frozen2 = CD.cell_stats(base2_rows), CD.cell_stats(frozen2_rows)
    base2_raw, frozen2_raw = raw_stats(base2_rows), raw_stats(frozen2_rows)
    n2_exits = sum(1 for t in frozen2_trades if t["reason"] == "trend")
    print(f"\nGeneralization check on delta={SECOND['target_delta']} p={SECOND['p']}, holdout window:")
    if base2:
        print(f"  baseline            mean_r={base2_raw['mean_r']:+.4f} CAGR={base2['cagr']:+.2%} "
              f"MDD={base2['mdd']:+.2%} Calmar={base2['calmar']:.2f}")
    if frozen2:
        print(f"  frozen trend-exit    mean_r={frozen2_raw['mean_r']:+.4f} CAGR={frozen2['cagr']:+.2%} "
              f"MDD={frozen2['mdd']:+.2%} Calmar={frozen2['calmar']:.2f}  exits={n2_exits}")

    # ---- mechanistic check: does the frozen rule actually engage the 5 known
    # disaster trades named in condor_delta_regime_diagnostic.py? ----
    known_disasters = ["2022-12-28", "2018-01-02", "2019-02-12", "2015-08-11", "2020-12-23"]
    all_trades = walk_rolling_chain_trend(chain_full, spot_full, *PRIMARY.values(),
                                           trend_sig, 60, NEVER)  # baseline chain (own expiries)
    all_frozen_trades = walk_rolling_chain_trend(chain_full, spot_full, *PRIMARY.values(),
                                                  trend_sig, frozen_window, frozen_thresh)
    print(f"\n--- Mechanistic check: known disaster trades vs frozen rule "
          f"(window={frozen_window}, thresh={frozen_thresh:.2f}) ---")
    for ds in known_disasters:
        d = pd.Timestamp(ds)
        base_match = next((t for t in all_trades if t["entry"] <= d <= t["exit"]), None)
        frozen_match = next((t for t in all_frozen_trades if t["entry"] <= d <= t["exit"]), None)
        if base_match is None:
            print(f"  {ds}: no matching continuous-chain trade found")
            continue
        base_r = base_match["pnl"] / base_match["max_loss"]
        print(f"  {ds}: baseline trade {base_match['entry'].date()}->{base_match['exit'].date()} "
              f"r={base_r:+.3f}", end="")
        if frozen_match is not None:
            fr_r = frozen_match["pnl"] / frozen_match["max_loss"]
            print(f"  |  frozen-rule trade {frozen_match['entry'].date()}->"
                  f"{frozen_match['exit'].date()} ({frozen_match['reason']}) r={fr_r:+.3f}")
        else:
            print("  |  no overlapping frozen-rule trade found")

    # ---- equity plot: full-history baseline vs frozen rule ----
    base_all_rows = rows_from_trades(all_trades)
    frozen_all_rows = rows_from_trades(all_frozen_trades)
    base_all, frozen_all = CD.cell_stats(base_all_rows), CD.cell_stats(frozen_all_rows)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    if base_all:
        curve = equity_curve(base_all_rows, base_all["f"])
        ax.plot(curve.index, curve.values, label="hold to expiry (baseline)",
                 color=C_SERIES[0], linewidth=1.6)
    if frozen_all:
        curve = equity_curve(frozen_all_rows, frozen_all["f"])
        ax.plot(curve.index, curve.values,
                 label=f"trend exit (window={frozen_window}, thresh={frozen_thresh:.2f})",
                 color=C_SERIES[1], linewidth=1.6)
    ax.axvline(SPLIT_DATE, color=C_MUTED, linestyle=":", linewidth=1)
    ax.set_title("Mid-trade trend exit vs hold-to-expiry, full 2015-2026 history "
                  f"(delta={PRIMARY['target_delta']} p={PRIMARY['p']}, continuous chain)",
                  color=C_INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=C_MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "condor_trend_exit_equity.png", dpi=150, facecolor=C_SURFACE)
    plt.close(fig)

    print(f"\nSaved: {OUTDIR/'condor_trend_exit_oos_discovery.csv'}, "
          f"{OUTDIR/'condor_trend_exit_oos_holdout.csv'}, and condor_trend_exit_equity.png")


if __name__ == "__main__":
    main()
