# -*- coding: utf-8 -*-
"""
condor_delta_dte_sweep.py
============================
The rolling/exit investigation (condor_delta_exit_backtest.py,
condor_delta_roll_backtest.py, condor_delta_roll_oos_test.py) settled on:
hold every condor to expiry, no early exit or roll of any kind -- every
mechanical intervention tested (DTE exit, profit-target, stop-loss, delta-
breach, and stop-loss ROLLING) either hurt expectancy outright or failed a
genuine out-of-sample test.

That leaves the ENTRY side unexplored: condor_delta_backtest.py and
condor_delta_roll_backtest.py both hardcoded TARGET_DTE=30 (target ~1 month
to expiry at entry) without ever testing whether that duration is good.
This script sweeps entry duration directly, holding every other choice fixed
(target_delta=0.20, p=3, hold-to-expiry-always, same real-quote pricing).

OOS discipline (learned from condor_delta_roll_oos_test.py -- a full-sample
grid search overstates edge; a mostly-pre-COVID discovery window doesn't
generalize past a regime break):
  DISCOVERY   2015-02-09 -> 2023-12-31 (spans both pre- and post-COVID
              regimes, same split that survived the roll test's second try).
  HOLDOUT     2024-01-01 -> 2026-07-08, touched only once, at the end, to
              score the single DTE value discovery picked.
  GRID        target DTE in {14, 21, 30, 45, 60, 90} calendar days. Actual
              realized DTE per entry can differ from the target -- SSE only
              lists ~4 expiries at a time (near month, next month, two
              further-out quarterlies), and choose_expiry_for() just picks
              whichever listed expiry is closest to the target (excluding
              <=5 DTE) -- so e.g. targets of 60 and 90 may often resolve to
              the same actual contract if nothing closer is listed that day.
              Reported avg_realized_dte makes this visible per cell.

Uses walk_rolling_chain() from condor_delta_roll_backtest.py with rule={}
(no exit trigger -- every position runs to its own expiry) and its
target_dte parameter, which condor_delta_backtest.py's own run_grid() never
exposed.

RUN IN A NORMAL PYTHON ENV. No network calls.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import iron_condor_backtest as IC        # noqa: E402
import condor_delta_backtest as CD       # noqa: E402
import condor_delta_roll_backtest as CR  # noqa: E402  walk_rolling_chain, rows_from_trades

OUTDIR = IC.OUTDIR

SPLIT_DATE = pd.Timestamp("2024-01-01")
TARGET_DELTA, P = 0.20, 3
DTE_GRID = [14, 21, 30, 45, 60, 90]


def window(chain: pd.DataFrame, spot_series: pd.Series, start, end):
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


def sweep(chain: pd.DataFrame, spot_series: pd.Series, label: str) -> pd.DataFrame:
    rows_out = []
    for dte in DTE_GRID:
        trades = CR.walk_rolling_chain(chain, spot_series, TARGET_DELTA, P, {}, target_dte=dte)
        rows = CR.rows_from_trades(trades)
        stats = CD.cell_stats(rows)
        if stats is None:
            print(f"  [{label}] dte={dte:>3}  n={len(rows):>3}  (too few trades, skipped)")
            continue
        hold_days = np.array([(t["exit"] - t["entry"]).days for t in trades])
        rows_out.append(dict(target_dte=dte, avg_realized_dte=float(hold_days.mean()), **stats))
        print(f"  [{label}] dte={dte:>3}  avg_realized={hold_days.mean():>5.1f}d  "
              f"n={stats['n']:>3}  f={stats['f']:.2f}  CAGR={stats['cagr']:+.2%}  "
              f"MDD={stats['mdd']:+.2%}  win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}")
    return pd.DataFrame(rows_out)


def main() -> None:
    chain_full = IC.load_local_chain()
    spot_full = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()
    disc_end = SPLIT_DATE - pd.Timedelta(days=1)

    print(f"Full chain: {data_start.date()} -> {data_end.date()}")
    print(f"Split date: {SPLIT_DATE.date()} (fixed in advance, same as the roll OOS test)")
    print(f"  Discovery window: {data_start.date()} -> {disc_end.date()}")
    print(f"  Holdout window:   {SPLIT_DATE.date()} -> {data_end.date()}")
    print(f"Config: delta={TARGET_DELTA} p={P}, hold-to-expiry-always (no exit/roll)\n")

    disc_chain, disc_spot = window(chain_full, spot_full, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_full, SPLIT_DATE, data_end)

    print(f"--- DISCOVERY sweep ({data_start.date()} -> {disc_end.date()}) ---")
    disc_grid = sweep(disc_chain, disc_spot, "discovery")
    disc_grid.to_csv(OUTDIR / "condor_delta_dte_oos_discovery.csv", index=False)

    if disc_grid.empty:
        print("\nNo discovery cell had enough trades to score -- cannot proceed.")
        return
    frozen_row = disc_grid.sort_values("calmar", ascending=False).iloc[0]
    frozen_dte = int(frozen_row["target_dte"])
    print(f"\nFROZEN pick from discovery: target_dte={frozen_dte} "
          f"(discovery CAGR={frozen_row['cagr']:+.2%}, Calmar={frozen_row['calmar']:.2f}, "
          f"avg_realized_dte={frozen_row['avg_realized_dte']:.1f}d)")

    print(f"\n--- HOLDOUT sweep ({SPLIT_DATE.date()} -> {data_end.date()}, for context "
          f"-- not what's being validated) ---")
    hold_grid = sweep(hold_chain, hold_spot, "holdout")
    hold_grid.to_csv(OUTDIR / "condor_delta_dte_oos_holdout.csv", index=False)

    print(f"\n--- OOS EVALUATION: frozen target_dte={frozen_dte} applied to holdout-only data ---")
    trades = CR.walk_rolling_chain(hold_chain, hold_spot, TARGET_DELTA, P, {}, target_dte=frozen_dte)
    rows = CR.rows_from_trades(trades)
    frozen_oos = CD.cell_stats(rows)
    default_row = hold_grid[hold_grid["target_dte"] == 30]
    hold_best = hold_grid.sort_values("calmar", ascending=False).iloc[0] if not hold_grid.empty else None

    if frozen_oos is None:
        print(f"  frozen dte={frozen_dte} on holdout: too few trades to score (n={len(rows)})")
    else:
        print(f"  frozen dte={frozen_dte} on holdout:   n={frozen_oos['n']:>3}  "
              f"f={frozen_oos['f']:.2f}  CAGR={frozen_oos['cagr']:+.2%}  "
              f"MDD={frozen_oos['mdd']:+.2%}  Calmar={frozen_oos['calmar']:.2f}")
    if not default_row.empty:
        d = default_row.iloc[0]
        print(f"  holdout at default dte=30 (unchanged status quo): n={int(d['n']):>3}  "
              f"f={d['f']:.2f}  CAGR={d['cagr']:+.2%}  MDD={d['mdd']:+.2%}  Calmar={d['calmar']:.2f}")
    if hold_best is not None:
        print(f"  holdout's OWN best (dte={int(hold_best['target_dte'])}, in-sample on "
              f"holdout, for context only): CAGR={hold_best['cagr']:+.2%}  Calmar={hold_best['calmar']:.2f}")

    print("\n--- Verdict ---")
    if frozen_oos is not None and not default_row.empty:
        d = default_row.iloc[0]
        beats_default = frozen_oos["cagr"] > d["cagr"]
        positive = frozen_oos["cagr"] > 0
        print(f"  Frozen discovery pick (dte={frozen_dte}) on holdout: "
              f"{'POSITIVE' if positive else 'non-positive'} edge, "
              f"{'beats' if beats_default else 'does not beat'} the unchanged dte=30 status quo.")
    else:
        print("  Could not score cleanly on holdout (too few trades) -- inconclusive.")

    print(f"\nSaved: {OUTDIR/'condor_delta_dte_oos_discovery.csv'} and "
          f"{OUTDIR/'condor_delta_dte_oos_holdout.csv'}")


if __name__ == "__main__":
    main()
