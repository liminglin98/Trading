# -*- coding: utf-8 -*-
"""
condor_delta_roll_oos_test.py
================================
condor_delta_roll_backtest.py found a positive-edge stop-loss "roll" plateau
(~1.6x-2.1x credit received) at delta=0.20/p=3, and the same shape at a
second entry config (delta=0.25/p=3, ~1.7x-2.3x). Both of those searched
ELEVEN stop-loss multiples against the SAME full 2015-2026 sample and picked
the best cell -- that's in-sample data-snooping (with only ~120-140 trades
total, testing ~30+ parameter combinations and reporting the winner is
guaranteed to look good even under pure noise). Neither run is an
out-of-sample test.

This script is one. Setting, fixed in advance (not tuned):

  SPLIT DATE   2021-01-01, a round calendar boundary near the temporal
               midpoint of the chain, chosen for landing after the 2020
               COVID vol shock so the discovery half still contains a real
               stress period. Not chosen by trying dates and picking the one
               that makes the result look best.
  DISCOVERY    2015-02-09 -> 2020-12-31. The full stop-loss grid (1.5x-3.0x)
               is swept HERE ONLY, using chain/ETF data trimmed to this
               window (no leakage: a trade's entry, life, and realization
               all come from within-window data only -- see window() below).
               Whichever multiple wins on Calmar in this window is FROZEN.
  HOLDOUT      2021-01-01 -> 2026-07-08. The frozen multiple is applied here
               -- once, not re-optimized -- and scored on trades built
               entirely from holdout-only data. Holdout's own independently-
               best multiple (from sweeping the same grid on holdout data)
               is also reported, for context on what "best" looks like when
               you're allowed to look -- NOT as the thing being validated.
  CONFIG       delta=0.20/p=3 (the primary config from condor_delta_roll_
               backtest.py). Not repeated for delta=0.25/p=3 here to keep
               this focused; the same window() split can be reapplied there.

Honest caveat: this is not a fully blind test. The analyst (and whoever
reads condor_delta_roll_backtest.py's output first) already knows the
full-sample answer skews toward 1.6x-2.1x, which anchors expectations before
this script runs. What makes the mechanism meaningful anyway is that the
DISCOVERY grid search is computed fresh, from scratch, on the restricted
window, without being told that answer -- so freeze-then-evaluate is a real
test of whether discovery's pick still works on data it never touched, even
if the prior isn't literally blind.

RUN IN A NORMAL PYTHON ENV. No network calls.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import iron_condor_backtest as IC         # noqa: E402
import condor_delta_backtest as CD        # noqa: E402
import condor_delta_roll_backtest as CR   # noqa: E402  walk_rolling_chain, rows_from_trades

OUTDIR = IC.OUTDIR

# SPLIT_DATE: override via `python condor_delta_roll_oos_test.py 2024-01-01`.
# Default 2021-01-01 puts a mostly-pre-COVID window into discovery; the
# alternate 2024-01-01 split instead gives discovery BOTH regimes (pre- and
# post-COVID) so it isn't being asked to survive a regime break, only to
# generalize forward within a regime it has already seen examples of.
import sys
SPLIT_DATE = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2021-01-01")
TARGET_DELTA, P = 0.20, 3
SL_GRID = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 3.0]


def window(chain: pd.DataFrame, spot_series: pd.Series, start, end):
    """Trim chain + spot series to [start, end] with no cross-window leakage
    -- walk_rolling_chain only ever sees rows inside this slice, so a trade
    that would need post-window data to realize its own expiry is dropped
    (existing 'not yet realized' logic), not filled in from the other side."""
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


def sweep(chain: pd.DataFrame, spot_series: pd.Series, label: str) -> pd.DataFrame:
    rows_out = []
    variants = [("baseline", {})] + [(f"{m:.1f}x", {"stop_loss_mult": m}) for m in SL_GRID]
    for name, rule in variants:
        trades = CR.walk_rolling_chain(chain, spot_series, TARGET_DELTA, P, rule)
        rows = CR.rows_from_trades(trades)
        stats = CD.cell_stats(rows)
        if stats is None:
            print(f"  [{label}] {name:<10} n={len(rows):>3}  (too few trades, skipped)")
            continue
        rows_out.append(dict(variant=name, **rule, **stats))
        print(f"  [{label}] {name:<10} n={stats['n']:>3}  f={stats['f']:.2f}  "
              f"CAGR={stats['cagr']:+.2%}  MDD={stats['mdd']:+.2%}  "
              f"win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}")
    return pd.DataFrame(rows_out)


def main() -> None:
    chain_full = IC.load_local_chain()
    spot_full = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()

    disc_end = SPLIT_DATE - pd.Timedelta(days=1)
    print(f"Full chain: {data_start.date()} -> {data_end.date()}")
    print(f"Split date: {SPLIT_DATE.date()} (fixed in advance)")
    print(f"  Discovery window: {data_start.date()} -> {disc_end.date()}")
    print(f"  Holdout window:   {SPLIT_DATE.date()} -> {data_end.date()}")
    print(f"Config: delta={TARGET_DELTA} p={P}\n")

    disc_chain, disc_spot = window(chain_full, spot_full, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_full, SPLIT_DATE, data_end)
    tag = f"_split{SPLIT_DATE.year}"

    print(f"--- DISCOVERY sweep ({data_start.date()} -> {disc_end.date()}) ---")
    disc_grid = sweep(disc_chain, disc_spot, "discovery")
    disc_grid.to_csv(OUTDIR / f"condor_delta_roll_oos_discovery{tag}.csv", index=False)

    sl_disc = disc_grid[disc_grid["variant"] != "baseline"]
    if sl_disc.empty:
        print("\nNo discovery cell had enough trades to score -- cannot proceed.")
        return
    frozen_row = sl_disc.sort_values("calmar", ascending=False).iloc[0]
    frozen_mult = frozen_row["stop_loss_mult"]
    print(f"\nFROZEN pick from discovery: stop_loss={frozen_mult:.1f}x "
          f"(discovery CAGR={frozen_row['cagr']:+.2%}, Calmar={frozen_row['calmar']:.2f})")

    print(f"\n--- HOLDOUT sweep ({SPLIT_DATE.date()} -> {data_end.date()}, for context "
          f"-- not what's being validated) ---")
    hold_grid = sweep(hold_chain, hold_spot, "holdout")
    hold_grid.to_csv(OUTDIR / f"condor_delta_roll_oos_holdout{tag}.csv", index=False)

    print(f"\n--- OOS EVALUATION: frozen {frozen_mult:.1f}x applied to holdout-only data ---")
    trades = CR.walk_rolling_chain(hold_chain, hold_spot, TARGET_DELTA, P,
                                    {"stop_loss_mult": float(frozen_mult)})
    rows = CR.rows_from_trades(trades)
    frozen_oos = CD.cell_stats(rows)
    base_hold = hold_grid[hold_grid["variant"] == "baseline"]
    sl_hold = hold_grid[hold_grid["variant"] != "baseline"]
    hold_best = sl_hold.sort_values("calmar", ascending=False).iloc[0] if not sl_hold.empty else None

    if frozen_oos is None:
        print(f"  frozen {frozen_mult:.1f}x on holdout: too few trades to score (n={len(rows)})")
    else:
        print(f"  frozen {frozen_mult:.1f}x on holdout:  n={frozen_oos['n']:>3}  "
              f"f={frozen_oos['f']:.2f}  CAGR={frozen_oos['cagr']:+.2%}  "
              f"MDD={frozen_oos['mdd']:+.2%}  Calmar={frozen_oos['calmar']:.2f}")
    if not base_hold.empty:
        b = base_hold.iloc[0]
        print(f"  holdout baseline (no roll):        n={int(b['n']):>3}  "
              f"f={b['f']:.2f}  CAGR={b['cagr']:+.2%}  MDD={b['mdd']:+.2%}  Calmar={b['calmar']:.2f}")
    if hold_best is not None:
        print(f"  holdout's OWN best ({hold_best['variant']}, in-sample "
              f"on holdout, for context only): CAGR={hold_best['cagr']:+.2%}  "
              f"Calmar={hold_best['calmar']:.2f}")

    print("\n--- Verdict ---")
    if frozen_oos is not None and base_hold is not None and not base_hold.empty:
        b = base_hold.iloc[0]
        beat_baseline = frozen_oos["cagr"] > b["cagr"]
        positive = frozen_oos["cagr"] > 0
        print(f"  Frozen discovery pick ({frozen_mult:.1f}x) on holdout: "
              f"{'POSITIVE' if positive else 'non-positive'} edge, "
              f"{'beats' if beat_baseline else 'does not beat'} holdout's own no-roll baseline.")
    else:
        print("  Could not score the frozen pick on holdout (too few trades) -- inconclusive, "
              "not a failure of the rule, just insufficient holdout sample.")

    print(f"\nSaved: {OUTDIR/f'condor_delta_roll_oos_discovery{tag}.csv'} and "
          f"{OUTDIR/f'condor_delta_roll_oos_holdout{tag}.csv'}")


if __name__ == "__main__":
    main()
