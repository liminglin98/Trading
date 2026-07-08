# -*- coding: utf-8 -*-
"""
condor_delta_wing_sweep.py
=============================
condor_delta_regime_diagnostic.py traced the discovery-period (2015-2023)
tail losses to persistent directional moves blowing through the protective
wing (p=3 listed strikes past the short). condor_delta_backtest.py and
condor_delta_roll_backtest.py only ever swept p in {1,2,3,4}. This tests
whether going meaningfully wider (more distance before max_loss is reached)
actually reduces those tail losses enough to matter, at the cost of a larger
max_loss (deposit) per contract -- which is fine to compare directly since
every stat here is on r = pnl/max_loss (return per unit of capital at risk),
not per contract, so wider-wing configs aren't unfairly advantaged/penalized
by their larger deposit size.

Same OOS discipline as condor_delta_dte_sweep.py:
  DISCOVERY   2015-02-09 -> 2023-12-31 (spans both regimes)
  HOLDOUT     2024-01-01 -> 2026-07-08, touched once at the end
  GRID        p in {2, 3, 4, 6, 8, 10} listed strikes past the short.
  CONFIG      target_delta=0.20, target_dte=30, hold-to-expiry-always (no
              exit/roll -- the one mechanism that's actually held up).

Also reports the credit and max_loss at each p (from a representative entry)
so the deposit-size tradeoff is visible, not just asserted, and re-checks
the five known discovery disaster trades at each p to see how much of the
loss each wing width actually would have absorbed.

RUN IN A NORMAL PYTHON ENV. No network calls.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import iron_condor_backtest as IC        # noqa: E402
import condor_delta_backtest as CD       # noqa: E402
import condor_delta_roll_backtest as CR  # noqa: E402

OUTDIR = IC.OUTDIR
SPLIT_DATE = pd.Timestamp("2024-01-01")
TARGET_DELTA, TARGET_DTE = 0.20, 30
P_GRID = [2, 3, 4, 6, 8, 10]

WORST_DISCOVERY_ENTRIES = [pd.Timestamp(d) for d in
    ["2022-12-28", "2018-01-02", "2019-02-12", "2015-08-11", "2020-12-23"]]


def window(chain, spot_series, start, end):
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


def sweep(chain, spot_series, label: str) -> pd.DataFrame:
    rows_out = []
    for p in P_GRID:
        trades = CR.walk_rolling_chain(chain, spot_series, TARGET_DELTA, p, {}, target_dte=TARGET_DTE)
        rows = CR.rows_from_trades(trades)
        stats = CD.cell_stats(rows)
        if stats is None:
            print(f"  [{label}] p={p:>2}  n={len(rows):>3}  (too few trades, skipped)")
            continue
        r = np.array([x for _, x in rows])
        rows_out.append(dict(p=p, mean_r=float(r.mean()), **stats))
        print(f"  [{label}] p={p:>2}  n={stats['n']:>3}  mean_r={r.mean():+.4f}  "
              f"f={stats['f']:.2f}  CAGR={stats['cagr']:+.2%}  MDD={stats['mdd']:+.2%}  "
              f"win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}  worst={stats['worst']:+.3f}")
    return pd.DataFrame(rows_out)


def deposit_check(chain: pd.DataFrame) -> None:
    """Credit/max_loss at each p, from one representative entry, so the
    capital tradeoff is visible in real numbers, not asserted."""
    sample_date = pd.Timestamp("2022-06-30")
    day = chain[chain[CD.C_DATE] == sample_date]
    if day.empty:
        return
    exp_date = CD.choose_expiry_for(day, TARGET_DTE)
    if exp_date is None:
        return
    sub = day[day[CD.C_EXP] == pd.Timestamp(exp_date)]
    print(f"\nDeposit-size tradeoff (sample entry {sample_date.date()}, exp {exp_date}):")
    for p in P_GRID:
        built = CD.build_condor_by_delta(sub, TARGET_DELTA, p)
        if built is None:
            print(f"  p={p:>2}: could not build")
            continue
        print(f"  p={p:>2}  credit={built['credit']:.4f}  max_loss={built['max_loss']:.4f}  "
              f"strikes={built['strikes']}")


def disaster_recheck(chain_full: pd.DataFrame, spot_full: pd.Series) -> None:
    print("\nRe-checking the 5 known discovery disaster trades at each wing width "
          "(r = pnl/max_loss, held to expiry, no exit):")
    for entry_date in WORST_DISCOVERY_ENTRIES:
        day = chain_full[chain_full[CD.C_DATE] == entry_date]
        exp_date = CD.choose_expiry_for(day, TARGET_DTE)
        if exp_date is None:
            continue
        exp_date = pd.Timestamp(exp_date)
        sub = day[day[CD.C_EXP] == exp_date]
        S_exp = float(spot_full.loc[exp_date]) if exp_date in spot_full.index else np.nan
        line = f"  {entry_date.date()}: "
        for p in P_GRID:
            built = CD.build_condor_by_delta(sub, TARGET_DELTA, p)
            if built is None:
                line += f"p={p}:n/a  "
                continue
            pnl = CD.realized_pnl(built["strikes"], built["credit"], S_exp)
            r = pnl / built["max_loss"]
            line += f"p={p}:{r:+.2f}  "
        print(line)


def main() -> None:
    chain_full = IC.load_local_chain()
    spot_full = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()
    disc_end = SPLIT_DATE - pd.Timedelta(days=1)

    print(f"Config: delta={TARGET_DELTA} target_dte={TARGET_DTE}, hold-to-expiry-always")
    deposit_check(chain_full)
    disaster_recheck(chain_full, spot_full)

    disc_chain, disc_spot = window(chain_full, spot_full, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_full, SPLIT_DATE, data_end)

    print(f"\n--- DISCOVERY sweep ({data_start.date()} -> {disc_end.date()}) ---")
    disc_grid = sweep(disc_chain, disc_spot, "discovery")
    disc_grid.to_csv(OUTDIR / "condor_delta_wing_oos_discovery.csv", index=False)

    if disc_grid.empty:
        print("\nNo discovery cell had enough trades -- cannot proceed.")
        return
    frozen_row = disc_grid.sort_values("calmar", ascending=False).iloc[0]
    frozen_p = int(frozen_row["p"])
    print(f"\nFROZEN pick from discovery: p={frozen_p} "
          f"(discovery CAGR={frozen_row['cagr']:+.2%}, Calmar={frozen_row['calmar']:.2f}, "
          f"mean_r={frozen_row['mean_r']:+.4f})")

    print(f"\n--- HOLDOUT sweep ({SPLIT_DATE.date()} -> {data_end.date()}, for context) ---")
    hold_grid = sweep(hold_chain, hold_spot, "holdout")
    hold_grid.to_csv(OUTDIR / "condor_delta_wing_oos_holdout.csv", index=False)

    print(f"\n--- OOS EVALUATION: frozen p={frozen_p} applied to holdout-only data ---")
    trades = CR.walk_rolling_chain(hold_chain, hold_spot, TARGET_DELTA, frozen_p, {}, target_dte=TARGET_DTE)
    rows = CR.rows_from_trades(trades)
    frozen_oos = CD.cell_stats(rows)
    default_row = hold_grid[hold_grid["p"] == 3]

    if frozen_oos is None:
        print(f"  frozen p={frozen_p} on holdout: too few trades (n={len(rows)})")
    else:
        print(f"  frozen p={frozen_p} on holdout:  n={frozen_oos['n']:>3}  f={frozen_oos['f']:.2f}  "
              f"CAGR={frozen_oos['cagr']:+.2%}  MDD={frozen_oos['mdd']:+.2%}  Calmar={frozen_oos['calmar']:.2f}")
    if not default_row.empty:
        d = default_row.iloc[0]
        print(f"  holdout at default p=3 (status quo): n={int(d['n']):>3}  f={d['f']:.2f}  "
              f"CAGR={d['cagr']:+.2%}  MDD={d['mdd']:+.2%}  Calmar={d['calmar']:.2f}")

    print(f"\nSaved: {OUTDIR/'condor_delta_wing_oos_discovery.csv'} and "
          f"{OUTDIR/'condor_delta_wing_oos_holdout.csv'}")


if __name__ == "__main__":
    main()
