# -*- coding: utf-8 -*-
"""
condor_300etf_backtest.py
==========================
The whole condor_delta_* investigation (see condor_delta_investigation_summary
.md) found negative expectancy for the delta-targeted iron condor on SSE
510050 (50ETF) options, and every management/overlay lever tested (exit,
roll, DTE, wing width, trend/momentum exit, tail hedge, trend sleeve,
asymmetric skew) failed to fix it. §4 of that summary flagged one untested
lever that needed zero new data sourcing: Data/options/option_daily_510300.csv
(CSI 300 ETF options) already exists locally, in the identical schema, and
had never been run through the condor construction itself.

This is that test -- the IDENTICAL construction as condor_delta_backtest.py
(monthly entry, target-delta short strikes, protective wings p listed
strikes further OTM, hold-to-expiry-always, real quoted closes, fractional
Kelly sizing), just swapped onto 510300. Reuses condor_delta_backtest.py's
(CD) pure functions directly -- they take a chain DataFrame as an argument
and never reference the underlying ticker, so passing a 510300-loaded chain
through them is exactly equivalent to what condor_delta_backtest.py did for
510050, no logic duplicated/forked.

CSI 300 is a broader, less single-sector-concentrated index than the SSE 50
(which skews heavily bank/financial) -- the open question is whether that
changes the tail-event profile that drove 510050's negative expectancy, not
whether the general VRP-harvesting idea works (already established that it
theoretically should, see condor_delta_regime_diagnostic.py).

Data: option_daily_510300.csv covers 2019-12-23 -> 2026-07-08 (300ETF options
only started listing in Dec 2019) -- ~6.5 years, meaningfully shorter than
510050's 9-year discovery window. Same SPLIT_DATE (2024-01-01) as every other
script in this investigation is used here too, so holdout periods line up
across underlyings even though discovery is shorter (~4 years / ~48 months
vs 510050's ~9 years / ~107 months) -- flagged explicitly in the printed
output, not glossed over.

Swept: target_delta x p (identical grid to condor_delta_backtest.py).
Fixed by convention: target_dte=30, hold-to-expiry-always (no exit/roll --
established as the best-performing choice on 510050), same Kelly/slippage
constants as the rest of the project.

Frozen-pick selection: mean_r (not Calmar) -- Kelly-rejection collapses
Calmar to nan across much of this grid, same reason as every OOS-disciplined
script since condor_trend_exit_backtest.py.

RUN IN A NORMAL PYTHON ENV. No network calls.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import iron_condor_backtest as IC   # noqa: E402  OUTDIR only
import condor_delta_backtest as CD  # noqa: E402  reused pure functions

OUTDIR = IC.OUTDIR
DATA_DIR = Path(__file__).resolve().parents[2] / "Data" / "options"
UNDERLYING = "510300"
CHAIN_CSV = DATA_DIR / f"option_daily_{UNDERLYING}.csv"
ETF_CSV = DATA_DIR / f"etf_daily_{UNDERLYING}.csv"

SPLIT_DATE = pd.Timestamp("2024-01-01")
DELTA_GRID = CD.DELTA_GRID   # [0.10, 0.15, 0.20, 0.25, 0.30]
P_GRID = CD.P_GRID           # [1, 2, 3, 4]


def load_chain() -> pd.DataFrame:
    """Same schema/cleaning as IC.load_local_chain, just pointed at 510300."""
    df = pd.read_csv(CHAIN_CSV, encoding="utf-8-sig", dtype={"合约代码": str})
    df = df[~df["合约简称"].astype(str).str.endswith("A")]   # drop adjusted contracts
    df["交易日期"] = pd.to_datetime(df["交易日期"])
    df["到期日"] = pd.to_datetime(df["到期日"])
    for col in ("行权价(元)", "距到期日剩余天数", "隐含波动率", "期权标的收盘价"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("交易日期")


def load_etf() -> pd.Series:
    df = pd.read_csv(ETF_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def window(chain, spot_series, start, end):
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


def run_grid(chain: pd.DataFrame, spot_series: pd.Series) -> dict[tuple, list]:
    """Identical logic to CD.run_grid, just inlined so it can print a
    label-free progress line (CD.run_grid always prints its own header)."""
    entry_dates = CD.monthly_entry_dates(chain)
    trades: dict[tuple, list] = {}
    for d in entry_dates:
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
        S_exp = float(spot_series.loc[exp_date])
        for target_delta in DELTA_GRID:
            for p in P_GRID:
                built = CD.build_condor_by_delta(sub, target_delta, p)
                if built is None:
                    continue
                pnl = CD.realized_pnl(built["strikes"], built["credit"], S_exp)
                r = pnl / built["max_loss"]
                trades.setdefault((target_delta, p), []).append((d, r))
    return trades


def sweep(chain, spot_series, label: str) -> pd.DataFrame:
    trades = run_grid(chain, spot_series)
    rows = []
    for (target_delta, p), lst in trades.items():
        stats = CD.cell_stats(lst)
        if stats is None:
            continue
        r = np.array([x for _, x in lst])
        rows.append(dict(target_delta=target_delta, p=p, mean_r=float(r.mean()), **stats))
    grid = pd.DataFrame(rows)
    if grid.empty:
        print(f"  [{label}] no cell had >=10 trades")
        return grid
    for _, row in grid.sort_values("mean_r", ascending=False).iterrows():
        print(f"  [{label}] delta={row['target_delta']:.2f} p={int(row['p'])}  "
              f"n={int(row['n']):>3}  mean_r={row['mean_r']:+.4f}  f={row['f']:.2f}  "
              f"CAGR={row['cagr']:+.2%}  MDD={row['mdd']:+.2%}  win={row['win']:.0%}  "
              f"Calmar={row['calmar']:.2f}  worst={row['worst']:+.2f}")
    return grid


def disaster_scan(chain: pd.DataFrame, spot_series: pd.Series, target_delta=0.20, p=3) -> None:
    """510300's own worst trades at the default config, discovery period only
    -- purely descriptive (which calendar months hurt 300ETF condors), not an
    OOS claim. Useful context for a future 2-way diversification test: do
    these dates line up with 510050's 5 known disasters or not."""
    trades = run_grid(chain, spot_series).get((target_delta, p), [])
    if not trades:
        print("  (no trades built at default config for disaster scan)")
        return
    worst = sorted(trades, key=lambda t: t[1])[:5]
    print(f"\n  Worst 5 trades, delta={target_delta} p={p}, discovery period:")
    for d, r in worst:
        print(f"    {d.date()}: r={r:+.2%}")


def main() -> None:
    chain_full = load_chain()
    etf = load_etf()
    spot_series = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()
    disc_end = SPLIT_DATE - pd.Timedelta(days=1)

    print(f"510300 chain: {data_start.date()} -> {data_end.date()}, "
          f"{chain_full[CD.C_DATE].nunique()} trading days")
    print(f"NOTE: discovery window here is {data_start.date()} -> {disc_end.date()} "
          f"(~{(disc_end - data_start).days/365.25:.1f} yrs) -- meaningfully shorter than "
          f"510050's 9-year discovery, since 300ETF options only began listing "
          f"2019-12-23. Same SPLIT_DATE ({SPLIT_DATE.date()}) used throughout this "
          f"investigation, so the holdout window lines up across underlyings.")

    disc_chain, disc_spot = window(chain_full, spot_series, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_series, SPLIT_DATE, data_end)

    print(f"\n--- DISCOVERY sweep ({data_start.date()} -> {disc_end.date()}) ---")
    disc_grid = sweep(disc_chain, disc_spot, "discovery")
    disc_grid.to_csv(OUTDIR / "condor_300etf_oos_discovery.csv", index=False)

    if disc_grid.empty:
        print("\nNo discovery cell had enough trades -- cannot proceed.")
        return

    frozen_row = disc_grid.sort_values("mean_r", ascending=False).iloc[0]
    frozen_delta, frozen_p = float(frozen_row["target_delta"]), int(frozen_row["p"])
    print(f"\nFROZEN pick from discovery (by mean_r): delta={frozen_delta:.2f} p={frozen_p}  "
          f"(discovery mean_r={frozen_row['mean_r']:+.4f}, CAGR={frozen_row['cagr']:+.2%}, "
          f"Calmar={frozen_row['calmar']:.2f})")

    default_disc = disc_grid[(disc_grid["target_delta"] == 0.20) & (disc_grid["p"] == 3)]
    if not default_disc.empty:
        d = default_disc.iloc[0]
        print(f"  (for reference, delta=0.20 p=3 discovery: mean_r={d['mean_r']:+.4f}, "
              f"CAGR={d['cagr']:+.2%}, Calmar={d['calmar']:.2f})")

    disaster_scan(disc_chain, disc_spot)

    print(f"\n--- HOLDOUT sweep ({SPLIT_DATE.date()} -> {data_end.date()}, for context) ---")
    hold_grid = sweep(hold_chain, hold_spot, "holdout")
    hold_grid.to_csv(OUTDIR / "condor_300etf_oos_holdout.csv", index=False)

    print(f"\n--- OOS EVALUATION: frozen delta={frozen_delta:.2f} p={frozen_p} "
          f"applied to holdout-only data ---")
    hold_trades = run_grid(hold_chain, hold_spot).get((frozen_delta, frozen_p), [])
    frozen_oos = CD.cell_stats(hold_trades)
    if frozen_oos is None:
        print(f"  frozen delta={frozen_delta:.2f} p={frozen_p} on holdout: "
              f"too few trades (n={len(hold_trades)})")
    else:
        r = np.array([x for _, x in hold_trades])
        print(f"  frozen delta={frozen_delta:.2f} p={frozen_p} on holdout:  "
              f"n={frozen_oos['n']:>3}  mean_r={r.mean():+.4f}  f={frozen_oos['f']:.2f}  "
              f"CAGR={frozen_oos['cagr']:+.2%}  MDD={frozen_oos['mdd']:+.2%}  "
              f"Calmar={frozen_oos['calmar']:.2f}")

    default_hold = hold_grid[(hold_grid["target_delta"] == 0.20) & (hold_grid["p"] == 3)] if not hold_grid.empty else pd.DataFrame()
    if not default_hold.empty:
        d = default_hold.iloc[0]
        print(f"  holdout at default delta=0.20 p=3 (status quo): n={int(d['n']):>3}  "
              f"mean_r={d['mean_r']:+.4f}  CAGR={d['cagr']:+.2%}  Calmar={d['calmar']:.2f}")

    bh = CD.buy_and_hold(etf, data_start, data_end)
    print(f"\nBuy & hold 300ETF over the full window ({data_start.date()} -> {data_end.date()}):")
    print(f"  CAGR={bh['cagr']:+.2%}  MDD={bh['mdd']:+.2%}  Calmar={bh['calmar']:.2f}")

    print(f"\nSaved: {OUTDIR/'condor_300etf_oos_discovery.csv'} and "
          f"{OUTDIR/'condor_300etf_oos_holdout.csv'}")
    print("Caveats: ~4-year discovery window (vs 510050's 9 years) means far fewer "
          "tail-event realizations to learn from or generalize about -- any 'frozen "
          "pick beats holdout' result here carries much wider uncertainty than the "
          "510050 findings elsewhere in this investigation, given the shorter history.")


if __name__ == "__main__":
    main()
