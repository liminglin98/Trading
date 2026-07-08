# -*- coding: utf-8 -*-
"""
condor_delta_regime_diagnostic.py
====================================
condor_delta_dte_sweep.py's OOS test found that the delta=0.20/p=3
hold-to-expiry condor has NEGATIVE mean expectancy over 2015-2023 (all DTE
choices, Kelly f=0) but looks strong over 2024-2026 (the holdout window).
Before trusting or discarding either number, this asks WHY they differ:
realized vol, implied vol (the volatility risk premium the condor is
actually harvesting), and the concentration/timing of large losing trades.

For each hold-to-expiry trade (delta=0.20, p=3, target_dte=30, same as the
DTE sweep's default cell) in each period:
  entry ATM IV     average quoted IV of the strike(s) nearest spot on entry
                    day (the vol level being sold)
  realized vol      annualized stdev of the underlying's daily log returns
                    over the trade's actual holding window (the vol that
                    actually showed up)
  VRP               entry ATM IV - realized vol (positive = condor should
                    profit on average; this is the core edge premium
                    sellers are supposed to harvest)
  move              net % move of the underlying from entry to exit
                    (direction/magnitude context for losing trades)

Also reports period-level realized vol from the DAILY return series
(unconditional on any trade), and lists the worst-N trades per period with
dates, for cross-referencing against known market events.

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
TARGET_DELTA, P, TARGET_DTE = 0.20, 3, 30


def window(chain, spot_series, start, end):
    c = chain[(chain[CD.C_DATE] >= start) & (chain[CD.C_DATE] <= end)]
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    return c, s


def entry_atm_iv(day: pd.DataFrame, spot: float) -> float:
    """Average quoted IV of the strike(s) closest to spot on entry day,
    across whatever expiry(ies) are present in `day` -- a simple, real,
    model-free read of 'how rich is the vol being sold right now.'"""
    strikes = day[CD.C_STRIKE].unique()
    if len(strikes) == 0:
        return np.nan
    nearest = strikes[np.argmin(np.abs(strikes - spot))]
    ivs = day[(day[CD.C_STRIKE] == nearest) & day["隐含波动率"].notna()]["隐含波动率"]
    return float(ivs.mean()) if len(ivs) else np.nan


def realized_vol(spot_series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    s = spot_series[(spot_series.index >= start) & (spot_series.index <= end)]
    if len(s) < 3:
        return np.nan
    ret = np.log(s / s.shift(1)).dropna()
    return float(ret.std() * np.sqrt(252))


def analyze(chain_full: pd.DataFrame, spot_full: pd.Series, chain_w: pd.DataFrame,
            spot_w: pd.Series, label: str) -> pd.DataFrame:
    trades = CR.walk_rolling_chain(chain_w, spot_w, TARGET_DELTA, P, {}, target_dte=TARGET_DTE)
    rows = []
    for t in trades:
        # rule={} means every trade holds to its own expiry, so t["exit"]
        # IS the chosen expiry date exactly -- filter to that expiry's own
        # smile rather than averaging IV across whatever else is listed.
        day_all = chain_full[chain_full[CD.C_DATE] == t["entry"]]
        day = day_all[day_all[CD.C_EXP] == t["exit"]]
        if day.empty:
            day = day_all
        spot_entry = float(spot_full.asof(t["entry"]))
        spot_exit = float(spot_full.asof(t["exit"]))
        iv = entry_atm_iv(day, spot_entry)
        rv = realized_vol(spot_full, t["entry"], t["exit"])
        rows.append(dict(entry=t["entry"], exit=t["exit"],
                          hold_days=(t["exit"] - t["entry"]).days,
                          r=t["pnl"] / t["max_loss"], atm_iv=iv, realized_vol=rv,
                          vrp=iv - rv if pd.notna(iv) and pd.notna(rv) else np.nan,
                          pct_move=spot_exit / spot_entry - 1))
    df = pd.DataFrame(rows)
    print(f"\n=== {label}: {len(df)} trades ===")
    print(f"  mean r/trade:       {df['r'].mean():+.4f}   win rate: {(df['r']>0).mean():.0%}")
    print(f"  mean entry ATM IV:  {df['atm_iv'].mean():.1%}")
    print(f"  mean realized vol:  {df['realized_vol'].mean():.1%}")
    print(f"  mean VRP (IV-RV):   {df['vrp'].mean():+.1%}   (positive = vol sold was rich, as intended)")
    print(f"  mean |monthly move|: {df['pct_move'].abs().mean():.1%}   "
          f"std(move): {df['pct_move'].std():.1%}")
    print(f"  underlying daily realized vol (period-level, unconditional): "
          f"{realized_vol(spot_w, spot_w.index.min(), spot_w.index.max()):.1%}")
    worst = df.sort_values("r").head(5)
    print(f"  worst 5 trades:")
    for _, r in worst.iterrows():
        print(f"    {r['entry'].date()} -> {r['exit'].date()}  r={r['r']:+.3f}  "
              f"move={r['pct_move']:+.1%}  IV={r['atm_iv']:.1%}  RV={r['realized_vol']:.1%}")
    return df


def main() -> None:
    chain_full = IC.load_local_chain()
    spot_full = CD.chain_spot_series(chain_full)
    data_start, data_end = chain_full[CD.C_DATE].min(), chain_full[CD.C_DATE].max()
    disc_end = SPLIT_DATE - pd.Timedelta(days=1)

    disc_chain, disc_spot = window(chain_full, spot_full, data_start, disc_end)
    hold_chain, hold_spot = window(chain_full, spot_full, SPLIT_DATE, data_end)

    disc_df = analyze(chain_full, spot_full, disc_chain, disc_spot,
                       f"DISCOVERY {data_start.date()} -> {disc_end.date()}")
    hold_df = analyze(chain_full, spot_full, hold_chain, hold_spot,
                       f"HOLDOUT {SPLIT_DATE.date()} -> {data_end.date()}")

    disc_df.to_csv(OUTDIR / "condor_delta_regime_discovery_trades.csv", index=False)
    hold_df.to_csv(OUTDIR / "condor_delta_regime_holdout_trades.csv", index=False)

    print("\n=== Side-by-side ===")
    print(f"{'metric':<28} {'discovery':>12} {'holdout':>12}")
    for name, d, h in [
        ("mean r/trade", disc_df["r"].mean(), hold_df["r"].mean()),
        ("win rate", (disc_df['r']>0).mean(), (hold_df['r']>0).mean()),
        ("mean entry ATM IV", disc_df["atm_iv"].mean(), hold_df["atm_iv"].mean()),
        ("mean realized vol", disc_df["realized_vol"].mean(), hold_df["realized_vol"].mean()),
        ("mean VRP (IV-RV)", disc_df["vrp"].mean(), hold_df["vrp"].mean()),
        ("frac trades r<-0.5", (disc_df['r']<-0.5).mean(), (hold_df['r']<-0.5).mean()),
        ("frac trades r<-0.9", (disc_df['r']<-0.9).mean(), (hold_df['r']<-0.9).mean()),
    ]:
        print(f"{name:<28} {d:>12.2%} {h:>12.2%}")

    print(f"\nSaved: {OUTDIR/'condor_delta_regime_discovery_trades.csv'} and "
          f"{OUTDIR/'condor_delta_regime_holdout_trades.csv'}")


if __name__ == "__main__":
    main()
