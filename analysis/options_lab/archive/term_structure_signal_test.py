"""Does near-month ATM IV exceeding far-month ATM IV ("term-structure
inversion"/backwardation) predict elevated realized vol over the following
days? A predictive/validation study, NOT a trading strategy backtest --
there's no position, no P&L, just: does today's spread carry information
about tomorrow's actual volatility?

Method: for each date with a computable spread (consolidated_store.
term_structure_series), look FORWARD `window` trading days and compute
realized vol from the underlying's own returns over that window. Using
forward data as the TARGET is intentional here (not lookahead bias) --
we're testing whether the spread observed today has predictive information,
not building an executable live rule. (If this test comes back positive,
turning it into an actual entry rule would then need to use only backward-
looking data at decision time, same as every other script in this repo.)

Reports, split discovery (<2024-01-01) / holdout (>=2024-01-01) per this
repo's standard discipline:
  - correlation(spread, forward RV) -- does the SPREAD predict?
  - correlation(near_iv level, forward RV) -- baseline: does just "IV is
    high right now" already predict comparably, via ordinary vol clustering?
    (if the spread adds nothing beyond this, it's not a useful signal on
    its own)
  - mean forward RV conditional on inverted (spread>0) vs not

Usage:
    python term_structure_signal_test.py --underlying 510500
    python term_structure_signal_test.py --underlying all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # archived: options_lab/ modules live one level up

import consolidated_store as cs
import data_store as ds

SPLIT_DATE = pd.Timestamp("2024-01-01")
FORWARD_WINDOWS = [5, 10, 20]

UNDERLYINGS = {
    "510500": "500ETF", "588000": "科创50ETF", "588080": "科创板50ETF",
    "159915": "创业板ETF", "159901": "深证100ETF",
}


def forward_rv(underlying: str, window: int) -> pd.Series:
    """Realized vol (annualized) over the `window` trading days AFTER each
    date (excludes the date's own return)."""
    etf = ds.load_etf_ohlc(underlying)
    s = etf.set_index("date")["close"].sort_index()
    logret = np.log(s).diff()
    return logret.rolling(window).std().shift(-window) * np.sqrt(252)


def _report_split(df: pd.DataFrame, label: str, window: int) -> None:
    if len(df) < 20:
        print(f"    {label}: n={len(df)} (too few to report)")
        return
    r_spread, p_spread = pearsonr(df["spread"], df["fwd_rv"])
    rho_spread, _ = spearmanr(df["spread"], df["fwd_rv"])
    r_level, p_level = pearsonr(df["near_iv"], df["fwd_rv"])

    inv = df[df["spread"] > 0]["fwd_rv"]
    norm = df[df["spread"] <= 0]["fwd_rv"]
    tstat, tp = (np.nan, np.nan) if len(inv) < 5 or len(norm) < 5 else ttest_ind(inv, norm, equal_var=False)

    print(f"    {label}: n={len(df)}  n_inverted={len(inv)} ({len(inv)/len(df):.0%})")
    print(f"      corr(spread, fwd_rv_{window}d): pearson={r_spread:+.3f} (p={p_spread:.3f})  "
          f"spearman={rho_spread:+.3f}")
    print(f"      corr(near_iv level, fwd_rv_{window}d): pearson={r_level:+.3f} (p={p_level:.3f})  <- baseline")
    print(f"      fwd_rv_{window}d mean: inverted={inv.mean():.1%}  normal={norm.mean():.1%}  "
          f"diff={inv.mean()-norm.mean():+.1%}  (t={tstat:+.2f}, p={tp:.3f})")


def run(underlying: str) -> None:
    name = UNDERLYINGS.get(underlying, underlying)
    ts = cs.term_structure_series(underlying)
    if ts.empty:
        print(f"{underlying} ({name}): no dates with >=2 listed expiries -- skipping.")
        return
    print(f"\n{underlying} ({name}): term structure spread computed on {len(ts)} days "
          f"({ts.index.min().date()} -> {ts.index.max().date()}), "
          f"{(ts['spread']>0).mean():.0%} inverted overall")

    for window in FORWARD_WINDOWS:
        fwd = forward_rv(underlying, window)
        df = ts.join(fwd.rename("fwd_rv"), how="inner").dropna(subset=["fwd_rv"])
        print(f"\n  forward window = {window}d:")
        _report_split(df[df.index < SPLIT_DATE], "discovery", window)
        _report_split(df[df.index >= SPLIT_DATE], "holdout  ", window)


def main(underlying: str) -> None:
    targets = list(UNDERLYINGS) if underlying == "all" else [underlying]
    for u in targets:
        run(u)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="all", choices=list(UNDERLYINGS) + ["all"])
    args = ap.parse_args()
    main(args.underlying)
