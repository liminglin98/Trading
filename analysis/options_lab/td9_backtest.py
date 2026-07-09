"""TD Sequential (DeMark) 9-count Setup backtest on 50ETF/300ETF daily closes.

Tests whether a completed TD Setup (9 consecutive closes each beyond the
close 4 bars back -- see xuanxue/6.25.5.png's "日线高9共振") has any
measurable edge, on two axes:
  * forward DIRECTIONAL return  (the classic DeMark claim: 9 = exhaustion,
    expect a bounce/pullback)
  * forward REALIZED VOLATILITY (what actually matters for iron-condor /
    butterfly premium selling: does a 9-count predict calmer or wilder
    markets over the following days?)

Both are compared against the unconditional baseline (all days), with a
Welch t-test. Caveat: signal days are not independent draws (TD counts and
their forward windows overlap during trends), so p-values here are an
optimistic lower bound on significance, not a rigorous one -- fine for a
first gut-check on whether this is worth taking seriously.

Usage:
    python td9_backtest.py                        # 510050, default horizons
    python td9_backtest.py --underlying 510300
    python td9_backtest.py --horizons 1,3,5,10,20
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import data_store as ds


# ------------------------------------------------------------------ TD Setup
def td_setup_counts(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Consecutive TD Setup counts (uncapped past 9).

    buy[i]  increments while close[i] < close[i-4]  (downtrend leg -> a run
            of 9 flags bottom exhaustion, the classic "buy" signal).
    sell[i] increments while close[i] > close[i-4]  (uptrend leg -> a run
            of 9 flags top exhaustion, the classic "sell" signal).
    Either count resets to 0 the day the condition breaks.
    """
    c = close.to_numpy()
    n = len(c)
    buy = np.zeros(n, dtype=int)
    sell = np.zeros(n, dtype=int)
    for i in range(4, n):
        buy[i] = buy[i - 1] + 1 if c[i] < c[i - 4] else 0
        sell[i] = sell[i - 1] + 1 if c[i] > c[i - 4] else 0
    return pd.Series(buy, index=close.index), pd.Series(sell, index=close.index)


def first_nine_signals(count: pd.Series) -> pd.DatetimeIndex:
    """Bars where the count first reaches exactly 9 in an uninterrupted run
    (so a run that continues to 10, 11, ... only signals once, at 9)."""
    is_nine = count == 9
    prev_at_least_nine = count.shift(1).fillna(0) >= 9
    return count.index[is_nine & ~prev_at_least_nine]


# ------------------------------------------------------------------ forward metrics
def forward_returns(close: pd.Series, signal_idx: pd.DatetimeIndex,
                    horizons: list[int]) -> pd.DataFrame:
    """log return from the signal day's close to close h bars later."""
    pos = {d: i for i, d in enumerate(close.index)}
    logc = np.log(close.to_numpy())
    rows = []
    for d in signal_idx:
        i = pos[d]
        row = {"date": d}
        for h in horizons:
            j = i + h
            row[f"h{h}"] = logc[j] - logc[i] if j < len(logc) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def forward_vol(close: pd.Series, signal_idx: pd.DatetimeIndex,
                horizons: list[int]) -> pd.DataFrame:
    """Annualized realized vol of daily log returns over the h bars
    immediately following the signal day."""
    logret = np.log(close).diff()
    ret = logret.to_numpy()
    pos = {d: i for i, d in enumerate(close.index)}
    rows = []
    for d in signal_idx:
        i = pos[d]
        row = {"date": d}
        for h in horizons:
            j = i + h
            window = ret[i + 1:j + 1] if j < len(ret) else np.array([])
            row[f"h{h}"] = (np.std(window, ddof=1) * np.sqrt(252)
                            if len(window) > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def summarize(sig: pd.DataFrame, base: pd.DataFrame, horizons: list[int],
             label: str, kind: str) -> None:
    print(f"\n--- {label} ({kind}), n_signals={len(sig)} ---")
    for h in horizons:
        col = f"h{h}"
        s, a = sig[col].dropna(), base[col].dropna()
        if len(s) < 5:
            print(f"  {h:>3}d: too few signals ({len(s)}) -- skip")
            continue
        tstat, pval = stats.ttest_ind(s, a, equal_var=False)
        line = (f"  {h:>3}d: signal mean={s.mean():+.4%} median={s.median():+.4%}"
               f" | baseline mean={a.mean():+.4%} | diff={s.mean() - a.mean():+.4%}"
               f"  t={tstat:+.2f} p={pval:.3f}")
        if kind == "return":
            line += f"  win%={(s > 0).mean():.0%}"
        print(line)


# ------------------------------------------------------------------ main
def main(underlying: str, horizons: list[int]) -> None:
    df = ds.load_etf_ohlc(underlying)
    close = df.set_index("date")["close"].sort_index()
    buy_cnt, sell_cnt = td_setup_counts(close)
    buy_sig = first_nine_signals(buy_cnt)     # bottom-exhaustion ("buy") signals
    sell_sig = first_nine_signals(sell_cnt)   # top-exhaustion ("sell") signals

    print(f"{underlying}: {len(close)} bars, "
          f"{close.index.min().date()} -> {close.index.max().date()}")
    print(f"Buy-9 (expect bounce) signals:    {len(buy_sig)}")
    print(f"Sell-9 (expect pullback) signals: {len(sell_sig)}")

    all_rets = forward_returns(close, close.index, horizons)
    all_vols = forward_vol(close, close.index, horizons)

    if len(buy_sig):
        summarize(forward_returns(close, buy_sig, horizons), all_rets,
                  horizons, "Buy-9", "return")
        summarize(forward_vol(close, buy_sig, horizons), all_vols,
                  horizons, "Buy-9", "vol")
    if len(sell_sig):
        summarize(forward_returns(close, sell_sig, horizons), all_rets,
                  horizons, "Sell-9", "return")
        summarize(forward_vol(close, sell_sig, horizons), all_vols,
                  horizons, "Sell-9", "vol")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="510050")
    ap.add_argument("--horizons", default="1,3,5,10,20")
    args = ap.parse_args()
    main(args.underlying, [int(x) for x in args.horizons.split(",")])
