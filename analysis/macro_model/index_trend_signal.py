# -*- coding: utf-8 -*-
"""
index_trend_signal.py
======================
Trend-persistence signal on the SSE Composite (上证指数, 000001) -- the broad-
market index, deliberately NOT the 50ETF the condor itself trades. Two
independent uses:

  1. condor_trend_exit_backtest.py -- a MID-TRADE exit trigger. The regime
     diagnostic (condor_delta_regime_diagnostic.py) found the discovery
     period's 5 disaster trades were mostly persistent directional grinds,
     not vol spikes (one -- Jan 2018 -- had realized vol BELOW what was
     priced, so a vol screen alone wouldn't have flagged it). A quick check
     (see condor_delta_investigation_summary.md addendum) found trailing
     signal AT ENTRY does not reliably predict which trades go bad -- so this
     is built as a REACTIVE mid-trade monitor, not a predictive entry filter.
  2. sse_4000_level_read.py -- a standalone descriptive read of where the
     index sits relative to a round-number level (4000 as of 2026), how many
     times it's tested that level recently, and whether recent crossings look
     like genuine breaks or rejections.

Core measure: Kaufman's Efficiency Ratio (ER) = net move / sum of |daily
moves| over a trailing N-day window. ER in [0, 1]:
  ER near 1 -- smooth, persistent trend ("grind"), most days move the same
              direction. This is the shape that can occur at LOW realized
              vol (Jan 2018's failure mode) -- a pure stdev/vol screen can't
              see it, but ER can, since it measures directional efficiency,
              not move magnitude.
  ER near 0 -- choppy / range-bound / mean-reverting -- the regime a short-
              premium condor wants.

All signals are computed causally (trailing windows only, no lookahead) so
they're safe to use as of any given day without leaking future information.

Data: Data/index/index_daily_000001.csv (date, open, close, high, low,
volume), fetched via Tencent's public kline endpoint, 2010-01-14 -> present.
Gitignored/local, same as Data/options/ -- refresh by re-running the fetch
(see the one-off pagination snippet in this investigation's chat log; not
wired into a script since the index rarely needs same-day refresh for this
use case).

RUN IN A NORMAL PYTHON ENV. No network calls (reads the local CSV cache).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDEX_CSV = Path(__file__).resolve().parents[2] / "Data" / "index" / "index_daily_000001.csv"

ER_WINDOWS = (10, 20, 40, 60)
MOM_WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}   # trading days


def load_index() -> pd.Series:
    """Daily close, SSE Composite (000001), sorted ascending by date."""
    df = pd.read_csv(INDEX_CSV, parse_dates=["date"])
    return df.set_index("date")["close"].sort_index()


def efficiency_ratio(s: pd.Series, n: int) -> pd.Series:
    """Kaufman ER over trailing n days, vectorized (causal: value at date t
    only uses data <= t)."""
    net = (s - s.shift(n)).abs()
    path = s.diff().abs().rolling(n).sum()
    return (net / path).replace([np.inf, -np.inf], np.nan)


def trend_signal_frame(s: pd.Series, windows: tuple[int, ...] = ER_WINDOWS) -> pd.DataFrame:
    """One row per trading day: ER_{n} for each window, plus 1m/3m momentum.
    Every column is causal (trailing-only)."""
    out = pd.DataFrame(index=s.index)
    for n in windows:
        out[f"ER_{n}"] = efficiency_ratio(s, n)
    for label, n in MOM_WINDOWS.items():
        out[f"mom_{label}"] = s / s.shift(n) - 1
    out["px"] = s
    return out


# ------------------------------------------------------------ key-level read
def level_crossings(s: pd.Series, level: float, confirm_days: int = 5) -> pd.DataFrame:
    """Genuine crossings of `level`: a side-change that PERSISTS for at least
    confirm_days trading days (filters out single-day whipsaws that cross and
    immediately snap back -- not a real break of the level). Returns one row
    per genuine crossing: date, direction ('up'/'down'), px_at_cross,
    held_days (trading days the new side lasted before the next genuine
    crossing, or still-open run length for the most recent one)."""
    above = (s > level)
    raw_changes = above.ne(above.shift(1))
    raw_changes.iloc[0] = False
    change_dates = s.index[raw_changes]

    genuine = []
    for d in change_dates:
        pos = s.index.get_loc(d)
        if pos + confirm_days > len(s) - 1:
            # not enough trailing days yet to confirm -- still, keep if it's
            # the most recent candidate (still holding as of today counts)
            still_holds = bool((above.iloc[pos:] == above.iloc[pos]).all())
            if not still_holds:
                continue
        else:
            if not bool((above.iloc[pos:pos + confirm_days + 1] == above.iloc[pos]).all()):
                continue  # snapped back within confirm_days -- a false break, not genuine
        genuine.append(dict(date=d, direction="up" if above.iloc[pos] else "down",
                             px_at_cross=float(s.iloc[pos])))

    df = pd.DataFrame(genuine)
    if df.empty:
        return df
    df["held_days"] = [
        (s.index.get_loc(df["date"].iloc[i + 1]) - s.index.get_loc(df["date"].iloc[i])
         if i + 1 < len(df) else len(s) - 1 - s.index.get_loc(df["date"].iloc[i]))
        for i in range(len(df))
    ]
    return df


def key_level_read(s: pd.Series, level: float, tol_pct: float = 0.015,
                    lookback_days: int = 250, confirm_days: int = 5) -> dict:
    """Descriptive (not predictive) summary of how the index is behaving
    around a round-number level. tol_pct: how close (as a fraction of level)
    counts as "testing" the level. lookback_days: recent window for the
    touch count (default ~1 trading year)."""
    recent = s.iloc[-lookback_days:]
    touches = recent[(recent / level - 1).abs() <= tol_pct]
    crossings = level_crossings(s, level, confirm_days=confirm_days)
    recent_crossings = crossings[crossings["date"] >= recent.index[0]] if not crossings.empty else crossings

    last_px = float(s.iloc[-1])
    side = "above" if last_px > level else "below"
    dist_pct = last_px / level - 1

    current_run_days = None
    if not crossings.empty:
        last_cross = crossings.iloc[-1]
        current_run_days = int(s.index.get_loc(s.index[-1]) - s.index.get_loc(last_cross["date"]))

    return dict(
        level=level, last_date=s.index[-1], last_px=last_px, side=side, dist_pct=dist_pct,
        touches_last_n=len(touches), lookback_days=lookback_days, tol_pct=tol_pct,
        n_genuine_crossings_all_time=len(crossings),
        recent_crossings=recent_crossings,
        days_since_last_genuine_cross=current_run_days,
        last_cross=(crossings.iloc[-1].to_dict() if not crossings.empty else None),
    )


if __name__ == "__main__":
    s = load_index()
    print(f"SSE Composite: {s.index.min().date()} -> {s.index.max().date()}, {len(s)} trading days")
    sig = trend_signal_frame(s)
    print(sig.tail(5))
    read = key_level_read(s, 4000.0)
    print(f"\nlevel=4000  last_px={read['last_px']:.1f} ({read['side']}, "
          f"{read['dist_pct']:+.2%})  touches(last {read['lookback_days']}d)={read['touches_last_n']}  "
          f"genuine crossings all-time={read['n_genuine_crossings_all_time']}  "
          f"days since last genuine cross={read['days_since_last_genuine_cross']}")
