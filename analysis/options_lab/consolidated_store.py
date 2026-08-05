"""Loader for the Data/买断日更/consolidated/{underlying}.parquet chain history
(see Data/买断日更/README.md), adapted into data_store.py's Chinese-column
schema so it drops into the same backtest helpers (quarterly_atm_short_straddle's
_atm_strike/_leg_row/_intrinsic, backtest.py's engine) as the original
Data/options/option_daily_{510050,510300}.csv feed.

Covers underlyings the old CSV feed never had: 500ETF, 科创50ETF, 科创板50ETF,
创业板ETF, plus the CFFEX index options and the other SSE/SZSE ETFs. Two gaps
vs. the old feed, both handled here rather than left silently broken:

- No underlying spot price column in the consolidated data (raw vendor drops
  don't carry it either -- checked). C_UND_CLOSE is derived per (date, expiry)
  via put-call parity from the chain's own close prices (_parity_forward
  below) -- NOT from the cached Data/options/etf_daily_{code}.csv ETF OHLC
  file, which was tried first (2026-08) and found broken: that cache is
  fetched qfq (前复权, dividend-adjusted, see updater.py's _tencent_kline),
  which understates historical prices more the further back in time, while
  option strikes are always on the raw/unadjusted price -- feeding the
  qfq-adjusted price into ATM strike selection and BS inversion produced a
  spurious implied-vol bias that decayed smoothly from ~200% (2015) to a
  correct ~15-20% (2026), confirmed via Data/exploring/huidian_index_replication/
  (an independent, QVIX-validated vol-index replication built from this same
  settle-price data with no spot dependency at all). Put-call parity is
  self-consistent with the raw strikes and needs no external price series.
- No implied vol column (vendor schema has Greeks -- delta/theta/gamma/vega/
  rho -- but not IV). atm_iv_series() below backs it out via Black-Scholes
  inversion (bs_iv.py) from the close price + derived spot, so iv_pct/vrp-style
  signals work the same way here as in iv_low_long_straddle_backtest.py --
  see long_straddle_rv_contraction_backtest.py's --signal option.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import bs_iv
import data_store as ds

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONSOLIDATED_DIR = REPO / "Data" / "买断日更" / "consolidated"

# short code -> consolidated parquet underlying symbol
UNDERLYING_SYMBOLS = {
    "510050": "510050.XSHG", "510300": "510300.XSHG", "510500": "510500.XSHG",
    "588000": "588000.XSHG", "588080": "588080.XSHG",
    "159901": "159901.XSHE", "159915": "159915.XSHE",
    "159919": "159919.XSHE", "159922": "159922.XSHE",
    "000016": "000016.XSHG", "000300": "000300.XSHG", "000852": "000852.XSHG",
}

_CP_MAP = {"CO": "C", "PO": "P"}

# Vendor-snapshot-only underlyings shown in the Data tab: no Sina live-quote
# coverage (checked 2026-07-24 -- Sina's StockOptionService category list has
# no ChiNext/Shenzhen-100 entry, unlike 500ETF/科创50/科创板50 which do), so
# the option chain itself can only be refreshed by re-running
# consolidate_etf_index_options.py over a fresh vendor data drop, not by
# updater.py.
EXT_VENDOR_ONLY_UNDERLYINGS = {"159915": "创业板ETF", "159901": "深证100ETF"}


def consolidated_path(underlying: str) -> Path:
    sym = UNDERLYING_SYMBOLS.get(underlying, underlying)
    return CONSOLIDATED_DIR / f"{sym}.parquet"


def _parity_forward(chain: pd.DataFrame, r: float = bs_iv.RISK_FREE_DEFAULT) -> pd.DataFrame:
    """One row per (date, expiry): put-call-parity-implied forward price,
    derived from that day's own call/put close prices at the strike
    minimizing |C-P| -- self-consistent with the chain's own raw strikes
    (see module docstring for why this replaced a cached external ETF
    price). At DTE=0 this reduces to K*+(C-P), i.e. the true
    settlement-consistent spot exactly, since e^(r*0)=1. Silently omits
    (date, expiry) pairs where no strike has a positive close on both the
    call and put leg (thin/no-liquidity day) -- C_UND_CLOSE is NaN there,
    same as the old cache-miss behavior. Fully vectorized (merge + groupby-
    idxmin, no per-group Python loop) -- needed since a per-(date,expiry)
    loop with a pivot_table call each is far too slow on the larger
    underlyings (100k+ rows -> tens of thousands of groups)."""
    key = [ds.C_DATE, ds.C_EXPIRY, ds.C_STRIKE]
    calls = chain.loc[chain[ds.C_CP] == "C", key + [ds.C_CLOSE]].rename(columns={ds.C_CLOSE: "C"})
    puts = chain.loc[chain[ds.C_CP] == "P", key + [ds.C_CLOSE]].rename(columns={ds.C_CLOSE: "P"})
    both = calls.merge(puts, on=key, how="inner")
    both = both[(both["C"] > 0) & (both["P"] > 0)]
    if both.empty:
        return pd.DataFrame(columns=[ds.C_DATE, ds.C_EXPIRY, ds.C_UND_CLOSE])

    idx = (both["C"] - both["P"]).abs().groupby([both[ds.C_DATE], both[ds.C_EXPIRY]]).idxmin()
    best = both.loc[idx].copy()
    t_years = (best[ds.C_EXPIRY] - best[ds.C_DATE]).dt.days.clip(lower=0) / 365.0
    best[ds.C_UND_CLOSE] = best[ds.C_STRIKE] + np.exp(r * t_years) * (best["C"] - best["P"])
    return best[[ds.C_DATE, ds.C_EXPIRY, ds.C_UND_CLOSE]]


@lru_cache(maxsize=8)
def load_chain(underlying: str) -> pd.DataFrame:
    """Chain history for `underlying` (e.g. "510500"), remapped into
    data_store.py's Chinese-column schema. C_IV is all-NaN (not in this
    dataset -- see module docstring); C_UND_CLOSE is a put-call-parity
    forward per (date, expiry) (_parity_forward), NaN where no strike has
    both legs quoted positive that day."""
    path = consolidated_path(underlying)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)

    out = pd.DataFrame({
        ds.C_CODE: df["code"].astype(str),
        ds.C_NAME: df["name"].astype(str),
        ds.C_MKT: df["underlying_symbol"].astype(str),
        ds.C_CP: df["contract_type"].map(_CP_MAP),
        ds.C_STRIKE: pd.to_numeric(df["strike"], errors="coerce"),
        ds.C_LIST: pd.to_datetime(df["list_date"]),
        ds.C_EXPIRY: pd.to_datetime(df["expire_date"]),
        ds.C_DATE: pd.to_datetime(df["date"]),
        ds.C_PREV_CLOSE: pd.to_numeric(df["prev_close"], errors="coerce"),
        ds.C_PREV_SETTLE: pd.to_numeric(df["prev_settle"], errors="coerce"),
        ds.C_OPEN: pd.to_numeric(df["open"], errors="coerce"),
        ds.C_HIGH: pd.to_numeric(df["high"], errors="coerce"),
        ds.C_LOW: pd.to_numeric(df["low"], errors="coerce"),
        ds.C_CLOSE: pd.to_numeric(df["close"], errors="coerce"),
        ds.C_SETTLE: pd.to_numeric(df["settle"], errors="coerce"),
        ds.C_VOL: pd.to_numeric(df["volume"], errors="coerce"),
        ds.C_AMT: pd.to_numeric(df["turnover"], errors="coerce"),
        ds.C_OI: pd.to_numeric(df["open_interest"], errors="coerce"),
        ds.C_DELTA: pd.to_numeric(df["delta"], errors="coerce"),
        ds.C_THETA: pd.to_numeric(df["theta"], errors="coerce"),
        ds.C_GAMMA: pd.to_numeric(df["gamma"], errors="coerce"),
        ds.C_VEGA: pd.to_numeric(df["vega"], errors="coerce"),
        ds.C_RHO: pd.to_numeric(df["rho"], errors="coerce"),
        ds.C_UND: df["underlying_symbol"].astype(str).str.split(".").str[0],
    })
    out[ds.C_IV] = float("nan")
    out[ds.C_DTE] = (out[ds.C_EXPIRY] - out[ds.C_DATE]).dt.days

    fwd = _parity_forward(out)
    if fwd.empty:
        out[ds.C_UND_CLOSE] = float("nan")
    else:
        out = out.merge(fwd, on=[ds.C_DATE, ds.C_EXPIRY], how="left")

    out.sort_values([ds.C_DATE, ds.C_EXPIRY, ds.C_CP, ds.C_STRIKE], inplace=True)
    return out.reset_index(drop=True)


def _cp_avg_iv(day: pd.DataFrame, expiry, strike: float, spot: float) -> float:
    """Average of call and put Black-Scholes IV at (expiry, strike) on this
    day's full chain -- more robust than either leg alone. Checked against
    the trusted 510050 feed's real quoted IV: even within 2% of spot, real
    call and put IV differ by a median ~1.8 vol points (skew/liquidity
    noise that doesn't vanish right at the money) -- averaging both legs
    cancels most of that one-sided noise instead of inheriting whichever
    side the call happens to sit on. Falls back to whichever single leg is
    available if the other's price is missing/zero."""
    rows = day[(day[ds.C_EXPIRY] == expiry) & (day[ds.C_STRIKE] == strike)]
    ivs = []
    for cp in ("C", "P"):
        r = rows[rows[ds.C_CP] == cp]
        if r.empty or r[ds.C_CLOSE].iloc[0] <= 0:
            continue
        row = r.iloc[0]
        iv = bs_iv.implied_vol(cp, spot, strike, float(row[ds.C_DTE]) / 365.0, float(row[ds.C_CLOSE]))
        if iv == iv:  # not NaN
            ivs.append(iv)
    return sum(ivs) / len(ivs) if ivs else float("nan")


@lru_cache(maxsize=8)
def atm_iv_series(underlying: str, dte_min: int = 7) -> pd.Series:
    """Daily near-month ATM IV, averaged across the call and put leg at the
    same strike (see _cp_avg_iv) -- same expiry/strike selection as
    backtest.py's _atm_iv_series (nearest expiry with DTE>=dte_min, then
    nearest strike to spot), just computing IV instead of reading a column
    that doesn't exist in this dataset."""
    chain = load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    chain = chain[chain[ds.C_CLOSE] > 0]
    calls = chain[(chain[ds.C_CP] == "C") & chain[ds.C_UND_CLOSE].notna()]
    calls = calls.loc[calls.groupby(ds.C_DATE)[ds.C_DTE].transform(
        lambda s: s == s[s >= dte_min].min() if (s >= dte_min).any() else s == s.min())]
    idx = (calls[ds.C_STRIKE] - calls[ds.C_UND_CLOSE]).abs().groupby(calls[ds.C_DATE]).idxmin()
    atm = calls.loc[idx].sort_values(ds.C_DATE)

    by_date = dict(tuple(chain.groupby(ds.C_DATE)))
    ivs = [
        _cp_avg_iv(by_date[row[ds.C_DATE]], row[ds.C_EXPIRY], row[ds.C_STRIKE], float(row[ds.C_UND_CLOSE]))
        for _, row in atm.iterrows()
    ]
    s = pd.Series(ivs, index=atm[ds.C_DATE].values).sort_index()
    return s[s.notna()]


@lru_cache(maxsize=8)
def term_structure_series(underlying: str, dte_min: int = 7) -> pd.DataFrame:
    """Daily near-month vs. next-month ATM IV (call+put average, see
    _cp_avg_iv), for testing whether term-structure inversion (near > far,
    'backwardation') carries information about near-term realized vol.
    Only dates with >=2 listed expiries produce a row."""
    chain = load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    chain = chain[chain[ds.C_CLOSE] > 0]
    calls = chain[(chain[ds.C_CP] == "C") & chain[ds.C_UND_CLOSE].notna()]

    rows = []
    for dt, day_calls in calls.groupby(ds.C_DATE):
        dtes = day_calls.drop_duplicates(ds.C_EXPIRY).set_index(ds.C_EXPIRY)[ds.C_DTE]
        expiries = sorted(dtes.index)
        eligible = [e for e in expiries if dtes[e] >= dte_min]
        if len(eligible) < 2:
            continue
        near_exp, far_exp = eligible[0], eligible[1]

        near_calls = day_calls[day_calls[ds.C_EXPIRY] == near_exp]
        far_calls = day_calls[day_calls[ds.C_EXPIRY] == far_exp]
        # each expiry gets its own parity forward now (C_UND_CLOSE varies by
        # expiry, not just by date) -- using one shared "spot" for both legs
        # here would silently reintroduce a mismatch bug
        near_spot = float(near_calls[ds.C_UND_CLOSE].iloc[0])
        far_spot = float(far_calls[ds.C_UND_CLOSE].iloc[0])
        near_strike = float(near_calls.loc[(near_calls[ds.C_STRIKE] - near_spot).abs().idxmin(), ds.C_STRIKE])
        far_strike = float(far_calls.loc[(far_calls[ds.C_STRIKE] - far_spot).abs().idxmin(), ds.C_STRIKE])

        day = chain[chain[ds.C_DATE] == dt]
        near_iv = _cp_avg_iv(day, near_exp, near_strike, near_spot)
        far_iv = _cp_avg_iv(day, far_exp, far_strike, far_spot)
        if near_iv != near_iv or far_iv != far_iv:  # NaN check
            continue
        near_dte = float(near_calls.loc[near_calls[ds.C_STRIKE] == near_strike, ds.C_DTE].iloc[0])
        far_dte = float(far_calls.loc[far_calls[ds.C_STRIKE] == far_strike, ds.C_DTE].iloc[0])
        rows.append({"date": dt, "near_dte": near_dte, "far_dte": far_dte,
                    "near_iv": near_iv, "far_iv": far_iv})

    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["spread"] = out["near_iv"] - out["far_iv"]
    return out
