"""Net gamma exposure (GEX) signal.

Aggregates per-strike open interest x gamma across the FULL listed chain
(every strike, every expiry) per trading date -- unlike rv_pct/iv_pct/vrp
elsewhere in this repo, which reduce to a single near-month-ATM value per
day, dealer gamma exposure is a market-wide quantity summed across
everything currently listed.

Sign convention (industry-standard, UNVERIFIED for this market -- see
gamma_exposure investigation plan / feedback_backtest_rigor memory): dealers
assumed net long calls, net short puts (retail buys calls & sells puts,
dealer is the counterparty). This repo's chain data has aggregate OI per
strike/right only, not dealer-vs-customer identity, so this is a hypothesis
the backtest must validate, not a known fact -- gex_magnitude_pct_series()
(unsigned) is provided as a fallback in case the signed direction turns out
backwards or too noisy.

Coverage: 510050/510300 use data_store.py's real vendor Greeks (one spot
price per date). The other five underlyings use consolidated_store.py,
whose C_UND_CLOSE is a put-call-parity forward derived PER (date, expiry),
not a single per-date spot -- _spot_per_date() below collapses that to one
value per date by taking the nearest listed expiry's quote. For 159901/
159915 (SZSE tier), Greeks are Black-Scholes derived, not vendor-validated
(see reference_data_sources memory) -- treat GEX there as lower-confidence,
same caveat already carried for iv_pct on that tier.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

import consolidated_store as cs
import data_store as ds

LEGACY_UNDERLYINGS = {"510050": "50ETF", "510300": "300ETF"}


@lru_cache(maxsize=8)
def _raw_chain(underlying: str) -> pd.DataFrame:
    """Full chain (every strike, every expiry) -- deliberately NOT
    restricted to near-month like the _prepped_chain helpers elsewhere."""
    if underlying in LEGACY_UNDERLYINGS:
        chain = ds.load_chain(underlying)
    else:
        chain = cs.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    return chain[chain[ds.C_CLOSE] > 0]


@lru_cache(maxsize=8)
def _spot_per_date(underlying: str) -> pd.Series:
    """One spot price per trading date, preferring the nearest listed
    expiry -- chain is sorted [date, expiry, cp, strike], so the first
    non-null C_UND_CLOSE per date is the nearest-expiry quote."""
    chain = _raw_chain(underlying)
    spot = chain.dropna(subset=[ds.C_UND_CLOSE])
    return spot.groupby(ds.C_DATE)[ds.C_UND_CLOSE].first()


@lru_cache(maxsize=8)
def net_gex_series(underlying: str) -> pd.Series:
    """Net dealer gamma exposure: RMB P&L per 1% underlying move, signed
    under the dealer-long-calls/short-puts convention (see module
    docstring). spot^2 x 0.01 x MULTIPLIER x sum(OI_call*gamma_call -
    OI_put*gamma_put) across the full chain for that date."""
    chain = _raw_chain(underlying)
    # Guard against a dataset with no per-contract gamma at all. Without this,
    # the .fillna(0) below turns an entirely absent Greek into a legitimate-
    # looking all-zero exposure series, whose rolling percentile rank is a
    # CONSTANT ~0.5 -- so gex_pct_series would never cross any threshold and a
    # backtest over it would silently produce zero trades (or, worse, a full
    # sweep grid of meaningless numbers) rather than failing. This is the
    # actual state of the CFFEX index options (000300/000852/000016) in the
    # 买断日更 consolidated dataset: C_GAMMA/C_VEGA/C_THETA are 0% populated
    # while C_DELTA is ~95%, so nothing else about the chain looks wrong.
    if chain[ds.C_GAMMA].notna().sum() == 0:
        raise ValueError(
            f"no per-contract gamma for {underlying!r} -- ds.C_GAMMA is entirely "
            f"null in this dataset, so dealer gamma exposure is not computable. "
            f"(Known for the CFFEX index options; see put_call_ratio.py for an "
            f"OI/volume-based positioning signal that does work there.)")
    signed = np.where(chain[ds.C_CP] == "C", 1.0, -1.0)
    contrib = signed * chain[ds.C_OI].fillna(0) * chain[ds.C_GAMMA].fillna(0)
    daily = contrib.groupby(chain[ds.C_DATE]).sum()

    spot = _spot_per_date(underlying)
    idx = daily.index.intersection(spot.index)
    gex = (spot.loc[idx] ** 2) * 0.01 * ds.MULTIPLIER * daily.loc[idx]
    return gex.sort_index()


@lru_cache(maxsize=8)
def gex_pct_series(underlying: str) -> pd.Series:
    """Rolling 252-trading-day percentile rank of net_gex_series -- same
    transform as rv_pct_series/iv_pct_series so it drops into the existing
    threshold-sweep machinery unchanged."""
    return net_gex_series(underlying).rolling(252, min_periods=60).rank(pct=True)


@lru_cache(maxsize=8)
def gex_magnitude_pct_series(underlying: str) -> pd.Series:
    """Rolling 252-day percentile of |net GEX| -- total gamma "pinning
    intensity" regardless of sign. Fallback signal in case the signed
    dealer-convention direction (see module docstring) turns out backwards
    or too noisy in the backtest."""
    return net_gex_series(underlying).abs().rolling(252, min_periods=60).rank(pct=True)
