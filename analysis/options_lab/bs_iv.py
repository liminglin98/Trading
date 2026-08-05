"""Black-Scholes European option pricing, implied-vol inversion, and Greeks.

SSE/SZSE ETF options are European-exercise and pay the option holder no
dividend, so plain Black-Scholes (no early-exercise premium, no dividend
yield term) applies exactly -- this isn't an approximation for this market.
Risk-free rate convention (flat 2%, no dividend yield) matches updater.py's
bs_rho, which already computes one Greek this way (rho, back-filled from
Sina's quoted IV since Sina doesn't publish it) for the live 510050/510300
feed. Used here to back out IV itself, for underlyings whose only chain
source (Data/买断日更/consolidated/, see consolidated_store.py) has Greeks
but no IV column.

The bs_delta/gamma/vega/theta functions below are cross-validated (2026-07-28,
analysis/options_lab/backfill_csv_greeks.py) against Sina's real quoted
Greeks on 200 liquid, non-stale 510050 rows: plain, *unscaled* textbook
formulas (T in calendar days/365, no per-day or per-percent rescaling
anywhere) reproduce them with median real/computed ratios of 0.97-1.00 for
all five Greeks including rho -- so despite "Theta" and "Vega" often being
quoted per-day or per-1%-vol-move on other platforms, this data's convention
is the plain per-year/per-unit-vol textbook value, no conversion needed.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq

RISK_FREE_DEFAULT = 0.02
_N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
_n = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bs_price(cp: str, s: float, k: float, t: float, sigma: float,
            r: float = RISK_FREE_DEFAULT) -> float:
    """European option price. t in years."""
    if t <= 0 or sigma <= 0:
        return max(s - k, 0.0) if cp == "C" else max(k - s, 0.0)
    d1 = (math.log(s / k) + (r + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if cp == "C":
        return s * _N(d1) - k * math.exp(-r * t) * _N(d2)
    return k * math.exp(-r * t) * _N(-d2) - s * _N(-d1)


def implied_vol(cp: str, s: float, k: float, t: float, price: float,
                r: float = RISK_FREE_DEFAULT, lo: float = 1e-4, hi: float = 5.0
                ) -> float:
    """Invert bs_price for sigma via Brent's method. NaN on invalid inputs or
    a price outside the [intrinsic, bs_price(hi)] no-arbitrage band -- a
    stale/illiquid quote shouldn't be forced to a spurious vol."""
    if not (s > 0 and k > 0 and t > 0 and price > 0):
        return float("nan")
    intrinsic = max(s - k, 0.0) if cp == "C" else max(k - s, 0.0)
    if price < intrinsic - 1e-6:
        return float("nan")
    f = lambda sigma: bs_price(cp, s, k, t, sigma, r) - price
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    return brentq(f, lo, hi, xtol=1e-6)


def _d1_d2(s: float, k: float, t: float, sigma: float, r: float) -> tuple[float, float]:
    d1 = (math.log(s / k) + (r + sigma * sigma / 2) * t) / (sigma * math.sqrt(t))
    return d1, d1 - sigma * math.sqrt(t)


def bs_delta(cp: str, s: float, k: float, t: float, sigma: float,
            r: float = RISK_FREE_DEFAULT) -> float:
    if not (s > 0 and k > 0 and t > 0 and sigma > 0):
        return float("nan")
    d1, _ = _d1_d2(s, k, t, sigma, r)
    return _N(d1) if cp == "C" else _N(d1) - 1


def bs_gamma(s: float, k: float, t: float, sigma: float,
            r: float = RISK_FREE_DEFAULT) -> float:
    """Same for call and put."""
    if not (s > 0 and k > 0 and t > 0 and sigma > 0):
        return float("nan")
    d1, _ = _d1_d2(s, k, t, sigma, r)
    return _n(d1) / (s * sigma * math.sqrt(t))


def bs_vega(s: float, k: float, t: float, sigma: float,
           r: float = RISK_FREE_DEFAULT) -> float:
    """Same for call and put. Per 1.0 (100 vol points) change in sigma --
    validated against real quotes at this scale, no /100 rescaling."""
    if not (s > 0 and k > 0 and t > 0 and sigma > 0):
        return float("nan")
    d1, _ = _d1_d2(s, k, t, sigma, r)
    return s * _n(d1) * math.sqrt(t)


def bs_theta(cp: str, s: float, k: float, t: float, sigma: float,
            r: float = RISK_FREE_DEFAULT) -> float:
    """Per year -- validated against real quotes at this scale, no /365
    rescaling despite "theta" often being quoted per-day elsewhere."""
    if not (s > 0 and k > 0 and t > 0 and sigma > 0):
        return float("nan")
    d1, d2 = _d1_d2(s, k, t, sigma, r)
    common = -(s * _n(d1) * sigma) / (2 * math.sqrt(t))
    if cp == "C":
        return common - r * k * math.exp(-r * t) * _N(d2)
    return common + r * k * math.exp(-r * t) * _N(-d2)


def bs_rho(cp: str, s: float, k: float, t: float, sigma: float,
          r: float = RISK_FREE_DEFAULT) -> float:
    """Per 1.0 (100 percentage points) change in r -- same convention
    updater.py's own bs_rho already used for the live feed."""
    if not (s > 0 and k > 0 and t > 0 and sigma > 0):
        return float("nan")
    _, d2 = _d1_d2(s, k, t, sigma, r)
    if cp == "C":
        return k * t * math.exp(-r * t) * _N(d2)
    return -k * t * math.exp(-r * t) * _N(-d2)
