"""cffex_margin.py -- margin (deposit) estimation for CFFEX index-option
short legs (MO/IO/HO). Parallel to margin.py, not a replacement for it:
margin.py implements the SSE/SZSE exchange-minimum formula, which is wrong
here on two counts -- wrong multiplier (ds.MULTIPLIER=10000 is the SSE ETF
options multiplier) and wrong formula shape (these are futures-style
margined, cash-settled index options, not premium-paid ETF options).

Formula (CFFEX 股指期权合约交易细则, seller's opening margin per contract):
  call: settle*mult + max(spot*mult*margin_ratio - otm_call, min_guarantee*spot*mult*margin_ratio)
  put:  settle*mult + max(spot*mult*margin_ratio - otm_put,  min_guarantee*strike*mult*margin_ratio)
  otm_call = max(strike-spot, 0)*mult
  otm_put  = max(spot-strike, 0)*mult

Multiplier: CFFEX contract spec, ¥100 per index point for MO/IO/HO alike.

CAVEAT -- read before trusting any return-on-margin number this produces:
MARGIN_PARAMS below are CSI1000 (MO)'s 2022-07-18 LAUNCH-DATE parameters
(margin_ratio=15%, min_guarantee=0.5), held constant across this repo's
entire 2022-2026 backtest window. CFFEX is known to periodically revise
these coefficients; no verified historical schedule of MO-specific
revisions was locatable. A static value likely UNDERSTATES margin (and so
OVERSTATES return-on-margin) in any subperiod where the exchange actually
raised the ratio. Every return-on-margin/leverage figure computed from
this module should be captioned as approximate for this specific reason --
a real, acknowledged data gap, not a rounding error.

No combo/portfolio-margin discount is modeled for multi-leg structures
(e.g. a risk reversal's offsetting short-put + long-call) -- each short
leg is margined independently and naked, matching margin.py's own
conservative philosophy for un-recognized combo shapes. IO/HO have no
MARGIN_PARAMS entry -- explicit TODO, not guessed; only needed if this is
extended to those underlyings as a robustness check.
"""
from __future__ import annotations

import pandas as pd

MULTIPLIER = 100  # yuan per index point -- CFFEX contract spec (MO/IO/HO)

MARGIN_PARAMS = {
    "000852": {"margin_ratio": 0.15, "min_guarantee": 0.5},  # MO, 2022-07-18 launch params
    # "000300": {...},  # IO -- not sourced, TODO before trusting IO results
    # "000016": {...},  # HO -- not sourced, TODO before trusting HO results
}

LEG_COLS = ["expiry", "cp", "strike", "direction", "qty", "entry_price"]


def cffex_index_option_margin(cp: str, strike: float, spot: float, settle_price: float,
                              margin_ratio: float, min_guarantee: float,
                              multiplier: float = MULTIPLIER) -> float:
    """CFFEX seller's opening margin for one short index-option contract."""
    if cp == "C":
        otm = max(strike - spot, 0.0) * multiplier
        return settle_price * multiplier + max(
            spot * multiplier * margin_ratio - otm,
            min_guarantee * spot * multiplier * margin_ratio)
    otm = max(spot - strike, 0.0) * multiplier
    return settle_price * multiplier + max(
        spot * multiplier * margin_ratio - otm,
        min_guarantee * strike * multiplier * margin_ratio)


def _explode(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        for _ in range(int(r["qty"])):
            out.append({**r, "qty": 1})
    return out


def compute_deposit_cffex(legs: pd.DataFrame, spot: float, underlying: str,
                          multiplier: float = MULTIPLIER, buffer: float = 1.0
                          ) -> tuple[float, pd.DataFrame]:
    """legs: rows with expiry, cp, strike, direction (+1/-1), qty, entry_price
    -- entry_price is used as the settle-price stand-in (deliberately, per
    the module docstring: this formula calls for settle, unlike
    margin.compute_deposit which uses the mark/close). Only short legs are
    margined -- a long leg is a pure debit with no margin requirement -- and
    no cross-leg netting is applied even for an offsetting shape like a
    risk reversal (conservative, consistent with margin.py's own
    unrecognized-combo-shape philosophy)."""
    if underlying not in MARGIN_PARAMS:
        raise ValueError(f"No CFFEX margin params for {underlying!r} -- see module docstring TODO")
    params = MARGIN_PARAMS[underlying]

    if legs.empty:
        return 0.0, pd.DataFrame(columns=["expiry", "cp", "strike", "margin_yuan"])

    shorts = legs[legs["direction"] == -1]
    rows = []
    total = 0.0
    for r in _explode(shorts.to_dict("records")):
        m = cffex_index_option_margin(r["cp"], r["strike"], spot, r["entry_price"],
                                      params["margin_ratio"], params["min_guarantee"],
                                      multiplier)
        total += m
        rows.append({"expiry": r["expiry"], "cp": r["cp"], "strike": r["strike"],
                    "margin_yuan": round(m, 0)})

    breakdown = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["expiry", "cp", "strike", "margin_yuan"])
    return total * buffer, breakdown
