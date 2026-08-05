"""Deposit (margin) estimation for simulated option combos.

Implements the official SSE/SZSE exchange-minimum formula for a single naked
(short) option, and the standard reduced-margin treatment for a recognized
vertical spread (defined-risk: one short + one long, same expiry/cp,
different strikes). Anything else that still carries a short leg -- short
straddles/strangles, ratio spreads, unmatched legs -- falls back to summing
the naked margin of each short leg. That's deliberately conservative: broker
combo-margin rules for those shapes vary and aren't asserted here, so this
tool never *understates* the capital you'd actually need to post.

All formulas operate on the option's own current mark price and the
underlying's current spot as a stand-in for the "previous settlement price"
/ "previous close" the real exchange formula uses -- fine for a pre-trade
sizing estimate, not exact for tomorrow's actual margin call.
"""
from __future__ import annotations

import pandas as pd

import data_store as ds

LEG_COLS = ["expiry", "cp", "strike", "direction", "qty", "entry_price"]


def naked_margin_per_unit(cp: str, strike: float, spot: float, mark: float) -> float:
    """Exchange-minimum opening margin for one naked short option, per
    contract unit (i.e. NOT yet multiplied by the contract multiplier)."""
    if cp == "C":
        otm = max(strike - spot, 0.0)
        return mark + max(0.12 * spot - otm, 0.07 * spot)
    otm = max(spot - strike, 0.0)
    return min(mark + max(0.12 * spot - otm, 0.07 * strike), strike)


def _explode(records: list[dict]) -> list[dict]:
    """One row per contract unit, so mismatched qty pairs partially."""
    out = []
    for r in records:
        for _ in range(int(r["qty"])):
            out.append({**r, "qty": 1})
    return out


def compute_deposit(legs: pd.DataFrame, spot: float,
                    multiplier: float = ds.MULTIPLIER,
                    buffer: float = 1.0) -> tuple[float, pd.DataFrame]:
    """legs: rows with expiry, cp, strike, direction (+1/-1), qty,
    entry_price (used as the mark-price stand-in). Groups by (expiry, cp),
    pairs opposite-direction legs into vertical spreads where possible, and
    naked-margins any leftover short legs. Returns (total_yuan_with_buffer,
    breakdown_df) -- breakdown margins are pre-buffer, per group."""
    rows = []
    total = 0.0
    if legs.empty:
        return 0.0, pd.DataFrame(
            columns=["expiry", "type", "detail", "margin_yuan"])

    for (expiry, cp), grp in legs.groupby(["expiry", "cp"]):
        shorts = _explode(grp[grp["direction"] == -1].to_dict("records"))
        longs = _explode(grp[grp["direction"] == 1].to_dict("records"))
        shorts.sort(key=lambda r: r["strike"])
        longs.sort(key=lambda r: r["strike"])

        paired = min(len(shorts), len(longs))
        for i in range(paired):
            s, l = shorts[i], longs[i]
            if s["strike"] == l["strike"]:
                continue  # fully offsetting, no residual risk, no margin
            width = abs(s["strike"] - l["strike"]) * multiplier
            net_credit = (s["entry_price"] - l["entry_price"]) * multiplier
            m = max(width - net_credit, 0.0) if net_credit > 0 else 0.0
            total += m
            rows.append({"expiry": expiry, "type": "vertical spread",
                        "detail": f"{cp} short {s['strike']:g} / "
                                  f"long {l['strike']:g}",
                        "margin_yuan": round(m, 0)})

        for s in shorts[paired:]:
            m = naked_margin_per_unit(cp, s["strike"], spot,
                                      s["entry_price"]) * multiplier
            total += m
            rows.append({"expiry": expiry, "type": "naked short",
                        "detail": f"{cp} short {s['strike']:g}",
                        "margin_yuan": round(m, 0)})

    breakdown = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["expiry", "type", "detail", "margin_yuan"])
    return total * buffer, breakdown
