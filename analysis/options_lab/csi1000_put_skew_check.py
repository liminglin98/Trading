"""csi1000_put_skew_check.py -- verify the claim that CSI1000 (MO) index
options carry a persistent put-side IV premium from quant funds hedging
small-cap holdings.

Two sub-claims, tested separately:
  1. Put IV is persistently richer than call IV on CSI1000 options -- directly
     testable via Black-Scholes IV inversion on the option chain.
  2. That premium is specific to small-cap hedging demand, not just the
     generic leverage/crash-fear skew every equity index option shows --
     tested by comparing CSI1000 (000852/MO) against CSI300 (000300/IO) and
     SSE50 (000016/HO) as controls. If all three show similar skew, it's
     ordinary index-option skew, not CSI1000-specific hedging pressure.

Reuses consolidated_store.py's chain loader and parity-forward spot
(_parity_forward) and the same per-leg Black-Scholes inversion pattern as
_cp_avg_iv -- that function already inverts both legs' IV and then averages
them together, discarding the sign; this script keeps both legs instead of
averaging, which is the whole skew signal.

Two skew metrics per day, near-month expiry (DTE>=7, same selection rule as
atm_iv_series/term_structure_series):
  - ATM skew: nearest-strike-to-spot put IV minus call IV.
  - OTM (risk-reversal) skew: ~5%-OTM put IV minus ~5%-OTM call IV -- the
    more standard "crash insurance" metric, since hedgers buying downside
    protection concentrate demand in OTM puts specifically.

Cross-checked against put_call_ratio.py's OI-based put/call ratio (a
non-IV, pure positioning signal) for the same three underlyings.

Usage: python csi1000_put_skew_check.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import bs_iv
import consolidated_store as cs
import data_store as ds
import put_call_ratio as pcr

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"

UNDERLYINGS = {"000852": "CSI1000 (MO)", "000300": "CSI300 (IO)", "000016": "SSE50 (HO)"}
DTE_MIN = 7
OTM_PCT = 0.05


def _leg_iv(day: pd.DataFrame, cp: str, expiry, strike: float, spot: float) -> float:
    rows = day[(day[ds.C_CP] == cp) & (day[ds.C_EXPIRY] == expiry) & (day[ds.C_STRIKE] == strike)]
    if rows.empty or rows[ds.C_CLOSE].iloc[0] <= 0:
        return float("nan")
    row = rows.iloc[0]
    return bs_iv.implied_vol(cp, spot, strike, float(row[ds.C_DTE]) / 365.0, float(row[ds.C_CLOSE]))


def _near_expiry(day_calls: pd.DataFrame, dte_min: int):
    dtes = day_calls.drop_duplicates(ds.C_EXPIRY).set_index(ds.C_EXPIRY)[ds.C_DTE]
    eligible = dtes[dtes >= dte_min]
    return eligible.idxmin() if not eligible.empty else None


def skew_series(underlying: str, dte_min: int = DTE_MIN, otm_pct: float = OTM_PCT) -> pd.DataFrame:
    """Daily ATM and OTM put-minus-call IV skew (decimal vol, positive = put
    richer) for the near-month expiry. Same expiry/strike selection
    convention as atm_iv_series (nearest expiry with DTE>=dte_min, nearest
    strike to the parity-forward spot for ATM); OTM strikes are the nearest
    listed strike to spot*(1-otm_pct) / spot*(1+otm_pct)."""
    chain = cs.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    chain = chain[chain[ds.C_CLOSE] > 0]
    calls = chain[(chain[ds.C_CP] == "C") & chain[ds.C_UND_CLOSE].notna()]

    by_date = dict(tuple(chain.groupby(ds.C_DATE)))
    rows = []
    for dt, day_calls in calls.groupby(ds.C_DATE):
        near_exp = _near_expiry(day_calls, dte_min)
        if near_exp is None:
            continue
        exp_calls = day_calls[day_calls[ds.C_EXPIRY] == near_exp]
        spot = float(exp_calls[ds.C_UND_CLOSE].iloc[0])
        strikes = sorted(exp_calls[ds.C_STRIKE].unique())
        if not strikes:
            continue

        atm_k = min(strikes, key=lambda k: abs(k - spot))
        otm_put_k = min(strikes, key=lambda k: abs(k - spot * (1 - otm_pct)))
        otm_call_k = min(strikes, key=lambda k: abs(k - spot * (1 + otm_pct)))

        day = by_date[dt]
        rows.append({
            "date": dt, "spot": spot, "dte": float(exp_calls[ds.C_DTE].iloc[0]),
            "atm_call_iv": _leg_iv(day, "C", near_exp, atm_k, spot),
            "atm_put_iv": _leg_iv(day, "P", near_exp, atm_k, spot),
            "otm_call_iv": _leg_iv(day, "C", near_exp, otm_call_k, spot),
            "otm_put_iv": _leg_iv(day, "P", near_exp, otm_put_k, spot),
        })

    out = pd.DataFrame(rows).set_index("date").sort_index()
    out["atm_skew"] = out["atm_put_iv"] - out["atm_call_iv"]
    out["otm_skew"] = out["otm_put_iv"] - out["otm_call_iv"]
    return out


def summarize(skew: pd.DataFrame, label: str) -> dict:
    atm, otm = skew["atm_skew"].dropna(), skew["otm_skew"].dropna()
    return {
        "underlying": label,
        "n_days": len(skew),
        "atm_pct_put_richer": float((atm > 0).mean()) if len(atm) else float("nan"),
        "atm_mean_skew_volpts": float(atm.mean() * 100) if len(atm) else float("nan"),
        "atm_median_skew_volpts": float(atm.median() * 100) if len(atm) else float("nan"),
        "otm_pct_put_richer": float((otm > 0).mean()) if len(otm) else float("nan"),
        "otm_mean_skew_volpts": float(otm.mean() * 100) if len(otm) else float("nan"),
        "otm_median_skew_volpts": float(otm.median() * 100) if len(otm) else float("nan"),
    }


def main():
    OUT_DIR.mkdir(exist_ok=True)
    skews, summaries = {}, []
    for code, label in UNDERLYINGS.items():
        print(f"Computing skew for {label} ({code}) ...")
        skews[code] = skew_series(code)
        summaries.append(summarize(skews[code], label))

    summary_df = pd.DataFrame(summaries)
    pd.set_option("display.width", 160)
    print("\n=== Put-Call IV Skew Summary (positive = put IV richer than call IV) ===")
    print(summary_df.to_string(index=False))

    pc_rows = []
    for code, label in UNDERLYINGS.items():
        oi_ratio = pcr.pc_oi_ratio_series(code)
        pc_rows.append({
            "underlying": label,
            "median_pc_oi_ratio": float(oi_ratio.median()) if len(oi_ratio) else float("nan"),
            "pct_days_oi_ratio_gt1": float((oi_ratio > 1).mean()) if len(oi_ratio) else float("nan"),
        })
    pc_df = pd.DataFrame(pc_rows)
    print("\n=== OI-based Put/Call Ratio (median; >1 = more put OI outstanding than call OI) ===")
    print(pc_df.to_string(index=False))

    combined = []
    for code, label in UNDERLYINGS.items():
        d = skews[code].reset_index()
        d["underlying"] = label
        combined.append(d)
    csv_path = OUT_DIR / "csi1000_put_skew_check.csv"
    pd.concat(combined, ignore_index=True).to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for code, label in UNDERLYINGS.items():
        s = (skews[code]["otm_skew"].dropna() * 100).rolling(20, min_periods=5).mean()
        axes[0].plot(s.index, s.values, label=label)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("vol pts (20d avg)")
    axes[0].set_title(f"{int(OTM_PCT*100)}%-OTM Put IV minus {int(OTM_PCT*100)}%-OTM Call IV")
    axes[0].legend()

    for code, label in UNDERLYINGS.items():
        s = (skews[code]["atm_skew"].dropna() * 100).rolling(20, min_periods=5).mean()
        axes[1].plot(s.index, s.values, label=label)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("vol pts (20d avg)")
    axes[1].set_title("ATM Put IV minus ATM Call IV")
    axes[1].legend()

    fig.tight_layout()
    png_path = OUT_DIR / "csi1000_put_skew_check.png"
    fig.savefig(png_path, dpi=120)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
