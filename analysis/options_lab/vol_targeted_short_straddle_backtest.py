"""Vol-targeted de-risking of the quarterly ATM short straddle
(quarterly_atm_short_straddle.py), tested against real 510050 + 510300
history.

Why this instead of another hedge overlay: external research (this
session) found no strong evidence that any single cross-asset instrument
reliably hedges a short-vol book -- and three overlay ideas already tried
here (domestic QVIX-triggered long straddle, global S&P-triggered
overlay, crude-oil straddle -- see commodity_straddle_overlay_backtest.py)
were all rejected. The one approach with real empirical backing in the
literature is architecturally different: rules-based position sizing /
vol-targeting / drawdown stop-outs applied to the book itself, not a
second position. quarterly_atm_short_straddle.py's run() has neither an
entry filter nor a stop-loss today -- it sells every quarter regardless
of conditions and holds every position to expiry no matter how far it
moves. This script tests two ways of closing that specific gap:

  * entry gate: skip the quarter's roll if QVIX is already elevated at
    entry (don't sell into an already-stressed regime).
  * stop-loss: close the position early if QVIX spikes while it's held,
    instead of always riding it out to expiry/intrinsic settlement.

Signal: reuses the QVIX-crossing convention already established in
analysis/macro_model/qvix_spike_put_backtest.py -- QVIX crossing above
its own trailing 250-day mean + k*std (a crossing event, not "sustained
above"), swept over k in {1.0, 1.5, 2.0}, each underlying using its own
QVIX (510050<-qvix_50, 510300<-qvix_300). That script uses the same
signal to trigger buying puts (a directional bet); this reuses only the
signal construction for a different action entirely -- gating/exiting
the *existing* short straddle, not adding a new position. This is not a
re-test of the deleted shock_overlay_backtest.py's QVIX-triggered hedge
overlay (which added a long straddle on trigger and made things worse) --
here nothing new is bought, exposure to the same book is just reduced.

Usage:
    python vol_targeted_short_straddle_backtest.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import data_store as ds
import margin as mg
from quarterly_atm_short_straddle import _atm_strike, _intrinsic, _leg_row, _target_expiries

REPO = Path(__file__).resolve().parents[2]
QVIX_DIR = REPO / "Data" / "akshare" / "dashboard"
OUT_DIR = Path(__file__).resolve().parent / "out"

QVIX_FILES = {"510050": "qvix_50.csv", "510300": "qvix_300.csv"}
K_SWEEP = [1.0, 1.5, 2.0]
LOOKBACK = 250


def load_qvix(underlying: str) -> pd.Series:
    df = pd.read_csv(QVIX_DIR / QVIX_FILES[underlying], parse_dates=["date"])
    return df.set_index("date")["value"].sort_index()


def qvix_threshold(qvix: pd.Series, k: float) -> pd.Series:
    roll_mean = qvix.rolling(LOOKBACK).mean()
    roll_std = qvix.rolling(LOOKBACK).std()
    return roll_mean + k * roll_std


def run(underlying: str, qvix: pd.Series, k: float | None,
       entry_gate: bool, stop_loss: bool) -> dict:
    """k=None (or both flags False) reproduces the unmodified base
    strategy exactly -- no entry filter, no stop-loss, same as
    quarterly_atm_short_straddle.py's run(roll_freq="quarterly")."""
    thresh = qvix_threshold(qvix, k) if k is not None else None

    df = ds.load_chain(underlying)
    df = df[~df[ds.C_NAME].astype(str).str.endswith("A")]
    df = df[df[ds.C_CLOSE] > 0]
    dates = sorted(df[ds.C_DATE].unique())
    by_date = dict(tuple(df.groupby(ds.C_DATE)))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_pos: dict | None = None
    realized = 0.0
    # Once a quarter's entry is gated off, commit to skipping the *whole*
    # quarter (don't retry daily until stress happens to clear -- that
    # would just delay entry by a few days within the same target expiry,
    # not actually sit the quarter out, since _target_expiries doesn't
    # advance to the next quarter until the current one's expiry passes).
    skip_until: pd.Timestamp | None = None

    def is_stressed(dt) -> bool:
        if thresh is None or dt not in qvix.index or dt not in thresh.index:
            return False
        t = thresh.loc[dt]
        return bool(pd.notna(t) and qvix.loc[dt] > t)

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        if open_pos is not None:
            call = day[day[ds.C_CODE] == open_pos["call_code"]]
            put = day[day[ds.C_CODE] == open_pos["put_code"]]
            expired = pd.Timestamp(dt) >= open_pos["expiry"]
            triggered_stop = stop_loss and is_stressed(pd.Timestamp(dt))
            if expired or call.empty or put.empty:
                call_px = _intrinsic("C", open_pos["strike"], spot)
                put_px = _intrinsic("P", open_pos["strike"], spot)
                exit_reason = "expiry"
            elif triggered_stop:
                call_px = float(call[ds.C_CLOSE].iloc[0])
                put_px = float(put[ds.C_CLOSE].iloc[0])
                exit_reason = "stop_loss"
            else:
                call_px = float(call[ds.C_CLOSE].iloc[0])
                put_px = float(put[ds.C_CLOSE].iloc[0])
                mark = -(call_px + put_px)
                pnl = mark - open_pos["cost"]
                equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl})
                continue

            mark = -(call_px + put_px)
            pnl = mark - open_pos["cost"]
            realized += pnl
            trades.append({
                "entry_date": open_pos["entry_date"], "expiry": open_pos["expiry"],
                "exit_date": dt, "exit_reason": exit_reason, "strike": open_pos["strike"],
                "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                "call_entry_px": open_pos["call_px"], "put_entry_px": open_pos["put_px"],
                "credit": -open_pos["cost"], "pnl": pnl,
                "days_held": (pd.Timestamp(dt) - open_pos["entry_date"]).days,
                "margin_yuan": open_pos["margin"],
            })
            open_pos = None
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
            continue

        # ---- entry
        if skip_until is not None:
            if pd.Timestamp(dt) <= skip_until:
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
                continue
            skip_until = None

        qexp = _target_expiries(day, "quarterly")
        if not qexp:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
            continue
        expiry = pd.Timestamp(qexp[0])

        if entry_gate and is_stressed(pd.Timestamp(dt)):
            skip_until = expiry  # sit out the rest of this quarter entirely
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
            continue

        strike = _atm_strike(day, expiry, spot)
        call_row = _leg_row(day, expiry, "C", strike) if strike is not None else None
        put_row = _leg_row(day, expiry, "P", strike) if strike is not None else None
        if call_row is None or put_row is None:
            equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})
            continue

        call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
        legs_df = pd.DataFrame([
            {"expiry": expiry, "cp": "C", "strike": strike, "direction": -1,
             "qty": 1, "entry_price": call_px},
            {"expiry": expiry, "cp": "P", "strike": strike, "direction": -1,
             "qty": 1, "entry_price": put_px},
        ])
        margin_yuan, _ = mg.compute_deposit(legs_df, spot)
        open_pos = {
            "entry_date": pd.Timestamp(dt), "entry_spot": spot, "expiry": expiry,
            "strike": strike, "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
            "call_px": call_px, "put_px": put_px, "cost": -(call_px + put_px),
            "margin": margin_yuan,
        }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0})

    return {"trades": pd.DataFrame(trades), "equity": pd.DataFrame(equity_rows)}


def _mdd_yuan(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    eq = (trades["pnl"] * ds.MULTIPLIER).cumsum()
    return float((eq - eq.cummax()).min())


def summarize(underlying: str, label: str, res: dict) -> dict:
    trades = res["trades"]
    if trades.empty:
        return {"underlying": underlying, "variant": label, "n_trades": 0,
                "total_pnl": 0.0, "worst_quarter": 0.0, "max_drawdown": 0.0,
                "n_skipped_entries": None, "n_stop_losses": None}
    pnl_yuan = trades["pnl"] * ds.MULTIPLIER
    n_stop = (trades["exit_reason"] == "stop_loss").sum() if "exit_reason" in trades else 0
    return {
        "underlying": underlying, "variant": label, "n_trades": len(trades),
        "total_pnl": pnl_yuan.sum(), "worst_quarter": pnl_yuan.min(),
        "max_drawdown": _mdd_yuan(trades), "n_stop_losses": n_stop,
    }


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    all_rows = []
    for underlying in QVIX_FILES:
        qvix = load_qvix(underlying)
        print(f"\n=== {underlying} (QVIX {qvix.index.min().date()} -> {qvix.index.max().date()}) ===")

        base = run(underlying, qvix, k=None, entry_gate=False, stop_loss=False)
        all_rows.append(summarize(underlying, "base (no rule)", base))
        base["trades"].to_csv(OUT_DIR / f"vol_targeted_straddle_{underlying}_base.csv", index=False)

        for k in K_SWEEP:
            for label, gate, stop in (
                (f"entry_gate k={k}", True, False),
                (f"stop_loss k={k}", False, True),
                (f"both k={k}", True, True),
            ):
                res = run(underlying, qvix, k=k, entry_gate=gate, stop_loss=stop)
                all_rows.append(summarize(underlying, label, res))
                fname = f"vol_targeted_straddle_{underlying}_{label.replace(' ', '_').replace('=', '')}.csv"
                res["trades"].to_csv(OUT_DIR / fname, index=False)

    summary = pd.DataFrame(all_rows)
    pd.set_option("display.width", 140)
    for c in ("total_pnl", "worst_quarter", "max_drawdown"):
        summary[c] = summary[c].round(0)
    print("\n" + summary.to_string(index=False))
    summary.to_csv(OUT_DIR / "vol_targeted_straddle_summary.csv", index=False)
    print(f"\nSummary saved to {OUT_DIR / 'vol_targeted_straddle_summary.csv'}")
