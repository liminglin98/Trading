# -*- coding: utf-8 -*-
"""
test_rules_dryrun.py
====================
Offline dry-run of the stop/roll rule engine in tracking.py. Exercises the PURE
logic (no network / no akshare chain) on a SYNTHETIC saved butterfly, and the
schema migration + save/load/status/rule/lineage round-trip against TEMP CSVs so
your real saved_positions.csv / position_tracking.csv are never touched.

It deliberately does NOT call build_roll_suggestion() or snapshot_all_open() end
to end -- those need the live SSE chain; run them from the app / pipeline with a
real position. Their pure sub-parts (detect_butterfly, position_max_profit,
evaluate_rules) ARE covered here.

RUN IN A NORMAL PYTHON ENV from inside macro_model\:
    python test_rules_dryrun.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import tracking as TK


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def main():
    # redirect the store to a throwaway dir so real journals are untouched
    tmp = Path(tempfile.mkdtemp(prefix="tracking_dryrun_"))
    TK.SAVE_PATH = tmp / "saved_positions.csv"
    TK.TRACK_PATH = tmp / "position_tracking.csv"
    print(f"[temp store] {tmp}")

    # --- synthetic long call butterfly 2.50 / 2.60 / 2.70, net debit 0.015 ---
    legs = [dict(type="C", strike=2.50, qty=1.0,  entry=0.12),
            dict(type="C", strike=2.60, qty=-2.0, entry=0.055),
            dict(type="C", strike=2.70, qty=1.0,  entry=0.005)]
    cost = sum(l["qty"] * l["entry"] for l in legs)
    mp = TK.position_max_profit(legs)
    print(f"net debit = {cost:+.4f}   max_profit = {mp:+.4f}")
    assert approx(cost, 0.015), cost
    assert approx(mp, 0.085), mp           # (2.60-2.50) - 0.015

    geom = TK.detect_butterfly(legs)
    assert geom and geom["body"] == 2.60 and geom["wing_low"] == 2.50 \
        and geom["wing_high"] == 2.70 and geom["cp"] == "C", geom
    assert TK.detect_butterfly(legs[:2]) is None
    print("detect_butterfly OK:", geom)

    # --- rule evaluation (two fractions on the combination P&L; roll on DTE) ---
    R = dict(tp_frac=0.5, sl_frac=0.5, roll_dte=14)
    assert TK.evaluate_rules(0.01, cost, mp, 30, R)["status"] == "open"
    assert TK.evaluate_rules(0.05, cost, mp, 30, R)["status"] == "hit_tp"   # >= 0.5*0.085
    assert TK.evaluate_rules(-0.01, cost, mp, 30, R)["status"] == "hit_sl"  # <= -0.5*0.015
    assert TK.evaluate_rules(0.0, cost, mp, 10, R)["roll"] is True
    assert TK.evaluate_rules(0.0, cost, mp, 30, R)["roll"] is False
    Roff = dict(tp_frac=np.nan, sl_frac=np.nan, roll_dte=np.nan)
    assert TK.evaluate_rules(99, cost, mp, 1, Roff)["status"] == "open"
    assert TK.evaluate_rules(99, cost, mp, 1, Roff)["roll"] is False
    print("evaluate_rules OK (TP frac of max gain, SL frac of max loss, roll, all-off)")

    # --- schema migration: old-schema frame gains new cols with safe defaults ---
    old = pd.DataFrame([dict(saved_at="x", name="legacy", position_id="legacy @ t",
                             side="Buy", type="C", strike=2.6, qty=1, entry=0.05,
                             expiry="2026-07-22", spot_at_save=2.6, dte_at_save=29,
                             cost_basis=0.015, mark_at_save=0.015, pnl_at_save=0.0)])
    mig = TK._ensure_cols(old.copy(), TK.SAVE_COLS, TK._SAVE_DEFAULTS)
    for c in TK._SAVE_RULES:
        assert c in mig.columns, c
    assert mig["status"].iloc[0] == "open"
    assert not np.isfinite(pd.to_numeric(mig["tp_frac"].iloc[0], errors="coerce"))
    print("schema migration OK (old rows -> status=open, rules=NaN)")

    # --- save/load + rules + status round-trip against temp store ---
    legs_df = pd.DataFrame([dict(side="Buy", type="C", strike=2.50, qty=1, entry=0.12),
                            dict(side="Sell", type="C", strike=2.60, qty=2, entry=0.055),
                            dict(side="Buy", type="C", strike=2.70, qty=1, entry=0.005)])
    snap = dict(expiry="2026-07-22", spot_at_save=2.60, dte_at_save=29,
                cost_basis=0.015, mark_at_save=0.015, pnl_at_save=0.0)
    pid = TK.save_position("dryrun fly", legs_df, snap,
                           rules=dict(tp_frac=0.5, sl_frac=0.5, roll_dte=14))
    saved = TK.load_saved()
    sub = saved[saved["position_id"] == pid]
    assert len(sub) == 3
    assert str(sub["roll_chain_id"].iloc[0]) == pid           # root lineage = own id
    assert TK.position_rules(sub)["tp_frac"] == 0.5
    assert TK.position_status(sub) == "open"

    TK.set_position_rules(pid, dict(sl_frac=0.4))
    assert TK.position_rules(TK.load_saved().pipe(lambda d: d[d.position_id == pid]))["sl_frac"] == 0.4

    TK.set_position_status(pid, "hit_tp", trigger_date="2026-07-01",
                           trigger_level=0.05, trigger_reason="TP 0.50*maxP")
    assert TK.position_status(TK.load_saved().pipe(lambda d: d[d.position_id == pid])) == "hit_tp"
    print("save/load/set_rules/set_status round-trip OK")

    # --- lineage cumulative across a manually-linked child ---
    child_pid = TK.save_position("dryrun fly (roll)", legs_df,
                                 dict(expiry="2026-08-26", spot_at_save=2.62,
                                      dte_at_save=30, cost_basis=0.016,
                                      mark_at_save=0.016, pnl_at_save=0.0),
                                 rules=dict(tp_frac=0.5, sl_frac=0.5, roll_dte=14),
                                 roll_parent_id=pid, roll_chain_id=pid)
    TK.record_tracking(dict(position_id=pid, recorded_at="2026-07-01 15:30:00",
                            asof_date="2026-07-01", expiry="2026-07-22", spot=2.61,
                            dte=21, cost_basis=0.015, mark=0.05, pnl=0.035,
                            status="hit_tp", alert="HIT_TP"))
    TK.record_tracking(dict(position_id=child_pid, recorded_at="2026-08-01 15:30:00",
                            asof_date="2026-08-01", expiry="2026-08-26", spot=2.63,
                            dte=25, cost_basis=0.016, mark=0.02, pnl=0.004,
                            status="open", alert=""))
    lin = TK.roll_lineage(pid)
    assert len(lin) == 2, lin
    assert approx(lin["cumulative_pnl"].iloc[-1], 0.035 + 0.004), lin["cumulative_pnl"].iloc[-1]
    print("roll_lineage OK: cumulative P&L =",
          f"{lin['cumulative_pnl'].iloc[-1]:+.4f}")

    print("\nALL DRY-RUN ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
