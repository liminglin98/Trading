# -*- coding: utf-8 -*-
"""
precompute.py -- build the app snapshot OFFLINE for the deployed website
========================================================================
Runs the heavy model fit + SSE IV fetches once (where the Chinese feeds are
reachable -- i.e. your local machine / a China VPS), then pickles the entire
app state to out/app_snapshot.pkl. The deployed Streamlit app READS that file
instead of computing, so cold starts are instant and the website never depends
on a live fetch.

Run (from inside macro_model\):
    python precompute.py

Typical daily flow (see run_snapshot.bat): record the EOD strategy snapshot,
build this app snapshot, then git add/commit/push so Streamlit Cloud redeploys
and serves the fresh data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

import butterfly_param_search as PS
import butterfly_wing_exit_optimizer as WX
import tracking as TK

OUT = Path(__file__).resolve().parent / "out" / "app_snapshot.pkl"


def main():
    print("building contexts (model fit + SSE IV chain)...")
    contexts, etf, last_daily, X, y, df = PS.build_contexts()

    print("pulling fresh 50ETF close...")
    etf_fresh = TK.load_etf()

    print("building live context (most recent chain)...")
    live_ctx = WX.build_live_context(etf_fresh, X, y)
    live = live_ctx is not None
    if not live:
        print("  WARNING: live context unavailable; app will fall back to last "
              "settled month for the suggestion tab.")

    payload = dict(
        contexts=contexts, etf=etf, last_daily=last_daily,
        X=X, y=y, df=df,
        etf_fresh=etf_fresh, live_ctx=live_ctx, live=live,
        built_at=pd.Timestamp.now(),
    )

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)

    print(f"wrote {OUT}")
    print(f"  contexts: {len(contexts)} months | live: {live} | "
          f"built_at: {payload['built_at']:%Y-%m-%d %H:%M}")


if __name__ == "__main__":
    main()
