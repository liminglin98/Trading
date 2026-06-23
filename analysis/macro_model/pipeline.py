# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Upper-level runner that makes sure the model is fed FRESH data before the wing/
exit optimizer runs, and is also imported by the Streamlit app (app.py) for its
"refresh" button + freshness banner.

Two data layers (see the project notes):
  * ETF daily + SSE option IV  -> auto-fetched at run time (sina always fresh;
    the SSE chain is date-cached and any uncached/most-recent date is fetched).
  * factor PANEL (Data/akshare/panel_monthly.csv) -> a STATIC file built by
    build_model_panel.py. This is the ONLY piece that does not self-refresh, so
    this runner checks its freshness and can rebuild it.

CLI:
    python pipeline.py                 # check freshness, warn if stale, run optimizer
    python pipeline.py --refresh       # rebuild the panel first (slow; some
                                       # sources need the proxy), then optimize
    python pipeline.py --check         # only print panel freshness and exit
    python pipeline.py --snapshot      # EOD: snapshot the two model strategies
                                       # (butterfly + iron condor) into
                                       # strategy_tracking.csv AND mark any hand-
                                       # saved positions, then exit. Point Windows
                                       # Task Scheduler at this ~30 min after the
                                       # 15:00 close for a hands-off daily track.

RUN IN A NORMAL PYTHON ENV from inside macro_model\.
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BUILD_SCRIPT = HERE / "build_model_panel.py"


def panel_path():
    """Resolve the panel CSV path from the model module (single source of truth)."""
    import butterfly_factor_forecast as BF
    return Path(BF.PANEL)


def panel_last_month():
    """Return the last month_end Timestamp in the panel, or None if unreadable."""
    p = panel_path()
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["month_end"])
        return pd.Timestamp(df["month_end"].max())
    except Exception:
        return None


def freshness():
    """(last_month, is_stale, message). Stale = panel's last month is before the
    previous completed month (i.e. a monthly update is plausibly available)."""
    last = panel_last_month()
    now = pd.Timestamp.now().normalize()
    if last is None:
        return None, True, "panel not found or unreadable -- rebuild needed."
    # stale if the last month_end is more than ~35 days old (a monthly update is due)
    stale = (now - last).days > 35
    msg = (f"panel last month_end = {last.date()}  "
           f"({(now - last).days} days old) -> {'STALE' if stale else 'fresh'}")
    return last, stale, msg


def rebuild_panel(timeout=1800):
    """Run build_model_panel.py as a subprocess. Returns (ok, combined_output)."""
    if not BUILD_SCRIPT.exists():
        return False, f"build script not found: {BUILD_SCRIPT}"
    try:
        proc = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT)],
            cwd=str(HERE), capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
        return proc.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, f"build_model_panel.py timed out after {timeout}s."
    except Exception as e:
        return False, f"rebuild failed: {e}"


def run_optimizer():
    """Import and run the wing/exit optimizer (prints report, writes CSV/PNG)."""
    import butterfly_wing_exit_optimizer as WX
    WX.main()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    do_refresh = "--refresh" in argv
    check_only = "--check" in argv

    if "--snapshot" in argv:
        import tracking
        print("=" * 70)
        print(f"EOD SNAPSHOT  {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 70)

        # 1) the two model strategies (butterfly + iron condor) -- the primary
        #    daily forward track. Build contexts once to get the model-band wing
        #    so the tracked structure matches the dashboard's suggestion exactly.
        print("model strategies (butterfly + iron condor):")
        try:
            import butterfly_param_search as PS
            import butterfly_wing_exit_optimizer as WX
            contexts, etf, last_daily, X, y, df = PS.build_contexts()
            params = dict(bfly_wing=WX.model_band_wing(contexts),
                          **{k: tracking.STRAT_DEFAULTS[k] for k in ("condor_m", "condor_p")})
            for strat, m in tracking.snapshot_strategies(etf=etf, params=params):
                print(f"  {strat} -> {m}")
        except Exception as e:
            print(f"  [strategy snapshot failed: {e}]")

        # 2) any hand-saved combinations from the Holdings tab (legacy journal)
        print("saved combinations:")
        results = tracking.snapshot_all_open()
        alerts = []
        for pid, m in results:
            print(f"  {pid} -> {m}")
            if "ALERT" in m:
                alerts.append((pid, m.split("::")[0].replace("ALERT", "").strip()))
        print("-" * 70)
        if alerts:
            print("STOP / ROLL ALERTS (analytics only -- act in your broker):")
            for pid, a in alerts:
                print(f"  [!] {pid} -> {a}")
        else:
            print("No TP/SL/roll alerts this run.")
        print("=" * 70)
        return

    last, stale, msg = freshness()
    print("=" * 70)
    print("PIPELINE: data freshness check")
    print("  " + msg)
    print("=" * 70)
    if check_only:
        return

    if do_refresh:
        print("\nRebuilding panel (build_model_panel.py)... this can take a while.")
        ok, out = rebuild_panel()
        print(out[-4000:] if out else "(no output)")
        if not ok:
            print("\n[WARN] panel rebuild did not succeed cleanly. Continuing with the")
            print("       existing panel -- results may be stale. Check the output above.")
        else:
            last, stale, msg = freshness()
            print("\nAfter rebuild: " + msg)
    elif stale:
        print("\n[WARN] panel looks STALE. Run  python pipeline.py --refresh  to rebuild")
        print("       it first (the optimizer's dispersion band uses these predictors).")
        print("       Proceeding on the existing panel for now.\n")

    run_optimizer()


if __name__ == "__main__":
    main()
