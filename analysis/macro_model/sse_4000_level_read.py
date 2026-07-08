# -*- coding: utf-8 -*-
"""
sse_4000_level_read.py
=========================
Standalone descriptive read of the SSE Composite (000001) around the 4000-
point level -- independent of the condor filter work in condor_trend_exit_
backtest.py, since where the broad index sits relative to a widely-watched
round number is useful context on its own, not just as a model input.

Not a trading signal or a prediction -- level_crossings()/key_level_read()
in index_trend_signal.py just count how many times the index has genuinely
crossed (persisted past confirm_days, filtering single-day whipsaws) vs.
merely brushed the level, and report the current run length. Interpretation
is left to the reader.

RUN IN A NORMAL PYTHON ENV. No network calls (reads the local index cache).
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import index_trend_signal as TS
import iron_condor_backtest as IC

OUTDIR = IC.OUTDIR
LEVEL = 4000.0

C_SURFACE, C_INK, C_MUTED = "#fcfcfb", "#0b0b0b", "#898781"
C_PRICE, C_LEVEL, C_UP, C_DOWN = "#3b6ea5", "#c1502e", "#5a8f5a", "#a34a4a"


def main() -> None:
    s = TS.load_index()
    read = TS.key_level_read(s, LEVEL, tol_pct=0.015, lookback_days=250, confirm_days=5)
    all_crossings = TS.level_crossings(s, LEVEL, confirm_days=5)

    print(f"SSE Composite (000001): {s.index.min().date()} -> {s.index.max().date()}")
    print(f"\n=== Level read: {LEVEL:.0f} as of {read['last_date'].date()} ===")
    print(f"  last close: {read['last_px']:.1f}  ({read['side']} the level, {read['dist_pct']:+.2%})")
    print(f"  touches within {read['tol_pct']:.1%} in the last {read['lookback_days']} trading days: "
          f"{read['touches_last_n']}")
    print(f"  genuine crossings (persist >=5 trading days), full 2010-2026 history: "
          f"{read['n_genuine_crossings_all_time']}")
    if read["last_cross"] is not None:
        lc = read["last_cross"]
        print(f"  most recent genuine crossing: {lc['date'].date()} ({lc['direction']}, "
              f"px={lc['px_at_cross']:.1f}) -- {read['days_since_last_genuine_cross']} trading "
              f"days ago, held_days={lc['held_days']}")

    print("\n=== All genuine crossings, full history ===")
    for _, r in all_crossings.iterrows():
        print(f"  {r['date'].date()}  {r['direction']:>4}  px={r['px_at_cross']:.1f}  "
              f"held {r['held_days']} trading days before next genuine cross (or still-open)")

    print("\n--- Read ---")
    n_decade_below = None
    long_runs = all_crossings[all_crossings["held_days"] > 500]
    if not long_runs.empty:
        biggest = long_runs.sort_values("held_days", ascending=False).iloc[0]
        n_decade_below = biggest
        print(f"  The dominant historical fact: after crossing below {LEVEL:.0f} on "
              f"{biggest['date'].date()}, the index stayed on that side for "
              f"{biggest['held_days']} trading days (~{biggest['held_days']/245:.1f} years) "
              f"before the next genuine cross -- 4000 spent most of 2015-2025 as a ceiling, "
              f"not a floor.")
    recent = all_crossings[all_crossings["date"] >= (s.index[-1] - pd.Timedelta(days=550))]
    print(f"  Last ~18 months: {len(recent)} genuine crossings -- the index is chopping back "
          f"and forth across {LEVEL:.0f} rather than trending cleanly through it either way. "
          f"Current side: {read['side']} ({read['dist_pct']:+.2%}), "
          f"{read['days_since_last_genuine_cross']} trading days since the last one held.")

    # ---------------------------------------------------------------- chart
    lookback = s.iloc[-380:]   # ~18 months of trading days
    fig, ax = plt.subplots(figsize=(9.5, 5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    ax.plot(lookback.index, lookback.values, color=C_PRICE, linewidth=1.6, label="SSE Composite (000001)")
    ax.axhline(LEVEL, color=C_LEVEL, linewidth=1.2, linestyle="--", label=f"{LEVEL:.0f} level")

    recent_in_view = all_crossings[all_crossings["date"] >= lookback.index[0]]
    for _, r in recent_in_view.iterrows():
        color = C_UP if r["direction"] == "up" else C_DOWN
        ax.scatter([r["date"]], [r["px_at_cross"]], color=color, s=45, zorder=5,
                   edgecolor=C_SURFACE, linewidth=0.8)
        ax.annotate(f"{r['date'].date()}", (r["date"], r["px_at_cross"]),
                    textcoords="offset points", xytext=(0, 10 if r["direction"] == "up" else -14),
                    fontsize=7.5, color=C_MUTED, ha="center")

    ax.set_title(f"SSE Composite around the {LEVEL:.0f} level, last ~18 months",
                 color=C_INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=C_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(C_MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    out_path = OUTDIR / "sse_4000_level.png"
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    main()
