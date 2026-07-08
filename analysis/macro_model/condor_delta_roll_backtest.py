# -*- coding: utf-8 -*-
"""
condor_delta_roll_backtest.py
===============================
condor_delta_exit_backtest.py showed that closing a delta-targeted condor
early on a trigger (DTE / profit-target / stop-loss / delta-breach) and then
sitting in CASH until the next monthly entry date destroys expectancy -- the
edge in these OTM condors comes from surviving the mid-life mark-to-market
drawdown, not reacting to it.

This script asks the natural follow-up: what if, instead of going to cash,
the triggered position is immediately ROLLED -- closed and replaced same-day
by a fresh delta-targeted condor off that day's chain -- so capital never
sits idle? That's the standard meaning of "rolling" in options trading
(close the current leg, open the replacement, same transaction), as opposed
to condor_delta_backtest.py's baseline of "hold to expiry, open fresh next
calendar month" which just chains independent trades with a cash gap between
them.

Because a rolled position never goes flat, this is modeled as ONE continuous
chain per (rule, target_delta, p) config across the full 2015-2026 history
(not one campaign per calendar month -- an aggressive roll trigger can span
years before any leg naturally reaches its own expiry, so monthly campaigns
would badly overlap). Whenever a leg ends -- by trigger or by expiring
naturally -- the next leg opens the same day off the current chain. Each
completed leg is its own Kelly-sized bet (pnl / that leg's own max_loss),
fed through the same fractional-Kelly compounding used everywhere else in
this project, so a rule that rolls every 9 days just produces more, smaller
bets in the same equity curve.

Policies swept (each ALONE, holding entry target_delta/p fixed):
  DTE roll        -- roll once <= N days remain, regardless of P&L (the
                     "42/21 DTE" scheduled-defense style)
  STOP-LOSS roll  -- roll once cost-to-close >= K x credit received
                     (defensive: only roll when the trade is going against you)
  STOP-GAIN roll  -- roll once captured P&L >= X% of credit (opportunistic:
                     bank the win early and redeploy capital immediately
                     instead of waiting out the rest of theta decay)
plus a combined 21-DTE/50%-gain/2x-loss policy and the baseline (no trigger
-- roll only at natural expiry, i.e. condor_delta_backtest.py's cadence
chained back-to-back with no cash gap, used as the sanity-check reference).

RUN IN A NORMAL PYTHON ENV. No network calls.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import iron_condor_backtest as IC        # noqa: E402  local chain/ETF loaders
import condor_delta_backtest as CD       # noqa: E402  entry logic
import condor_delta_exit_backtest as CX  # noqa: E402  build_life, find_exit
import butterfly_timing_backtest as B    # noqa: E402  kelly_opt

OUTDIR = IC.OUTDIR
C_SURFACE, C_INK, C_MUTED = "#fcfcfb", "#0b0b0b", "#898781"
C_SERIES = ["#3b6ea5", "#c1502e", "#5a8f5a", "#8a7ab5"]

MAX_LEGS = 3000   # safety cap so a degenerate rule can't loop forever


def walk_rolling_chain(chain: pd.DataFrame, spot_series: pd.Series,
                        target_delta: float, p: int, rule: dict,
                        target_dte: int = CD.TARGET_DTE) -> list[dict]:
    """One continuous chain of trades: whenever a leg ends (trigger or its
    own expiry), the next leg opens the same day. Stops when the current
    leg's own expiry hasn't happened yet (still open -- can't realize it)."""
    trading_days = np.sort(chain[CD.C_DATE].unique())
    last_daily = spot_series.index.max()
    cur_date = pd.Timestamp(trading_days[0])
    trades = []

    def next_day(d):
        later = trading_days[trading_days > np.datetime64(d)]
        return pd.Timestamp(later[0]) if len(later) else None

    for _ in range(MAX_LEGS):
        day = chain[chain[CD.C_DATE] == cur_date]
        if day.empty:
            nd = next_day(cur_date)
            if nd is None:
                break
            cur_date = nd
            continue
        exp_date = CD.choose_expiry_for(day, target_dte)
        if exp_date is None:
            nd = next_day(cur_date)
            if nd is None:
                break
            cur_date = nd
            continue
        exp_date = pd.Timestamp(exp_date)
        sub = day[day[CD.C_EXP] == exp_date]
        built = CD.build_condor_by_delta(sub, target_delta, p)
        if built is None:
            nd = next_day(cur_date)
            if nd is None:
                break
            cur_date = nd
            continue
        life = CX.build_life(chain, exp_date, built["strikes"], cur_date)
        if life is None or life.empty:
            break
        trig_day, value, reason = CX.find_exit(life, built["credit"], rule)
        if trig_day is not None:
            pnl = built["credit"] - value - 4 * CD.SLIP_PER_LEG
            trades.append(dict(entry=cur_date, exit=trig_day, pnl=pnl,
                                max_loss=built["max_loss"], reason=reason))
            cur_date = trig_day
            continue
        # no trigger before expiry -- held naturally; only realizable if past
        if exp_date > last_daily or exp_date not in spot_series.index:
            break
        pnl = CD.realized_pnl(built["strikes"], built["credit"],
                               float(spot_series.loc[exp_date]))
        trades.append(dict(entry=cur_date, exit=exp_date, pnl=pnl,
                            max_loss=built["max_loss"], reason="expiry"))
        cur_date = exp_date
    return trades


def rows_from_trades(trades: list[dict]) -> list[tuple]:
    return [(t["entry"], t["pnl"] / t["max_loss"]) for t in trades]


def equity_curve(rows: list, f_used: float) -> pd.Series:
    dates = [d for d, _ in rows]
    r = np.array([x for _, x in rows])
    eq = np.exp(np.log1p((f_used * r).clip(min=-0.999)).cumsum())
    return pd.Series(eq, index=dates)


FULL_VARIANTS = [
    ("baseline (roll only at expiry)", {}),
    ("dte_roll=21",  {"dte_exit": 21}),
    ("dte_roll=14",  {"dte_exit": 14}),
    ("dte_roll=10",  {"dte_exit": 10}),
    ("dte_roll=7",   {"dte_exit": 7}),
    ("stop_gain=0.25", {"profit_target": 0.25}),
    ("stop_gain=0.50", {"profit_target": 0.50}),
    ("stop_gain=0.75", {"profit_target": 0.75}),
    ("stop_loss=1.5x", {"stop_loss_mult": 1.5}),
    ("stop_loss=1.6x", {"stop_loss_mult": 1.6}),
    ("stop_loss=1.7x", {"stop_loss_mult": 1.7}),
    ("stop_loss=1.8x", {"stop_loss_mult": 1.8}),
    ("stop_loss=1.9x", {"stop_loss_mult": 1.9}),
    ("stop_loss=2.0x", {"stop_loss_mult": 2.0}),
    ("stop_loss=2.1x", {"stop_loss_mult": 2.1}),
    ("stop_loss=2.2x", {"stop_loss_mult": 2.2}),
    ("stop_loss=2.3x", {"stop_loss_mult": 2.3}),
    ("stop_loss=2.5x", {"stop_loss_mult": 2.5}),
    ("stop_loss=3.0x", {"stop_loss_mult": 3.0}),
    ("combined 21dte/50%gain/2xloss",
     {"dte_exit": 21, "profit_target": 0.50, "stop_loss_mult": 2.0}),
]

# lighter grid for the generalization check -- just the stop-loss axis where
# the primary config found a plateau, no need to re-prove dte/stop-gain are bad
SL_ONLY_VARIANTS = [("baseline (roll only at expiry)", {})] + [
    (f"stop_loss={m:.1f}x", {"stop_loss_mult": m})
    for m in (1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.5, 3.0)
]


def run_config(chain, spot_series, etf, target_delta: float, p: int,
               variants: list, tag: str) -> tuple[pd.DataFrame, dict | None]:
    print(f"\n=== Rolling chain config: delta={target_delta} p={p} ({tag or 'primary'}), "
          f"full history {chain[CD.C_DATE].min().date()} -> "
          f"{chain[CD.C_DATE].max().date()} ===")

    out_rows, curves = [], {}
    for name, rule in variants:
        trades = walk_rolling_chain(chain, spot_series, target_delta, p, rule)
        rows = rows_from_trades(trades)
        stats = CD.cell_stats(rows)
        r = np.array([x for _, x in rows])
        reasons = pd.Series([t["reason"] for t in trades]).value_counts().to_dict()
        hold_days = np.array([(t["exit"] - t["entry"]).days for t in trades])
        if stats is None:
            print(f"{name:<32} n={len(rows):>4}  (too few trades, skipped from grid)")
            continue
        out_rows.append(dict(variant=name, **rule, **stats,
                              mean_r=float(r.mean()), avg_hold_days=float(hold_days.mean()),
                              reasons=reasons))
        curves[name] = equity_curve(rows, stats["f"])
        print(f"{name:<32} n={stats['n']:>4} avg_hold={hold_days.mean():>5.1f}d  "
              f"mean_r={r.mean():+.4f}  f={stats['f']:.2f}  CAGR={stats['cagr']:+.2%}  "
              f"MDD={stats['mdd']:+.2%}  win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}  "
              f"legs/reason={reasons}")

    grid = pd.DataFrame(out_rows)
    suffix = f"_{tag}" if tag else ""
    grid.to_csv(OUTDIR / f"condor_delta_roll_grid{suffix}.csv", index=False)
    base_stats = next((r for r in out_rows if r["variant"].startswith("baseline")), None)

    sl_rows = grid[grid["variant"].str.match(r"stop_loss=\d")].copy()
    if len(sl_rows) >= 3:
        sl_rows["mult"] = sl_rows["variant"].str.extract(r"=([\d.]+)x").astype(float)
        sl_rows = sl_rows.sort_values("mult")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), facecolor=C_SURFACE, sharex=True)
        for ax in (ax1, ax2):
            ax.set_facecolor(C_SURFACE)
            ax.tick_params(colors=C_MUTED)
        ax1.plot(sl_rows["mult"], sl_rows["cagr"] * 100, "o-", color=C_SERIES[0])
        ax1.axhline(base_stats["cagr"] * 100 if base_stats else 0, color=C_MUTED,
                    linestyle="--", linewidth=1, label="baseline CAGR")
        ax1.set_ylabel("CAGR (%)", color=C_INK)
        ax1.legend(fontsize=8, frameon=False)
        ax2.plot(sl_rows["mult"], sl_rows["calmar"], "o-", color=C_SERIES[1])
        ax2.axhline(base_stats["calmar"] if base_stats else 0, color=C_MUTED,
                    linestyle="--", linewidth=1, label="baseline Calmar")
        ax2.set_ylabel("Calmar", color=C_INK)
        ax2.set_xlabel("stop-loss multiple of credit received (x)", color=C_INK)
        ax2.legend(fontsize=8, frameon=False)
        ax1.set_title(f"Fine stop-loss roll sweep, delta={target_delta} p={p}",
                      color=C_INK, fontsize=12, loc="left", pad=10)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"condor_delta_roll_sl_sweep{suffix}.png", dpi=150, facecolor=C_SURFACE)
        plt.close(fig)
        print(f"\nFine stop-loss sweep ({tag or 'primary'}):")
        for _, r in sl_rows.iterrows():
            print(f"  {r['mult']:.1f}x  n={int(r['n']):>4}  mean_r={r['mean_r']:+.4f}  "
                  f"CAGR={r['cagr']:+.2%}  MDD={r['mdd']:+.2%}  Calmar={r['calmar']:.2f}")

    if not grid.empty and grid["variant"].str.startswith("baseline").any() and len(grid) > 1:
        best_row = grid[~grid["variant"].str.startswith("baseline")].sort_values(
            "calmar", ascending=False).iloc[0]
        plot_names = list(dict.fromkeys(["baseline (roll only at expiry)", best_row["variant"]]))
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_SURFACE)
        ax.set_facecolor(C_SURFACE)
        for i, name in enumerate(plot_names):
            if name in curves:
                ax.plot(curves[name].index, curves[name].values, label=name,
                         color=C_SERIES[i % len(C_SERIES)], linewidth=1.6)
        anchor = curves[plot_names[0]].index.min()
        bh = etf[etf.index >= anchor]
        bh = bh / bh.iloc[0]
        ax.plot(bh.index, bh.values, label="buy & hold 50ETF", color=C_MUTED,
                 linewidth=1.2, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"Rolling policies, delta={target_delta} p={p}, "
                     f"continuous 2015-2026 chain", color=C_INK, fontsize=12,
                     loc="left", pad=10)
        ax.tick_params(colors=C_MUTED)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"condor_delta_roll_equity{suffix}.png", dpi=150, facecolor=C_SURFACE)
        plt.close(fig)

    print(f"\nSaved: {OUTDIR/f'condor_delta_roll_grid{suffix}.csv'} and companion plots")
    return grid, base_stats


def main() -> None:
    chain = IC.load_local_chain()
    etf = IC.load_local_etf()
    spot_series = CD.chain_spot_series(chain)

    PRIMARY = dict(target_delta=0.20, p=3)
    grid1, base1 = run_config(chain, spot_series, etf, PRIMARY["target_delta"], PRIMARY["p"],
                               FULL_VARIANTS, tag="")

    # sanity check: baseline chain should be close to condor_delta_backtest.py's
    # monthly-cadence baseline (not identical -- this chains off prior expiry
    # date, not calendar month-end -- but same ballpark).
    orig_grid = CD.run_grid(chain, spot_series)
    orig_stats = CD.cell_stats(orig_grid.get((PRIMARY["target_delta"], PRIMARY["p"]), []))
    if orig_stats is not None and base1 is not None:
        print(f"\nSanity check -- rolling-chain baseline vs condor_delta_backtest.py's "
              f"monthly baseline: CAGR {base1['cagr']:+.2%} vs {orig_stats['cagr']:+.2%}, "
              f"n {base1['n']} vs {orig_stats['n']} (different cadence, same ballpark expected)")

    # generalization check: same stop-loss plateau, different entry config
    SECOND = dict(target_delta=0.25, p=3)
    grid2, base2 = run_config(chain, spot_series, etf, SECOND["target_delta"], SECOND["p"],
                               SL_ONLY_VARIANTS, tag="delta025p3")

    print(f"\n=== Generalization comparison: delta={PRIMARY['target_delta']}/p={PRIMARY['p']} "
          f"vs delta={SECOND['target_delta']}/p={SECOND['p']} ===")
    for g, tag in ((grid1, "0.20/3"), (grid2, "0.25/3")):
        sl = g[g["variant"].str.match(r"stop_loss=\d")].copy()
        if sl.empty:
            continue
        sl["mult"] = sl["variant"].str.extract(r"=([\d.]+)x").astype(float)
        best = sl.sort_values("calmar", ascending=False).iloc[0]
        pos_band = sl[sl["cagr"] > 0]["mult"].tolist()
        print(f"  {tag}: best Calmar at {best['mult']:.1f}x (CAGR={best['cagr']:+.2%}, "
              f"Calmar={best['calmar']:.2f}); positive-CAGR multiples = {pos_band}")


if __name__ == "__main__":
    main()
