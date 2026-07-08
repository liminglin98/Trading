# -*- coding: utf-8 -*-
"""
condor_delta_exit_backtest.py
===============================
Adds a MECHANICAL EXIT to condor_delta_backtest.py's hold-to-expiry baseline
and sweeps exit parameters to see whether managing the trade before expiry
beats letting it expire.

condor_delta_backtest.py enters a delta-targeted condor once a month and
marks it exactly twice: entry credit, expiry intrinsic payoff. This script
reuses its identical entry logic (build_condor_by_delta / choose_expiry_for /
monthly_entry_dates, imported from condor_delta_backtest.py) but then WALKS
the position day by day using the same local Sina chain's daily quoted
closes + Greeks (Data/options/option_daily_510050.csv) up to expiry, and
exits the first day one of these rules trips:

  DTE_EXIT       -- close once <= N calendar days remain (time-based "get out
                    before gamma risk ramps up", e.g. the popular 21-DTE rule)
  PROFIT_TARGET  -- close once captured P&L >= X% of the credit received
                    (stop collecting theta once most of it is banked)
  STOP_LOSS_MULT -- close once cost-to-close >= K x the credit received
                    (cap the loss instead of riding it to max_loss at expiry)
  DELTA_STOP     -- close once either SHORT leg's |delta| breaches a
                    threshold (the market now prices that strike as likely
                    ITM -- a defensive "the short is being tested" signal)

Each axis is swept ALONE (others off) holding entry (target_delta, p) fixed,
so the marginal effect of each mechanism is comparable; a COMBINED cell
(21 DTE / 50% PT / 2x SL -- the common "manage winners, cap losers" retail
recipe) is run for reference, plus a second entry config to sanity-check
that the winning rule isn't an artifact of one delta/wing choice.

Position value before expiry = (short-long put spread) + (short-long call
spread) quoted closes -- same formula used for entry credit, so cost-to-
close is priced consistently with how the position was entered (real
quotes, no Black-Scholes reconstruction). Exit charges the same
SLIP_PER_LEG x 4 as entry. With every rule off this script reproduces
condor_delta_backtest.py's hold-to-expiry numbers exactly (sanity-checked
in main()).

RUN IN A NORMAL PYTHON ENV. No network calls.
"""
from __future__ import annotations

import collections
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import iron_condor_backtest as IC       # noqa: E402  local chain/ETF loaders
import condor_delta_backtest as CD      # noqa: E402  entry logic (delta-targeted condor)
import butterfly_timing_backtest as B   # noqa: E402  kelly_opt, KELLY_FRAC/CAP

OUTDIR = IC.OUTDIR

# palette (light mode; see dataviz skill references/palette.md)
C_SURFACE, C_INK, C_MUTED = "#fcfcfb", "#0b0b0b", "#898781"
C_SERIES = ["#3b6ea5", "#c1502e", "#5a8f5a", "#8a7ab5"]


# ============================================================== life-of-trade
def build_life(chain: pd.DataFrame, exp_date: pd.Timestamp, strikes: tuple,
                entry_date: pd.Timestamp) -> pd.DataFrame | None:
    """Daily closes + short-leg Deltas for the 4 legs, entry_date (exclusive)
    through exp_date (inclusive). One row per cached trading day."""
    Kp_l, Kp_s, Kc_s, Kc_l = strikes
    legs = chain[(chain[CD.C_EXP] == exp_date)
                 & (chain[CD.C_DATE] > entry_date)
                 & (chain[CD.C_DATE] <= exp_date)
                 & (chain[CD.C_STRIKE].isin([Kp_l, Kp_s, Kc_s, Kc_l]))]
    if legs.empty:
        return None
    piv_close = legs.pivot_table(index=CD.C_DATE, columns=[CD.C_STRIKE, CD.C_CP],
                                  values=CD.C_CLOSE, aggfunc="first")
    piv_delta = legs.pivot_table(index=CD.C_DATE, columns=[CD.C_STRIKE, CD.C_CP],
                                  values="Delta", aggfunc="first")
    dte = legs.groupby(CD.C_DATE)[CD.C_DTE].first()

    def col(piv, K, cp):
        return piv[(K, cp)] if (K, cp) in piv.columns else pd.Series(index=piv.index, dtype=float)

    out = pd.DataFrame({
        "dte": dte,
        "pl_close": col(piv_close, Kp_l, "P"), "ps_close": col(piv_close, Kp_s, "P"),
        "cs_close": col(piv_close, Kc_s, "C"), "cl_close": col(piv_close, Kc_l, "C"),
        "ps_delta": col(piv_delta, Kp_s, "P"), "cs_delta": col(piv_delta, Kc_s, "C"),
    }).sort_index()
    return out


def build_entries(chain: pd.DataFrame, spot_series: pd.Series,
                   target_delta: float, p: int) -> list[dict]:
    entries = []
    for d in CD.monthly_entry_dates(chain):
        day = chain[chain[CD.C_DATE] == d]
        if day.empty:
            continue
        exp_date = CD.choose_expiry_for(day)
        if exp_date is None:
            continue
        exp_date = pd.Timestamp(exp_date)
        if exp_date not in spot_series.index:
            continue                      # not yet realized
        sub = day[day[CD.C_EXP] == exp_date]
        built = CD.build_condor_by_delta(sub, target_delta, p)
        if built is None:
            continue
        life = build_life(chain, exp_date, built["strikes"], d)
        if life is None or life.empty:
            continue
        entries.append(dict(entry_date=d, exp_date=exp_date, strikes=built["strikes"],
                             credit=built["credit"], max_loss=built["max_loss"], life=life,
                             S_exp=float(spot_series.loc[exp_date])))
    return entries


# ============================================================== exit rules
def find_exit(life: pd.DataFrame, entry_credit: float, rule: dict):
    """First day one of the active rule legs trips. Returns
    (exit_date, cost_to_close, reason) or (None, None, None) if it never
    trips (position held to expiry)."""
    dte_exit = rule.get("dte_exit")
    pt = rule.get("profit_target")
    sl = rule.get("stop_loss_mult")
    dstop = rule.get("delta_stop")
    for day, row in life.iterrows():
        have_value = pd.notna(row[["pl_close", "ps_close", "cs_close", "cl_close"]]).all()
        value = ((row["ps_close"] - row["pl_close"]) + (row["cs_close"] - row["cl_close"])
                 if have_value else None)
        if dte_exit is not None and pd.notna(row["dte"]) and row["dte"] <= dte_exit and value is not None:
            return day, value, "dte"
        if pt is not None and value is not None and (entry_credit - value) >= pt * entry_credit:
            return day, value, "pt"
        if sl is not None and value is not None and value >= sl * entry_credit:
            return day, value, "sl"
        if dstop is not None:
            ps_d, cs_d = row["ps_delta"], row["cs_delta"]
            breach = (pd.notna(ps_d) and abs(ps_d) >= dstop) or (pd.notna(cs_d) and abs(cs_d) >= dstop)
            if breach and value is not None:
                return day, value, "delta"
    return None, None, None


def apply_rule(entries: list[dict], rule: dict):
    rows, reasons = [], collections.Counter()
    for e in entries:
        day, value, reason = find_exit(e["life"], e["credit"], rule)
        if day is not None:
            pnl = e["credit"] - value - 4 * CD.SLIP_PER_LEG
            reasons[reason] += 1
        else:
            pnl = CD.realized_pnl(e["strikes"], e["credit"], e["S_exp"])
            reasons["expiry"] += 1
        rows.append((e["entry_date"], pnl / e["max_loss"]))
    return rows, reasons


def equity_curve(rows: list, f_used: float) -> pd.Series:
    dates = [d for d, _ in rows]
    r = np.array([x for _, x in rows])
    eq = np.exp(np.log1p((f_used * r).clip(min=-0.999)).cumsum())
    return pd.Series(eq, index=dates)


# ============================================================== main
def main() -> None:
    chain = IC.load_local_chain()
    etf = IC.load_local_etf()
    spot_series = CD.chain_spot_series(chain)

    PRIMARY = dict(target_delta=0.20, p=3)
    entries = build_entries(chain, spot_series, **PRIMARY)
    print(f"Primary entry config: delta={PRIMARY['target_delta']} p={PRIMARY['p']}  "
          f"-> {len(entries)} monthly entries with a full daily life series")

    variants = [
        ("baseline (hold to expiry)", {}),
        ("dte_exit=21",  {"dte_exit": 21}),
        ("dte_exit=14",  {"dte_exit": 14}),
        ("dte_exit=10",  {"dte_exit": 10}),
        ("dte_exit=7",   {"dte_exit": 7}),
        ("profit_target=0.25", {"profit_target": 0.25}),
        ("profit_target=0.50", {"profit_target": 0.50}),
        ("profit_target=0.75", {"profit_target": 0.75}),
        ("stop_loss=1.5x", {"stop_loss_mult": 1.5}),
        ("stop_loss=2.0x", {"stop_loss_mult": 2.0}),
        ("stop_loss=3.0x", {"stop_loss_mult": 3.0}),
        ("delta_stop=0.30", {"delta_stop": 0.30}),
        ("delta_stop=0.40", {"delta_stop": 0.40}),
        ("delta_stop=0.50", {"delta_stop": 0.50}),
        ("combined 21dte/50%pt/2xsl",
         {"dte_exit": 21, "profit_target": 0.50, "stop_loss_mult": 2.0}),
    ]

    out_rows = []
    curves = {}
    for name, rule in variants:
        rows, reasons = apply_rule(entries, rule)
        stats = CD.cell_stats(rows)
        if stats is None:
            continue
        out_rows.append(dict(variant=name, **rule, **stats,
                              reasons=dict(reasons)))
        curves[name] = equity_curve(rows, stats["f"])
        print(f"{name:<28} n={stats['n']:>3} f={stats['f']:.2f}  "
              f"CAGR={stats['cagr']:+.2%}  MDD={stats['mdd']:+.2%}  "
              f"win={stats['win']:.0%}  Calmar={stats['calmar']:.2f}  "
              f"exits={dict(reasons)}")

    grid = pd.DataFrame(out_rows)
    grid.to_csv(OUTDIR / "condor_delta_exit_grid.csv", index=False)

    # sanity check: baseline must reproduce condor_delta_backtest.py's own numbers
    base_rows, _ = apply_rule(entries, {})
    base_stats = CD.cell_stats(base_rows)
    orig_grid = CD.run_grid(chain, spot_series)
    orig_stats = CD.cell_stats(orig_grid.get((PRIMARY["target_delta"], PRIMARY["p"]), []))
    if orig_stats is not None:
        print(f"\nSanity check vs condor_delta_backtest.py directly: "
              f"CAGR {base_stats['cagr']:+.2%} vs {orig_stats['cagr']:+.2%}, "
              f"n {base_stats['n']} vs {orig_stats['n']}")

    # second entry config, winning rule only, to check it's not delta/p-specific
    best_row = grid[grid["variant"] != "baseline (hold to expiry)"].sort_values(
        "calmar", ascending=False).iloc[0]
    best_variant = next(r for n, r in variants if n == best_row["variant"])
    SECOND = dict(target_delta=0.25, p=3)
    entries2 = build_entries(chain, spot_series, **SECOND)
    base2_rows, _ = apply_rule(entries2, {})
    best2_rows, _ = apply_rule(entries2, best_variant)
    base2_stats, best2_stats = CD.cell_stats(base2_rows), CD.cell_stats(best2_rows)
    print(f"\nGeneralization check on delta={SECOND['target_delta']} p={SECOND['p']} "
          f"(n={len(entries2)}):")
    if base2_stats:
        print(f"  baseline    CAGR={base2_stats['cagr']:+.2%} MDD={base2_stats['mdd']:+.2%} "
              f"Calmar={base2_stats['calmar']:.2f}")
    if best2_stats:
        print(f"  {best_row['variant']:<24} CAGR={best2_stats['cagr']:+.2%} "
              f"MDD={best2_stats['mdd']:+.2%} Calmar={best2_stats['calmar']:.2f}")

    # equity plot: baseline vs best single-rule vs combined vs buy&hold
    plot_names = ["baseline (hold to expiry)", best_row["variant"],
                  "combined 21dte/50%pt/2xsl"]
    plot_names = list(dict.fromkeys(plot_names))  # dedupe, keep order
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    for i, name in enumerate(plot_names):
        if name in curves:
            ax.plot(curves[name].index, curves[name].values, label=name,
                     color=C_SERIES[i % len(C_SERIES)], linewidth=1.6)
    bh = (etf / etf.loc[etf.index >= curves[plot_names[0]].index.min()].iloc[0])
    bh = bh[bh.index >= curves[plot_names[0]].index.min()]
    ax.plot(bh.index, bh.values, label="buy & hold 50ETF", color=C_MUTED,
             linewidth=1.2, linestyle="--")
    ax.set_yscale("log")
    ax.set_title(f"Mechanical exit vs hold-to-expiry, delta={PRIMARY['target_delta']} "
                 f"p={PRIMARY['p']}", color=C_INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(colors=C_MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "condor_delta_exit_equity.png", dpi=150, facecolor=C_SURFACE)
    plt.close(fig)

    print(f"\nSaved: {OUTDIR/'condor_delta_exit_grid.csv'} and condor_delta_exit_equity.png")


if __name__ == "__main__":
    main()
