# -*- coding: utf-8 -*-
"""cffex_vol_signal_transfer.py -- do the long-straddle vol signals that are
already live on ETF options transfer to the CFFEX index options?

Context. `expiry_week_atm_straddle_backtest.py` established that buying
expiry-week ATM gamma on IO/MO/HO unconditionally loses money (see
expiry_week_straddle_investigation.md: -32,423 yuan on 2.48M of premium, and
every positive cell dies once its three best trades are removed). The obvious
follow-up is whether *timing* the purchase with this lab's validated
cheap-vol signals rescues it. Six long-straddle picks are frozen and live in
`alerts_core.VOL_SIGNAL_ALERTS` -- all six on ETF underlyings, none on CFFEX.

Two stages, deliberately in this order:

STAGE A -- transfer test (`--stage a`, the honest one). Take each live pick's
PARAMETERS EXACTLY AS FROZEN (signal, threshold, max_hold, profit_target) and
run them on IO/MO/HO with no refitting whatsoever. Because the parameters were
chosen on entirely different underlyings, this is a genuine out-of-sample
test: there is no knob to turn and therefore no way to flatter the result.
The picks are read live from `alerts_core.VOL_SIGNAL_ALERTS` rather than
copied here, so this cannot silently drift out of sync with what is actually
running. Duplicate (signal, threshold, max_hold, profit_target) tuples across
the six picks collapse to one configuration each.

STAGE B -- IO-only sweep (`--stage b`). IO has four years of pre-2024
discovery, enough to grid-search and freeze a pick honestly. MO (listed
2022-07) and HO (2022-12) have ~1.5 years, which will not clear
MIN_CELL_N=15; rather than fit them on a thin sample, IO's frozen pick is
carried over to MO/HO as an independent out-of-sample read. Sweeps run via
`long_straddle_rv_contraction_backtest.sweep`.

Benchmark. Every stage-A row is printed against the unconditional
expiry-week straddle's return on debit for the same underlying (IO -7.8%,
MO +7.3%, HO -10.3% on premium deployed -- and MO's is three months of luck,
see the investigation memo). A timing signal that cannot beat *that* has not
earned anything; beating zero is not the bar.

Signal availability on CFFEX was checked rather than assumed:
`rv_pct`/`iv_pct`/`vrp`/`pc_oi_pct`/`pc_vol_pct` all produce healthy
non-degenerate series; `gex_pct`/`gex_mag_pct` are blocked (no per-contract
gamma in this dataset -- see long_straddle_rv_contraction_backtest.
CFFEX_UNSUPPORTED_SIGNALS) and `flip_dist` is 510050/510300-only.

`rv_pct` and `vrp` only became computable for these underlyings once
`index_price_updater.py` cached the real index daily closes -- run it first.

Usage:
    python cffex_vol_signal_transfer.py                # stage A
    python cffex_vol_signal_transfer.py --stage b      # IO sweep + MO/HO carry-over
    python cffex_vol_signal_transfer.py --stage a --lag-signal-days 1
    python cffex_vol_signal_transfer.py --stage robustness   # cost/direction/lag/year checks on IO's picks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import alerts_core as ac
import consolidated_store as cs
import data_store as ds
import long_straddle_rv_contraction_backtest as ls

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = ls.SPLIT_DATE
MIN_CELL_N = ls.MIN_CELL_N

TARGETS = list(ls.CFFEX_UNDERLYINGS)  # 000300, 000852, 000016

# IO's two frozen picks from the stage-B sweep, referenced by all four
# robustness checks below so a threshold/hold/target typo can't silently
# diverge between the memo, stage_b's own printed pick, and these checks.
IO_VRP_PICK = dict(underlying="000300", signal="vrp", threshold=0.0,
                   max_hold_days=30, profit_target=0.5)
IO_IVPCT_PICK = dict(underlying="000300", signal="iv_pct", threshold=0.30,
                     max_hold_days=20, profit_target=2.0)
SLIPPAGE_GRID = [0.0, 0.2, 0.5, 1.0]  # index points per leg, CFFEX tick=0.2

# Unconditional expiry-week ATM straddle, Mon+Wed+Fri held to expiry, return
# on premium deployed -- expiry_week_straddle_investigation.md section 3.1.
# This is the null the signal has to beat, not zero.
NULL_BASELINE = {"000300": -0.078, "000852": +0.073, "000016": -0.103}


def frozen_configs() -> list[dict]:
    """The live long-straddle picks, deduped to distinct parameter sets.
    Read from alerts_core so this tracks whatever is actually deployed."""
    seen: dict[tuple, dict] = {}
    for cfg in ac.VOL_SIGNAL_ALERTS:
        key = (cfg["signal"], cfg["threshold"], cfg["max_hold"], cfg["profit_target"])
        seen.setdefault(key, {
            "signal": cfg["signal"], "threshold": cfg["threshold"],
            "max_hold": cfg["max_hold"], "profit_target": cfg["profit_target"],
            "from": [],
        })
        seen[key]["from"].append(cfg["u"])
    return list(seen.values())


def _stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "mean_r": np.nan, "median_r": np.nan, "win_rate": np.nan,
                "mean_r_ex_top3": np.nan, "disc_n": 0, "disc_mean_r": np.nan,
                "hold_n": 0, "hold_mean_r": np.nan, "avg_days": np.nan}
    r = trades["pnl"] / trades["debit"]
    ranked = r.sort_values(ascending=False)
    disc = trades[trades["entry_date"] < SPLIT_DATE]
    hold = trades[trades["entry_date"] >= SPLIT_DATE]
    return {
        "n": len(trades), "mean_r": r.mean(), "median_r": r.median(),
        "win_rate": (trades["pnl"] > 0).mean(),
        # a long straddle's payoff is right-skewed by construction; a positive
        # mean carried by two or three trades is not an edge (this is exactly
        # how the unconditional version's apparent MO profit fell apart)
        "mean_r_ex_top3": ranked.iloc[3:].mean() if len(r) > 3 else np.nan,
        "avg_days": trades["days_held"].mean(),
        "disc_n": len(disc),
        "disc_mean_r": (disc["pnl"] / disc["debit"]).mean() if len(disc) else np.nan,
        "hold_n": len(hold),
        "hold_mean_r": (hold["pnl"] / hold["debit"]).mean() if len(hold) else np.nan,
    }


def ungated_control(underlying: str, max_hold: int, profit_target: float | None,
                    lag_signal_days: int = 0) -> dict:
    """The same straddle, same exit rules, with the entry gate disabled --
    threshold=1.0 on a percentile signal admits every day, so the engine
    simply re-opens a straddle whenever it is flat.

    This is the control that actually isolates the signal's contribution.
    Comparing a gated result against the expiry-week baseline confounds two
    different things at once (the timing rule AND a different DTE/exit
    structure); comparing it against this holds the structure fixed and
    varies only whether the signal was consulted. If gated does not beat
    ungated, the signal added nothing and the P&L is coming from the
    straddle-buying regime, not from the timing."""
    res = ls.run(underlying, signal="rv_pct", threshold=1.0, max_hold_days=max_hold,
                 profit_target=profit_target, lag_signal_days=lag_signal_days)
    return _stats(res["trades"])


def stage_a(lag_signal_days: int = 0) -> pd.DataFrame:
    configs = frozen_configs()
    rows = []
    for u in TARGETS:
        for cfg in configs:
            res = ls.run(u, signal=cfg["signal"], threshold=cfg["threshold"],
                         max_hold_days=cfg["max_hold"], profit_target=cfg["profit_target"],
                         lag_signal_days=lag_signal_days)
            st = _stats(res["trades"])
            ctl = ungated_control(u, cfg["max_hold"], cfg["profit_target"], lag_signal_days)
            rows.append({"underlying": u, "name": ls.UNDERLYINGS[u],
                         "signal": cfg["signal"], "threshold": cfg["threshold"],
                         "max_hold": cfg["max_hold"],
                         "profit_target": ("none" if cfg["profit_target"] is None
                                           else cfg["profit_target"]),
                         "frozen_from": "+".join(cfg["from"]),
                         "null_baseline_r": NULL_BASELINE[u],
                         "ctl_n": ctl["n"], "ctl_mean_r": ctl["mean_r"],
                         "ctl_median_r": ctl["median_r"], "ctl_win_rate": ctl["win_rate"],
                         "edge_vs_ctl": st["mean_r"] - ctl["mean_r"], **st})
    df = pd.DataFrame(rows)

    print(f"\n{'=' * 108}")
    print("STAGE A -- live ETF long-straddle picks applied VERBATIM to CFFEX index options")
    print(f"no refitting; parameters frozen on other underlyings "
          f"(lag_signal_days={lag_signal_days})")
    print(f"{'=' * 108}")
    for u in TARGETS:
        g = df[df["underlying"] == u]
        print(f"\n{ls.UNDERLYINGS[u]} ({u})   "
              f"unconditional expiry-week straddle benchmark: {NULL_BASELINE[u]:+.1%} on premium")
        print(f"  {'signal':<11}{'thr':>6}{'hold':>6}{'pt':>6}{'n':>5}{'mean r':>9}"
              f"{'med r':>9}{'win%':>7}{'ex-top3':>9}{'disc r':>9}{'hold r':>9}"
              f" | {'ungated':>9}{'n':>5}{'EDGE':>9}")
        for _, x in g.iterrows():
            thin = " *thin*" if x["n"] < MIN_CELL_N else ""
            print(f"  {x['signal']:<11}{x['threshold']:>6.2f}{x['max_hold']:>6}"
                  f"{str(x['profit_target']):>6}{x['n']:>5}"
                  f"{x['mean_r']:>+8.1%}{x['median_r']:>+9.1%}{x['win_rate']:>6.0%}"
                  f"{x['mean_r_ex_top3']:>+9.1%}{x['disc_mean_r']:>+9.1%}"
                  f"{x['hold_mean_r']:>+9.1%}"
                  f" | {x['ctl_mean_r']:>+8.1%}{x['ctl_n']:>5}{x['edge_vs_ctl']:>+9.1%}{thin}")

    print(f"\n  ungated = same structure, entry gate disabled (every day admitted); EDGE = mean r - ungated.")
    print(f"  EDGE is what the signal itself contributed. A positive 'mean r' with a")
    print(f"  negative EDGE means the money came from being long straddles in that")
    print(f"  period, not from the timing rule.")
    print(f"  *thin* = fewer than MIN_CELL_N={MIN_CELL_N} trades; no conclusion may rest on it.")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "cffex_vol_signal_transfer_stage_a.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  Saved -> {out}")
    return df


def stage_b(lag_signal_days: int = 0) -> None:
    signals = [s for s in ["rv_pct", "iv_pct", "vrp", "pc_oi_pct", "pc_vol_pct"]]
    print(f"\n{'=' * 108}")
    print("STAGE B -- full discovery/holdout sweep on IO (000300) only")
    print("MO/HO have ~1.5y of pre-2024 discovery and are NOT swept -- see module docstring")
    print(f"{'=' * 108}")
    for sig in signals:
        ls.sweep("000300", sig, lag_signal_days=lag_signal_days)


# ============================================================ robustness checks
# Four diagnostics on IO's two frozen picks (IO_VRP_PICK / IO_IVPCT_PICK),
# requested after the transfer verdict to answer: is "candidate, not
# deployable" too generous, or does it still hold up? Each is a check, not a
# new backtest -- they read the same trades tables the picks already produce.

def cost_sensitivity(pick: dict, name: str) -> pd.DataFrame:
    """mean/median return-on-debit and trade count across SLIPPAGE_GRID.
    Uses long_straddle_rv_contraction_backtest.run()'s slippage parameter
    (added for this check): entry friction on both legs, exit friction on
    close-based exits only, none at cash-settled expiry -- same convention
    expiry_week_atm_straddle_backtest.py established. Trade COUNT can shift
    across the grid (higher slippage inflates the cost basis, which changes
    when profit_target is reached), which is itself informative, not noise
    -- reported alongside the returns rather than hidden."""
    rows = []
    for slip in SLIPPAGE_GRID:
        t = ls.run(pick["underlying"], signal=pick["signal"], threshold=pick["threshold"],
                   max_hold_days=pick["max_hold_days"], profit_target=pick["profit_target"],
                   slippage=slip)["trades"]
        st = _stats(t)
        rows.append({"pick": name, "slippage": slip, **st})
    df = pd.DataFrame(rows)
    print(f"\n{name}: return on debit vs. adverse per-leg slippage (index points)")
    print(f"  {'slippage':>9}{'n':>5}{'mean_r':>9}{'median_r':>10}{'win%':>7}{'ex-top3':>9}")
    for _, x in df.iterrows():
        print(f"  {x['slippage']:>9.2f}{x['n']:>5.0f}{x['mean_r']:>+8.1%}"
              f"{x['median_r']:>+9.1%}{x['win_rate']:>6.0%}{x['mean_r_ex_top3']:>+8.1%}")
    return df


def _entry_net_delta(underlying: str, entry_date: pd.Timestamp, expiry: pd.Timestamp,
                     strike: float) -> float:
    """call_delta + put_delta at entry, from the vendor's own C_DELTA (put
    already signed negative in this dataset -- checked directly, e.g. IO
    2021-02-05 5500-strike: call +0.487, put -0.508). NaN if either leg's
    vendor delta is missing (measured ~95% populated for CFFEX, not 100%)."""
    chain = cs.load_chain(underlying)
    day = chain[(chain[ds.C_DATE] == entry_date) & (chain[ds.C_EXPIRY] == expiry) &
               (chain[ds.C_STRIKE] == strike)]
    call = day[day[ds.C_CP] == "C"]
    put = day[day[ds.C_CP] == "P"]
    if call.empty or put.empty:
        return np.nan
    cd, pd_ = call[ds.C_DELTA].iloc[0], put[ds.C_DELTA].iloc[0]
    if pd.isna(cd) or pd.isna(pd_):
        return np.nan
    return float(cd) + float(pd_)


def directional_confound_check(pick: dict, name: str,
                               compare_pick: dict | None = None,
                               compare_name: str | None = None) -> dict:
    """Does the pick's P&L come from harvesting volatility (correlated with
    the MAGNITUDE of the underlying's move over the hold) or from a
    directional bias smuggled in through strike selection (correlated with
    the SIGN of the move)? This is exactly the confound
    csi1000_skew_risk_reversal_backtest.py found in a risk reversal (net
    delta ~+0.39, "profits" that were really a leveraged long-CSI1000 bet
    through a period CSI1000 happened to rise). A long straddle has no
    structural reason to carry that bias -- both legs are bought, not one
    bought and one sold -- but that is a claim to verify, not assume.

    compare_pick (optional): the same config run on a second underlying, as
    a cross-check. If IO's edge is really just "IO went up a lot in
    2025/2026" then whichever other CFFEX name shared that rally should show
    a similar direction correlation -- whether or not IT made money is a
    separate question from whether direction explains WHY."""
    t = ls.run(pick["underlying"], signal=pick["signal"], threshold=pick["threshold"],
              max_hold_days=pick["max_hold_days"], profit_target=pick["profit_target"])["trades"]
    t = t.copy()
    t["signed_ret"] = (t["exit_spot"] - t["entry_spot"]) / t["entry_spot"]
    t["abs_ret"] = t["signed_ret"].abs()
    t["r"] = t["pnl"] / t["debit"]
    t["net_delta"] = [
        _entry_net_delta(pick["underlying"], row.entry_date, row.expiry, row.strike)
        for row in t.itertuples()
    ]
    corr_signed = t["r"].corr(t["signed_ret"])
    corr_abs = t["r"].corr(t["abs_ret"])
    delta_stats = t["net_delta"].describe()

    print(f"\n{name}: directional confound check (n={len(t)})")
    print(f"  corr(return_on_debit, SIGNED underlying move)   = {corr_signed:+.3f}"
          f"  <- should be small if this is vol-harvesting, not direction")
    print(f"  corr(return_on_debit, |underlying move|)         = {corr_abs:+.3f}"
          f"  <- should be positive if this is genuinely long gamma")
    print(f"  entry net delta (call+put, vendor C_DELTA): mean={delta_stats['mean']:+.3f} "
          f"sd={delta_stats['std']:.3f} n={int(delta_stats['count'])}/{len(t)} non-null")

    out = {"pick": name, "n": len(t), "corr_signed": corr_signed, "corr_abs": corr_abs,
          "mean_net_delta": delta_stats["mean"], "sd_net_delta": delta_stats["std"]}

    if compare_pick is not None:
        c = ls.run(compare_pick["underlying"], signal=compare_pick["signal"],
                  threshold=compare_pick["threshold"], max_hold_days=compare_pick["max_hold_days"],
                  profit_target=compare_pick["profit_target"])["trades"].copy()
        c["signed_ret"] = (c["exit_spot"] - c["entry_spot"]) / c["entry_spot"]
        c["abs_ret"] = c["signed_ret"].abs()
        c["r"] = c["pnl"] / c["debit"]
        c_signed, c_abs = c["r"].corr(c["signed_ret"]), c["r"].corr(c["abs_ret"])
        print(f"\n  cross-check [{compare_name}] same config, n={len(c)}:")
        print(f"    corr(r, signed move) = {c_signed:+.3f}   corr(r, |move|) = {c_abs:+.3f}")
        if len(c) > 1:
            print(f"    underlying's own realized drift over these windows: "
                  f"mean signed move {c['signed_ret'].mean():+.1%} "
                  f"(vs {name.split()[0]}'s {t['signed_ret'].mean():+.1%})")
        out.update({"compare": compare_name, "compare_corr_signed": c_signed,
                   "compare_corr_abs": c_abs})

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = t[["entry_date", "exit_date", "strike", "entry_spot", "exit_spot",
                    "signed_ret", "abs_ret", "r", "net_delta"]]
    p = OUT_DIR / "cffex_vol_signal_robustness_direction.csv"
    trades_out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n  Per-trade detail -> {p}")
    return out


def lag_divergence_diagnostic(pick: dict, name: str,
                              boundary_epsilons: tuple[float, ...] = (0.01, 0.02, 0.03)) -> pd.DataFrame:
    """Which trades differ between lag=0 (same-day signal) and lag=1 (prior
    day's signal, this repo's stricter "rich/extreme" convention), and are
    they concentrated at the threshold boundary? A trade that only exists
    under one lag convention because the signal crossed the threshold WITHIN
    that single day is not something a real order could have captured either
    way -- same-day info isn't tradeable, and a one-day-later fill would see
    a different (possibly non-triggering) value. If most of the flipped
    trades' P&L sits within a few points of the threshold, the lag=0 grid's
    apparent edge is largely this same-bar-execution artifact, not a real
    margin."""
    sig = ls._signal_series(pick["underlying"], pick["signal"])
    t0 = ls.run(**pick, lag_signal_days=0)["trades"].copy()
    t1 = ls.run(**pick, lag_signal_days=1)["trades"].copy()
    t0["lag"], t1["lag"] = 0, 1

    only0 = t0[~t0["entry_date"].isin(t1["entry_date"])]
    only1 = t1[~t1["entry_date"].isin(t0["entry_date"])]
    flipped = pd.concat([only0, only1], ignore_index=True)

    dates = sorted(sig.index)
    def _prior_signal(dt: pd.Timestamp) -> float:
        idx = [d for d in dates if d < dt]
        return sig.loc[idx[-1]] if idx else np.nan

    flipped = flipped.copy()
    flipped["signal_same_day"] = flipped["entry_date"].map(lambda d: sig.get(d, np.nan))
    flipped["signal_prior_day"] = flipped["entry_date"].map(_prior_signal)
    flipped["dist_to_threshold"] = (flipped[["signal_same_day", "signal_prior_day"]]
                                    .min(axis=1) - pick["threshold"]).abs()

    print(f"\n{name}: lag=0 vs lag=1 divergence (threshold={pick['threshold']:.2f})")
    print(f"  lag=0 n={len(t0)}  lag=1 n={len(t1)}  "
          f"only-in-lag0={len(only0)}  only-in-lag1={len(only1)}")
    for eps in boundary_epsilons:
        boundary = flipped[flipped["dist_to_threshold"] < eps]
        share_n = len(boundary) / len(flipped) if len(flipped) else np.nan
        pnl_all = flipped["pnl"].abs().sum()
        share_pnl = boundary["pnl"].abs().sum() / pnl_all if pnl_all else np.nan
        print(f"  epsilon={eps:.2f}: {len(boundary)}/{len(flipped)} flipped trades "
              f"({share_n:.0%}) are within {eps:.2f} of the threshold, "
              f"accounting for {share_pnl:.0%} of flipped trades' |pnl|")

    OUT_DIR.mkdir(exist_ok=True)
    p = OUT_DIR / "cffex_vol_signal_robustness_lag_divergence.csv"
    flipped.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  Flipped-trade detail -> {p}")
    return flipped


def leave_one_year_out(pick: dict, name: str) -> pd.DataFrame:
    """Recompute mean_r/median_r/win_rate/total_pnl with each entry-year's
    trades excluded once. If dropping any single year flips the sign or the
    median, the headline number is not resting on a broad base -- this is
    the same check that unmasked the unconditional expiry-week straddle's
    apparent MO profit as three specific months (expiry_week_straddle_
    investigation.md section 3.1)."""
    t = ls.run(pick["underlying"], signal=pick["signal"], threshold=pick["threshold"],
              max_hold_days=pick["max_hold_days"], profit_target=pick["profit_target"])["trades"].copy()
    t["year"] = t["entry_date"].dt.year
    t["r"] = t["pnl"] / t["debit"]

    full = _stats(t)
    total_pnl = t["pnl"].sum()
    mult = ls._multiplier(pick["underlying"])
    by_year_pnl = t.groupby("year")["pnl"].sum()

    rows = [{"excluded_year": "none (full sample)", "n": full["n"],
            "mean_r": full["mean_r"], "median_r": full["median_r"],
            "win_rate": full["win_rate"], "total_pnl_yuan": total_pnl * mult}]
    for y, g in t.groupby("year"):
        rest = t[t["year"] != y]
        st = _stats(rest)
        rows.append({"excluded_year": y, "n": st["n"], "mean_r": st["mean_r"],
                    "median_r": st["median_r"], "win_rate": st["win_rate"],
                    "total_pnl_yuan": rest["pnl"].sum() * mult,
                    "this_years_pnl_share": g["pnl"].sum() / total_pnl if total_pnl else np.nan})
    df = pd.DataFrame(rows)

    print(f"\n{name}: leave-one-year-out (full sample n={full['n']}, "
          f"mean_r={full['mean_r']:+.1%}, median_r={full['median_r']:+.1%})")
    print(f"  {'excluded':>12}{'n':>5}{'mean_r':>9}{'median_r':>10}{'win%':>7}{'yr pnl share':>14}")
    for _, x in df.iterrows():
        share = x.get("this_years_pnl_share", np.nan)
        share_s = f"{share:+.0%}" if pd.notna(share) else ""
        flag = "  <- sign flips" if pd.notna(share) and (
            (full["mean_r"] > 0) != (x["mean_r"] > 0)) else ""
        print(f"  {str(x['excluded_year']):>12}{x['n']:>5.0f}{x['mean_r']:>+8.1%}"
              f"{x['median_r']:>+9.1%}{x['win_rate']:>6.0%}{share_s:>14}{flag}")

    # No single year's exclusion flips the sign above -- but per-year shares
    # can each be under ~55% while still jointly accounting for essentially
    # all the profit if two years' shares happen to sum near 100%. That is
    # exactly what happens here and the single-year table above cannot show
    # it, so check it explicitly: cumulative share of the top-K years by
    # contribution, and the result of excluding the top-2 together.
    ranked = by_year_pnl.sort_values(ascending=False)
    cum_share = (ranked / total_pnl).cumsum() if total_pnl else ranked * np.nan
    print(f"\n  Cumulative P&L share by year, ranked (top-2 excluded jointly below):")
    for y, share in cum_share.items():
        print(f"    {y}: this year {ranked[y]/total_pnl:+.1%}, cumulative {share:+.1%}")

    top2_years = list(ranked.index[:2])
    rest2 = t[~t["year"].isin(top2_years)]
    st2 = _stats(rest2)
    print(f"\n  Excluding BOTH top-2 years ({top2_years}): n={st2['n']}  "
          f"mean_r={st2['mean_r']:+.1%}  median_r={st2['median_r']:+.1%}  "
          f"win%={st2['win_rate']:.0%}  total_pnl={rest2['pnl'].sum()*mult:+,.0f} yuan")
    if rest2["pnl"].sum() * mult < 0.10 * (total_pnl * mult):
        print(f"  -> the other {len(ranked)-2} years combined are close to breakeven; "
              f"almost the entire result rests on {top2_years}")

    OUT_DIR.mkdir(exist_ok=True)
    p = OUT_DIR / "cffex_vol_signal_robustness_year_loo.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"  Saved -> {p}")
    return df


def robustness() -> None:
    print(f"\n{'=' * 108}")
    print("ROBUSTNESS CHECKS -- IO's two frozen picks (vrp: candidate; iv_pct: unconfirmed)")
    print(f"{'=' * 108}")

    print(f"\n{'-' * 108}\n1. COST SENSITIVITY\n{'-' * 108}")
    cost_vrp = cost_sensitivity(IO_VRP_PICK, "IO vrp<=0.00/30d/+50%")
    cost_iv = cost_sensitivity(IO_IVPCT_PICK, "IO iv_pct<=0.30/20d/+200%")
    pd.concat([cost_vrp, cost_iv], ignore_index=True).to_csv(
        OUT_DIR / "cffex_vol_signal_robustness_cost.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'-' * 108}\n2. DIRECTIONAL CONFOUND CHECK\n{'-' * 108}")
    mo_vrp_pick = {**IO_VRP_PICK, "underlying": "000852"}
    directional_confound_check(IO_VRP_PICK, "IO vrp<=0.00/30d/+50%",
                               compare_pick=mo_vrp_pick, compare_name="MO, same config")

    print(f"\n{'-' * 108}\n3. iv_pct LAG=0 vs LAG=1 DIVERGENCE\n{'-' * 108}")
    lag_divergence_diagnostic(IO_IVPCT_PICK, "IO iv_pct<=0.30/20d/+200%")

    print(f"\n{'-' * 108}\n4. vrp LEAVE-ONE-YEAR-OUT\n{'-' * 108}")
    leave_one_year_out(IO_VRP_PICK, "IO vrp<=0.00/30d/+50%")

    print(f"\n{'=' * 108}")
    print(f"All robustness CSVs saved under {OUT_DIR}/cffex_vol_signal_robustness_*.csv")
    print(f"{'=' * 108}")


def main(stage: str, lag_signal_days: int) -> None:
    if stage in ("a", "both"):
        stage_a(lag_signal_days)
    if stage in ("b", "both"):
        stage_b(lag_signal_days)
    if stage == "robustness":
        robustness()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="a", choices=["a", "b", "both", "robustness"])
    ap.add_argument("--lag-signal-days", type=int, default=0, dest="lag_signal_days",
                    help="gate on the signal from N trading days earlier (robustness)")
    args = ap.parse_args()
    main(args.stage, args.lag_signal_days)
