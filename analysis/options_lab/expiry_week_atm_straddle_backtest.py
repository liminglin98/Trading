# -*- coding: utf-8 -*-
"""expiry_week_atm_straddle_backtest.py -- 到期周平值双买 (expiry-week ATM long
straddle) on the three CFFEX index options: IO (沪深300/000300), MO
(中证1000/000852), HO (上证50/000016).

The trade being tested, as asked: 股指期权每月第三个周五到期; on the 周一 /
周三 / 周五 of that expiry week, buy an at-the-money straddle (双买 = long
call + long put, same strike) on the **当月** contract -- the one expiring
that same Friday. This is a pure gamma-vs-theta question with nothing else
in it: expiry-week ATM options carry the most gamma and bleed the most
theta, and which side wins is an empirical matter, not something to reason
out from first principles. There is no signal, no filter, no discretion --
entry is decided by the calendar alone, which is exactly why it is worth
measuring: it is the null hypothesis that every signal-gated long-straddle
script in this lab (iv_low_long_straddle_backtest.py,
long_straddle_rv_contraction_backtest.py) should have to beat.

Unlike every other backtest here, this one holds positions that OVERLAP (a
Monday leg held to expiry is still open when the Wednesday leg opens), so
it does not use the lab's usual single-`open_pos` sequential engine. It
produces a flat per-trade table instead (the shape
analysis/macro_model/ma_signal_put_backtest.py's run_trades() uses), and
builds a sequential equity curve only per (entry weekday, exit rule) cell,
where by construction there is at most one open position at a time.

Data: Data/买断日更/consolidated/{000300,000852,000016}.XSHG.parquet, read
through consolidated_store.load_chain() -- never read directly. Two things
that loader provides and this script depends on:
  - ds.C_UND_CLOSE is a **put-call-parity forward** per (date, expiry), not
    a stored index level (the dataset has no spot column, and the ETF OHLC
    caches are 前复权-adjusted, hence incomparable to raw strikes -- see
    consolidated_store.py's module docstring). At DTE=0 that forward
    reduces exactly to the settlement-consistent spot, which is what makes
    the expiry-day intrinsic cross-check below meaningful.
  - ds.C_IV is all-NaN for this dataset. Nothing here needs IV.

Coverage: IO 2019-12-23 -> 2026-08-14 (84 expiries), MO 2022-07-22 ->
(54), HO 2022-12-19 -> (49).

Entry-day definition -- BOTH are run, and the difference is a result:
  - calendar (the literal reading): the Monday / Wednesday / Friday of the
    expiry week. If one of those is a holiday the leg is skipped and
    counted, never substituted with a neighbouring trading day -- a
    substitute would change DTE and quietly contaminate the comparison.
  - tradingday (robustness): T-4 / T-2 / T-0 **trading** days before
    expiry. In an ordinary week these are the same three dates, but this
    basis also recovers the ~12 expiry weeks with a holiday and the 3
    expiries that are not on a Friday at all (2024-02-19, 2026-02-24,
    2026-06-22 -- exchange-shifted, verified against the data, not
    assumed). If a conclusion only holds under one basis, it is not a
    conclusion.

Exit rules -- all three are run side by side:
  | rule      | Mon leg      | Wed leg      | Fri leg       |
  | to_expiry | Fri delivery | Fri delivery | same-day      |
  | next_obs  | Wed close    | Fri close    | same-day      |
  | one_day   | Tue close    | Thu close    | same-day      |
The Friday leg is 0DTE open-and-settle and is therefore IDENTICAL under all
three rules. That is stated in the report rather than presented as three
independent results.

**The Friday leg is not a tradeable result and must not be read as one.**
CFFEX settles index options at the arithmetic average of the underlying
index over the LAST TWO HOURS of the expiry day, and trading ends at 15:00
-- so an entry at the 15:00 close is an entry into a payoff that is already
fully determined. Measured (fri_leg_diagnostic()): entry_date == exit_date
for every one of these, and the P&L is exactly the same-day close-minus-
settlement basis, mean -0.00/+0.08/-0.07 index points with SD ~0.2-0.3 on
IO/MO/HO -- noise around zero, with no market exposure in it at all. The
"+1.8% mean return on debit" this produces for IO is that ±0.3-point basis
divided by a ~12-point debit, not an edge. Part of it is pure stale-quote
artifact: the ~worthless leg's last traded tick is a median 0.20 points
while its settlement value is 0.00. Testing a real 0DTE Friday entry needs
intraday quotes (Data/options/minute/index_option_minute_*.csv, via
minute_updater.py -- but that only starts 2026-08-11), not daily bars.

Marks: mid-life exits at ds.C_CLOSE (this lab's convention -- fills at the
daily close). Expiry exits at the expiring contract's own ds.C_SETTLE on
the expiry date, which IS the cash delivery value; validate_settlement()
cross-checks every one of those against intrinsic recomputed from the
DTE=0 parity forward and reports the worst disagreement rather than
silently preferring one source.

Costs: the lab models none by convention (README.md: "Fills are assumed at
daily close; no fees/slippage ... treat results as upper bounds"), so the
headline numbers follow suit and stay comparable to every other script
here. But a round trip on this structure is 4 legs and an ATM 0DTE MO
straddle can cost only ~60 index points, so 1 point per leg is ~7% of the
debit -- far too big to leave as a footnote. --slippage / the sensitivity
table therefore reruns everything at 0 / 0.2 / 0.5 / 1.0 index points of
adverse half-spread per leg (CFFEX tick is 0.2). Slippage is charged on
entry and on close-based exits, NOT on cash delivery at expiry (there is
no spread to cross when the exchange settles you out).

Return basis: return on **debit** (premium paid), not on margin. A long
option position posts no margin, and cffex_margin.py only carries verified
MARGIN_PARAMS for MO -- IO/HO are an explicit un-sourced TODO there, so any
return-on-margin figure for those two would be fiction. Debit is the true
capital at risk. Charts use compounding=False for the same reason
long_straddle_chart.py does.

Discovery/holdout: SPLIT_DATE 2024-01-01, this repo's standard
(condor_delta_investigation_summary.md). Note this script has NO parameters
to fit -- there is no grid search and nothing is frozen on discovery, so
the split here is a stability read, not an out-of-sample test.

Usage:
    python expiry_week_atm_straddle_backtest.py                    # all 3, all rules, both bases
    python expiry_week_atm_straddle_backtest.py --underlying 000852
    python expiry_week_atm_straddle_backtest.py --entry-basis tradingday
    python expiry_week_atm_straddle_backtest.py --slippage 0.5
    python expiry_week_atm_straddle_backtest.py --no-costs         # skip the sensitivity table
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cffex_margin as cm
import consolidated_store as cs
import data_store as ds
import quarterly_atm_short_straddle as qs

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = pd.Timestamp("2024-01-01")  # condor_delta_investigation_summary.md convention

UNDERLYINGS = {
    "000300": "IO 沪深300",
    "000852": "MO 中证1000",
    "000016": "HO 上证50",
}
# matplotlib has no CJK font in this environment (checked: fontManager carries
# 21 families, none with CJK coverage), so PNG titles use ASCII names. Console
# output and the CSVs keep the Chinese names.
PLOT_NAMES = {"000300": "IO (CSI300)", "000852": "MO (CSI1000)", "000016": "HO (SSE50)"}

MULTIPLIER = cm.MULTIPLIER  # 100 yuan per index point -- CFFEX spec, same for IO/MO/HO
TICK = 0.2                  # CFFEX index-option minimum price increment, index points
SLIPPAGE_GRID = [0.0, 0.2, 0.5, 1.0]  # adverse half-spread per leg, index points

# Entry weekday labels -> (calendar weekday, trading days before expiry).
ENTRY_DAYS = {"Mon": (0, 4), "Wed": (2, 2), "Fri": (4, 0)}
ENTRY_ORDER = ["Mon", "Wed", "Fri"]
EXIT_RULES = ["to_expiry", "next_obs", "one_day"]

# Exchange-shifted expiries that are genuinely not a third Friday, verified
# against the parquet on 2026-08-17. validate_expiries() raises on a fourth.
KNOWN_NON_FRIDAY_EXPIRIES = {
    pd.Timestamp("2024-02-19"),
    pd.Timestamp("2026-02-24"),
    pd.Timestamp("2026-06-22"),
}

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"
C_AQUA = "#1baf7a"
ENTRY_COLORS = {"Mon": C_BLUE, "Wed": C_AQUA, "Fri": C_RED}


# ------------------------------------------------------------------ calendar
def third_friday(ts: pd.Timestamp) -> pd.Timestamp:
    """Third Friday of ts's calendar month."""
    first = pd.Timestamp(year=ts.year, month=ts.month, day=1)
    first_friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + pd.Timedelta(days=14)


def validate_expiries(expiries: list[pd.Timestamp], underlying: str) -> list[pd.Timestamp]:
    """Every expiry must be its month's third Friday, or one of the three
    known exchange-shifted dates. A fourth exception means the data or this
    assumption changed -- raise rather than silently backtest something
    other than what the user asked for."""
    unexpected = [e for e in expiries
                  if e != third_friday(e) and e not in KNOWN_NON_FRIDAY_EXPIRIES]
    if unexpected:
        raise ValueError(
            f"{underlying}: {len(unexpected)} expiry date(s) are neither the month's "
            f"third Friday nor a known shifted expiry: "
            f"{[str(e.date()) for e in unexpected[:8]]} -- "
            f"add them to KNOWN_NON_FRIDAY_EXPIRIES only after confirming with CFFEX.")
    return expiries


# ------------------------------------------------------------------ chain prep
@lru_cache(maxsize=4)
def _prepped_chain(underlying: str) -> pd.DataFrame:
    """Full chain with a usable close. Adjusted-contract filtering (the
    `名称 endswith "A"` rule the ETF scripts use) does not apply to CFFEX --
    index options are cash-settled and never adjusted."""
    chain = cs.load_chain(underlying)
    return chain[chain[ds.C_CLOSE].notna()].copy()


def _is_liquid(row: pd.Series) -> bool:
    """Volume is missing for the `akshare_cffex_spot` rows that make up part
    of the 2026-07-28-onward tail, and open interest is missing for the
    `akshare_cffex_daily` rows -- so a naive `volume > 0` filter would
    silently drop the most recent month. Use whichever of the two the row
    actually carries; only reject when a present field says zero."""
    vol, oi = row.get(ds.C_VOL), row.get(ds.C_OI)
    if pd.notna(vol):
        return float(vol) > 0
    if pd.notna(oi):
        return float(oi) > 0
    return True


def _atm_strike_liquid(day: pd.DataFrame, expiry: pd.Timestamp, fwd: float) -> float | None:
    """Strike minimising |K - forward| among strikes where BOTH legs are
    quoted and tradeable. Ties (equidistant strikes, which happens whenever
    the forward sits exactly between two ladder rungs) break toward the
    higher combined volume/OI, not toward the arbitrary first one."""
    exp_day = day[day[ds.C_EXPIRY] == expiry]
    best: tuple[float, float, float] | None = None  # (dist, -activity, strike)
    for strike in exp_day[ds.C_STRIKE].unique():
        call = qs._leg_row(exp_day, expiry, "C", strike)
        put = qs._leg_row(exp_day, expiry, "P", strike)
        if call is None or put is None:
            continue
        if not (_is_liquid(call) and _is_liquid(put)):
            continue
        activity = float(np.nansum([call.get(ds.C_VOL), put.get(ds.C_VOL),
                                    call.get(ds.C_OI), put.get(ds.C_OI)]))
        key = (abs(float(strike) - fwd), -activity, float(strike))
        if best is None or key < best:
            best = key
    return best[2] if best else None


# ------------------------------------------------------------------ entry days
def _entry_dates(dates: list[pd.Timestamp], expiry: pd.Timestamp,
                 basis: str) -> dict[str, pd.Timestamp | None]:
    """Map Mon/Wed/Fri -> the trading date to enter on, or None if that leg
    does not exist under this basis (holiday, or a shifted expiry that
    lands mid-week)."""
    if basis == "calendar":
        week_start = expiry - pd.Timedelta(days=expiry.weekday())
        out = {}
        for label, (weekday, _) in ENTRY_DAYS.items():
            want = week_start + pd.Timedelta(days=weekday)
            # never enter after the contract has already settled
            out[label] = want if (want in dates and want <= expiry) else None
        return out

    if basis == "tradingday":
        try:
            idx = dates.index(expiry)
        except ValueError:
            return {label: None for label in ENTRY_DAYS}
        return {label: (dates[idx - offset] if idx - offset >= 0 else None)
                for label, (_, offset) in ENTRY_DAYS.items()}

    raise ValueError(f"unknown entry basis {basis!r}")


def _exit_date(rule: str, label: str, entry: pd.Timestamp, expiry: pd.Timestamp,
               entries: dict[str, pd.Timestamp | None],
               dates: list[pd.Timestamp]) -> pd.Timestamp:
    """Exit date for one (rule, leg). Always clamped at expiry -- these are
    当月 contracts, they cannot be held past delivery."""
    if rule == "to_expiry" or label == "Fri":
        return expiry
    if rule == "next_obs":
        nxt = ENTRY_ORDER[ENTRY_ORDER.index(label) + 1:]
        for nxt_label in nxt:
            cand = entries.get(nxt_label)
            if cand is not None and cand > entry:
                return min(cand, expiry)
        return expiry
    if rule == "one_day":
        i = dates.index(entry)
        return min(dates[i + 1], expiry) if i + 1 < len(dates) else expiry
    raise ValueError(f"unknown exit rule {rule!r}")


# ------------------------------------------------------------------ engine
@lru_cache(maxsize=8)
def _raw_trades(underlying: str, basis: str = "calendar") -> pd.DataFrame:
    """Flat per-trade table at RAW (zero-slippage) marks: one row per
    (expiry, entry weekday, exit rule). Contract selection does not depend
    on slippage, so this is computed once per (underlying, basis) and the
    cost grid is applied on top by apply_costs() -- otherwise the ATM-pair
    scan would rerun for every point of the sensitivity table."""
    chain = _prepped_chain(underlying)
    dates = sorted(pd.Timestamp(d) for d in chain[ds.C_DATE].unique())
    by_date = {pd.Timestamp(d): g for d, g in chain.groupby(ds.C_DATE)}
    expiries = validate_expiries(
        sorted(pd.Timestamp(e) for e in chain[ds.C_EXPIRY].dropna().unique()), underlying)

    rows: list[dict] = []
    skips: list[dict] = []

    for expiry in expiries:
        if expiry not in by_date:  # expiry falls after the data ends
            continue
        entries = _entry_dates(dates, expiry, basis)

        for label in ENTRY_ORDER:
            entry = entries[label]
            if entry is None:
                skips.append({"expiry": expiry, "entry_day": label, "why": "no trading day"})
                continue

            day = by_date[entry]
            exp_day = day[day[ds.C_EXPIRY] == expiry]
            if exp_day.empty:
                skips.append({"expiry": expiry, "entry_day": label, "why": "contract not listed"})
                continue
            fwd_series = exp_day[ds.C_UND_CLOSE].dropna()
            if fwd_series.empty:
                skips.append({"expiry": expiry, "entry_day": label, "why": "no parity forward"})
                continue
            fwd = float(fwd_series.iloc[0])

            strike = _atm_strike_liquid(day, expiry, fwd)
            if strike is None:
                skips.append({"expiry": expiry, "entry_day": label, "why": "no tradeable ATM pair"})
                continue

            call_in = qs._leg_row(day, expiry, "C", strike)
            put_in = qs._leg_row(day, expiry, "P", strike)

            for rule in EXIT_RULES:
                exit_dt = _exit_date(rule, label, entry, expiry, entries, dates)
                out = _exit_value(by_date, exit_dt, expiry, strike, fwd)
                if out is None:
                    skips.append({"expiry": expiry, "entry_day": label,
                                  "why": f"no exit mark ({rule})"})
                    continue
                call_out, put_out, exit_fwd, at_delivery = out

                rows.append({
                    "underlying": underlying,
                    "name": UNDERLYINGS[underlying],
                    "basis": basis,
                    "entry_day": label,
                    "exit_rule": rule,
                    "expiry": expiry,
                    "expiry_month": expiry.strftime("%Y-%m"),
                    "entry_date": entry,
                    "exit_date": exit_dt,
                    "dte_at_entry": int((expiry - entry).days),
                    "days_held": dates.index(exit_dt) - dates.index(entry),
                    "strike": strike,
                    "entry_fwd": fwd,
                    "exit_fwd": exit_fwd,
                    "moneyness": fwd / strike - 1.0,
                    "call_entry_raw": float(call_in[ds.C_CLOSE]),
                    "put_entry_raw": float(put_in[ds.C_CLOSE]),
                    "call_exit_raw": call_out, "put_exit_raw": put_out,
                    "at_delivery": at_delivery,
                    "entry_vol": call_in.get(ds.C_VOL), "entry_oi": call_in.get(ds.C_OI),
                })

    trades = pd.DataFrame(rows)
    # plain list of dicts, not a DataFrame -- pandas compares .attrs values with
    # == during merge/concat __finalize__, which raises on a nested DataFrame
    trades.attrs["skips"] = skips
    trades.attrs["n_expiries"] = len(expiries)
    return trades


def apply_costs(raw: pd.DataFrame, slippage: float = 0.0) -> pd.DataFrame:
    """Charge `slippage` index points of adverse half-spread per leg and
    derive debit / P&L / return-on-debit. Paid on entry (buy the offer) and
    on close-based exits (sell the bid), but NOT on cash delivery at expiry
    -- there is no spread to cross when the exchange settles you out."""
    t = raw.copy()
    t["call_entry"] = t["call_entry_raw"] + slippage
    t["put_entry"] = t["put_entry_raw"] + slippage
    exit_slip = np.where(t["at_delivery"], 0.0, slippage)
    t["call_exit"] = np.maximum(t["call_exit_raw"] - exit_slip, 0.0)
    t["put_exit"] = np.maximum(t["put_exit_raw"] - exit_slip, 0.0)
    t["slippage"] = slippage
    t["debit"] = t["call_entry"] + t["put_entry"]
    t["debit_yuan"] = t["debit"] * MULTIPLIER
    t["pnl"] = t["call_exit"] + t["put_exit"] - t["debit"]
    t["pnl_yuan"] = t["pnl"] * MULTIPLIER
    t["return_on_debit"] = np.where(t["debit"] > 0, t["pnl"] / t["debit"], np.nan)
    t.attrs["skips"] = raw.attrs.get("skips", [])
    return t


def run(underlying: str, basis: str = "calendar", slippage: float = 0.0) -> pd.DataFrame:
    """Priced per-trade table for one underlying."""
    return apply_costs(_raw_trades(underlying, basis), slippage)


def _exit_value(by_date: dict, exit_dt: pd.Timestamp, expiry: pd.Timestamp,
                strike: float, entry_fwd: float):
    """(call_px, put_px, forward_at_exit, at_delivery) or None.

    On the expiry date the exchange's own ds.C_SETTLE for the expiring
    contract IS the cash delivery value, so that is what is used; intrinsic
    from the DTE=0 parity forward is computed alongside purely so
    validate_settlement() can compare the two. Mid-life exits mark at
    ds.C_CLOSE, this lab's convention."""
    day = by_date.get(exit_dt)
    if day is None:
        return None
    exp_day = day[day[ds.C_EXPIRY] == expiry]
    if exp_day.empty:
        return None
    fwd_series = exp_day[ds.C_UND_CLOSE].dropna()
    fwd = float(fwd_series.iloc[0]) if not fwd_series.empty else np.nan

    at_delivery = exit_dt >= expiry
    col = ds.C_SETTLE if at_delivery else ds.C_CLOSE
    legs = {}
    for cp in ("C", "P"):
        m = exp_day[(exp_day[ds.C_CP] == cp) & (exp_day[ds.C_STRIKE] == strike)]
        if not m.empty and pd.notna(m[col].iloc[0]):
            legs[cp] = float(m[col].iloc[0])
        elif at_delivery and np.isfinite(fwd):
            legs[cp] = qs._intrinsic(cp, strike, fwd)  # contract row gone; settle at intrinsic
        else:
            return None
    return legs["C"], legs["P"], fwd, at_delivery


# ------------------------------------------------------------------ validation
def validate_settlement(underlying: str, tol: float = TICK) -> dict:
    """Every DTE=0 ds.C_SETTLE vs. intrinsic recomputed from the parity
    forward. _parity_forward's docstring claims e^(r*0)=1 makes the DTE=0
    forward the exact settlement-consistent spot -- this is the check that
    the claim holds on these three underlyings before any P&L depends on
    it."""
    chain = _prepped_chain(underlying)
    d0 = chain[(chain[ds.C_DTE] == 0) & chain[ds.C_UND_CLOSE].notna()
               & chain[ds.C_SETTLE].notna()].copy()
    if d0.empty:
        return {"n": 0}
    intrinsic = np.where(d0[ds.C_CP] == "C",
                         np.maximum(d0[ds.C_UND_CLOSE] - d0[ds.C_STRIKE], 0.0),
                         np.maximum(d0[ds.C_STRIKE] - d0[ds.C_UND_CLOSE], 0.0))
    d0["diff"] = (d0[ds.C_SETTLE] - intrinsic).abs()
    # Only the near-the-money strikes are ever traded by this script, and a
    # perfect match is not expected even there: settle is struck off the
    # 交割结算价 (index average over the last two hours) while the parity
    # forward is backed out of closing option prices. What matters is that
    # the gap stays small in index points where we actually transact.
    near = d0[(d0[ds.C_STRIKE] - d0[ds.C_UND_CLOSE]).abs()
              <= 0.02 * d0[ds.C_UND_CLOSE]]
    return {"n": len(d0), "max_abs_diff": float(d0["diff"].max()),
            "median_abs_diff": float(d0["diff"].median()),
            "pct_within_tick": float((d0["diff"] <= tol).mean()),
            "atm_n": len(near),
            "atm_median_abs_diff": float(near["diff"].median()) if len(near) else np.nan,
            "atm_max_abs_diff": float(near["diff"].max()) if len(near) else np.nan}


def fri_leg_diagnostic(underlying: str, basis: str = "calendar") -> dict:
    """Shows that the Friday leg carries no market exposure: entry and exit
    are the same date, so its P&L is identically the close-minus-settlement
    basis. See the module docstring -- this is the check behind that claim,
    not a strategy statistic."""
    t = _raw_trades(underlying, basis)
    f = t[(t["entry_day"] == "Fri") & (t["exit_rule"] == "to_expiry")]
    if f.empty:
        return {"n": 0}
    b = (f["call_exit_raw"] + f["put_exit_raw"]) - (f["call_entry_raw"] + f["put_entry_raw"])
    return {"n": len(f), "all_same_day": bool((f["entry_date"] == f["exit_date"]).all()),
            "mean_basis_pts": float(b.mean()), "sd_basis_pts": float(b.std()),
            "median_debit_pts": float((f["call_entry_raw"] + f["put_entry_raw"]).median()),
            "median_worthless_leg_tick": float(
                np.minimum(f["call_entry_raw"], f["put_entry_raw"]).median())}


def validate_basis_agreement(underlying: str) -> dict:
    """calendar and tradingday must produce byte-identical trades on every
    (expiry, entry_day) both bases resolved to the same date. This is the
    cross-check between two independently written entry-selection paths."""
    a = run(underlying, basis="calendar")
    b = run(underlying, basis="tradingday")
    key = ["expiry", "entry_day", "exit_rule"]
    merged = a.merge(b, on=key, suffixes=("_c", "_t"))
    same_entry = merged[merged["entry_date_c"] == merged["entry_date_t"]]
    if same_entry.empty:
        return {"n_shared": 0}
    mismatch = same_entry[
        ~np.isclose(same_entry["pnl_c"], same_entry["pnl_t"], atol=1e-9)]
    return {"n_shared": len(same_entry), "n_mismatch": len(mismatch),
            "n_calendar": len(a), "n_tradingday": len(b)}


# ------------------------------------------------------------------ reporting
def _cell_stats(g: pd.DataFrame) -> dict:
    """Per-cell stats. A long straddle's payoff is right-skewed by
    construction -- most months bleed a little and the occasional gap pays
    for all of them -- so a positive mean_r means nothing on its own. The
    ex-top-N columns exist to answer the only question that matters for a
    cell that looks profitable: is this an edge, or is it one or three
    lucky months? mean_r_ex_top3 going negative is the standard signature
    of the latter."""
    r = g["return_on_debit"]
    disc, hold = g[g["entry_date"] < SPLIT_DATE], g[g["entry_date"] >= SPLIT_DATE]
    ranked = g.sort_values("pnl", ascending=False)
    return {
        "n": len(g),
        "total_pnl_yuan": g["pnl_yuan"].sum(),
        "mean_r": r.mean(),
        "median_r": r.median(),
        "win_rate": (g["pnl"] > 0).mean(),
        "avg_debit_yuan": g["debit_yuan"].mean(),
        "best_r": r.max(),
        "worst_r": r.min(),
        "mean_r_ex_top1": ranked["return_on_debit"].iloc[1:].mean() if len(g) > 1 else np.nan,
        "mean_r_ex_top3": ranked["return_on_debit"].iloc[3:].mean() if len(g) > 3 else np.nan,
        "top1_pnl_share": (ranked["pnl_yuan"].iloc[0] / g["pnl_yuan"].sum()
                           if g["pnl_yuan"].sum() > 0 else np.nan),
        "disc_n": len(disc), "disc_mean_r": disc["return_on_debit"].mean() if len(disc) else np.nan,
        "hold_n": len(hold), "hold_mean_r": hold["return_on_debit"].mean() if len(hold) else np.nan,
    }


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (u, label, rule), g in trades.groupby(["underlying", "entry_day", "exit_rule"]):
        rows.append({"underlying": u, "name": UNDERLYINGS[u], "entry_day": label,
                     "exit_rule": rule, **_cell_stats(g)})
    out = pd.DataFrame(rows)
    out["entry_day"] = pd.Categorical(out["entry_day"], ENTRY_ORDER, ordered=True)
    out["exit_rule"] = pd.Categorical(out["exit_rule"], EXIT_RULES, ordered=True)
    return out.sort_values(["underlying", "entry_day", "exit_rule"]).reset_index(drop=True)


def report(trades: pd.DataFrame, basis: str, slippage: float) -> pd.DataFrame:
    summary = summarize(trades)
    slip_note = "no fees/slippage (lab convention)" if slippage == 0 else \
        f"{slippage:.2f} index pts adverse half-spread per leg"
    print(f"\n{'=' * 78}")
    print(f"到期周平值双买 (expiry-week ATM long straddle) -- entry basis: {basis}")
    print(f"当月合约, 1 call + 1 put, multiplier {MULTIPLIER} yuan/pt, {slip_note}")
    print(f"{'=' * 78}")

    for u, g in summary.groupby("underlying", sort=False):
        tr = trades[trades["underlying"] == u]
        print(f"\n{UNDERLYINGS[u]} ({u})   "
              f"{tr['entry_date'].min().date()} -> {tr['exit_date'].max().date()}   "
              f"avg debit {tr['debit_yuan'].mean():,.0f} yuan")
        print(f"  {'entry':<6} {'exit rule':<10} {'n':>4} {'total P&L':>12} "
              f"{'mean r':>9} {'median r':>9} {'win%':>6} {'ex-top3':>9} "
              f"{'disc r':>9} {'hold r':>9}")
        for _, row in g.iterrows():
            print(f"  {row['entry_day']:<6} {row['exit_rule']:<10} {row['n']:>4} "
                  f"{row['total_pnl_yuan']:>12,.0f} {row['mean_r']:>+8.1%} "
                  f"{row['median_r']:>+8.1%} {row['win_rate']:>5.0%} "
                  f"{row['mean_r_ex_top3']:>+8.1%} "
                  f"{row['disc_mean_r']:>+8.1%} {row['hold_mean_r']:>+8.1%}")

    fri = trades[trades["entry_day"] == "Fri"]
    if not fri.empty:
        per_rule = fri.groupby("exit_rule", observed=True)["pnl_yuan"].sum()
        identical = per_rule.nunique() == 1
        print(f"\n  Fri leg: 0DTE open-and-settle, identical under all three exit rules "
              f"({'confirmed' if identical else 'DIFFERS -- investigate'}). NOT A TRADEABLE")
        print("  RESULT -- CFFEX settles on the last two hours' index average and trading")
        print("  ends 15:00, so a close entry buys an already-determined payoff. Its P&L is")
        print("  exactly the same-day close-vs-settlement basis:")
        for u in sorted(fri["underlying"].unique()):
            d = fri_leg_diagnostic(u, basis)
            print(f"    {UNDERLYINGS[u]:<16} n={d['n']}  same-day={d['all_same_day']}  "
                  f"basis mean={d['mean_basis_pts']:+.3f} sd={d['sd_basis_pts']:.2f} pts "
                  f"on a {d['median_debit_pts']:.1f}-pt debit  "
                  f"(worthless leg's stale tick: {d['median_worthless_leg_tick']:.2f} pts)")

    skips = trades.attrs.get("skips") or []
    if skips:
        why = pd.Series([s["why"] for s in skips]).value_counts().to_dict()
        print(f"  Skipped legs ({len(skips)}): {why}")

    OUT_DIR.mkdir(exist_ok=True)
    trades.to_csv(OUT_DIR / "expiry_week_atm_straddle_trades.csv",
                  index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "expiry_week_atm_straddle_summary.csv",
                   index=False, encoding="utf-8-sig")
    print(f"\n  Trades  -> {OUT_DIR / 'expiry_week_atm_straddle_trades.csv'}")
    print(f"  Summary -> {OUT_DIR / 'expiry_week_atm_straddle_summary.csv'}")
    return summary


def cost_sensitivity(underlyings: list[str], basis: str) -> pd.DataFrame:
    """mean return-on-debit per (underlying, entry day, exit rule) across the
    slippage grid. The headline run is the 0.00 column; everything to its
    right is how much of that survives a real spread."""
    raw = pd.concat([_raw_trades(u, basis) for u in underlyings], ignore_index=True)
    frames = []
    for slip in SLIPPAGE_GRID:
        tr = apply_costs(raw, slip)
        s = summarize(tr)[["underlying", "entry_day", "exit_rule", "n", "mean_r", "win_rate"]]
        s["slippage"] = slip
        frames.append(s)
    grid = pd.concat(frames, ignore_index=True)
    pivot = grid.pivot_table(index=["underlying", "entry_day", "exit_rule"],
                             columns="slippage", values="mean_r", observed=True)

    print(f"\n{'=' * 78}")
    print("Cost sensitivity -- mean return on debit vs. adverse half-spread per leg")
    print(f"(index points; CFFEX tick = {TICK}. Charged on entry and on close-based "
          f"exits, not on cash delivery.)")
    print(f"{'=' * 78}")
    header = "  ".join(f"{s:>8.2f}" for s in SLIPPAGE_GRID)
    print(f"  {'underlying':<16} {'entry':<6} {'exit rule':<10} {header}")
    for (u, label, rule), row in pivot.iterrows():
        cells = "  ".join(f"{row.get(s, np.nan):>+7.1%}" for s in SLIPPAGE_GRID)
        print(f"  {UNDERLYINGS[u]:<16} {label:<6} {rule:<10} {cells}")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "expiry_week_atm_straddle_cost_sensitivity.csv"
    grid.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  Cost grid -> {out}")
    return grid


# ------------------------------------------------------------------ equity / plot
def equity_curve(trades: pd.DataFrame, underlying: str, entry_day: str,
                 exit_rule: str) -> pd.DataFrame:
    """Sequential daily equity for one cell. Within a single (entry day,
    exit rule) there is at most one position open at a time, so a plain
    realized-at-exit cumulative curve is exact -- no overlap to net out."""
    g = trades[(trades["underlying"] == underlying) & (trades["entry_day"] == entry_day)
               & (trades["exit_rule"] == exit_rule)].sort_values("exit_date")
    if g.empty:
        return pd.DataFrame(columns=["date", "equity", "open_pnl", "spot"])
    chain = _prepped_chain(underlying)
    dates = sorted(pd.Timestamp(d) for d in chain[ds.C_DATE].unique())
    dates = [d for d in dates if d >= g["entry_date"].min()]
    realized = g.set_index("exit_date")["pnl"].groupby(level=0).sum()
    spot = (chain.dropna(subset=[ds.C_UND_CLOSE]).groupby(ds.C_DATE)[ds.C_UND_CLOSE].first())
    cum = realized.reindex(dates).fillna(0.0).cumsum()
    return pd.DataFrame({"date": dates, "equity": cum.values, "open_pnl": 0.0,
                         "spot": [spot.get(d, np.nan) for d in dates]})


def plot(trades: pd.DataFrame, underlying: str, exit_rule: str, out_path: Path) -> None:
    """Cumulative P&L by entry weekday for one underlying and exit rule."""
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=C_AXIS, linewidth=1.0)

    for label in ENTRY_ORDER:
        eq = equity_curve(trades, underlying, label, exit_rule)
        if eq.empty:
            continue
        g = trades[(trades["underlying"] == underlying) & (trades["entry_day"] == label)
                   & (trades["exit_rule"] == exit_rule)]
        ax.plot(eq["date"], eq["equity"] * MULTIPLIER, color=ENTRY_COLORS[label],
                linewidth=1.6,
                label=f"{label}  n={len(g)}  mean r={g['return_on_debit'].mean():+.1%}")

    ax.set_ylabel("cumulative P&L, 1 straddle (yuan)", color=C_INK2, fontsize=10)
    ax.set_title(f"{PLOT_NAMES[underlying]}: expiry-week ATM long straddle, "
                 f"front month (exit={exit_rule})",
                 color=C_INK, fontsize=12, loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for t in leg.get_texts():
        t.set_color(C_INK2)
    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(underlyings: list[str], basis: str, slippage: float, do_costs: bool,
         do_validate: bool) -> None:
    if do_validate:
        print("Validation")
        for u in underlyings:
            v = validate_settlement(u)
            print(f"  {UNDERLYINGS[u]:<16} settle vs parity intrinsic at DTE=0: "
                  f"n={v['n']:,}  median|diff|={v.get('median_abs_diff', float('nan')):.3f}  "
                  f"max|diff|={v.get('max_abs_diff', float('nan')):.2f} pts   "
                  f"| within 2% of forward: n={v.get('atm_n', 0):,}  "
                  f"median={v.get('atm_median_abs_diff', float('nan')):.3f}  "
                  f"max={v.get('atm_max_abs_diff', float('nan')):.2f} pts")
        for u in underlyings:
            b = validate_basis_agreement(u)
            print(f"  {UNDERLYINGS[u]:<16} calendar vs tradingday on shared dates: "
                  f"n_shared={b.get('n_shared', 0)}  mismatches={b.get('n_mismatch', 0)}  "
                  f"(n_calendar={b.get('n_calendar', 0)}, n_tradingday={b.get('n_tradingday', 0)})")

    per_u = [run(u, basis=basis, slippage=slippage) for u in underlyings]
    skips = [s for t in per_u for s in t.attrs["skips"]]
    trades = pd.concat(per_u, ignore_index=True)
    trades.attrs["skips"] = skips
    report(trades, basis, slippage)

    for u in underlyings:
        for rule in EXIT_RULES:
            out_path = OUT_DIR / f"expiry_week_atm_straddle_{u}_{rule}.png"
            plot(trades, u, rule, out_path)
    print(f"  Charts  -> {OUT_DIR}/expiry_week_atm_straddle_<underlying>_<rule>.png")

    if do_costs:
        cost_sensitivity(underlyings, basis)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default=None, choices=list(UNDERLYINGS),
                    help="default: all three (IO/MO/HO)")
    ap.add_argument("--entry-basis", default="calendar", dest="basis",
                    choices=["calendar", "tradingday"],
                    help="calendar: Mon/Wed/Fri of the expiry week (literal). "
                         "tradingday: T-4/T-2/T-0 trading days before expiry (robustness).")
    ap.add_argument("--slippage", type=float, default=0.0,
                    help="adverse half-spread per leg, index points (default 0, lab convention)")
    ap.add_argument("--no-costs", action="store_true", dest="no_costs",
                    help="skip the slippage sensitivity table")
    ap.add_argument("--no-validate", action="store_true", dest="no_validate",
                    help="skip the settlement / entry-basis cross-checks")
    args = ap.parse_args()
    us = [args.underlying] if args.underlying else list(UNDERLYINGS)
    main(us, args.basis, args.slippage, not args.no_costs, not args.no_validate)
