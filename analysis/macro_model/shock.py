# -*- coding: utf-8 -*-
"""
shock.py -- global risk-off ("outside shock") detector for the 50ETF book
=========================================================================
Both model strategies (long butterfly, iron condor) are SHORT realized vol and
price-neutral, so an imported global risk-off (e.g. the Jun-2026 AI/chip rout:
KOSPI circuit-breaker, Nasdaq -2%+, Micron/Nvidia tumbling) is exactly the
regime that hurts them. This module turns daily GLOBAL equity moves into a daily
risk-off flag for the China session, with NO lookahead.

Signal sources (direct / sina -- no proxy needed; see project data notes):
    S&P 500   -> ak.index_us_stock_sina(".INX")
    Nasdaq    -> ak.index_us_stock_sina(".IXIC")   <- core tech-risk proxy
(KOSPI was dropped: its only akshare feed needs the eastmoney proxy. Nasdaq
 carries the same global AI/chip-risk factor, so it stands in fine.)

Two notions of "risk-off":
    acute  : a single overnight crash (core 1-day log return <= ACUTE_1D).
             -> used as an intramonth EARLY-EXIT trigger for the short-vol book.
    regime : a sustained risk-off state (rolling drawdown <= DD_THR over DD_WIN,
             OR a <= Z_THR sigma daily move vs trailing vol).
             -> used as the month-ENTRY filter (de-risk and/or hedge).

Overnight alignment (no lookahead): the foreign close the China session at day t
can react to is, at the latest, the foreign close of the PRIOR calendar day. We
therefore read every global signal `as-of (t - LAG_DAYS)`.

CLI:
    python shock.py                 # fetch, build flags, print recent rows + a
                                    # count of historical risk-off days, save CSV
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import butterfly_timing_backtest as B        # gives us the akshare handle
ak = B.ak

OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)
CACHE = OUTDIR / "global_daily_cache.csv"

# --- default detector parameters (calibrate after the first backtest run) -----
ACUTE_1D = -0.025      # core 1-day log return <= this  -> acute overnight shock
DD_WIN   = 5           # rolling window (sessions) for the drawdown regime test
DD_THR   = -0.06       # core drawdown over DD_WIN <= this -> regime on
Z_WIN    = 60          # trailing window for the return z-score
Z_THR    = -2.0        # core 1-day z-score <= this -> regime on
LAG_DAYS = 1           # read foreign signal as-of (cn_date - LAG_DAYS): no lookahead

US_SYMBOLS = {"spx": ".INX", "ndx": ".IXIC"}

# --- DOMESTIC detector parameters (QVIX + 50ETF's own move) -------------------
# The global overlay backtest showed the book's tail is DOMESTIC, not imported:
# the worst short-vol months were China-specific. These flags watch the things
# that actually hurt a short-vol book -- the underlying gapping and implied vol
# spiking. CN signals are same-session: the decision is made at the close using
# that close's data and acted at the close, so DOMESTIC_LAG = 0 (no lookahead).
ETF_ACUTE_1D  = -0.030   # 50ETF 1-day log return <= this -> acute price shock
QVIX_JUMP     =  0.150   # QVIX 1-day log change >= this   -> acute vol spike
ETF_DD_WIN    =  5       # rolling window (sessions) for the 50ETF drawdown regime
ETF_DD_THR    = -0.060   # 50ETF drawdown over ETF_DD_WIN <= this -> regime on
ETF_Z_WIN     =  60      # trailing window for the 50ETF return z-score
ETF_Z_THR     = -2.0     # 50ETF 1-day z-score <= this -> regime on
QVIX_PCTL_WIN =  252     # trailing window (~1y) for the QVIX level percentile
QVIX_PCTL_THR =  0.80    # QVIX in the top (1-this) of the past year -> regime on
DOMESTIC_LAG  =  0       # CN signals are same-session (decided + acted at close)


# =============================================================================
# Fetch
# =============================================================================
def _sina_us(symbol: str) -> pd.Series:
    d = ak.index_us_stock_sina(symbol=symbol)
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].astype(float).sort_index()


def fetch_global_daily(use_cache: bool = True) -> pd.DataFrame:
    """Daily close levels for the global risk proxies (index = date)."""
    frames = {}
    for name, sym in US_SYMBOLS.items():
        try:
            frames[name] = _sina_us(sym)
            print(f"[shock] {name} ({sym}): {len(frames[name])} rows")
        except Exception as e:
            print(f"[shock] failed {name} ({sym}): {e}")
    if not frames:
        if use_cache and CACHE.exists():
            print("[shock] live fetch failed; using cached global_daily_cache.csv")
            return pd.read_csv(CACHE, parse_dates=["date"]).set_index("date")
        raise RuntimeError("no global series fetched and no cache available")

    out = pd.concat(frames.values(), axis=1)
    out.columns = list(frames.keys())
    out = out.sort_index()
    out.to_csv(CACHE, index_label="date")
    return out


# =============================================================================
# Risk-off flags
# =============================================================================
def risk_off_table(glob: pd.DataFrame | None = None,
                   acute_1d: float = ACUTE_1D, dd_win: int = DD_WIN,
                   dd_thr: float = DD_THR, z_win: int = Z_WIN,
                   z_thr: float = Z_THR) -> pd.DataFrame:
    """Daily features + boolean flags. `core` = Nasdaq if present else S&P."""
    if glob is None:
        glob = fetch_global_daily()

    core = "ndx" if "ndx" in glob.columns else ("spx" if "spx" in glob.columns
                                                else glob.columns[0])
    r = np.log(glob[core]).diff()

    f = pd.DataFrame(index=glob.index)
    f["core"] = core
    f["core_ret"] = r
    f["core_dd"] = glob[core] / glob[core].rolling(dd_win).max() - 1.0
    mu = r.rolling(z_win).mean()
    sd = r.rolling(z_win).std()
    f["core_z"] = (r - mu) / sd
    # per-source 1-day returns (handy for inspection / multi-market confirmation)
    for c in glob.columns:
        f[f"{c}_ret"] = np.log(glob[c]).diff()

    f["acute"] = f["core_ret"] <= acute_1d
    f["regime"] = (f["core_dd"] <= dd_thr) | (f["core_z"] <= z_thr)
    f["risk_off"] = f["acute"] | f["regime"]
    return f


# =============================================================================
# DOMESTIC risk-off (QVIX implied-vol spike + 50ETF's own price move)
# =============================================================================
def fetch_etf(symbol: str = "sh510050") -> pd.Series:
    e = ak.fund_etf_hist_sina(symbol=symbol).copy()
    e["date"] = pd.to_datetime(e["date"])
    return e.set_index("date")["close"].astype(float).sort_index()


def fetch_qvix() -> pd.Series:
    """50ETF QVIX (implied-vol index) daily close -- same feed as the panel."""
    d = ak.index_option_50etf_qvix().copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date")["close"].astype(float).sort_index()


def domestic_risk_table(etf: pd.Series | None = None,
                        qvix: pd.Series | None = None,
                        etf_acute: float = ETF_ACUTE_1D, qvix_jump: float = QVIX_JUMP,
                        dd_win: int = ETF_DD_WIN, dd_thr: float = ETF_DD_THR,
                        z_win: int = ETF_Z_WIN, z_thr: float = ETF_Z_THR,
                        qvix_win: int = QVIX_PCTL_WIN,
                        qvix_pctl: float = QVIX_PCTL_THR) -> pd.DataFrame:
    """Daily domestic flags on the 50ETF trading-day grid.

    acute  : a same-day 50ETF crash (ret <= etf_acute) OR a QVIX spike
             (1-day log change >= qvix_jump). -> intramonth early-exit trigger.
    regime : a sustained stressed state -- 50ETF rolling drawdown <= dd_thr, OR a
             <= z_thr sigma day, OR QVIX already in the top (1-qvix_pctl) of the
             past year (elevated implied vol). -> entry de-risk / hedge gate.
    """
    if etf is None:
        etf = fetch_etf()
    if qvix is None:
        try:
            qvix = fetch_qvix()
        except Exception as e:
            print(f"[shock] QVIX fetch failed ({e}); domestic table = price-only.")
            qvix = None

    r = np.log(etf).diff()
    f = pd.DataFrame(index=etf.index)
    f["core"] = "50ETF"
    f["etf_ret"] = r
    f["etf_dd"] = etf / etf.rolling(dd_win).max() - 1.0
    f["etf_z"] = (r - r.rolling(z_win).mean()) / r.rolling(z_win).std()

    acute = r <= etf_acute
    regime = (f["etf_dd"] <= dd_thr) | (f["etf_z"] <= z_thr)

    if qvix is not None:
        q = qvix.reindex(etf.index).ffill()
        f["qvix"] = q
        f["qvix_chg"] = np.log(q).diff()
        # rank of today within the trailing window (1.0 = highest in ~1y)
        f["qvix_pctl"] = q.rolling(qvix_win).apply(
            lambda a: float((a[-1] >= a).mean()), raw=True)
        acute = acute | (f["qvix_chg"] >= qvix_jump)
        regime = regime | (f["qvix_pctl"] >= qvix_pctl)

    f["acute"] = acute.fillna(False)
    f["regime"] = regime.fillna(False)
    f["risk_off"] = f["acute"] | f["regime"]
    return f


# =============================================================================
# No-lookahead alignment helpers (use these from the backtest / live app)
# =============================================================================
def asof_flag(tbl: pd.DataFrame, col: str, cn_date, lag_days: int = LAG_DAYS):
    """Value of `col` as known to the China session at cn_date: the most recent
    foreign observation on or before (cn_date - lag_days). Returns the scalar, or
    NaN/False if nothing precedes that date."""
    cutoff = pd.Timestamp(pd.Timestamp(cn_date).date()) - pd.Timedelta(days=lag_days)
    s = tbl[col]
    idx = s.index[s.index <= cutoff]
    if len(idx) == 0:
        return False if s.dtype == bool else np.nan
    return s.loc[idx[-1]]


def is_risk_off(tbl: pd.DataFrame, cn_date, kind: str = "regime",
                lag_days: int = LAG_DAYS) -> bool:
    """kind: 'regime' (entry filter) | 'acute' (intramonth shock) | 'risk_off'."""
    return bool(asof_flag(tbl, kind, cn_date, lag_days))


# =============================================================================
# CLI
# =============================================================================
def main():
    glob = fetch_global_daily()
    tbl = risk_off_table(glob)
    out = OUTDIR / "global_risk_off.csv"
    tbl.to_csv(out, index_label="date")

    n = len(tbl)
    print("\n=== global risk-off summary ===")
    print(f"core proxy            : {tbl['core'].iloc[-1]}")
    print(f"sessions              : {n}  ({tbl.index[0].date()} .. {tbl.index[-1].date()})")
    print(f"acute days  (<= {ACUTE_1D:+.1%}) : {int(tbl['acute'].sum())} "
          f"({tbl['acute'].mean():.1%})")
    print(f"regime days           : {int(tbl['regime'].sum())} ({tbl['regime'].mean():.1%})")
    print(f"any risk-off          : {int(tbl['risk_off'].sum())} ({tbl['risk_off'].mean():.1%})")
    print(f"\nsaved -> {out}")
    cols = ["core_ret", "core_dd", "core_z", "acute", "regime", "risk_off"]
    print("\n--- last 10 sessions ---")
    print(tbl[cols].tail(10).to_string(
        float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
