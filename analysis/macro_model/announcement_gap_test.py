"""
Announcement-Gap Test: does a big policy surprise predict a big overnight move?
==============================================================================

This is the STRADDLE-trigger companion to lead_lag_hfi_qvix.py.

Design rationale
----------------
The butterfly is the normal-regime (calm) position. The straddle is the move
we flip into when we expect a VOL/PRICE BREAKOUT. A straddle is long vega and
long gamma, so it profits from a big jump in implied vol and/or a big move in
the underlying -- in EITHER direction. So the right question is not "does the
signed shock predict a signed drift" (that's the lead-lag/butterfly test), but:

    does the MAGNITUDE of a policy surprise predict the MAGNITUDE of the
    announcement-window move, in IV (QVIX) and in the underlying (50ETF)?

We therefore measure the ANNOUNCEMENT GAP (overnight, close->open), not the
close-to-close drift, because the gap isolates the announcement's own impact:

    IV gap    = QVIX_open(effective)  - QVIX_close(prior trading day)
    price gap = ETF50_open(effective) - ETF50_close(prior trading day)

and regress |gap| on |shock| with HAC standard errors.

Announcement timing (which "open" is the effective one)
-------------------------------------------------------
Per the desk rule:
  * the 7-day OMO (reverse repo, `ReRepo7days`) is announced SAME DAY, BEFORE
    the market -> its effect lands in THAT day's open.
  * every other core instrument (MLF, LPR, RRR, deposit/loan benchmarks) is
    announced AFTER the close -> its effect lands in the NEXT trading day's open.

Mixed days (OMO fires together with a post-market instrument) are classified
POST-MARKET: the later announcement is what actually moves the next open.
Rule: pre-market iff (ReRepo7days == 1 AND no other core instrument fired);
otherwise post-market.

Data note
---------
QVIX open: 2015-02-09 -> present (full coverage).
ETF50 open: 2020-01-02 -> present, so the PRICE-gap test is limited to 2020+.
The IV-gap test uses the longer QVIX window.

Run:  python announcement_gap_test.py
Needs: pandas, numpy, statsmodels, matplotlib, openpyxl, scipy.
"""

from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]            # ...\Trading
QMT = ROOT / "qmt_export"
DERIVED = ROOT / "Data" / "derived"
RAW = ROOT / "Data" / "raw"
PYDIR = ROOT / "python"
OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

QVIX_FILE = QMT / "qvix_50etf.csv"
ETF_FILE = QMT / "etf50_daily.csv"
FR007_CACHE = RAW / "cache" / "fr007_daily.csv"

# which shock magnitude to use as the regressor:
#   "any"    -> FR007 change on any policy-announcement day  (broader)
#   "change" -> FR007 change only on days a core rate changed (cleaner surprise)
SHOCK_KIND = "change"

# the 7-day OMO instrument column in hfi_core (pre-market announcement)
PREMARKET_COL = "ReRepo7days"


# =============================================================================
# 1. BUILD TIMING-TAGGED ANNOUNCEMENT TABLE (reuses prep_hfi logic)
# =============================================================================
def build_announcements() -> pd.DataFrame:
    """
    Return one row per announcement day with:
        date, timing ('pre'|'post'), shock_any, shock_change
    by reusing prep_hfi's parsing/instrument logic against cached inputs.
    """
    # Stub akshare + config so importing prep_hfi needs no network and writes
    # nowhere unexpected.
    fake = types.ModuleType("fetch_akshare")
    fake.fetch_fr007_daily = lambda *a, **k: pd.read_csv(FR007_CACHE)
    sys.modules["fetch_akshare"] = fake
    cfg = types.ModuleType("config")
    cfg.RAW_DIR = RAW
    cfg.DERIVED_DIR = DERIVED
    sys.modules["config"] = cfg

    sys.path.insert(0, str(PYDIR))
    import prep_hfi  # noqa: E402

    cmpi = prep_hfi.parse_cmpi_sheets()
    hfi_df, _ = prep_hfi.build_hfi_matrix(cmpi)
    core = prep_hfi.select_core_instruments(hfi_df, cmpi)          # daily flags
    daily = prep_hfi.build_daily_shock(core)                       # + FR007 shock

    df = core.merge(
        daily[["date", "shock_policy", "shock_policy_change"]], on="date", how="left"
    )
    df = df[df["any_policy"] == 1].copy()

    others = [c for c in ["MLF1year", "LPR1year", "RRRLarge", "TD1year", "ML1-3years"]
              if c in df.columns]
    df["other_fired"] = df[others].any(axis=1).astype(int)
    df["timing"] = np.where(
        (df[PREMARKET_COL] == 1) & (df["other_fired"] == 0), "pre", "post"
    )
    out = df[["date", "timing", "shock_policy", "shock_policy_change"]].copy()
    out = out.rename(columns={"shock_policy": "shock_any",
                              "shock_policy_change": "shock_change"})
    return out.sort_values("date").reset_index(drop=True)


# =============================================================================
# 2. LOAD MARKET OHLC
# =============================================================================
def load_qvix() -> pd.DataFrame:
    q = pd.read_csv(QVIX_FILE, parse_dates=["date"])
    return q[["date", "open", "close"]].sort_values("date").reset_index(drop=True)


def load_etf() -> pd.DataFrame:
    e = pd.read_csv(ETF_FILE)
    e["date"] = pd.to_datetime(e["date"].astype(str), format="%Y%m%d")
    return e[["date", "open", "close"]].sort_values("date").reset_index(drop=True)


# =============================================================================
# 3. GAP CONSTRUCTION
# =============================================================================
def attach_gap(ann: pd.DataFrame, mkt: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    For each announcement day, compute the overnight gap on series `mkt`
    (columns date, open, close), respecting pre/post timing:

      pre  : effective open = announcement day's open;
             prior close    = previous trading day's close.
      post : effective open = next trading day's open;
             prior close    = announcement day's close.

    gap_pct = open(effective)/close(prior) - 1   (in %)
    Rows where the required trading days fall outside the market window are NaN.
    """
    m = mkt.sort_values("date").reset_index(drop=True)
    tdays = m["date"].to_numpy()
    open_by = dict(zip(m["date"], m["open"]))
    close_by = dict(zip(m["date"], m["close"]))

    def prev_tday(d):
        pos = np.searchsorted(tdays, np.datetime64(d), side="left")
        return pd.Timestamp(tdays[pos - 1]) if pos >= 1 else None

    def this_or_next_tday(d):
        pos = np.searchsorted(tdays, np.datetime64(d), side="left")
        return pd.Timestamp(tdays[pos]) if pos < len(tdays) else None

    gaps = []
    for _, r in ann.iterrows():
        d = r["date"]
        if d < m["date"].iloc[0] or d > m["date"].iloc[-1]:
            gaps.append(np.nan); continue
        if r["timing"] == "pre":
            eff_open_day = this_or_next_tday(d)          # the announcement day itself (or next open)
            prior_close_day = prev_tday(eff_open_day) if eff_open_day is not None else None
        else:  # post-market: effect lands next session
            ann_day = this_or_next_tday(d)               # the (close) anchor day
            prior_close_day = ann_day
            # next trading day strictly after ann_day
            if ann_day is not None:
                pos = np.searchsorted(tdays, np.datetime64(ann_day), side="right")
                eff_open_day = pd.Timestamp(tdays[pos]) if pos < len(tdays) else None
            else:
                eff_open_day = None
        if eff_open_day is None or prior_close_day is None:
            gaps.append(np.nan); continue
        o = open_by.get(eff_open_day); c = close_by.get(prior_close_day)
        gaps.append((o / c - 1.0) * 100.0 if (o and c) else np.nan)

    out = ann.copy()
    out[f"{name}_gap_pct"] = gaps
    return out


# =============================================================================
# 4. MAGNITUDE REGRESSION  |gap| ~ |shock|
# =============================================================================
def nw_lags(n: int) -> int:
    return max(int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))), 1)


def mag_regression(df: pd.DataFrame, gap_col: str, shock_col: str, label: str) -> dict:
    d = df.dropna(subset=[gap_col, shock_col]).copy()
    d = d[d[shock_col] != 0.0]                         # only real surprises
    y = d[gap_col].abs()
    x = d[shock_col].abs()
    if len(d) < 10:
        return dict(test=label, n=len(d), beta=np.nan, se=np.nan, t=np.nan,
                    p=np.nan, r2=np.nan)
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags(len(d))})
    bname = shock_col
    return dict(test=label, n=int(res.nobs),
                beta=res.params[bname], se=res.bse[bname],
                t=res.tvalues[bname], p=res.pvalues[bname], r2=res.rsquared)


def plot_scatter(df, gap_col, shock_col, label, fname):
    d = df.dropna(subset=[gap_col, shock_col]).copy()
    d = d[d[shock_col] != 0.0]
    if len(d) < 5:
        return
    x, y = d[shock_col].abs(), d[gap_col].abs()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x, y, s=24, alpha=0.7, color="#1f4e79")
    if len(d) >= 3:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(0, x.max(), 50)
        ax.plot(xs, a + b * xs, color="#c0392b", lw=1.5, label=f"slope={b:.2f}")
        ax.legend()
    ax.set_xlabel("|policy shock|  (|FR007 change|)")
    ax.set_ylabel(f"|{label} gap| (%)")
    ax.set_title(f"Surprise magnitude vs announcement-gap magnitude â€?{label}")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Announcement-gap (straddle-trigger) test")
    print("=" * 70)

    ann = build_announcements()
    shock_col = "shock_any" if SHOCK_KIND == "any" else "shock_change"
    print(f"announcement days: {len(ann)}  "
          f"(pre-market {int((ann.timing=='pre').sum())}, "
          f"post-market {int((ann.timing=='post').sum())})  | shock = {shock_col}")

    qvix = load_qvix()
    etf = load_etf()

    ann = attach_gap(ann, qvix, "iv")
    ann = attach_gap(ann, etf, "price")

    ann.to_csv(OUTDIR / "announcement_gaps.csv", index=False)

    # IV gap available on QVIX window; price gap on ETF (2020+)
    rows = [
        mag_regression(ann, "iv_gap_pct", shock_col, "IV gap (QVIX)"),
        mag_regression(ann, "price_gap_pct", shock_col, "Price gap (50ETF)"),
    ]
    table = pd.DataFrame(rows)
    print("\n--- |gap| ~ |shock|  (HAC SE) ---")
    print(table.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    plot_scatter(ann, "iv_gap_pct", shock_col, "IV", OUTDIR / "gap_iv_vs_shock.png")
    plot_scatter(ann, "price_gap_pct", shock_col, "price", OUTDIR / "gap_price_vs_shock.png")

    # quick descriptive: average |gap| on real-surprise days vs all
    real = ann[(ann[shock_col].abs() > 0)]
    print("\n--- descriptive (real-surprise announcement days) ---")
    for col, nm in [("iv_gap_pct", "QVIX IV gap"), ("price_gap_pct", "50ETF price gap")]:
        s = real[col].dropna()
        if len(s):
            print(f"  {nm:16s}: n={len(s):3d}  mean|gap|={s.abs().mean():.2f}%  "
                  f"max|gap|={s.abs().max():.2f}%")

    print(f"\nSaved: {OUTDIR/'announcement_gaps.csv'} and gap_*_vs_shock.png")


if __name__ == "__main__":
    main()
