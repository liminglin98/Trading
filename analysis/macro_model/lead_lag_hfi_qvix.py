"""
Lead-Lag Test: HFI Shock -> QVIX (Delta-IV)
============================================

Decision-gate regression for the rolling-butterfly strategy:

    Delta_IV(t -> t+k) = alpha + beta * HFI_shock(t) + controls + eps

for k in {1, 2, 3, 5, 10} trading days, with Newey-West (HAC) standard errors.

This is a SIGNAL-VALIDATION test only. The HFI shock contains a narrative /
look-ahead component and CANNOT be used as-is for backtesting. Here we only ask:
does the shock carry information that moves implied volatility?

------------------------------------------------------------------------------
IMPORTANT DATA NOTE (read before trusting any output)
------------------------------------------------------------------------------
As of writing, the folder does NOT contain a daily intraday FR007-surprise
series. The closest available object is `shock_policy` in
Data/derived/hfi_core_data.csv, which is a MONTHLY policy-rule residual
(see fr007_policy_rule_coefficients.csv). A monthly shock cannot be honestly
mapped to a specific trading day / 9:20am announcement window.

Therefore this script has two modes:

    HFI_MODE = "proxy_monthly"   -> runs NOW on the monthly shock as an
                                    explicitly-labeled SMOKE TEST / pipeline
                                    validation. Treat coefficients as
                                    plumbing checks, NOT economic results.

    HFI_MODE = "external"        -> plug in your real daily HFI file. Set
                                    HFI_FILE and confirm it matches
                                    EXPECTED_EXTERNAL_SCHEMA below. Everything
                                    downstream (regressions, plots, robustness)
                                    is identical and is the real analysis.

Switch modes by editing the CONFIG block. Nothing else needs to change.
------------------------------------------------------------------------------

Author: generated for Liming's master's-thesis butterfly project.
Python 3.13; needs pandas, numpy, scipy, statsmodels, matplotlib.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # write PNGs without a display
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIG
# =============================================================================
# Folder layout assumed:
#   <ROOT>/qmt_export/qvix_*.csv
#   <ROOT>/Data/derived/hfi_core_data.csv
#   <ROOT>/analysis/<this script>   <- output goes to ./out
ROOT = Path(__file__).resolve().parents[2]          # ...\Trading
QMT = ROOT / "qmt_export"
DERIVED = ROOT / "Data" / "derived"
OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

# ---- which shock to use ----------------------------------------------------
HFI_MODE = "external"           # "proxy_monthly" | "external"

# external daily HFI file (only used when HFI_MODE == "external")
HFI_FILE = DERIVED / "hfi_daily_shock.csv"
# Expected schema for the external file: one row per PBoC announcement day.
#   date         : YYYY-MM-DD, the announcement trading day (close used for t)
#   hfi_shock    : signed FR007 surprise (bp or std units; document which)
EXPECTED_EXTERNAL_SCHEMA = ["date", "hfi_shock"]

# proxy column from hfi_core_data.csv (monthly). Either:
#   "shock_policy"        -> level of policy-rule residual
#   "shock_policy_change" -> first difference (more "surprise"-like)
PROXY_COL = "shock_policy"

# ---- regression design -----------------------------------------------------
HORIZONS = [1, 2, 3, 5, 10]
QVIX_TARGETS = {
    "50ETF": QMT / "qvix_50etf.csv",
    "300ETF": QMT / "qvix_300etf.csv",
}
PRIMARY_TARGET = "50ETF"           # run first / used for diagnostic plots

# controls toggles
USE_AR_CONTROL = True              # lagged 1-day Delta-IV at t
USE_LEVEL_CONTROL = True           # lagged log QVIX level at t
GLOBAL_RISK_FILE = None            # e.g. QMT/"vix.csv" with [date, close]; else None

STRUCTURAL_BREAK = pd.Timestamp("2022-01-01")   # pre/post subsample split

# =============================================================================
# 1. LOADERS
# =============================================================================
def load_qvix(path: Path) -> pd.DataFrame:
    """Return daily QVIX with a clean trading-day index and log close."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    df = df[["close"]].rename(columns={"close": "qvix"})
    df["log_qvix"] = np.log(df["qvix"])
    return df


def load_global_risk(path: Path | None) -> pd.Series | None:
    if path is None:
        return None
    g = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
    return np.log(g["close"]).rename("log_global")


def load_hfi_external(path: Path) -> pd.DataFrame:
    """Daily HFI shock: one row per announcement day."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in EXPECTED_EXTERNAL_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(
            f"External HFI file missing columns {missing}. "
            f"Expected {EXPECTED_EXTERNAL_SCHEMA}."
        )
    df = df[["date", "hfi_shock"]].dropna(subset=["hfi_shock"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    df["announcement"] = True
    return df


def load_hfi_proxy_monthly(qvix_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    PLACEHOLDER shock. The monthly policy-rule residual is mapped to ONE trading
    day per month so the daily pipeline can run. We use the first QVIX trading
    day on/after the month-start timestamp as the pseudo 'announcement day'.

    This is a smoke test. The mapping has no claim to being the true OMO date,
    and a monthly shock spread over k<=10 trading days is econometrically weak.
    """
    core = pd.read_csv(DERIVED / "hfi_core_data.csv", parse_dates=["date"])
    if PROXY_COL not in core.columns:
        raise ValueError(f"{PROXY_COL!r} not in hfi_core_data.csv")
    m = core[["date", PROXY_COL]].dropna().rename(columns={PROXY_COL: "hfi_shock"})
    m = m[m["hfi_shock"] != 0.0]

    qidx = pd.DatetimeIndex(sorted(qvix_index))
    rows = []
    for _, r in m.iterrows():
        # skip months outside QVIX coverage (else they collapse onto day 0)
        if r["date"] < qidx[0] or r["date"] > qidx[-1]:
            continue
        # first trading day on/after month-start
        pos = qidx.searchsorted(r["date"], side="left")
        if pos >= len(qidx):
            continue
        rows.append((qidx[pos], r["hfi_shock"]))
    out = pd.DataFrame(rows, columns=["date", "hfi_shock"]).drop_duplicates("date")
    out = out.set_index("date")
    out["announcement"] = True
    return out


# =============================================================================
# 2. DESIGN MATRIX
# =============================================================================
def forward_dlog(qvix: pd.DataFrame, k: int) -> pd.Series:
    """
    Delta_IV(t -> t+k) on the QVIX TRADING-DAY grid:
        log(qvix_{t+k}) - log(qvix_t)
    Indexed by t. Uses trading-day shift, so calendar gaps are ignored.
    """
    fwd = qvix["log_qvix"].shift(-k) - qvix["log_qvix"]
    return fwd.rename(f"dIV_{k}")


def lag1_dlog(qvix: pd.DataFrame) -> pd.Series:
    return (qvix["log_qvix"] - qvix["log_qvix"].shift(1)).rename("dIV_lag1")


def nw_lags(n: int, k: int) -> int:
    """
    'Automatic' Newey-West bandwidth: rule-of-thumb L = floor(4*(n/100)^(2/9)),
    bumped to at least (k-1) because overlapping k-day forward returns induce an
    MA(k-1) error structure that MUST be covered by the HAC window.
    """
    rot = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    return max(rot, k - 1, 1)


def build_design(
    qvix: pd.DataFrame,
    hfi: pd.DataFrame,
    k: int,
    sample: str = "announcement",
    global_risk: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Assemble regression frame for horizon k.

    sample = "announcement" : only announcement days (shock present).
    sample = "zero_fill"    : every trading day; shock=0 off announcement days.
    """
    dv = forward_dlog(qvix, k)
    base = pd.DataFrame(index=qvix.index)
    base["dIV_k"] = dv
    base["log_qvix_lag"] = qvix["log_qvix"]          # level at t (the "lag" wrt t+k)
    base["dIV_lag1"] = lag1_dlog(qvix)
    if global_risk is not None:
        base["log_global"] = global_risk.reindex(qvix.index).ffill()

    if sample == "zero_fill":
        base["hfi_shock"] = 0.0
        base.loc[base.index.isin(hfi.index), "hfi_shock"] = hfi["hfi_shock"]
        base["announcement"] = base.index.isin(hfi.index)
    elif sample == "announcement":
        base = base.loc[base.index.isin(hfi.index)].copy()
        base["hfi_shock"] = hfi["hfi_shock"].reindex(base.index)
        base["announcement"] = True
    else:
        raise ValueError(sample)

    # Drop rows with NaN/inf in the dependent var, the shock, OR any control
    # that will enter the design (e.g. dIV_lag1 is NaN on the first trading day,
    # log_qvix_lag where QVIX<=0). Otherwise statsmodels' HAC fit errors out.
    ctrl_cols = [c for c in ("dIV_lag1", "log_qvix_lag", "log_global") if c in base]
    base = base.replace([np.inf, -np.inf], np.nan)
    return base.dropna(subset=["dIV_k", "hfi_shock"] + ctrl_cols)


def regressors(frame: pd.DataFrame, extra: list[str] | None = None) -> list[str]:
    cols = ["hfi_shock"]
    if USE_AR_CONTROL and "dIV_lag1" in frame:
        cols.append("dIV_lag1")
    if USE_LEVEL_CONTROL and "log_qvix_lag" in frame:
        cols.append("log_qvix_lag")
    if "log_global" in frame:
        cols.append("log_global")
    if extra:
        cols += [c for c in extra if c in frame]
    return cols


def run_ols(frame: pd.DataFrame, k: int, xcols: list[str]) -> sm.regression.linear_model.RegressionResultsWrapper:
    X = sm.add_constant(frame[xcols], has_constant="add")
    y = frame["dIV_k"]
    L = nw_lags(len(frame), k)
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})


# =============================================================================
# 3. CORE: horizon sweep
# =============================================================================
def horizon_sweep(
    qvix: pd.DataFrame,
    hfi: pd.DataFrame,
    sample: str,
    label: str,
    global_risk: pd.Series | None = None,
) -> pd.DataFrame:
    out = []
    for k in HORIZONS:
        frame = build_design(qvix, hfi, k, sample=sample, global_risk=global_risk)
        if len(frame) < 10:
            out.append(dict(target=label, k=k, n=len(frame), beta=np.nan,
                            se=np.nan, t=np.nan, p=np.nan, r2=np.nan, nw_lags=np.nan))
            continue
        xcols = regressors(frame)
        res = run_ols(frame, k, xcols)
        out.append(dict(
            target=label, k=k, n=int(res.nobs),
            beta=res.params["hfi_shock"],
            se=res.bse["hfi_shock"],
            t=res.tvalues["hfi_shock"],
            p=res.pvalues["hfi_shock"],
            r2=res.rsquared,
            nw_lags=nw_lags(len(frame), k),
        ))
    return pd.DataFrame(out)


# =============================================================================
# 4. PLOTS
# =============================================================================
def plot_irf(results: pd.DataFrame, label: str, fname: Path) -> None:
    r = results.dropna(subset=["beta"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(r["k"], r["beta"], "o-", color="#1f4e79", label="beta(k)")
    ax.fill_between(r["k"], r["beta"] - 1.96 * r["se"], r["beta"] + 1.96 * r["se"],
                    alpha=0.2, color="#1f4e79", label="95% CI (NW)")
    ax.set_xlabel("horizon k (trading days)")
    ax.set_ylabel("beta: d(log QVIX) per unit shock")
    ax.set_title(f"Impulse response of Delta-IV to HFI shock â€?{label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_scatter(qvix, hfi, k_star: int, label: str, fname: Path) -> None:
    frame = build_design(qvix, hfi, k_star, sample="announcement")
    x, y = frame["hfi_shock"], frame["dIV_k"]
    b, a = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(x, y, s=22, alpha=0.7, color="#1f4e79")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color="#c0392b", lw=1.5, label=f"fit slope={b:.3f}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("HFI shock")
    ax.set_ylabel(f"Delta-IV over k={k_star}")
    ax.set_title(f"Shock vs Delta-IV at k={k_star} â€?{label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def plot_residuals(qvix, hfi, k_star: int, label: str, fname: Path) -> None:
    frame = build_design(qvix, hfi, k_star, sample="announcement")
    res = run_ols(frame, k_star, regressors(frame))
    resid, fitted = res.resid, res.fittedvalues
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(fitted, resid, s=18, alpha=0.7, color="#1f4e79")
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[0].set_title("Residuals vs fitted")
    axes[0].set_xlabel("fitted"); axes[0].set_ylabel("residual")

    stats.probplot(resid, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q")

    nlags = min(20, len(resid) // 2)
    acf = [resid.autocorr(lag=i) for i in range(1, nlags + 1)]
    axes[2].bar(range(1, nlags + 1), acf, color="#1f4e79")
    ci = 1.96 / np.sqrt(len(resid))
    axes[2].axhline(ci, color="#c0392b", ls="--", lw=0.8)
    axes[2].axhline(-ci, color="#c0392b", ls="--", lw=0.8)
    axes[2].axhline(0, color="grey", lw=0.8)
    axes[2].set_title("Residual ACF")
    axes[2].set_xlabel("lag")

    fig.suptitle(f"Residual diagnostics k={k_star} â€?{label}")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    plt.close(fig)


# =============================================================================
# 5. ROBUSTNESS
# =============================================================================
def robustness_zero_fill(qvix, hfi, label: str) -> pd.DataFrame:
    ann = horizon_sweep(qvix, hfi, "announcement", f"{label}|ann_only")
    zf = horizon_sweep(qvix, hfi, "zero_fill", f"{label}|zero_fill")
    return pd.concat([ann, zf], ignore_index=True)


def robustness_asymmetry(qvix, hfi, label: str) -> pd.DataFrame:
    """Split shock into positive and negative parts; report both betas."""
    out = []
    for k in HORIZONS:
        frame = build_design(qvix, hfi, k, sample="announcement")
        frame["shock_pos"] = frame["hfi_shock"].clip(lower=0)
        frame["shock_neg"] = frame["hfi_shock"].clip(upper=0)
        xcols = [c for c in regressors(frame) if c != "hfi_shock"]
        xcols = ["shock_pos", "shock_neg"] + xcols
        if len(frame) < 12:
            continue
        res = run_ols(frame, k, xcols)
        out.append(dict(
            target=label, k=k, n=int(res.nobs),
            beta_pos=res.params["shock_pos"], p_pos=res.pvalues["shock_pos"],
            beta_neg=res.params["shock_neg"], p_neg=res.pvalues["shock_neg"],
        ))
    return pd.DataFrame(out)


def robustness_subsample(qvix, hfi, label: str) -> pd.DataFrame:
    pre_q = qvix[qvix.index < STRUCTURAL_BREAK]
    post_q = qvix[qvix.index >= STRUCTURAL_BREAK]
    pre = horizon_sweep(pre_q, hfi, "announcement", f"{label}|pre2022")
    post = horizon_sweep(post_q, hfi, "announcement", f"{label}|post2022")
    return pd.concat([pre, post], ignore_index=True)


# =============================================================================
# 6. STRATEGY MAPPING
# =============================================================================
def strategy_mapping(results: pd.DataFrame, hfi: pd.DataFrame, label: str) -> str:
    """
    Translate the best-horizon beta into a butterfly-signal magnitude.

    A long-butterfly profits when realized vol stays inside the wings (vol falls
    or is stable); a short-butterfly profits when vol breaks out. So the sign and
    size of expected Delta-IV after a shock maps to which side to lean.
    """
    r = results.dropna(subset=["p"])
    if r.empty:
        return "No estimable horizon."
    best = r.loc[r["p"].idxmin()]
    sd = float(np.nanstd(hfi["hfi_shock"]))
    move_1sd = best["beta"] * sd                      # d(log IV) for a 1-sd shock
    pct = (np.exp(move_1sd) - 1) * 100
    lines = [
        f"[{label}] most significant horizon: k={int(best['k'])} "
        f"(beta={best['beta']:.3f}, p={best['p']:.3g}, n={int(best['n'])}).",
        f"1-sd shock = {sd:.4f} units -> expected Delta_IV(k) ~ {move_1sd:+.4f} "
        f"log points ({pct:+.2f}% in QVIX).",
        "Mapping to the butterfly regime switch:",
        f"  - sign(beta) tells you the conditional IV drift after a shock; "
        f"a {'rise' if move_1sd > 0 else 'fall'} favors a "
        f"{'short (breakout)' if move_1sd > 0 else 'long (mean-revert)'} butterfly tilt.",
        "  - Set the regime trigger on |fitted Delta_IV| exceeding the residual "
        "sigma of the regression, not on the raw shock, so the threshold is in "
        "IV units the strategy actually trades.",
        "  - Only act when the coefficient is stable across the pre/post-2022 "
        "subsamples; otherwise the trigger is fitting a regime that has passed.",
    ]
    return "\n".join(lines)


# =============================================================================
# 7. MAIN
# =============================================================================
def main() -> None:
    print("=" * 70)
    print(f"HFI -> QVIX lead-lag test   |   HFI_MODE = {HFI_MODE}")
    if HFI_MODE == "proxy_monthly":
        print("!! PROXY MODE: monthly shock as smoke test. NOT economic results.")
    print("=" * 70)

    global_risk = load_global_risk(GLOBAL_RISK_FILE)
    all_results = []

    # load primary first to build the trading-day grid for proxy mapping
    primary_qvix = load_qvix(QVIX_TARGETS[PRIMARY_TARGET])

    if HFI_MODE == "external":
        hfi = load_hfi_external(HFI_FILE)
    else:
        hfi = load_hfi_proxy_monthly(primary_qvix.index)
    print(f"HFI shock rows: {len(hfi)}  "
          f"({hfi.index.min().date()} -> {hfi.index.max().date()})")

    for name, path in QVIX_TARGETS.items():
        qvix = primary_qvix if name == PRIMARY_TARGET else load_qvix(path)
        print(f"\n----- {name} ({len(qvix)} trading days) -----")

        res = horizon_sweep(qvix, hfi, "announcement", name, global_risk)
        all_results.append(res)
        print(res.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

        if name == PRIMARY_TARGET:
            r = res.dropna(subset=["p"])
            k_star = int(r.loc[r["p"].idxmin(), "k"]) if not r.empty else HORIZONS[0]
            plot_irf(res, name, OUTDIR / f"irf_{name}.png")
            plot_scatter(qvix, hfi, k_star, name, OUTDIR / f"scatter_{name}_k{k_star}.png")
            plot_residuals(qvix, hfi, k_star, name, OUTDIR / f"residuals_{name}_k{k_star}.png")

            print("\n--- robustness: announcement-only vs zero-fill ---")
            print(robustness_zero_fill(qvix, hfi, name).to_string(index=False))
            print("\n--- robustness: asymmetry (pos vs neg shock) ---")
            print(robustness_asymmetry(qvix, hfi, name).to_string(index=False))
            print("\n--- robustness: pre/post 2022 ---")
            print(robustness_subsample(qvix, hfi, name).to_string(index=False))
            print("\n--- strategy mapping ---")
            print(strategy_mapping(res, hfi, name))

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTDIR / "lead_lag_results.csv", index=False)
    print(f"\nSaved tables + plots to: {OUTDIR}")
    if HFI_MODE == "proxy_monthly":
        print("\nREMINDER: switch HFI_MODE to 'external' once the daily "
              "FR007-surprise file exists. See EXPECTED_EXTERNAL_SCHEMA.")


if __name__ == "__main__":
    main()
