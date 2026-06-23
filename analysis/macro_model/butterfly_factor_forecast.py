# -*- coding: utf-8 -*-
"""
Butterfly FACTOR forecast - 50ETF next-month return (Bayesian, walk-forward)
============================================================================
Successor to butterfly_macro_forecast.py. The macro-only model had no directional
skill (the surveyed real-economy series are the wrong input for monthly index
direction). This version keeps the SAME Bayesian walk-forward machinery but swaps
in the MARKET / FINANCIAL / FLOW factor panel built by build_model_panel.py.

Choices (per user): target = 50ETF, backend = conjugate Normal-Inverse-Gamma
(ridge-shrinkage Bayesian linear regression), predictors = full factor set on the
modern window (~2018+, where margin / QVIX / etc. are all populated).

What it produces each month:
  * point forecast (posterior mean) -> butterfly BODY (center) strike
  * posterior-predictive band (50% / 80%) -> WING width
  * P(up) and direction, plus a walk-forward OOS evaluation (hit rate, RMSE vs a
    predict-zero naive, and interval coverage -- the macro model's real strength).

Look-ahead discipline: every predictor in the panel is already month-end and
known at time t (market data, same-day); the target is the t->t+1 log return.
Walk-forward is expanding-window; standardization uses TRAIN moments only.

Run:  python butterfly_factor_forecast.py
Needs: pandas, numpy, matplotlib, scipy.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = 20260622

# =============================================================================
# CONFIG
# =============================================================================
ROOT     = Path(__file__).resolve().parents[2]
PANEL    = ROOT / "Data" / "akshare" / "panel_monthly.csv"
OUTDIR   = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

TARGET   = "etf50_ret_next"          # 50ETF next-month log return
SPOT_COL = "etf50_close"             # month-end 50ETF close (for strike mapping)

# Full factor set on the modern window. Deliberately EXCLUDES:
#   - level prices / margin_bal level  (non-stationary)
#   - nbnd_net_m  (northbound dies 2024-08 -> would truncate the sample)
#   - tsf_increment_lag1 (currently unfetchable: mofcom SSL)
# erp_50 is derived below from ey_50 and cn10y.
PREDICTORS = [
    # momentum / trend / vol (50ETF own price)
    "etf50_mom_3m", "etf50_mom_6m", "etf50_mom_12m",
    "etf50_px_ma50", "etf50_px_ma200",
    "etf50_rv20", "etf50_rv60",
    # valuation
    "ey_50", "erp_50",
    # liquidity / monetary
    "margin_mom", "shibor_1w", "shibor_chg",
    # sentiment
    "qvix_50", "qvix_50_pct",
    # cross-asset / global
    "usdcny_mom", "cn10y", "us10y", "us10y_chg",
    "sp500_ret", "gold_ret",
]

PRIOR_TAU = 0.25     # prior SD on standardized coefficients (shrinkage strength)
MIN_TRAIN = 48       # initial training window (months) before first OOS forecast
BANDS     = [0.50, 0.80]

# =============================================================================
# 1. PANEL
# =============================================================================
def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL, parse_dates=["month_end"]).set_index("month_end").sort_index()
    # derive SSE50 equity-risk-premium = earnings yield - China 10Y (decimal)
    if "ey_50" in df and "cn10y" in df:
        df["erp_50"] = df["ey_50"] - df["cn10y"] / 100.0
    return df


def build_xy(df: pd.DataFrame):
    """Complete-case panel of (target + predictors) on the modern window."""
    have = [c for c in PREDICTORS if c in df.columns]
    missing = [c for c in PREDICTORS if c not in df.columns]
    if missing:
        print("  [warn] predictors not in panel, skipped:", missing)
    cols = [TARGET] + have
    sub = df[cols].dropna()
    y = sub[TARGET]
    X = sub[have]
    return X, y, have


# =============================================================================
# 2. CONJUGATE BAYESIAN LINEAR REGRESSION (Normal-Inverse-Gamma)
# =============================================================================
def _standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_predict_conjugate(Xtr, ytr, Xte, n_draws=4000):
    """
    Gaussian (ridge) prior on standardized-predictor coefficients + Inverse-Gamma
    on noise variance. Posterior predictive is Student-t; we draw from it so the
    banding code is backend-agnostic. Ideal at small n.
    """
    Xtr_s, Xte_s = _standardize(Xtr, Xte)
    n, p = Xtr_s.shape
    A   = np.column_stack([np.ones(n), Xtr_s])
    Ate = np.column_stack([np.ones(Xte_s.shape[0]), Xte_s])

    prior_prec = np.diag([1e-6] + [1.0 / PRIOR_TAU**2] * p)   # weak on intercept
    a0, b0 = 1.0, 1e-4                                         # weak NIG

    Vn_inv = prior_prec + A.T @ A
    Vn = np.linalg.inv(Vn_inv)
    mn = Vn @ (A.T @ ytr)
    an = a0 + n / 2.0
    bn = max(b0 + 0.5 * (ytr @ ytr - mn @ Vn_inv @ mn), 1e-10)

    mean   = Ate @ mn
    scale2 = (bn / an) * (1.0 + np.einsum("ij,jk,ik->i", Ate, Vn, Ate))
    dof    = 2.0 * an
    rng = np.random.default_rng(RNG)
    draws = mean[:, None] + np.sqrt(scale2)[:, None] * rng.standard_t(dof, size=(len(mean), n_draws))
    return draws


# =============================================================================
# 3. WALK-FORWARD
# =============================================================================
def walk_forward(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    Xv, yv, idx = X.values, y.values, X.index
    rows = []
    for t in range(MIN_TRAIN, len(X)):
        dr = fit_predict_conjugate(Xv[:t], yv[:t], Xv[t:t + 1])[0]
        actual = yv[t]
        pmean = dr.mean()
        q = {b: (np.quantile(dr, (1 - b) / 2), np.quantile(dr, 1 - (1 - b) / 2)) for b in BANDS}
        rows.append(dict(
            date=idx[t], pred_mean=pmean, actual=actual,
            dir_pred=int(np.sign(pmean)), dir_actual=int(np.sign(actual)),
            p_up=float((dr > 0).mean()),
            lo50=q[0.50][0], hi50=q[0.50][1], lo80=q[0.80][0], hi80=q[0.80][1],
        ))
        print(f"  {pd.Timestamp(idx[t]).date()}  pred={pmean:+.4f}  actual={actual:+.4f}  P(up)={(dr>0).mean():.2f}")
    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# 4. EVAL + BUTTERFLY MAPPING
# =============================================================================
def evaluate(wf: pd.DataFrame) -> dict:
    hit = (wf["dir_pred"] == wf["dir_actual"]).mean()
    rmse = np.sqrt(((wf["pred_mean"] - wf["actual"]) ** 2).mean())
    rmse_naive = np.sqrt((wf["actual"] ** 2).mean())     # predict-zero benchmark
    cov50 = ((wf["actual"] >= wf["lo50"]) & (wf["actual"] <= wf["hi50"])).mean()
    cov80 = ((wf["actual"] >= wf["lo80"]) & (wf["actual"] <= wf["hi80"])).mean()
    return dict(n=len(wf), dir_hit_rate=hit, rmse=rmse, rmse_naive=rmse_naive,
                skill_vs_naive=1 - rmse / rmse_naive, cov50=cov50, cov80=cov80)


def to_butterfly(wf: pd.DataFrame, spot: pd.Series) -> pd.DataFrame:
    out = wf.copy()
    px = spot.reindex(out.index)
    out["spot"] = px.values
    out["center_strike"] = out["spot"] * np.exp(out["pred_mean"])
    out["wing_lo"] = out["spot"] * np.exp(out["lo80"])
    out["wing_hi"] = out["spot"] * np.exp(out["hi80"])
    out["half_width_pct"] = (np.exp((out["hi80"] - out["lo80"]) / 2) - 1) * 100
    return out


def plot_forecast(wf: pd.DataFrame, fname: Path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(wf.index, wf["lo80"] * 100, wf["hi80"] * 100, alpha=0.18, color="#1f4e79", label="80% band")
    ax.fill_between(wf.index, wf["lo50"] * 100, wf["hi50"] * 100, alpha=0.30, color="#1f4e79", label="50% band")
    ax.plot(wf.index, wf["pred_mean"] * 100, "-", color="#1f4e79", lw=1.3, label="forecast mean")
    ax.scatter(wf.index, wf["actual"] * 100, s=22, color="#c0392b", zorder=5, label="actual")
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_ylabel("next-month 50ETF return (%)")
    ax.set_title("Walk-forward Bayesian FACTOR forecast - 50ETF next-month return")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


def latest_forecast(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame):
    """Live forecast for the upcoming month using ALL complete-case history to
    train and the most recent month-end features (target still unknown)."""
    full = df[[c for c in X.columns]].dropna()
    if full.index[-1] <= X.index[-1]:
        return None  # no newer feature row than the last trainable row
    Xtr, ytr = X.values, y.values
    Xte = full.iloc[[-1]].values
    dr = fit_predict_conjugate(Xtr, ytr, Xte)[0]
    spot = df[SPOT_COL].reindex([full.index[-1]]).iloc[0]
    pmean = dr.mean()
    return dict(date=full.index[-1], spot=spot, pred_mean=pmean, p_up=float((dr > 0).mean()),
                center=spot * np.exp(pmean),
                wing_lo=spot * np.exp(np.quantile(dr, 0.10)),
                wing_hi=spot * np.exp(np.quantile(dr, 0.90)))


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    print("=" * 72)
    print("Butterfly FACTOR-forecast model (Bayesian conjugate, walk-forward) - 50ETF")
    print("=" * 72)
    df = load_panel()
    X, y, used = build_xy(df)
    print(f"panel: {X.index.min().date()} -> {X.index.max().date()} | "
          f"n={len(X)} complete months | {len(used)} predictors")
    print("predictors:", used)
    if len(X) <= MIN_TRAIN + 6:
        print(f"\n!! Only {len(X)} complete months; need > MIN_TRAIN({MIN_TRAIN})+6. "
              "Lower MIN_TRAIN or trim PREDICTORS.")
        return

    print(f"\nwalk-forward (min train = {MIN_TRAIN} months):")
    wf = walk_forward(X, y)

    m = evaluate(wf)
    print("\n--- out-of-sample evaluation ---")
    for k, v in m.items():
        print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")
    print("  (dir_hit_rate>0.5 = directional skill; skill_vs_naive>0 = beats predicting 0;")
    print("   cov50~0.50 & cov80~0.80 = well-calibrated bands for wing sizing.)")

    bf = to_butterfly(wf, df[SPOT_COL])
    bf.to_csv(OUTDIR / "butterfly_factor_forecast.csv")
    plot_forecast(wf, OUTDIR / "factor_forecast_50ETF.png")

    print("\n--- last 6 OOS months: forecast -> butterfly strikes ---")
    cols = ["spot", "pred_mean", "p_up", "center_strike", "wing_lo", "wing_hi", "half_width_pct"]
    print(bf[cols].tail(6).to_string(float_format=lambda v: f"{v:.4f}"))

    nf = latest_forecast(X, y, df)
    if nf:
        print("\n--- LIVE forecast for the upcoming month (features known, target unknown) ---")
        print(f"  as of {pd.Timestamp(nf['date']).date()}  spot={nf['spot']:.4f}  "
              f"pred={nf['pred_mean']:+.4f}  P(up)={nf['p_up']:.2f}")
        print(f"  center strike ~ {nf['center']:.4f} | 80% wings ~ [{nf['wing_lo']:.4f}, {nf['wing_hi']:.4f}]")

    print(f"\nSaved: {OUTDIR/'butterfly_factor_forecast.csv'} and factor_forecast_50ETF.png")


if __name__ == "__main__":
    main()
