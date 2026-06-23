"""
Butterfly center/wing model: macro indicators -> next-month 50ETF move
======================================================================

Goal
----
The butterfly is the normal-regime position. It pays off best when, one month
out, the 50ETF sits near the body strike. So we need, each month:

    * a DIRECTION + point forecast  -> sets the BODY (center) strike
    * a MAGNITUDE band (dispersion) -> sets the WING width

We forecast the next-month log return of the 50ETF from a panel of Chinese
macro indicators, using a BAYESIAN linear model. Bayesian is the right tool
for the small sample (~65 monthly obs): a sparsity (horseshoe) prior shrinks
irrelevant predictors toward zero, and the POSTERIOR PREDICTIVE distribution
gives the magnitude band directly -- mean -> center, quantiles -> wings, with
honest uncertainty instead of a point estimate plus an ad-hoc error bar.

Look-ahead discipline (strict)
------------------------------
Each macro series is shifted by its real publication lag so a forecast made at
month t only uses data PUBLISHED on or before t. The target is the return over
month t+1 (month-end t -> month-end t+1). Evaluation is walk-forward
(expanding window): every out-of-sample prediction is trained only on data
strictly before it.

Data
----
50ETF daily: qmt_export/etf50_daily.csv (from 2020-01) -> month-end closes.
Macro panel: Data/derived/china_longterm_data.csv (monthly, to ~2025).

Run:  python butterfly_macro_forecast.py
Needs: pandas, numpy, pymc, arviz, matplotlib.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats as _st

warnings.filterwarnings("ignore")
RNG = 20260622

# Estimator backend:
#   "conjugate" -> analytic Normal-Inverse-Gamma Bayesian linear regression
#                  (ridge-style Gaussian prior on coefs). Closed-form Student-t
#                  posterior predictive. Fast, robust at small n. DEFAULT.
#   "pymc"      -> horseshoe-prior MCMC (heavier; needs pymc). Sparser selection
#                  but slow over a long walk-forward.
BACKEND = "conjugate"
PRIOR_TAU = 0.25   # prior SD on standardized-predictor coefficients (shrinkage)

# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]
QMT = ROOT / "qmt_export"
DERIVED = ROOT / "Data" / "derived"
OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

ETF_FILE = QMT / "etf50_daily.csv"
MACRO_FILE = DERIVED / "china_longterm_data.csv"

# Publication lag (in months) for each macro series: how many months AFTER the
# reference month the figure is actually released. Market series (FX, repo) are
# ~real-time (0); surveyed macro is ~1 month.
RELEASE_LAG = {
    "realgdp_monthly_yoy": 1,
    "cpi": 1,
    "IP_yoy": 1,
    "M2_growth": 1,
    "proxy_fai": 1,
    "trade balance": 1,
    "current consumption": 1,
    "US_IP_yoy": 1,
    "CNYUSDSpot": 0,     # FX, real-time
    "neer_yoy": 0,       # FX-derived
    "FR007": 0,          # money-market rate, real-time
}

# Walk-forward: minimum months in the initial training window.
MIN_TRAIN = 36
# Posterior bands to report (for wing sizing).
BANDS = [0.50, 0.80]

# =============================================================================
# 1. PANEL
# =============================================================================
def load_etf_monthly() -> pd.Series:
    e = pd.read_csv(ETF_FILE)
    e["date"] = pd.to_datetime(e["date"].astype(str), format="%Y%m%d")
    e = e.sort_values("date").set_index("date")
    # month-end close, indexed to month-START to match macro stamping
    me = e["close"].resample("MS").last()
    return me.rename("etf_close")


def load_macro() -> pd.DataFrame:
    m = pd.read_csv(MACRO_FILE, parse_dates=["date"]).sort_values("date").set_index("date")
    return m


def build_panel() -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (panel, predictor_cols).
    panel index = forecast month t (month-start).
    target 'y' = log return of 50ETF over month t -> t+1.
    predictors = macro series shifted by release lag (so known by month t).
    """
    etf = load_etf_monthly()
    macro = load_macro()

    # target: next-month log return, attributed to the month it's forecast FROM
    logp = np.log(etf)
    y_next = (logp.shift(-1) - logp).rename("y")   # return over t -> t+1, indexed at t

    preds = {}
    for col, lag in RELEASE_LAG.items():
        if col in macro.columns:
            preds[col] = macro[col].shift(lag)     # value known at month t
    X = pd.DataFrame(preds)

    panel = pd.concat([y_next, X], axis=1)
    # restrict to 50ETF era and rows with complete predictors + target
    panel = panel.loc[panel.index >= etf.index.min()]
    panel = panel.dropna()
    predictor_cols = [c for c in X.columns]
    return panel, predictor_cols


# =============================================================================
# 2. BAYESIAN MODEL (horseshoe prior)
# =============================================================================
def _standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_predict_conjugate(Xtr, ytr, Xte, n_draws=4000):
    """
    Analytic Bayesian linear regression with a Gaussian (ridge) prior on the
    standardized-predictor coefficients and an Inverse-Gamma on the noise
    variance (Normal-Inverse-Gamma conjugate). Returns posterior-PREDICTIVE
    draws for each test row.

    Prior: beta ~ N(0, tau^2 I) on standardized X (intercept weakly informative).
    This is the conjugate analogue of ridge / a Bayesian shrinkage prior, ideal
    at small n. The posterior predictive is Student-t; we draw from it so the
    downstream banding code is identical to the MCMC path.
    """
    Xtr_s, Xte_s = _standardize(Xtr, Xte)
    n, p = Xtr_s.shape
    # design with intercept; large prior variance on the intercept (col 0)
    A = np.column_stack([np.ones(n), Xtr_s])
    Ate = np.column_stack([np.ones(Xte_s.shape[0]), Xte_s])
    k = A.shape[1]

    # prior precision: tiny on intercept, 1/tau^2 on slopes
    prior_prec = np.diag([1e-6] + [1.0 / PRIOR_TAU**2] * p)
    # NIG hyperparams (weak)
    a0, b0 = 1.0, 1e-4

    Vn_inv = prior_prec + A.T @ A
    Vn = np.linalg.inv(Vn_inv)
    mn = Vn @ (A.T @ ytr)                       # posterior mean coefficients
    an = a0 + n / 2.0
    bn = b0 + 0.5 * (ytr @ ytr - mn @ Vn_inv @ mn)
    bn = max(bn, 1e-10)

    # posterior predictive at test rows is Student-t:
    #   mean   = Ate @ mn
    #   scale^2= (bn/an) * (1 + diag(Ate Vn Ate^T))
    #   dof    = 2*an
    mean = Ate @ mn
    scale2 = (bn / an) * (1.0 + np.einsum("ij,jk,ik->i", Ate, Vn, Ate))
    dof = 2.0 * an
    rng = np.random.default_rng(RNG)
    draws = mean[:, None] + np.sqrt(scale2)[:, None] * rng.standard_t(dof, size=(len(mean), n_draws))
    return draws


def fit_predict_pymc(Xtr, ytr, Xte, draws=800, tune=800):
    import pymc as pm
    Xtr_s, Xte_s = _standardize(Xtr, Xte)
    p = Xtr_s.shape[1]
    with pm.Model():
        tau = pm.HalfCauchy("tau", beta=0.1)
        lam = pm.HalfCauchy("lam", beta=1.0, shape=p)
        z = pm.Normal("z", 0.0, 1.0, shape=p)
        beta = pm.Deterministic("beta", z * lam * tau)
        intercept = pm.Normal("intercept", 0.0, 0.1)
        sigma = pm.HalfNormal("sigma", 0.1)
        pm.Normal("y_obs", mu=intercept + pm.math.dot(Xtr_s, beta), sigma=sigma, observed=ytr)
        idata = pm.sample(draws=draws, tune=tune, chains=2, cores=1,
                          target_accept=0.97, random_seed=RNG, progressbar=False)
    post = idata.posterior
    b = post["beta"].stack(s=("chain", "draw")).values
    a = post["intercept"].stack(s=("chain", "draw")).values
    s = post["sigma"].stack(s=("chain", "draw")).values
    rng = np.random.default_rng(RNG)
    mu_te = a[None, :] + Xte_s @ b
    return mu_te + rng.normal(size=mu_te.shape) * s[None, :]


def fit_predict(Xtr, ytr, Xte):
    """Standardization uses TRAIN moments only (no leakage)."""
    if BACKEND == "pymc":
        return fit_predict_pymc(Xtr, ytr, Xte)
    return fit_predict_conjugate(Xtr, ytr, Xte)


# =============================================================================
# 3. WALK-FORWARD
# =============================================================================
def walk_forward(panel: pd.DataFrame, predictor_cols: list[str]) -> pd.DataFrame:
    y = panel["y"].values
    X = panel[predictor_cols].values
    idx = panel.index
    rows = []
    for t in range(MIN_TRAIN, len(panel)):
        Xtr, ytr = X[:t], y[:t]
        Xte = X[t:t + 1]
        dr = fit_predict(Xtr, ytr, Xte)[0]      # (S,) posterior-predictive draws
        actual = y[t]
        pmean = dr.mean()
        q = {b: (np.quantile(dr, (1 - b) / 2), np.quantile(dr, 1 - (1 - b) / 2))
             for b in BANDS}
        rows.append(dict(
            date=idx[t], pred_mean=pmean, actual=actual,
            dir_pred=int(np.sign(pmean)), dir_actual=int(np.sign(actual)),
            p_up=float((dr > 0).mean()),
            lo50=q[0.50][0], hi50=q[0.50][1],
            lo80=q[0.80][0], hi80=q[0.80][1],
        ))
        print(f"  {idx[t].date()}  pred={pmean:+.4f}  actual={actual:+.4f}  "
              f"P(up)={(dr>0).mean():.2f}")
    return pd.DataFrame(rows).set_index("date")


# =============================================================================
# 4. EVAL + BUTTERFLY MAPPING
# =============================================================================
def evaluate(wf: pd.DataFrame) -> dict:
    hit = (wf["dir_pred"] == wf["dir_actual"]).mean()
    rmse = np.sqrt(((wf["pred_mean"] - wf["actual"]) ** 2).mean())
    rmse_naive = np.sqrt((wf["actual"] ** 2).mean())     # predict 0 return
    cov50 = ((wf["actual"] >= wf["lo50"]) & (wf["actual"] <= wf["hi50"])).mean()
    cov80 = ((wf["actual"] >= wf["lo80"]) & (wf["actual"] <= wf["hi80"])).mean()
    return dict(n=len(wf), dir_hit_rate=hit, rmse=rmse, rmse_naive=rmse_naive,
                skill_vs_naive=1 - rmse / rmse_naive,
                cov50=cov50, cov80=cov80)


def to_butterfly(wf: pd.DataFrame, etf_monthly: pd.Series) -> pd.DataFrame:
    """Translate each month's forecast into butterfly strikes (relative to last close)."""
    out = wf.copy()
    # center strike = current price * exp(pred_mean); wings from the 80% band
    px = etf_monthly.reindex(out.index)
    out["spot"] = px.values
    out["center_strike"] = out["spot"] * np.exp(out["pred_mean"])
    out["wing_lo"] = out["spot"] * np.exp(out["lo80"])
    out["wing_hi"] = out["spot"] * np.exp(out["hi80"])
    out["half_width_pct"] = (np.exp((out["hi80"] - out["lo80"]) / 2) - 1) * 100
    return out


def plot_forecast(wf: pd.DataFrame, fname: Path):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(wf.index, wf["lo80"] * 100, wf["hi80"] * 100, alpha=0.18,
                    color="#1f4e79", label="80% band")
    ax.fill_between(wf.index, wf["lo50"] * 100, wf["hi50"] * 100, alpha=0.30,
                    color="#1f4e79", label="50% band")
    ax.plot(wf.index, wf["pred_mean"] * 100, "-", color="#1f4e79", lw=1.3,
            label="forecast mean")
    ax.scatter(wf.index, wf["actual"] * 100, s=22, color="#c0392b", zorder=5,
               label="actual")
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_ylabel("next-month 50ETF return (%)")
    ax.set_title("Walk-forward Bayesian macro forecast â€?50ETF next-month return")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("Butterfly macro-forecast model (Bayesian, walk-forward)")
    print("=" * 70)
    panel, predictor_cols = build_panel()
    print(f"panel: {panel.index.min().date()} -> {panel.index.max().date()}  "
          f"| n={len(panel)} months | {len(predictor_cols)} predictors")
    print("predictors:", predictor_cols)

    print(f"\nwalk-forward (min train = {MIN_TRAIN} months):")
    wf = walk_forward(panel, predictor_cols)

    metrics = evaluate(wf)
    print("\n--- out-of-sample evaluation ---")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v:.4f}" if isinstance(v, float) else f"  {k:16s}: {v}")

    etf_monthly = load_etf_monthly()
    bf = to_butterfly(wf, etf_monthly)
    bf.to_csv(OUTDIR / "butterfly_macro_forecast.csv")
    plot_forecast(wf, OUTDIR / "macro_forecast_50ETF.png")

    print("\n--- last 6 months: forecast -> butterfly strikes ---")
    cols = ["spot", "pred_mean", "p_up", "center_strike", "wing_lo", "wing_hi", "half_width_pct"]
    print(bf[cols].tail(6).to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved: {OUTDIR/'butterfly_macro_forecast.csv'} and macro_forecast_50ETF.png")


if __name__ == "__main__":
    main()
