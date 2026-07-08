# -*- coding: utf-8 -*-
"""
iron_condor_backtest.py
=======================
The flip side of the timing test. Buying vol lost (variance risk premium works
against the buyer), so here we test the favored side: systematically SELL a capped
short-vol structure -- an iron condor -- and see whether harvesting the VRP beats
buy & hold, after fees & slippage.

Iron condor (defined risk, net CREDIT):
    + buy  put  Kp_long   (far OTM protection)
    - sell put  Kp_short
    - sell call Kc_short
    + buy  call Kc_long   (far OTM protection)
Max profit = net credit (if 50ETF expires between the short strikes).
Max loss   = wing width - credit (capped). Profits when the realized move is small.

Each month-end t (no look-ahead):
  1. Train factor model on data <= t; draw next-month returns (scaled to option DTE).
  2. IV smile at t from the LOCAL option-chain cache (Data/options/option_daily_
     510050.csv, built daily by analysis/options_lab/updater.py from real Sina
     quotes), OTM side, expiry closest to ~1m out. local_iv_smile() below replaces
     butterfly_timing_backtest.fetch_iv_table/choose_expiry/otm_smile (which hit
     SSE live over the network) for backtesting ONLY -- those three are untouched
     and still used live by app.py/tracking.py for TODAY's data, since the
     deployed Streamlit Cloud app has no access to this local cache (Data/options/
     is gitignored, dev-machine only).
  3. Pick condor width (short strikes m steps OTM, protection p steps) by best model
     Kelly growth; trade only if model E[PnL] > 0 AND implied/model >= RATIO_GATE.
  4. Size at fractional Kelly on PnL/max-loss; realize at expiry from actual 50ETF.
Compare: model-filtered condor vs always-on condor vs buy & hold.

Fees & slippage: SLIP_PER_LEG charged on all 4 entry legs; held to expiry (cash
settle, no exit trade).

RUN IN A NORMAL PYTHON ENV. No network calls -- reads the local option-chain +
ETF caches under Data/options/ (run analysis/options_lab/updater.py first to
accumulate/refresh them). Needs the same deps + butterfly_timing_backtest.py
and butterfly_factor_forecast.py alongside.
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

# reuse helpers (importing does not run their main())
from butterfly_factor_forecast import load_panel, build_xy, fit_predict_conjugate, SPOT_COL
from butterfly_timing_backtest import bs_call, bs_put, kelly_opt, R_FREE, MIN_TRAIN

# =============================================================================
# CONFIG
# =============================================================================
OUTDIR        = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)

UNDERLYING    = "510050"
# Data/options/ lives at the repo root, two levels up from analysis/macro_model/.
# Untracked/gitignored on purpose -- local accumulation only, see updater.py.
DATA_DIR      = Path(__file__).resolve().parents[2] / "Data" / "options"
CHAIN_CSV     = DATA_DIR / f"option_daily_{UNDERLYING}.csv"
ETF_CSV       = DATA_DIR / f"etf_daily_{UNDERLYING}.csv"
TARGET_DTE    = 30         # prefer the expiry closest to ~30 calendar days out
MAX_LOOKBACK  = 7          # max days to step back looking for a cached trading day

SLIP_PER_LEG  = 0.0008      # entry friction per leg (yuan/share): bid-ask+commission
SHORT_STEPS   = [1, 2, 3]   # short strikes this many listed steps OTM (grid)
PROT_STEPS    = 2           # protective wings this many steps beyond the shorts
RATIO_GATE    = 1.00        # only sell when ATM implied vol >= model vol (implied not cheap)
KELLY_FRAC    = 0.33        # fraction of full Kelly deployed
KELLY_CAP     = 0.50        # cap on bankroll fraction at risk per trade
ALWAYS_M      = 2           # always-on benchmark: fixed short distance (steps)
FIXED_FRAC    = 0.10        # raw-VRP benchmark: risk this fixed % of bankroll every month


# =============================================================================
# Local data source (replaces live SSE fetch_iv_table/choose_expiry/otm_smile)
# =============================================================================
def load_local_chain() -> pd.DataFrame:
    """Local option-chain cache -- same schema updater.py/data_store.py write,
    read directly here (by design macro_model does not import options_lab code,
    since only macro_model is part of the deployed/tracked app)."""
    df = pd.read_csv(CHAIN_CSV, encoding="utf-8-sig", dtype={"合约代码": str})
    df = df[~df["合约简称"].astype(str).str.endswith("A")]   # drop adjusted contracts
    df["交易日期"] = pd.to_datetime(df["交易日期"])
    df["到期日"] = pd.to_datetime(df["到期日"])
    for col in ("行权价(元)", "距到期日剩余天数", "隐含波动率", "期权标的收盘价"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("交易日期")


def load_local_etf() -> pd.Series:
    df = pd.read_csv(ETF_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].sort_index()


def local_iv_smile(chain: pd.DataFrame, trade_date: pd.Timestamp,
                   target_dte: int = TARGET_DTE, max_back: int = MAX_LOOKBACK):
    """Nearest cached trading day on/before trade_date; the listed expiry
    closest to target_dte (skipping <=5 DTE, same rule as choose_expiry); OTM
    smile (puts below spot, calls above -- same rule as otm_smile), built
    straight off real DTE/expiry dates rather than reconstructing them from a
    YYMM contract code.

    Returns (entry_date, spot, smile, exp_date) or (None, None, None, None).
    """
    day, d = pd.DataFrame(), None
    for back in range(max_back + 1):
        d = trade_date - pd.Timedelta(days=back)
        day = chain[chain["交易日期"] == d]
        if not day.empty:
            break
    if day.empty:
        return None, None, None, None

    spot = float(day["期权标的收盘价"].iloc[0])
    exp_info = day[["到期日", "距到期日剩余天数"]].drop_duplicates()
    exp_info = exp_info[exp_info["距到期日剩余天数"] > 5]
    if exp_info.empty:
        return None, None, None, None
    exp_info = exp_info.assign(
        score=(exp_info["距到期日剩余天数"] - target_dte).abs())
    exp_date = exp_info.sort_values("score").iloc[0]["到期日"]

    sub = day[day["到期日"] == exp_date]
    out = {}
    for k in sorted(sub["行权价(元)"].unique()):
        side = "P" if k < spot else "C"
        row = sub[(sub["行权价(元)"] == k) & (sub["认购(C)认沽(P)类型"] == side)]
        alt = sub[(sub["行权价(元)"] == k) & (sub["认购(C)认沽(P)类型"] != side)]
        iv = row["隐含波动率"].mean() if len(row) else (
            alt["隐含波动率"].mean() if len(alt) else np.nan)
        if np.isfinite(iv) and iv > 0:
            out[k] = iv
    smile = pd.Series(out).sort_index()
    return d, spot, smile, exp_date.date()

# =============================================================================
# Condor pricing / payoff
# =============================================================================
def price_condor(smile, spot, T, m, p, slip=SLIP_PER_LEG):
    ks = smile.index.values
    i0 = int(np.argmin(np.abs(ks - spot)))
    il_p, is_p, is_c, il_c = i0 - m - p, i0 - m, i0 + m, i0 + m + p
    if il_p < 0 or il_c >= len(ks):
        return None
    Kp_l, Kp_s, Kc_s, Kc_l = ks[il_p], ks[is_p], ks[is_c], ks[il_c]
    put = lambda K: bs_put(spot, K, T, R_FREE, smile[K])
    cal = lambda K: bs_call(spot, K, T, R_FREE, smile[K])
    credit = (put(Kp_s) - put(Kp_l)) + (cal(Kc_s) - cal(Kc_l))
    credit -= 4 * slip                                   # entry friction
    wing = max(Kp_s - Kp_l, Kc_l - Kc_s)
    max_loss = wing - credit
    if credit <= 0 or max_loss <= 1e-6:
        return None
    return dict(strikes=(Kp_l, Kp_s, Kc_s, Kc_l), credit=credit, max_loss=max_loss)


def payoff_condor(S, strikes, credit):
    Kp_l, Kp_s, Kc_s, Kc_l = strikes
    payout = (np.maximum(Kp_s - S, 0) - np.maximum(Kp_l - S, 0)
              + np.maximum(S - Kc_s, 0) - np.maximum(S - Kc_l, 0))
    return credit - payout                                # PnL


# =============================================================================
# Backtest
# =============================================================================
def run():
    df = load_panel()
    X, y, used = build_xy(df)
    idx = X.index

    etf = load_local_etf()
    last_daily = etf.index.max()
    chain = load_local_chain()

    rows = []
    for ti in range(MIN_TRAIN, len(X)):
        t = idx[ti]
        draws = fit_predict_conjugate(X.values[:ti], y.values[:ti], X.values[ti:ti + 1])[0]
        model_ann = float(draws.std()) * np.sqrt(12)

        entry_date, spot, smile, exp_date = local_iv_smile(chain, t)
        if entry_date is None:
            rows.append(dict(date=t, traded=0, note="no local chain snapshot"))
            continue
        if pd.Timestamp(exp_date) > last_daily:
            rows.append(dict(date=t, traded=0, note="no realized expiry"))
            continue
        T = max((exp_date - entry_date.date()).days, 1) / 365.0
        if len(smile) < 7:
            rows.append(dict(date=t, traded=0, note="thin smile"))
            continue

        atm_iv = float(smile.iloc[(smile.index - spot).to_series().abs().values.argmin()])
        ratio = atm_iv / model_ann if model_ann > 0 else np.nan

        Tm = T * 12.0
        d_h = draws.mean() * Tm + (draws - draws.mean()) * np.sqrt(Tm)
        ST = spot * np.exp(d_h)
        S_exp = float(etf.asof(pd.Timestamp(exp_date)))

        # --- always-on benchmarks (fixed width, no gate) ---
        #   ao_contrib       : model-Kelly sized (throttled when model sees no edge)
        #   ao_fixed_contrib : RAW VRP -> risk FIXED_FRAC of bankroll every month,
        #                      model-independent. Isolates "does selling vol pay at all".
        ao = price_condor(smile, spot, T, ALWAYS_M, PROT_STEPS)
        ao_contrib, ao_fixed_contrib = np.nan, np.nan
        if ao:
            r_ao = payoff_condor(np.array([S_exp]), ao["strikes"], ao["credit"])[0] / ao["max_loss"]
            f_ao, _ = kelly_opt(payoff_condor(ST, ao["strikes"], ao["credit"]) / ao["max_loss"])
            ao_contrib = min(KELLY_FRAC * f_ao, KELLY_CAP) * r_ao
            ao_fixed_contrib = FIXED_FRAC * r_ao

        # --- model-filtered condor: best width by model Kelly growth, gated ---
        best = None
        for m in SHORT_STEPS:
            pr = price_condor(smile, spot, T, m, PROT_STEPS)
            if not pr:
                continue
            pnl = payoff_condor(ST, pr["strikes"], pr["credit"])
            f_, g_ = kelly_opt(pnl / pr["max_loss"])
            if pnl.mean() > 0 and g_ > 0 and (best is None or g_ > best[1]):
                best = (pr, g_, f_)

        traded, contrib, r_trade, side = 0, np.nan, np.nan, "flat"
        if best is not None and ratio >= RATIO_GATE:
            pr, _, f_ = best
            r_trade = payoff_condor(np.array([S_exp]), pr["strikes"], pr["credit"])[0] / pr["max_loss"]
            f_used = min(KELLY_FRAC * f_, KELLY_CAP)
            contrib = f_used * r_trade
            traded, side = 1, "condor"
            print(f"  {t.date()} condor ratio={ratio:4.2f} cr={pr['credit']:.4f} "
                  f"maxL={pr['max_loss']:.4f} r={r_trade:+.2f} f={f_used:.2f} -> {contrib:+.3f}")

        rows.append(dict(date=t, traded=traded, side=side, ratio=ratio, spot=spot,
                         atm_iv=atm_iv, model_ann=model_ann, expiry=exp_date,
                         S_exp=S_exp, r_trade=r_trade, contrib=contrib,
                         ao_contrib=ao_contrib, ao_fixed_contrib=ao_fixed_contrib))

    return pd.DataFrame(rows).set_index("date"), y


# =============================================================================
# Eval
# =============================================================================
def curve(series_contrib):
    g = np.log1p(series_contrib.fillna(0.0).clip(lower=-0.999))
    return np.exp(g.cumsum())


def main():
    print("=" * 72)
    print("Iron condor VRP backtest (capped short-vol), 50ETF")
    print(f"RATIO_GATE={RATIO_GATE} short_steps={SHORT_STEPS} prot={PROT_STEPS} "
          f"fracKelly={KELLY_FRAC} slip/leg={SLIP_PER_LEG}")
    print("=" * 72)
    bt, y = run()

    rule_eq = curve(bt["contrib"])
    ao_eq = curve(bt["ao_contrib"]) if "ao_contrib" in bt else None
    aof_eq = curve(bt["ao_fixed_contrib"]) if "ao_fixed_contrib" in bt else None
    bh = y.reindex(bt.index).fillna(0.0)
    bh_eq = np.exp(bh.cumsum())

    yrs = max((bt.index[-1] - bt.index[0]).days / 365.25, 1e-9)
    cagr = lambda eq: eq.iloc[-1] ** (1 / yrs) - 1
    mdd = lambda eq: float((eq / eq.cummax() - 1).min())
    rt = bt.loc[bt["traded"] == 1, "r_trade"].dropna()

    bt.to_csv(OUTDIR / "condor_backtest_trades.csv")
    print("\n--- metrics ---")
    print(f"  months              : {len(bt)}")
    print(f"  condor trades       : {int((bt['traded']==1).sum())}  "
          f"({(bt['traded']==1).mean()*100:.0f}% of months)")
    print(f"  rule(model)  tot/CAGR/MDD: {rule_eq.iloc[-1]:.3f} / {cagr(rule_eq):+.3f} / {mdd(rule_eq):+.3f}")
    if ao_eq is not None:
        print(f"  always(Kelly)tot/CAGR/MDD: {ao_eq.iloc[-1]:.3f} / {cagr(ao_eq):+.3f} / {mdd(ao_eq):+.3f}")
    if aof_eq is not None:
        n_aof = int(bt['ao_fixed_contrib'].notna().sum())
        print(f"  RAW VRP fixed{FIXED_FRAC:.0%} tot/CAGR/MDD: {aof_eq.iloc[-1]:.3f} / "
              f"{cagr(aof_eq):+.3f} / {mdd(aof_eq):+.3f}   (sold {n_aof}/{len(bt)} months)")
        rao = bt['ao_fixed_contrib'].dropna() / FIXED_FRAC
        print(f"  RAW VRP per-trade  : winrate {(rao>0).mean()*100:.0f}%  avg r {rao.mean():+.3f}  "
              f"worst {rao.min():+.3f}")
    print(f"  b&h          tot/CAGR/MDD: {bh_eq.iloc[-1]:.3f} / {cagr(bh_eq):+.3f} / {mdd(bh_eq):+.3f}")
    if len(rt):
        print(f"  trade winrate       : {(rt>0).mean()*100:.0f}%   avg trade r: {rt.mean():+.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rule_eq.index, rule_eq, label="model-filtered condor", color="#1f4e79", lw=1.6)
    if aof_eq is not None:
        ax.plot(aof_eq.index, aof_eq, label=f"RAW VRP (fixed {FIXED_FRAC:.0%})", color="#8e44ad", lw=1.3)
    if ao_eq is not None:
        ax.plot(ao_eq.index, ao_eq, label="always-on (model-Kelly)", color="#2e8b57", lw=1.0, ls="--")
    ax.plot(bh_eq.index, bh_eq, label="buy & hold 50ETF", color="#c0392b", lw=1.1)
    ax.axhline(1.0, color="grey", lw=0.7); ax.legend(); ax.set_ylabel("growth of 1")
    ax.set_title("Iron condor (capped short-vol) vs buy & hold, real IV-priced + fees")
    fig.tight_layout(); fig.savefig(OUTDIR / "condor_backtest_equity.png", dpi=130); plt.close(fig)

    print(f"\nSaved: {OUTDIR/'condor_backtest_trades.csv'} and condor_backtest_equity.png")
    print("Caveats: SLIP_PER_LEG is a rough friction proxy; short vol has fat left-tail")
    print("risk (caps help but 2020/2024-type gaps still hit). Not investment advice.")


if __name__ == "__main__":
    main()
