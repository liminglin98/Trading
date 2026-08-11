"""csi1000_skew_risk_reversal_backtest.py -- backtest harvesting the CSI1000
(000852/MO) put-IV skew premium documented in csi1000_put_skew_check.py
(5%-OTM put IV richer than call IV on 88.5% of days, mean 2.24 vol pts,
2022-07-22 to 2026-08-10, vs. much weaker CSI300/near-zero SSE50 controls --
a small-cap-hedging-demand signature, not generic index skew).

Two structures, run through the same engine via --structure:
  - risk_reversal (primary): short a target-delta OTM put + long a
    target-delta OTM call, same expiry. Isolates the skew premium
    specifically -- the short put collects the rich put vol, the long call
    caps the tail-gap risk that made the put rich in the first place, and
    the pair stays closer to vol-neutral than an outright short put. Still
    carries real delta exposure from the strike asymmetry, and real gap
    risk on exactly the small-cap-crash scenarios that price the put rich
    (see the Feb-2024 DMA-liquidation check in csi1000_put_skew_check.py) --
    that's the risk this structure is being paid to bear, not a flaw in the
    backtest.
  - strangle (control): short OTM call + short OTM put, same target delta on
    both sides -- the generic short-vol / variance-risk-premium structure
    every other short-vol script in this repo already sells. Comparing the
    two isolates the skew-specific edge from generic vol-selling P&L (see
    compare_structures()) -- without this control, any risk-reversal P&L is
    unattributable to skew specifically.

Delta source: BS-derived (bs_iv.implied_vol + bs_iv.bs_delta) rather than
the raw vendor ds.C_DELTA column, for consistency with the already-BS-derived
IV pipeline this whole feature sits on (ds.C_IV is always NaN for this
dataset; ds.C_THETA/GAMMA/VEGA are 0% populated despite the same vendor feed
claiming ds.C_DELTA is 93% populated, which is reason for some doubt about
that field). Validated 2026-08-11 on the full CSI1000 chain (n=17,615
non-null, cp*expiry*strike sample): raw C_DELTA vs. BS-derived delta
correlate at 0.996, mean diff -0.030 (raw runs slightly more negative/less
positive across all moneyness buckets), median abs diff 0.017 -- close
agreement, not identical. Raw C_DELTA is still logged on every selected leg
as a permanent cross-check column.

Margin: CFFEX index-option futures-style formula (cffex_margin.py), NOT
margin.py's SSE/SZSE formula (wrong multiplier and formula shape for these
contracts). Margin is priced off ds.C_SETTLE (the exchange's own margin
formula input), computed once at entry and held fixed for the trade's life,
same convention delta_strangle_short_backtest.py uses. P&L is marked on
ds.C_CLOSE throughout, as in every other script in this repo. See
cffex_margin.py's docstring for the margin-parameter approximation caveat
(MO's 2022 launch-date margin_ratio/min_guarantee, held static across the
whole backtest window) -- every return-on-margin figure this script prints
should be read with that caveat in mind.

Skew-credit decomposition: at each risk-reversal entry, the put leg's
BS-implied price is compared against a counterfactual price computed at the
call leg's own IV (i.e., "what would this put cost with no skew, at the same
delta"), isolating the entry-time skew premium independent of what happens
to spot/vol afterward -- see skew_credit_decomposition(). Cross-checked
against the first-order vega approximation vega*(put_iv-call_iv).

Measured results (2026-08-11, target_delta=0.20, dte=[15,45], full history
2022-07-22 to 2026-08-10; see skew_signal_investigation_leaderboard.md #4
for the full writeup): risk_reversal (n=96) total pnl +70,560 yuan, 33% win
rate (asymmetric-payoff signature -- most trades are small stop-losses,
gains come from the minority held to expiry); discovery (<2024) -24,500
yuan / -0.17% mean_r, holdout (>=2024) +95,060 yuan / +3.24% mean_r -- the
edge does not decay post-2024, if anything strengthens. The strangle
control (n=77) is net negative in BOTH periods (-33,160 / -209,480 yuan),
confirming the risk-reversal's edge is skew-specific, not generic
short-vol harvesting. Skew-credit decomposition: mean +1,408 yuan/trade,
91% of trades positive at entry, corr=1.00 vs. the vega approximation --
realized mean pnl/trade (~735 yuan) runs below this, consistent with part
of the entry-time edge being eaten by the same small-cap gap risk that
prices the put rich to begin with. Not yet swept over target_delta/DTE --
these numbers are one parameter cell, not a validated optimum.

Usage:
    python csi1000_skew_risk_reversal_backtest.py --structure risk_reversal
    python csi1000_skew_risk_reversal_backtest.py --structure strangle
    python csi1000_skew_risk_reversal_backtest.py --compare
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bs_iv
import cffex_margin as cm
import consolidated_store as cs
import data_store as ds

OUT_DIR = Path(__file__).resolve().parent / "out"
SPLIT_DATE = pd.Timestamp("2024-01-01")

UNDERLYING = "000852"
UNDERLYING_NAME = "CSI1000 (MO)"
DTE_MIN = 15
DTE_MAX = 45
TARGET_DELTA = 0.20
MAX_HOLD_DAYS = 25
STOP_LOSS_MULT = 2.5
PROFIT_TARGET_MULT = 0.5  # strangle only

# ---- palette (light mode; see dataviz skill references/palette.md) ----
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"
C_RED = "#e34948"
C_GREEN = "#2f9e5c"


# ------------------------------------------------------------------ strikes
def _leg_metrics(day: pd.DataFrame, expiry: pd.Timestamp, cp: str, spot: float) -> pd.DataFrame:
    """Rows for this (expiry, cp) with BS-derived iv/delta columns attached.
    Drops rows where IV can't be inverted (stale/illiquid/off-band quotes,
    see bs_iv.implied_vol)."""
    rows = day[(day[ds.C_EXPIRY] == expiry) & (day[ds.C_CP] == cp) & (day[ds.C_CLOSE] > 0)].copy()
    if rows.empty or spot <= 0:
        return rows.iloc[0:0]
    t = float(rows[ds.C_DTE].iloc[0]) / 365.0
    ivs, deltas = [], []
    for _, r in rows.iterrows():
        iv = bs_iv.implied_vol(cp, spot, float(r[ds.C_STRIKE]), t, float(r[ds.C_CLOSE]))
        d = bs_iv.bs_delta(cp, spot, float(r[ds.C_STRIKE]), t, iv) if iv == iv else float("nan")
        ivs.append(iv)
        deltas.append(d)
    rows["bs_iv"] = ivs
    rows["bs_delta"] = deltas
    return rows.dropna(subset=["bs_delta"])


def _delta_leg(day: pd.DataFrame, expiry: pd.Timestamp, cp: str, target_delta: float,
               spot: float) -> pd.Series | None:
    """Listed strike (for this expiry/cp) whose BS-derived delta is closest
    to +target_delta (calls) or -target_delta (puts)."""
    rows = _leg_metrics(day, expiry, cp, spot)
    if rows.empty:
        return None
    want = target_delta if cp == "C" else -target_delta
    idx = (rows["bs_delta"] - want).abs().idxmin()
    return rows.loc[idx]


def _intrinsic(cp: str, strike: float, spot: float) -> float:
    return max(spot - strike, 0.0) if cp == "C" else max(strike - spot, 0.0)


def _near_expiry(day: pd.DataFrame, dte_min: int, dte_max: int) -> pd.Timestamp | None:
    exps = day.drop_duplicates(ds.C_EXPIRY).set_index(ds.C_EXPIRY)[ds.C_DTE]
    eligible = exps[(exps >= dte_min) & (exps <= dte_max)]
    if eligible.empty:
        return None
    return pd.Timestamp(eligible.idxmin())


def _prepped_chain(underlying: str, start_date: str | None = None) -> pd.DataFrame:
    chain = cs.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    chain = chain[chain[ds.C_CLOSE] > 0]
    if start_date is not None:
        chain = chain[chain[ds.C_DATE] >= pd.Timestamp(start_date)]
    return chain


# ------------------------------------------------------------------ engine
def run(underlying: str = UNDERLYING, structure: str = "risk_reversal",
       target_delta: float = TARGET_DELTA, call_target_delta: float | None = None,
       dte_min: int = DTE_MIN, dte_max: int = DTE_MAX, max_hold_days: int = MAX_HOLD_DAYS,
       stop_loss_mult: float = STOP_LOSS_MULT, profit_target_mult: float | None = PROFIT_TARGET_MULT,
       start_date: str | None = None) -> dict:
    """structure: "risk_reversal" (short put + long call) or "strangle"
    (short put + short call), both delta-targeted, always-on roll (re-enter
    immediately the day a position closes -- the discovery script found the
    put premium present on 88.5% of days, i.e. structural, not event-driven,
    so no entry signal gate is used). profit_target_mult is ignored for
    risk_reversal (its asymmetric payoff is the point; capping gains early
    defeats the structure)."""
    assert structure in ("risk_reversal", "strangle")
    call_target_delta = call_target_delta if call_target_delta is not None else target_delta
    chain = _prepped_chain(underlying, start_date)
    dates = sorted(chain[ds.C_DATE].unique())
    by_date = dict(tuple(chain.groupby(ds.C_DATE)))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_pos: dict | None = None
    realized = 0.0

    for i, dt in enumerate(dates):
        dt = pd.Timestamp(dt)
        day = by_date[dt]
        spot_rows = day[day[ds.C_UND_CLOSE].notna()]
        spot = float(spot_rows[ds.C_UND_CLOSE].iloc[0]) if not spot_rows.empty else float("nan")

        if open_pos is not None:
            call = day[day[ds.C_CODE] == open_pos["call_code"]]
            put = day[day[ds.C_CODE] == open_pos["put_code"]]
            call_px = float(call[ds.C_CLOSE].iloc[0]) if not call.empty else open_pos["last_call_px"]
            put_px = float(put[ds.C_CLOSE].iloc[0]) if not put.empty else open_pos["last_put_px"]
            open_pos["last_call_px"], open_pos["last_put_px"] = call_px, put_px

            expired = dt >= open_pos["expiry"]
            days_held = i - open_pos["entry_idx"]
            reason = None
            if expired:
                call_px = _intrinsic("C", open_pos["call_strike"], spot)
                put_px = _intrinsic("P", open_pos["put_strike"], spot)
                reason = "expiry"

            call_sign = open_pos["call_dir"]
            put_sign = -1  # put is always short in both structures
            mark = call_sign * call_px + put_sign * put_px
            pnl = mark - open_pos["cost"]
            basis = abs(open_pos["cost"]) if open_pos["cost"] != 0 else (call_px + put_px)

            if reason is None:
                if stop_loss_mult is not None and pnl <= -stop_loss_mult * basis:
                    reason = "stop_loss"
                elif structure == "strangle" and profit_target_mult is not None and pnl >= profit_target_mult * basis:
                    reason = "profit_target"
                elif days_held >= max_hold_days:
                    reason = "max_hold"

            if reason:
                realized += pnl
                trades.append({
                    "structure": structure, "entry_date": open_pos["entry_date"], "exit_date": dt,
                    "expiry": open_pos["expiry"], "exit_reason": reason,
                    "call_strike": open_pos["call_strike"], "put_strike": open_pos["put_strike"],
                    "entry_spot": open_pos["entry_spot"], "exit_spot": spot,
                    "call_delta_at_entry": open_pos["call_delta_at_entry"],
                    "put_delta_at_entry": open_pos["put_delta_at_entry"],
                    "call_raw_delta_at_entry": open_pos["call_raw_delta_at_entry"],
                    "put_raw_delta_at_entry": open_pos["put_raw_delta_at_entry"],
                    "call_iv_at_entry": open_pos["call_iv_at_entry"],
                    "put_iv_at_entry": open_pos["put_iv_at_entry"],
                    "skew_edge_yuan": open_pos.get("skew_edge_yuan"),
                    "vega_approx_edge_yuan": open_pos.get("vega_approx_edge_yuan"),
                    "net_cost": open_pos["cost"], "pnl": pnl,
                    "days_held": days_held, "margin_yuan": open_pos["margin"],
                })
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl, "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry: always-on roll, near-month expiry in [dte_min, dte_max]
        if spot == spot:  # not NaN
            expiry = _near_expiry(day, dte_min, dte_max)
            if expiry is not None:
                put_row = _delta_leg(day, expiry, "P", target_delta, spot)
                call_row = _delta_leg(day, expiry, "C", call_target_delta, spot)
                if put_row is not None and call_row is not None:
                    call_dir = -1 if structure == "strangle" else 1
                    call_px, put_px = float(call_row[ds.C_CLOSE]), float(put_row[ds.C_CLOSE])
                    call_strike, put_strike = float(call_row[ds.C_STRIKE]), float(put_row[ds.C_STRIKE])
                    cost = call_dir * call_px + (-1) * put_px

                    legs = [{"expiry": expiry, "cp": "P", "strike": put_strike, "direction": -1,
                            "qty": 1, "entry_price": float(put_row[ds.C_SETTLE])}]
                    if call_dir == -1:
                        legs.append({"expiry": expiry, "cp": "C", "strike": call_strike, "direction": -1,
                                    "qty": 1, "entry_price": float(call_row[ds.C_SETTLE])})
                    margin_yuan, _ = cm.compute_deposit_cffex(pd.DataFrame(legs), spot, underlying)

                    skew_edge_yuan = vega_approx_edge_yuan = None
                    if structure == "risk_reversal":
                        t = float(put_row[ds.C_DTE]) / 365.0
                        counterfactual_put = bs_iv.bs_price(
                            "P", spot, put_strike, t, float(call_row["bs_iv"]))
                        skew_edge_yuan = (put_px - counterfactual_put) * cm.MULTIPLIER
                        vega = bs_iv.bs_vega(spot, put_strike, t, float(put_row["bs_iv"]))
                        vega_approx_edge_yuan = vega * (float(put_row["bs_iv"]) - float(call_row["bs_iv"])) * cm.MULTIPLIER

                    open_pos = {
                        "entry_date": dt, "entry_idx": i, "entry_spot": spot,
                        "expiry": expiry, "call_strike": call_strike, "put_strike": put_strike,
                        "call_code": call_row[ds.C_CODE], "put_code": put_row[ds.C_CODE],
                        "call_dir": call_dir,
                        "last_call_px": call_px, "last_put_px": put_px,
                        "cost": cost, "margin": margin_yuan,
                        "call_delta_at_entry": float(call_row["bs_delta"]),
                        "put_delta_at_entry": float(put_row["bs_delta"]),
                        "call_raw_delta_at_entry": float(call_row[ds.C_DELTA]) if call_row[ds.C_DELTA] == call_row[ds.C_DELTA] else None,
                        "put_raw_delta_at_entry": float(put_row[ds.C_DELTA]) if put_row[ds.C_DELTA] == put_row[ds.C_DELTA] else None,
                        "call_iv_at_entry": float(call_row["bs_iv"]),
                        "put_iv_at_entry": float(put_row["bs_iv"]),
                        "skew_edge_yuan": skew_edge_yuan,
                        "vega_approx_edge_yuan": vega_approx_edge_yuan,
                    }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0, "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df}


# ------------------------------------------------------------------ skew decomposition
def skew_credit_decomposition(trades: pd.DataFrame) -> dict:
    """Summarizes the entry-time skew_edge_yuan (actual put price minus the
    counterfactual put price at the call leg's own IV -- see run()) computed
    for every risk_reversal trade. Positive means the put leg was sold at a
    genuine skew premium over its "no-skew" counterfactual price at entry,
    independent of what happened to spot/vol afterward. Cross-checked
    against the first-order vega approximation."""
    edge = trades["skew_edge_yuan"].dropna()
    vega_approx = trades["vega_approx_edge_yuan"].dropna()
    if edge.empty:
        return {"n": 0}
    return {
        "n": len(edge),
        "mean_yuan": float(edge.mean()), "median_yuan": float(edge.median()),
        "pct_positive": float((edge > 0).mean()),
        "vega_approx_mean_yuan": float(vega_approx.mean()) if len(vega_approx) else float("nan"),
        "corr_vs_vega_approx": float(edge.corr(vega_approx)) if len(vega_approx) > 1 else float("nan"),
    }


# ------------------------------------------------------------------ report
def _split(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    disc = trades[trades["entry_date"] < SPLIT_DATE]
    hold = trades[trades["entry_date"] >= SPLIT_DATE]
    return disc, hold


def _period_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "total_pnl": np.nan, "mean_r": np.nan, "win_rate": np.nan}
    pnl_yuan = trades["pnl"] * cm.MULTIPLIER
    r = pnl_yuan / trades["margin_yuan"]
    return {"n": len(trades), "total_pnl": float(pnl_yuan.sum()),
            "mean_r": float(r.mean()), "win_rate": float((trades["pnl"] > 0).mean())}


def report(res: dict, underlying: str = UNDERLYING, structure: str = "risk_reversal",
          target_delta: float = TARGET_DELTA) -> None:
    trades, equity = res["trades"], res["equity"]
    if trades.empty:
        print(f"{underlying} ({UNDERLYING_NAME}) [{structure}]: no trades.")
        return
    pnl_yuan = trades["pnl"] * cm.MULTIPLIER
    eq = equity.set_index("date")["equity"] * cm.MULTIPLIER
    dd = eq - eq.cummax()
    daily = eq.diff().dropna()
    sharpe = daily.mean() / daily.std() * np.sqrt(244) if daily.std() > 0 else np.nan
    avg_margin = trades["margin_yuan"].mean()
    total_pnl = pnl_yuan.sum()
    span_years = (equity["date"].max() - equity["date"].min()).days / 365.25
    return_on_margin_yr = (total_pnl / avg_margin) / span_years if span_years > 0 and avg_margin else np.nan

    print(f"{underlying} ({UNDERLYING_NAME}) [{structure}]: target_delta={target_delta:.2f}, "
          f"dte=[{DTE_MIN},{DTE_MAX}], max_hold={MAX_HOLD_DAYS}d, stop_loss={STOP_LOSS_MULT:.1f}x")
    print(f"  {equity['date'].min().date()} -> {equity['date'].max().date()}")
    print(f"  n_trades={len(trades)}  avg_days_held={trades['days_held'].mean():.1f}  "
          f"avg call/put BS-delta at entry={trades['call_delta_at_entry'].mean():+.2f}/"
          f"{trades['put_delta_at_entry'].mean():+.2f}")
    print(f"  total_pnl={total_pnl:,.0f} yuan  win_rate={(trades['pnl']>0).mean():.0%}  "
          f"mean_r={(pnl_yuan/trades['margin_yuan']).mean():+.2%} (pnl/margin per trade)")
    print(f"  max_drawdown={dd.min():,.0f} yuan  sharpe(daily,ann)={sharpe:.2f}")
    print(f"  avg entry margin (CFFEX naked, APPROX -- see cffex_margin.py caveat)={avg_margin:,.0f} yuan  "
          f"return_on_margin/yr~={return_on_margin_yr:.1%}")
    print("  exit reasons:", trades["exit_reason"].value_counts().to_dict())

    disc, hold = _split(trades)
    d_stats, h_stats = _period_stats(disc), _period_stats(hold)
    print(f"  discovery (<{SPLIT_DATE.date()}): n={d_stats['n']}  total_pnl={d_stats['total_pnl']:,.0f}  "
          f"mean_r={d_stats['mean_r']:+.2%}  win_rate={d_stats['win_rate']:.0%}" if d_stats["n"] else
          f"  discovery (<{SPLIT_DATE.date()}): no trades")
    print(f"  holdout   (>={SPLIT_DATE.date()}): n={h_stats['n']}  total_pnl={h_stats['total_pnl']:,.0f}  "
          f"mean_r={h_stats['mean_r']:+.2%}  win_rate={h_stats['win_rate']:.0%}" if h_stats["n"] else
          f"  holdout   (>={SPLIT_DATE.date()}): no trades")

    if structure == "risk_reversal":
        dec = skew_credit_decomposition(trades)
        if dec["n"]:
            print(f"  skew-credit decomposition (entry-time, put vs. counterfactual at call IV): "
                  f"n={dec['n']}  mean={dec['mean_yuan']:,.0f} yuan  median={dec['median_yuan']:,.0f} yuan  "
                  f"pct_positive={dec['pct_positive']:.0%}")
            print(f"    vega-approx cross-check: mean={dec['vega_approx_mean_yuan']:,.0f} yuan  "
                  f"corr(exact, approx)={dec['corr_vs_vega_approx']:.2f}")

    trades["year"] = trades["entry_date"].dt.year
    print("  by year:")
    for yr, g in trades.groupby("year"):
        print(f"    {yr}: n={len(g)}  pnl={g['pnl'].sum()*cm.MULTIPLIER:>10,.0f}  win%={(g['pnl']>0).mean():.0%}")

    OUT_DIR.mkdir(exist_ok=True)
    trades_out = OUT_DIR / f"csi1000_skew_{structure}_trades_{underlying}_{target_delta:.2f}.csv"
    trades.to_csv(trades_out, index=False, encoding="utf-8-sig")
    print(f"  Trades saved to {trades_out}\n")


def plot(res: dict, structure: str, out_path: Path, target_delta: float = TARGET_DELTA) -> None:
    equity = res["equity"]
    eq = equity.set_index("date")["equity"] * cm.MULTIPLIER
    dd = eq - eq.cummax()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]}, facecolor=C_SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURFACE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(C_AXIS)
        ax.spines["bottom"].set_color(C_AXIS)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        ax.grid(axis="y", color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    ax1.plot(eq.index, eq.values, color=C_BLUE, linewidth=1.6)
    ax1.set_ylabel("cumulative P&L (yuan)", color=C_INK2, fontsize=10)
    ax1.set_title(f"{UNDERLYING_NAME}: skew {structure} (target_delta={target_delta:.2f})",
                  color=C_INK, fontsize=12, loc="left", pad=10)

    ax2.fill_between(dd.index, 0, dd.values, color=C_RED, alpha=0.55, linewidth=0)
    ax2.set_ylabel("drawdown (yuan)", color=C_INK2, fontsize=10)

    fig.autofmt_xdate()
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=C_SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------ comparison
def compare_structures(underlying: str = UNDERLYING, target_delta: float = TARGET_DELTA,
                       **kwargs) -> pd.DataFrame:
    """Runs risk_reversal and strangle through the identical engine (same
    target_delta/dte/roll cadence) and reports them side by side, discovery
    vs. holdout -- the marginal difference between the two is the closest
    full-backtest measurement of whether tilting into the skew (risk
    reversal) beats generic short-vol harvesting (strangle) at the same
    delta. They will not share entry/exit dates (different exit rules ->
    different holding periods), so this is an independent-equity-curves
    comparison, not a paired backtest."""
    rows = []
    results = {}
    for structure in ("risk_reversal", "strangle"):
        res = run(underlying=underlying, structure=structure, target_delta=target_delta, **kwargs)
        results[structure] = res
        trades = res["trades"]
        disc, hold = (trades[trades["entry_date"] < SPLIT_DATE], trades[trades["entry_date"] >= SPLIT_DATE]) \
            if not trades.empty else (trades, trades)
        d, h = _period_stats(disc), _period_stats(hold)
        rows.append({
            "structure": structure, "n_trades": len(trades),
            "disc_n": d["n"], "disc_total_pnl": d["total_pnl"], "disc_mean_r": d["mean_r"],
            "hold_n": h["n"], "hold_total_pnl": h["total_pnl"], "hold_mean_r": h["mean_r"],
        })
    grid = pd.DataFrame(rows)
    print(f"{underlying} ({UNDERLYING_NAME}): risk_reversal vs. strangle @ target_delta={target_delta:.2f}, "
          f"discovery < {SPLIT_DATE.date()} <= holdout")
    print(grid.to_string(index=False))
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"csi1000_skew_structure_comparison_{underlying}_{target_delta:.2f}.csv"
    grid.to_csv(out_path, index=False)
    print(f"  Comparison saved to {out_path}\n")
    return grid


# ------------------------------------------------------------------ chart_common adapter
def build_underlying_price_series(underlying: str = UNDERLYING) -> pd.DataFrame:
    """date/close DataFrame of the near-month parity-forward spot, for
    chart_common.py's 3-panel HTML chart -- no CSI1000 OHLC CSV exists in
    Data/index/ (unlike the SSE ETF underlyings chart_common.load_price_series
    expects), so this rebuilds the same near-month spot series
    csi1000_put_skew_check.py's skew_series() already extracts, standalone."""
    chain = _prepped_chain(underlying)
    calls = chain[(chain[ds.C_CP] == "C") & chain[ds.C_UND_CLOSE].notna()]
    rows = []
    for dt, day_calls in calls.groupby(ds.C_DATE):
        expiry = _near_expiry(day_calls, DTE_MIN, DTE_MAX)
        if expiry is None:
            eligible = day_calls.drop_duplicates(ds.C_EXPIRY).set_index(ds.C_EXPIRY)[ds.C_DTE]
            if eligible.empty:
                continue
            expiry = eligible.idxmin()
        exp_calls = day_calls[day_calls[ds.C_EXPIRY] == expiry]
        rows.append({"date": dt, "close": float(exp_calls[ds.C_UND_CLOSE].iloc[0])})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------------ main
def main(underlying: str, structure: str, target_delta: float, do_compare: bool) -> None:
    if do_compare:
        compare_structures(underlying=underlying, target_delta=target_delta)
        return
    res = run(underlying=underlying, structure=structure, target_delta=target_delta)
    report(res, underlying, structure, target_delta)
    out_path = OUT_DIR / f"csi1000_skew_{structure}_{underlying}_{target_delta:.2f}.png"
    plot(res, structure, out_path, target_delta)
    print(f"  Chart saved to {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default=UNDERLYING)
    ap.add_argument("--structure", default="risk_reversal", choices=["risk_reversal", "strangle"])
    ap.add_argument("--target-delta", type=float, default=TARGET_DELTA, dest="target_delta")
    ap.add_argument("--compare", action="store_true", dest="do_compare",
                    help="run both structures and report them side by side")
    args = ap.parse_args()
    main(args.underlying, args.structure, args.target_delta, args.do_compare)
