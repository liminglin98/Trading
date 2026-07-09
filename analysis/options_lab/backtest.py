"""Template + rules backtest engine over the daily option chain history.

Strategy = leg template (straddle/strangle/butterfly/condor/vertical/
calendar/custom) + entry rules (DTE window, IV percentile filter) + exit
rules (profit target, stop loss, DTE exit, expiry settlement).

Fills are assumed at the daily close of the signal day. P&L is per one
combo unit, in underlying-price units; multiply by data_store.MULTIPLIER
for yuan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import data_store as ds


# ------------------------------------------------------------------ specs
@dataclass
class LegSpec:
    cp: str               # "C" or "P"
    direction: int        # +1 long, -1 short
    ratio: int = 1
    select: str = "offset"  # "offset" (strike steps from ATM) or "delta"
    value: float = 0.0      # offset steps (signed) or target |delta|
    expiry_rank: int = 0    # 0 = entry expiry, 1 = next month (calendar)


@dataclass
class Rules:
    dte_min: int = 20
    dte_max: int = 40
    iv_pct_min: float = 0.0   # ATM IV percentile gate [0,1]
    iv_pct_max: float = 1.0
    profit_target: float | None = 0.5   # fraction of |net entry cost|
    stop_loss: float | None = 1.0
    exit_dte: int = 5
    exclude_adjusted: bool = True


def template_legs(name: str, **p) -> list[LegSpec]:
    w = int(p.get("width", 2))          # wing width in strike steps
    sd = float(p.get("short_delta", 0.25))
    ld = float(p.get("long_delta", 0.10))
    if name == "short_straddle":
        return [LegSpec("C", -1), LegSpec("P", -1)]
    if name == "long_straddle":
        return [LegSpec("C", +1), LegSpec("P", +1)]
    if name == "short_strangle":
        return [LegSpec("C", -1, select="delta", value=sd),
                LegSpec("P", -1, select="delta", value=sd)]
    if name == "long_strangle":
        return [LegSpec("C", +1, select="delta", value=sd),
                LegSpec("P", +1, select="delta", value=sd)]
    if name == "iron_condor":
        return [LegSpec("C", -1, select="delta", value=sd),
                LegSpec("P", -1, select="delta", value=sd),
                LegSpec("C", +1, select="delta", value=ld),
                LegSpec("P", +1, select="delta", value=ld)]
    if name == "call_butterfly":
        return [LegSpec("C", +1, value=-w), LegSpec("C", -1, ratio=2, value=0),
                LegSpec("C", +1, value=+w)]
    if name == "put_butterfly":
        return [LegSpec("P", +1, value=-w), LegSpec("P", -1, ratio=2, value=0),
                LegSpec("P", +1, value=+w)]
    if name == "bull_call_spread":
        return [LegSpec("C", +1, value=0), LegSpec("C", -1, value=+w)]
    if name == "bear_put_spread":
        return [LegSpec("P", +1, value=0), LegSpec("P", -1, value=-w)]
    if name == "call_calendar":
        return [LegSpec("C", -1, value=0, expiry_rank=0),
                LegSpec("C", +1, value=0, expiry_rank=1)]
    raise ValueError(f"unknown template {name}")


TEMPLATE_NAMES = ["short_straddle", "long_straddle", "short_strangle",
                  "long_strangle", "iron_condor", "call_butterfly",
                  "put_butterfly", "bull_call_spread", "bear_put_spread",
                  "call_calendar"]


# ------------------------------------------------------------------ engine
def _prep(df: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    d = df.copy()
    if rules.exclude_adjusted:
        d = d[~d[ds.C_NAME].astype(str).str.endswith("A")]
    d = d[(d[ds.C_CLOSE] > 0)]
    return d


def _atm_iv_series(df: pd.DataFrame) -> pd.Series:
    """Daily near-month ATM call IV — used for the entry percentile gate."""
    d = df[(df[ds.C_CP] == "C") & (df[ds.C_IV] > 0)]
    d = d.loc[d.groupby(ds.C_DATE)[ds.C_DTE].transform(
        lambda s: s == s[s >= 7].min() if (s >= 7).any() else s == s.min())]
    idx = (d[ds.C_STRIKE] - d[ds.C_UND_CLOSE]).abs().groupby(
        d[ds.C_DATE]).idxmin()
    s = d.loc[idx].set_index(ds.C_DATE)[ds.C_IV].sort_index()
    return s


def _pick_leg(day_chain: pd.DataFrame, spec: LegSpec, expiries: list,
              spot: float) -> pd.Series | None:
    if spec.expiry_rank >= len(expiries):
        return None
    exp = expiries[spec.expiry_rank]
    c = day_chain[(day_chain[ds.C_EXPIRY] == exp) &
                  (day_chain[ds.C_CP] == spec.cp)].sort_values(ds.C_STRIKE)
    if c.empty:
        return None
    if spec.select == "delta":
        tgt = abs(spec.value)
        c = c[c[ds.C_DELTA].notna()]
        if c.empty:
            return None
        i = (c[ds.C_DELTA].abs() - tgt).abs().idxmin()
        return c.loc[i]
    # offset from ATM in strike steps
    strikes = c[ds.C_STRIKE].to_numpy()
    atm_i = int(np.argmin(np.abs(strikes - spot)))
    j = atm_i + int(spec.value)
    if j < 0 or j >= len(strikes):
        return None
    return c.iloc[j]


def run_backtest(df: pd.DataFrame, legs: list[LegSpec], rules: Rules
                 ) -> dict:
    d = _prep(df, rules)
    iv_atm = _atm_iv_series(d)
    iv_pct = iv_atm.rolling(252, min_periods=60).rank(pct=True)

    dates = sorted(d[ds.C_DATE].unique())
    by_date = dict(tuple(d.groupby(ds.C_DATE)))

    trades: list[dict] = []
    equity_rows: list[dict] = []
    open_pos: dict | None = None
    realized = 0.0

    for dt in dates:
        day = by_date[dt]
        spot = float(day[ds.C_UND_CLOSE].iloc[0])

        # ---- mark / exit
        if open_pos is not None:
            mark, all_alive = 0.0, True
            for leg in open_pos["legs"]:
                row = day[day[ds.C_CODE] == leg["code"]]
                if row.empty:
                    px = leg["last_px"]
                    if pd.Timestamp(dt) >= open_pos["expiry"]:
                        px = _intrinsic(leg, spot)
                        all_alive = False
                else:
                    px = float(row[ds.C_CLOSE].iloc[0])
                leg["last_px"] = px
                mark += leg["direction"] * leg["ratio"] * px
            pnl = mark - open_pos["cost"]
            basis = max(abs(open_pos["cost"]), 1e-6)
            dte_now = (open_pos["expiry"] - pd.Timestamp(dt)).days
            reason = None
            if rules.profit_target is not None and pnl >= rules.profit_target * basis:
                reason = "profit_target"
            elif rules.stop_loss is not None and pnl <= -rules.stop_loss * basis:
                reason = "stop_loss"
            elif dte_now <= rules.exit_dte:
                reason = "dte_exit"
            elif not all_alive or pd.Timestamp(dt) >= open_pos["expiry"]:
                reason = "expiry"
            if reason:
                realized += pnl
                trades.append({**open_pos["info"], "exit_date": dt,
                               "exit_reason": reason, "pnl": pnl,
                               "exit_spot": spot,
                               "days_held": (pd.Timestamp(dt) -
                                             open_pos["info"]["entry_date"]).days})
                open_pos = None
                equity_rows.append({"date": dt, "equity": realized,
                                    "open_pnl": 0.0, "spot": spot})
                continue
            equity_rows.append({"date": dt, "equity": realized + pnl,
                                "open_pnl": pnl, "spot": spot})
            continue

        # ---- entry
        expiries = sorted(day[ds.C_EXPIRY].unique())
        target = [e for e in expiries
                  if rules.dte_min <= (e - pd.Timestamp(dt)).days <= rules.dte_max]
        ok_iv = True
        if (rules.iv_pct_min > 0 or rules.iv_pct_max < 1):
            p = iv_pct.get(dt, np.nan)
            ok_iv = (p == p) and rules.iv_pct_min <= p <= rules.iv_pct_max
        if target and ok_iv:
            exp0 = target[0]
            ranked = [e for e in expiries if e >= exp0]
            picked = []
            for spec in legs:
                row = _pick_leg(day, spec, ranked, spot)
                if row is None or row[ds.C_CLOSE] <= 0:
                    picked = []
                    break
                picked.append({"code": row[ds.C_CODE], "name": row[ds.C_NAME],
                               "cp": spec.cp, "strike": float(row[ds.C_STRIKE]),
                               "direction": spec.direction, "ratio": spec.ratio,
                               "entry_px": float(row[ds.C_CLOSE]),
                               "last_px": float(row[ds.C_CLOSE]),
                               "expiry": row[ds.C_EXPIRY]})
            if picked and len({p["code"] for p in picked}) == len(picked):
                cost = sum(p["direction"] * p["ratio"] * p["entry_px"]
                           for p in picked)
                open_pos = {
                    "legs": picked, "cost": cost, "expiry": min(
                        p["expiry"] for p in picked),
                    "info": {"entry_date": pd.Timestamp(dt), "entry_spot": spot,
                             "cost": cost,
                             "legs_desc": " | ".join(
                                 f"{'+' if p['direction']>0 else '-'}"
                                 f"{p['ratio']} {p['cp']}{p['strike']:g} "
                                 f"{p['expiry']:%y%m}" for p in picked)},
                }
        equity_rows.append({"date": dt, "equity": realized, "open_pnl": 0.0,
                            "spot": spot})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return {"trades": trades_df, "equity": equity_df,
            "stats": _stats(trades_df, equity_df)}


def _intrinsic(leg: dict, spot: float) -> float:
    if leg["cp"] == "C":
        return max(spot - leg["strike"], 0.0)
    return max(leg["strike"] - spot, 0.0)


def _stats(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n_trades": 0}
    pnl = trades["pnl"]
    eq = equity.set_index("date")["equity"]
    dd = (eq - eq.cummax())
    daily = eq.diff().dropna()
    sharpe = (daily.mean() / daily.std() * np.sqrt(244)
              if daily.std() > 0 else np.nan)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "n_trades": len(trades),
        "total_pnl_yuan": round(pnl.sum() * ds.MULTIPLIER, 0),
        "win_rate": round((pnl > 0).mean(), 3),
        "avg_win_yuan": round(wins.mean() * ds.MULTIPLIER, 0) if len(wins) else 0,
        "avg_loss_yuan": round(losses.mean() * ds.MULTIPLIER, 0) if len(losses) else 0,
        "profit_factor": round(wins.sum() / abs(losses.sum()), 2)
        if losses.sum() != 0 else np.inf,
        "max_drawdown_yuan": round(dd.min() * ds.MULTIPLIER, 0),
        "sharpe": round(sharpe, 2) if sharpe == sharpe else None,
        "avg_days_held": round(trades["days_held"].mean(), 1),
    }
