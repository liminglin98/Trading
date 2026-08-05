"""How long does a short-straddle vol-selling trigger (vrp/iv_pct, alone or in
the two combo signals) usually stay above its entry threshold once it
crosses, and does that persistence duration relate to trade gain/loss?

Never computed anywhere in this codebase before -- reuses the already-frozen
signal series and run() functions from the four backtest modules below
rather than reimplementing any formula or entry logic. Read-only: writes only
new CSVs to out/, no changes to any existing module.

Covers the frozen picks from project_short_straddle_vol_signals memory:
vrp-only and iv_pct-only (both fully validated) on 510050/510300, plus the
iv_pct+vrp AND-combo (510050 only -- 300ETF never froze, too thin) and the
IV-spike+VRP combo (510050/510300) -- both preliminary. Delta-strangle is
excluded: it's a strike-selection variant on the same vrp entry signal, not a
distinct signal, so it would just duplicate the vrp episode data.

Usage:
    python signal_persistence_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # archived: options_lab/ modules live one level up

import data_store as ds
import vol_rich_short_straddle_backtest as vr
import iv_pct_vrp_combo_short_straddle_backtest as combo_and
import iv_spike_vrp_combo_short_straddle_backtest as combo_spike
import iv_low_long_straddle_backtest as il
import iv_spike_short_straddle_backtest as spk

OUT_DIR = Path(__file__).resolve().parent / "out"


# ------------------------------------------------------------------ dates
def _chain_dates(underlying: str) -> list[pd.Timestamp]:
    """Same trading-day list every run() below iterates over (matches
    vr._prepped_chain's filter for 510050/510300)."""
    chain = ds.load_chain(underlying)
    chain = chain[~chain[ds.C_NAME].astype(str).str.endswith("A")]
    chain = chain[chain[ds.C_CLOSE] > 0]
    return sorted(pd.Timestamp(d) for d in chain[ds.C_DATE].unique())


def _prev_day(d, dates, date_idx):
    i = date_idx.get(pd.Timestamp(d))
    if i is None or i == 0:
        return None
    return dates[i - 1]


def _same_day(d, dates, date_idx):
    return pd.Timestamp(d)


# ------------------------------------------------------------------ episodes
def _episodes(dates: list[pd.Timestamp], active: pd.Series) -> pd.DataFrame:
    """Contiguous runs of `active`==True over the trading-day index `dates`."""
    vals = active.reindex(dates).fillna(False).to_numpy(dtype=bool)
    rows = []
    i, n = 0, len(vals)
    while i < n:
        if vals[i]:
            j = i
            while j + 1 < n and vals[j + 1]:
                j += 1
            rows.append({"start_idx": i, "end_idx": j, "start_date": dates[i],
                        "end_date": dates[j], "length_days": j - i + 1})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(rows)


def _episode_for(episodes: pd.DataFrame, t_idx: int):
    m = episodes[(episodes["start_idx"] <= t_idx) & (episodes["end_idx"] >= t_idx)]
    return None if m.empty else m.iloc[0]


def _attach_episode_metrics(trades: pd.DataFrame, date_idx: dict, episodes: pd.DataFrame) -> pd.DataFrame:
    total_len, remaining, elapsed = [], [], []
    for t_date in trades["trigger_date"]:
        t_idx = date_idx.get(pd.Timestamp(t_date)) if t_date is not None else None
        ep = _episode_for(episodes, t_idx) if t_idx is not None else None
        if ep is None:
            total_len.append(np.nan); remaining.append(np.nan); elapsed.append(np.nan)
        else:
            total_len.append(ep["length_days"])
            remaining.append(ep["end_idx"] - t_idx)
            elapsed.append(t_idx - ep["start_idx"])
    return trades.assign(episode_total_len=total_len,
                         episode_days_remaining_after_entry=remaining,
                         episode_days_elapsed_before_entry=elapsed)


# ------------------------------------------------------------------ reporting
def _print_episode_stats(label: str, episodes: pd.DataFrame) -> None:
    if episodes.empty:
        print(f"  {label}: no episodes found")
        return
    L = episodes["length_days"]
    print(f"  regime persistence: n_episodes={len(episodes)}  mean={L.mean():.1f}d  "
          f"median={L.median():.0f}d  p25={L.quantile(.25):.0f}d  p75={L.quantile(.75):.0f}d  "
          f"max={L.max():.0f}d")


def _print_duration_return_relation(trades: pd.DataFrame) -> None:
    t = trades.dropna(subset=["episode_total_len", "mean_r"])
    if len(t) < 5:
        print(f"    too few trades ({len(t)}) for a duration/return relation")
        return
    for col in ["episode_total_len", "episode_days_remaining_after_entry",
               "episode_days_elapsed_before_entry"]:
        corr = t[col].corr(t["mean_r"])
        print(f"    corr(mean_r, {col}) = {corr:+.2f}  (n={len(t)})")
    try:
        t = t.assign(bucket=pd.qcut(t["episode_total_len"], 3,
                                    labels=["short", "medium", "long"], duplicates="drop"))
    except ValueError:
        print("    (not enough distinct episode lengths for tercile buckets)")
        return
    for name, g in t.groupby("bucket", observed=True):
        flag = "" if len(g) >= 10 else "  [thin, n<10 -- don't trust]"
        print(f"    {name}: n={len(g)}  mean_r={g['mean_r'].mean():+.2%}  "
              f"win_rate={(g['pnl'] > 0).mean():.0%}{flag}")


# ------------------------------------------------------------------ configs
CONFIGS = [
    dict(label="vrp_510050", underlying="510050", validated=True, trigger_fn=_prev_day,
        active_fn=lambda dates: vr._signal_series("510050", "vrp") >= 0.06,
        run_fn=lambda: vr.run("510050", "vrp", 0.06, 20)),
    dict(label="vrp_510300", underlying="510300", validated=True, trigger_fn=_prev_day,
        active_fn=lambda dates: vr._signal_series("510300", "vrp") >= 0.03,
        run_fn=lambda: vr.run("510300", "vrp", 0.03, 20)),
    dict(label="iv_pct_510050", underlying="510050", validated=True, trigger_fn=_prev_day,
        active_fn=lambda dates: vr._signal_series("510050", "iv_pct") >= 0.90,
        run_fn=lambda: vr.run("510050", "iv_pct", 0.90, 20)),
    dict(label="iv_pct_510300", underlying="510300", validated=True, trigger_fn=_prev_day,
        active_fn=lambda dates: vr._signal_series("510300", "iv_pct") >= 0.80,
        run_fn=lambda: vr.run("510300", "iv_pct", 0.80, 20)),
    dict(label="and_combo_510050", underlying="510050", validated=False, trigger_fn=_prev_day,
        active_fn=lambda dates: (il.iv_pct_series("510050") >= 0.90) & (il.vrp_trailing_series("510050") >= 0.03),
        run_fn=lambda: combo_and.run("510050", iv_pct_threshold=0.90, vrp_threshold=0.03, max_hold_days=20)),
    dict(label="spike_combo_510050", underlying="510050", validated=False, trigger_fn=_same_day,
        active_fn=lambda dates: spk.spike_flags(spk.iv_atm_series("510050"), 10, 0.15) & (il.vrp_trailing_series("510050") >= 0.03),
        run_fn=lambda: combo_spike.run("510050", spike_window=10, spike_pct=0.15, vrp_threshold=0.03, max_hold_days=20)),
    dict(label="spike_combo_510300", underlying="510300", validated=False, trigger_fn=_same_day,
        active_fn=lambda dates: spk.spike_flags(spk.iv_atm_series("510300"), 10, 0.15) & (il.vrp_trailing_series("510300") >= 0.03),
        run_fn=lambda: combo_spike.run("510300", spike_window=10, spike_pct=0.15, vrp_threshold=0.03, max_hold_days=20)),
]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    pooled = []

    for cfg in CONFIGS:
        print(f"\n=== {cfg['label']} ===")
        dates = _chain_dates(cfg["underlying"])
        date_idx = {d: i for i, d in enumerate(dates)}

        active = cfg["active_fn"](dates)
        episodes = _episodes(dates, active)
        _print_episode_stats(cfg["label"], episodes)

        trades = cfg["run_fn"]()["trades"]
        if trades.empty:
            print("  no trades")
            continue
        trades = trades.copy()
        trades["trigger_date"] = trades["entry_date"].apply(
            lambda d: cfg["trigger_fn"](d, dates, date_idx))
        trades = _attach_episode_metrics(trades, date_idx, episodes)
        trades["mean_r"] = trades["pnl"] * ds.MULTIPLIER / trades["margin_yuan"]

        print(f"  n_trades={len(trades)}")
        _print_duration_return_relation(trades)

        out_path = OUT_DIR / f"signal_persistence_{cfg['label']}.csv"
        trades.to_csv(out_path, index=False)
        print(f"  saved: {out_path.name}")

        if cfg["validated"]:
            pooled.append(trades.assign(config=cfg["label"]))

    if pooled:
        print("\n=== POOLED (vrp-only + iv_pct-only, both validated signals) ===")
        pooled_df = pd.concat(pooled, ignore_index=True)
        print(f"  n_trades={len(pooled_df)}")
        _print_duration_return_relation(pooled_df)


if __name__ == "__main__":
    main()
