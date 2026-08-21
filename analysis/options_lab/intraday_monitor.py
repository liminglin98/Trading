"""Intraday follow-up to minute_updater.py: turns the pre-market "close to
triggering" watchlist (alerts_core.near_trigger_detail, the same logic
push_alerts.py uses for the 8:30am push) into a live check against
minute_updater's polled option quotes, and pushes via PushPlus the moment a
signal actually crosses its trigger level during the trading day.

Design (see the plan this was built from for the full reasoning): realized
vol barely moves within a session, so every near-trigger signal type EXCEPT
the pure rv_pct-gated ones is treated as "live ATM IV vs. a raw IV level
fixed this morning" -- RV itself is held at its EOD value all day, and the
raw trigger level for each signal (already computed daily by
alerts_core.vol_trigger_level/vrp_trigger_level) is resolved once at
startup, not recomputed per cycle. Coverage of the IV-based path is also
limited to underlyings minute_updater.py actually polls (ETF_UNDERLYINGS);
near-trigger rows for other underlyings (e.g. the SZSE-only long-straddle
alerts) are skipped with a logged reason.

Pure rv_pct-gated signals (RV *is* the metric, not IV -- the three
VOL_SIGNAL_ALERTS rows for 510500/159915/159901) get a SEPARATE, once-a-day
check at ~11:30 (morning session close) instead: rv_proxy_now() splices a
scaled morning-session return into the same 20-day trailing-RV window
_trailing_rv_series uses, in place of today's not-yet-known full-day return.
This is explicitly a noisier proxy than the real EOD figure (see
RV_PROXY_SCALE/RV_PROXY_MARGIN below for how it was calibrated and why the
push is labeled "exploratory") -- checked once, not every cycle, since
checking earlier in the session would only add noise, and checking more
than once can't add information (the proxy doesn't change until the next
day's price data).

Usage:
    python intraday_monitor.py                  # run the live monitoring loop
    python intraday_monitor.py --dry-run --once  # one check cycle, print only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

import alerts_core as ac
import dashboard_data as DD
import data_store as ds
import minute_updater as MU
import pushplus
import updater as U

HERE = Path(__file__).resolve().parent
STATE_DIR = ds.DATA_DIR / "intraday"
LOG_PATH = HERE / "intraday_monitor.log"
DEFAULT_INTERVAL = 300  # 5 minutes -- coarser than minute_updater's own 60s
                        # poll (options signals don't need second-level
                        # precision), so each cycle just re-reads whatever
                        # minute_updater has already appended by then.

RV_CHECKPOINT_TIME = dtime(11, 30)

# Live Sina spot symbol for each rv_pct-gated underlying (159915/159901 are
# SZSE, not polled by minute_updater at all -- confirmed live 2026-08-11 via
# updater._hq_batch that Sina serves these fine under the "sz" prefix, same
# pattern minute_updater.index_spot already uses for CFFEX indices).
RV_SPOT_SINA_SYM = {"510500": "sh510500", "159915": "sz159915", "159901": "sz159901"}

# Empirically calibrated 2026-08-11 against ~42 trading days of Sina 5-min
# bars (scratchpad rv_proxy_calibration.py, not committed): the scale that
# makes the morning session's (09:30-11:30) RMS return match the actual
# full-day return's RMS, per underlying. The naive theoretical sqrt(2)
# (assuming variance splits 50/50 by clock time between the two sessions)
# overshot for 510500 (1.30x too high) and slightly undershot for
# 159915/159901 (1.07x) -- the intraday vol U-shape means variance doesn't
# actually split evenly by clock time, so this fits per-underlying instead.
RV_PROXY_SCALE = {"510500": 1.086, "159915": 1.323, "159901": 1.316}

# Same calibration run's end-to-end check: splicing the scaled proxy into
# the real 20-day trailing-RV window (all other 19 days real) vs. the
# window using the actual eventual close, across those 42 days: nearly
# unbiased (mean diff -0.0002 to -0.0006) with RMSE 0.009-0.015 and a
# worst-case max diff of ~0.03-0.05 (annualized-vol points). MARGIN below is
# a round ~2x-RMSE cushion so the alert only fires on a clear miss, not a
# borderline one this proxy's noise could easily produce.
RV_PROXY_MARGIN = 0.03

# ------------------------------------------------------------------ IV basis
# A trigger level is a percentile of some DAILY historical ATM-IV series, so
# the live number compared against it has to be measured the same way that
# series was. Two different series feed the triggers, and they disagree:
#
#   "call_only" -- backtest._atm_iv_series: near-month ATM CALL only, read
#       straight from the exchange's published 隐含波动率 column.
#   "cp_avg"    -- consolidated_store.atm_iv_series: same strike/expiry pick,
#       but BS-inverts (bs_iv.implied_vol, as we do) and averages call+put.
#
# Until 2026-08-17 live_atm_iv() always averaged call+put, so every
# call_only-backed alert was compared against a reading ~1.2-2.2 vol points
# too high (put IV sits above call IV at the same strike). That is one-sided
# harm, not noise: it made every direction="below" (long-vol) alert
# essentially unfireable and every direction="above" (short-vol) alert fire
# early. Each watchlist row now records its basis and is measured on it.
IV_BASIS_CALL_ONLY = "call_only"
IV_BASIS_CP_AVG = "cp_avg"

# Residual method offset once the legs match: subtracted from the live
# reading to put it in the trigger series' units. For call_only that is our
# BS inversion vs. the exchange's (different rate/daycount conventions); for
# cp_avg it is only intraday mid vs. daily close, which is ~0 as expected.
# Calibrated by tools/iv_basis_calibration.py (median over its window; rerun
# it to refresh). 510500/588000 populate the exchange IV column on only a
# minority of sessions, so their call_only figures rest on a much shorter
# sample than 510050/510300's. Underlyings absent here get no correction.
#
# A constant is the right shape of correction: trigger levels are percentiles
# of the historical series, and shifting a whole series by a constant shifts
# its percentiles by that same constant -- so correcting the live reading is
# equivalent to re-deriving the level from a shifted series, and disturbs no
# backtested signal definition.
IV_METHOD_BIAS = {
    "510050": {"call_only": +0.0041, "cp_avg": -0.0002},
    "510300": {"call_only": +0.0042, "cp_avg": -0.0008},
    "510500": {"call_only": +0.0047, "cp_avg": -0.0003},
    "588000": {"call_only": +0.0206, "cp_avg": +0.0006},
}


def _log(msg: str) -> None:
    line = f"[{MU.now_cn().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _cached_chain_loader():
    cache: dict = {}

    def loader(u: str):
        if u not in cache:
            cache[u] = ds.load_chain(u)
        return cache[u]

    return loader


# ------------------------------------------------------------------ watchlist (pre-market phase)
def _resolve_trigger(row: dict, near: dict) -> tuple[str, float, str] | None:
    """(direction, raw IV-point trigger level, IV basis) for one near-trigger
    row, or None if this row has no live-trackable IV level (an idx
    near_trigger_detail doesn't cover at all -- see its own docstring). Pure
    rv_pct-gated rows are intercepted earlier in build_watchlist(), before
    this is called, and routed to the RV-proxy path instead -- see
    RV_PROXY_SCALE above.
    'direction': "above" fires when live ATM IV >= level (short-vol signals
    richening), "below" fires when live ATM IV <= level (long-vol signals
    cheapening).
    'basis': which daily ATM-IV series the level came from, so live_atm_iv()
    can measure the same thing -- see IV_BASIS_CALL_ONLY above. Everything
    here traces back to backtest._atm_iv_series (call-only) except the
    module=="rv" long-straddle signals, which come from
    consolidated_store.atm_iv_series (call+put average)."""
    side, idx, cfg = row["side"], row["strategy_idx"], row.get("cfg")

    if side == "buy" and idx == 1:
        if "trigger_raw" not in near:
            return None
        # alerts_core.vol_trigger_level: module "rv" (non-rv_pct signal) reads
        # cs.atm_iv_series; anything else reads bt._atm_iv_series.
        basis = IV_BASIS_CP_AVG if (cfg or {}).get("module") == "rv" else IV_BASIS_CALL_ONLY
        return "below", float(near["trigger_raw"]), basis

    if side == "sell" and idx in (0, 1, 2):
        if "trigger_raw" not in near:
            return None
        # Both branches (vrp_trigger_level, vol_trigger_level(.., "iv_low", ..))
        # bottom out in bt._atm_iv_series.
        return "above", float(near["trigger_raw"]), IV_BASIS_CALL_ONLY

    if side == "sell" and idx == 3:
        # near_trigger_detail only surfaces this row once ok_spike is the
        # sole missing piece (see its idx==3 branch) -- but confirm here
        # too, since ok_vrp is anchored on YESTERDAY's data and can't change
        # today: if it wasn't already satisfied this morning, this alert
        # cannot fully trigger today no matter how live IV moves.
        if not near.get("ok_vrp") or near.get("spike_needed") is None:
            return None
        # spike_needed is scaled off spk.iv_atm_series -> bt._atm_iv_series.
        return "above", float(near["spike_needed"]), IV_BASIS_CALL_ONLY

    if side == "sell" and idx == 4:
        iv_pct_trig = ac.vol_trigger_level(cfg["u"], "iv_low", "iv_pct", cfg["iv_pct_threshold"])
        vrp_trig = ac.vrp_trigger_level(cfg["u"], cfg["vrp_threshold"])
        if not iv_pct_trig or not vrp_trig:
            return None
        # Both sub-conditions are ">=" checks against the SAME live ATM IV
        # number, so the AND-combo collapses to one ">= max(...)" check.
        # Both legs are bt._atm_iv_series-derived, so one shared basis.
        return "above", max(float(iv_pct_trig["level"]), float(vrp_trig["needed"])), IV_BASIS_CALL_ONLY

    return None  # idx 5/6 (gamma flip, delta-strangle iv_pct): not covered by near_trigger_detail either


def build_watchlist(chain_loader=None) -> list[dict]:
    """Every currently-inactive, live-trackable alert row (not just ones
    already near their threshold at the open -- see
    alerts_core.full_trigger_detail), resolved down to the specific ATM
    contracts to poll live and the raw IV-point level that would flip each
    one active -- ready for check_triggers() to compare against
    minute_updater's feed all day."""
    if chain_loader is None:
        chain_loader = _cached_chain_loader()
    idx_sh = DD.load_series("idx_sh")
    rows = ac.evaluate_summary(idx_sh, chain_loader=chain_loader)

    watchlist = []
    for row in rows:
        if row["active"]:
            continue
        positions = ac.position_detail(row, chain_loader=chain_loader, idx_sh_series=idx_sh)
        if len(positions or []) >= ac.MAX_CONCURRENT_POSITIONS:
            continue
        near = ac.full_trigger_detail(row)
        if not near:
            continue

        cfg = row.get("cfg")
        u = cfg["u"] if cfg else None
        if u is None:
            continue

        if row["side"] == "buy" and row["strategy_idx"] == 1 and cfg.get("signal") == "rv_pct":
            # RV itself is the metric here, not IV -- routed to the separate
            # once-a-day 11:30 proxy check instead of the continuous
            # IV-comparison path below (see module docstring / RV_PROXY_SCALE).
            if u not in RV_PROXY_SCALE:
                _log(f"skip {row['strategy']} ({u}): rv_pct underlying not covered by the RV proxy calibration")
                continue
            if "trigger_raw" not in near:
                continue
            watchlist.append(dict(
                label=ac.ALERT_LABELS_ZH.get((row["side"], row["strategy_idx"]), row["strategy"]),
                instrument=row["instrument"], u=u, side=row["side"], strategy_idx=row["strategy_idx"],
                metric="rv_proxy", direction="below", trigger_level=float(near["trigger_raw"]),
                baseline_raw=float(near.get("current_raw", float("nan"))),
            ))
            continue

        if u not in MU.ETF_UNDERLYINGS:
            _log(f"skip {row['strategy']} ({u}): not in minute_updater's polled universe")
            continue

        trig = _resolve_trigger(row, near)
        if trig is None:
            continue
        direction, level, basis = trig

        atm = (ac.suggest_atm_straddle(chain_loader(u)) if row["side"] == "buy"
               else ac.suggest_atm_short_straddle(chain_loader(u)))
        if atm is None or "call_code" not in atm:
            _log(f"skip {row['strategy']} ({u}): no current ATM contract to track")
            continue

        watchlist.append(dict(
            label=ac.ALERT_LABELS_ZH.get((row["side"], row["strategy_idx"]), row["strategy"]),
            instrument=row["instrument"], u=u, side=row["side"], strategy_idx=row["strategy_idx"],
            metric="iv", call_code=atm["call_code"], put_code=atm["put_code"],
            direction=direction, trigger_level=level, iv_basis=basis,
            method_bias=IV_METHOD_BIAS.get(u, {}).get(basis, 0.0),
            baseline_raw=float(near.get("current_raw", near.get("iv_now", near.get("iv_pct_now", float("nan"))))),
        ))
    return watchlist


def _watchlist_path(when: date | None = None) -> Path:
    when = when or MU.now_cn().date()
    return STATE_DIR / f"watchlist_{when.strftime('%Y%m%d')}.json"


def _fired_path(when: date | None = None) -> Path:
    when = when or MU.now_cn().date()
    return STATE_DIR / f"fired_{when.strftime('%Y%m%d')}.json"


def _item_key(item: dict) -> str:
    return f"{item['side']}_{item['strategy_idx']}_{item['u']}"


def save_watchlist(watchlist: list[dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _watchlist_path().write_text(json.dumps(watchlist, ensure_ascii=False, indent=2), encoding="utf-8")


def load_fired() -> set[str]:
    path = _fired_path()
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def save_fired(fired: set[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _fired_path().write_text(json.dumps(sorted(fired), ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ live check (intraday phase)
def _load_minute_df(u: str, cache: dict) -> pd.DataFrame | None:
    if u not in cache:
        path = MU.minute_csv_path(u, is_index=False)
        cache[u] = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig") if path.exists() else None
    return cache[u]


def live_atm_iv(u: str, call_code: str, put_code: str, minute_df_cache: dict | None = None,
                basis: str = IV_BASIS_CALL_ONLY, method_bias: float = 0.0) -> float | None:
    """Live ATM IV for one watchlist item, expressed in the units of the
    daily series its trigger level came from: latest bid/ask-mid quote for
    the tracked contracts, BS-inverted via minute_updater.iv_for_row (same
    on-demand pattern minute_updater.py was built around).

    `basis` selects which legs to read -- IV_BASIS_CP_AVG averages call and
    put, IV_BASIS_CALL_ONLY reads the call alone -- and MUST match the basis
    of the trigger being compared against (see IV_BASIS_CALL_ONLY above;
    mismatching them is the 2026-08-17 bug). `method_bias` is then subtracted
    to absorb the residual inversion/price-source offset.

    None if a needed contract hasn't been quoted yet today. `minute_df_cache`
    lets a single check_triggers() cycle share one read of each underlying's
    CSV across watchlist items instead of re-reading it per item."""
    df = _load_minute_df(u, minute_df_cache if minute_df_cache is not None else {})
    if df is None:
        return None
    codes = (call_code,) if basis == IV_BASIS_CALL_ONLY else (call_code, put_code)
    ivs = []
    for code in codes:
        rows = df[df["code"] == code]
        if rows.empty:
            return None
        iv = MU.iv_for_row(rows.iloc[-1])
        if pd.isna(iv):
            return None
        ivs.append(iv)
    return sum(ivs) / len(ivs) - method_bias


def check_triggers(watchlist: list[dict], fired: set[str]) -> list[dict]:
    """Newly-crossed watchlist items this cycle (already-fired ones are
    skipped -- once-per-alert-per-day dedupe)."""
    triggered = []
    minute_df_cache: dict = {}
    for item in watchlist:
        if item.get("metric") != "iv":
            continue
        key = _item_key(item)
        if key in fired:
            continue
        # Older watchlist files (pre-2026-08-17) carry no basis; they were all
        # measured call+put-averaged, so fall back to that rather than
        # silently re-interpreting them as call-only.
        iv = live_atm_iv(item["u"], item["call_code"], item["put_code"], minute_df_cache,
                         basis=item.get("iv_basis", IV_BASIS_CP_AVG),
                         method_bias=float(item.get("method_bias", 0.0)))
        if iv is None:
            continue
        hit = (iv >= item["trigger_level"] if item["direction"] == "above"
               else iv <= item["trigger_level"])
        if hit:
            triggered.append({**item, "live_iv": iv})
    return triggered


def push_trigger(item: dict) -> None:
    arrow = "≥" if item["direction"] == "above" else "≤"
    content = (
        f"🔔 {item['label']} · {item['instrument']}（已触发，日内信号）\n"
        f"当前ATM IV：{item['live_iv']:.2%} {arrow} 触发阈值：{item['trigger_level']:.2%}\n"
        f"触发时间：{MU.now_cn().strftime('%H:%M:%S')}"
    )
    title = f"期权信号日内触发 {item['label']} {MU.now_cn().date().isoformat()}"
    pushplus.send(title, content)


# ------------------------------------------------------------------ RV proxy (11:30 checkpoint)
def rv_proxy_now(u: str) -> float | None:
    """Intraday proxy for _trailing_rv_series's most recent value: splices a
    scaled morning-session (09:30-11:30) return into the same 20-day
    trailing window of daily log returns, in place of today's not-yet-known
    full-day return, then std()*sqrt(252) exactly as the real function does.
    None if the daily OHLC cache or the live spot quote isn't available."""
    daily = ds.load_etf_ohlc(u)
    if daily is None or len(daily) < 20:
        return None
    close = daily.set_index(daily["date"].dt.date)["close"].sort_index()
    logret = np.log(close).diff().dropna()
    if len(logret) < 19:
        return None
    prev_close = float(close.iloc[-1])

    sina_sym = RV_SPOT_SINA_SYM.get(u)
    if sina_sym is None:
        return None
    got = U._hq_batch([sina_sym])
    v = got.get(sina_sym)
    if not v or len(v) < 4:
        return None
    try:
        spot = float(v[3])
    except (TypeError, ValueError):
        return None
    if not (spot > 0 and prev_close > 0):
        return None

    r_morning = np.log(spot / prev_close)
    r_proxy = r_morning * RV_PROXY_SCALE[u]

    window = pd.concat([logret.iloc[-19:], pd.Series([r_proxy])], ignore_index=True)
    return float(window.std() * np.sqrt(252))


def check_rv_proxy_triggers(watchlist: list[dict], fired: set[str]) -> list[dict]:
    """Same once-per-alert-per-day dedupe as check_triggers, but for the
    metric="rv_proxy" rows and gated by RV_PROXY_MARGIN -- fires only on a
    clear miss past the trigger level, not a borderline touch, since this
    proxy is strictly noisier than a real close (see RV_PROXY_MARGIN)."""
    triggered = []
    for item in watchlist:
        if item.get("metric") != "rv_proxy":
            continue
        key = _item_key(item)
        if key in fired:
            continue
        rv = rv_proxy_now(item["u"])
        if rv is None:
            continue
        if rv <= item["trigger_level"] - RV_PROXY_MARGIN:
            triggered.append({**item, "live_rv": rv})
    return triggered


def push_rv_proxy_trigger(item: dict) -> None:
    content = (
        f"⚠️ 探索性/近似（日内RV代理，非完整收盘数据）\n"
        f"{item['label']} · {item['instrument']}\n"
        f"代理RV（09:30-11:30外推）：{item['live_rv']:.2%} ≤ 触发阈值：{item['trigger_level']:.2%}"
        f"（已扣除{RV_PROXY_MARGIN:.0%}安全边际）\n"
        f"该读数基于上午盘中收益率外推，非完整收盘数据，请结合实际情况判断；"
        f"收盘后的次日早盘推送会给出准确读数。\n"
        f"检查时间：{MU.now_cn().strftime('%H:%M:%S')}"
    )
    title = f"期权信号日内触发(探索性RV代理) {item['label']} {MU.now_cn().date().isoformat()}"
    pushplus.send(title, content)


# ------------------------------------------------------------------ main loop
def run(interval: int = DEFAULT_INTERVAL, dry_run: bool = False, once: bool = False) -> None:
    today = MU.now_cn().date()
    if not once and not MU.is_trading_day(today):
        _log(f"{today} is not a trading day -- exiting.")
        return

    _log("Building pre-market near-trigger watchlist...")
    watchlist = build_watchlist()
    if not dry_run:
        save_watchlist(watchlist)
    if watchlist:
        names = ", ".join(f"{w['label']}({w['u']}, {w.get('iv_basis', w['metric'])})" for w in watchlist)
        _log(f"Watchlist: {len(watchlist)} trackable near-trigger signal(s): {names}")
    else:
        _log("Watchlist: empty")

    fired = load_fired() if not dry_run else set()
    rv_checked = False

    while True:
        now = MU.now_cn()
        if now.date() != today:
            _log("date rolled over -- exiting.")
            break
        if not once and MU.past_close(now):
            _log("past close -- exiting.")
            break
        if not once and not MU.in_session(now):
            time.sleep(5)
            continue

        newly_triggered = check_triggers(watchlist, fired)
        for item in newly_triggered:
            _log(f"TRIGGERED: {item['label']} ({item['u']}) live_iv={item['live_iv']:.2%} "
                 f"vs level={item['trigger_level']:.2%} [basis={item.get('iv_basis', '?')}]"
                 + (" (dry-run, not pushed)" if dry_run else ""))
            if not dry_run:
                try:
                    push_trigger(item)
                except Exception as e:
                    _log(f"FAILED pushing {item['label']}: {type(e).__name__}: {e}")
                    continue
                fired.add(_item_key(item))
                save_fired(fired)

        # Once-a-day RV-proxy checkpoint, right after the morning session
        # closes -- see module docstring for why this isn't on the regular
        # 5-min cadence.
        if not rv_checked and now.time() >= RV_CHECKPOINT_TIME:
            rv_checked = True
            rv_triggered = check_rv_proxy_triggers(watchlist, fired)
            for item in rv_triggered:
                _log(f"RV-PROXY TRIGGERED: {item['label']} ({item['u']}) live_rv={item['live_rv']:.2%} "
                     f"vs level={item['trigger_level']:.2%}" + (" (dry-run, not pushed)" if dry_run else ""))
                if not dry_run:
                    try:
                        push_rv_proxy_trigger(item)
                    except Exception as e:
                        _log(f"FAILED pushing RV-proxy {item['label']}: {type(e).__name__}: {e}")
                        continue
                    fired.add(_item_key(item))
                    save_fired(fired)

        if once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--once", action="store_true", help="one check cycle and exit (testing)")
    args = ap.parse_args()
    try:
        run(interval=args.interval, dry_run=args.dry_run, once=args.once)
    except KeyboardInterrupt:
        print("interrupted, exiting.")
        sys.exit(0)
