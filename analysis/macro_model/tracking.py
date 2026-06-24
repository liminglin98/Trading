# -*- coding: utf-8 -*-
"""
tracking.py
===========
Shared (no-Streamlit) store + valuation for saved butterfly/option combinations
and their FORWARD live-test history. Imported by app.py (the dashboard) AND by
pipeline.py (--snapshot, for an end-of-day cron). Keeping it here means the app
and the scheduled job mark positions with EXACTLY the same code.

Data sources (both end-of-day):
  * option marks  : akshare option_risk_indicator_sse (official SSE EOD IV per
                    contract) -> BS value per leg (via butterfly_timing_backtest).
  * underlying    : sina fund_etf_hist_sina (EOD close).
So an EOD snapshot needs no extra data plumbing; it reuses the warm IV cache.

Files (next to this module):
  saved_positions.csv    -- one row per leg + a save-time snapshot + the per-combo
                            stop/roll RULE columns + roll LINEAGE columns.
  position_tracking.csv  -- one row per forward snapshot (the live-test history)
                            now carrying the evaluated status + any alert text.

STOP / ROLL RULES (analytics + suggestions only -- this module NEVER places,
moves, or auto-executes a trade; it detects, flags, logs, and alerts. You execute
in your broker). Both stops are simple fractions on the WHOLE combination's P&L
(net of every leg, buys and sells together -- nothing is evaluated per-leg):
  * take-profit : tp_frac -- take profit when open P&L >= tp_frac * max gain,
                  where max gain is the combination's max payoff at expiry minus
                  its net debit. Blank=off.
  * stop-loss   : sl_frac -- stop out when open P&L <= -sl_frac * max loss, where
                  max loss for a long butterfly is simply the net debit. Blank=off.
  * roll        : a SUGGESTION fires when dte <= roll_dte (per-combo; default 14).
                  A roll re-centres the body at the current spot, keeps the SAME
                  wing steps as the parent, and is priced on the next listed
                  expiry's REAL chain. Hitting TP/SL flags the position to CLOSE;
                  it does NOT auto-generate a roll. Status is sticky once terminal.

Position status (stored on saved_positions, also stamped on each snapshot row):
  open | hit_tp | hit_sl | expired | rolled
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import butterfly_timing_backtest as B
import butterfly_param_search as PS      # wires the disk IV cache + monkeypatch

R_FREE = B.R_FREE
HERE = Path(__file__).resolve().parent
SAVE_PATH = HERE / "saved_positions.csv"
TRACK_PATH = HERE / "position_tracking.csv"

DEFAULT_ROLL_DTE = 14                      # matches the chosen operational exit-DTE
TERMINAL = {"hit_tp", "hit_sl", "expired", "rolled"}

# Original (v1) columns -- never renamed.
_SAVE_BASE = ["saved_at", "name", "position_id", "side", "type", "strike", "qty",
              "entry", "expiry", "spot_at_save", "dte_at_save", "cost_basis",
              "mark_at_save", "pnl_at_save"]
# Added (v2): per-combo stop/roll rules + lineage. Migration-safe defaults below.
# (tp_frac = frac of combination max GAIN; sl_frac = frac of combination max LOSS)
_SAVE_RULES = ["tp_frac", "sl_frac", "roll_dte", "status",
               "trigger_date", "trigger_level", "trigger_reason",
               "roll_chain_id", "roll_parent_id", "roll_child_id"]
SAVE_COLS = _SAVE_BASE + _SAVE_RULES

_TRACK_BASE = ["position_id", "recorded_at", "asof_date", "expiry", "spot",
               "dte", "cost_basis", "mark", "pnl"]
_TRACK_ADD = ["status", "alert"]
TRACK_COLS = _TRACK_BASE + _TRACK_ADD

# Default value for every added column (applied to OLD rows on load).
_SAVE_DEFAULTS = dict(tp_frac=np.nan, sl_frac=np.nan,
                      roll_dte=np.nan, status="open", trigger_date="",
                      trigger_level=np.nan, trigger_reason="",
                      roll_chain_id="", roll_parent_id="", roll_child_id="")
_TRACK_DEFAULTS = dict(status="open", alert="")


# =============================================================================
# Schema migration helper
# =============================================================================
def _ensure_cols(df, cols, defaults):
    """Add any missing columns with their safe default (in place-ish). Preserves
    existing data; never renames. Returns the column-ordered frame."""
    for c in cols:
        if c not in df.columns:
            df[c] = defaults.get(c, np.nan)
    # keep any extra/unknown columns at the end so nothing is dropped
    ordered = [c for c in cols if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    return df[ordered + extra]


def _finite(x):
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# =============================================================================
# Option valuation helpers
# =============================================================================
def leg_value_now(cp, K, spot, T, smile):
    """BS value of one option leg. IV from the smile (interp) if given, else 0.2."""
    iv = B.smile_iv(smile, K) if (smile is not None and len(smile)) else 0.2
    if cp == "C":
        return B.bs_call(spot, K, T, R_FREE, iv), iv
    return B.bs_put(spot, K, T, R_FREE, iv), iv

def legs_from_saved(sub):
    """Reconstruct signed legs from saved rows (Buy->+, Sell->-)."""
    legs = []
    for _, r in sub.iterrows():
        try:
            sign = -1.0 if str(r["side"]).strip().lower().startswith("s") else 1.0
            legs.append(dict(type=str(r["type"]).strip().upper()[0],
                             strike=float(r["strike"]),
                             qty=sign * abs(float(r["qty"])), entry=float(r["entry"])))
        except Exception:
            continue
    return legs

def mark_position(legs, smile, spot, T):
    """Return (mark, cost_basis, pnl) for signed legs at (spot, T) on a smile."""
    mark = cost = 0.0
    for lg in legs:
        v, _iv = leg_value_now(lg["type"], lg["strike"], spot, T, smile)
        mark += lg["qty"] * v
        cost += lg["qty"] * lg["entry"]
    return mark, cost, mark - cost

def position_max_profit(legs):
    """Max profit at EXPIRY (over the underlying) for signed legs, net of cost.
    Self-contained (no model) -- evaluates the piecewise-linear payoff on a dense
    grid spanning the strikes. For a long butterfly this is the body apex."""
    if not legs:
        return np.nan
    ks = sorted({lg["strike"] for lg in legs})
    lo, hi = min(ks) * 0.5, max(ks) * 1.5
    grid = np.unique(np.concatenate([np.asarray(ks, float),
                                     np.linspace(lo, hi, 400)]))
    cost = sum(lg["qty"] * lg["entry"] for lg in legs)
    pnl = np.zeros_like(grid)
    for lg in legs:
        intr = (np.maximum(grid - lg["strike"], 0.0) if lg["type"] == "C"
                else np.maximum(lg["strike"] - grid, 0.0))
        pnl += lg["qty"] * intr
    return float(np.nanmax(pnl - cost))


# =============================================================================
# Stop-gain / stop-loss / roll rule evaluation  (pure, no I/O)
# =============================================================================
def evaluate_rules(pnl, cost, max_profit, dte, rules):
    """Evaluate the per-combo rules at one EOD mark.

    All thresholds are on the WHOLE combination's open P&L (= net mark - net
    cost, every leg together). Two simple fractions:
      take-profit : P&L >= tp_frac * max_profit   (max_profit = combo max gain)
      stop-loss   : P&L <= -sl_frac * debit       (debit = combo max loss)
    Blank (NaN) disables a rule. A long butterfly cannot be both >=TP and <=SL at
    one mark, but TP is checked first as a tie-break for safety.

    Returns dict(status, reason, level, roll, roll_reason):
      status      : 'hit_tp' | 'hit_sl' | 'open'   (expiry handled by caller)
      reason      : human text for why TP/SL fired (or '')
      level       : the P&L at which it fired (rounded), np.nan if none
      roll        : True if a roll SUGGESTION should be raised (dte <= roll_dte)
      roll_reason : human text for the roll alert (or '')"""
    tp_frac = rules.get("tp_frac")
    sl_frac = rules.get("sl_frac")
    roll_dte = rules.get("roll_dte")
    debit = abs(float(cost)) if _finite(cost) else np.nan   # = max loss (long fly)

    status, reason = "open", ""
    if _finite(tp_frac) and _finite(max_profit) and max_profit > 0 \
            and pnl >= float(tp_frac) * max_profit:
        status = "hit_tp"
        reason = f"TP {float(tp_frac):.2f} of max gain (>= {float(tp_frac)*max_profit:+.4f})"
    elif _finite(sl_frac) and _finite(debit) and debit > 0 \
            and pnl <= -float(sl_frac) * debit:
        status = "hit_sl"
        reason = f"SL {float(sl_frac):.2f} of max loss (<= {-float(sl_frac)*debit:+.4f})"

    roll = bool(_finite(roll_dte) and _finite(dte) and dte <= float(roll_dte))
    roll_reason = (f"roll: {int(dte)} DTE <= {int(float(roll_dte))} threshold"
                   if roll else "")
    level = round(float(pnl), 4) if status != "open" else np.nan
    return dict(status=status, reason=reason, level=level,
                roll=roll, roll_reason=roll_reason)


# =============================================================================
# Saved-positions store
# =============================================================================
def load_saved():
    if SAVE_PATH.exists():
        try:
            df = pd.read_csv(SAVE_PATH)
            return _ensure_cols(df, SAVE_COLS, _SAVE_DEFAULTS)
        except Exception:
            pass
    return pd.DataFrame(columns=SAVE_COLS)

def save_position(name, legs_df, snapshot, rules=None,
                  roll_parent_id="", roll_chain_id=""):
    """Save one combination. `rules` (optional) carries tp_frac/sl_frac/roll_dte;
    missing keys default to disabled. roll_chain_id defaults to
    this position's own id (a fresh lineage root); pass the parent's chain id +
    parent id to record a rolled child."""
    existing = load_saved()
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    name = (str(name).strip() or "unnamed")
    pid = f"{name} @ {ts}"
    rules = rules or {}
    chain = str(roll_chain_id).strip() or pid           # root lineage = own id
    rule_fields = dict(
        tp_frac=rules.get("tp_frac", np.nan), sl_frac=rules.get("sl_frac", np.nan),
        roll_dte=rules.get("roll_dte", np.nan), status="open",
        trigger_date="", trigger_level=np.nan, trigger_reason="",
        roll_chain_id=chain, roll_parent_id=str(roll_parent_id), roll_child_id="")
    rows = []
    for _, r in legs_df.iterrows():
        rows.append(dict(saved_at=ts, name=name, position_id=pid,
                         side=r.get("side", ""), type=r.get("type", ""),
                         strike=r.get("strike", np.nan), qty=r.get("qty", np.nan),
                         entry=r.get("entry", np.nan), **snapshot, **rule_fields))
    out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    _ensure_cols(out, SAVE_COLS, _SAVE_DEFAULTS).to_csv(SAVE_PATH, index=False)
    return pid

def delete_position(pid):
    df = load_saved()
    df[df["position_id"] != pid].to_csv(SAVE_PATH, index=False)

def set_position_rules(pid, rules):
    """Update the per-combo stop/roll rule fields for every leg-row of `pid`.
    Only the keys present in `rules` are written (others left as-is)."""
    df = load_saved()
    m = df["position_id"] == pid
    for k in ("tp_frac", "sl_frac", "roll_dte"):
        if k in rules:
            v = rules[k]
            df.loc[m, k] = (np.nan if v is None or v == "" else v)
    df.to_csv(SAVE_PATH, index=False)

def set_position_status(pid, status, trigger_date="", trigger_level=np.nan,
                        trigger_reason="", roll_child_id=None):
    """Stamp a (usually terminal) status + trigger metadata onto a position."""
    df = load_saved()
    m = df["position_id"] == pid
    df.loc[m, "status"] = status
    df.loc[m, "trigger_date"] = str(trigger_date)
    df.loc[m, "trigger_level"] = trigger_level
    df.loc[m, "trigger_reason"] = str(trigger_reason)
    if roll_child_id is not None:
        df.loc[m, "roll_child_id"] = str(roll_child_id)
    df.to_csv(SAVE_PATH, index=False)

def position_rules(sub):
    """Pull the rule dict from a saved sub-frame (rules are identical per row)."""
    r0 = sub.iloc[0]
    return dict(tp_frac=r0.get("tp_frac", np.nan),
                sl_frac=r0.get("sl_frac", np.nan),
                roll_dte=r0.get("roll_dte", np.nan))

def position_status(sub):
    s = sub.iloc[0].get("status", "open")
    return str(s) if isinstance(s, str) and s else "open"


# =============================================================================
# Forward live-test history store
# =============================================================================
def load_tracking():
    if TRACK_PATH.exists():
        try:
            df = pd.read_csv(TRACK_PATH)
            return _ensure_cols(df, TRACK_COLS, _TRACK_DEFAULTS)
        except Exception:
            pass
    return pd.DataFrame(columns=TRACK_COLS)

def record_tracking(row):
    df = pd.concat([load_tracking(), pd.DataFrame([row])], ignore_index=True)
    _ensure_cols(df, TRACK_COLS, _TRACK_DEFAULTS).to_csv(TRACK_PATH, index=False)

def delete_tracking(pid):
    df = load_tracking()
    df[df["position_id"] != pid].to_csv(TRACK_PATH, index=False)


# =============================================================================
# Live chain access
# =============================================================================
def load_etf(symbol="sh510050"):
    e = B.ak.fund_etf_hist_sina(symbol=symbol)
    e["date"] = pd.to_datetime(e["date"])
    return e.set_index("date")["close"].sort_index()

def _recent_iv_table(etf, underlying="510050", max_back=8):
    """Most recent trading day's full SSE IV table + (asof, spot). None on miss."""
    B.UNDERLYING = underlying
    today = pd.Timestamp(pd.Timestamp.now().date())
    for back in range(0, max_back):
        d = pd.Timestamp((today - pd.Timedelta(days=back)).date())
        tbl = B.fetch_iv_table(d)
        if tbl is not None:
            spot = float(etf.asof(d))
            if np.isfinite(spot):
                return tbl, d, spot
    return None

def latest_chain_for_expiry(etf, expiry_date, underlying="510050"):
    """Most recent trading day's real chain; smile for the LISTED expiry nearest
    `expiry_date`. Returns dict(smile, spot, T, asof, exp4) or None."""
    rec = _recent_iv_table(etf, underlying=underlying)
    if rec is None:
        return None
    tbl, asof, spot = rec
    best = None
    for e in tbl["expiry"].unique():
        ed = B.fourth_wednesday(e)
        if (ed - asof.date()).days <= 0:
            continue
        score = abs((ed - pd.Timestamp(expiry_date).date()).days)
        if best is None or score < best[0]:
            best = (score, e, ed)
    if best is None:
        return None
    _, e, ed = best
    smile = B.otm_smile(tbl, e, spot)
    if len(smile) < 3:
        return None
    T = max((ed - asof.date()).days, 1) / 365.0
    return dict(smile=smile, spot=spot, T=T, asof=asof, exp4=ed)


# =============================================================================
# Roll suggestion (analytics only -- detect, price, log the INTENDED roll)
# =============================================================================
def detect_butterfly(legs):
    """Recognise a (long) butterfly from signed legs: one sold body (qty<0) and
    two bought wings (qty>0) of the same option type, body between the wings.
    Returns dict(cp, body, wing_low, wing_high, lots) or None."""
    if len(legs) != 3:
        return None
    cps = {lg["type"] for lg in legs}
    if len(cps) != 1:
        return None
    cp = cps.pop()
    body = [lg for lg in legs if lg["qty"] < 0]
    wings = [lg for lg in legs if lg["qty"] > 0]
    if len(body) != 1 or len(wings) != 2:
        return None
    bk = float(body[0]["strike"])
    ws = sorted(float(w["strike"]) for w in wings)
    if not (ws[0] < bk < ws[1]):
        return None
    return dict(cp=cp, body=bk, wing_low=ws[0], wing_high=ws[1],
                lots=abs(float(wings[0]["qty"])))

def next_listed_expiry(tbl, after_date, asof_date):
    """Listed expiry whose 4th-Wed is the SOONEST strictly after `after_date`
    (and after asof). Returns (expiry_code, expiry_date) or (None, None)."""
    after = pd.Timestamp(after_date).date()
    best = None
    for e in tbl["expiry"].unique():
        ed = B.fourth_wednesday(e)
        if ed <= after or ed <= asof_date:
            continue
        if best is None or ed < best[1]:
            best = (e, ed)
    return best if best else (None, None)

def build_roll_suggestion(pid, etf=None, underlying="510050"):
    """Build (but do NOT persist) the roll for a saved butterfly: close current,
    open the equivalent fly (body=current spot, SAME wing steps) on the NEXT
    listed expiry, priced on the real chain. Returns a spec dict or
    dict(error=...). Pure suggestion -- you execute the roll in your broker."""
    etf = load_etf() if etf is None else etf
    saved = load_saved()
    sub = saved[saved["position_id"] == pid]
    if not len(sub):
        return dict(error="position not found")
    legs = legs_from_saved(sub)
    geom = detect_butterfly(legs)
    if geom is None:
        return dict(error="not a recognised butterfly (3 legs, 1 body / 2 wings)")
    exp_raw = sub["expiry"].iloc[0]
    try:
        parent_exp = pd.Timestamp(exp_raw).date()
    except Exception:
        return dict(error="parent expiry unreadable")

    rec = _recent_iv_table(etf, underlying=underlying)
    if rec is None:
        return dict(error="no recent real chain")
    tbl, asof, spot = rec
    new_e, new_ed = next_listed_expiry(tbl, parent_exp, asof.date())
    if new_e is None:
        return dict(error="no listed expiry beyond the current one")
    smile = B.otm_smile(tbl, new_e, spot)
    if len(smile) < 5:
        return dict(error="next-expiry smile too thin to price")

    ks = smile.index.values.astype(float)
    step = float(np.median(np.diff(np.sort(ks))))
    if step <= 0:
        return dict(error="bad strike grid")
    steps = int(round((geom["wing_high"] - geom["body"]) / step))
    steps = max(steps, 1)
    i0 = int(np.argmin(np.abs(ks - spot)))
    lo, hi = i0 - steps, i0 + steps
    if lo < 0 or hi >= len(ks):
        return dict(error=f"wing ({steps} steps) runs off the listed grid at the new expiry")
    klow, kmid, khigh = float(ks[lo]), float(ks[i0]), float(ks[hi])
    T = max((new_ed - asof.date()).days, 1) / 365.0
    pr = B.price_butterfly(smile, spot, T, steps)
    cost = float(pr["cost"]) if pr else np.nan
    max_profit = (kmid - klow) - cost if np.isfinite(cost) else np.nan

    # mark the CURRENT position to estimate the close value of the parent
    cur = latest_chain_for_expiry(etf, parent_exp, underlying=underlying)
    if cur is not None:
        cmark, ccost, cpnl = mark_position(legs, cur["smile"], cur["spot"], cur["T"])
    else:
        cmark = ccost = cpnl = np.nan

    return dict(parent_id=pid, asof=asof, spot=spot,
                close_value=cmark, close_pnl=cpnl,
                wing_steps=steps, cp=geom["cp"], lots=geom["lots"],
                new_expiry=new_e, new_exp_date=new_ed,
                body=kmid, wing_low=klow, wing_high=khigh,
                cost=cost, max_profit=max_profit,
                dte=int(round(T * 365)), smile=smile, T=T)

def record_roll(pid, suggestion, name=None, etf=None, underlying="510050"):
    """Persist a roll: create the linked CHILD position from `suggestion`, then
    mark the PARENT as 'rolled' (child id + asof). The child inherits the parent's
    rule set and roll_chain_id so the journal shows one continuous lineage.
    This logs an INTENDED roll for the journal -- it does not touch your broker."""
    if "error" in suggestion:
        return dict(error=suggestion["error"])
    saved = load_saved()
    sub = saved[saved["position_id"] == pid]
    if not len(sub):
        return dict(error="parent not found")
    r0 = sub.iloc[0]
    chain = str(r0.get("roll_chain_id", "") or pid)
    rules = position_rules(sub)

    spot, T, smile = suggestion["spot"], suggestion["T"], suggestion["smile"]
    cp = suggestion["cp"]; lots = suggestion["lots"]
    klow, kmid, khigh = suggestion["wing_low"], suggestion["body"], suggestion["wing_high"]
    e_low = round(leg_value_now(cp, klow, spot, T, smile)[0], 4)
    e_mid = round(leg_value_now(cp, kmid, spot, T, smile)[0], 4)
    e_high = round(leg_value_now(cp, khigh, spot, T, smile)[0], 4)
    legs_df = pd.DataFrame([
        dict(side="Buy",  type=cp, strike=round(klow, 3),  qty=lots, entry=e_low),
        dict(side="Sell", type=cp, strike=round(kmid, 3),  qty=2 * lots, entry=e_mid),
        dict(side="Buy",  type=cp, strike=round(khigh, 3), qty=lots, entry=e_high),
    ])
    child_name = name or (str(r0.get("name", "roll")) + " (roll)")
    snap = dict(expiry=str(suggestion["new_exp_date"]),
                spot_at_save=round(spot, 4), dte_at_save=int(suggestion["dte"]),
                cost_basis=round(float(suggestion["cost"]), 4),
                mark_at_save=round(float(suggestion["cost"]), 4), pnl_at_save=0.0)
    child_pid = save_position(child_name, legs_df, snap, rules=rules,
                              roll_parent_id=pid, roll_chain_id=chain)
    set_position_status(pid, "rolled",
                        trigger_date=str(suggestion["asof"].date()),
                        trigger_level=round(float(suggestion.get("close_pnl", np.nan)), 4)
                        if _finite(suggestion.get("close_pnl")) else np.nan,
                        trigger_reason=f"rolled to {child_pid}",
                        roll_child_id=child_pid)
    return dict(child_id=child_pid, chain=chain)

def roll_lineage(pid):
    """Return the lineage (DataFrame, one row per position in the roll chain that
    contains `pid`) ordered parent->child, with the latest tracked P&L for each
    and the cumulative across the chain. Empty frame if nothing found."""
    saved = load_saved()
    sub = saved[saved["position_id"] == pid]
    if not len(sub):
        return pd.DataFrame()
    chain = str(sub.iloc[0].get("roll_chain_id", "") or pid)
    members = saved[saved["roll_chain_id"].astype(str) == chain]
    track = load_tracking()
    rows = []
    for cpid, g in members.groupby("position_id"):
        r0 = g.iloc[0]
        h = track[track["position_id"] == cpid]
        last_pnl = float(h.sort_values("recorded_at")["pnl"].iloc[-1]) if len(h) \
            else float(r0.get("pnl_at_save", np.nan))
        rows.append(dict(position_id=cpid, name=r0.get("name", ""),
                         parent=str(r0.get("roll_parent_id", "")),
                         expiry=r0.get("expiry", ""), status=r0.get("status", "open"),
                         latest_pnl=round(last_pnl, 4) if np.isfinite(last_pnl) else np.nan,
                         saved_at=r0.get("saved_at", "")))
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("saved_at").reset_index(drop=True)
        out["cumulative_pnl"] = out["latest_pnl"].cumsum().round(4)
    return out


# =============================================================================
# Headless EOD snapshot of every (open) saved position  (+ rule evaluation)
# =============================================================================
def snapshot_all_open(only_open=True, etf=None, underlying="510050"):
    """Mark every saved combination from the latest EOD chain, EVALUATE the
    stop/roll rules, stamp the status, and append to the tracking history.
    De-duplicates by (position_id, chain-asof-date). Returns a list of
    (position_id, message); TP/SL/roll alerts are surfaced in the message and
    the position's status is updated (sticky once terminal). Analytics only -- no
    orders are placed."""
    etf = load_etf() if etf is None else etf
    today = pd.Timestamp(pd.Timestamp.now().date())
    saved = load_saved()
    if not len(saved):
        return [("-", "no saved positions")]
    track = load_tracking()
    done = set()
    if len(track):
        done = set(zip(track["position_id"].astype(str), track["asof_date"].astype(str)))

    out = []
    for pid, sub in saved.groupby("position_id"):
        cur_status = position_status(sub)
        exp_raw = sub["expiry"].iloc[0] if "expiry" in sub.columns else None
        try:
            exp_ts = pd.Timestamp(exp_raw)
        except Exception:
            exp_ts = None

        # expired -> flag once, then stop tracking
        if exp_ts is not None and exp_ts.date() < today.date():
            if cur_status not in TERMINAL:
                set_position_status(pid, "expired",
                                    trigger_date=str(today.date()),
                                    trigger_reason="reached expiry")
                out.append((pid, "ALERT expired -> flagged (close/settle in broker)"))
            else:
                out.append((pid, f"skipped ({cur_status})"))
            continue
        if only_open and cur_status == "rolled":
            out.append((pid, "skipped (rolled)")); continue

        target_exp = exp_ts if exp_ts is not None else (today + pd.Timedelta(days=30))
        info = latest_chain_for_expiry(etf, target_exp, underlying=underlying)
        if info is None:
            out.append((pid, "skipped (no chain/expiry match)")); continue
        asof = str(info["asof"].date())
        if (str(pid), asof) in done:
            out.append((pid, f"already logged for {asof}")); continue

        legs = legs_from_saved(sub)
        mark, cost, pnl = mark_position(legs, info["smile"], info["spot"], info["T"])
        dte = int(round(info["T"] * 365))
        max_profit = position_max_profit(legs)
        rules = position_rules(sub)
        ev = evaluate_rules(pnl, cost, max_profit, dte, rules)

        # status: sticky once terminal; otherwise adopt a newly-fired TP/SL
        if cur_status in TERMINAL:
            row_status = cur_status
        elif ev["status"] != "open":
            row_status = ev["status"]
            set_position_status(pid, ev["status"], trigger_date=asof,
                                trigger_level=ev["level"], trigger_reason=ev["reason"])
        else:
            row_status = "open"

        alerts = []
        if ev["status"] != "open" and cur_status not in TERMINAL:
            alerts.append(f"{ev['status'].upper()} ({ev['reason']})")
        if ev["roll"]:
            alerts.append(ev["roll_reason"])
        alert_txt = " | ".join(alerts)

        record_tracking(dict(position_id=pid,
                             recorded_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                             asof_date=asof, expiry=str(exp_raw),
                             spot=round(info["spot"], 4), dte=dte,
                             cost_basis=round(cost, 4), mark=round(mark, 4),
                             pnl=round(pnl, 4), status=row_status, alert=alert_txt))
        msg = f"logged pnl={pnl:+.4f} spot={info['spot']:.3f} dte={dte} asof {asof}"
        if alert_txt:
            msg = "ALERT " + alert_txt + " :: " + msg
        out.append((pid, msg))
    return out


# =============================================================================
# Two-strategy auto-tracker (butterfly + iron condor)
# =============================================================================
# A daily FORWARD paper-trade of the two model strategies. Each strategy's legs
# are LOCKED once per expiry month (the structure the model suggested when that
# month's expiry first became the front contract); every subsequent day we just
# re-mark those locked legs on the latest real chain and log the P&L. This is the
# "daily fresh snapshot" tracker -- it records the daily movement of the two
# strategies, with no manually entered combination. Suggestions only; no orders.
#
# Files (next to this module):
#   strategy_entries.csv   -- one row per leg of each (strategy, expiry) entry
#                             (the locked structure for that month).
#   strategy_tracking.csv  -- one row per (strategy, expiry, asof-date) snapshot.
STRAT_ENTRY_PATH = HERE / "strategy_entries.csv"
STRAT_TRACK_PATH = HERE / "strategy_tracking.csv"

STRATEGIES = ("butterfly", "condor", "condor10")
# Default structure parameters. butterfly wing is overridden by the model band
# when the snapshot is taken from the dashboard / pipeline (which pass bfly_wing
# = WX.model_band_wing(contexts)); the defaults here are the standalone fallback.
#   condor    -- aggressive front-month VRP condor (shorts 2 steps OTM, 2-step wings)
#   condor10  -- conservative income sleeve: edge-bearing ~1-sigma (~20-delta) shorts
#                with wider (3-step) protective wings, sized small (CONDOR10_FRACTION
#                of bankroll/trade) to a ~-35% max-drawdown budget. A passive monthly
#                50ETF condor caps near 5-6%/yr after costs at sane sizing, so this
#                sleeve targets the realistic ~6% (not 10%). See out/condor10_spec.json.
STRAT_DEFAULTS = dict(bfly_wing=3, condor_m=2, condor_p=2,
                      condor10_m=2, condor10_p=3)

# condor10 sizing: bankroll fraction risked per trade (max-loss based) and the
# annualized return it delivered in the 2015-2026 backtest under the realistic
# (flat-vol + slippage) case: f=0.12 -> ~5.8% CAGR, ~-35% MDD, 80% win.
CONDOR10_FRACTION = 0.12
CONDOR10_TARGET_RETURN = 0.06

STRAT_ENTRY_COLS = ["strategy", "expiry", "entry_date",
                    "side", "type", "strike", "qty", "entry"]
STRAT_TRACK_COLS = ["strategy", "expiry", "entry_date", "recorded_at",
                    "asof_date", "spot", "dte", "cost_basis", "mark", "pnl"]

STRAT_LABEL = {"butterfly": "Long call butterfly",
               "condor": "Iron condor (short vol, defined risk)",
               "condor10": "Iron condor 10%-target (income sleeve, ~6% realistic)"}


# ---- structure builders (priced off the live smile, same BS as everything else)
def build_butterfly_legs(smile, spot, T, wing):
    """3-leg long call butterfly (body=ATM, symmetric wings `wing` steps out).
    Returns a legs DataFrame (side/type/strike/qty/entry) or None if off-grid."""
    ks = smile.index.values.astype(float)
    if len(ks) < 3:
        return None
    i0 = int(np.argmin(np.abs(ks - spot)))
    lo, hi = i0 - int(wing), i0 + int(wing)
    if lo < 0 or hi >= len(ks):
        return None
    klow, kmid, khigh = float(ks[lo]), float(ks[i0]), float(ks[hi])
    e = lambda cp, K: round(leg_value_now(cp, K, spot, T, smile)[0], 4)
    return pd.DataFrame([
        dict(side="Buy",  type="C", strike=round(klow, 3),  qty=1, entry=e("C", klow)),
        dict(side="Sell", type="C", strike=round(kmid, 3),  qty=2, entry=e("C", kmid)),
        dict(side="Buy",  type="C", strike=round(khigh, 3), qty=1, entry=e("C", khigh)),
    ])

def build_condor_legs(smile, spot, T, m, p):
    """4-leg iron condor: sell a put `m` steps OTM + a call `m` steps OTM, buy
    protective wings `p` steps further out. Net CREDIT structure. Returns a legs
    DataFrame or None if the protective wings run off the listed grid."""
    ks = smile.index.values.astype(float)
    if len(ks) < 7:
        return None
    i0 = int(np.argmin(np.abs(ks - spot)))
    il_p, is_p, is_c, il_c = i0 - m - p, i0 - m, i0 + m, i0 + m + p
    if il_p < 0 or il_c >= len(ks):
        return None
    Kp_l, Kp_s, Kc_s, Kc_l = (float(ks[il_p]), float(ks[is_p]),
                              float(ks[is_c]), float(ks[il_c]))
    e = lambda cp, K: round(leg_value_now(cp, K, spot, T, smile)[0], 4)
    return pd.DataFrame([
        dict(side="Buy",  type="P", strike=round(Kp_l, 3), qty=1, entry=e("P", Kp_l)),
        dict(side="Sell", type="P", strike=round(Kp_s, 3), qty=1, entry=e("P", Kp_s)),
        dict(side="Sell", type="C", strike=round(Kc_s, 3), qty=1, entry=e("C", Kc_s)),
        dict(side="Buy",  type="C", strike=round(Kc_l, 3), qty=1, entry=e("C", Kc_l)),
    ])

def build_strategy_legs(strategy, smile, spot, T, params=None):
    params = {**STRAT_DEFAULTS, **(params or {})}
    if strategy == "butterfly":
        return build_butterfly_legs(smile, spot, T, int(params["bfly_wing"]))
    if strategy == "condor":
        return build_condor_legs(smile, spot, T, int(params["condor_m"]),
                                 int(params["condor_p"]))
    if strategy == "condor10":
        return build_condor_legs(smile, spot, T, int(params["condor10_m"]),
                                 int(params["condor10_p"]))
    return None


# ---- current front-month chain (latest real SSE chain + model ~30d expiry) ----
def current_expiry_chain(etf=None, underlying="510050"):
    """Most recent real chain + the model's preferred (~30 DTE) front expiry.
    Returns dict(smile, spot, T, asof, expiry, exp_date) or None."""
    etf = load_etf() if etf is None else etf
    rec = _recent_iv_table(etf, underlying=underlying)
    if rec is None:
        return None
    tbl, asof, spot = rec
    expiry, exp_date = B.choose_expiry(tbl, asof)
    if expiry is None:
        return None
    smile = B.otm_smile(tbl, expiry, spot)
    if len(smile) < 5:
        return None
    T = max((exp_date - asof.date()).days, 1) / 365.0
    return dict(smile=smile, spot=spot, T=T, asof=asof,
                expiry=expiry, exp_date=exp_date)


# ---- strategy-entry store (locked structure per (strategy, expiry)) ----------
def load_strategy_entries():
    if STRAT_ENTRY_PATH.exists():
        try:
            return pd.read_csv(STRAT_ENTRY_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=STRAT_ENTRY_COLS)

def get_locked_legs(strategy, expiry):
    """Return the locked legs DataFrame for (strategy, expiry), or None."""
    df = load_strategy_entries()
    if not len(df):
        return None
    m = (df["strategy"].astype(str) == str(strategy)) & \
        (df["expiry"].astype(str) == str(expiry))
    sub = df[m]
    if not len(sub):
        return None
    return sub[["side", "type", "strike", "qty", "entry"]].reset_index(drop=True)

def get_entry_date(strategy, expiry):
    df = load_strategy_entries()
    if not len(df):
        return None
    m = (df["strategy"].astype(str) == str(strategy)) & \
        (df["expiry"].astype(str) == str(expiry))
    sub = df[m]
    return str(sub["entry_date"].iloc[0]) if len(sub) else None

def save_strategy_entry(strategy, expiry, entry_date, legs_df):
    """Persist the locked legs for (strategy, expiry) (no-op if already present)."""
    df = load_strategy_entries()
    if get_locked_legs(strategy, expiry) is not None:
        return
    rows = []
    for _, r in legs_df.iterrows():
        rows.append(dict(strategy=strategy, expiry=str(expiry),
                         entry_date=str(entry_date), side=r["side"], type=r["type"],
                         strike=r["strike"], qty=r["qty"], entry=r["entry"]))
    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(STRAT_ENTRY_PATH, index=False)


# ---- strategy-tracking history store ----------------------------------------
def load_strategy_tracking():
    if STRAT_TRACK_PATH.exists():
        try:
            return pd.read_csv(STRAT_TRACK_PATH)
        except Exception:
            pass
    return pd.DataFrame(columns=STRAT_TRACK_COLS)

def record_strategy_tracking(row):
    df = pd.concat([load_strategy_tracking(), pd.DataFrame([row])],
                   ignore_index=True)
    df.to_csv(STRAT_TRACK_PATH, index=False)

def delete_strategy_tracking(strategy=None):
    """Clear the whole strategy track, or just one strategy's rows."""
    df = load_strategy_tracking()
    if strategy is None:
        df.head(0).to_csv(STRAT_TRACK_PATH, index=False)
    else:
        df[df["strategy"].astype(str) != str(strategy)].to_csv(STRAT_TRACK_PATH, index=False)


def snapshot_strategies(etf=None, params=None, underlying="510050"):
    """Daily snapshot of BOTH model strategies. Locks each strategy's legs the
    first time its current front expiry is seen, then re-marks those locked legs
    on the latest real chain and appends a P&L row. De-duplicates by
    (strategy, expiry, chain-asof-date). Returns a list of (strategy, message).
    Analytics only -- no orders are placed."""
    etf = load_etf() if etf is None else etf
    params = {**STRAT_DEFAULTS, **(params or {})}
    chain = current_expiry_chain(etf, underlying=underlying)
    if chain is None:
        return [("-", "no recent real chain to mark the strategies")]
    asof = str(chain["asof"].date())
    expiry = str(chain["expiry"])
    track = load_strategy_tracking()
    done = set()
    if len(track):
        done = set(zip(track["strategy"].astype(str), track["expiry"].astype(str),
                       track["asof_date"].astype(str)))

    out = []
    for strat in STRATEGIES:
        legs_df = get_locked_legs(strat, expiry)
        if legs_df is None:
            built = build_strategy_legs(strat, chain["smile"], chain["spot"],
                                        chain["T"], params)
            if built is None:
                out.append((strat, f"skip ({expiry}): structure runs off the listed grid"))
                continue
            save_strategy_entry(strat, expiry, asof, built)
            legs_df = built
        if (strat, expiry, asof) in done:
            out.append((strat, f"already logged for {asof} ({expiry})"))
            continue
        legs = legs_from_saved(legs_df)
        mark, cost, pnl = mark_position(legs, chain["smile"], chain["spot"], chain["T"])
        dte = int(round(chain["T"] * 365))
        record_strategy_tracking(dict(
            strategy=strat, expiry=expiry,
            entry_date=str(get_entry_date(strat, expiry) or asof),
            recorded_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            asof_date=asof, spot=round(chain["spot"], 4), dte=dte,
            cost_basis=round(cost, 4), mark=round(mark, 4), pnl=round(pnl, 4)))
        out.append((strat, f"logged pnl={pnl:+.4f} spot={chain['spot']:.3f} "
                           f"dte={dte} ({expiry}) asof {asof}"))
    return out


if __name__ == "__main__":
    print("-- saved positions --")
    for pid, msg in snapshot_all_open():
        print(f"  {pid} -> {msg}")
    print("-- model strategies --")
    for strat, msg in snapshot_strategies():
        print(f"  {strat} -> {msg}")
