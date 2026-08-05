"""Custom combo position store + mark-to-market against the chain CSVs.

positions.csv holds one row per leg; position_id groups legs into a combo.
Daily P&L history is reconstructed from the chain history on demand, so
nothing needs to be appended day by day — keep the data CSV updated and
every open position re-marks automatically.

Legs within a combo are NOT all-or-nothing: a leg can be closed individually
(close_leg), split into a closed portion + a remaining open portion
(partial_close_leg), corrected in place (edit_leg), or a brand new leg can be
added to an already-open combo (add_leg_to_position, e.g. for a roll). Each
leg row therefore carries its OWN status/close_date/close_price and its own
leg_open_date (when that leg specifically came on) — the combo-level
open_date is just the date the combo was first created, kept for display.

Invariant relied on elsewhere: at most one non-closed row per
(position_id, leg_code) at a time — close_leg/partial_close_leg only ever
touch the single open row for a leg_code, so this holds as long as
add_leg_to_position isn't used to add a second position in an already-open
contract.
"""
from __future__ import annotations

import uuid
from datetime import date

import pandas as pd

import data_store as ds

LEG_COLS = ["position_id", "pos_name", "underlying", "open_date",
            "leg_open_date", "status", "close_date", "leg_code", "leg_name",
            "cp", "strike", "expiry", "direction", "qty", "entry_price",
            "close_price", "notes"]


def load() -> pd.DataFrame:
    if not ds.POSITIONS_CSV.exists():
        return pd.DataFrame(columns=LEG_COLS)
    df = pd.read_csv(ds.POSITIONS_CSV, dtype={"leg_code": str,
                                              "underlying": str})
    if "leg_open_date" not in df.columns:
        df["leg_open_date"] = df["open_date"]
    else:
        blank = df["leg_open_date"].isna() | (df["leg_open_date"].astype(str) == "")
        df.loc[blank, "leg_open_date"] = df.loc[blank, "open_date"]
    return df


def _save(df: pd.DataFrame) -> None:
    df[LEG_COLS].to_csv(ds.POSITIONS_CSV, index=False)


def add_position(pos_name: str, underlying: str, open_date: str,
                 legs: list[dict], notes: str = "") -> str:
    """legs: [{leg_code, leg_name, cp, strike, expiry, direction, qty,
    entry_price}]"""
    pid = uuid.uuid4().hex[:8]
    rows = [{"position_id": pid, "pos_name": pos_name,
             "underlying": underlying, "open_date": open_date,
             "leg_open_date": open_date,
             "status": "open", "close_date": "", "close_price": "",
             "notes": notes, **leg} for leg in legs]
    old = load()
    new = pd.DataFrame(rows)
    df = new if old.empty else pd.concat([old, new], ignore_index=True)
    _save(df)
    return pid


def close_position(pid: str, close_prices: dict[str, float],
                   close_date: str | None = None) -> None:
    """Close every still-open leg of a combo at once. close_prices:
    {leg_code: price}. Missing codes use last mark."""
    df = load()
    df["close_date"] = df["close_date"].astype("object")
    df["close_price"] = df["close_price"].astype("object")
    m = (df["position_id"] == pid) & (df["status"] != "closed")
    df.loc[m, "status"] = "closed"
    df.loc[m, "close_date"] = close_date or date.today().isoformat()
    for code, px in close_prices.items():
        df.loc[m & (df["leg_code"] == code), "close_price"] = px
    _save(df)


def close_leg(pid: str, leg_code: str, price: float,
             close_date: str | None = None) -> None:
    """Close ONE leg of a combo, leaving the rest open."""
    df = load()
    df["close_date"] = df["close_date"].astype("object")
    df["close_price"] = df["close_price"].astype("object")
    m = ((df["position_id"] == pid) & (df["leg_code"] == leg_code) &
         (df["status"] != "closed"))
    if not m.any():
        raise ValueError(f"no open leg {leg_code} in position {pid}")
    df.loc[m, "status"] = "closed"
    df.loc[m, "close_date"] = close_date or date.today().isoformat()
    df.loc[m, "close_price"] = price
    _save(df)


def partial_close_leg(pid: str, leg_code: str, qty: int, price: float,
                      close_date: str | None = None) -> None:
    """Split `qty` units off an open leg as closed at `price`; the remainder
    (original qty - qty) stays open under the same leg_code."""
    df = load()
    df["close_date"] = df["close_date"].astype("object")
    df["close_price"] = df["close_price"].astype("object")
    m = ((df["position_id"] == pid) & (df["leg_code"] == leg_code) &
         (df["status"] != "closed"))
    if not m.any():
        raise ValueError(f"no open leg {leg_code} in position {pid}")
    idx = df.index[m][0]
    total_qty = int(df.loc[idx, "qty"])
    if not (0 < qty < total_qty):
        raise ValueError(f"partial-close qty must be between 1 and "
                         f"{total_qty - 1} (leg has {total_qty})")
    closed_row = df.loc[idx].copy()
    df.loc[idx, "qty"] = total_qty - qty
    closed_row["qty"] = qty
    closed_row["status"] = "closed"
    closed_row["close_date"] = close_date or date.today().isoformat()
    closed_row["close_price"] = price
    df = pd.concat([df, pd.DataFrame([closed_row])], ignore_index=True)
    _save(df)


def edit_leg(pid: str, leg_code: str, updates: dict) -> None:
    """Directly correct fields (qty, strike, entry_price, cp, expiry, ...) on
    an OPEN leg -- for data-entry fixes, not trade actions."""
    df = load()
    m = ((df["position_id"] == pid) & (df["leg_code"] == leg_code) &
         (df["status"] != "closed"))
    if not m.any():
        raise ValueError(f"no open leg {leg_code} in position {pid}")
    for k, v in updates.items():
        df.loc[m, k] = v
    _save(df)


def add_leg_to_position(pid: str, leg: dict, entry_date: str | None = None) -> None:
    """Append a new leg to an existing combo -- e.g. a roll or an adjustment
    where one leg closes and a replacement opens without starting a whole new
    position. leg: {leg_code, leg_name, cp, strike, expiry, direction, qty,
    entry_price}."""
    df = load()
    g = df[df["position_id"] == pid]
    if g.empty:
        raise ValueError(f"no such position {pid}")
    base = g.iloc[0]
    row = {"position_id": pid, "pos_name": base["pos_name"],
          "underlying": base["underlying"], "open_date": base["open_date"],
          "leg_open_date": entry_date or date.today().isoformat(),
          "status": "open", "close_date": "", "close_price": "",
          "notes": "", **leg}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)


def delete_position(pid: str) -> None:
    df = load()
    _save(df[df["position_id"] != pid])


# ------------------------------------------------------------------ marking
def latest_marks(chain: pd.DataFrame) -> tuple[pd.Series, pd.Timestamp]:
    return marks_at_offset(chain, offset=0)


def marks_at_offset(chain: pd.DataFrame, offset: int = 0) -> tuple[pd.Series, pd.Timestamp]:
    """Marks as of the offset-th most recent chain date (0=latest, 1=T-1, ...).
    Clamped to the earliest available date if the chain is shorter than offset."""
    dates = sorted(chain[ds.C_DATE].unique())
    idx = max(0, len(dates) - 1 - offset)
    dt = dates[idx]
    day = chain[chain[ds.C_DATE] == dt]
    return day.set_index(ds.C_CODE)[ds.C_CLOSE], dt


def position_summary(chains: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per position with cost, mark, P&L (yuan). Status is an
    aggregate across legs: 'open' if any leg is still open, else 'closed'."""
    df = load()
    if df.empty:
        return df
    out = []
    marks_by_underlying: dict[str, tuple] = {}  # avoid re-filtering the same
    # underlying's full chain history once per position that shares it
    for pid, g in df.groupby("position_id"):
        u = g["underlying"].iloc[0]
        if u not in marks_by_underlying:
            chain = chains.get(u)
            marks_by_underlying[u] = (None, None) if chain is None else latest_marks(chain)
        marks, asof = marks_by_underlying[u]
        cost = mark = 0.0
        n_open = int((g["status"] != "closed").sum())
        for _, leg in g.iterrows():
            signed = leg["direction"] * leg["qty"]
            cost += signed * leg["entry_price"]
            if leg["status"] == "closed" and pd.notna(leg["close_price"]) \
                    and str(leg["close_price"]) != "":
                px = float(leg["close_price"])
            elif marks is not None and leg["leg_code"] in marks.index:
                px = float(marks[leg["leg_code"]])
            else:
                px = leg["entry_price"]
            mark += signed * px
        close_dates = g.loc[g["status"] == "closed", "close_date"]
        out.append({"position_id": pid, "pos_name": g["pos_name"].iloc[0],
                    "underlying": u, "open_date": g["open_date"].iloc[0],
                    "status": "open" if n_open else "closed",
                    "close_date": close_dates.max() if n_open == 0 and len(close_dates) else "",
                    "n_legs": len(g), "n_open": n_open,
                    "cost_yuan": round(cost * ds.MULTIPLIER, 0),
                    "mark_yuan": round(mark * ds.MULTIPLIER, 0),
                    "pnl_yuan": round((mark - cost) * ds.MULTIPLIER, 0),
                    "asof": asof})
    return pd.DataFrame(out)


def pnl_history(pid: str, chains: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily mark-to-market P&L series for one position since open.

    Cost basis is time-varying: a leg only contributes (to both cost and
    mark) from its own leg_open_date onward, so a leg added later (a roll,
    an adjustment) doesn't retroactively distort earlier P&L. Once a leg is
    closed its contribution freezes at its close_price from close_date
    onward -- realized P&L carries forward flat instead of disappearing.
    """
    df = load()
    g = df[df["position_id"] == pid]
    if g.empty:
        return pd.DataFrame()
    u = g["underlying"].iloc[0]
    chain = chains.get(u)
    if chain is None:
        return pd.DataFrame()

    g = g.copy()
    g["leg_open_dt"] = pd.to_datetime(g["leg_open_date"])
    is_closed = g["status"] == "closed"
    g["leg_close_dt"] = pd.NaT
    g.loc[is_closed, "leg_close_dt"] = pd.to_datetime(g.loc[is_closed, "close_date"])

    start = g["leg_open_dt"].min()
    end = (g["leg_close_dt"].max() if is_closed.all()
           else chain[ds.C_DATE].max())

    codes = list(g["leg_code"].unique())
    sub = chain[(chain[ds.C_CODE].isin(codes)) &
                (chain[ds.C_DATE] >= start) & (chain[ds.C_DATE] <= end)]
    px = sub.pivot_table(index=ds.C_DATE, columns=ds.C_CODE,
                         values=ds.C_CLOSE).ffill()
    spot = sub.groupby(ds.C_DATE)[ds.C_UND_CLOSE].first()

    rows = []
    for dt, r in px.iterrows():
        cost = mark = 0.0
        for _, l in g.iterrows():
            if dt < l["leg_open_dt"]:
                continue  # leg didn't exist yet
            signed = l["direction"] * l["qty"]
            cost += signed * l["entry_price"]
            if pd.notna(l["leg_close_dt"]) and dt >= l["leg_close_dt"]:
                px_leg = float(l["close_price"])  # frozen realized mark
            else:
                px_leg = r.get(l["leg_code"], l["entry_price"])
            mark += signed * px_leg
        rows.append({"date": dt, "mark": mark,
                     "pnl_yuan": (mark - cost) * ds.MULTIPLIER,
                     "spot": spot.get(dt)})
    return pd.DataFrame(rows)


def account_equity_curve(chains: dict[str, pd.DataFrame],
                         baseline_value: float,
                         baseline_asof: str | None = None) -> pd.DataFrame:
    """Whole-account asset-tracking curve across EVERY tracked position (open
    and closed combined), anchored to `baseline_value` -- the last known
    TRUE account value (cash + all positions marked to market), verified as
    of `baseline_asof`.

    Each position's own pnl_history is reindexed onto the shared date axis:
    0 before that position's own start, forward-filled (flat) after its own
    last available date (this naturally covers closed positions, whose
    pnl_history stops at close but whose realized P&L must keep counting
    forward). Summing across positions gives a combined P&L path; equity(t)
    = baseline_value + (total_pnl(t) - total_pnl(baseline_asof)) backs out
    an implied account value at every date, including dates AFTER
    baseline_asof -- so as the daily updater appends fresh chain data, the
    latest point on this curve rolls forward automatically instead of
    staying pinned to whatever was last typed into the anchor box.

    If baseline_asof is missing or falls outside the tracked P&L window, the
    anchor falls back to the latest available date (old behavior: equity at
    the most recent point equals baseline_value exactly).

    This assumes the ONLY thing moving the account within the window is
    these tracked option positions -- no deposits/withdrawals/other trades.
    A simplifying assumption for a local combo tracker, not a brokerage-grade
    ledger; re-anchor baseline_value periodically against your real balance.
    """
    df = load()
    if df.empty:
        return pd.DataFrame()
    series = {}
    for pid, g in df.groupby("position_id"):
        h = pnl_history(pid, chains)
        if h.empty:
            continue
        series[pid] = h.set_index("date")["pnl_yuan"]
    if not series:
        return pd.DataFrame()

    full_index = pd.DatetimeIndex(
        sorted(set().union(*[s.index for s in series.values()])))
    # One extra day before the earliest position's start, so the curve shows
    # the account's value just before any tracked position existed (all
    # per-position series reindex to NaN->0 there, since it precedes their
    # own start too).
    pre_start = full_index.min() - pd.Timedelta(days=1)
    full_index = pd.DatetimeIndex(sorted(set(full_index) | {pre_start}))
    total = pd.Series(0.0, index=full_index)
    for s in series.values():
        total = total.add(s.reindex(full_index).ffill().fillna(0.0))

    anchor_dt = pd.Timestamp(baseline_asof) if baseline_asof else None
    if anchor_dt is not None and full_index.min() <= anchor_dt:
        anchor_idx = full_index[full_index <= anchor_dt][-1]
        anchor_total = float(total.loc[anchor_idx])
    else:
        anchor_total = float(total.iloc[-1])
    equity = baseline_value + (total - anchor_total)
    return pd.DataFrame({"date": full_index, "total_pnl_yuan": total.values,
                        "equity_yuan": equity.values})


# ------------------------------------------------------------------ lookup
def contract_picker(chain: pd.DataFrame, expiry=None, cp=None) -> pd.DataFrame:
    """Contracts as of the latest chain date, for UI dropdowns."""
    last_dt = chain[ds.C_DATE].max()
    day = chain[chain[ds.C_DATE] == last_dt]
    if expiry is not None:
        day = day[day[ds.C_EXPIRY] == expiry]
    if cp:
        day = day[day[ds.C_CP] == cp]
    cols = [ds.C_CODE, ds.C_NAME, ds.C_CP, ds.C_STRIKE, ds.C_EXPIRY,
            ds.C_CLOSE, ds.C_IV, ds.C_DELTA, ds.C_OI]
    return day[cols].sort_values([ds.C_EXPIRY, ds.C_CP, ds.C_STRIKE])
