"""Intraday IV seasonality check: does ATM IV run higher near the open and
decay through the trading day ("vol crush" into the close)?

Uses the live minute-level option chain in Data/options/minute/ (started
2026-08-11 -- the only source with real intraday granularity; updater.py's
daily chain has just one EOD snapshot per contract).

Method
------
For each underlying x trading day: take the front-month (min dte) contracts
for that day, and at every polled minute find the strike nearest that
minute's spot, invert call+put IV via bs_iv.implied_vol (same on-demand
basic-data-in/IV-out pattern minute_updater.iv_for_row uses; IV_BASIS_CP_AVG
convention -- average of call and put, matching the live-monitor default),
and average the two legs. That gives one ATM-IV path per underlying per day.

Day-to-day vol *level* swings (e.g. a risk-off week vs a calm week) are much
bigger than any plausible intraday effect, so each day's path is indexed to
that day's own mean IV (iv / day_mean) before pooling -- otherwise a handful
of high-vol days would dominate the average and manufacture a false shape.
Morning (09:30-11:30) and afternoon (13:00-15:00) sessions are kept as
separate minutes-since-session-open axes (there's a 90-minute closed lunch
in between, not tradeable time).

Output: per-underlying and pooled-across-underlyings mean indexed IV by
minute-of-session bucket, printed as a table, plus the morning-vs-afternoon
and first-30min-vs-last-30min comparison actually asked for.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import bs_iv
import data_store as ds

# Data/options/minute is gitignored (large runtime data, not repo-tracked) so
# it doesn't exist inside a git worktree checkout -- read it from the real
# repo path directly regardless of where this script itself is running from.
MINUTE_DIR = Path("/mnt/data/Trading/Data/options/minute")

FILE_RE = re.compile(r"^(index_)?option_minute_(\d+)_(\d{8})\.csv$")

BUCKET_MIN = 10  # minutes-since-session-open bucket width


def _iv_row(cp: str, spot: float, strike: float, dte: float, bid: float, ask: float, last: float) -> float:
    price = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    if not (price > 0 and spot > 0 and strike > 0 and dte > 0):
        return float("nan")
    return bs_iv.implied_vol(cp, spot, strike, dte / 365.0, price)


def _session_minute(ts: pd.Timestamp) -> tuple[str, int] | None:
    t = ts.time()
    if t < pd.Timestamp("09:30").time() or t > pd.Timestamp("15:00").time():
        return None
    if t <= pd.Timestamp("11:30").time():
        open_t = ts.replace(hour=9, minute=30, second=0)
        return "morning", int((ts - open_t).total_seconds() // 60)
    if t >= pd.Timestamp("13:00").time():
        open_t = ts.replace(hour=13, minute=0, second=0)
        return "afternoon", int((ts - open_t).total_seconds() // 60)
    return None


def day_atm_iv_path(path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(path)
    if df.empty:
        return None
    front_dte = df["dte"].min()
    df = df[df["dte"] == front_dte].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    rows = []
    for ts, g in df.groupby("timestamp"):
        spot = g["spot"].iloc[0]
        if not (spot > 0):
            continue
        idx = (g["strike"] - spot).abs().idxmin()
        atm_strike = g.loc[idx, "strike"]
        leg = g[g["strike"] == atm_strike]
        ivs = []
        for _, r in leg.iterrows():
            iv = _iv_row(r["cp"], spot, r["strike"], r["dte"], r["bid"], r["ask"], r["last"])
            if pd.notna(iv):
                ivs.append(iv)
        if not ivs:
            continue
        rows.append((ts, sum(ivs) / len(ivs)))

    if not rows:
        return None
    out = pd.DataFrame(rows, columns=["timestamp", "iv"]).sort_values("timestamp")
    day_mean = out["iv"].mean()
    if not (day_mean > 0):
        return None
    out["iv_idx"] = out["iv"] / day_mean

    sess_min = out["timestamp"].apply(_session_minute)
    out = out[sess_min.notna()].copy()
    out["session"] = [s for s, _ in sess_min[sess_min.notna()]]
    out["min_since_open"] = [m for _, m in sess_min[sess_min.notna()]]
    out["bucket"] = (out["min_since_open"] // BUCKET_MIN) * BUCKET_MIN
    return out


def main():
    files = sorted(MINUTE_DIR.glob("*.csv"))
    matched = [(f, m) for f in files if (m := FILE_RE.match(f.name))]
    print(f"{len(matched)} minute files found under {MINUTE_DIR}")

    all_days = []
    for f, m in matched:
        underlying = m.group(2)
        day_df = day_atm_iv_path(f)
        if day_df is None:
            continue
        day_df["underlying"] = underlying
        day_df["file"] = f.name
        all_days.append(day_df)

    if not all_days:
        print("No usable IV paths extracted.")
        return

    pooled = pd.concat(all_days, ignore_index=True)
    n_days = pooled.groupby("underlying")["file"].nunique()
    print("\nTrading days per underlying:")
    print(n_days.to_string())

    print(f"\nTotal (underlying, day) paths: {pooled.groupby(['underlying', 'file']).ngroups}, "
          f"total minute observations: {len(pooled)}")

    # Pooled shape: mean indexed IV by session + bucket, across all underlyings/days
    print(f"\n=== Pooled indexed ATM IV by minutes-since-session-open (bucket={BUCKET_MIN}min) ===")
    shape = pooled.groupby(["session", "bucket"])["iv_idx"].agg(["mean", "std", "count"])
    for session in ["morning", "afternoon"]:
        if session not in shape.index.get_level_values(0):
            continue
        print(f"\n-- {session} --")
        print(shape.loc[session].to_string(float_format=lambda x: f"{x:.4f}"))

    # Per-underlying shape (mean indexed IV in first 30min vs last 30min of each session)
    print("\n=== Per-underlying: first-30min vs last-30min of session (indexed IV, 1.0 = day mean) ===")
    for u, g in pooled.groupby("underlying"):
        first_m = g[(g.session == "morning") & (g.min_since_open < 30)]["iv_idx"].mean()
        last_m = g[(g.session == "morning") & (g.min_since_open >= 90)]["iv_idx"].mean()
        first_a = g[(g.session == "afternoon") & (g.min_since_open < 30)]["iv_idx"].mean()
        last_a = g[(g.session == "afternoon") & (g.min_since_open >= 90)]["iv_idx"].mean()
        close = g[(g.session == "afternoon") & (g.min_since_open >= 100)]["iv_idx"].mean()
        print(f"{u}: morning open={first_m:.4f} morning late={last_m:.4f} | "
              f"afternoon open={first_a:.4f} afternoon late={last_a:.4f} | last~20min={close:.4f}")

    # Overall summary: open (first 30min of morning) vs close (last 20min of afternoon)
    open_iv = pooled[(pooled.session == "morning") & (pooled.min_since_open < 30)]["iv_idx"]
    close_iv = pooled[(pooled.session == "afternoon") & (pooled.min_since_open >= 100)]["iv_idx"]
    print(f"\n=== Overall: open (first 30min) mean indexed IV = {open_iv.mean():.4f} (n={len(open_iv)}) "
          f"vs close (last ~20min) = {close_iv.mean():.4f} (n={len(close_iv)}) ===")
    print(f"Open - Close = {open_iv.mean() - close_iv.mean():+.4f} "
          f"({(open_iv.mean() / close_iv.mean() - 1) * 100:+.2f}% relative)")

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "intraday_iv_seasonality_pooled.csv"
    shape.to_csv(out_path)
    print(f"\nSaved pooled bucket table to {out_path}")

    make_charts(shape, out_dir)


def make_charts(shape: pd.DataFrame, out_dir: Path) -> None:
    """Static PNG (matplotlib, same convention as chart_common.render_matplotlib_png
    -- no kaleido dependency for plotly PNG export) + an interactive HTML (plotly)
    of the pooled indexed-IV-by-session-minute shape. Categorical palette slots 1
    (blue, morning) / 2 (orange, afternoon) per the project's validated dataviz
    palette -- 2 series, both direct-labeled, so no legend ambiguity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    C_SURFACE, C_INK, C_INK2, C_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    C_GRID, C_AXIS = "#e1e0d9", "#c3c2b7"
    C_MORNING, C_AFTERNOON = "#2a78d6", "#eb6834"  # categorical slots 1 / 2

    morning = shape.loc["morning"].reset_index() if "morning" in shape.index.get_level_values(0) else None
    afternoon = shape.loc["afternoon"].reset_index() if "afternoon" in shape.index.get_level_values(0) else None

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130, facecolor=C_SURFACE)
    ax.set_facecolor(C_SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_AXIS)
    ax.spines["bottom"].set_color(C_AXIS)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=C_MUTED, linewidth=0.8, linestyle=":")

    if morning is not None:
        ax.plot(morning["bucket"], morning["mean"], color=C_MORNING, linewidth=2.2, marker="o", markersize=4)
        ax.text(morning["bucket"].iloc[-1] + 2, morning["mean"].iloc[-1], "morning",
                color=C_MORNING, fontsize=10, va="center", fontweight="bold")
    if afternoon is not None:
        ax.plot(afternoon["bucket"], afternoon["mean"], color=C_AFTERNOON, linewidth=2.2, marker="o", markersize=4)
        ax.text(afternoon["bucket"].iloc[-1] + 2, afternoon["mean"].iloc[-1], "afternoon",
                color=C_AFTERNOON, fontsize=10, va="center", fontweight="bold")

    ax.set_xlabel("minutes since session open", color=C_INK2, fontsize=10)
    ax.set_ylabel("ATM IV, indexed to that day's mean (1.0 = day avg)", color=C_INK2, fontsize=10)
    ax.set_title(
        "Intraday ATM IV shape: pooled across 7 underlyings, 22-23 trading days each\n"
        "(2026-08-11 to 2026-09-11 minute chain, front-month ATM, call+put avg)",
        color=C_INK, fontsize=11, loc="left", pad=10,
    )
    fig.tight_layout()
    png_path = out_dir / "intraday_iv_seasonality.png"
    fig.savefig(png_path, facecolor=C_SURFACE)
    plt.close(fig)
    print(f"Saved PNG chart to {png_path}")

    fig2 = go.Figure()
    if morning is not None:
        fig2.add_trace(go.Scatter(
            x=morning["bucket"], y=morning["mean"], mode="lines+markers", name="morning",
            line=dict(color=C_MORNING, width=2.5), marker=dict(size=6),
            hovertemplate="morning +%{x}min: %{y:.4f}<extra></extra>",
        ))
    if afternoon is not None:
        fig2.add_trace(go.Scatter(
            x=afternoon["bucket"], y=afternoon["mean"], mode="lines+markers", name="afternoon",
            line=dict(color=C_AFTERNOON, width=2.5), marker=dict(size=6),
            hovertemplate="afternoon +%{x}min: %{y:.4f}<extra></extra>",
        ))
    fig2.add_hline(y=1.0, line=dict(color=C_MUTED, width=1, dash="dot"))
    fig2.update_layout(
        title="Intraday ATM IV shape (pooled, 7 underlyings x 22-23 days)",
        xaxis_title="minutes since session open",
        yaxis_title="ATM IV indexed to day mean (1.0 = day avg)",
        plot_bgcolor=C_SURFACE, paper_bgcolor=C_SURFACE,
        font=dict(color=C_INK),
        xaxis=dict(gridcolor=C_GRID, zerolinecolor=C_AXIS),
        yaxis=dict(gridcolor=C_GRID, zerolinecolor=C_AXIS),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    html_path = out_dir / "intraday_iv_seasonality.html"
    fig2.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive HTML chart to {html_path}")


if __name__ == "__main__":
    main()
