"""Overnight overseas market-moves section of the daily WeChat push --
market moves only (index/asset levels + % change), no headlines: there's no
news-headline API wired into this codebase, and levels-only is simpler and
more reliable than trying to curate headlines.

Every feed here is a China-reachable Sina/BOC mirror (no VPN/proxy needed),
following the pattern already proven in the deleted-but-recovered
analysis/macro_model/shock.py (S&P500/Nasdaq via ak.index_us_stock_sina --
see its docstring: "direct/sina -- no proxy needed"). A real VIX isn't
available through any no-proxy akshare feed (checked index_global_hist_sina's
own symbol map -- no VIX entry); 50ETF QVIX is shown instead and labeled
explicitly as a domestic proxy, not relabeled as VIX.

Each feed is independently wrapped in try/except (same resilience pattern
updater.py uses per-underlying) so one failing source doesn't blank the
whole digest.

    python overseas_digest.py
"""
from __future__ import annotations

import akshare as ak
import pandas as pd


def _sina_us_change(symbol: str) -> tuple[float, float, pd.Timestamp] | None:
    """(last_close, pct_change_vs_prior_close, asof) for a Sina US index."""
    d = ak.index_us_stock_sina(symbol=symbol)
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    if len(d) < 2:
        return None
    last, prior = float(d["close"].iloc[-1]), float(d["close"].iloc[-2])
    return last, last / prior - 1, d["date"].iloc[-1]


def _wti_crude_change() -> tuple[float, float, pd.Timestamp] | None:
    d = ak.futures_foreign_hist(symbol="CL")
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["close"] > 0].sort_values("date")
    if len(d) < 2:
        return None
    last, prior = float(d["close"].iloc[-1]), float(d["close"].iloc[-2])
    return last, last / prior - 1, d["date"].iloc[-1]


def _usdcny_change() -> tuple[float, float, pd.Timestamp] | None:
    """BOC central parity (折算价, quoted as RMB per 100 USD) -- an official,
    no-VPN-needed proxy for USD/CNY. Not real-time offshore CNH, but close
    enough for an overnight-moves digest."""
    d = ak.currency_boc_safe()
    d["日期"] = pd.to_datetime(d["日期"])
    d = d[["日期", "美元"]].dropna().sort_values("日期")
    if len(d) < 2:
        return None
    last, prior = float(d["美元"].iloc[-1]) / 100, float(d["美元"].iloc[-2]) / 100
    return last, last / prior - 1, d["日期"].iloc[-1]


def _qvix_change() -> tuple[float, float, pd.Timestamp] | None:
    """50ETF QVIX -- China's own implied-vol index. Domestic proxy shown in
    place of the real VIX (not obtainable via any no-proxy akshare feed)."""
    d = ak.index_option_50etf_qvix()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date")
    if len(d) < 2:
        return None
    last, prior = float(d["close"].iloc[-1]), float(d["close"].iloc[-2])
    return last, last / prior - 1, d["date"].iloc[-1]


FEEDS = [
    ("标普500", ".INX", lambda: _sina_us_change(".INX")),
    ("纳斯达克", ".IXIC", lambda: _sina_us_change(".IXIC")),
    ("WTI原油", "CL", _wti_crude_change),
    ("美元/人民币（中间价）", None, _usdcny_change),
    ("50ETF期权QVIX（境内波动率指数，非VIX）", None, _qvix_change),
]


def build_overseas_section() -> str:
    lines = ["### 隔夜海外市场"]
    any_ok = False
    for label, _sym, fetch in FEEDS:
        try:
            result = fetch()
        except Exception as e:
            lines.append(f"{label}：获取失败（{type(e).__name__}）")
            continue
        if result is None:
            lines.append(f"{label}：数据不足")
            continue
        last, chg, asof = result
        any_ok = True
        lines.append(f"{label}：{last:,.2f}（{chg:+.2%}，截至{pd.Timestamp(asof).date()}）")
    if not any_ok:
        lines.append("（所有数据源均获取失败，请检查网络或稍后重试）")
    return "\n".join(lines)


if __name__ == "__main__":
    print(build_overseas_section())
