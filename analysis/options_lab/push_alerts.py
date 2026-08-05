"""Builds the Chinese-language options_lab alert section of the daily WeChat
push. Only includes alerts that are currently active or close to triggering
(see alerts_core.near_trigger_detail's NEAR_BAND_* constants) -- not all ~15
tracked rows every day. daily_push.py owns actually sending the message;
this module only builds text, so it's independently testable:

    python push_alerts.py
"""
from __future__ import annotations

import alerts_core as ac
import dashboard_data as DD


def _fmt_pct(x: float) -> str:
    return f"{x:+.1%}"


def _fmt_yuan(x: float) -> str:
    return f"{x:+,.0f}元"


def _active_block(row: dict, pos: dict) -> str:
    label = ac.ALERT_LABELS_ZH.get((row["side"], row["strategy_idx"]), row["strategy"])
    lines = [f"🔔 {label} · {pos['name']}",
             f"开仓日期：{pos['entry_date'].date()}",
             f"净盈亏：{_fmt_pct(pos['pnl_pct'])}（{_fmt_yuan(pos['pnl_yuan'])}）"]
    if "stop_loss_pct" in pos:
        lines.append(f"止损线：亏损达到 {_fmt_pct(pos['stop_loss_pct'])}（约{_fmt_yuan(pos['stop_loss_yuan'])}）")
    if "profit_target_pct" in pos:
        lines.append(f"止盈线：盈利达到 {_fmt_pct(pos['profit_target_pct'])}（约{_fmt_yuan(pos['profit_target_yuan'])}）")
    end_note = "（已达最长持有期）" if pos["max_hold_reached"] else ""
    lines.append(f"策略结束日：{pos['end_date'].date()}{end_note}")
    return "\n".join(lines)


def _near_block(row: dict, near: dict) -> str:
    label = ac.ALERT_LABELS_ZH.get((row["side"], row["strategy_idx"]), row["strategy"])
    parts = [f"⚪ {label} · {row['instrument']}（接近触发）"]
    if near.get("current_pct") is not None:
        parts.append(f"当前指标值：{near['current_pct']:.0%} | 触发阈值：{near['threshold_pct']:.0%}")
    if "current_vrp" in near:
        parts.append(f"当前VRP：{near['current_vrp']:+.1%} | 触发阈值：{near['threshold_vrp']:+.0%}")
    if near.get("current_raw") is not None:
        parts.append(f"当前IV：{near['current_raw']:.2%} | 触发所需IV：{near['trigger_raw']:.2%}")
    if "ok_spike" in near:
        parts.append(f"IV跳升：{'已满足' if near['ok_spike'] else '未满足'} | VRP：{'已满足' if near['ok_vrp'] else '未满足'}")
        if near.get("iv_now") is not None:
            parts.append(f"当前IV：{near['iv_now']:.2%} | 跳升所需IV：{near['spike_needed']:.2%}")
        if near.get("vrp_yst") is not None:
            extra = f" | 跳升所需VRP对应IV：{near['iv_needed_yst']:.2%}" if near.get("iv_needed_yst") is not None else ""
            parts.append(f"昨日VRP：{near['vrp_yst']:+.1%}{extra}")
    if "iv_pct_now" in near:
        parts.append(f"当前iv_pct：{near['iv_pct_now']:.0%}（阈值{near['iv_pct_threshold']:.0%}） | "
                      f"当前VRP：{near['vrp_now']:+.1%}（阈值{near['vrp_threshold']:+.0%}）")
    return "\n".join(parts)


def build_alerts_section(chain_loader=None) -> str:
    """Chinese markdown section for the options_lab alerts, ready to concat
    into the daily push message."""
    idx = DD.load_series("idx_sh")
    rows = ac.evaluate_summary(idx, chain_loader=chain_loader)

    active_blocks, near_blocks = [], []
    for row in rows:
        if row["active"]:
            positions = ac.position_detail(row, chain_loader=chain_loader, idx_sh_series=idx)
            for pos in (positions or []):
                active_blocks.append(_active_block(row, pos))
        else:
            near = ac.near_trigger_detail(row)
            if near:
                near_blocks.append(_near_block(row, near))

    if not active_blocks and not near_blocks:
        return "### 期权信号\n今日无预警触发，也无接近触发的信号。"

    sections = ["### 期权信号"]
    if active_blocks:
        sections.append("**已触发：**\n\n" + "\n\n".join(active_blocks))
    if near_blocks:
        sections.append("**接近触发：**\n\n" + "\n\n".join(near_blocks))
    return "\n\n".join(sections)


if __name__ == "__main__":
    print(build_alerts_section())
