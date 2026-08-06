"""Orchestrator + Task Scheduler entry point for the daily WeChat push:
combines push_alerts.py's options_lab alert section with overseas_digest.py's
overnight market-moves section into one message and sends it via PushPlus.

Run manually:      python daily_push.py
Scheduled (daily):  run_daily_push.bat via Windows Task Scheduler, ~8:30am
                     China time, weekdays only.

Credentials come from pushplus_config.json (gitignored, sibling to this
file): {"token": "...", "topic": "..."}.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import requests

import data_store as ds
import overseas_digest
import push_alerts
import updater

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "pushplus_config.json"
LOG_PATH = HERE / "daily_push.log"
PUSHPLUS_URL = "https://www.pushplus.plus/send"


def _log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_message() -> str:
    today = datetime.now().date().isoformat()
    # Yesterday's SSE Greeks/IV file is often not published yet when the
    # 18:00 update_data.bat run pulls the chain, leaving that date's IV
    # column blank; alerts_core then anchors "today" one trading day further
    # back than the data actually available by push time (~8:30am, well
    # after SSE catches up). Retry the backfill here so the push reflects
    # the latest date, not a stale active/near-trigger read.
    updater.backfill_missing_greeks(list(ds.UNDERLYINGS) + list(ds.EXT_LIVE_UNDERLYINGS))
    alerts_section = push_alerts.build_alerts_section()
    overseas_section = overseas_digest.build_overseas_section()
    return f"## Options Lab 每日推送 {today}\n\n{alerts_section}\n\n{overseas_section}"


def send(content: str) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload = {
        "token": config["token"],
        "topic": config["topic"],
        "title": f"Options Lab 每日推送 {datetime.now().date().isoformat()}",
        "content": content,
        "template": "markdown",
    }
    resp = requests.post(PUSHPLUS_URL, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("code") != 200:
        raise RuntimeError(f"PushPlus rejected the push: {result}")


def main() -> int:
    try:
        content = build_message()
    except Exception as e:
        _log(f"FAILED building message: {type(e).__name__}: {e}")
        return 1
    try:
        send(content)
    except Exception as e:
        _log(f"FAILED sending push: {type(e).__name__}: {e}")
        return 1
    _log("OK: push sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
