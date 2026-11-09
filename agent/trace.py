"""执行轨迹。每条事件一行 JSON，方便 grep、回放和排障。

记录粒度刻意做得比较细：LLM 调用（含耗时与 token）、解析结果、工具调用与
返回、压缩动作、异常。面试时可以直接拿 trace 讲"这一步为什么这么走"。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, trace_dir: str | Path, session_id: str, *, echo: bool = False) -> None:
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.path = self.dir / f"{session_id}.jsonl"
        self.echo = echo
        # 同一会话的多次对话会复用同一个 trace 文件，turn 必须接着历史往下数，
        # 否则每条记录都显示 turn=1，按轮复盘时无法区分。
        self.turn = self._existing_turns()
        self.step = 0

    def _existing_turns(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        try:
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if '"event": "user_input"' in line:
                        count += 1
        except OSError:
            return 0
        return count

    def new_turn(self) -> int:
        self.turn += 1
        self.step = 0
        return self.turn

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def log(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "epoch": round(time.time(), 3),
            "session": self.session_id,
            "turn": self.turn,
            "step": self.step,
            "event": event,
        }
        record.update(fields)
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # trace 写失败绝不影响主流程
        if self.echo:
            print(f"  · [{event}] " + ", ".join(f"{k}={v}" for k, v in fields.items() if k != "raw"))
        return record

    def read(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-limit:]


def load_trace(trace_dir: str | Path, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
    return Tracer(trace_dir, session_id).read(limit)
