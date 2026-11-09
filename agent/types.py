"""贯穿全链路的数据结构。

刻意保持「可 JSON 序列化」，方便落库、写 trace、以及喂给 LLM。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """一条对话消息。assistant 消息可携带 tool_calls，tool 消息回填结果。"""

    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    reasoning: str | None = None
    seq: int = 0
    created_at: float = field(default_factory=time.time)

    def to_openai(self) -> dict[str, Any]:
        """转成 OpenAI 兼容协议的 message。"""
        if self.role == "assistant":
            payload: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
            if self.tool_calls:
                payload["tool_calls"] = self.tool_calls
            return payload
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.content,
            }
        return {"role": self.role, "content": self.content}

    def to_record(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
            "reasoning": self.reasoning,
            "seq": self.seq,
            "created_at": self.created_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Message:
        return cls(
            role=record["role"],
            content=record.get("content") or "",
            name=record.get("name"),
            tool_call_id=record.get("tool_call_id"),
            tool_calls=record.get("tool_calls"),
            reasoning=record.get("reasoning"),
            seq=record.get("seq", 0),
            created_at=record.get("created_at", time.time()),
        )

    def brief(self, limit: int = 120) -> str:
        text = (self.content or "").replace("\n", " ")
        if len(text) > limit:
            text = text[:limit] + "…"
        return f"[{self.role}] {text}"


@dataclass
class ToolCall:
    """模型发起的一次工具调用。arguments 已解析成 dict。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass
class Decision:
    """解析后的模型意图：要么调用工具，要么给出最终答案。"""

    kind: Literal["final", "tool_calls", "invalid"]
    thinking: str = ""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None
    raw: str = ""
    protocol: str = "native"  # native | text | fallback

    @property
    def has_tool_calls(self) -> bool:
        return self.kind == "tool_calls" and bool(self.tool_calls)


@dataclass
class ToolResult:
    ok: bool
    content: str
    error: str | None = None
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> str:
        """回填给模型的文本。失败时也要给出可读信息，让模型有机会自愈。"""
        if self.ok:
            return self.content
        return f"[工具执行失败] {self.error}\n请根据错误信息修正参数或改用其他工具；如果无法完成，请如实告知用户。"


@dataclass
class ChatResult:
    """一次用户输入的完整结果。"""

    session_id: str
    reply: str
    steps: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str = "final"  # final | max_steps | error
    usage: dict[str, Any] = field(default_factory=dict)
    trace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reply": self.reply,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "stopped_reason": self.stopped_reason,
            "usage": self.usage,
            "trace_path": self.trace_path,
        }


@dataclass
class Session:
    """一个会话窗口。messages 只保留未被压缩的部分。"""

    id: str
    user_id: str
    title: str = ""
    summary: str = ""
    compressed_upto: int = 0
    facts: dict[str, Any] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def last_tool_call(self) -> dict[str, Any] | None:
        value = self.facts.get("last_tool_call")
        return value if isinstance(value, dict) else None

    def set_last_tool_call(self, name: str, arguments: dict[str, Any], summary: str) -> None:
        self.facts["last_tool_call"] = {
            "name": name,
            "arguments": arguments,
            "result_summary": summary,
        }

    def append(self, message: Message) -> Message:
        message.seq = (self.messages[-1].seq + 1) if self.messages else 1
        self.messages.append(message)
        self.updated_at = time.time()
        return message

    def dump_facts(self) -> str:
        return json.dumps(self.facts, ensure_ascii=False)
