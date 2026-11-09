"""上下文装配与压缩。

装配顺序（稳定在前、易变在后）：
  1. system   —— 角色、工具规范、输出格式。启动即固定。
  2. summary  —— 早期对话的滚动摘要，压缩时才重写。
  3. state    —— 结构化状态：时间、待办、上次工具调用、已知事实。永不压缩。
  4. history  —— 最近若干轮原文（含本轮用户输入）。

压缩策略：按"轮"切分（user 开头），保留最近 keep_recent_turns 轮原文，
更早的交给 LLM 压成摘要，并推进水位线 compressed_upto。
原始消息仍留在库里，只是不再进入上下文——既省 token 又可审计。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

from .config import Config
from .llm import BaseLLM
from .session import SessionStore, todo_owner
from .trace import Tracer
from .types import Message, Session

SYSTEM_PROMPT = """你是一个可靠、克制的中文助手，运行在一个自行实现的最小 Agent Runtime 上。

工作方式：
1. 能直接回答的就直接回答，不要为了调用工具而调用工具。
2. 需要外部信息（算术、资料检索、天气、待办、文档）时调用对应工具；一次可以调用多个。
3. 拿到工具结果后判断是否还需要继续调用；信息足够时立即给出最终回答。
4. 绝不编造工具结果。工具报错时，根据错误信息修正参数重试，或如实告知用户。

工具使用规范：
- 任何算术都用 calculator，不要心算。
- 查公司制度/流程先用 search，需要全文再用 read_docs。
- 用户说「记一下 / 待办 / 提醒我」用 todo 的 add；说「做完了」用 done。
- 用户追问省略了主语或地点时，参考【会话状态】里的「上次工具调用」补全参数。

输出要求：
- 简体中文，结论先行，简洁。
- 若当前接口不支持 function calling，请只输出一个 JSON 对象：
  {"tool": "工具名", "arguments": {...}}
  需要直接回答时，正常输出文本即可。"""

COMPRESS_SYSTEM = (
    "你是一个对话压缩器。把给定的历史对话合并进已有摘要，输出一份新的摘要。"
    "只输出摘要正文，不要解释、不要标题。"
)

COMPRESS_TEMPLATE = """已有摘要：
{summary}

需要并入摘要的新对话：
{chunk}

要求：
1. 合并去重，输出一份新的摘要，不超过 400 字。
2. 必须保留：用户的目标与偏好、已确认的事实与数字、已完成事项、未完成或待跟进事项、重要工具调用的结论。
3. 丢弃寒暄、重复内容和被否定的中间尝试。
4. 用短句和分号连接，不要分点标题。"""


def estimate_tokens(text: str) -> int:
    """粗估 token：中文按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return int(cjk + (len(text) - cjk) * 0.3) + 1


def estimate_messages(messages: Sequence[Message]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(message.content) + 4
        if message.tool_calls:
            total += estimate_tokens(json.dumps(message.tool_calls, ensure_ascii=False))
    return total


def group_turns(messages: Sequence[Message]) -> list[list[Message]]:
    """按 user 消息切分轮次，保证 assistant.tool_calls 与 tool 结果不被拆开。"""
    turns: list[list[Message]] = []
    current: list[Message] = []
    for message in messages:
        if message.role == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


class ContextBuilder:
    def __init__(
        self,
        config: Config,
        *,
        llm: BaseLLM | None = None,
        store: SessionStore | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.store = store
        self.tracer = tracer

    # ------------------------------------------------------------------
    def active_messages(self, session: Session) -> list[Message]:
        return [m for m in session.messages if m.seq > session.compressed_upto]

    def build(self, session: Session) -> list[Message]:
        self.maybe_compress(session)

        messages: list[Message] = [Message(role="system", content=SYSTEM_PROMPT)]
        if session.summary:
            messages.append(
                Message(role="system", content="【历史对话摘要】\n" + session.summary.strip())
            )
        state = self._state_block(session)
        if state:
            messages.append(Message(role="system", content=state))
        messages.extend(self.active_messages(session))
        return messages

    # ------------------------------------------------------------------
    def _state_block(self, session: Session) -> str:
        lines = ["【会话状态】"]
        now = datetime.now()
        weekday = "一二三四五六日"[now.weekday()]
        lines.append(f"当前时间：{now.strftime('%Y-%m-%d %H:%M')}（周{weekday}）")
        lines.append(f"用户：{session.user_id}")

        if self.store is not None:
            owner = todo_owner(session.user_id, session.id, self.config.todo_scope)
            items = self.store.list_todos(owner, status="open")
            if items:
                preview = "；".join(f"#{item['id']} {item['content']}" for item in items[:8])
                suffix = f" 等 {len(items)} 项" if len(items) > 8 else ""
                lines.append(f"未完成待办（{len(items)} 项）：{preview}{suffix}")

        last_call = session.last_tool_call
        if last_call:
            args = json.dumps(last_call.get("arguments", {}), ensure_ascii=False)
            summary = str(last_call.get("result_summary", ""))[:120]
            lines.append(f"上次工具调用：{last_call.get('name')}({args}) → {summary}")

        others = {k: v for k, v in session.facts.items() if k != "last_tool_call"}
        if others:
            lines.append("已知事实：" + json.dumps(others, ensure_ascii=False))

        return "\n".join(lines) if len(lines) > 2 else ""

    # ------------------------------------------------------------------
    def _needs_compression(self, session: Session) -> bool:
        active = self.active_messages(session)
        used = estimate_messages(active) + estimate_tokens(SYSTEM_PROMPT)
        if session.summary:
            used += estimate_tokens(session.summary)
        return used > self.config.max_context_tokens * self.config.compress_threshold

    def maybe_compress(self, session: Session) -> bool:
        if self.llm is None or self.store is None:
            return False
        if not self._needs_compression(session):
            return False

        active = self.active_messages(session)
        turns = group_turns(active)
        if len(turns) <= 1:
            return False

        keep_n = min(self.config.keep_recent_turns, len(turns) - 1)
        old_turns = turns[: len(turns) - keep_n]
        if not old_turns:
            return False

        chunk = self._render_chunk(old_turns)
        if not chunk.strip():
            return False

        prompt = COMPRESS_TEMPLATE.format(summary=session.summary or "（无）", chunk=chunk)
        try:
            response = self.llm.chat(
                [
                    Message(role="system", content=COMPRESS_SYSTEM),
                    Message(role="user", content=prompt),
                ]
            )
            new_summary = (response.content or "").strip()
        except Exception as exc:  # 压缩失败不能拖垮主流程
            new_summary = ""
            if self.tracer:
                self.tracer.log("compress_failed", error=str(exc))

        if not new_summary:
            new_summary = self._mechanical_summary(session.summary, chunk)

        compressed_upto = old_turns[-1][-1].seq
        self.store.save_summary(session, new_summary, compressed_upto)
        if self.tracer:
            self.tracer.log(
                "compress",
                turns=len(old_turns),
                kept_turns=keep_n,
                compressed_upto=compressed_upto,
                summary_chars=len(new_summary),
            )
        return True

    @staticmethod
    def _render_chunk(turns: list[list[Message]], per_message: int = 400, total: int = 6000) -> str:
        lines: list[str] = []
        for turn in turns:
            for message in turn:
                text = (message.content or "").replace("\n", " ").strip()
                if len(text) > per_message:
                    text = text[:per_message] + "…"
                if message.role == "assistant" and message.tool_calls:
                    names = ", ".join(
                        str(call.get("function", {}).get("name", "?")) for call in message.tool_calls
                    )
                    text = f"{text} [调用工具: {names}]".strip()
                if text:
                    lines.append(f"{message.role}: {text}")
        chunk = "\n".join(lines)
        return chunk[:total]

    @staticmethod
    def _mechanical_summary(existing: str, chunk: str) -> str:
        """兜底：LLM 不可用时退化为截断，保证上下文不会无限增长。"""
        head = (existing + " " if existing else "").strip()
        merged = f"{head} {chunk}".strip()
        return merged[-800:]
