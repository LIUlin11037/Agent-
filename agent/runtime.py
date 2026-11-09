"""Agent 主循环。

这是整个项目唯一"知道怎么转圈"的地方，不依赖任何 Agent 框架：

    用户输入 → 装配上下文 → 调 LLM → 解析意图
        ├─ 最终答案 → 落库 → 返回
        └─ 工具调用 → 校验执行 → 结果回填 → 回到装配上下文

终止条件有三类：模型给出最终答案、达到 max_steps、出现不可恢复错误。
"""

from __future__ import annotations

import json
from typing import Any

from .config import Config
from .context import ContextBuilder, estimate_messages
from .errors import LLMError
from .llm import BaseLLM, build_llm
from .parser import parse_response
from .session import SessionStore
from .tools import ToolContext, execute_tool, openai_tools, tool_names
from .trace import Tracer
from .types import ChatResult, Message, Session, ToolResult

REPAIR_HINT = (
    "[系统提示] 你上一条输出无法被解析（{error}）。"
    "请重新作答：要么直接给出最终回答，要么严格输出一个 JSON 对象 "
    '{{"tool": "工具名", "arguments": {{...}}}}。'
)

REPEAT_HINT = (
    "[系统提示] 你刚刚已经用完全相同的参数调用过 {name}，结果不会变化。"
    "请基于已有结果作答；如果信息不足，就如实说明缺少什么。"
)

# 同一 (工具, 参数) 允许执行的次数上限。达到之前用提示纠正，达到即拦截。
REPEAT_LIMIT = 3


class AgentRuntime:
    def __init__(
        self,
        config: Config | None = None,
        *,
        llm: BaseLLM | None = None,
        store: SessionStore | None = None,
        echo_trace: bool = False,
    ) -> None:
        self.config = config or Config.from_env()
        self.store = store or SessionStore(self.config.db_path)
        self.llm = llm or build_llm(self.config)
        self.context = ContextBuilder(self.config, llm=self.llm, store=self.store)
        self.echo_trace = echo_trace

    # ------------------------------------------------------------------
    def chat(self, user_id: str, text: str, session_id: str | None = None) -> ChatResult:
        text = (text or "").strip()
        if not text:
            raise ValueError("用户输入不能为空")

        session = self.store.ensure_session(user_id, session_id)
        tracer = Tracer(self.config.trace_dir, session.id, echo=self.echo_trace)
        tracer.new_turn()
        # 让压缩等发生在 ContextBuilder 里的事件也能写进本次对话的 trace
        self.context.tracer = tracer

        self.store.append_message(session, Message(role="user", content=text))
        if not session.title:
            session.title = text[:24]
            self.store.rename_session(session.id, session.title)
        tracer.log("user_input", text=text, session=session.id, history=len(session.messages))

        tool_records: list[dict[str, Any]] = []
        usage_total: dict[str, int] = {}
        repaired = False
        # 按 (工具, 参数) 指纹累计重复次数。用 dict 而不是「上一个 key」，
        # 否则模型一步并行调用两个工具时，计数会被另一个 key 反复重置，死循环检测失效。
        repeat_counts: dict[str, int] = {}
        hinted: set[str] = set()

        for step in range(1, self.config.max_steps + 1):
            tracer.next_step()
            messages = self.context.build(session)
            est = estimate_messages(messages)
            tracer.log("llm_request", step=step, messages=len(messages), est_tokens=est)

            try:
                response = self.llm.chat(messages, tools=openai_tools())
            except LLMError as exc:
                tracer.log("llm_error", error=str(exc))
                return self._finish_with_error(session, tracer, f"模型服务暂时不可用：{exc}", tool_records)
            except Exception as exc:  # 防御性兜底
                tracer.log("llm_error", error=f"{type(exc).__name__}: {exc}")
                return self._finish_with_error(session, tracer, f"调用模型时出错：{exc}", tool_records)

            for key, value in (response.usage or {}).items():
                if isinstance(value, int):
                    usage_total[key] = usage_total.get(key, 0) + value

            decision = parse_response(response)
            tracer.log(
                "llm_response",
                content=(response.content or "")[:200],
                tool_calls=[call.get("name") for call in response.tool_calls],
                latency_ms=round(response.latency_ms, 1),
                usage=response.usage,
            )
            tracer.log(
                "parse",
                kind=decision.kind,
                protocol=decision.protocol,
                thinking=decision.thinking[:200],
                tool_calls=[call.name for call in decision.tool_calls],
                error=decision.error,
                latency_ms=round(response.latency_ms, 1),
                usage=response.usage,
            )

            if decision.kind == "final":
                reply = decision.text.strip() or "（模型没有返回内容）"
                self.store.append_message(
                    session,
                    Message(role="assistant", content=reply, reasoning=decision.thinking or None),
                )
                tracer.log("final", step=step, chars=len(reply))
                return ChatResult(
                    session_id=session.id,
                    reply=reply,
                    steps=step,
                    tool_calls=tool_records,
                    stopped_reason="final",
                    usage=usage_total,
                    trace_path=str(tracer.path),
                )

            if decision.kind == "invalid":
                if not repaired:
                    repaired = True
                    tracer.log("repair", reason=decision.error)
                    self.store.append_message(
                        session,
                        Message(role="user", content=REPAIR_HINT.format(error=decision.error or "未知原因")),
                    )
                    continue
                return self._finish_with_error(
                    session, tracer, "抱歉，我没能理解要执行的操作，可以换个说法再试一次吗？", tool_records
                )

            # ---- 工具调用 ----
            # 先算指纹和"预计执行次数"：写工具不能等到执行完才发现是第三次重复，
            # 否则 todo(add) 会真的落三条数据。重复达到上限时，在执行前直接拦截。
            fingerprints = [
                f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
                for call in decision.tool_calls
            ]
            projected = dict(repeat_counts)
            blocked_key: str | None = None
            for key in fingerprints:
                projected[key] = projected.get(key, 0) + 1
                if projected[key] >= REPEAT_LIMIT:
                    blocked_key = key
                    break

            if blocked_key:
                name = blocked_key.split(":", 1)[0]
                tracer.log(
                    "loop_guard",
                    name=name,
                    repeats=projected[blocked_key],
                    blocked_before_execute=True,
                )
                reply = (
                    f"我准备第 {projected[blocked_key]} 次用相同参数调用 {name}，结果不会变化，"
                    "所以这次没有再执行。可以补充一些信息，或者换个问法。"
                )
                self.store.append_message(session, Message(role="assistant", content=reply))
                return ChatResult(
                    session_id=session.id,
                    reply=reply,
                    steps=step,
                    tool_calls=tool_records,
                    stopped_reason="loop_guard",
                    usage=usage_total,
                    trace_path=str(tracer.path),
                )

            self._append_assistant_tool_calls(session, decision)

            for call, key in zip(decision.tool_calls, fingerprints, strict=True):
                repeat_counts[key] = repeat_counts.get(key, 0) + 1

                tracer.log("tool_call", name=call.name, arguments=call.arguments, step=step)
                result = execute_tool(
                    call.name,
                    call.arguments,
                    ToolContext(
                        session_id=session.id,
                        user_id=session.user_id,
                        store=self.store,
                        config=self.config,
                    ),
                )
                tracer.log(
                    "tool_result",
                    name=call.name,
                    ok=result.ok,
                    latency_ms=round(result.latency_ms, 2),
                    preview=result.content[:200] if result.ok else result.error,
                )

                self._append_tool_observation(session, call, result, decision.protocol)

                tool_records.append(
                    {
                        "name": call.name,
                        "arguments": call.arguments,
                        "ok": result.ok,
                        "output": result.content[:500] if result.ok else result.error,
                        "latency_ms": round(result.latency_ms, 2),
                    }
                )

                if result.ok:
                    summary = " ".join(result.content.split())[:200]
                    session.set_last_tool_call(call.name, call.arguments, summary)
                else:
                    session.set_last_tool_call(call.name, call.arguments, f"失败：{result.error}")

            self.store.save_facts(session)

            worst_key, worst_count = max(repeat_counts.items(), key=lambda item: item[1], default=("", 0))

            if worst_count == REPEAT_LIMIT - 1 and worst_key not in hinted:
                name = worst_key.split(":", 1)[0]
                hinted.add(worst_key)
                tracer.log("repeat_hint", name=name)
                self.store.append_message(
                    session, Message(role="user", content=REPEAT_HINT.format(name=name))
                )

        tracer.log("max_steps", steps=self.config.max_steps)
        reply = (
            f"我在本轮已经调用了 {self.config.max_steps} 步工具仍未得出结论，先停下来避免无限循环。"
            "可以把问题拆小一点，或者补充更多信息。"
        )
        self.store.append_message(session, Message(role="assistant", content=reply))
        return ChatResult(
            session_id=session.id,
            reply=reply,
            steps=self.config.max_steps,
            tool_calls=tool_records,
            stopped_reason="max_steps",
            usage=usage_total,
            trace_path=str(tracer.path),
        )

    # ------------------------------------------------------------------
    def _append_assistant_tool_calls(self, session: Session, decision) -> None:
        if decision.protocol == "native":
            payload = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in decision.tool_calls
            ]
            message = Message(
                role="assistant",
                content=decision.text or "",
                tool_calls=payload,
                reasoning=decision.thinking or None,
            )
        else:
            # 文本协议下没有真正的 tool_call_id，原样保留模型的输出作为 assistant 消息
            message = Message(
                role="assistant",
                content=decision.raw or decision.text,
                reasoning=decision.thinking or None,
            )
        self.store.append_message(session, message)

    def _append_tool_observation(self, session: Session, call, result: ToolResult, protocol: str) -> None:
        observation = result.to_observation()
        if protocol == "native":
            message = Message(
                role="tool",
                content=observation,
                tool_call_id=call.id,
                name=call.name,
            )
        else:
            message = Message(role="user", content=f"[工具 {call.name} 返回]\n{observation}")
        self.store.append_message(session, message)

    def _finish_with_error(
        self, session: Session, tracer: Tracer, reply: str, tool_records: list[dict[str, Any]]
    ) -> ChatResult:
        self.store.append_message(session, Message(role="assistant", content=reply))
        tracer.log("abort", reply=reply[:120])
        return ChatResult(
            session_id=session.id,
            reply=reply,
            steps=tracer.step,
            tool_calls=tool_records,
            stopped_reason="error",
            trace_path=str(tracer.path),
        )

    # ------------------------------------------------------------------
    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        return self.store.list_sessions(user_id)

    def history(self, session_id: str, user_id: str | None = None) -> list[Message]:
        """读取会话原文。传 user_id 时会校验归属，防止跨窗口/跨用户窥探。"""
        return self.store.get_session(session_id, user_id=user_id).messages

    def close(self) -> None:
        self.store.close()

    @staticmethod
    def available_tools() -> list[str]:
        return tool_names()
