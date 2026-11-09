"""模型输出解析。

这是「自己实现 Runtime」的核心之一：不把决策权交给框架，而是把模型
的任意输出收敛成三种意图——调用工具 / 直接回答 / 无法解析。

支持两条协议：
1. 原生 function calling（tool_calls 字段）——主路径；
2. 文本协议（```json {"tool": ..., "arguments": ...}```）——兼容不支持
   function calling 的模型，也让「解析逻辑」本身可被单独测试。
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from .llm import LLMResponse
from .types import Decision, ToolCall

_THINK_RE = re.compile(
    r"<(thinking|thought|思考|reasoning)>(.*?)</\1>",
    re.IGNORECASE | re.DOTALL,
)
_FENCED_RE = re.compile(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", re.DOTALL)

_TOOL_KEYS = ("tool", "tool_name", "name")
_ARG_KEYS = ("arguments", "args", "parameters", "input", "action_input")
_FINAL_ACTIONS = {"final", "answer", "respond"}
_META_ACTIONS = {"tool", "call", "use", ""}
# 出现这些键，说明模型大概率是在描述一次工具调用，而不是在回答数据
_BROKEN_TOOL_KEYS = ('"tool":', '"tool_name":', '"arguments":', '"action_input":')


def _looks_like_broken_tool_call(body: str) -> bool:
    """判断一段解析不了的正文，是不是"想调工具但 JSON 坏了"。

    必须同时满足两件事才判 invalid：结构上像调用（以 `{` 开头，或带 json 代码块），
    并且出现了工具/参数键。否则会把普通的 JSON 回答误伤成解析失败。
    """
    text = (body or "").strip()
    if not text or not any(key in text for key in _BROKEN_TOOL_KEYS):
        return False
    if text.startswith("{"):
        return True
    return "```" in text and "json" in text.lower()


def extract_thinking(text: str) -> tuple[str, str]:
    """剥离思考标签，返回 (thinking, 剩余正文)。"""
    if not text:
        return "", ""
    parts = [m.group(2).strip() for m in _THINK_RE.finditer(text)]
    rest = _THINK_RE.sub("", text).strip()
    return "\n".join(p for p in parts if p), rest


def _strip_trailing_commas(raw: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", raw)


def _first_json_object(raw: str) -> str | None:
    """从混杂文本里截出第一个括号配平的 JSON 对象（跳过字符串内的括号）。"""
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None


def loads_lenient(raw: str) -> dict[str, Any] | None:
    """尽量把模型吐出来的东西读成 dict，失败返回 None。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return {}
    candidates = [text]
    stripped = _strip_trailing_commas(text)
    if stripped != text:
        candidates.append(stripped)
    first = _first_json_object(text)
    if first and first != text:
        candidates.append(first)
        candidates.append(_strip_trailing_commas(first))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value

    # 模型偶尔会输出 Python 字面量风格（单引号、True/False）
    for candidate in candidates:
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _coerce_arguments(value: Any) -> dict[str, Any] | None:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return loads_lenient(value)
    return None


def _decision_from_payload(payload: dict[str, Any], thinking: str, raw: str) -> Decision | None:
    """把文本协议里的 JSON 对象翻译成 Decision。"""
    action = str(payload.get("action", "")).lower()
    if action in _FINAL_ACTIONS:
        answer = payload.get("answer") or payload.get("content") or payload.get("final") or ""
        return Decision(
            kind="final",
            thinking=thinking,
            text=str(answer).strip(),
            raw=raw,
            protocol="text",
        )

    name = ""
    name_from = ""
    for key in _TOOL_KEYS:
        if payload.get(key):
            name = str(payload[key])
            name_from = key
            break

    args_key_present = any(key in payload for key in _ARG_KEYS)

    # "name" 也是普通数据里常见的键（比如 {"name": "小林"}）。
    # 只有同时带了参数键，才把它当成工具名，避免把正常 JSON 回答误判成调用。
    if name_from == "name" and not args_key_present:
        name = ""

    # ReAct 风格：{"thought": ..., "action": "weather", "action_input": {...}}
    # action 不是 final/tool 这类元动作，又带着参数，就把它当工具名。
    if not name and args_key_present and action not in _FINAL_ACTIONS | _META_ACTIONS:
        name = str(payload["action"])

    if name:
        args_value = None
        for key in _ARG_KEYS:
            if key in payload:
                args_value = payload[key]
                break
        args = _coerce_arguments(args_value)
        if args is None:
            return Decision(
                kind="invalid",
                thinking=thinking,
                raw=raw,
                protocol="text",
                error=f"工具 {name} 的 arguments 不是合法 JSON 对象",
            )
        return Decision(
            kind="tool_calls",
            thinking=thinking,
            raw=raw,
            protocol="text",
            tool_calls=[
                ToolCall(
                    id="text_call_0",
                    name=name,
                    arguments=args,
                    raw_arguments=json.dumps(args, ensure_ascii=False),
                )
            ],
        )
    return None


def parse_response(response: LLMResponse) -> Decision:
    """把一次 LLM 响应解析成 Decision。"""
    raw = response.content or ""
    tag_thinking, body = extract_thinking(raw)
    thinking = "\n".join(x for x in (response.reasoning or "", tag_thinking) if x).strip()

    if response.tool_calls:
        calls: list[ToolCall] = []
        for index, call in enumerate(response.tool_calls):
            name = call.get("name") or ""
            raw_args = call.get("arguments")
            args = _coerce_arguments(raw_args)
            if not name:
                return Decision(kind="invalid", thinking=thinking, raw=raw, error="工具调用缺少 name")
            if args is None:
                return Decision(
                    kind="invalid",
                    thinking=thinking,
                    raw=raw,
                    error=f"工具 {name} 的参数无法解析：{raw_args!r}",
                )
            calls.append(
                ToolCall(
                    id=call.get("id") or f"call_{index}",
                    name=name,
                    arguments=args,
                    raw_arguments=raw_args if isinstance(raw_args, str) else json.dumps(args, ensure_ascii=False),
                )
            )
        return Decision(kind="tool_calls", thinking=thinking, raw=raw, tool_calls=calls, protocol="native")

    if not body:
        return Decision(kind="invalid", thinking=thinking, raw=raw, error="模型返回为空")

    # 文本协议：优先看围栏代码块，其次看正文里的第一个 JSON 对象
    for match in _FENCED_RE.finditer(body):
        payload = loads_lenient(match.group(1))
        if payload:
            decision = _decision_from_payload(payload, thinking, raw)
            if decision:
                return decision

    payload = loads_lenient(body)
    if payload:
        decision = _decision_from_payload(payload, thinking, raw)
        if decision:
            return decision

    # 正文非空、但明显是"坏掉的工具调用"时，不能当最终答案直接给用户，
    # 要交给 runtime 走修复重试（例如被截断的 {"tool": ..., "arguments": ...}）。
    if _looks_like_broken_tool_call(body):
        return Decision(
            kind="invalid",
            thinking=thinking,
            raw=raw,
            protocol="text",
            error="输出看起来是工具调用，但 JSON 不完整或格式错误",
        )

    return Decision(kind="final", thinking=thinking, text=body.strip(), raw=raw, protocol="fallback")
