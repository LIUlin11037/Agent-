"""LLM 客户端。

只依赖官方 SDK 的 HTTP 调用能力，不依赖任何 Agent 框架。
对外暴露统一的 `BaseLLM.chat()`，让主循环与具体厂商解耦；
测试用 `FakeLLM` 按脚本回放，完全离线。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .errors import LLMError
from .types import Message


@dataclass
class LLMResponse:
    """归一化后的模型响应，屏蔽不同厂商的字段差异。"""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    latency_ms: float = 0.0


class BaseLLM:
    name = "base"

    def chat(self, messages: Sequence[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        raise NotImplementedError


class DeepSeekLLM(BaseLLM):
    """DeepSeek（OpenAI 兼容协议）。也适用于任何兼容端点。"""

    name = "deepseek"

    def __init__(self, config: Config) -> None:
        config.require_llm()
        from openai import OpenAI  # 延迟导入：离线测试无需安装/初始化

        self.config = config
        self.model = config.model
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.llm_timeout,
            max_retries=0,  # 重试逻辑自己控制，便于记录 trace
        )

    def chat(self, messages: Sequence[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        payload = [m.to_openai() for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": 0.2,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(self.config.llm_retries + 1):
            started = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # 网络 / 限流 / 服务端错误
                last_error = exc
                if attempt >= self.config.llm_retries:
                    break
                time.sleep(0.8 * (2**attempt))
                continue

            latency = (time.perf_counter() - started) * 1000
            choice = resp.choices[0]
            msg = choice.message
            raw_calls = getattr(msg, "tool_calls", None) or []
            calls = []
            for call in raw_calls:
                calls.append(
                    {
                        "id": getattr(call, "id", "") or f"call_{len(calls)}",
                        "name": getattr(call.function, "name", ""),
                        "arguments": getattr(call.function, "arguments", "") or "{}",
                    }
                )
            usage_obj = getattr(resp, "usage", None)
            usage = {}
            if usage_obj is not None:
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                    "total_tokens": getattr(usage_obj, "total_tokens", 0),
                }
            return LLMResponse(
                content=getattr(msg, "content", "") or "",
                tool_calls=calls,
                reasoning=getattr(msg, "reasoning_content", "") or "",
                usage=usage,
                raw=str(getattr(msg, "content", "") or ""),
                latency_ms=latency,
            )

        raise LLMError(f"LLM 调用失败：{last_error}", retryable=False)


class FakeLLM(BaseLLM):
    """脚本化假模型，用于离线测试与本地演示。

    script 里每一项可以是：
      - dict：{"content": ..., "tool_calls": [...], "reasoning": ...}
      - Exception：抛出，模拟接口异常
      - callable(messages, tools) -> dict：按当前上下文动态生成
    脚本耗尽后抛 AssertionError，避免测试悄悄跑偏。
    """

    name = "fake"

    def __init__(self, script: Iterable[Any], *, model: str = "fake-model") -> None:
        self.script: list[Any] = list(script)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    @property
    def remaining(self) -> int:
        return len(self.script)

    def chat(self, messages: Sequence[Message], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        snapshot = [m.to_openai() for m in messages]
        self.calls.append({"messages": snapshot, "tools": tools})
        if not self.script:
            raise AssertionError(
                f"FakeLLM 脚本已耗尽，但仍发起了第 {len(self.calls)} 次调用。"
                "请检查测试脚本是否覆盖了所有循环步数。"
            )
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(snapshot, tools)
        if isinstance(item, str):
            item = {"content": item}
        return self._build(item)

    def _build(self, item: dict[str, Any]) -> LLMResponse:
        calls = []
        for index, call in enumerate(item.get("tool_calls") or []):
            args = call.get("arguments", {})
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False)
            calls.append(
                {
                    "id": call.get("id") or f"call_{index}",
                    "name": call.get("name", ""),
                    "arguments": args,
                }
            )
        return LLMResponse(
            content=item.get("content", "") or "",
            tool_calls=calls,
            reasoning=item.get("reasoning", "") or "",
            usage=item.get("usage", {}) or {},
            raw=item.get("content", "") or "",
        )


def build_llm(config: Config) -> BaseLLM:
    return DeepSeekLLM(config)
