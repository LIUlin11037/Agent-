"""测试辅助工具。"""

from __future__ import annotations

from agent import FakeLLM
from agent.context import COMPRESS_SYSTEM
from agent.llm import LLMResponse


class SummaryLLM(FakeLLM):
    """压缩请求返回固定摘要，其余请求按脚本回放。

    压缩发生在装配上下文时，会插到正常对话调用之间，因此单独识别，
    避免打乱脚本顺序。
    """

    def __init__(self, script, summary: str = "【摘要】用户此前问过天气，并记录了待办。") -> None:
        super().__init__(script)
        self.summary = summary
        self.compress_calls = 0

    def chat(self, messages, tools=None):
        if messages and messages[0].role == "system" and messages[0].content == COMPRESS_SYSTEM:
            self.compress_calls += 1
            self.calls.append({"messages": [m.to_openai() for m in messages], "tools": tools})
            return LLMResponse(content=self.summary)
        return super().chat(messages, tools)


class RecordingSummaryLLM(SummaryLLM):
    """记录每次压缩实际喂进去的文本，用来断言"没有历史被静默丢掉"。"""

    def __init__(self, summary: str = "【摘要】用户连续问了很多轮。") -> None:
        super().__init__([], summary=summary)
        self.prompts: list[str] = []

    def chat(self, messages, tools=None):
        if messages and messages[0].role == "system" and messages[0].content == COMPRESS_SYSTEM:
            self.prompts.append(messages[1].content)
        return super().chat(messages, tools)
