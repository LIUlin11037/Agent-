"""从零实现的最小可用 Agent Runtime。

核心入口：

    from agent import AgentRuntime, Config

    runtime = AgentRuntime(Config.from_env())
    result = runtime.chat(user_id="user_a", text="北京天气怎么样？")
    print(result.reply)
"""

from __future__ import annotations

from .config import Config
from .llm import BaseLLM, DeepSeekLLM, FakeLLM
from .runtime import AgentRuntime
from .session import SessionStore
from .types import ChatResult, Message, Session, ToolResult

__version__ = "0.1.0"

__all__ = [
    "AgentRuntime",
    "BaseLLM",
    "ChatResult",
    "Config",
    "DeepSeekLLM",
    "FakeLLM",
    "Message",
    "Session",
    "SessionStore",
    "ToolResult",
]
