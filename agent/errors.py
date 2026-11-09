"""领域异常。区分「可恢复」与「不可恢复」，主循环据此决定是否继续。"""

from __future__ import annotations


class AgentError(Exception):
    """所有 Agent 相关异常的基类。"""


class LLMError(AgentError):
    """LLM 调用失败（网络 / 限流 / 服务端错误）。"""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class ParseError(AgentError):
    """模型输出无法解析成工具调用或最终答案。"""


class ToolError(AgentError):
    """工具执行失败。会被包装成 observation 回喂模型，而不是中断主循环。"""


class ToolNotFound(ToolError):
    pass


class ToolValidationError(ToolError):
    """参数不符合 Schema。"""


class SessionNotFound(AgentError):
    pass


class SessionAccessDenied(AgentError):
    """会话存在，但不属于当前用户。窗口隔离的最后一道门。"""


class InvalidSessionId(AgentError):
    """会话 id 含非法字符。id 会被拼进 trace 文件名，必须先校验。"""


class StepLimitExceeded(AgentError):
    """超过单轮最大步数。"""
