"""工具包。import 本模块即完成全部工具注册。"""

from __future__ import annotations

# 下面这些 import 的副作用就是注册工具，不要删。
from . import calculator as _calculator  # noqa: F401
from . import read_docs as _read_docs  # noqa: F401
from . import search as _search  # noqa: F401
from . import todo as _todo  # noqa: F401
from . import weather as _weather  # noqa: F401
from .base import (
    ToolContext,
    ToolSpec,
    all_specs,
    clear_registry,
    execute_tool,
    get_spec,
    openai_tools,
    tool,
    tool_names,
    validate_arguments,
)

__all__ = [
    "ToolContext",
    "ToolSpec",
    "all_specs",
    "clear_registry",
    "execute_tool",
    "get_spec",
    "openai_tools",
    "tool",
    "tool_names",
    "validate_arguments",
]
