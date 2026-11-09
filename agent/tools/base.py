"""工具注册表与参数校验。

每个工具自带 名称 / 描述 / 参数 Schema，注册后自动转成 OpenAI 的
function calling 描述，模型据此自主决策调用哪个工具、传什么参数。

校验器是手写的（只覆盖本仓库用到的 JSON Schema 子集），好处是报错信息
可以说人话，并且能原样回喂给模型让它自我修正。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..errors import ToolNotFound, ToolValidationError
from ..types import ToolResult

Handler = Callable[[dict[str, Any], "ToolContext"], str]


@dataclass
class ToolContext:
    """工具执行时能拿到的最小上下文。"""

    session_id: str
    user_id: str
    store: Any = None
    config: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, parameters: dict[str, Any]) -> Callable[[Handler], Handler]:
    """装饰器注册。import 该模块即完成注册。"""

    def decorator(func: Handler) -> Handler:
        if name in _REGISTRY:
            raise ValueError(f"工具重名：{name}")
        _REGISTRY[name] = ToolSpec(name=name, description=description, parameters=parameters, handler=func)
        return func

    return decorator


def get_spec(name: str) -> ToolSpec:
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ToolNotFound(f"不存在名为 {name} 的工具")
    return spec


def all_specs() -> list[ToolSpec]:
    return list(_REGISTRY.values())


def tool_names() -> list[str]:
    return list(_REGISTRY.keys())


def openai_tools() -> list[dict[str, Any]]:
    return [spec.to_openai() for spec in _REGISTRY.values()]


def clear_registry() -> None:
    """仅供测试使用。"""
    _REGISTRY.clear()


# --------------------------------------------------------------------------
# JSON Schema 校验（子集）
# --------------------------------------------------------------------------

_TYPE_MAP = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _check_type(value: Any, expected: str) -> bool:
    if expected == "boolean":
        return isinstance(value, bool)
    if expected in ("number", "integer") and isinstance(value, bool):
        return False  # bool 是 int 的子类，这里要排除
    py_type = _TYPE_MAP.get(expected)
    if py_type is None:
        return True
    if expected == "integer" and isinstance(value, float):
        return value.is_integer()
    return isinstance(value, py_type)


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """返回错误列表；空列表代表通过。"""
    errors: list[str] = []
    if not isinstance(arguments, dict):
        return ["arguments 必须是一个 JSON 对象"]

    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    for key in required:
        if key not in arguments or arguments[key] is None:
            errors.append(f"缺少必填参数 `{key}`")

    for key, value in arguments.items():
        prop = properties.get(key)
        if prop is None:
            if schema.get("additionalProperties") is False:
                errors.append(f"不支持参数 `{key}`，可用参数：{', '.join(properties) or '无'}")
            continue

        expected = prop.get("type")
        if expected and not _check_type(value, expected):
            errors.append(f"参数 `{key}` 类型应为 {expected}，实际是 {type(value).__name__}")
            continue

        if "enum" in prop and value not in prop["enum"]:
            errors.append(f"参数 `{key}` 只能取 {prop['enum']} 之一，实际是 {value!r}")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in prop and value < prop["minimum"]:
                errors.append(f"参数 `{key}` 不能小于 {prop['minimum']}")
            if "maximum" in prop and value > prop["maximum"]:
                errors.append(f"参数 `{key}` 不能大于 {prop['maximum']}")

        if isinstance(value, str):
            if "minLength" in prop and len(value) < prop["minLength"]:
                errors.append(f"参数 `{key}` 长度不能小于 {prop['minLength']}")
            if "maxLength" in prop and len(value) > prop["maxLength"]:
                errors.append(f"参数 `{key}` 长度不能超过 {prop['maxLength']}")

        if isinstance(value, list):
            if "minItems" in prop and len(value) < prop["minItems"]:
                errors.append(f"参数 `{key}` 至少需要 {prop['minItems']} 项")
            if "maxItems" in prop and len(value) > prop["maxItems"]:
                errors.append(f"参数 `{key}` 最多 {prop['maxItems']} 项")
            item_schema = prop.get("items") or {}
            item_type = item_schema.get("type")
            if item_type:
                for index, item in enumerate(value):
                    if not _check_type(item, item_type):
                        errors.append(f"参数 `{key}[{index}]` 类型应为 {item_type}")
                        break

    return errors


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n…（输出过长，已截断，原始长度 {len(text)} 字符）"


def execute_tool(name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    """执行工具。任何异常都收敛成 ToolResult，绝不向上抛。"""
    started = time.perf_counter()

    try:
        spec = get_spec(name)
    except ToolNotFound as exc:
        return ToolResult(ok=False, content="", error=str(exc), latency_ms=0.0)

    errors = validate_arguments(spec.parameters, arguments)
    if errors:
        return ToolResult(
            ok=False,
            content="",
            error="参数校验失败：" + "；".join(errors),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    try:
        output = spec.handler(arguments, context)
    except ToolValidationError as exc:
        return ToolResult(
            ok=False,
            content="",
            error=str(exc),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # 工具内部任何意外都不该拖垮主循环
        return ToolResult(
            ok=False,
            content="",
            error=f"{type(exc).__name__}: {exc}",
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    limit = getattr(context.config, "max_tool_output_chars", 4000)
    text = output if isinstance(output, str) else str(output)
    return ToolResult(
        ok=True,
        content=truncate(text, limit),
        latency_ms=(time.perf_counter() - started) * 1000,
    )
