"""calculator：用 AST 白名单求值，绝不使用 eval()。"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from ..errors import ToolValidationError
from .base import ToolContext, tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}

_NORMALIZE = str.maketrans({"×": "*", "÷": "/", "^": "**", "（": "(", "）": ")", "，": ","})


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise TypeError("只支持数字常量")
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        func = _BIN_OPS.get(op_type)
        if func is None:
            raise ValueError(f"不支持的运算符：{op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > 64:
            raise ValueError("指数过大，已拒绝计算")
        return func(left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("只支持这些函数：abs/round/min/max/sum/sqrt/log/exp/floor/ceil/pow")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError(f"不支持的表达式结构：{type(node).__name__}")


def _format_number(value: float) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "NaN/Inf（结果无意义）"
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


@tool(
    name="calculator",
    description=(
        "执行数学计算。支持 + - * / // % **、括号，以及 "
        "abs/round/min/max/sum/sqrt/log/exp/floor/ceil 等函数。"
        "任何涉及算术的问题都应该用它，不要心算。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "待计算的表达式，例如 (23*7+11)/4 或 sqrt(144)+2**10",
                "minLength": 1,
                "maxLength": 200,
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)
def calculator(args: dict[str, Any], ctx: ToolContext) -> str:
    expression = str(args["expression"]).strip()
    normalized = expression.translate(_NORMALIZE)
    try:
        tree = ast.parse(normalized, mode="eval")
        value = _eval_node(tree)
    except ZeroDivisionError:
        raise ToolValidationError("除数为 0") from None
    except SyntaxError:
        raise ToolValidationError(f"表达式语法有误：{expression}") from None
    return f"{expression} = {_format_number(value)}"
