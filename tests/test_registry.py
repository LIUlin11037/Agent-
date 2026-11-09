"""工具注册表与 Schema 校验测试。"""

from __future__ import annotations

import pytest

from agent.tools import (
    ToolContext,
    all_specs,
    execute_tool,
    get_spec,
    openai_tools,
    tool_names,
    validate_arguments,
)


def test_required_tools_registered():
    names = set(tool_names())
    assert {"calculator", "search", "todo", "weather", "read_docs"} <= names


def test_spec_has_name_description_schema():
    for spec in all_specs():
        assert spec.name
        assert len(spec.description) > 10
        assert spec.parameters["type"] == "object"
        assert "properties" in spec.parameters


def test_openai_tool_schema_shape():
    for item in openai_tools():
        assert item["type"] == "function"
        fn = item["function"]
        assert {"name", "description", "parameters"} <= set(fn)
        assert fn["parameters"]["type"] == "object"


def test_validate_missing_required():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    errors = validate_arguments(schema, {})
    assert errors and "缺少必填参数" in errors[0]


def test_validate_wrong_type():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert validate_arguments(schema, {"n": "abc"})


def test_validate_bool_is_not_integer():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert validate_arguments(schema, {"n": True})


def test_validate_enum():
    schema = {"type": "object", "properties": {"a": {"type": "string", "enum": ["x", "y"]}}}
    assert validate_arguments(schema, {"a": "z"})
    assert validate_arguments(schema, {"a": "x"}) == []


def test_validate_range_and_length():
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1, "maximum": 5},
            "s": {"type": "string", "maxLength": 3},
        },
    }
    assert validate_arguments(schema, {"n": 0})
    assert validate_arguments(schema, {"n": 9})
    assert validate_arguments(schema, {"s": "abcd"})
    assert validate_arguments(schema, {"n": 3, "s": "ab"}) == []


def test_validate_additional_properties():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
    }
    assert validate_arguments(schema, {"a": "x", "b": 1})


def test_unknown_tool_returns_error_not_raise():
    result = execute_tool("not_exists", {}, ToolContext(session_id="s", user_id="u"))
    assert result.ok is False
    assert "不存在" in (result.error or "")


def test_bad_arguments_returned_not_raised():
    result = execute_tool("weather", {"city": 123}, ToolContext(session_id="s", user_id="u"))
    assert result.ok is False
    assert "类型" in (result.error or "")


def test_tool_exception_is_captured(tool_ctx):
    result = execute_tool("calculator", {"expression": "1/0"}, tool_ctx())
    assert result.ok is False
    assert "除数为 0" in (result.error or "")


def test_get_spec_unknown_raises():
    from agent.errors import ToolNotFound

    with pytest.raises(ToolNotFound):
        get_spec("nope")


def test_long_output_truncated(tool_ctx, config, store):
    for index in range(5):
        store.add_todo("u1", f"待办事项编号 {index} 的内容比较长一些")
    config.max_tool_output_chars = 40
    result = execute_tool("todo", {"action": "list"}, tool_ctx())
    assert result.ok
    assert "已截断" in result.content
