"""五个工具的行为测试。"""

from __future__ import annotations

from agent.tools import execute_tool


def test_calculator_basic(tool_ctx):
    result = execute_tool("calculator", {"expression": "(23*7+11)/4"}, tool_ctx())
    assert result.ok
    assert "43" in result.content


def test_calculator_power_and_function(tool_ctx):
    assert "1024" in execute_tool("calculator", {"expression": "2**10"}, tool_ctx()).content
    assert "13" in execute_tool("calculator", {"expression": "sqrt(144)+1"}, tool_ctx()).content


def test_calculator_fullwidth_operators(tool_ctx):
    result = execute_tool("calculator", {"expression": "6×7"}, tool_ctx())
    assert result.ok and "42" in result.content


def test_calculator_div_zero(tool_ctx):
    result = execute_tool("calculator", {"expression": "1/0"}, tool_ctx())
    assert result.ok is False
    assert "除数为 0" in (result.error or "")


def test_calculator_rejects_code_injection(tool_ctx):
    result = execute_tool("calculator", {"expression": "__import__('os').system('echo hi')"}, tool_ctx())
    assert result.ok is False


def test_calculator_rejects_unknown_function(tool_ctx):
    result = execute_tool("calculator", {"expression": "open('x')"}, tool_ctx())
    assert result.ok is False


def test_calculator_syntax_error(tool_ctx):
    result = execute_tool("calculator", {"expression": "1++"}, tool_ctx())
    assert result.ok is False


def test_search_hits_corpus(tool_ctx):
    result = execute_tool("search", {"query": "报销"}, tool_ctx())
    assert result.ok
    assert "差旅与报销流程" in result.content


def test_search_no_result(tool_ctx):
    result = execute_tool("search", {"query": "量子纠缠退火"}, tool_ctx())
    assert result.ok
    assert "没有检索到" in result.content


def test_search_top_k_limit(tool_ctx):
    result = execute_tool("search", {"query": "流程", "top_k": 1}, tool_ctx())
    assert result.ok
    assert result.content.count("（相关度") == 1


def test_weather_known_city(tool_ctx):
    result = execute_tool("weather", {"city": "北京"}, tool_ctx())
    assert result.ok
    assert "北京" in result.content and "°C" in result.content


def test_weather_unknown_city(tool_ctx):
    result = execute_tool("weather", {"city": "火星"}, tool_ctx())
    assert result.ok is False
    assert "暂不支持城市" in (result.error or "")


def test_weather_accepts_pinyin(tool_ctx):
    result = execute_tool("weather", {"city": "shanghai"}, tool_ctx())
    assert result.ok and "上海" in result.content


def test_weather_bad_date(tool_ctx):
    result = execute_tool("weather", {"city": "北京", "date": "2026/01/01"}, tool_ctx())
    assert result.ok is False


def test_read_docs_list(tool_ctx):
    result = execute_tool("read_docs", {"action": "list"}, tool_ctx())
    assert result.ok
    assert "release" in result.content


def test_read_docs_read(tool_ctx):
    result = execute_tool("read_docs", {"action": "read", "doc": "release"}, tool_ctx())
    assert result.ok
    assert "灰度" in result.content


def test_read_docs_query_filter(tool_ctx):
    result = execute_tool(
        "read_docs", {"action": "read", "doc": "weekly-report", "query": "500"}, tool_ctx()
    )
    assert result.ok
    assert "500 字" in result.content


def test_read_docs_missing_doc(tool_ctx):
    result = execute_tool("read_docs", {"action": "read", "doc": "不存在的文档"}, tool_ctx())
    assert result.ok is False


def test_todo_lifecycle(tool_ctx, store):
    added = execute_tool("todo", {"action": "add", "content": "买牛奶"}, tool_ctx())
    assert added.ok and "买牛奶" in added.content
    todo_id = store.list_todos("u1")[0]["id"]

    listed = execute_tool("todo", {"action": "list"}, tool_ctx())
    assert listed.ok and "买牛奶" in listed.content

    done = execute_tool("todo", {"action": "done", "todo_id": todo_id}, tool_ctx())
    assert done.ok
    assert store.count_todos("u1", "open") == 0

    deleted = execute_tool("todo", {"action": "delete", "todo_id": todo_id}, tool_ctx())
    assert deleted.ok
    assert store.list_todos("u1") == []


def test_todo_add_requires_content(tool_ctx):
    result = execute_tool("todo", {"action": "add"}, tool_ctx())
    assert result.ok is False
    assert "content" in (result.error or "")


def test_todo_done_unknown_id(tool_ctx):
    result = execute_tool("todo", {"action": "done", "todo_id": 999}, tool_ctx())
    assert result.ok is False


def test_todo_clear(tool_ctx, store):
    for index in range(3):
        store.add_todo("u1", f"事项{index}")
    result = execute_tool("todo", {"action": "clear"}, tool_ctx())
    assert result.ok and "3" in result.content
    assert store.list_todos("u1") == []


def test_todo_invalid_action_rejected_by_schema(tool_ctx):
    result = execute_tool("todo", {"action": "explode"}, tool_ctx())
    assert result.ok is False
    assert "只能取" in (result.error or "")
