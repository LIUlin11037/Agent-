"""解析层测试：原生 tool_calls、文本协议、容错与降级。"""

from __future__ import annotations

from agent.llm import LLMResponse
from agent.parser import extract_thinking, loads_lenient, parse_response


def resp(content: str = "", tool_calls=None, reasoning: str = "") -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls or [], reasoning=reasoning)


def test_native_tool_calls():
    decision = parse_response(
        resp(tool_calls=[{"id": "c1", "name": "weather", "arguments": '{"city": "北京"}'}])
    )
    assert decision.kind == "tool_calls"
    assert decision.protocol == "native"
    assert decision.tool_calls[0].name == "weather"
    assert decision.tool_calls[0].arguments == {"city": "北京"}


def test_native_tool_calls_broken_arguments():
    decision = parse_response(
        resp(tool_calls=[{"id": "c1", "name": "weather", "arguments": '{"city": '}])
    )
    assert decision.kind == "invalid"
    assert "无法解析" in (decision.error or "")


def test_native_tool_call_without_name():
    decision = parse_response(resp(tool_calls=[{"id": "c1", "name": "", "arguments": "{}"}]))
    assert decision.kind == "invalid"


def test_text_protocol_fenced_json():
    content = '好的，我来算一下：\n```json\n{"tool": "calculator", "arguments": {"expression": "1+1"}}\n```'
    decision = parse_response(resp(content))
    assert decision.kind == "tool_calls"
    assert decision.protocol == "text"
    assert decision.tool_calls[0].name == "calculator"
    assert decision.tool_calls[0].arguments == {"expression": "1+1"}


def test_text_protocol_bare_json():
    decision = parse_response(resp('{"tool": "search", "arguments": {"query": "报销"}}'))
    assert decision.kind == "tool_calls"
    assert decision.tool_calls[0].name == "search"


def test_text_protocol_action_final():
    decision = parse_response(resp('{"action": "final", "answer": "答案是 2"}'))
    assert decision.kind == "final"
    assert decision.text == "答案是 2"


def test_text_protocol_python_literal():
    decision = parse_response(resp("{'tool': 'weather', 'arguments': {'city': '上海'}}"))
    assert decision.kind == "tool_calls"
    assert decision.tool_calls[0].arguments["city"] == "上海"


def test_text_protocol_react_style_action():
    """ReAct 风格：action 直接写工具名，参数放在 action_input。"""
    content = '{"thought": "先查天气", "action": "weather", "action_input": {"city": "北京"}}'
    decision = parse_response(resp(content))
    assert decision.kind == "tool_calls"
    assert decision.tool_calls[0].name == "weather"
    assert decision.tool_calls[0].arguments == {"city": "北京"}


def test_text_protocol_action_without_arguments_is_final():
    """只有 action、没有参数，不能瞎猜成工具调用。"""
    decision = parse_response(resp('{"action": "add", "content": "写周报"}'))
    assert decision.kind == "final"


def test_thinking_tag_extracted():
    decision = parse_response(resp("<thinking>先算乘法再算加法</thinking>结果是 12"))
    assert "先算乘法" in decision.thinking
    assert decision.text == "结果是 12"


def test_reasoning_field_used_as_thinking():
    decision = parse_response(resp("答案是 2", reasoning="内心独白"))
    assert decision.thinking == "内心独白"


def test_empty_response_is_invalid():
    decision = parse_response(resp(""))
    assert decision.kind == "invalid"
    assert decision.error


def test_plain_text_is_final():
    decision = parse_response(resp("你好，有什么可以帮你？"))
    assert decision.kind == "final"
    assert decision.text.startswith("你好")


def test_extract_thinking_multiple_blocks():
    thinking, rest = extract_thinking("<thinking>一</thinking>正文<thought>二</thought>")
    assert "一" in thinking and "二" in thinking
    assert rest == "正文"


def test_loads_lenient_trailing_comma():
    assert loads_lenient('{"a": 1, "b": [1, 2,],}') == {"a": 1, "b": [1, 2]}


def test_loads_lenient_embedded_object():
    text = '这是结果 {"city": "北京"} 请查收'
    assert loads_lenient(text) == {"city": "北京"}


def test_loads_lenient_garbage():
    assert loads_lenient("完全不是 JSON") is None
