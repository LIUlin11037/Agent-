"""主循环测试：直接回答、工具调用、多步、终止条件与异常路径。"""

from __future__ import annotations

from agent.errors import LLMError


def test_direct_reply(make_runtime):
    runtime, llm = make_runtime(["你好，我是助手。"])
    result = runtime.chat("u1", "你好")
    assert result.reply == "你好，我是助手。"
    assert result.steps == 1
    assert result.tool_calls == []
    assert result.stopped_reason == "final"
    assert len(llm.calls) == 1


def test_single_tool_call(make_runtime):
    runtime, llm = make_runtime(
        [
            {"content": "我查一下", "tool_calls": [{"name": "weather", "arguments": {"city": "上海"}}]},
            "上海今天晴，19°C。",
        ]
    )
    result = runtime.chat("u1", "上海天气怎么样")
    assert [item["name"] for item in result.tool_calls] == ["weather"]
    assert result.steps == 2
    assert result.stopped_reason == "final"

    second_round = llm.calls[1]["messages"]
    assert any(message["role"] == "tool" for message in second_round)
    assert any("上海" in message.get("content", "") for message in second_round if message["role"] == "tool")


def test_multi_step_chain(make_runtime):
    runtime, llm = make_runtime(
        [
            {"tool_calls": [{"name": "search", "arguments": {"query": "周报"}}]},
            {"tool_calls": [{"name": "read_docs", "arguments": {"action": "read", "doc": "weekly-report"}}]},
            "周报四段结构，500 字以内。",
        ]
    )
    result = runtime.chat("u1", "周报怎么写")
    assert [item["name"] for item in result.tool_calls] == ["search", "read_docs"]
    assert result.steps == 3
    assert len(llm.calls) == 3


def test_multiple_tool_calls_in_one_step(make_runtime):
    runtime, _ = make_runtime(
        [
            {
                "tool_calls": [
                    {"name": "calculator", "arguments": {"expression": "12*12"}},
                    {"name": "weather", "arguments": {"city": "杭州"}},
                ]
            },
            "144，杭州多云。",
        ]
    )
    result = runtime.chat("u1", "12*12 等于多少，杭州天气呢")
    assert len(result.tool_calls) == 2
    assert all(item["ok"] for item in result.tool_calls)


def test_max_steps_terminates(make_runtime, config):
    config.max_steps = 2
    runtime, _ = make_runtime(
        [
            {"tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]},
            {"tool_calls": [{"name": "calculator", "arguments": {"expression": "2+2"}}]},
            "永远不会被消费到",
        ]
    )
    result = runtime.chat("u1", "一直算下去")
    assert result.stopped_reason == "max_steps"
    assert result.steps == 2


def test_repair_after_invalid_output(make_runtime):
    runtime, llm = make_runtime(["", "好的，我直接回答：2。"])
    result = runtime.chat("u1", "1+1 等于几")
    assert result.reply.startswith("好的")
    assert result.stopped_reason == "final"
    assert len(llm.calls) == 2
    assert any("系统提示" in message.get("content", "") for message in llm.calls[1]["messages"])


def test_repair_gives_up_after_second_failure(make_runtime):
    runtime, _ = make_runtime(["", ""])
    result = runtime.chat("u1", "随便问问")
    assert result.stopped_reason == "error"
    assert "没能理解" in result.reply


def test_tool_error_is_fed_back_and_recovers(make_runtime):
    runtime, llm = make_runtime(
        [
            {"tool_calls": [{"name": "weather", "arguments": {"city": "火星"}}]},
            "抱歉，暂不支持火星，你可以问北京、上海等城市。",
        ]
    )
    result = runtime.chat("u1", "火星天气")
    assert result.tool_calls[0]["ok"] is False
    assert result.stopped_reason == "final"
    tool_message = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
    assert "工具执行失败" in tool_message["content"]


def test_unknown_tool_does_not_crash(make_runtime):
    runtime, _ = make_runtime(
        [
            {"tool_calls": [{"name": "nonexistent", "arguments": {}}]},
            "这个工具不存在，我换个方式回答。",
        ]
    )
    result = runtime.chat("u1", "试试不存在的工具")
    assert result.tool_calls[0]["ok"] is False
    assert result.stopped_reason == "final"


def test_loop_guard_breaks_repeat(make_runtime):
    same_call = {"tool_calls": [{"name": "weather", "arguments": {"city": "北京"}}]}
    runtime, _ = make_runtime([same_call, same_call, same_call, same_call])
    result = runtime.chat("u1", "北京天气")
    assert result.stopped_reason == "loop_guard"
    assert "重复" in result.reply or "相同参数" in result.reply


def test_loop_guard_handles_parallel_calls(make_runtime):
    """一步里并行调两个工具时，重复计数不能被另一个 key 重置。"""
    same_step = {
        "tool_calls": [
            {"name": "weather", "arguments": {"city": "北京"}},
            {"name": "weather", "arguments": {"city": "上海"}},
        ]
    }
    runtime, _ = make_runtime([same_step, same_step, same_step])
    result = runtime.chat("u1", "北京和上海天气")
    assert result.stopped_reason == "loop_guard"
    assert result.steps == 3


def test_loop_guard_blocks_before_side_effect(make_runtime):
    """写工具不能"先执行三次再拦截"，第三次必须在执行前被挡住。"""
    same_call = {
        "tool_calls": [{"name": "todo", "arguments": {"action": "add", "content": "重复记的事"}}]
    }
    runtime, _ = make_runtime([same_call, same_call, same_call])
    result = runtime.chat("u1", "一直记同一条")

    assert result.stopped_reason == "loop_guard"
    # 第 1、2 次真的执行，第 3 次被拦，所以库里只能有两条
    assert len(runtime.store.list_todos("u1")) == 2


def test_trace_turn_increments_across_chats(make_runtime):
    """同一会话多次对话，trace 的 turn 要接着数，不能每条都是 1。"""
    import json
    from pathlib import Path

    runtime, _ = make_runtime(["第一轮回答", "第二轮回答"])
    runtime.chat("u1", "第一轮", session_id="t1")
    second = runtime.chat("u1", "第二轮", session_id="t1")

    events = [
        json.loads(line)
        for line in Path(second.trace_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    turns = sorted({event["turn"] for event in events if event["event"] == "user_input"})
    assert turns == [1, 2]


def test_llm_error_returns_friendly_message(make_runtime):
    runtime, _ = make_runtime([LLMError("连接超时")])
    result = runtime.chat("u1", "你好")
    assert result.stopped_reason == "error"
    assert "模型服务" in result.reply


def test_unexpected_exception_is_contained(make_runtime):
    runtime, _ = make_runtime([RuntimeError("boom")])
    result = runtime.chat("u1", "你好")
    assert result.stopped_reason == "error"


def test_text_protocol_end_to_end(make_runtime):
    runtime, llm = make_runtime(
        [
            '```json\n{"tool": "calculator", "arguments": {"expression": "6*7"}}\n```',
            "答案是 42。",
        ]
    )
    result = runtime.chat("u1", "6*7 是多少")
    assert result.tool_calls[0]["name"] == "calculator"
    assert "42" in result.tool_calls[0]["output"]

    second_round = llm.calls[1]["messages"]
    assert any(
        message["role"] == "user" and "工具 calculator 返回" in message["content"]
        for message in second_round
    )


def test_trace_file_written(make_runtime):
    runtime, _ = make_runtime(["好的"])
    result = runtime.chat("u1", "你好")
    from pathlib import Path

    trace_path = Path(result.trace_path)
    assert trace_path.exists()
    content = trace_path.read_text(encoding="utf-8")
    assert "user_input" in content
    assert "llm_response" in content
    assert "final" in content
