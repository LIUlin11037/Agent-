"""上下文装配、追问与压缩测试。"""

from __future__ import annotations

from agent import AgentRuntime
from agent.context import (
    ContextBuilder,
    estimate_messages,
    estimate_tokens,
    group_turns,
)
from agent.types import Message
from tests.helpers import SummaryLLM


def _system_text(messages) -> str:
    return "\n".join(
        message.get("content", "") if isinstance(message, dict) else message.content
        for message in messages
        if (message.get("role") if isinstance(message, dict) else message.role) == "system"
    )


def test_estimate_tokens_prefers_cjk():
    assert estimate_tokens("你好世界") >= 4
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 100) < estimate_tokens("中" * 100)


def test_group_turns_keeps_tool_pairs():
    messages = [
        Message(role="user", content="q1", seq=1),
        Message(
            role="assistant",
            content="",
            seq=2,
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}],
        ),
        Message(role="tool", content="r1", tool_call_id="c1", seq=3),
        Message(role="assistant", content="a1", seq=4),
        Message(role="user", content="q2", seq=5),
    ]
    turns = group_turns(messages)
    assert len(turns) == 2
    assert len(turns[0]) == 4
    assert len(turns[1]) == 1


def test_state_block_contains_last_tool_call(make_runtime):
    runtime, llm = make_runtime(
        [
            {"tool_calls": [{"name": "weather", "arguments": {"city": "北京"}}]},
            "北京多云，12°C。",
            "上海晴，19°C。",
        ]
    )
    runtime.chat("user_a", "北京天气怎么样", session_id="w1")
    runtime.chat("user_a", "那上海呢", session_id="w1")

    second_turn = llm.calls[2]["messages"]
    system_text = _system_text(second_turn)
    assert "上次工具调用" in system_text
    assert "weather" in system_text
    assert "北京" in system_text


def test_pure_chat_follow_up(make_runtime):
    runtime, llm = make_runtime(["我叫小林。", "你叫小林。"])
    runtime.chat("user_a", "我叫小林", session_id="w1")
    runtime.chat("user_a", "我叫什么", session_id="w1")

    second_turn = llm.calls[1]["messages"]
    assert any("我叫小林" in message.get("content", "") for message in second_turn)


def test_state_block_includes_todos(make_runtime, store):
    store.add_todo("user_a", "准备周会材料")
    runtime, llm = make_runtime(["好的"])
    runtime.chat("user_a", "你好", session_id="w1")

    system_text = _system_text(llm.calls[0]["messages"])
    assert "准备周会材料" in system_text


def test_context_contains_system_prompt(make_runtime):
    runtime, llm = make_runtime(["好的"])
    runtime.chat("user_a", "你好", session_id="w1")
    system_text = _system_text(llm.calls[0]["messages"])
    assert "工具使用规范" in system_text


def test_compression_triggers_and_keeps_raw_messages(config):
    config.max_context_tokens = 400
    config.compress_threshold = 0.5
    config.keep_recent_turns = 1

    script = [f"第 {index} 轮回答：" + "这是一段比较长的回复内容。" * 8 for index in range(6)]
    llm = SummaryLLM(script)
    runtime = AgentRuntime(config, llm=llm)

    session_id = None
    for index in range(6):
        result = runtime.chat("user_a", f"第 {index} 个问题：" + "请详细说明" * 5, session_id=session_id)
        session_id = result.session_id

    session = runtime.store.get_session(session_id)
    assert session.summary, "压缩后应生成摘要"
    assert session.compressed_upto > 0
    assert llm.compress_calls >= 1

    active = [message for message in session.messages if message.seq > session.compressed_upto]
    assert 0 < len(active) < len(session.messages), "原文应保留在库中，但不进入上下文"


def test_summary_injected_after_compression(config):
    config.max_context_tokens = 400
    config.compress_threshold = 0.5
    config.keep_recent_turns = 1

    script = [f"回答 {index}：" + "内容很长。" * 12 for index in range(5)]
    llm = SummaryLLM(script, summary="【摘要】用户连续问了几个问题。")
    runtime = AgentRuntime(config, llm=llm)

    session_id = None
    for index in range(5):
        result = runtime.chat("user_a", f"问题 {index}：" + "补充背景" * 4, session_id=session_id)
        session_id = result.session_id

    last_messages = llm.calls[-1]["messages"]
    system_text = _system_text(last_messages)
    assert "【历史对话摘要】" in system_text
    assert "用户连续问了几个问题" in system_text


def test_compression_failure_falls_back(config, monkeypatch):
    """压缩用的 LLM 调用失败时，必须退化为机械摘要，不能中断主流程。"""
    config.max_context_tokens = 400
    config.compress_threshold = 0.5
    config.keep_recent_turns = 1

    llm = SummaryLLM([f"回答 {index}：" + "内容很长。" * 12 for index in range(5)])
    runtime = AgentRuntime(config, llm=llm)

    original = llm.chat

    def flaky(messages, tools=None):
        if messages and messages[0].role == "system" and "对话压缩器" in messages[0].content:
            raise RuntimeError("压缩服务不可用")
        return original(messages, tools)

    monkeypatch.setattr(llm, "chat", flaky)

    session_id = None
    for index in range(4):
        result = runtime.chat("user_a", f"问题 {index}：" + "补充背景" * 4, session_id=session_id)
        session_id = result.session_id

    session = runtime.store.get_session(session_id)
    assert session.compressed_upto > 0
    assert session.summary, "LLM 压缩失败时应写入机械摘要兜底"


def test_follow_up_survives_compression(config):
    """压缩掉早期原文后，「那上海呢」仍然要靠状态块补全参数。"""
    config.max_context_tokens = 400
    config.compress_threshold = 0.5
    config.keep_recent_turns = 1

    llm = SummaryLLM(
        [
            {"tool_calls": [{"name": "weather", "arguments": {"city": "北京"}}]},
            "北京多云。",
            "上海晴，19°C。",
        ]
    )
    runtime = AgentRuntime(config, llm=llm)
    first = runtime.chat("user_a", "北京天气怎么样", session_id="w1")
    assert first.tool_calls[0]["name"] == "weather"

    # 灌入足够长的历史，逼迫下一次装配上下文时触发压缩
    session = runtime.store.get_session("w1")
    for _ in range(8):
        runtime.store.append_message(session, Message(role="user", content="补充背景" * 60))
        runtime.store.append_message(session, Message(role="assistant", content="好的，记下了。" * 60))

    runtime.chat("user_a", "那上海呢", session_id="w1")

    session = runtime.store.get_session("w1")
    assert session.summary, "应当已经触发压缩"
    sent = _system_text(llm.calls[-1]["messages"])
    assert "上次工具调用" in sent
    assert "北京" in sent


def test_context_builder_direct_use(config, store):
    session = store.create_session("user_a", title="t")
    store.append_message(session, Message(role="user", content="你好"))
    builder = ContextBuilder(config, llm=None, store=store)
    messages = builder.build(session)
    assert messages[0].role == "system"
    assert messages[-1].content == "你好"
    assert estimate_messages(messages) > 0
