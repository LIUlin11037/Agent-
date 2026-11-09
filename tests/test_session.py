"""会话隔离与持久化测试。"""

from __future__ import annotations

import pytest

from agent.errors import InvalidSessionId, SessionAccessDenied


def test_two_windows_are_isolated(make_runtime):
    runtime, _ = make_runtime(
        [
            {"tool_calls": [{"name": "weather", "arguments": {"city": "北京"}}]},
            "北京多云，12°C。",
            {"tool_calls": [{"name": "todo", "arguments": {"action": "add", "content": "写周报"}}]},
            "已记录：写周报。",
        ]
    )
    first = runtime.chat("user_a", "北京天气怎么样", session_id="win1")
    second = runtime.chat("user_a", "记一下写周报", session_id="win2")

    assert first.session_id == "win1"
    assert second.session_id == "win2"

    history_1 = runtime.history("win1")
    history_2 = runtime.history("win2")
    assert any("北京" in message.content for message in history_1)
    assert not any("北京" in message.content for message in history_2)
    assert any("写周报" in message.content for message in history_2)
    assert not any("写周报" in message.content for message in history_1)


def test_session_survives_restart(make_runtime):
    runtime_1, _ = make_runtime(["记住了，你叫小林。"])
    runtime_1.chat("user_a", "我叫小林，请记住", session_id="win1")

    runtime_2, llm_2 = make_runtime(["你叫小林。"])
    result = runtime_2.chat("user_a", "我叫什么名字", session_id="win1")

    assert result.session_id == "win1"
    sent = " ".join(message.get("content", "") for message in llm_2.calls[0]["messages"])
    assert "小林" in sent


def test_todo_shared_across_windows(make_runtime):
    runtime, llm = make_runtime(
        [
            {"tool_calls": [{"name": "todo", "arguments": {"action": "add", "content": "买牛奶"}}]},
            "记好了。",
            {"tool_calls": [{"name": "todo", "arguments": {"action": "list"}}]},
            "你有 1 项待办：买牛奶。",
        ]
    )
    runtime.chat("user_a", "记一下买牛奶", session_id="win1")
    runtime.chat("user_a", "看看我的待办", session_id="win2")

    assert len(runtime.store.list_todos("user_a")) == 1
    # 第二个窗口虽然看不到窗口 1 的对话，但能看到共享的待办状态
    system_text = "\n".join(
        message["content"] for message in llm.calls[2]["messages"] if message["role"] == "system"
    )
    assert "买牛奶" in system_text


def test_todo_isolated_when_scope_is_session(make_runtime, config):
    config.todo_scope = "session"
    runtime, _ = make_runtime(
        [
            {"tool_calls": [{"name": "todo", "arguments": {"action": "add", "content": "写周报"}}]},
            "记好了。",
        ]
    )
    runtime.chat("user_a", "记一下写周报", session_id="win1")

    assert len(runtime.store.list_todos("win1")) == 1
    assert runtime.store.list_todos("win2") == []


def test_list_sessions_orders_by_update(make_runtime):
    runtime, _ = make_runtime(["一", "二"])
    runtime.chat("user_a", "第一件事", session_id="win1")
    runtime.chat("user_a", "第二件事", session_id="win2")

    sessions = runtime.list_sessions("user_a")
    assert [item["id"] for item in sessions] == ["win2", "win1"]
    assert sessions[0]["title"].startswith("第二件事")


def test_other_user_cannot_see_session(make_runtime):
    runtime, _ = make_runtime(["好的"])
    runtime.chat("user_a", "你好", session_id="win1")
    assert runtime.list_sessions("user_b") == []


def test_other_user_cannot_continue_session(make_runtime):
    """知道 session_id 也不等于有权接着聊——否则窗口隔离形同虚设。"""
    runtime, _ = make_runtime(["好的"])
    runtime.chat("user_a", "你好", session_id="win1")

    with pytest.raises(SessionAccessDenied):
        runtime.chat("user_b", "我也来看看", session_id="win1")

    with pytest.raises(SessionAccessDenied):
        runtime.history("win1", user_id="user_b")

    # 被拒绝的请求不能污染原会话
    history = runtime.history("win1", user_id="user_a")
    assert [message.role for message in history] == ["user", "assistant"]
    assert all("我也来看看" not in message.content for message in history)


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "a b", "x" * 65])
def test_invalid_session_id_rejected(make_runtime, bad_id):
    """session_id 会被拼进 trace 文件名，必须先校验，避免路径穿越。"""
    runtime, _ = make_runtime(["好的"])
    with pytest.raises(InvalidSessionId):
        runtime.chat("user_a", "你好", session_id=bad_id)


def test_history_keeps_raw_messages(make_runtime):
    runtime, _ = make_runtime(["第一轮回答", "第二轮回答"])
    runtime.chat("user_a", "第一个问题", session_id="win1")
    runtime.chat("user_a", "第二个问题", session_id="win1")

    history = runtime.history("win1")
    assert [message.role for message in history] == ["user", "assistant", "user", "assistant"]
    assert history[0].seq < history[-1].seq
