"""Web 接口的会话鉴权测试（不启动服务，直接调用路由函数）。"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import server
from agent import AgentRuntime


@pytest.fixture
def api_runtime(config, monkeypatch):
    from agent import FakeLLM

    runtime = AgentRuntime(config, llm=FakeLLM(["好的"]))
    monkeypatch.setattr(server, "_runtime", runtime)
    return runtime


def test_chat_endpoint_creates_session(api_runtime):
    payload = server.ChatRequest(user_id="user_a", message="你好")
    result = server.chat(payload)
    assert result["session_id"]
    assert result["reply"] == "好的"


def test_messages_endpoint_rejects_other_user(api_runtime):
    api_runtime.chat("user_a", "你好", session_id="win1")
    with pytest.raises(HTTPException) as exc:
        server.session_messages("win1", user_id="user_b")
    assert exc.value.status_code == 403


def test_trace_endpoint_rejects_other_user(api_runtime):
    api_runtime.chat("user_a", "你好", session_id="win1")
    with pytest.raises(HTTPException) as exc:
        server.session_trace("win1", user_id="user_b")
    assert exc.value.status_code == 403


def test_trace_endpoint_rejects_path_traversal(api_runtime):
    with pytest.raises(HTTPException) as exc:
        server.session_trace("../../server", user_id="user_a")
    assert exc.value.status_code == 404
