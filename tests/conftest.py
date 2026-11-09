"""测试夹具。所有用例默认离线，不需要任何 API Key。"""

from __future__ import annotations

import pytest

from agent import AgentRuntime, Config, FakeLLM, SessionStore


@pytest.fixture
def config(tmp_path):
    cfg = Config(
        data_dir=tmp_path / "data",
        trace_dir=tmp_path / "data" / "traces",
        db_path=tmp_path / "data" / "test.db",
        max_steps=8,
        max_context_tokens=12_000,
        keep_recent_turns=6,
        compress_threshold=0.7,
        todo_scope="user",
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def store(config):
    session_store = SessionStore(config.db_path)
    yield session_store
    session_store.close()


@pytest.fixture
def make_runtime(config):
    def _make(script, llm_cls=FakeLLM):
        llm = llm_cls(script)
        return AgentRuntime(config, llm=llm), llm

    return _make


@pytest.fixture
def tool_ctx(store, config):
    from agent.tools import ToolContext

    def _ctx(user: str = "u1", session: str = "s1"):
        return ToolContext(session_id=session, user_id=user, store=store, config=config)

    return _ctx
