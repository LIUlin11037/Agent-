"""真实模型的端到端用例。

默认不执行（不产生费用、不依赖网络）。需要时显式开启：

    AGENT_LIVE_TEST=1 pytest tests/test_live.py -s
"""

from __future__ import annotations

import os

import pytest

from agent import AgentRuntime, Config

pytestmark = pytest.mark.skipif(
    os.getenv("AGENT_LIVE_TEST") != "1",
    reason="真实模型用例默认跳过，设置 AGENT_LIVE_TEST=1 开启",
)


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    base = tmp_path_factory.mktemp("live")
    config = Config.from_env(
        data_dir=base / "data",
        trace_dir=base / "data" / "traces",
        db_path=base / "data" / "live.db",
    )
    config.ensure_dirs()
    return AgentRuntime(config)


def test_live_direct_answer(runtime):
    result = runtime.chat("live_user", "用一句话介绍你自己")
    assert result.reply.strip()
    assert result.stopped_reason == "final"


def test_live_tool_flow(runtime):
    result = runtime.chat("live_user", "北京今天天气怎么样？")
    assert result.reply.strip()
    assert result.tool_calls, "问天气时模型应当调用 weather 工具"
    assert result.tool_calls[0]["name"] == "weather"
    assert result.tool_calls[0]["ok"] is True


def test_live_follow_up(runtime):
    first = runtime.chat("live_user", "北京天气怎么样？", session_id="live_win")
    assert first.reply
    second = runtime.chat("live_user", "那上海呢？", session_id="live_win")
    assert second.tool_calls, "追问应当触发对另一座城市的查询"
    assert "上海" in str(second.tool_calls[0]["arguments"])
