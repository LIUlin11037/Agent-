"""离线演示：不需要 API Key，用脚本化的假模型跑通完整链路。

    python examples/demo_offline.py

会依次展示：工具调用 → 结果回填 → 二次决策 → 落库，以及 trace 长什么样。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, Config, FakeLLM

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo"


def weather_reply(messages, tools):
    """把工具返回原样整理成回答，用来演示「结果确实回填进了上下文」。"""
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    content = tool_messages[-1]["content"] if tool_messages else "（没有拿到工具结果）"
    return {"content": "我查到了：\n" + content}


def reset_demo_state() -> None:
    """演示数据每次重跑都从头开始，否则待办会一行行累加，看不清「隔离」这件事。"""
    db = DEMO_DIR / "demo.db"
    if db.exists():
        db.unlink()
    for trace in (DEMO_DIR / "traces").glob("demo_win*.jsonl"):
        trace.unlink()


def build_runtime() -> AgentRuntime:
    reset_demo_state()
    config = Config(
        data_dir=DEMO_DIR,
        trace_dir=DEMO_DIR / "traces",
        db_path=DEMO_DIR / "demo.db",
    )
    config.ensure_dirs()
    script = [
        {"content": "我先查一下北京现在的天气。", "tool_calls": [{"name": "weather", "arguments": {"city": "北京"}}]},
        weather_reply,
        {
            "content": "好的，记一下。",
            "tool_calls": [
                {"name": "todo", "arguments": {"action": "add", "content": "周五前交周报"}}
            ],
        },
        {"content": "已经记到待办里了：周五前交周报。"},
    ]
    return AgentRuntime(config, llm=FakeLLM(script))


def show(title: str, result) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"助手：{result.reply}")
    print(f"步数：{result.steps}　停止原因：{result.stopped_reason}")
    for call in result.tool_calls:
        print(f"  · {call['name']}({call['arguments']}) -> ok={call['ok']} {call['output'][:60]}")


def main() -> None:
    runtime = build_runtime()

    print(">>> 窗口 1：问天气")
    first = runtime.chat("user_a", "北京天气怎么样？", session_id="demo_win1")
    show("窗口 1", first)

    print("\n>>> 窗口 2：记待办（同一用户，不同窗口）")
    second = runtime.chat("user_a", "记一下：周五前交周报", session_id="demo_win2")
    show("窗口 2", second)

    print("\n>>> 待办是用户级共享的：")
    for item in runtime.store.list_todos("user_a"):
        print(f"  - [{item['status']}] #{item['id']} {item['content']}")

    print("\n>>> 窗口 1 的 trace：")
    trace_path = Path(first.trace_path)
    for line in trace_path.read_text(encoding="utf-8").splitlines()[:8]:
        print("  " + line[:160])

    print("\n>>> 两个窗口的消息条数（互不干扰）：")
    print(f"  win1 = {len(runtime.history('demo_win1'))} 条，win2 = {len(runtime.history('demo_win2'))} 条")


if __name__ == "__main__":
    main()
