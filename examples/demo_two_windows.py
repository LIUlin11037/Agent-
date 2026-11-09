"""两个窗口交错对话（需要真实 API Key）。

    python examples/demo_two_windows.py

窗口 1 和窗口 2 属于同一个用户，但对话上下文完全隔离：
- 窗口 1 追问"那上海呢"，靠的是本窗口的上一次工具调用；
- 两个窗口的待办是同一份（user 级共享），可以在任一窗口查看。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import AgentRuntime, Config

WINDOW_1 = [
    "北京天气怎么样？",
    "那上海呢？",
    "记一下：周五前交周报",
    "我今天让你记过什么？",
]

WINDOW_2 = [
    "周报有什么格式要求？",
    "把「补测试」也加到待办里",
    "现在一共有哪些待办？",
]


def main() -> None:
    try:
        runtime = AgentRuntime(Config.from_env())
    except RuntimeError as exc:
        print(exc)
        return

    windows = {"窗口1": ("win_a", WINDOW_1), "窗口2": ("win_b", WINDOW_2)}
    max_rounds = max(len(turns) for _, turns in windows.values())

    for index in range(max_rounds):
        for label, (session_id, turns) in windows.items():
            if index >= len(turns):
                continue
            question = turns[index]
            print(f"\n[{label}] 用户：{question}")
            result = runtime.chat("user_a", question, session_id=session_id)
            print(f"[{label}] 助手：{result.reply}")
            if result.tool_calls:
                names = "、".join(call["name"] for call in result.tool_calls)
                print(f"[{label}] 调用：{names}（{result.steps} 步）")

    print("\n===== 会话隔离检查 =====")
    for label, (session_id, _) in windows.items():
        history = runtime.history(session_id)
        print(f"{label}（{session_id}）共 {len(history)} 条消息")

    print("\n===== 待办（用户级共享） =====")
    for item in runtime.store.list_todos("user_a"):
        print(f"- [{item['status']}] #{item['id']} {item['content']}")


if __name__ == "__main__":
    main()
