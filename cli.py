"""命令行入口。

支持在一个进程里开多个会话窗口（/new + /use），用来直观演示
「窗口 1 和窗口 2 互不干扰、随时可以切回去接着聊」。

    python cli.py
    python cli.py --user user_a --session sess_xxxx
"""

from __future__ import annotations

import argparse
import json
import sys

from agent import AgentRuntime, Config

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:  # pragma: no cover
    console = None
    HAS_RICH = False


HELP_TEXT = """命令：
  /new [标题]      新建一个会话窗口
  /sessions        列出当前用户的会话
  /use <序号|id>   切换到指定会话
  /history         查看当前会话的消息（压缩前的原文都在库里）
  /trace [n]       查看最近 n 条执行轨迹（默认 20）
  /todo            查看当前用户的待办
  /tools           列出已注册的工具
  /help            显示本帮助
  /quit            退出
直接输入内容即可对话。"""


def _print(text: str, style: str | None = None) -> None:
    if HAS_RICH:
        console.print(text, style=style)
    else:
        print(text)


def _print_reply(result) -> None:
    if HAS_RICH:
        console.print(Panel(result.reply, border_style="cyan", title="助手", title_align="left"))
        if result.tool_calls:
            table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
            table.add_column("工具")
            table.add_column("参数", overflow="fold")
            table.add_column("结果", overflow="fold")
            for call in result.tool_calls:
                args = json.dumps(call["arguments"], ensure_ascii=False)
                table.add_row(call["name"], args, call["output"][:80])
            console.print(table)
        console.print(
            f"[dim]step={result.steps}  stop={result.stopped_reason}  "
            f"session={result.session_id}[/dim]"
        )
    else:
        print(f"\n助手：{result.reply}")
        for call in result.tool_calls:
            print(f"  · {call['name']}({json.dumps(call['arguments'], ensure_ascii=False)})")
        print(f"  (step={result.steps}, stop={result.stopped_reason})")


class ChatCLI:
    def __init__(self, runtime: AgentRuntime, user_id: str, session_id: str | None = None) -> None:
        self.runtime = runtime
        self.user_id = user_id
        self.current = session_id

    def run(self) -> None:
        if not self.current:
            session = self.runtime.store.create_session(self.user_id, title="默认窗口")
            self.current = session.id
        _print(f"已进入会话 {self.current}（用户 {self.user_id}）。输入 /help 查看命令。", "dim")

        while True:
            try:
                raw = input("\n你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                _print("\n再见。", "dim")
                break
            if not raw:
                continue
            if raw.startswith("/"):
                if self._handle_command(raw):
                    break
                continue

            try:
                result = self.runtime.chat(self.user_id, raw, session_id=self.current)
            except Exception as exc:
                _print(f"[错误] {type(exc).__name__}: {exc}", "red")
                continue
            _print_reply(result)

    # ------------------------------------------------------------------
    def _handle_command(self, raw: str) -> bool:
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            return True
        if cmd in ("/help", "/h", "/?"):
            _print(HELP_TEXT)
            return False
        if cmd == "/new":
            session = self.runtime.store.create_session(self.user_id, title=arg or "新窗口")
            self.current = session.id
            _print(f"已新建会话 {session.id}", "green")
            return False
        if cmd == "/sessions":
            self._show_sessions()
            return False
        if cmd == "/use":
            self._switch(arg)
            return False
        if cmd == "/history":
            self._show_history()
            return False
        if cmd == "/trace":
            self._show_trace(arg)
            return False
        if cmd == "/todo":
            from agent.session import todo_owner

            owner = todo_owner(self.user_id, self.current or "", self.runtime.config.todo_scope)
            items = self.runtime.store.list_todos(owner, status="all")
            if not items:
                _print("当前没有待办。", "dim")
            for item in items:
                mark = "x" if item["status"] == "done" else " "
                _print(f"- [{mark}] #{item['id']} {item['content']}")
            return False
        if cmd == "/tools":
            from agent.tools import all_specs

            for spec in all_specs():
                _print(f"· {spec.name}: {spec.description[:60]}")
            return False

        _print(f"未知命令 {cmd}，输入 /help 查看可用命令。", "yellow")
        return False

    def _show_sessions(self) -> None:
        sessions = self.runtime.list_sessions(self.user_id)
        if not sessions:
            _print("还没有任何会话。", "dim")
            return
        for index, item in enumerate(sessions, start=1):
            mark = "*" if item["id"] == self.current else " "
            _print(
                f"{mark} {index}. {item['id']}  {item['title'] or '(无标题)'}  "
                f"{item['message_count']} 条消息"
            )

    def _switch(self, arg: str) -> None:
        sessions = self.runtime.list_sessions(self.user_id)
        if not arg:
            _print("用法：/use <序号|会话id>", "yellow")
            return
        if arg.isdigit():
            index = int(arg) - 1
            if 0 <= index < len(sessions):
                self.current = sessions[index]["id"]
                _print(f"已切换到 {self.current}", "green")
                return
            _print("序号超出范围。", "red")
            return
        self.current = arg
        _print(f"已切换到 {arg}", "green")

    def _show_history(self) -> None:
        try:
            messages = self.runtime.history(self.current or "", user_id=self.user_id)
        except Exception as exc:
            _print(f"[错误] {exc}", "red")
            return
        for message in messages:
            calls = ""
            if message.tool_calls:
                names = [c.get("function", {}).get("name") for c in message.tool_calls]
                calls = f" [tool_calls: {', '.join(str(n) for n in names)}]"
            _print(f"{message.seq:>3} {message.role:<9} {message.content[:100]}{calls}")

    def _show_trace(self, arg: str) -> None:
        from agent.trace import load_trace

        limit = int(arg) if arg.isdigit() else 20
        records = load_trace(self.runtime.config.trace_dir, self.current or "", limit=limit)
        if not records:
            _print("当前会话还没有 trace。", "dim")
            return
        for record in records:
            extras = {
                k: v
                for k, v in record.items()
                if k not in {"ts", "epoch", "session", "turn", "step", "event"}
            }
            detail = " ".join(f"{k}={v}" for k, v in extras.items() if v not in (None, "", []))
            _print(f"{record['ts'][11:]} [{record['event']}] {detail[:160]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="最小可用 Agent CLI")
    parser.add_argument("--user", default="user_a", help="用户 id，默认 user_a")
    parser.add_argument("--session", default=None, help="恢复指定会话 id")
    parser.add_argument("--echo-trace", action="store_true", help="实时打印执行轨迹")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass

    try:
        runtime = AgentRuntime(Config.from_env(), echo_trace=args.echo_trace)
    except RuntimeError as exc:
        _print(str(exc), "red")
        return 1

    ChatCLI(runtime, args.user, args.session).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
