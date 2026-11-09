"""todo：待办管理。题目要求的必做工具。

存储归属由 `AGENT_TODO_SCOPE` 决定：
- user（默认）：同一用户的所有窗口共享一份待办，符合"待办是用户资产"的直觉；
- session：严格按窗口隔离，两个窗口的待办互不可见。
"""

from __future__ import annotations

from typing import Any

from ..errors import ToolValidationError
from ..session import todo_owner
from .base import ToolContext, tool


def _owner(ctx: ToolContext) -> str:
    return todo_owner(ctx.user_id, ctx.session_id, getattr(ctx.config, "todo_scope", "user"))


def _render(items: list[dict[str, Any]]) -> str:
    if not items:
        return "当前没有待办。"
    open_count = sum(1 for item in items if item["status"] == "open")
    done_count = len(items) - open_count
    lines = [f"共 {len(items)} 项待办（未完成 {open_count} / 已完成 {done_count}）："]
    for item in items:
        mark = "x" if item["status"] == "done" else " "
        lines.append(f"- [{mark}] #{item['id']} {item['content']}")
    return "\n".join(lines)


@tool(
    name="todo",
    description="管理用户的待办清单：新增(add)、查看(list)、完成(done)、删除(delete)、清空(clear)。用户提到「记一下」「待办」「提醒我」「完成了」等意图时使用。",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "done", "delete", "clear"],
                "description": "要执行的动作",
            },
            "content": {"type": "string", "description": "待办内容，action=add 时必填", "maxLength": 200},
            "todo_id": {"type": "integer", "description": "待办编号，action=done/delete 时必填", "minimum": 1},
            "status": {
                "type": "string",
                "enum": ["all", "open", "done"],
                "description": "action=list 时的过滤条件，默认 all",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
)
def todo(args: dict[str, Any], ctx: ToolContext) -> str:
    store = ctx.store
    if store is None:
        raise ToolValidationError("待办存储不可用")

    action = str(args["action"]).lower()
    owner = _owner(ctx)

    if action == "add":
        content = str(args.get("content") or "").strip()
        if not content:
            raise ToolValidationError("action=add 时必须提供 content")
        item = store.add_todo(owner, content, ctx.session_id)
        remaining = store.count_todos(owner, status="open")
        return f"已添加待办 #{item['id']}：{content}\n当前未完成 {remaining} 项。"

    if action == "list":
        status = str(args.get("status") or "all")
        items = store.list_todos(owner, status=status)
        return _render(items)

    if action == "done":
        todo_id = args.get("todo_id")
        if todo_id is None:
            raise ToolValidationError("action=done 时必须提供 todo_id")
        ok = store.complete_todo(owner, int(todo_id))
        if not ok:
            raise ToolValidationError(f"待办 #{todo_id} 不存在或已完成")
        remaining = store.count_todos(owner, status="open")
        return f"待办 #{todo_id} 已标记完成，剩余未完成 {remaining} 项。"

    if action == "delete":
        todo_id = args.get("todo_id")
        if todo_id is None:
            raise ToolValidationError("action=delete 时必须提供 todo_id")
        ok = store.delete_todo(owner, int(todo_id))
        if not ok:
            raise ToolValidationError(f"待办 #{todo_id} 不存在")
        return f"待办 #{todo_id} 已删除。"

    if action == "clear":
        removed = store.clear_todos(owner)
        return f"已清空 {removed} 项待办。"

    raise ToolValidationError(f"不支持的动作：{action}")
