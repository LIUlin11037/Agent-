"""read_docs：读取本地文档。演示"先检索、再读全文、最后总结"的多步链路。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import ToolValidationError
from .base import ToolContext, tool

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "docs"


def _doc_files() -> list[Path]:
    if not DOCS_DIR.exists():
        return []
    return sorted(p for p in DOCS_DIR.glob("*.md") if p.is_file())


def _resolve_doc(name: str) -> Path:
    key = name.strip().lower().removesuffix(".md")
    files = _doc_files()
    exact = [p for p in files if p.stem.lower() == key]
    if exact:
        return exact[0]
    partial = [p for p in files if key in p.stem.lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ToolValidationError(
            f"「{name}」匹配到多份文档：{'、'.join(p.stem for p in partial)}，请指定更精确的名字"
        )
    available = "、".join(p.stem for p in files) or "（无）"
    raise ToolValidationError(f"没有找到文档「{name}」。可用文档：{available}")


@tool(
    name="read_docs",
    description="读取本地知识库文档。action=list 列出所有文档；action=read 读取指定文档全文，可用 query 只取相关段落。",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "read"], "description": "list 或 read"},
            "doc": {"type": "string", "description": "文档名（可省略 .md 后缀），action=read 时必填", "maxLength": 100},
            "query": {"type": "string", "description": "可选，只返回包含该关键词的段落", "maxLength": 100},
            "max_chars": {
                "type": "integer",
                "description": "最多返回字符数，默认 3000",
                "minimum": 200,
                "maximum": 8000,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
)
def read_docs(args: dict[str, Any], ctx: ToolContext) -> str:
    action = str(args["action"]).lower()
    files = _doc_files()

    if action == "list":
        if not files:
            return f"文档目录为空：{DOCS_DIR}"
        lines = [f"共 {len(files)} 份文档："]
        for path in files:
            title = ""
            try:
                first_line = path.read_text(encoding="utf-8").splitlines()[0]
                if first_line.startswith("#"):
                    title = first_line.lstrip("# ").strip()
            except (OSError, IndexError):
                pass
            lines.append(f"- {path.stem}" + (f"（{title}）" if title else ""))
        return "\n".join(lines)

    if action != "read":
        raise ToolValidationError(f"不支持的动作：{action}")

    doc_name = str(args.get("doc") or "").strip()
    if not doc_name:
        raise ToolValidationError("action=read 时必须提供 doc")

    path = _resolve_doc(doc_name)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolValidationError(f"读取文档失败：{exc}") from exc

    max_chars = int(args.get("max_chars") or 3000)
    query = str(args.get("query") or "").strip()

    if query:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        hits = [p for p in paragraphs if query.lower() in p.lower()]
        if hits:
            body = "\n\n".join(hits[:5])
            header = f"文档《{path.stem}》中与「{query}」相关的 {len(hits[:5])} 段：\n\n"
        else:
            body = text[:max_chars]
            header = f"文档《{path.stem}》未命中「{query}」，以下是开头部分：\n\n"
    else:
        body = text[:max_chars]
        header = f"文档《{path.stem}》全文（共 {len(text)} 字符）：\n\n"

    if len(body) > max_chars:
        body = body[:max_chars] + "\n…（已截断）"
    return header + body
