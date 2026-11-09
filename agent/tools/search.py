"""search：本地 mock 检索。

不联网也能演示"先查资料再回答"的链路。打分规则刻意做得可解释：
标题命中权重最高，其次标签，最后正文；中文按 2-gram 切分，英文按单词切分。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .base import ToolContext, tool

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_LATIN_WORD = re.compile(r"[a-zA-Z0-9_]+")

CORPUS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "corpus.json"


@lru_cache(maxsize=1)
def _load_corpus() -> list[dict[str, Any]]:
    if not CORPUS_PATH.exists():
        return []
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = list(_LATIN_WORD.findall(lowered))
    for run in _CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def _score(query_tokens: list[str], doc: dict[str, Any]) -> int:
    title = str(doc.get("title", "")).lower()
    tags = " ".join(doc.get("tags", [])).lower()
    body = str(doc.get("text", "")).lower()

    score = 0
    for token in set(query_tokens):
        if not token:
            continue
        if token in title:
            score += 6
        if token in tags:
            score += 3
        if token in body:
            score += 2
    return score


def _snippet(text: str, tokens: list[str], width: int = 90) -> str:
    lowered = text.lower()
    positions = [lowered.find(t) for t in tokens if t and lowered.find(t) != -1]
    start = max(0, (min(positions) if positions else 0) - 20)
    piece = text[start : start + width].replace("\n", " ").strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if start + width < len(text) else ""
    return f"{prefix}{piece}{suffix}"


@tool(
    name="search",
    description=(
        "检索公司内部知识库（本地 mock 语料）。适用于查制度、流程、技术规范等问题。"
        "返回文档标题与摘要，需要全文时再用 read_docs。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索关键词，中文或英文均可", "minLength": 1, "maxLength": 100},
            "top_k": {"type": "integer", "description": "返回条数，默认 3", "minimum": 1, "maximum": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)
def search(args: dict[str, Any], ctx: ToolContext) -> str:
    query = str(args["query"]).strip()
    top_k = int(args.get("top_k") or 3)
    corpus = _load_corpus()
    if not corpus:
        return "知识库为空（data/corpus.json 未找到或没有内容）。"

    tokens = _tokenize(query)
    scored = [(doc, _score(tokens, doc)) for doc in corpus]
    scored = [(doc, score) for doc, score in scored if score > 0]
    scored.sort(key=lambda item: item[1], reverse=True)

    if not scored:
        return f"没有检索到与「{query}」相关的内容。可以换更宽泛的关键词，或直接用 read_docs 列出可用文档。"

    lines = [f"检索「{query}」命中 {min(len(scored), top_k)} 条："]
    for index, (doc, score) in enumerate(scored[:top_k], start=1):
        lines.append(f"{index}. [{doc['id']}] {doc['title']}（相关度 {score}）")
        lines.append(f"   {_snippet(str(doc.get('text', '')), tokens)}")
    return "\n".join(lines)
