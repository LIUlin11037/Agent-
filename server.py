"""极简 Web 演示（可选）。

    uvicorn server:app --reload --port 8000

左侧是会话列表（相当于多个窗口），右侧是对话区。用来直观演示
「两个窗口各自独立、可以随时切回去接着聊」。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent import AgentRuntime, Config
from agent.errors import InvalidSessionId, SessionAccessDenied, SessionNotFound
from agent.tools import all_specs

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Minimal Agent", version="0.1.0")
_runtime: AgentRuntime | None = None


def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        try:
            _runtime = AgentRuntime(Config.from_env())
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _runtime


class ChatRequest(BaseModel):
    user_id: str = "user_a"
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


class SessionRequest(BaseModel):
    user_id: str = "user_a"
    title: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/tools")
def list_tools() -> list[dict[str, Any]]:
    return [
        {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
        for spec in all_specs()
    ]


@app.get("/api/sessions")
def list_sessions(user_id: str = "user_a") -> list[dict[str, Any]]:
    return get_runtime().list_sessions(user_id)


@app.post("/api/sessions")
def create_session(payload: SessionRequest) -> dict[str, Any]:
    session = get_runtime().store.create_session(payload.user_id, title=payload.title or "新窗口")
    return {"id": session.id, "title": session.title}


@app.get("/api/sessions/{session_id}/messages")
def session_messages(session_id: str, user_id: str = "user_a") -> list[dict[str, Any]]:
    try:
        messages = get_runtime().history(session_id, user_id=user_id)
    except SessionAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (SessionNotFound, InvalidSessionId) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [
        {
            "seq": message.seq,
            "role": message.role,
            "content": message.content,
            "tool_calls": [
                call.get("function", {}).get("name") for call in (message.tool_calls or [])
            ],
        }
        for message in messages
    ]


@app.get("/api/sessions/{session_id}/trace")
def session_trace(session_id: str, limit: int = 60, user_id: str = "user_a") -> list[dict[str, Any]]:
    runtime = get_runtime()
    # trace 路径由 session_id 拼成，先校验会话归属，避免按 id 读到别人的日志
    try:
        runtime.store.get_session(session_id, with_messages=False, user_id=user_id)
    except SessionAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (SessionNotFound, InvalidSessionId) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = Path(runtime.config.trace_dir) / f"{session_id}.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records[-limit:]


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    runtime = get_runtime()
    try:
        result = runtime.chat(payload.user_id, payload.message, session_id=payload.session_id)
    except SessionAccessDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except InvalidSessionId as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return result.to_dict()


@app.get("/api/todos")
def list_todos(user_id: str = "user_a") -> list[dict[str, Any]]:
    return get_runtime().store.list_todos(user_id, status="all")
