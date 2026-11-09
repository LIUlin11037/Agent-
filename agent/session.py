"""会话存储（SQLite）。

三张表对应三种不同生命周期的数据，这也是"窗口隔离"能被验证的根本原因：

- sessions  ：会话元信息 + 滚动摘要 + 结构化事实（session 级）
- messages  ：未被压缩的消息（session 级）
- todos     ：待办（默认 user 级，可由 AGENT_TODO_SCOPE 切换为 session 级）

进程重启后 Session 可原样恢复，因为压缩水位线 compressed_upto 也落库了。
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import InvalidSessionId, SessionAccessDenied, SessionNotFound
from .types import Message, Session

# session_id 会被拼进 trace 文件名（data/traces/<id>.jsonl）。
# 只允许白名单字符，避免 ../ 之类的路径穿越把日志写到仓库外面。
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_session_id(session_id: str) -> str:
    if not _SESSION_ID_RE.match(session_id or ""):
        raise InvalidSessionId(
            f"非法的会话 id：{session_id!r}。只允许 1-64 位字母、数字、下划线和短横线。"
        )
    return session_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    title           TEXT DEFAULT '',
    summary         TEXT DEFAULT '',
    compressed_upto INTEGER DEFAULT 0,
    facts           TEXT DEFAULT '{}',
    created_at      REAL,
    updated_at      REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT DEFAULT '',
    name         TEXT,
    tool_call_id TEXT,
    tool_calls   TEXT,
    reasoning    TEXT,
    created_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
CREATE TABLE IF NOT EXISTS todos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    owner      TEXT NOT NULL,
    session_id TEXT,
    content    TEXT NOT NULL,
    status     TEXT DEFAULT 'open',
    created_at REAL,
    done_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_todos_owner ON todos(owner, status);
"""


def todo_owner(user_id: str, session_id: str, scope: str = "user") -> str:
    """待办的归属键。user 级共享是默认策略，session 级用于严格隔离场景。"""
    return session_id if scope == "session" else user_id


class SessionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def create_session(self, user_id: str, title: str = "", session_id: str | None = None) -> Session:
        sid = validate_session_id(session_id) if session_id else f"sess_{uuid.uuid4().hex[:10]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id, user_id, title, summary, compressed_upto, facts, created_at, updated_at)"
                " VALUES (?, ?, ?, '', 0, '{}', ?, ?)",
                (sid, user_id, title, now, now),
            )
            self._conn.commit()
        return Session(id=sid, user_id=user_id, title=title, created_at=now, updated_at=now)

    def ensure_session(self, user_id: str, session_id: str | None = None) -> Session:
        if session_id:
            try:
                # 带上 user_id：同一个 id 换个用户传进来，必须被拒绝，而不是接着聊
                return self.get_session(session_id, user_id=user_id)
            except SessionNotFound:
                return self.create_session(user_id, session_id=session_id)
        return self.create_session(user_id)

    def get_session(self, session_id: str, with_messages: bool = True, user_id: str | None = None) -> Session:
        validate_session_id(session_id)
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFound(f"会话 {session_id} 不存在")
        if user_id is not None and row["user_id"] != user_id:
            # 不暴露「会话存在但你没有权限」以外的时间差信息，直接拒绝
            raise SessionAccessDenied(f"会话 {session_id} 不属于用户 {user_id}")

        session = Session(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"] or "",
            summary=row["summary"] or "",
            compressed_upto=row["compressed_upto"] or 0,
            facts=json.loads(row["facts"] or "{}"),
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
        )
        if with_messages:
            session.messages = self.get_messages(session_id)
        return session

    def get_messages(self, session_id: str) -> list[Message]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY seq ASC", (session_id,)
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        return Message(
            role=row["role"],
            content=row["content"] or "",
            name=row["name"],
            tool_call_id=row["tool_call_id"],
            tool_calls=json.loads(row["tool_calls"]) if row["tool_calls"] else None,
            reasoning=row["reasoning"],
            seq=row["seq"],
            created_at=row["created_at"] or 0.0,
        )

    def list_sessions(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS msg_count"
                " FROM sessions s WHERE s.user_id = ? ORDER BY s.updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"] or "",
                "updated_at": row["updated_at"] or 0.0,
                "message_count": row["msg_count"],
                "has_summary": bool(row["summary"]),
            }
            for row in rows
        ]

    def rename_session(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            self._conn.commit()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()

    def touch(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time(), session_id))
            self._conn.commit()

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------
    def append_message(self, session: Session, message: Message) -> Message:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM messages WHERE session_id = ?",
                (session.id,),
            ).fetchone()
            message.seq = int(row["next"])
            self._conn.execute(
                "INSERT INTO messages (session_id, seq, role, content, name, tool_call_id,"
                " tool_calls, reasoning, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.id,
                    message.seq,
                    message.role,
                    message.content,
                    message.name,
                    message.tool_call_id,
                    json.dumps(message.tool_calls, ensure_ascii=False) if message.tool_calls else None,
                    message.reasoning,
                    message.created_at,
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time(), session.id)
            )
            self._conn.commit()
        session.messages.append(message)
        session.updated_at = time.time()
        return message

    def save_summary(self, session: Session, summary: str, compressed_upto: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET summary = ?, compressed_upto = ?, updated_at = ? WHERE id = ?",
                (summary, compressed_upto, time.time(), session.id),
            )
            self._conn.commit()
        session.summary = summary
        session.compressed_upto = compressed_upto

    def save_facts(self, session: Session) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET facts = ?, updated_at = ? WHERE id = ?",
                (session.dump_facts(), time.time(), session.id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 待办
    # ------------------------------------------------------------------
    def add_todo(self, owner: str, content: str, session_id: str = "") -> dict[str, Any]:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO todos (owner, session_id, content, status, created_at) VALUES (?, ?, ?, 'open', ?)",
                (owner, session_id, content, now),
            )
            self._conn.commit()
            todo_id = int(cur.lastrowid)
        return {"id": todo_id, "content": content, "status": "open", "created_at": now}

    def list_todos(self, owner: str, status: str = "all") -> list[dict[str, Any]]:
        sql = "SELECT * FROM todos WHERE owner = ?"
        params: list[Any] = [owner]
        if status in ("open", "done"):
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY CASE status WHEN 'open' THEN 0 ELSE 1 END, id ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def count_todos(self, owner: str, status: str = "open") -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM todos WHERE owner = ? AND status = ?", (owner, status)
            ).fetchone()
        return int(row["c"])

    def complete_todo(self, owner: str, todo_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE todos SET status = 'done', done_at = ? WHERE id = ? AND owner = ? AND status = 'open'",
                (time.time(), todo_id, owner),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def delete_todo(self, owner: str, todo_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM todos WHERE id = ? AND owner = ?", (todo_id, owner)
            )
            self._conn.commit()
        return cur.rowcount > 0

    def clear_todos(self, owner: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM todos WHERE owner = ?", (owner,))
            self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
