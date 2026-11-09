"""集中配置。

安全约定：任何密钥都不允许写进源码。这里只从环境变量 / .env 读取，
缺失时给出可操作的报错，而不是静默使用空字符串去请求接口。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv 是可选依赖，没有也能跑
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 兜底分支

    def load_dotenv(*_args, **_kwargs):
        return False


ROOT = Path(__file__).resolve().parent.parent


def _env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return value.strip() if value else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip())
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip())
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    """运行期配置。默认值即为推荐值，测试里可直接构造覆盖。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"

    max_steps: int = 8
    max_context_tokens: int = 12_000
    compress_threshold: float = 0.7
    keep_recent_turns: int = 6
    max_tool_output_chars: int = 4_000

    llm_timeout: float = 60.0
    llm_retries: int = 2

    data_dir: Path = ROOT / "data"
    trace_dir: Path = ROOT / "data" / "traces"
    db_path: Path = ROOT / "data" / "agent.db"

    todo_scope: str = "user"  # user | session

    @classmethod
    def from_env(cls, **overrides) -> Config:
        load_dotenv(ROOT / ".env")
        cfg = cls(
            api_key=_env_str("DEEPSEEK_API_KEY") or _env_str("OPENAI_API_KEY"),
            base_url=_env_str("LLM_BASE_URL", "https://api.deepseek.com"),
            model=_env_str("LLM_MODEL", "deepseek-chat"),
            max_steps=_env_int("AGENT_MAX_STEPS", 8),
            max_context_tokens=_env_int("AGENT_MAX_CONTEXT_TOKENS", 12_000),
            compress_threshold=_env_float("AGENT_COMPRESS_THRESHOLD", 0.7),
            keep_recent_turns=_env_int("AGENT_KEEP_RECENT_TURNS", 6),
            max_tool_output_chars=_env_int("AGENT_MAX_TOOL_OUTPUT", 4_000),
            todo_scope=_env_str("AGENT_TODO_SCOPE", "user"),
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        cfg.data_dir = Path(cfg.data_dir)
        cfg.trace_dir = Path(cfg.trace_dir)
        cfg.db_path = Path(cfg.db_path)
        cfg.ensure_dirs()
        return cfg

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def require_llm(self) -> None:
        """真正要发请求前才校验，保证纯离线测试不需要密钥。"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 LLM 密钥。请复制 .env.example 为 .env 并填写 DEEPSEEK_API_KEY；"
                "离线测试不需要密钥，运行 `pytest` 即可。"
            )
