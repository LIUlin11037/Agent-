"""weather：本地 mock 天气。结果按「城市 + 日期」稳定生成，便于复现与测试。"""

from __future__ import annotations

import hashlib
import math
from datetime import date
from typing import Any

from ..errors import ToolValidationError
from .base import ToolContext, tool

_CITIES: dict[str, dict[str, Any]] = {
    "北京": {"aliases": ["beijing", "bj"], "lat": 39.9, "base": 13},
    "上海": {"aliases": ["shanghai", "sh"], "lat": 31.2, "base": 19},
    "广州": {"aliases": ["guangzhou", "gz"], "lat": 23.1, "base": 24},
    "深圳": {"aliases": ["shenzhen", "sz"], "lat": 22.5, "base": 25},
    "杭州": {"aliases": ["hangzhou", "hz"], "lat": 30.3, "base": 19},
    "成都": {"aliases": ["chengdu", "cd"], "lat": 30.6, "base": 18},
    "武汉": {"aliases": ["wuhan", "wh"], "lat": 30.6, "base": 19},
    "西安": {"aliases": ["xian", "xa"], "lat": 34.3, "base": 15},
    "南京": {"aliases": ["nanjing", "nj"], "lat": 32.1, "base": 18},
    "重庆": {"aliases": ["chongqing", "cq"], "lat": 29.6, "base": 20},
}

_CONDITIONS = ["晴", "多云", "阴", "小雨", "阵雨", "晴间多云"]
_ALIAS_INDEX = {alias: name for name, meta in _CITIES.items() for alias in meta["aliases"]}
_ALIAS_INDEX.update({name: name for name in _CITIES})


def _resolve(city: str) -> str:
    key = city.strip().lower()
    resolved = _ALIAS_INDEX.get(key)
    if not resolved:
        raise ToolValidationError(
            f"暂不支持城市「{city}」，可用城市：{'、'.join(_CITIES)}（也接受拼音）"
        )
    return resolved


def _stable_seed(city: str, day: date) -> int:
    digest = hashlib.sha256(f"{city}-{day.isoformat()}".encode()).hexdigest()
    return int(digest[:8], 16)


def _seasonal_base(city: str, day: date) -> float:
    """用正弦曲线近似气温年周期，让 mock 数据不至于一年四季一个温度。"""
    meta = _CITIES[city]
    day_of_year = day.timetuple().tm_yday
    amplitude = 14 if meta["lat"] > 35 else 9
    return meta["base"] + amplitude * math.sin((day_of_year - 105) / 365 * 2 * math.pi)


@tool(
    name="weather",
    description=(
        "查询城市天气（本地 mock 数据，支持北京/上海/广州/深圳/杭州/成都/武汉/西安/南京/重庆）。"
        "用户问天气、温度、要不要带伞时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，中文或拼音", "minLength": 1, "maxLength": 20},
            "date": {
                "type": "string",
                "description": "日期，格式 YYYY-MM-DD，默认今天",
                "maxLength": 10,
            },
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)
def weather(args: dict[str, Any], ctx: ToolContext) -> str:
    city = _resolve(str(args["city"]))
    raw_date = str(args.get("date") or "").strip()
    if raw_date:
        try:
            day = date.fromisoformat(raw_date)
        except ValueError:
            raise ToolValidationError(f"日期格式应为 YYYY-MM-DD，收到 {raw_date!r}") from None
    else:
        day = date.today()

    seed = _stable_seed(city, day)
    base = _seasonal_base(city, day)
    variation = (seed % 70) / 10 - 3.5
    temp = round(base + variation)
    condition = _CONDITIONS[seed % len(_CONDITIONS)]
    humidity = 40 + seed % 50
    wind_level = 1 + seed % 5
    feels_like = temp - 1 if wind_level >= 4 else temp

    advice = "出门记得带伞。" if condition in {"小雨", "阵雨"} else "适合外出。"
    if temp <= 5:
        advice = "气温偏低，注意保暖。"

    return (
        f"城市：{city}\n"
        f"日期：{day.isoformat()}\n"
        f"天气：{condition}\n"
        f"气温：{temp}°C（体感 {feels_like}°C）\n"
        f"湿度：{humidity}%\n"
        f"风力：{wind_level} 级\n"
        f"提示：{advice}\n"
        f"（数据来源：本地 mock，仅供演示）"
    )
