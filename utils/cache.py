import time
from typing import Optional

# Simple TTL in-memory cache. Replace with Redis in production.
_cache: dict[str, tuple[any, float]] = {}
DEFAULT_TTL = 300  # 5 minutes


def cache_get(key: str) -> Optional[any]:
    entry = _cache.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.time() > expires_at:
        del _cache[key]
        return None
    return value


def cache_set(key: str, value: any, ttl: int = DEFAULT_TTL) -> None:
    _cache[key] = (value, time.time() + ttl)


def make_cache_key(document_id: str, question: str) -> str:
    return f"{document_id}::{question.strip().lower()}"