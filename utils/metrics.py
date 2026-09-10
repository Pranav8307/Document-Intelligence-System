import threading
from collections import defaultdict


def _empty_stats() -> dict[str, float]:
    return {
        "count": 0,
        "total_ms": 0.0,
        "avg_ms": 0.0,
        "max_ms": 0.0,
        "min_ms": 0.0,
    }


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._timings: dict[str, dict[str, float]] = defaultdict(_empty_stats)

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def record_timing(self, name: str, duration_ms: float) -> None:
        with self._lock:
            stats = self._timings[name]
            stats["count"] += 1
            stats["total_ms"] += duration_ms
            stats["avg_ms"] = stats["total_ms"] / stats["count"]
            stats["max_ms"] = max(stats["max_ms"], duration_ms)
            stats["min_ms"] = (
                duration_ms if stats["count"] == 1 else min(stats["min_ms"], duration_ms)
            )

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            counters = dict(self._counters)
            timings = {name: dict(stats) for name, stats in self._timings.items()}
        return {"counters": counters, "timings_ms": timings}


metrics_store = MetricsStore()
