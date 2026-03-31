"""
EDEN OS — Conductor: Metrics Collector
Real-time pipeline performance metrics with rolling-window percentiles.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from loguru import logger

from eden_os.shared.types import PipelineMetrics


_WINDOW_SIZE = 100  # rolling window for percentile calculation


class MetricsCollector:
    """Collects and summarises pipeline metrics.

    Records named numeric values in a rolling window and exposes
    both a typed ``PipelineMetrics`` snapshot and a richer ``dict``
    summary including percentiles and error rates.
    """

    def __init__(self, window_size: int = _WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._data: dict[str, list[float]] = defaultdict(list)
        self._error_counts: dict[str, int] = defaultdict(int)
        self._total_calls: dict[str, int] = defaultdict(int)
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, metric_name: str, value: float) -> None:
        """Append *value* to the rolling window for *metric_name*."""
        buf = self._data[metric_name]
        buf.append(value)
        if len(buf) > self._window_size:
            buf.pop(0)
        self._total_calls[metric_name] += 1

    def record_error(self, engine_name: str) -> None:
        """Increment the error counter for *engine_name*."""
        self._error_counts[engine_name] += 1

    # ------------------------------------------------------------------
    # Percentile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(sorted_vals: list[float], p: float) -> float:
        """Return the *p*-th percentile (0-100) from a **sorted** list."""
        if not sorted_vals:
            return 0.0
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_vals):
            return sorted_vals[f]
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    def _pcts(self, name: str) -> dict[str, float]:
        vals = sorted(self._data.get(name, []))
        return {
            "p50": round(self._percentile(vals, 50), 2),
            "p95": round(self._percentile(vals, 95), 2),
            "p99": round(self._percentile(vals, 99), 2),
        }

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def get_metrics(self) -> PipelineMetrics:
        """Build a ``PipelineMetrics`` snapshot from the latest values."""
        def _last(name: str) -> float:
            buf = self._data.get(name)
            return buf[-1] if buf else 0.0

        return PipelineMetrics(
            asr_latency_ms=_last("asr_ms"),
            llm_first_token_ms=_last("llm_first_token_ms"),
            tts_first_chunk_ms=_last("tts_first_chunk_ms"),
            animation_fps=_last("animation_fps"),
            total_latency_ms=_last("total_ms"),
            gpu_memory_used_mb=_last("gpu_memory_used_mb"),
            gpu_memory_total_mb=_last("gpu_memory_total_mb"),
        )

    def get_summary(self) -> dict[str, Any]:
        """Return a rich summary with percentiles, counts, and error rates."""
        uptime_s = time.monotonic() - self._start_time
        summary: dict[str, Any] = {
            "uptime_seconds": round(uptime_s, 1),
            "stages": {},
            "error_rates": {},
        }

        for name, buf in self._data.items():
            if not buf:
                continue
            sorted_buf = sorted(buf)
            summary["stages"][name] = {
                "last": round(buf[-1], 2),
                "avg": round(sum(buf) / len(buf), 2),
                "min": round(sorted_buf[0], 2),
                "max": round(sorted_buf[-1], 2),
                "count": self._total_calls.get(name, len(buf)),
                **self._pcts(name),
            }

        # Error rates (errors / total calls) per engine
        all_engines = set(self._error_counts.keys()) | set(self._total_calls.keys())
        for eng in all_engines:
            total = self._total_calls.get(eng, 0)
            errors = self._error_counts.get(eng, 0)
            summary["error_rates"][eng] = {
                "errors": errors,
                "total": total,
                "rate": round(errors / total, 4) if total > 0 else 0.0,
            }

        return summary

    def reset(self) -> None:
        """Clear all collected data."""
        self._data.clear()
        self._error_counts.clear()
        self._total_calls.clear()
        self._start_time = time.monotonic()
