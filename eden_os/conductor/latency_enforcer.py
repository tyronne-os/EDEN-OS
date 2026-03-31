"""
EDEN OS — Conductor: Latency Enforcer
Monitors per-stage pipeline latency and enforces budgets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loguru import logger


# Budget limits in milliseconds (from CLAUDE.md spec)
DEFAULT_BUDGETS: dict[str, float] = {
    "asr": 500.0,
    "llm_first_token": 200.0,
    "tts_first_chunk": 300.0,
    "animation_frame": 50.0,
    "total": 1500.0,
}


@dataclass
class _StageTimer:
    """Internal tracker for a single stage invocation."""
    name: str
    start_ns: int = 0
    end_ns: int = 0
    elapsed_ms: float = 0.0
    finished: bool = False


class LatencyEnforcer:
    """Tracks wall-clock latency per pipeline stage and warns on budget overruns.

    Usage::

        enforcer = LatencyEnforcer()
        enforcer.start_stage("asr")
        # ... do ASR work ...
        enforcer.end_stage("asr")
        if not enforcer.check_budget("asr"):
            logger.warning("ASR exceeded budget")
    """

    def __init__(self, budgets: dict[str, float] | None = None) -> None:
        self._budgets: dict[str, float] = {**DEFAULT_BUDGETS, **(budgets or {})}
        self._active: dict[str, _StageTimer] = {}
        self._history: dict[str, list[float]] = {}  # stage -> list of elapsed_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_stage(self, name: str) -> None:
        """Mark the beginning of a pipeline stage."""
        self._active[name] = _StageTimer(name=name, start_ns=time.perf_counter_ns())

    def end_stage(self, name: str) -> float:
        """Mark the end of a pipeline stage. Returns elapsed ms."""
        timer = self._active.get(name)
        if timer is None:
            logger.warning("end_stage called for '{}' but no matching start_stage", name)
            return 0.0

        timer.end_ns = time.perf_counter_ns()
        timer.elapsed_ms = (timer.end_ns - timer.start_ns) / 1_000_000
        timer.finished = True

        # Store in history (rolling window of 200)
        hist = self._history.setdefault(name, [])
        hist.append(timer.elapsed_ms)
        if len(hist) > 200:
            hist.pop(0)

        # Check budget and warn
        budget = self._budgets.get(name)
        if budget is not None and timer.elapsed_ms > budget:
            logger.warning(
                "LATENCY BUDGET EXCEEDED — stage '{}': {:.1f}ms (budget {:.0f}ms)",
                name,
                timer.elapsed_ms,
                budget,
            )

        del self._active[name]
        return timer.elapsed_ms

    def check_budget(self, name: str) -> bool:
        """Return True if the *last recorded* latency for *name* is within budget.

        If no history exists yet, returns True (optimistic).
        """
        hist = self._history.get(name)
        if not hist:
            return True
        budget = self._budgets.get(name)
        if budget is None:
            return True
        return hist[-1] <= budget

    def get_last(self, name: str) -> float:
        """Return the last recorded latency (ms) for a stage, or 0.0."""
        hist = self._history.get(name)
        return hist[-1] if hist else 0.0

    def get_report(self) -> dict:
        """Return a summary report of all tracked stages.

        Returns a dict keyed by stage name, each containing:
        - last_ms, avg_ms, min_ms, max_ms, budget_ms, within_budget, count
        """
        report: dict[str, dict] = {}
        for name, hist in self._history.items():
            budget = self._budgets.get(name, float("inf"))
            last = hist[-1] if hist else 0.0
            report[name] = {
                "last_ms": round(last, 2),
                "avg_ms": round(sum(hist) / len(hist), 2) if hist else 0.0,
                "min_ms": round(min(hist), 2) if hist else 0.0,
                "max_ms": round(max(hist), 2) if hist else 0.0,
                "budget_ms": budget,
                "within_budget": last <= budget,
                "count": len(hist),
            }
        return report

    def reset(self) -> None:
        """Clear all history and active timers."""
        self._active.clear()
        self._history.clear()
