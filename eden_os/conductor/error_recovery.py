"""
EDEN OS — Conductor: Error Recovery
Graceful failure handling with fallback chains per engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger


class FallbackAction(Enum):
    """Actions the conductor can take when an engine fails."""
    RETRY = "retry"
    USE_LOCAL_FALLBACK = "use_local_fallback"
    SILENT_AUDIO = "silent_audio"
    FREEZE_LAST_FRAME = "freeze_last_frame"
    SKIP = "skip"
    ABORT = "abort"


# Ordered fallback chains per engine.
# The conductor tries each action in order until one succeeds.
_FALLBACK_CHAINS: dict[str, list[FallbackAction]] = {
    "brain": [
        FallbackAction.RETRY,
        FallbackAction.USE_LOCAL_FALLBACK,  # Qwen/BitNet local LLM
        FallbackAction.ABORT,
    ],
    "voice": [
        FallbackAction.RETRY,
        FallbackAction.SILENT_AUDIO,  # Emit silent audio so animator keeps running
        FallbackAction.ABORT,
    ],
    "animator": [
        FallbackAction.RETRY,
        FallbackAction.FREEZE_LAST_FRAME,  # Hold the last good frame
        FallbackAction.ABORT,
    ],
    "genesis": [
        FallbackAction.RETRY,
        FallbackAction.ABORT,
    ],
    "scholar": [
        FallbackAction.RETRY,
        FallbackAction.SKIP,  # Non-critical — conversation can continue without knowledge
        FallbackAction.ABORT,
    ],
    "asr": [
        FallbackAction.RETRY,
        FallbackAction.SKIP,  # Fall back to text-only input
        FallbackAction.ABORT,
    ],
}

MAX_RETRIES = 2


@dataclass
class _EngineErrorState:
    total_errors: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    retries_used: int = 0


class ErrorRecovery:
    """Tracks errors per engine and determines fallback actions.

    Usage::

        recovery = ErrorRecovery()
        action = recovery.handle_error("brain", error)
        if action == FallbackAction.USE_LOCAL_FALLBACK:
            # switch to local LLM
            ...
    """

    def __init__(self) -> None:
        self._states: dict[str, _EngineErrorState] = {}

    def _get(self, engine: str) -> _EngineErrorState:
        if engine not in self._states:
            self._states[engine] = _EngineErrorState()
        return self._states[engine]

    def handle_error(self, engine_name: str, error: Exception | str) -> FallbackAction:
        """Record an error for *engine_name* and return the recommended fallback.

        The fallback progresses through the chain as consecutive errors
        accumulate (retry up to MAX_RETRIES, then next fallback, etc.).
        """
        state = self._get(engine_name)
        state.total_errors += 1
        state.consecutive_errors += 1
        state.last_error = str(error)

        chain = _FALLBACK_CHAINS.get(engine_name, [FallbackAction.ABORT])

        # Walk the chain based on how many retries have been exhausted
        for action in chain:
            if action == FallbackAction.RETRY:
                if state.retries_used < MAX_RETRIES:
                    state.retries_used += 1
                    logger.warning(
                        "Engine '{}' error (retry {}/{}): {}",
                        engine_name,
                        state.retries_used,
                        MAX_RETRIES,
                        error,
                    )
                    return FallbackAction.RETRY
                # retries exhausted — continue to next fallback
                continue
            else:
                logger.error(
                    "Engine '{}' error — fallback to {}: {}",
                    engine_name,
                    action.value,
                    error,
                )
                return action

        # Nothing left in chain
        logger.critical("Engine '{}' — all fallbacks exhausted, aborting: {}", engine_name, error)
        return FallbackAction.ABORT

    def clear_engine(self, engine_name: str) -> None:
        """Reset error state for an engine (e.g. after a successful call)."""
        state = self._get(engine_name)
        state.consecutive_errors = 0
        state.retries_used = 0

    def get_error_stats(self) -> dict[str, dict]:
        """Return error statistics for all engines that have recorded errors."""
        stats: dict[str, dict] = {}
        for name, state in self._states.items():
            stats[name] = {
                "total_errors": state.total_errors,
                "consecutive_errors": state.consecutive_errors,
                "last_error": state.last_error,
                "retries_used": state.retries_used,
            }
        return stats

    def reset(self) -> None:
        """Clear all error state."""
        self._states.clear()
