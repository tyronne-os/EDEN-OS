"""
EDEN OS -- Brain Engine: Memory Manager
Sliding-window conversation history (last 20 turns).
Stores user/assistant messages. Extracts and stores key facts.
Returns formatted context for LLM.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


_MAX_TURNS = 20

# Simple patterns for extracting key facts from user messages
_FACT_PATTERNS = [
    # "My name is X" / "I'm X" / "I am X"
    re.compile(r"\bmy name is (\w[\w\s]{0,30})", re.IGNORECASE),
    re.compile(r"\bi(?:'m| am) (\w[\w\s]{0,30})", re.IGNORECASE),
    # "I have X" / "I work at X" / "I live in X"
    re.compile(r"\bi (?:have|work at|live in|am from) (.{3,50})", re.IGNORECASE),
    # "My X is Y"
    re.compile(r"\bmy (\w+) is (.{2,40})", re.IGNORECASE),
    # Age
    re.compile(r"\bi(?:'m| am) (\d{1,3}) years old", re.IGNORECASE),
    # Email
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    # Phone
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
]


@dataclass
class _Turn:
    """A single conversation turn."""
    role: str  # "user" or "assistant"
    content: str


class MemoryManager:
    """
    Manages conversation history with a sliding window and key-fact extraction.

    Keeps the most recent ``max_turns`` exchanges (user + assistant pairs)
    and a set of extracted key facts that persist for the session.
    """

    def __init__(self, max_turns: int = _MAX_TURNS) -> None:
        self.max_turns = max_turns
        self._history: deque[_Turn] = deque(maxlen=max_turns * 2)
        self._key_facts: list[str] = []
        self._turn_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """Store a user message and extract any key facts."""
        text = text.strip()
        if not text:
            return
        self._history.append(_Turn(role="user", content=text))
        self._extract_facts(text)
        self._turn_count += 1
        logger.debug("User message stored (turn {}): {}...", self._turn_count, text[:60])

    def add_assistant_message(self, text: str) -> None:
        """Store an assistant (avatar) response."""
        text = text.strip()
        if not text:
            return
        self._history.append(_Turn(role="assistant", content=text))
        logger.debug("Assistant message stored: {}...", text[:60])

    def get_history_for_llm(self) -> list[dict[str, str]]:
        """
        Return conversation history formatted for the Anthropic messages API.

        Returns:
            List of ``{"role": ..., "content": ...}`` dicts.
        """
        return [{"role": t.role, "content": t.content} for t in self._history]

    def get_context(self) -> dict[str, Any]:
        """
        Return the full context dict for the Brain Engine.

        Includes conversation history, key facts, and metadata.
        """
        return {
            "conversation_history": self.get_history_for_llm(),
            "key_facts": list(self._key_facts),
            "turn_count": self._turn_count,
            "window_size": self.max_turns,
        }

    def get_key_facts(self) -> list[str]:
        """Return extracted key facts."""
        return list(self._key_facts)

    def get_facts_prompt_section(self) -> str:
        """
        Format key facts as a prompt section to inject into the system prompt.

        Returns:
            A formatted string, or empty string if no facts.
        """
        if not self._key_facts:
            return ""
        facts_str = "\n".join(f"- {f}" for f in self._key_facts)
        return (
            "\n\n<user_context>\n"
            "Key facts about the current user gathered from this conversation:\n"
            f"{facts_str}\n"
            "</user_context>"
        )

    def clear(self) -> None:
        """Clear all history and facts."""
        self._history.clear()
        self._key_facts.clear()
        self._turn_count = 0
        logger.info("Memory cleared")

    @property
    def turn_count(self) -> int:
        return self._turn_count

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------

    def _extract_facts(self, text: str) -> None:
        """Extract key facts from user text using pattern matching."""
        for pattern in _FACT_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    fact = " ".join(part.strip() for part in match if part.strip())
                else:
                    fact = match.strip()

                # Deduplicate (case-insensitive)
                if fact and not any(
                    fact.lower() == existing.lower() for existing in self._key_facts
                ):
                    self._key_facts.append(fact)
                    logger.info("Key fact extracted: '{}'", fact)
