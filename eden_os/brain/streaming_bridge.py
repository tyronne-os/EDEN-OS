"""
EDEN OS -- Brain Engine: Streaming Bridge
Buffers LLM tokens until natural speech boundaries (sentence end, comma pause).
Yields TextChunk with is_sentence_end=True at boundaries.
Analyzes sentiment of each chunk for the emotion dict.
"""

from __future__ import annotations

import re
from typing import AsyncIterator

from loguru import logger

from eden_os.shared.types import TextChunk


# Sentence-ending punctuation
_SENTENCE_END = re.compile(r"[.!?]+\s*$")
# Clause-boundary punctuation (comma, semicolon, colon, dash)
_CLAUSE_BREAK = re.compile(r"[,;:\u2014—-]+\s*$")

# Simple keyword lists for lightweight sentiment analysis
_JOY_WORDS = {
    "happy", "glad", "great", "wonderful", "excellent", "love", "amazing",
    "fantastic", "beautiful", "delighted", "excited", "joy", "pleased",
    "cheerful", "thrilled", "awesome", "perfect", "brilliant",
}
_SAD_WORDS = {
    "sorry", "sad", "unfortunately", "regret", "loss", "difficult",
    "painful", "tragic", "grief", "mourn", "disappoint", "unhappy",
}
_CONFIDENCE_WORDS = {
    "certainly", "absolutely", "definitely", "sure", "confident",
    "clearly", "obviously", "undoubtedly", "indeed", "precisely",
    "exactly", "correct", "right",
}
_URGENCY_WORDS = {
    "immediately", "urgent", "asap", "critical", "emergency", "hurry",
    "quickly", "right away", "important", "now", "crucial",
}
_WARMTH_WORDS = {
    "welcome", "thank", "please", "care", "help", "support", "understand",
    "appreciate", "kind", "gentle", "warm", "comfort", "safe", "here for you",
}

# Minimum buffer size before we consider flushing at a clause break
_MIN_CLAUSE_BUFFER = 15
# Maximum buffer size -- force flush regardless
_MAX_BUFFER = 300


class StreamingBridge:
    """
    Buffers streaming LLM tokens and yields TextChunk objects
    at natural speech boundaries with per-chunk sentiment.
    """

    def __init__(
        self,
        emotion_baseline: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            emotion_baseline: Default emotion values from the persona.
                Sentiment analysis adjusts these values up/down.
        """
        self._baseline = emotion_baseline or {
            "joy": 0.5,
            "sadness": 0.0,
            "confidence": 0.7,
            "urgency": 0.0,
            "warmth": 0.6,
        }

    async def bridge(
        self,
        token_stream: AsyncIterator[str],
    ) -> AsyncIterator[TextChunk]:
        """
        Consume raw token strings and yield TextChunk objects
        at natural speech boundaries.

        Args:
            token_stream: Async iterator of raw LLM token strings.

        Yields:
            TextChunk with text, is_sentence_end flag, and emotion dict.
        """
        buffer = ""

        async for token in token_stream:
            buffer += token

            # Check for sentence end
            if _SENTENCE_END.search(buffer):
                chunk = self._make_chunk(buffer.strip(), is_sentence_end=True)
                logger.debug("Sentence chunk: '{}'", buffer.strip()[:60])
                buffer = ""
                yield chunk
                continue

            # Check for clause break (only if buffer is long enough)
            if len(buffer) >= _MIN_CLAUSE_BUFFER and _CLAUSE_BREAK.search(buffer):
                chunk = self._make_chunk(buffer.strip(), is_sentence_end=False)
                logger.debug("Clause chunk: '{}'", buffer.strip()[:60])
                buffer = ""
                yield chunk
                continue

            # Force flush on max buffer
            if len(buffer) >= _MAX_BUFFER:
                chunk = self._make_chunk(buffer.strip(), is_sentence_end=False)
                logger.debug("Max-buffer chunk: '{}'", buffer.strip()[:60])
                buffer = ""
                yield chunk

        # Flush remaining buffer
        if buffer.strip():
            chunk = self._make_chunk(buffer.strip(), is_sentence_end=True)
            logger.debug("Final chunk: '{}'", buffer.strip()[:60])
            yield chunk

    def _make_chunk(self, text: str, is_sentence_end: bool) -> TextChunk:
        """Create a TextChunk with sentiment-analyzed emotion dict."""
        emotion = self._analyze_sentiment(text)
        return TextChunk(
            text=text,
            is_sentence_end=is_sentence_end,
            emotion=emotion,
        )

    def _analyze_sentiment(self, text: str) -> dict[str, float]:
        """
        Lightweight keyword-based sentiment analysis.

        Starts from the persona's emotional baseline and adjusts
        each dimension based on keyword presence in the text.
        """
        words = set(text.lower().split())
        emotion = dict(self._baseline)

        # Count keyword hits per dimension
        joy_hits = len(words & _JOY_WORDS)
        sad_hits = len(words & _SAD_WORDS)
        conf_hits = len(words & _CONFIDENCE_WORDS)
        urg_hits = len(words & _URGENCY_WORDS)
        warm_hits = len(words & _WARMTH_WORDS)

        # Adjust: each hit shifts the value by 0.1, clamped to [0.0, 1.0]
        _bump = 0.1
        emotion["joy"] = _clamp(emotion["joy"] + joy_hits * _bump - sad_hits * _bump)
        emotion["sadness"] = _clamp(emotion["sadness"] + sad_hits * _bump - joy_hits * 0.05)
        emotion["confidence"] = _clamp(emotion["confidence"] + conf_hits * _bump)
        emotion["urgency"] = _clamp(emotion["urgency"] + urg_hits * _bump)
        emotion["warmth"] = _clamp(emotion["warmth"] + warm_hits * _bump)

        return emotion


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))
