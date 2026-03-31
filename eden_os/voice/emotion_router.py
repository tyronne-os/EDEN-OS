"""
EDEN OS -- Voice Engine: Emotion Router
Keyword/pattern-based sentiment analysis to derive emotion parameters from text.
Maps text to emotion dict {joy, sadness, confidence, urgency, warmth} each 0.0-1.0.
"""

import re
from typing import Dict

from loguru import logger


# ---------------------------------------------------------------------------
# Keyword banks -- each word maps to a partial emotion delta
# ---------------------------------------------------------------------------

_JOY_WORDS = {
    "happy", "glad", "wonderful", "amazing", "fantastic", "great", "love",
    "excited", "thrilled", "delighted", "excellent", "awesome", "joy",
    "beautiful", "brilliant", "celebrate", "cheers", "fun", "laugh",
    "smile", "pleased", "perfect", "magnificent", "hooray", "yay",
    "congrats", "congratulations", "bravo", "superb", "terrific",
}

_SADNESS_WORDS = {
    "sad", "sorry", "unfortunately", "regret", "loss", "grief", "mourn",
    "tragic", "devastating", "heartbroken", "disappointed", "depressed",
    "unhappy", "miserable", "painful", "suffer", "cry", "tears", "sorrow",
    "lonely", "hopeless", "despair", "condolence", "sympathy", "miss",
}

_CONFIDENCE_WORDS = {
    "certainly", "absolutely", "definitely", "sure", "confident", "proven",
    "evidence", "fact", "clear", "obvious", "undoubtedly", "indeed",
    "precisely", "exactly", "correct", "right", "accurate", "guarantee",
    "assured", "positive", "determined", "affirm", "strong", "know",
}

_URGENCY_WORDS = {
    "urgent", "immediately", "now", "hurry", "asap", "critical", "emergency",
    "important", "deadline", "quickly", "fast", "rush", "alert", "warning",
    "danger", "crucial", "vital", "priority", "time-sensitive", "must",
    "need", "required", "essential", "pressing", "imperative",
}

_WARMTH_WORDS = {
    "welcome", "thank", "thanks", "appreciate", "kind", "care", "help",
    "support", "understand", "comfort", "gentle", "friend", "dear",
    "pleasure", "glad", "here for you", "together", "safe", "trust",
    "compassion", "empathy", "hug", "embrace", "thoughtful", "generous",
    "patient", "listen", "reassure", "encourage",
}

# Patterns that boost certain emotions
_QUESTION_PATTERN = re.compile(r"\?")
_EXCLAMATION_PATTERN = re.compile(r"!")
_ELLIPSIS_PATTERN = re.compile(r"\.\.\.")
_ALL_CAPS_PATTERN = re.compile(r"\b[A-Z]{2,}\b")


class EmotionRouter:
    """Derives emotion parameters from text using keyword/pattern analysis."""

    def __init__(self) -> None:
        self._baseline: Dict[str, float] = {
            "joy": 0.5,
            "sadness": 0.0,
            "confidence": 0.7,
            "urgency": 0.0,
            "warmth": 0.6,
        }
        logger.info("EmotionRouter initialised with baseline emotions")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> Dict[str, float]:
        """Return emotion dict for the given text.

        Each value is clamped to [0.0, 1.0].
        """
        if not text or not text.strip():
            return dict(self._baseline)

        words = set(re.findall(r"[a-z]+", text.lower()))
        word_count = max(len(words), 1)

        # Count keyword hits
        joy_hits = len(words & _JOY_WORDS)
        sad_hits = len(words & _SADNESS_WORDS)
        conf_hits = len(words & _CONFIDENCE_WORDS)
        urg_hits = len(words & _URGENCY_WORDS)
        warm_hits = len(words & _WARMTH_WORDS)

        # Normalise hits relative to word count (cap contribution at 0.5)
        def _score(hits: int) -> float:
            return min(hits / max(word_count * 0.15, 1.0), 0.5)

        emotions: Dict[str, float] = {
            "joy": self._baseline["joy"] + _score(joy_hits),
            "sadness": self._baseline["sadness"] + _score(sad_hits),
            "confidence": self._baseline["confidence"] + _score(conf_hits),
            "urgency": self._baseline["urgency"] + _score(urg_hits),
            "warmth": self._baseline["warmth"] + _score(warm_hits),
        }

        # Pattern-based adjustments
        if _EXCLAMATION_PATTERN.search(text):
            emotions["urgency"] += 0.1
            emotions["confidence"] += 0.05
        if _QUESTION_PATTERN.search(text):
            emotions["confidence"] -= 0.1
            emotions["warmth"] += 0.05
        if _ELLIPSIS_PATTERN.search(text):
            emotions["sadness"] += 0.05
            emotions["confidence"] -= 0.05
        caps_count = len(_ALL_CAPS_PATTERN.findall(text))
        if caps_count >= 2:
            emotions["urgency"] += 0.15

        # Sadness suppresses joy and vice-versa
        if emotions["sadness"] > 0.4:
            emotions["joy"] = max(emotions["joy"] - 0.2, 0.0)
        if emotions["joy"] > 0.7:
            emotions["sadness"] = max(emotions["sadness"] - 0.1, 0.0)

        # Clamp all values
        for k in emotions:
            emotions[k] = max(0.0, min(1.0, emotions[k]))

        logger.debug("Emotion analysis: {}", emotions)
        return emotions

    def set_baseline(self, baseline: Dict[str, float]) -> None:
        """Override the default baseline emotions (e.g. from persona config)."""
        for k in self._baseline:
            if k in baseline:
                self._baseline[k] = max(0.0, min(1.0, baseline[k]))
        logger.info("Emotion baseline updated: {}", self._baseline)

    def get_baseline(self) -> Dict[str, float]:
        return dict(self._baseline)
