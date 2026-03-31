"""
EDEN OS -- Brain Engine: Reasoning Engine
LLM interface with streaming using Anthropic API (Claude claude-sonnet-4-20250514).
Falls back gracefully when no API key is available.
"""

import os
from typing import AsyncIterator

from loguru import logger

from eden_os.shared.types import TextChunk


# Default model
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 4096


class ReasoningEngine:
    """Streams LLM responses as TextChunk objects via Anthropic API."""

    def __init__(self, model: str = _DEFAULT_MODEL, max_tokens: int = _MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self._api_available = False
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Anthropic client if API key is present."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set. ReasoningEngine will use offline fallback."
            )
            self._api_available = False
            return
        try:
            import anthropic

            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._api_available = True
            logger.info("ReasoningEngine initialized with Anthropic API (model={})", self.model)
        except ImportError:
            logger.error("anthropic package not installed. pip install anthropic")
            self._api_available = False
        except Exception as exc:
            logger.error("Failed to initialize Anthropic client: {}", exc)
            self._api_available = False

    async def stream_response(
        self,
        user_input: str,
        system_prompt: str = "",
        conversation_history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """
        Async generator that yields raw text tokens from the LLM.
        Falls back to a canned echo response when API is unavailable.

        Args:
            user_input: The current user message.
            system_prompt: System prompt for persona injection.
            conversation_history: List of {"role": ..., "content": ...} dicts.

        Yields:
            Raw text token strings as they arrive from the LLM.
        """
        if not self._api_available or self._client is None:
            async for token in self._offline_fallback(user_input):
                yield token
            return

        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})

        try:
            kwargs: dict = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            logger.debug(
                "Streaming LLM request: model={} messages={} chars system_prompt",
                self.model,
                len(system_prompt),
            )

            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as exc:
            logger.error("Anthropic API error: {}. Falling back to offline.", exc)
            async for token in self._offline_fallback(user_input):
                yield token

    async def _offline_fallback(self, user_input: str) -> AsyncIterator[str]:
        """Simple fallback when LLM is unavailable."""
        fallback = (
            "I'm currently operating in offline mode without access to my "
            "full reasoning capabilities. I heard you say: \""
            + user_input
            + "\". Please configure an ANTHROPIC_API_KEY to enable full conversation."
        )
        # Yield word-by-word to simulate streaming
        words = fallback.split(" ")
        for i, word in enumerate(words):
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word
