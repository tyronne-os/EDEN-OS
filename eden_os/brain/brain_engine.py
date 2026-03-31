"""
EDEN OS -- Brain Engine: Main Engine Class
Implements IBrainEngine interface from eden_os.shared.interfaces.
Orchestrates reasoning, persona, memory, streaming bridge, and templates.
"""

from __future__ import annotations

from typing import AsyncIterator

from loguru import logger

from eden_os.shared.interfaces import IBrainEngine
from eden_os.shared.types import TextChunk
from eden_os.brain.reasoning_engine import ReasoningEngine
from eden_os.brain.persona_manager import PersonaManager
from eden_os.brain.memory_manager import MemoryManager
from eden_os.brain.streaming_bridge import StreamingBridge
from eden_os.brain.template_loader import TemplateLoader


class BrainEngine(IBrainEngine):
    """
    Agent 4: LLM Reasoning + Context Engine.

    Connects persona management, conversation memory, LLM streaming,
    and the streaming bridge that buffers tokens into speech-ready chunks.

    Usage::

        brain = BrainEngine()
        await brain.load_persona("medical_office")

        async for chunk in brain.reason_stream("Hello!", {}):
            print(chunk.text, chunk.is_sentence_end, chunk.emotion)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        templates_dir: str | None = None,
    ) -> None:
        self._reasoning = ReasoningEngine(model=model, max_tokens=max_tokens)
        self._persona = PersonaManager()
        self._memory = MemoryManager()
        self._template_loader = TemplateLoader(templates_dir=templates_dir)
        self._bridge = StreamingBridge()
        logger.info("BrainEngine initialised")

    # ------------------------------------------------------------------
    # IBrainEngine interface
    # ------------------------------------------------------------------

    async def reason_stream(
        self, user_input: str, context: dict
    ) -> AsyncIterator[TextChunk]:
        """
        Stream LLM response tokens as TextChunk objects.

        1. Stores user input in memory.
        2. Builds system prompt (persona + key facts).
        3. Streams raw tokens from the LLM.
        4. Passes tokens through the streaming bridge for sentence-boundary
           buffering and sentiment analysis.

        Args:
            user_input: The user's message text.
            context: Additional context dict (currently unused; reserved for
                Scholar RAG injection).

        Yields:
            TextChunk objects at natural speech boundaries.
        """
        # 1. Store user input
        self._memory.add_user_message(user_input)

        # 2. Build system prompt
        system_prompt = self._build_system_prompt(context)

        # 3. Get conversation history (excluding the message we just added,
        #    since the reasoning engine appends user_input itself)
        history = self._memory.get_history_for_llm()
        # Remove the last entry (the user_input we just added) because
        # ReasoningEngine.stream_response appends it.
        if history and history[-1]["role"] == "user":
            history = history[:-1]

        # 4. Stream tokens through bridge
        token_stream = self._reasoning.stream_response(
            user_input=user_input,
            system_prompt=system_prompt,
            conversation_history=history if history else None,
        )

        full_response_parts: list[str] = []

        async for chunk in self._bridge.bridge(token_stream):
            full_response_parts.append(chunk.text)
            yield chunk

        # 5. Store full assistant response in memory
        full_response = " ".join(full_response_parts)
        if full_response.strip():
            self._memory.add_assistant_message(full_response)

    async def load_persona(self, template_path: str) -> None:
        """
        Load agent persona from a YAML template.

        Args:
            template_path: Template name (e.g. ``"medical_office"``) or
                full path to a YAML file.
        """
        try:
            config = self._template_loader.load(template_path)
        except FileNotFoundError:
            # Try as absolute path
            config = self._template_loader.load_from_path(template_path)

        await self._persona.load(config)

        # Update streaming bridge with new emotional baseline
        self._bridge = StreamingBridge(
            emotion_baseline=self._persona.emotional_baseline
        )
        logger.info("Persona '{}' loaded into BrainEngine", self._persona.name)

    async def get_context(self) -> dict:
        """
        Get current conversation context and memory.

        Returns:
            Dict with keys: conversation_history, key_facts,
            turn_count, window_size, persona.
        """
        ctx = self._memory.get_context()
        ctx["persona"] = {
            "name": self._persona.name,
            "role": self._persona.role,
            "loaded": self._persona.is_loaded,
        }
        return ctx

    async def process_user_input(self, text: str) -> None:
        """
        Process and store user input in conversation history.

        This is used when the caller wants to record user input
        without triggering LLM reasoning (e.g., during ASR
        partial transcripts).
        """
        self._memory.add_user_message(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, context: dict) -> str:
        """
        Assemble the full system prompt from persona + memory facts + context.
        """
        parts: list[str] = []

        # Persona system prompt
        if self._persona.is_loaded:
            parts.append(self._persona.system_prompt)
        else:
            parts.append(
                "You are EVE, a friendly and helpful conversational AI assistant "
                "built by EDEN OS. You are warm, clear, and professional. "
                "Respond naturally and conversationally."
            )

        # Key facts from memory
        facts_section = self._memory.get_facts_prompt_section()
        if facts_section:
            parts.append(facts_section)

        # Injected context (e.g., from Scholar RAG)
        if context.get("knowledge_context"):
            parts.append(
                "\n\n<knowledge_context>\n"
                + str(context["knowledge_context"])
                + "\n</knowledge_context>"
            )

        return "\n".join(parts)
