"""
EDEN OS — Conductor: Orchestrator (Agent 5)
Master controller that wires all engines together and manages the
full ASR -> Brain -> Voice TTS -> Animator pipeline.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from loguru import logger

from eden_os.shared.interfaces import IConductor
from eden_os.shared.types import (
    AudioChunk,
    AvatarState,
    PipelineMetrics,
    SessionConfig,
    TextChunk,
    VideoFrame,
)

from eden_os.conductor.session_manager import SessionManager, SessionState
from eden_os.conductor.latency_enforcer import LatencyEnforcer
from eden_os.conductor.error_recovery import ErrorRecovery, FallbackAction
from eden_os.conductor.metrics_collector import MetricsCollector


class Conductor(IConductor):
    """Agent 5 — Pipeline Orchestrator.

    Single entry point for EDEN OS.  Initialises engines lazily (at
    session-creation time) and routes data through the full pipeline::

        User audio -> ASR -> Brain LLM -> Voice TTS -> Animator -> Video frames

    Parameters
    ----------
    config : dict
        Top-level configuration.  Recognised keys:

        * ``hardware_profile`` (str) – ``"auto"`` / ``"cpu"`` / ``"cuda"`` etc.
        * ``models_cache`` (str)     – path where HF model weights are cached.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config: dict = config or {}
        self._hardware_profile: str = self._config.get("hardware_profile", "auto")
        self._models_cache: str = self._config.get("models_cache", "models_cache")

        self._sessions = SessionManager()
        self._latency = LatencyEnforcer()
        self._recovery = ErrorRecovery()
        self._metrics = MetricsCollector()

        # Engine singletons — populated lazily on first session creation
        self._brain = None
        self._voice = None
        self._animator = None
        self._genesis = None
        self._scholar = None
        self._engines_loaded = False

        logger.info(
            "Conductor initialised (hardware={}, cache={})",
            self._hardware_profile,
            self._models_cache,
        )

    # ------------------------------------------------------------------
    # Lazy engine loading (import inside method to avoid top-level deps)
    # ------------------------------------------------------------------

    def _ensure_engines(self) -> None:
        """Import and instantiate all engine singletons once."""
        if self._engines_loaded:
            return

        logger.info("Conductor: loading engines...")

        from eden_os.brain import BrainEngine
        from eden_os.voice import VoiceEngine
        from eden_os.animator import AnimatorEngine
        from eden_os.genesis import GenesisEngine
        from eden_os.scholar import ScholarEngine

        self._brain = BrainEngine()
        self._voice = VoiceEngine()
        self._animator = AnimatorEngine()
        self._genesis = GenesisEngine()
        self._scholar = ScholarEngine()
        self._engines_loaded = True

        logger.info("Conductor: all engines loaded")

    # ------------------------------------------------------------------
    # IConductor interface
    # ------------------------------------------------------------------

    async def create_session(self, config: SessionConfig) -> str:
        """Create a session: init engines, process portrait, pre-cache idle.

        Returns the session_id.
        """
        self._ensure_engines()

        sid = self._sessions.create(config)
        session = self._sessions.get(sid)
        self._sessions.set_state(sid, SessionState.INITIALIZING)

        # Store engine refs on session for convenience
        session.engines = {
            "brain": self._brain,
            "voice": self._voice,
            "animator": self._animator,
            "genesis": self._genesis,
            "scholar": self._scholar,
        }

        # Run Genesis portrait processing if an image was provided
        if config.portrait_image is not None:
            try:
                self._latency.start_stage("genesis_upload")
                profile = await self._genesis.process_upload(config.portrait_image)
                self._latency.end_stage("genesis_upload")
                session.portrait_profile = profile

                # Pre-cache idle animations so avatar is alive on load
                self._latency.start_stage("genesis_idle_cache")
                idle_cache = await self._genesis.precompute_idle_cache(profile)
                self._latency.end_stage("genesis_idle_cache")
                session.idle_cache = idle_cache

                self._recovery.clear_engine("genesis")
                logger.info("Session '{}': portrait processed & idle cached", sid)

            except Exception as exc:
                action = self._recovery.handle_error("genesis", exc)
                logger.error("Session '{}': genesis init failed ({}): {}", sid, action.value, exc)

        # Load persona if template specified
        if config.template_name and config.template_name != "default":
            try:
                await self._brain.load_persona(config.template_name)
            except Exception as exc:
                logger.warning("Session '{}': persona load failed: {}", sid, exc)

        self._sessions.set_state(sid, SessionState.READY)
        logger.info("Session '{}' is READY", sid)
        return sid

    async def start_conversation(self, session_id: str) -> None:
        """Begin the conversation: start idle animation + ASR listening."""
        session = self._sessions.get(session_id)
        self._sessions.set_state(session_id, SessionState.ACTIVE)
        session.avatar_state = AvatarState.IDLE

        logger.info("Session '{}': conversation started (IDLE + ASR listening)", session_id)

    async def end_conversation(self, session_id: str) -> None:
        """End the session and clean up resources."""
        try:
            session = self._sessions.get(session_id)
        except KeyError:
            logger.warning("end_conversation: session '{}' not found", session_id)
            return

        self._sessions.set_state(session_id, SessionState.ENDED)
        session.avatar_state = AvatarState.IDLE
        self._sessions.destroy(session_id)
        logger.info("Session '{}': conversation ended and cleaned up", session_id)

    async def handle_user_input(
        self,
        session_id: str,
        text_or_audio: str | AudioChunk,
    ) -> AsyncIterator[VideoFrame]:
        """Route user input through the full pipeline.

        Supports both text (str) and audio (AudioChunk).  For text, the
        ASR stage is skipped.

        Yields ``VideoFrame`` objects as the animator produces them.
        """
        session = self._sessions.get(session_id)

        # ----------------------------------------------------------
        # Total pipeline timer
        # ----------------------------------------------------------
        self._latency.start_stage("total")

        # ----------------------------------------------------------
        # 1. ASR (only for audio input)
        # ----------------------------------------------------------
        if isinstance(text_or_audio, AudioChunk):
            session.avatar_state = AvatarState.LISTENING
            self._latency.start_stage("asr")
            try:
                # Wrap single chunk as an async iterator for transcribe_stream
                async def _audio_iter():
                    yield text_or_audio

                transcript_parts: list[str] = []
                async for part in self._voice.transcribe_stream(_audio_iter()):
                    transcript_parts.append(part)

                user_text = " ".join(transcript_parts)
                asr_ms = self._latency.end_stage("asr")
                self._metrics.record("asr_ms", asr_ms)
                self._recovery.clear_engine("asr")

            except Exception as exc:
                self._latency.end_stage("asr")
                action = self._recovery.handle_error("asr", exc)
                self._metrics.record_error("asr")
                if action == FallbackAction.SKIP:
                    logger.warning("ASR failed, cannot process audio input")
                    self._latency.end_stage("total")
                    return
                raise
        else:
            user_text = text_or_audio

        # Store in history
        self._sessions.add_history(session_id, "user", user_text)
        await self._brain.process_user_input(user_text)

        # ----------------------------------------------------------
        # 2. Brain LLM (streaming)
        # ----------------------------------------------------------
        session.avatar_state = AvatarState.THINKING

        self._latency.start_stage("llm_first_token")
        first_token_recorded = False
        text_chunks: list[TextChunk] = []

        try:
            context = await self._brain.get_context()

            async def _llm_stream() -> AsyncIterator[TextChunk]:
                nonlocal first_token_recorded
                async for chunk in self._brain.reason_stream(user_text, context):
                    if not first_token_recorded:
                        llm_ms = self._latency.end_stage("llm_first_token")
                        self._metrics.record("llm_first_token_ms", llm_ms)
                        first_token_recorded = True
                    text_chunks.append(chunk)
                    yield chunk

            self._recovery.clear_engine("brain")

        except Exception as exc:
            if not first_token_recorded:
                self._latency.end_stage("llm_first_token")
            action = self._recovery.handle_error("brain", exc)
            self._metrics.record_error("brain")

            if action == FallbackAction.USE_LOCAL_FALLBACK:
                # Yield a canned fallback response
                fallback = TextChunk(
                    text="I'm sorry, I'm having trouble thinking right now. Could you try again?",
                    is_sentence_end=True,
                )

                async def _llm_stream():
                    yield fallback

                text_chunks.append(fallback)
            else:
                self._latency.end_stage("total")
                raise

        # ----------------------------------------------------------
        # 3. Voice TTS (streaming from LLM chunks)
        # ----------------------------------------------------------
        session.avatar_state = AvatarState.SPEAKING

        self._latency.start_stage("tts_first_chunk")
        first_tts_recorded = False

        try:
            async def _tts_audio_stream() -> AsyncIterator[AudioChunk]:
                nonlocal first_tts_recorded
                async for audio_chunk in self._voice.synthesize_stream(_llm_stream()):
                    if not first_tts_recorded:
                        tts_ms = self._latency.end_stage("tts_first_chunk")
                        self._metrics.record("tts_first_chunk_ms", tts_ms)
                        first_tts_recorded = True
                    yield audio_chunk

            self._recovery.clear_engine("voice")

        except Exception as exc:
            if not first_tts_recorded:
                self._latency.end_stage("tts_first_chunk")
            action = self._recovery.handle_error("voice", exc)
            self._metrics.record_error("voice")

            if action == FallbackAction.SILENT_AUDIO:
                import numpy as np
                silent = AudioChunk(
                    data=np.zeros(16000, dtype=np.float32),
                    sample_rate=16000,
                    duration_ms=1000.0,
                    is_final=True,
                )

                async def _tts_audio_stream():
                    yield silent
            else:
                self._latency.end_stage("total")
                raise

        # ----------------------------------------------------------
        # 4. Animator (audio -> video frames)
        # ----------------------------------------------------------
        try:
            frame_count = 0
            async for frame in self._animator.drive_from_audio(_tts_audio_stream()):
                self._latency.start_stage("animation_frame")
                elapsed = self._latency.end_stage("animation_frame")
                self._metrics.record("animation_frame_ms", elapsed)
                frame_count += 1
                yield frame

            if frame_count > 0:
                # Approximate fps from frame count (rough)
                self._metrics.record("animation_fps", float(frame_count))
            self._recovery.clear_engine("animator")

        except Exception as exc:
            action = self._recovery.handle_error("animator", exc)
            self._metrics.record_error("animator")
            if action == FallbackAction.FREEZE_LAST_FRAME:
                logger.warning("Animator failed — freezing last frame")
                # Don't raise; the pipeline simply stops yielding frames
            else:
                raise
        finally:
            total_ms = self._latency.end_stage("total")
            self._metrics.record("total_ms", total_ms)

        # Store assistant response in history
        full_response = "".join(c.text for c in text_chunks)
        self._sessions.add_history(session_id, "assistant", full_response)

        session.avatar_state = AvatarState.IDLE
        logger.debug(
            "Session '{}': pipeline complete ({:.0f}ms total, {} frames)",
            session_id,
            total_ms,
            frame_count,
        )

    async def get_metrics(self, session_id: str) -> PipelineMetrics:
        """Return current pipeline metrics for a session."""
        # Ensure session exists (will raise if not)
        self._sessions.get(session_id)
        return self._metrics.get_metrics()

    # ------------------------------------------------------------------
    # Extended API (beyond IConductor)
    # ------------------------------------------------------------------

    def get_latency_report(self) -> dict:
        """Full latency report across all stages."""
        return self._latency.get_report()

    def get_error_stats(self) -> dict:
        """Error statistics for all engines."""
        return self._recovery.get_error_stats()

    def get_metrics_summary(self) -> dict:
        """Rich metrics summary with percentiles."""
        return self._metrics.get_summary()

    def get_session_manager(self) -> SessionManager:
        """Expose session manager for Gateway integration."""
        return self._sessions
