"""
EDEN OS -- VoiceEngine
Composite engine that implements IVoiceEngine by composing
ASREngine, TTSEngine, VoiceCloner, EmotionRouter, and InterruptionHandler.
"""

from __future__ import annotations

from typing import AsyncIterator

import numpy as np
from loguru import logger

from eden_os.shared.interfaces import IVoiceEngine
from eden_os.shared.types import AudioChunk, TextChunk
from eden_os.voice.asr_engine import ASREngine
from eden_os.voice.tts_engine import TTSEngine
from eden_os.voice.voice_cloner import VoiceCloner
from eden_os.voice.emotion_router import EmotionRouter
from eden_os.voice.interruption_handler import InterruptionHandler


class VoiceEngine(IVoiceEngine):
    """Agent 2: TTS + ASR + Voice Cloning.

    Composes all voice sub-modules and exposes the IVoiceEngine
    interface methods plus convenience accessors for sub-engines.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        sample_rate: int = 16000,
        tts_sample_rate: int = 22050,
        tts_speed: float = 1.0,
        tts_pitch: float = 0.0,
    ) -> None:
        self.asr = ASREngine(
            whisper_model_name=whisper_model,
            sample_rate=sample_rate,
        )
        self.tts = TTSEngine(
            sample_rate=tts_sample_rate,
            speed=tts_speed,
            pitch_shift=tts_pitch,
        )
        self.cloner = VoiceCloner()
        self.emotion = EmotionRouter()
        self.interruption = InterruptionHandler()

        # Wire interruption handler to halt TTS
        self.interruption.on_interrupt(self.tts.halt)

        logger.info("VoiceEngine initialised")

    # ------------------------------------------------------------------
    # IVoiceEngine: transcribe_stream
    # ------------------------------------------------------------------

    async def transcribe_stream(
        self, audio: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[str]:
        """Real-time speech-to-text from streaming audio.

        Yields transcribed text each time a speech endpoint is detected.
        """
        async for text in self.asr.transcribe_stream(audio):
            yield text

    # ------------------------------------------------------------------
    # IVoiceEngine: synthesize_stream
    # ------------------------------------------------------------------

    async def synthesize_stream(
        self, text: AsyncIterator[TextChunk]
    ) -> AsyncIterator[AudioChunk]:
        """Stream TTS audio from text chunks.

        Enriches each TextChunk with emotion analysis before synthesis.
        Signals the interruption handler that the avatar is speaking.
        """
        self.interruption.set_avatar_speaking(True)
        self.tts.resume()

        try:
            async def _emotion_enriched(
                source: AsyncIterator[TextChunk],
            ) -> AsyncIterator[TextChunk]:
                async for chunk in source:
                    # Run emotion analysis on the text
                    analysed_emotion = self.emotion.analyze(chunk.text)
                    # Merge with existing emotion (analysis overrides defaults)
                    merged = dict(chunk.emotion)
                    merged.update(analysed_emotion)
                    yield TextChunk(
                        text=chunk.text,
                        is_sentence_end=chunk.is_sentence_end,
                        emotion=merged,
                    )

            async for audio_chunk in self.tts.synthesize_stream(
                _emotion_enriched(text)
            ):
                if self.interruption.was_interrupted:
                    logger.info("Synthesis halted due to interruption")
                    break
                yield audio_chunk
        finally:
            self.interruption.set_avatar_speaking(False)

    # ------------------------------------------------------------------
    # IVoiceEngine: detect_interruption
    # ------------------------------------------------------------------

    async def detect_interruption(self, audio: AudioChunk) -> bool:
        """Detect if user has started speaking (interrupt)."""
        return await self.interruption.detect(audio)

    # ------------------------------------------------------------------
    # IVoiceEngine: clone_voice
    # ------------------------------------------------------------------

    async def clone_voice(self, reference_audio: np.ndarray) -> str:
        """Clone voice from reference audio, return voice_id."""
        voice_id = self.cloner.clone_voice(reference_audio)
        # Set the cloned voice as active for TTS
        embedding = self.cloner.get_voice_embedding(voice_id)
        self.tts.set_voice(voice_id, embedding)
        return voice_id

    # ------------------------------------------------------------------
    # Convenience methods (not in interface but useful for Conductor)
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """Batch transcribe a full audio array."""
        return await self.asr.transcribe(audio, language=language)

    def set_emotion_baseline(self, baseline: dict) -> None:
        """Set the emotion router baseline (e.g. from persona config)."""
        self.emotion.set_baseline(baseline)

    def list_voices(self) -> list:
        """List all cloned voice profiles."""
        return self.cloner.list_voices()

    def set_tts_speed(self, speed: float) -> None:
        self.tts.set_speed(speed)

    def set_tts_pitch(self, semitones: float) -> None:
        self.tts.set_pitch(semitones)
