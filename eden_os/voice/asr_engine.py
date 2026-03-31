"""
EDEN OS -- Voice Engine: ASR Engine
Real-time speech-to-text using OpenAI Whisper with Silero VAD
for voice-activity / endpoint detection.
Supports both batch (full audio) and streaming-style (chunked) transcription.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from eden_os.shared.types import AudioChunk

# ---------------------------------------------------------------------------
# Lazy-loaded globals -- heavy models are only imported/loaded on first use
# ---------------------------------------------------------------------------
_whisper_model = None
_vad_model = None
_vad_utils: Optional[dict] = None


def _load_whisper(model_name: str = "base") -> object:
    """Lazily load and cache the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper model '{}' ...", model_name)
        try:
            import whisper  # type: ignore[import-untyped]

            _whisper_model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load Whisper model: {}", exc)
            raise
    return _whisper_model


def _load_vad() -> tuple:
    """Lazily load Silero VAD model and utilities."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        logger.info("Loading Silero VAD ...")
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers5/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            _vad_model = model
            _vad_utils = {
                "get_speech_timestamps": utils[0],
                "save_audio": utils[1],
                "read_audio": utils[2],
                "VADIterator": utils[3],
                "collect_chunks": utils[4],
            }
            logger.info("Silero VAD loaded successfully")
        except Exception as exc:
            logger.warning("Silero VAD unavailable, falling back to energy VAD: {}", exc)
            _vad_model = "fallback"
            _vad_utils = None
    return _vad_model, _vad_utils


class ASREngine:
    """Real-time Automatic Speech Recognition engine.

    Uses OpenAI Whisper for transcription and Silero VAD for
    voice-activity / endpoint detection.
    """

    def __init__(
        self,
        whisper_model_name: str = "base",
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        silence_timeout_ms: float = 800.0,
        energy_threshold: float = 0.01,
    ) -> None:
        self._whisper_model_name = whisper_model_name
        self._sample_rate = sample_rate
        self._vad_threshold = vad_threshold
        self._silence_timeout_ms = silence_timeout_ms
        self._energy_threshold = energy_threshold

        # Streaming state
        self._audio_buffer: list[np.ndarray] = []
        self._is_speech_active = False
        self._silence_start_ms: Optional[float] = None
        self._total_buffered_ms: float = 0.0

        logger.info(
            "ASREngine created  model={}  sr={}",
            whisper_model_name,
            sample_rate,
        )

    # ------------------------------------------------------------------
    # Lazy model access
    # ------------------------------------------------------------------

    def _get_whisper(self):
        return _load_whisper(self._whisper_model_name)

    def _get_vad(self):
        return _load_vad()

    # ------------------------------------------------------------------
    # VAD helpers
    # ------------------------------------------------------------------

    def _energy_vad(self, audio: np.ndarray) -> bool:
        """Simple energy-based VAD fallback."""
        samples = audio.astype(np.float32)
        if np.max(np.abs(samples)) > 2.0:
            samples = samples / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        return rms >= self._energy_threshold

    def _silero_vad_check(self, audio: np.ndarray) -> bool:
        """Run Silero VAD on a chunk and return speech probability."""
        vad_model, vad_utils = self._get_vad()
        if vad_model == "fallback" or vad_utils is None:
            return self._energy_vad(audio)

        try:
            import torch

            samples = audio.astype(np.float32)
            if np.max(np.abs(samples)) > 2.0:
                samples = samples / 32768.0
            tensor = torch.from_numpy(samples)
            if tensor.dim() > 1:
                tensor = tensor.squeeze()
            speech_prob = vad_model(tensor, self._sample_rate).item()
            return speech_prob >= self._vad_threshold
        except Exception:
            return self._energy_vad(audio)

    # ------------------------------------------------------------------
    # Batch transcription
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """Transcribe a complete audio array to text.

        Args:
            audio: PCM samples (float32 or int16), mono, at self._sample_rate.
            language: Language hint for Whisper.

        Returns:
            Transcribed text string.
        """
        model = self._get_whisper()

        samples = audio.astype(np.float32)
        if np.max(np.abs(samples)) > 2.0:
            samples = samples / 32768.0

        # Run Whisper in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                samples,
                language=language,
                fp16=False,
            ),
        )
        text = result.get("text", "").strip()
        logger.debug("Transcribed ({:.1f}s audio): '{}'", len(samples) / self._sample_rate, text)
        return text

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    async def process_chunk(self, chunk: AudioChunk) -> Optional[str]:
        """Feed a single audio chunk for streaming transcription.

        Accumulates audio while speech is detected. When silence is
        sustained past the timeout, transcribes the buffered audio and
        returns the text. Returns None while still accumulating.

        Args:
            chunk: An AudioChunk with PCM data.

        Returns:
            Transcribed text when an utterance boundary is detected,
            None otherwise.
        """
        has_speech = self._silero_vad_check(chunk.data)

        if has_speech:
            self._is_speech_active = True
            self._silence_start_ms = None
            self._audio_buffer.append(chunk.data)
            self._total_buffered_ms += chunk.duration_ms
        else:
            if self._is_speech_active:
                # Speech was active, now silence
                if self._silence_start_ms is None:
                    self._silence_start_ms = self._total_buffered_ms
                    self._audio_buffer.append(chunk.data)
                    self._total_buffered_ms += chunk.duration_ms
                else:
                    self._audio_buffer.append(chunk.data)
                    self._total_buffered_ms += chunk.duration_ms
                    silence_duration = self._total_buffered_ms - self._silence_start_ms
                    if silence_duration >= self._silence_timeout_ms:
                        # Endpoint detected -- transcribe buffer
                        return await self._flush_buffer()
            # else: no speech yet, discard

        # Force flush if buffer is very long (>30s)
        if self._total_buffered_ms > 30000.0:
            return await self._flush_buffer()

        return None

    async def _flush_buffer(self) -> str:
        """Transcribe accumulated buffer and reset state."""
        if not self._audio_buffer:
            self._reset_stream()
            return ""

        combined = np.concatenate(self._audio_buffer)
        self._reset_stream()
        return await self.transcribe(combined)

    def _reset_stream(self) -> None:
        self._audio_buffer.clear()
        self._is_speech_active = False
        self._silence_start_ms = None
        self._total_buffered_ms = 0.0

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[str]:
        """Consume an async stream of AudioChunks and yield transcribed text.

        Yields text each time a speech endpoint is detected.
        """
        self._reset_stream()
        async for chunk in audio_stream:
            result = await self.process_chunk(chunk)
            if result is not None and result.strip():
                yield result

            if chunk.is_final:
                # Flush remaining buffer
                tail = await self._flush_buffer()
                if tail.strip():
                    yield tail
                break

    def reset(self) -> None:
        """Reset streaming state."""
        self._reset_stream()
        logger.debug("ASREngine stream state reset")
