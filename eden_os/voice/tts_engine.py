"""
EDEN OS -- Voice Engine: TTS Engine
Text-to-speech with streaming AudioChunk output.
Primary: torch-based synthesis. Fallback: silent audio generator.
Supports voice parameter control (speed, pitch via resampling).
"""

from __future__ import annotations

import asyncio
import math
from typing import AsyncIterator, Dict, Optional

import numpy as np
from loguru import logger

from eden_os.shared.types import AudioChunk, TextChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio via linear interpolation (lightweight, no librosa)."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_len = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _apply_speed(audio: np.ndarray, speed: float, sr: int) -> np.ndarray:
    """Change playback speed by resampling.

    speed > 1.0 = faster, speed < 1.0 = slower.
    """
    if abs(speed - 1.0) < 0.01:
        return audio
    # Resample to stretch/shrink, then back to original sr
    intermediate_sr = int(sr * speed)
    return _resample(audio, intermediate_sr, sr)


def _apply_pitch(audio: np.ndarray, pitch_shift: float, sr: int) -> np.ndarray:
    """Shift pitch by resampling up/down then truncating.

    pitch_shift > 0 = higher pitch, < 0 = lower pitch.
    Units: semitones.
    """
    if abs(pitch_shift) < 0.1:
        return audio
    factor = 2.0 ** (pitch_shift / 12.0)
    resampled_sr = int(sr * factor)
    shifted = _resample(audio, sr, resampled_sr)
    # Trim/pad to original length
    if len(shifted) > len(audio):
        shifted = shifted[: len(audio)]
    elif len(shifted) < len(audio):
        shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
    return shifted


def _generate_sine_speech(
    text: str,
    sample_rate: int = 22050,
    base_freq: float = 180.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a simple tonal waveform whose length is proportional to text.

    This is a deterministic torch-free fallback that produces audible
    output shaped by the text content (different characters modulate
    frequency slightly, giving a 'speaking' feel).
    """
    # ~80ms per character, minimum 200ms
    duration_s = max(len(text) * 0.08, 0.2)
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n_samples, dtype=np.float32)

    # Modulate frequency by character codes for variation
    char_codes = [ord(c) % 30 for c in text] if text else [0]
    # Build a slow frequency envelope
    freq_env = np.interp(
        np.linspace(0, len(char_codes) - 1, n_samples),
        np.arange(len(char_codes)),
        [base_freq + c * 3.0 for c in char_codes],
    ).astype(np.float32)

    # Synthesise with amplitude envelope (fade in/out)
    phase = np.cumsum(2.0 * np.pi * freq_env / sample_rate).astype(np.float32)
    wave = amplitude * np.sin(phase)

    # Fade in/out (10ms)
    fade_samples = min(int(0.01 * sample_rate), n_samples // 4)
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        wave[:fade_samples] *= fade_in
        wave[-fade_samples:] *= fade_out

    return wave


def _generate_silence(duration_s: float, sample_rate: int = 22050) -> np.ndarray:
    """Generate silent audio buffer."""
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


# ---------------------------------------------------------------------------
# Torch TTS backend (lazy-loaded)
# ---------------------------------------------------------------------------

_torch_tts = None
_torch_tts_failed = False


def _load_torch_tts():
    """Attempt to load a torch-based TTS model.

    Tries to import and set up a basic torch TTS pipeline.
    Falls back gracefully if unavailable.
    """
    global _torch_tts, _torch_tts_failed
    if _torch_tts_failed:
        return None
    if _torch_tts is not None:
        return _torch_tts

    try:
        import torch

        # Try loading a pre-trained TTS vocoder (e.g. from torchaudio)
        try:
            import torchaudio

            # Attempt to use torchaudio's Tacotron2/WaveRNN bundle
            bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
            processor = bundle.get_text_processor()
            tacotron2 = bundle.get_tacotron2()
            vocoder = bundle.get_vocoder()
            _torch_tts = {
                "processor": processor,
                "tacotron2": tacotron2,
                "vocoder": vocoder,
                "sample_rate": bundle.sample_rate,
                "type": "tacotron2",
            }
            logger.info("Torch TTS loaded: Tacotron2 + WaveRNN  sr={}", bundle.sample_rate)
            return _torch_tts
        except Exception as e1:
            logger.debug("Tacotron2 bundle unavailable: {}", e1)

        # If torchaudio pipeline unavailable, mark as failed
        _torch_tts_failed = True
        logger.info("No torch TTS backend available, using sine-wave fallback")
        return None

    except ImportError:
        _torch_tts_failed = True
        logger.info("torch not available, using sine-wave fallback")
        return None


def _torch_synthesize(text: str, tts_bundle: dict) -> tuple[np.ndarray, int]:
    """Synthesise audio using the loaded torch TTS pipeline.

    Returns (audio_array, sample_rate).
    """
    import torch

    if tts_bundle["type"] == "tacotron2":
        processor = tts_bundle["processor"]
        tacotron2 = tts_bundle["tacotron2"]
        vocoder = tts_bundle["vocoder"]
        sr = tts_bundle["sample_rate"]

        with torch.inference_mode():
            processed, lengths = processor(text)
            spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
            waveforms, _ = vocoder(spec, spec_lengths)
            audio = waveforms.squeeze(0).cpu().numpy()
        return audio, sr

    raise RuntimeError(f"Unknown TTS type: {tts_bundle['type']}")


# ---------------------------------------------------------------------------
# TTSEngine
# ---------------------------------------------------------------------------


class TTSEngine:
    """Text-to-Speech engine with streaming AudioChunk output.

    Tries torch-based TTS first, falls back to sine-wave synthesis,
    and ultimately to silence if everything fails.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        chunk_duration_ms: float = 100.0,
        speed: float = 1.0,
        pitch_shift: float = 0.0,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_duration_ms = chunk_duration_ms
        self._speed = speed
        self._pitch_shift = pitch_shift
        self._voice_id: Optional[str] = None
        self._voice_embedding: Optional[np.ndarray] = None
        self._halt_requested = False

        logger.info(
            "TTSEngine created  sr={}  speed={}  pitch={}",
            sample_rate,
            speed,
            pitch_shift,
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.25, min(4.0, speed))

    def set_pitch(self, semitones: float) -> None:
        self._pitch_shift = max(-12.0, min(12.0, semitones))

    def set_voice(self, voice_id: str, embedding: Optional[np.ndarray] = None) -> None:
        self._voice_id = voice_id
        self._voice_embedding = embedding
        logger.info("TTS voice set to '{}'", voice_id)

    def halt(self) -> None:
        """Signal the engine to stop current synthesis."""
        self._halt_requested = True
        logger.debug("TTS halt requested")

    def resume(self) -> None:
        self._halt_requested = False

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def _synthesize_text(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesise a single text string to audio.

        Returns (audio_np, sample_rate).
        """
        # Try torch TTS
        tts = _load_torch_tts()
        if tts is not None:
            try:
                audio, sr = _torch_synthesize(text, tts)
                return audio, sr
            except Exception as exc:
                logger.warning("Torch TTS synthesis failed, falling back: {}", exc)

        # Fallback: sine-wave speech
        try:
            audio = _generate_sine_speech(text, sample_rate=self._sample_rate)
            return audio, self._sample_rate
        except Exception as exc:
            logger.error("Sine-wave fallback failed, returning silence: {}", exc)
            return _generate_silence(0.5, self._sample_rate), self._sample_rate

    def _post_process(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        """Apply speed / pitch controls and resample to target sr."""
        audio = _apply_speed(audio, self._speed, sr)
        audio = _apply_pitch(audio, self._pitch_shift, sr)
        if sr != self._sample_rate:
            audio = _resample(audio, sr, self._sample_rate)
        return audio, self._sample_rate

    # ------------------------------------------------------------------
    # Streaming output
    # ------------------------------------------------------------------

    async def synthesize(self, text: str, emotion: Optional[Dict[str, float]] = None) -> AsyncIterator[AudioChunk]:
        """Synthesise text and yield AudioChunk objects in a stream.

        Args:
            text: The text to speak.
            emotion: Optional emotion dict to shape inflection.
                     Currently modulates speed/pitch slightly based on
                     urgency and warmth values.

        Yields:
            AudioChunk objects of ~chunk_duration_ms each.
        """
        self._halt_requested = False

        # Apply emotion-based modulation on copies of speed/pitch
        speed = self._speed
        pitch = self._pitch_shift
        if emotion:
            urgency = emotion.get("urgency", 0.0)
            warmth = emotion.get("warmth", 0.5)
            joy = emotion.get("joy", 0.5)
            sadness = emotion.get("sadness", 0.0)
            speed += (urgency - 0.3) * 0.3  # Urgent -> faster
            pitch += (warmth - 0.5) * 1.0   # Warmer -> slightly higher
            pitch += (joy - 0.5) * 0.5
            speed -= sadness * 0.15          # Sad -> slower

        old_speed, old_pitch = self._speed, self._pitch_shift
        self._speed = max(0.25, min(4.0, speed))
        self._pitch_shift = max(-12.0, min(12.0, pitch))

        try:
            # Run synthesis in executor
            loop = asyncio.get_event_loop()
            audio, sr = await loop.run_in_executor(None, self._synthesize_text, text)
            audio, sr = self._post_process(audio, sr)

            # Chunk the audio
            chunk_samples = int(self._sample_rate * self._chunk_duration_ms / 1000.0)
            total_samples = len(audio)
            offset = 0

            while offset < total_samples:
                if self._halt_requested:
                    logger.debug("TTS halted mid-stream at sample {}/{}", offset, total_samples)
                    return

                end = min(offset + chunk_samples, total_samples)
                chunk_data = audio[offset:end]
                is_final = end >= total_samples
                duration_ms = len(chunk_data) / self._sample_rate * 1000.0

                yield AudioChunk(
                    data=chunk_data,
                    sample_rate=self._sample_rate,
                    duration_ms=duration_ms,
                    is_final=is_final,
                )
                offset = end

                # Yield control to event loop between chunks
                await asyncio.sleep(0)
        finally:
            self._speed = old_speed
            self._pitch_shift = old_pitch

    async def synthesize_stream(
        self, text_stream: AsyncIterator[TextChunk]
    ) -> AsyncIterator[AudioChunk]:
        """Consume a stream of TextChunks and yield AudioChunks.

        Begins synthesis as soon as a sentence boundary is reached,
        enabling low-latency streaming TTS from LLM tokens.
        """
        buffer = ""
        async for text_chunk in text_stream:
            if self._halt_requested:
                return

            buffer += text_chunk.text

            # Synthesise at sentence boundaries or if buffer is long
            if text_chunk.is_sentence_end or len(buffer) > 200:
                if buffer.strip():
                    async for audio_chunk in self.synthesize(
                        buffer.strip(), emotion=text_chunk.emotion
                    ):
                        if self._halt_requested:
                            return
                        yield audio_chunk
                buffer = ""

        # Flush remaining buffer
        if buffer.strip() and not self._halt_requested:
            async for audio_chunk in self.synthesize(buffer.strip()):
                if self._halt_requested:
                    return
                yield audio_chunk
