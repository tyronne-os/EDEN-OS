"""
EDEN OS -- Audio Capture & Processing (Agent 6)
Decodes incoming base64 PCM audio, applies noise gate, resamples to 16kHz.
"""

from __future__ import annotations

import base64
from typing import Optional

import numpy as np
from loguru import logger

from eden_os.shared.types import AudioChunk


class AudioCapture:
    """Processes raw base64-encoded PCM audio from WebSocket clients.

    Pipeline:
        1. Decode base64 -> raw bytes -> int16 numpy array
        2. Normalise to float32 [-1, 1]
        3. Apply noise gate (threshold-based)
        4. Resample to target_sr if the source sample-rate differs
        5. Return an AudioChunk or None (if gated out)
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        noise_gate_threshold: float = 0.01,
        source_sr: int = 48_000,
        dtype: str = "int16",
    ) -> None:
        """
        Parameters
        ----------
        target_sr:
            Output sample rate (default 16 kHz for ASR models).
        noise_gate_threshold:
            RMS threshold below which audio is considered silence.
            Range [0.0, 1.0] relative to float32 normalised amplitude.
        source_sr:
            Expected sample rate of the incoming PCM data.
            Common values: 48000 (browser default), 44100, 16000.
        dtype:
            Numpy dtype of the incoming raw PCM samples.
        """
        self.target_sr = target_sr
        self.noise_gate_threshold = noise_gate_threshold
        self.source_sr = source_sr
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process(
        self,
        b64_pcm: str,
        source_sr: int | None = None,
        is_final: bool = False,
    ) -> Optional[AudioChunk]:
        """Decode, gate, resample, and return an AudioChunk.

        Returns None if the audio is below the noise gate.
        """
        sr = source_sr or self.source_sr

        # 1. Decode base64 -> numpy int16
        try:
            raw_bytes = base64.b64decode(b64_pcm)
            samples = np.frombuffer(raw_bytes, dtype=self.dtype)
        except Exception as exc:
            logger.warning(f"AudioCapture: failed to decode PCM: {exc}")
            return None

        if samples.size == 0:
            return None

        # 2. Normalise to float32 [-1, 1]
        audio = samples.astype(np.float32)
        if self.dtype == "int16":
            audio /= 32768.0
        elif self.dtype == "int32":
            audio /= 2147483648.0
        # float32 input is already in [-1, 1]

        # 3. Noise gate
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < self.noise_gate_threshold:
            return None

        # 4. Resample to target_sr if needed
        if sr != self.target_sr:
            audio = self._resample(audio, sr, self.target_sr)

        # 5. Build AudioChunk
        duration_ms = (len(audio) / self.target_sr) * 1000.0
        return AudioChunk(
            data=audio,
            sample_rate=self.target_sr,
            duration_ms=duration_ms,
            is_final=is_final,
        )

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(
        audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Simple linear-interpolation resample.

        For production quality, scipy.signal.resample_poly would be
        better, but we keep zero mandatory heavy dependencies here.
        If scipy is available we use it; otherwise fall back to
        numpy linear interpolation.
        """
        if orig_sr == target_sr:
            return audio

        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(orig_sr, target_sr)
            up = target_sr // g
            down = orig_sr // g
            return resample_poly(audio, up, down).astype(np.float32)
        except ImportError:
            # Fallback: numpy linear interpolation
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(
                np.float32
            )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_noise_gate(self, threshold: float) -> None:
        """Update the noise gate threshold at runtime."""
        self.noise_gate_threshold = max(0.0, min(1.0, threshold))
        logger.debug(f"AudioCapture noise gate set to {self.noise_gate_threshold}")

    def set_source_sample_rate(self, sr: int) -> None:
        """Update the expected source sample rate."""
        self.source_sr = sr
        logger.debug(f"AudioCapture source SR set to {sr}")
