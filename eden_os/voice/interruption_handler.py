"""
EDEN OS -- Voice Engine: Interruption Handler
Detects when user starts speaking while avatar is talking.
Uses energy-based VAD (RMS threshold). Tracks avatar speaking state.
Signals TTS halt on interruption.
"""

import asyncio
from typing import Callable, Optional

import numpy as np
from loguru import logger

from eden_os.shared.types import AudioChunk


class InterruptionHandler:
    """Energy-based voice-activity detector for interruption handling.

    Monitors incoming user audio while the avatar is speaking.
    When sustained energy above threshold is detected, fires the
    interruption callback to halt TTS output.
    """

    def __init__(
        self,
        rms_threshold: float = 0.02,
        sustained_frames: int = 3,
        cooldown_ms: float = 500.0,
    ) -> None:
        """
        Args:
            rms_threshold: RMS energy level that counts as speech.
            sustained_frames: Number of consecutive above-threshold
                chunks required before an interruption is declared.
            cooldown_ms: Minimum time between interruption events.
        """
        self._rms_threshold = rms_threshold
        self._sustained_frames = sustained_frames
        self._cooldown_ms = cooldown_ms

        # State
        self._is_avatar_speaking = False
        self._consecutive_active: int = 0
        self._last_interruption_time: float = 0.0
        self._interrupted = asyncio.Event()

        # Optional callback fired on interruption
        self._on_interrupt: Optional[Callable[[], None]] = None

        logger.info(
            "InterruptionHandler ready  rms_thresh={} sustained={}",
            rms_threshold,
            sustained_frames,
        )

    # ------------------------------------------------------------------
    # Avatar state control
    # ------------------------------------------------------------------

    def set_avatar_speaking(self, speaking: bool) -> None:
        """Called by TTS/Conductor to indicate avatar speech state."""
        self._is_avatar_speaking = speaking
        if not speaking:
            self._consecutive_active = 0
            self._interrupted.clear()
        logger.debug("Avatar speaking state -> {}", speaking)

    @property
    def is_avatar_speaking(self) -> bool:
        return self._is_avatar_speaking

    # ------------------------------------------------------------------
    # Interruption callback
    # ------------------------------------------------------------------

    def on_interrupt(self, callback: Callable[[], None]) -> None:
        """Register a callback to fire when an interruption is detected."""
        self._on_interrupt = callback

    @property
    def was_interrupted(self) -> bool:
        return self._interrupted.is_set()

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rms(audio: np.ndarray) -> float:
        """Compute root-mean-square energy of an audio buffer."""
        if audio.size == 0:
            return 0.0
        # Ensure float
        samples = audio.astype(np.float32) if audio.dtype != np.float32 else audio
        # Normalise int16 range if needed
        if np.max(np.abs(samples)) > 2.0:
            samples = samples / 32768.0
        return float(np.sqrt(np.mean(samples ** 2)))

    async def detect(self, audio: AudioChunk) -> bool:
        """Check a single audio chunk for interruption.

        Returns True if an interruption is detected (user is speaking
        while avatar is speaking), False otherwise.
        """
        if not self._is_avatar_speaking:
            self._consecutive_active = 0
            return False

        rms = self._compute_rms(audio.data)

        if rms >= self._rms_threshold:
            self._consecutive_active += 1
        else:
            # Reset if silence gap
            self._consecutive_active = max(0, self._consecutive_active - 1)

        if self._consecutive_active >= self._sustained_frames:
            now = asyncio.get_event_loop().time() * 1000.0
            if (now - self._last_interruption_time) < self._cooldown_ms:
                return False  # Within cooldown

            self._last_interruption_time = now
            self._interrupted.set()
            self._consecutive_active = 0

            logger.warning("INTERRUPTION detected  rms={:.4f}", rms)

            if self._on_interrupt is not None:
                self._on_interrupt()

            return True

        return False

    def reset(self) -> None:
        """Reset all state (e.g. at start of new turn)."""
        self._consecutive_active = 0
        self._interrupted.clear()
        self._is_avatar_speaking = False
        logger.debug("InterruptionHandler reset")
