"""
EDEN OS — Idle Animation Generator
Generates continuous LISTENING state idle loop with natural blinks,
micro head movements, breathing, and occasional eyebrow raises.
The avatar is NEVER frozen.
"""

import asyncio
import time
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from eden_os.shared.types import AvatarState, VideoFrame


class IdleGenerator:
    """Generates natural idle animations for the avatar."""

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self._running = False
        self._start_time: float = 0.0
        self._frame_count: int = 0

        # Blink parameters
        self._next_blink_time: float = 0.0
        self._blink_duration: float = 0.15  # seconds
        self._blink_progress: float = 0.0
        self._is_blinking: bool = False

        # Breathing parameters
        self._breath_cycle: float = 4.0  # seconds per breath
        self._breath_amplitude: float = 0.005

        # Head micro-movement
        self._head_sway_freq: float = 0.3  # Hz
        self._head_sway_amp: float = 0.01  # radians equivalent

        # Eyebrow micro-raise
        self._next_brow_time: float = 0.0
        self._brow_raise_duration: float = 0.5
        self._brow_progress: float = 0.0
        self._is_brow_raising: bool = False

    def _schedule_next_blink(self, current_time: float) -> None:
        """Schedule next blink at random interval (3-7 seconds)."""
        interval = np.random.uniform(3.0, 7.0)
        self._next_blink_time = current_time + interval

    def _schedule_next_brow_raise(self, current_time: float) -> None:
        """Schedule next eyebrow micro-raise (8-15 seconds)."""
        interval = np.random.uniform(8.0, 15.0)
        self._next_brow_time = current_time + interval

    def get_idle_keypoint_deltas(self, elapsed: float) -> dict:
        """
        Compute keypoint deltas for idle animation at given elapsed time.

        Returns dict with deltas for each facial feature.
        """
        deltas = {
            "eye_blink": 0.0,      # 0=open, 1=closed
            "brow_raise": 0.0,     # 0=neutral, 1=raised
            "head_x": 0.0,         # horizontal rotation delta
            "head_y": 0.0,         # vertical rotation delta
            "head_z": 0.0,         # tilt delta
            "breath_y": 0.0,       # vertical breathing motion
            "mouth_tension": 0.0,  # slight mouth movement
        }

        # --- Breathing ---
        breath_phase = (elapsed % self._breath_cycle) / self._breath_cycle
        breath_value = np.sin(breath_phase * 2 * np.pi) * self._breath_amplitude
        deltas["breath_y"] = float(breath_value)

        # --- Head micro-sway ---
        # Lissajous-like pattern for natural head movement
        deltas["head_x"] = float(
            np.sin(elapsed * self._head_sway_freq * 2 * np.pi) * self._head_sway_amp
        )
        deltas["head_y"] = float(
            np.sin(elapsed * self._head_sway_freq * 1.3 * 2 * np.pi + 0.7)
            * self._head_sway_amp * 0.6
        )
        deltas["head_z"] = float(
            np.sin(elapsed * self._head_sway_freq * 0.7 * 2 * np.pi + 1.4)
            * self._head_sway_amp * 0.3
        )

        # --- Blinking ---
        if not self._is_blinking and elapsed >= self._next_blink_time:
            self._is_blinking = True
            self._blink_progress = 0.0

        if self._is_blinking:
            self._blink_progress += self.frame_interval / self._blink_duration
            if self._blink_progress >= 1.0:
                self._is_blinking = False
                self._blink_progress = 0.0
                self._schedule_next_blink(elapsed)
                deltas["eye_blink"] = 0.0
            else:
                # Smooth blink curve: fast close, slower open
                if self._blink_progress < 0.3:
                    deltas["eye_blink"] = float(self._blink_progress / 0.3)
                else:
                    deltas["eye_blink"] = float(1.0 - (self._blink_progress - 0.3) / 0.7)

        # --- Eyebrow micro-raise ---
        if not self._is_brow_raising and elapsed >= self._next_brow_time:
            self._is_brow_raising = True
            self._brow_progress = 0.0

        if self._is_brow_raising:
            self._brow_progress += self.frame_interval / self._brow_raise_duration
            if self._brow_progress >= 1.0:
                self._is_brow_raising = False
                self._brow_progress = 0.0
                self._schedule_next_brow_raise(elapsed)
                deltas["brow_raise"] = 0.0
            else:
                # Smooth up and down
                deltas["brow_raise"] = float(
                    np.sin(self._brow_progress * np.pi) * 0.3
                )

        # --- Subtle mouth tension (micro-expression) ---
        deltas["mouth_tension"] = float(
            np.sin(elapsed * 0.15 * 2 * np.pi) * 0.002
        )

        return deltas

    def apply_idle_to_keypoints(
        self, base_keypoints: np.ndarray, deltas: dict
    ) -> np.ndarray:
        """Apply idle animation deltas to base keypoints."""
        keypoints = base_keypoints.copy()

        # Eye blink: move top and bottom eyelids
        blink = deltas.get("eye_blink", 0.0)
        keypoints[6][1] += blink * 0.02    # left eye top down
        keypoints[7][1] -= blink * 0.015   # left eye bottom up
        keypoints[9][1] += blink * 0.02    # right eye top down
        keypoints[10][1] -= blink * 0.015  # right eye bottom up

        # Brow raise
        brow = deltas.get("brow_raise", 0.0)
        keypoints[19][1] -= brow * 0.02  # left brow up
        keypoints[20][1] -= brow * 0.02  # right brow up

        # Head movement (apply to all keypoints)
        head_x = deltas.get("head_x", 0.0)
        head_y = deltas.get("head_y", 0.0)
        keypoints[:, 0] += head_x
        keypoints[:, 1] += head_y

        # Breathing (vertical shift on lower face)
        breath = deltas.get("breath_y", 0.0)
        keypoints[11:, 1] += breath

        # Mouth tension
        tension = deltas.get("mouth_tension", 0.0)
        keypoints[14][0] -= tension
        keypoints[15][0] += tension

        return keypoints

    async def generate_idle_frames(
        self,
        source_image: np.ndarray,
        base_keypoints: np.ndarray,
        render_fn,
    ) -> AsyncIterator[VideoFrame]:
        """
        Continuously generate idle animation frames.

        Args:
            source_image: The avatar portrait
            base_keypoints: Neutral face keypoints
            render_fn: Function that takes keypoints and returns rendered frame
        """
        self._running = True
        self._start_time = time.monotonic()
        self._frame_count = 0
        self._schedule_next_blink(0.0)
        self._schedule_next_brow_raise(0.0)

        logger.info("Idle animation loop started")

        while self._running:
            frame_start = time.monotonic()
            elapsed = frame_start - self._start_time

            # Get idle deltas
            deltas = self.get_idle_keypoint_deltas(elapsed)

            # Apply to keypoints
            animated_keypoints = self.apply_idle_to_keypoints(base_keypoints, deltas)

            # Render frame
            pixels = render_fn(animated_keypoints)

            frame = VideoFrame(
                pixels=pixels,
                timestamp_ms=elapsed * 1000.0,
                state=AvatarState.LISTENING,
                eden_score=1.0,
            )

            self._frame_count += 1
            yield frame

            # Maintain target FPS
            frame_time = time.monotonic() - frame_start
            sleep_time = self.frame_interval - frame_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        """Stop the idle animation loop."""
        self._running = False
        logger.info(f"Idle loop stopped after {self._frame_count} frames")
