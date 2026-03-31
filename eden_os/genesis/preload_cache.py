"""
EDEN OS — Genesis Engine: Preload Cache
Pre-computes idle animation seed data so the avatar is alive on load
with ZERO wait time.

Given a processed portrait, generates:
  - N seed frames with slight random perturbations:
      * micro blink positions (eyelid offset values)
      * slight head rotations (affine transforms)
  - Breathing cycle keyframes (6 frames of subtle vertical shift).

Returns a cache dict with 'seed_frames' and 'breathing_cycle' lists.
"""

from __future__ import annotations

import numpy as np
import cv2
from loguru import logger


class PreloadCache:
    """Pre-compute idle animation seed data for instant-ready avatars."""

    DEFAULT_NUM_SEEDS = 8
    BREATHING_FRAMES = 6

    def __init__(self, num_seeds: int = 8) -> None:
        self.num_seeds = num_seeds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def compute(self, profile: dict) -> dict:
        """Pre-compute idle seed frames and breathing keyframes.

        Parameters
        ----------
        profile : dict
            Must contain 'aligned_face' (np.ndarray, 512x512x3 RGB uint8)
            as produced by PortraitEngine.process().
            Optional keys: 'landmarks' (dict), 'bbox' (tuple).

        Returns
        -------
        dict with keys:
            seed_frames     – list[dict] each containing 'frame' (np.ndarray),
                              'blink_offset' (float), 'head_rotation_deg' (float)
            breathing_cycle – list[dict] each containing 'frame' (np.ndarray),
                              'vertical_shift_px' (float), 'phase' (float 0..1)
        """
        face = profile.get("aligned_face")
        if face is None:
            raise ValueError("profile must contain 'aligned_face' key")

        logger.info("PreloadCache.compute — generating {} seed frames + {} breathing frames",
                    self.num_seeds, self.BREATHING_FRAMES)

        rng = np.random.RandomState(seed=7)

        seed_frames = self._generate_seed_frames(face, rng)
        breathing_cycle = self._generate_breathing_cycle(face)

        cache = {
            "seed_frames": seed_frames,
            "breathing_cycle": breathing_cycle,
        }
        logger.info("PreloadCache.compute — done ({} seed, {} breathing)",
                    len(seed_frames), len(breathing_cycle))
        return cache

    # ------------------------------------------------------------------
    # Seed frames (blink + micro head rotation)
    # ------------------------------------------------------------------
    def _generate_seed_frames(self, face: np.ndarray,
                              rng: np.random.RandomState) -> list[dict]:
        """Generate N seed frames with subtle random perturbations."""
        h, w = face.shape[:2]
        center = (w / 2, h / 2)
        seed_frames: list[dict] = []

        for i in range(self.num_seeds):
            # Random micro head rotation: +/- 2 degrees
            rotation_deg = float(rng.uniform(-2.0, 2.0))
            # Random blink offset: 0 = eyes open, 1 = fully closed
            # Most frames are open; occasionally partially closed
            blink_offset = float(rng.choice(
                [0.0, 0.0, 0.0, 0.0, 0.05, 0.1, 0.3, 0.8],
            ))

            # Apply rotation via affine transform
            rot_mat = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
            frame = cv2.warpAffine(face, rot_mat, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)

            # Simulate blink by darkening the upper-face region proportionally
            if blink_offset > 0.01:
                frame = self._apply_blink_overlay(frame, blink_offset)

            seed_frames.append({
                "frame": frame,
                "blink_offset": blink_offset,
                "head_rotation_deg": rotation_deg,
                "index": i,
            })

        return seed_frames

    # ------------------------------------------------------------------
    # Breathing cycle
    # ------------------------------------------------------------------
    def _generate_breathing_cycle(self, face: np.ndarray) -> list[dict]:
        """Generate 6 keyframes for a subtle breathing motion.

        Breathing is modelled as a vertical sinusoidal shift of 1-3 pixels —
        just enough to feel organic without being distracting.
        """
        h, w = face.shape[:2]
        max_shift_px = 2.5  # peak of inhale
        frames: list[dict] = []

        for i in range(self.BREATHING_FRAMES):
            phase = i / self.BREATHING_FRAMES  # 0.0 → ~1.0
            # Sinusoidal vertical shift (inhale = up, exhale = down)
            shift_px = max_shift_px * np.sin(2 * np.pi * phase)

            # Apply vertical translation
            trans_mat = np.float32([[1, 0, 0], [0, 1, -shift_px]])
            frame = cv2.warpAffine(face, trans_mat, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)

            frames.append({
                "frame": frame,
                "vertical_shift_px": float(shift_px),
                "phase": float(phase),
                "index": i,
            })

        return frames

    # ------------------------------------------------------------------
    # Blink overlay
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_blink_overlay(frame: np.ndarray, blink_offset: float) -> np.ndarray:
        """Simulate eyelid closure by blending the eye region towards
        a skin-tone average.  blink_offset in [0, 1]."""
        result = frame.copy()
        h, w = frame.shape[:2]

        # Eye region: roughly the upper-middle band of the portrait
        eye_y_start = int(h * 0.28)
        eye_y_end = int(h * 0.42)
        eye_x_start = int(w * 0.2)
        eye_x_end = int(w * 0.8)

        eye_region = result[eye_y_start:eye_y_end, eye_x_start:eye_x_end]
        if eye_region.size == 0:
            return result

        # Compute the mean skin colour around the eyes as the "closed lid" colour
        skin_color = np.mean(eye_region, axis=(0, 1)).astype(np.uint8)
        closed_lid = np.full_like(eye_region, skin_color)

        # Blend: higher blink_offset → more closed
        alpha = np.clip(blink_offset, 0.0, 1.0)
        blended = cv2.addWeighted(eye_region, 1.0 - alpha, closed_lid, alpha, 0)
        result[eye_y_start:eye_y_end, eye_x_start:eye_x_end] = blended

        return result
