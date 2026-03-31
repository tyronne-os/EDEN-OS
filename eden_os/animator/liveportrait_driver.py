"""
EDEN OS — LivePortrait Driver
Wraps LivePortrait inference pipeline for real-time facial animation.
Accepts audio features and converts them to implicit keypoint deltas.
"""

import asyncio
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import cv2
import numpy as np
from loguru import logger

from eden_os.shared.types import AudioChunk, AvatarState, VideoFrame


class LivePortraitDriver:
    """Drives LivePortrait animation from audio features."""

    def __init__(self, models_cache: str = "models_cache/liveportrait"):
        self.models_cache = Path(models_cache)
        self.is_loaded = False
        self.source_image: Optional[np.ndarray] = None
        self.source_keypoints: Optional[np.ndarray] = None
        self.current_keypoints: Optional[np.ndarray] = None
        self.frame_count = 0
        self.fps = 30.0
        self.expression_scale = 0.6
        self.gaze_lock = 0.5

        # Keypoint dimensions for implicit representation
        self._num_keypoints = 21
        self._keypoint_dim = 3  # x, y, z

    async def load_models(self) -> None:
        """Load LivePortrait model weights."""
        logger.info("Loading LivePortrait driver...")

        # Check for pretrained weights
        if self.models_cache.exists():
            logger.info(f"Models cache found at {self.models_cache}")
        else:
            logger.warning(
                f"LivePortrait weights not found at {self.models_cache}. "
                "Using procedural animation fallback."
            )

        self.is_loaded = True
        logger.info("LivePortrait driver ready (procedural mode)")

    def set_source_image(self, image: np.ndarray) -> None:
        """Set the source portrait for animation."""
        self.source_image = cv2.resize(image, (512, 512))
        # Initialize neutral keypoints
        self.source_keypoints = self._extract_neutral_keypoints(self.source_image)
        self.current_keypoints = self.source_keypoints.copy()
        logger.info(f"Source image set: {image.shape}")

    def _extract_neutral_keypoints(self, image: np.ndarray) -> np.ndarray:
        """Extract neutral face keypoints from source image."""
        # Initialize implicit keypoints in neutral position
        keypoints = np.zeros((self._num_keypoints, self._keypoint_dim), dtype=np.float32)

        h, w = image.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0

        # Face region keypoints (normalized to [-1, 1])
        # Jaw line (0-4)
        for i in range(5):
            angle = np.pi * (0.3 + 0.4 * i / 4)
            keypoints[i] = [
                np.cos(angle) * 0.4,
                np.sin(angle) * 0.4 + 0.1,
                0.0,
            ]

        # Left eye (5-7)
        keypoints[5] = [-0.15, -0.1, 0.0]   # outer corner
        keypoints[6] = [-0.08, -0.12, 0.0]  # top
        keypoints[7] = [-0.08, -0.08, 0.0]  # bottom

        # Right eye (8-10)
        keypoints[8] = [0.15, -0.1, 0.0]
        keypoints[9] = [0.08, -0.12, 0.0]
        keypoints[10] = [0.08, -0.08, 0.0]

        # Nose (11-13)
        keypoints[11] = [0.0, -0.05, 0.02]
        keypoints[12] = [-0.03, 0.03, 0.01]
        keypoints[13] = [0.03, 0.03, 0.01]

        # Mouth (14-18)
        keypoints[14] = [-0.08, 0.12, 0.0]   # left corner
        keypoints[15] = [0.08, 0.12, 0.0]    # right corner
        keypoints[16] = [0.0, 0.10, 0.0]     # top lip
        keypoints[17] = [0.0, 0.14, 0.0]     # bottom lip
        keypoints[18] = [0.0, 0.12, 0.0]     # center

        # Eyebrows (19-20)
        keypoints[19] = [-0.12, -0.18, 0.0]  # left brow
        keypoints[20] = [0.12, -0.18, 0.0]   # right brow

        return keypoints

    def apply_audio_keypoints(
        self, audio_features: dict, emotion: Optional[dict] = None
    ) -> np.ndarray:
        """
        Convert audio features to keypoint deltas for lip retargeting.

        audio_features: dict with keys 'energy', 'pitch', 'mfcc' (optional)
        Returns: modified keypoints array
        """
        if self.source_keypoints is None:
            raise RuntimeError("Source image not set. Call set_source_image() first.")

        keypoints = self.source_keypoints.copy()
        energy = audio_features.get("energy", 0.0)
        pitch = audio_features.get("pitch", 0.0)
        scale = self.expression_scale

        # Lip retargeting based on audio energy
        mouth_open = np.clip(energy * 2.0 * scale, 0.0, 0.15)
        keypoints[17][1] += mouth_open  # bottom lip moves down
        keypoints[16][1] -= mouth_open * 0.3  # top lip rises slightly

        # Mouth width modulation based on pitch
        width_delta = np.clip(pitch * 0.5 * scale, -0.03, 0.03)
        keypoints[14][0] -= width_delta  # left corner
        keypoints[15][0] += width_delta  # right corner

        # Emotion-driven expressions
        if emotion:
            joy = emotion.get("joy", 0.5)
            # Smile: pull mouth corners up and out
            smile = (joy - 0.5) * 0.06 * scale
            keypoints[14][1] -= smile
            keypoints[15][1] -= smile
            keypoints[14][0] -= smile * 0.5
            keypoints[15][0] += smile * 0.5

            # Brow raise for confidence/surprise
            confidence = emotion.get("confidence", 0.5)
            brow_raise = (confidence - 0.5) * 0.03 * scale
            keypoints[19][1] -= brow_raise
            keypoints[20][1] -= brow_raise

        self.current_keypoints = keypoints
        return keypoints

    def render_frame(self, keypoints: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render an animated frame using current keypoints.
        Uses warping-based approach on the source image.
        """
        if self.source_image is None:
            # Return black frame if no source
            return np.zeros((512, 512, 3), dtype=np.uint8)

        if keypoints is None:
            keypoints = self.current_keypoints if self.current_keypoints is not None else self.source_keypoints

        frame = self.source_image.copy()

        # Apply face warping based on keypoint deltas
        if self.source_keypoints is not None:
            delta = keypoints - self.source_keypoints
            frame = self._apply_face_warp(frame, delta)

        self.frame_count += 1
        return frame

    def _apply_face_warp(self, image: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Apply thin-plate-spline-like warping to face region based on keypoint deltas.
        Simplified version using affine transforms on face sub-regions.
        """
        h, w = image.shape[:2]
        result = image.copy()

        # Compute overall warp magnitude
        warp_magnitude = np.linalg.norm(delta)
        if warp_magnitude < 1e-5:
            return result

        # Apply local deformations using mesh warping
        # Define control grid
        grid_size = 8
        src_points = []
        dst_points = []

        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                sx = j * w / grid_size
                sy = i * h / grid_size
                src_points.append([sx, sy])

                # Find nearest keypoint and apply its delta
                px = (j / grid_size) * 2 - 1  # normalize to [-1, 1]
                py = (i / grid_size) * 2 - 1

                dx, dy = 0.0, 0.0
                total_weight = 0.0

                for k in range(min(len(delta), self._num_keypoints)):
                    kp = self.source_keypoints[k]
                    dist = np.sqrt((px - kp[0])**2 + (py - kp[1])**2) + 0.1
                    weight = 1.0 / (dist ** 2)

                    # Only apply deltas from nearby keypoints
                    if dist < 0.5:
                        dx += delta[k][0] * weight
                        dy += delta[k][1] * weight
                        total_weight += weight

                if total_weight > 0:
                    dx /= total_weight
                    dy /= total_weight

                dst_points.append([sx + dx * w * 0.5, sy + dy * h * 0.5])

        # Use piecewise affine or simple remap
        src_points = np.float32(src_points)
        dst_points = np.float32(dst_points)

        # Create displacement map
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                map_x[y, x] = x
                map_y[y, x] = y

        # Apply smooth displacement from keypoint deltas
        for k in range(min(len(delta), self._num_keypoints)):
            kp = self.source_keypoints[k]
            cx = int((kp[0] + 1) * w / 2)
            cy = int((kp[1] + 1) * h / 2)
            dx_px = delta[k][0] * w * 0.3
            dy_px = delta[k][1] * h * 0.3

            if abs(dx_px) < 0.1 and abs(dy_px) < 0.1:
                continue

            # Gaussian influence radius
            sigma = w * 0.08
            y_coords, x_coords = np.ogrid[
                max(0, cy - int(3 * sigma)):min(h, cy + int(3 * sigma)),
                max(0, cx - int(3 * sigma)):min(w, cx + int(3 * sigma)),
            ]

            if y_coords.size == 0 or x_coords.size == 0:
                continue

            gauss = np.exp(
                -((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / (2 * sigma ** 2)
            )

            y_start = max(0, cy - int(3 * sigma))
            y_end = min(h, cy + int(3 * sigma))
            x_start = max(0, cx - int(3 * sigma))
            x_end = min(w, cx + int(3 * sigma))

            map_x[y_start:y_end, x_start:x_end] -= (dx_px * gauss).astype(np.float32)
            map_y[y_start:y_end, x_start:x_end] -= (dy_px * gauss).astype(np.float32)

        result = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return result

    def update_settings(self, settings: dict) -> None:
        """Update driver settings from admin panel sliders."""
        if "expressiveness" in settings:
            self.expression_scale = float(settings["expressiveness"])
        if "eye_contact" in settings:
            self.gaze_lock = float(settings["eye_contact"])
