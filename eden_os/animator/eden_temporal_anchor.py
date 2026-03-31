"""
EDEN OS — Eden Temporal Anchor
Implements temporal consistency system adapted from LONGLIVE's frame sink concept.
Prevents identity drift over long conversations by maintaining anchor frames.
"""

import time
from typing import Optional

import cv2
import numpy as np
from loguru import logger


class EdenTemporalAnchor:
    """
    Maintains identity consistency over long conversations.

    Adapted from LONGLIVE (arXiv:2509.22622):
    - Keeps the first frame of each conversation as a global anchor
    - Periodically refreshes anchor to prevent staleness
    - Blends current frames with anchor to maintain identity
    """

    def __init__(self, refresh_interval: int = 5, min_anchor_weight: float = 0.1):
        """
        Args:
            refresh_interval: Refresh anchor every N conversation turns
            min_anchor_weight: Minimum blend weight for anchor (never fully ignored)
        """
        self.refresh_interval = refresh_interval
        self.min_anchor_weight = min_anchor_weight

        self._global_anchor: Optional[np.ndarray] = None
        self._anchor_features: Optional[np.ndarray] = None
        self._turn_count: int = 0
        self._decay_rate: float = 0.02  # Weight decay per turn
        self._creation_time: float = 0.0

    def set_anchor(self, frame: np.ndarray) -> None:
        """Set the global identity anchor frame."""
        self._global_anchor = frame.copy()
        self._anchor_features = self._extract_identity_features(frame)
        self._creation_time = time.monotonic()
        self._turn_count = 0
        logger.info("Global identity anchor set")

    def _extract_identity_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract identity-critical features from a frame.
        Focuses on skin texture, facial proportions, and color profile.
        """
        # Resize to standard analysis size
        analysis = cv2.resize(frame, (128, 128))

        # Convert to LAB for perceptually-uniform color analysis
        lab = cv2.cvtColor(analysis, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Extract features:
        # 1. Color histogram (identity-linked skin tone)
        l_hist = np.histogram(lab[:, :, 0], bins=32, range=(0, 255))[0].astype(np.float32)
        a_hist = np.histogram(lab[:, :, 1], bins=32, range=(0, 255))[0].astype(np.float32)
        b_hist = np.histogram(lab[:, :, 2], bins=32, range=(0, 255))[0].astype(np.float32)

        # Normalize histograms
        l_hist /= l_hist.sum() + 1e-8
        a_hist /= a_hist.sum() + 1e-8
        b_hist /= b_hist.sum() + 1e-8

        # 2. Spatial structure (downsampled face layout)
        gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY)
        structure = cv2.resize(gray, (16, 16)).flatten().astype(np.float32) / 255.0

        # Combine into feature vector
        features = np.concatenate([l_hist, a_hist, b_hist, structure])
        return features

    def compute_drift(self, current_frame: np.ndarray) -> float:
        """
        Compute identity drift between current frame and anchor.
        Returns drift score (0.0 = identical, higher = more drift).
        """
        if self._anchor_features is None:
            return 0.0

        current_features = self._extract_identity_features(current_frame)
        drift = np.sqrt(np.mean((current_features - self._anchor_features) ** 2))
        return float(drift)

    def stabilize_frame(
        self, current_frame: np.ndarray, conversation_turn: int
    ) -> np.ndarray:
        """
        Apply temporal anchoring to stabilize a frame's identity.

        Blends the current frame with the anchor based on decay weight.
        This prevents gradual identity drift over long conversations.
        """
        if self._global_anchor is None:
            return current_frame

        self._turn_count = conversation_turn

        # Refresh anchor periodically
        if conversation_turn > 0 and conversation_turn % self.refresh_interval == 0:
            self._refresh_anchor(current_frame)

        # Compute anchor weight (decays over time but never reaches zero)
        anchor_weight = max(
            self.min_anchor_weight,
            1.0 - (conversation_turn * self._decay_rate),
        )

        # Only blend if drift is detected
        drift = self.compute_drift(current_frame)
        if drift < 0.05:
            return current_frame  # No significant drift

        # Blend in LAB space for perceptual smoothness
        anchor_resized = cv2.resize(self._global_anchor, (current_frame.shape[1], current_frame.shape[0]))

        current_lab = cv2.cvtColor(current_frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        anchor_lab = cv2.cvtColor(anchor_resized, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Blend color channels (preserve structure from current, tone from anchor)
        blended_lab = current_lab.copy()
        # Only blend color channels (a, b), keep lightness from current frame
        blend_strength = anchor_weight * 0.3  # Subtle color correction
        blended_lab[:, :, 1] = (
            current_lab[:, :, 1] * (1 - blend_strength)
            + anchor_lab[:, :, 1] * blend_strength
        )
        blended_lab[:, :, 2] = (
            current_lab[:, :, 2] * (1 - blend_strength)
            + anchor_lab[:, :, 2] * blend_strength
        )

        blended = cv2.cvtColor(
            np.clip(blended_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
        )

        return blended

    def _refresh_anchor(self, current_frame: np.ndarray) -> None:
        """Refresh anchor with current frame (partial update)."""
        if self._global_anchor is None:
            self.set_anchor(current_frame)
            return

        # Blend old anchor with current frame (70% old, 30% new)
        anchor_resized = cv2.resize(
            self._global_anchor, (current_frame.shape[1], current_frame.shape[0])
        )
        self._global_anchor = cv2.addWeighted(anchor_resized, 0.7, current_frame, 0.3, 0)
        self._anchor_features = self._extract_identity_features(self._global_anchor)
        logger.debug(f"Anchor refreshed at turn {self._turn_count}")

    def get_status(self) -> dict:
        """Get anchor status for monitoring."""
        return {
            "has_anchor": self._global_anchor is not None,
            "turn_count": self._turn_count,
            "uptime_seconds": time.monotonic() - self._creation_time if self._global_anchor is not None else 0,
            "current_weight": max(
                self.min_anchor_weight,
                1.0 - (self._turn_count * self._decay_rate),
            ),
        }
