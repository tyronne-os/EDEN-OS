"""
EDEN OS — Genesis Engine: Latent Encoder
Encodes a processed portrait into a compact latent representation
compatible with the animation engine (LivePortrait appearance extractor).

Pipeline:
  1. Resize to 256x256.
  2. Normalise pixel values to [-1, 1].
  3. Extract multi-scale feature maps via spatial pyramid pooling.
  4. Average-pool feature maps to produce a 1-D latent vector.
"""

from __future__ import annotations

import numpy as np
import cv2
from loguru import logger


class LatentEncoder:
    """Encode a portrait image into a latent vector for the animation engine."""

    ENCODE_SIZE = 256
    LATENT_DIM = 512  # output vector dimensionality

    def __init__(self, latent_dim: int = 512) -> None:
        self.LATENT_DIM = latent_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def encode(self, portrait: np.ndarray) -> np.ndarray:
        """Encode *portrait* (RGB uint8) into a 1-D latent vector.

        Parameters
        ----------
        portrait : np.ndarray
            RGB image, any size, dtype uint8.

        Returns
        -------
        np.ndarray of shape (LATENT_DIM,) and dtype float32.
        """
        logger.info("LatentEncoder.encode — input shape {}", portrait.shape)

        # Step 1: resize
        resized = cv2.resize(portrait, (self.ENCODE_SIZE, self.ENCODE_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)

        # Step 2: normalise to [-1, 1] float32
        normalised = resized.astype(np.float32) / 127.5 - 1.0  # (256, 256, 3)

        # Step 3: extract multi-scale feature maps via spatial pyramid
        features = self._spatial_pyramid_features(normalised)

        # Step 4: project to final latent dimension
        latent = self._project_to_latent(features)

        logger.info("LatentEncoder.encode — output shape {}", latent.shape)
        return latent

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _spatial_pyramid_features(self, image: np.ndarray) -> np.ndarray:
        """Compute multi-scale pooled features using a spatial pyramid.

        Levels: 1x1, 2x2, 4x4, 8x8 spatial grids.
        At each level, compute (mean, std) per channel in each cell.
        Total features = sum_over_levels(grid_cells * channels * 2).
        """
        h, w, c = image.shape
        features: list[float] = []

        for grid_size in [1, 2, 4, 8]:
            cell_h = h // grid_size
            cell_w = w // grid_size
            for gy in range(grid_size):
                for gx in range(grid_size):
                    cell = image[gy * cell_h:(gy + 1) * cell_h,
                                 gx * cell_w:(gx + 1) * cell_w, :]
                    for ch in range(c):
                        features.append(float(np.mean(cell[:, :, ch])))
                        features.append(float(np.std(cell[:, :, ch])))

        # Also add gradient-based features for edge/texture awareness
        gray = np.mean(image, axis=2)
        gx = np.gradient(gray, axis=1)
        gy = np.gradient(gray, axis=0)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        angle = np.arctan2(gy, gx)

        # Gradient histogram (8 bins of orientation, mean magnitude per bin)
        bin_edges = np.linspace(-np.pi, np.pi, 9)
        for i in range(8):
            mask = (angle >= bin_edges[i]) & (angle < bin_edges[i + 1])
            if np.any(mask):
                features.append(float(np.mean(mag[mask])))
            else:
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def _project_to_latent(self, features: np.ndarray) -> np.ndarray:
        """Deterministically project the feature vector to LATENT_DIM via
        a fixed random projection (seeded for reproducibility).

        This is a lightweight stand-in for a learned encoder; it preserves
        the structure of the feature space while producing a vector of the
        correct dimensionality for downstream consumption.
        """
        feat_dim = features.shape[0]

        if feat_dim == self.LATENT_DIM:
            return features

        # Fixed random projection matrix (seeded so identical images → identical latents)
        rng = np.random.RandomState(42)
        proj = rng.randn(feat_dim, self.LATENT_DIM).astype(np.float32)
        # Normalise columns for unit variance
        proj /= np.sqrt(np.sum(proj ** 2, axis=0, keepdims=True)) + 1e-8

        latent = features @ proj

        # L2-normalise the latent vector
        norm = np.linalg.norm(latent) + 1e-8
        latent = latent / norm

        return latent
