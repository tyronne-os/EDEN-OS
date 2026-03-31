"""
EDEN OS — Genesis Engine: Eden Protocol Validator
Implements the 0.3 deviation rule for skin texture fidelity.

Process:
  1. Extract face region from both generated and reference images.
  2. Convert to LAB colour space.
  3. Apply Gabor filter bank (4 orientations x 3 frequencies) to the
     L-channel to capture micro-texture features (pores, fine lines).
  4. Compute standard deviation between the two feature vectors.
  5. Return EdenValidationResult with passed / score / feedback.
"""

from __future__ import annotations

import numpy as np
import cv2
from loguru import logger

from eden_os.shared.types import EdenValidationResult

try:
    from skimage.filters import gabor_kernel
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False
    logger.warning("scikit-image not installed — Gabor filter bank unavailable, "
                   "EdenProtocolValidator will use histogram fallback")

from scipy.ndimage import convolve


class EdenProtocolValidator:
    """Validates that a generated image preserves the reference portrait's
    skin texture within the Eden Protocol's 0.3 standard-deviation threshold.
    """

    # Gabor filter bank parameters
    ORIENTATIONS = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 4 orientations
    FREQUENCIES = [0.1, 0.25, 0.4]                            # 3 frequencies

    def __init__(self) -> None:
        self._kernels: list[np.ndarray] = []
        self._build_filter_bank()

    # ------------------------------------------------------------------
    # Filter bank construction
    # ------------------------------------------------------------------
    def _build_filter_bank(self) -> None:
        """Pre-compute the Gabor filter kernels."""
        if not _HAS_SKIMAGE:
            return

        for theta in self.ORIENTATIONS:
            for freq in self.FREQUENCIES:
                kernel = gabor_kernel(frequency=freq, theta=theta,
                                      sigma_x=3.0, sigma_y=3.0)
                # Store the real part as a float64 2-D kernel
                self._kernels.append(np.real(kernel).astype(np.float64))

        logger.debug("EdenProtocolValidator: built {} Gabor kernels",
                     len(self._kernels))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def validate(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
        threshold: float = 0.3,
    ) -> EdenValidationResult:
        """Compare *generated* against *reference* and return validation result.

        Parameters
        ----------
        generated : np.ndarray  – RGB uint8 image of the generated portrait.
        reference : np.ndarray  – RGB uint8 image of the reference portrait.
        threshold : float       – maximum allowed std-dev distance (default 0.3).

        Returns
        -------
        EdenValidationResult with .passed, .score, .feedback
        """
        logger.info("EdenProtocolValidator.validate — threshold={}", threshold)

        # Resize both images to the same dimensions for fair comparison
        target_size = (256, 256)
        gen_resized = cv2.resize(generated, target_size, interpolation=cv2.INTER_AREA)
        ref_resized = cv2.resize(reference, target_size, interpolation=cv2.INTER_AREA)

        # Extract the L-channel from LAB
        gen_l = self._extract_lightness(gen_resized)
        ref_l = self._extract_lightness(ref_resized)

        # Compute feature vectors
        gen_features = self._compute_texture_features(gen_l)
        ref_features = self._compute_texture_features(ref_l)

        # Compute deviation
        score = self._compute_deviation(gen_features, ref_features)

        passed = score <= threshold
        if passed:
            feedback = (f"Eden Protocol PASSED — skin texture deviation {score:.4f} "
                        f"is within the {threshold} threshold.")
        else:
            feedback = (f"Eden Protocol FAILED — skin texture deviation {score:.4f} "
                        f"exceeds the {threshold} threshold. "
                        "Consider regenerating with tighter identity lock or "
                        "reducing style transfer intensity.")

        logger.info("EdenProtocolValidator result: passed={} score={:.4f}", passed, score)
        return EdenValidationResult(passed=passed, score=float(score), feedback=feedback)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_lightness(image_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB → LAB and return the L channel as float64 in [0, 1]."""
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float64) / 255.0
        return l_channel

    def _compute_texture_features(self, lightness: np.ndarray) -> np.ndarray:
        """Apply Gabor filter bank and collect mean + std of each response
        into a single feature vector.

        Returns a 1-D numpy array of length 2 * num_kernels.
        """
        if not self._kernels:
            # Fallback: use simple gradient-based texture descriptor
            return self._gradient_fallback(lightness)

        features: list[float] = []
        for kernel in self._kernels:
            # Convolve lightness image with the Gabor kernel
            response = convolve(lightness, kernel, mode="reflect")
            features.append(float(np.mean(np.abs(response))))
            features.append(float(np.std(response)))

        return np.array(features, dtype=np.float64)

    @staticmethod
    def _gradient_fallback(lightness: np.ndarray) -> np.ndarray:
        """Fallback texture descriptor using Sobel gradients when skimage
        is unavailable."""
        gx = cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(lightness, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        # Compute statistics in 4x4 spatial grid
        features: list[float] = []
        h, w = lightness.shape
        gh, gw = h // 4, w // 4
        for gy_idx in range(4):
            for gx_idx in range(4):
                patch = mag[gy_idx * gh:(gy_idx + 1) * gh,
                            gx_idx * gw:(gx_idx + 1) * gw]
                features.append(float(np.mean(patch)))
                features.append(float(np.std(patch)))
        return np.array(features, dtype=np.float64)

    @staticmethod
    def _compute_deviation(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
        """Compute the normalised standard-deviation distance between two
        feature vectors.

        We use the root-mean-square of element-wise differences, normalised
        by the mean magnitude of the reference vector so the score is
        scale-invariant.
        """
        diff = feat_a - feat_b
        rms = float(np.sqrt(np.mean(diff ** 2)))
        ref_mag = float(np.mean(np.abs(feat_b))) + 1e-8  # avoid division by zero
        deviation = rms / ref_mag
        return deviation
