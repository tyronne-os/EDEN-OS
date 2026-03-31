"""
EDEN OS -- Video Encoder (Agent 6)
Encodes numpy RGB frames to base64 JPEG/PNG for WebSocket transport.
"""

from __future__ import annotations

import base64
import io
from typing import Literal

import numpy as np
from loguru import logger

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


class VideoEncoder:
    """Encodes raw numpy RGB frames into base64 strings for WebSocket delivery.

    Supports JPEG (default, smaller payload) and PNG (lossless fallback).
    Uses OpenCV if available for speed; falls back to Pillow.
    """

    def __init__(
        self,
        fmt: Literal["jpeg", "png"] = "jpeg",
        quality: int = 80,
    ) -> None:
        """
        Parameters
        ----------
        fmt:
            Output image format. "jpeg" for lossy (smaller), "png" for lossless.
        quality:
            JPEG quality 1-100 (ignored for PNG). Higher = better quality, larger payload.
        """
        self.fmt = fmt.lower()
        self.quality = max(1, min(100, quality))

        if not _CV2_AVAILABLE and not _PIL_AVAILABLE:
            logger.warning(
                "VideoEncoder: neither cv2 nor Pillow available — "
                "encode_frame will return empty strings."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_frame(self, pixels: np.ndarray) -> str:
        """Encode a single RGB frame (H, W, 3) uint8 to a base64 string.

        Returns an empty string on failure.
        """
        if pixels is None or pixels.size == 0:
            return ""

        # Ensure uint8
        if pixels.dtype != np.uint8:
            pixels = np.clip(pixels, 0, 255).astype(np.uint8)

        if _CV2_AVAILABLE:
            return self._encode_cv2(pixels)
        if _PIL_AVAILABLE:
            return self._encode_pil(pixels)

        logger.error("VideoEncoder: no image library available.")
        return ""

    def encode_batch(self, frames: list[np.ndarray]) -> list[str]:
        """Encode a list of RGB frames, returning a list of base64 strings."""
        return [self.encode_frame(f) for f in frames]

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_quality(self, quality: int) -> None:
        """Update JPEG quality at runtime (1-100)."""
        self.quality = max(1, min(100, quality))
        logger.debug(f"VideoEncoder quality set to {self.quality}")

    def set_format(self, fmt: Literal["jpeg", "png"]) -> None:
        """Switch between jpeg and png encoding."""
        self.fmt = fmt.lower()
        logger.debug(f"VideoEncoder format set to {self.fmt}")

    # ------------------------------------------------------------------
    # Internal encoders
    # ------------------------------------------------------------------

    def _encode_cv2(self, pixels: np.ndarray) -> str:
        """Encode using OpenCV (fastest path)."""
        try:
            # OpenCV expects BGR
            bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            if self.fmt == "jpeg":
                params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                success, buf = cv2.imencode(".jpg", bgr, params)
            else:
                params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 0-9, 3 is reasonable
                success, buf = cv2.imencode(".png", bgr, params)

            if not success:
                logger.error("VideoEncoder: cv2.imencode failed.")
                return ""

            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as exc:
            logger.error(f"VideoEncoder cv2 error: {exc}")
            return ""

    def _encode_pil(self, pixels: np.ndarray) -> str:
        """Encode using Pillow (fallback)."""
        try:
            img = Image.fromarray(pixels, mode="RGB")
            buf = io.BytesIO()

            if self.fmt == "jpeg":
                img.save(buf, format="JPEG", quality=self.quality)
            else:
                img.save(buf, format="PNG")

            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.error(f"VideoEncoder PIL error: {exc}")
            return ""
