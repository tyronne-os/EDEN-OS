"""
EDEN OS — Genesis Engine: Portrait Processing Pipeline
Accepts uploaded image (numpy RGB), detects face via MediaPipe,
extracts landmarks, crops/aligns to 512x512, normalizes lighting.
"""

import numpy as np
import cv2
from loguru import logger

try:
    import mediapipe as mp
except ImportError:
    mp = None
    logger.warning("mediapipe not installed — PortraitEngine will use fallback face detection")


class PortraitEngine:
    """Face detection, alignment, and enhancement pipeline."""

    PORTRAIT_SIZE = 512

    def __init__(self) -> None:
        self._detector = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Lazy init — avoids loading MediaPipe until first use
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        if mp is not None and hasattr(mp, "solutions"):
            try:
                self._mp_face_detection = mp.solutions.face_detection
                self._detector = self._mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.5,
                )
            except (AttributeError, Exception) as e:
                logger.warning(f"MediaPipe solutions unavailable ({e}) — using Haar cascade")
                self._detector = None
        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def process(self, image: np.ndarray) -> dict:
        """Process an uploaded RGB image into a standardised portrait dict.

        Parameters
        ----------
        image : np.ndarray
            RGB image with shape (H, W, 3) and dtype uint8.

        Returns
        -------
        dict with keys:
            aligned_face  – np.ndarray (512, 512, 3) uint8 RGB
            landmarks     – dict of landmark name → (x, y) normalised coords
            bbox          – (x, y, w, h) in pixel coords of the original image
            original_image – the input image (unchanged)
        """
        self._ensure_initialized()
        logger.info("PortraitEngine.process — starting face detection")

        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input must be an RGB image with shape (H, W, 3)")

        h, w, _ = image.shape

        # ----- face detection -----
        bbox, landmarks = self._detect_face(image)

        if bbox is None:
            logger.warning("No face detected — using centre crop fallback")
            bbox, landmarks = self._centre_crop_fallback(h, w)

        # ----- crop face region with margin -----
        cropped = self._crop_with_margin(image, bbox, margin_factor=0.4)

        # ----- align face (rotate to upright using eye landmarks) -----
        aligned = self._align_face(cropped, landmarks, bbox)

        # ----- resize to standard portrait size -----
        aligned = cv2.resize(aligned, (self.PORTRAIT_SIZE, self.PORTRAIT_SIZE),
                             interpolation=cv2.INTER_LANCZOS4)

        # ----- normalise lighting (histogram eq on LAB L-channel) -----
        aligned = self._normalise_lighting(aligned)

        logger.info("PortraitEngine.process — done  bbox={}", bbox)
        return {
            "aligned_face": aligned,
            "landmarks": landmarks,
            "bbox": bbox,
            "original_image": image,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _detect_face(self, image: np.ndarray):
        """Return (bbox, landmarks) using MediaPipe or fallback."""
        h, w, _ = image.shape

        if self._detector is None:
            # Fallback: simple Haar cascade via OpenCV
            return self._haar_fallback(image)

        results = self._detector.process(image)
        if not results.detections:
            return None, {}

        det = results.detections[0]  # pick highest-confidence face
        bb = det.location_data.relative_bounding_box
        x = int(bb.xmin * w)
        y = int(bb.ymin * h)
        bw = int(bb.width * w)
        bh = int(bb.height * h)
        bbox = (max(x, 0), max(y, 0), bw, bh)

        # Extract the 6 MediaPipe face-detection keypoints
        keypoint_names = [
            "right_eye", "left_eye", "nose_tip",
            "mouth_center", "right_ear_tragion", "left_ear_tragion",
        ]
        landmarks = {}
        for i, kp in enumerate(det.location_data.relative_keypoints):
            name = keypoint_names[i] if i < len(keypoint_names) else f"kp_{i}"
            landmarks[name] = (float(kp.x), float(kp.y))

        return bbox, landmarks

    def _haar_fallback(self, image: np.ndarray):
        """Fallback face detection using OpenCV Haar cascade."""
        h, w, _ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(60, 60))
        if len(faces) == 0:
            return None, {}

        # pick largest face
        areas = [fw * fh for (_, _, fw, fh) in faces]
        idx = int(np.argmax(areas))
        x, y, fw, fh = faces[idx]
        bbox = (int(x), int(y), int(fw), int(fh))

        # approximate landmarks from bbox
        cx, cy = x + fw / 2, y + fh / 2
        landmarks = {
            "right_eye": ((cx - fw * 0.15) / w, (cy - fh * 0.15) / h),
            "left_eye": ((cx + fw * 0.15) / w, (cy - fh * 0.15) / h),
            "nose_tip": (cx / w, cy / h),
            "mouth_center": (cx / w, (cy + fh * 0.2) / h),
        }
        return bbox, landmarks

    def _centre_crop_fallback(self, h: int, w: int):
        """When no face is found, assume the face is centered."""
        side = min(h, w)
        x = (w - side) // 2
        y = (h - side) // 2
        bbox = (x, y, side, side)
        cx, cy = 0.5, 0.5
        landmarks = {
            "right_eye": (cx - 0.05, cy - 0.05),
            "left_eye": (cx + 0.05, cy - 0.05),
            "nose_tip": (cx, cy),
            "mouth_center": (cx, cy + 0.08),
        }
        return bbox, landmarks

    def _crop_with_margin(self, image: np.ndarray, bbox: tuple,
                          margin_factor: float = 0.4) -> np.ndarray:
        """Crop face region with extra margin around the bounding box."""
        h, w, _ = image.shape
        x, y, bw, bh = bbox
        mx = int(bw * margin_factor)
        my = int(bh * margin_factor)
        x1 = max(x - mx, 0)
        y1 = max(y - my, 0)
        x2 = min(x + bw + mx, w)
        y2 = min(y + bh + my, h)
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return image  # safety fallback
        return cropped

    def _align_face(self, cropped: np.ndarray, landmarks: dict,
                    bbox: tuple) -> np.ndarray:
        """Rotate image so that the line between the eyes is horizontal."""
        if "left_eye" not in landmarks or "right_eye" not in landmarks:
            return cropped

        h, w, _ = cropped.shape
        # landmarks are in normalised coords relative to the original image;
        # convert to pixel coords within the cropped region is impractical
        # without the crop offset, so we use a simpler heuristic: compute
        # angle from the normalised coords (they preserve relative positions).
        lx, ly = landmarks["left_eye"]
        rx, ry = landmarks["right_eye"]
        angle_rad = np.arctan2(ly - ry, lx - rx)
        angle_deg = float(np.degrees(angle_rad))

        # only rotate if tilt is significant but not extreme
        if abs(angle_deg) < 0.5 or abs(angle_deg) > 45:
            return cropped

        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        aligned = cv2.warpAffine(cropped, rot_mat, (w, h),
                                 flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REFLECT_101)
        return aligned

    @staticmethod
    def _normalise_lighting(image_rgb: np.ndarray) -> np.ndarray:
        """Histogram equalisation on the L channel of LAB colour space."""
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        self._initialized = False
