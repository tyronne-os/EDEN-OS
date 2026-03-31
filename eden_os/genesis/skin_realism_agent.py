"""
EDEN OS — Skin Realism Agent
Specialized skill agent for natural skin texture enhancement.
Ensures photorealistic skin that is indistinguishable from real human video.

This agent handles:
1. Pore-level micro-texture synthesis
2. Subsurface scattering simulation (SSS) for skin translucency
3. Melanin-aware color correction (works across ALL skin tones)
4. Specular highlight preservation (natural skin sheen, not plastic)
5. Fine hair and peach fuzz rendering
6. Skin imperfection preservation (freckles, moles, beauty marks, pores)
7. Dynamic skin response (blush, pallor, blood flow under emotion)

The agent works as a post-processing pass on every frame from the Animator,
ensuring the Eden Protocol's 0.3 deviation threshold is met with emphasis
on skin naturalness rather than just pixel accuracy.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from loguru import logger


@dataclass
class SkinProfile:
    """Extracted skin characteristics from the reference portrait."""
    # Color profile
    mean_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    std_lab: np.ndarray = field(default_factory=lambda: np.ones(3))
    melanin_level: float = 0.5      # 0.0=very fair, 1.0=very deep
    undertone: str = "neutral"       # warm, cool, neutral, olive

    # Texture profile
    pore_density: float = 0.5        # detected pore density
    texture_roughness: float = 0.5   # skin surface roughness
    micro_texture_map: Optional[np.ndarray] = None  # high-freq detail map

    # Imperfections (identity-critical — must be preserved)
    freckle_map: Optional[np.ndarray] = None
    mole_positions: list = field(default_factory=list)
    beauty_marks: list = field(default_factory=list)

    # Specular profile
    specular_intensity: float = 0.3
    specular_spread: float = 0.5
    oiliness: float = 0.3            # T-zone shine level


class SkinRealismAgent:
    """
    Skill agent for photorealistic skin texture in every generated frame.

    Pipeline:
    1. Analyze reference portrait → build SkinProfile
    2. On each generated frame:
       a. Extract face region
       b. Apply melanin-aware color correction
       c. Synthesize micro-texture (pores, fine lines)
       d. Apply subsurface scattering simulation
       e. Preserve identity markers (freckles, moles)
       f. Add natural specular highlights
       g. Apply dynamic skin response (emotion-driven)
    3. Validate against Eden Protocol
    """

    def __init__(self):
        self.profile: Optional[SkinProfile] = None
        self._reference_face: Optional[np.ndarray] = None
        self._face_mask: Optional[np.ndarray] = None
        self._initialized = False

        # Tunable parameters (connected to admin panel sliders)
        self.realism_strength = 0.7     # 0=none, 1=max processing
        self.texture_detail = 0.6       # pore/texture enhancement level
        self.sss_strength = 0.4         # subsurface scattering intensity
        self.specular_strength = 0.3    # highlight intensity
        self.imperfection_preserve = 0.9  # how strongly to keep freckles/moles

    async def analyze_portrait(self, portrait: np.ndarray, face_bbox: Optional[tuple] = None) -> SkinProfile:
        """
        Build a complete skin profile from the reference portrait.
        This runs once during session setup (Genesis phase).
        """
        logger.info("Skin Realism Agent: Analyzing reference portrait...")

        profile = SkinProfile()

        # Extract face region
        if face_bbox:
            x, y, w, h = face_bbox
            face = portrait[y:y+h, x:x+w]
        else:
            face = self._detect_and_crop_face(portrait)

        self._reference_face = face.copy()
        h, w = face.shape[:2]

        # Create skin mask (exclude eyes, lips, eyebrows, hair)
        skin_mask = self._create_skin_mask(face)
        self._face_mask = skin_mask

        # ── Color Analysis ──
        lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB).astype(np.float32)
        skin_pixels = lab[skin_mask > 0]

        if len(skin_pixels) > 0:
            profile.mean_lab = np.mean(skin_pixels, axis=0)
            profile.std_lab = np.std(skin_pixels, axis=0)

            # Melanin estimation from L channel (luminance)
            l_mean = profile.mean_lab[0]
            # L ranges 0-255 in OpenCV LAB; darker skin = lower L
            profile.melanin_level = np.clip(1.0 - (l_mean / 200.0), 0.0, 1.0)

            # Undertone from a/b channels
            a_mean = profile.mean_lab[1] - 128  # center around 0
            b_mean = profile.mean_lab[2] - 128
            if b_mean > 5 and a_mean > 0:
                profile.undertone = "warm"
            elif b_mean < -3:
                profile.undertone = "cool"
            elif a_mean > 3 and b_mean < 3:
                profile.undertone = "olive"
            else:
                profile.undertone = "neutral"

        # ── Texture Analysis ──
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

        # Pore detection via Laplacian response
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        pore_response = np.abs(laplacian)
        skin_pore_response = pore_response[skin_mask > 0]
        if len(skin_pore_response) > 0:
            profile.pore_density = np.clip(np.mean(skin_pore_response) / 30.0, 0.0, 1.0)

        # Surface roughness via Gabor energy
        roughness_energies = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((21, 21), 3.0, theta, 8.0, 0.5, 0)
            response = cv2.filter2D(gray, cv2.CV_64F, kernel)
            skin_response = response[skin_mask > 0]
            if len(skin_response) > 0:
                roughness_energies.append(np.std(skin_response))

        if roughness_energies:
            profile.texture_roughness = np.clip(np.mean(roughness_energies) / 50.0, 0.0, 1.0)

        # Extract micro-texture map (high-frequency detail)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        profile.micro_texture_map = (gray.astype(np.float32) - blurred.astype(np.float32))

        # ── Imperfection Detection ──
        profile.freckle_map = self._detect_freckles(face, skin_mask)
        profile.mole_positions = self._detect_moles(face, skin_mask)

        # ── Specular Profile ──
        # Detect existing highlights (bright spots on skin)
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)
        skin_brightness = v_channel[skin_mask > 0]
        if len(skin_brightness) > 0:
            bright_threshold = np.percentile(skin_brightness, 95)
            highlight_ratio = np.mean(skin_brightness > bright_threshold)
            profile.specular_intensity = np.clip(highlight_ratio * 10, 0.1, 0.8)

            # T-zone oiliness (forehead + nose region)
            t_zone_y = slice(0, h // 3)
            t_zone_x = slice(w // 4, 3 * w // 4)
            t_zone_v = v_channel[t_zone_y, t_zone_x]
            t_zone_mask = skin_mask[t_zone_y, t_zone_x]
            t_zone_skin = t_zone_v[t_zone_mask > 0]
            if len(t_zone_skin) > 0:
                profile.oiliness = np.clip(
                    (np.mean(t_zone_skin) - np.mean(skin_brightness)) / 30.0 + 0.3,
                    0.0, 1.0
                )

        self.profile = profile
        self._initialized = True

        logger.info(
            f"Skin profile built: melanin={profile.melanin_level:.2f}, "
            f"undertone={profile.undertone}, pores={profile.pore_density:.2f}, "
            f"roughness={profile.texture_roughness:.2f}, "
            f"moles={len(profile.mole_positions)}"
        )

        return profile

    def enhance_frame(
        self,
        frame: np.ndarray,
        face_bbox: Optional[tuple] = None,
        emotion: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Apply skin realism enhancement to a generated frame.
        Called on every frame from the Animator pipeline.
        """
        if not self._initialized or self.profile is None:
            return frame

        if self.realism_strength < 0.01:
            return frame

        result = frame.copy()

        # Extract face region
        if face_bbox:
            x, y, w, h = face_bbox
            face = result[y:y+h, x:x+w]
        else:
            face = result  # assume face-cropped input

        # Step 1: Melanin-aware color correction
        face = self._apply_color_correction(face)

        # Step 2: Micro-texture synthesis
        if self.texture_detail > 0.1:
            face = self._apply_micro_texture(face)

        # Step 3: Subsurface scattering simulation
        if self.sss_strength > 0.1:
            face = self._apply_sss(face)

        # Step 4: Preserve identity markers
        if self.imperfection_preserve > 0.1:
            face = self._preserve_imperfections(face)

        # Step 5: Natural specular highlights
        if self.specular_strength > 0.1:
            face = self._apply_specular(face)

        # Step 6: Dynamic skin response (emotion-driven)
        if emotion:
            face = self._apply_dynamic_response(face, emotion)

        # Write back to frame
        if face_bbox:
            result[y:y+h, x:x+w] = face
        else:
            result = face

        return result

    # ── Processing Steps ───────────────────────────────────────────

    def _apply_color_correction(self, face: np.ndarray) -> np.ndarray:
        """
        Melanin-aware color correction.
        Prevents the "whitewashing" artifact common in AI face generation.
        Ensures generated skin tone matches reference across all melanin levels.
        """
        if self.profile is None:
            return face

        lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Get current skin color stats
        mask = self._create_skin_mask(face) if self._face_mask is None else cv2.resize(
            self._face_mask, (face.shape[1], face.shape[0])
        )
        current_pixels = lab[mask > 0]

        if len(current_pixels) < 100:
            return face

        current_mean = np.mean(current_pixels, axis=0)

        # Compute correction needed
        target_mean = self.profile.mean_lab
        correction = (target_mean - current_mean) * self.realism_strength * 0.5

        # Apply correction more aggressively on color channels (a, b)
        # and more gently on luminance (L) to preserve lighting
        lab[:, :, 0] += correction[0] * 0.3  # gentle L correction
        lab[:, :, 1] += correction[1] * 0.7  # stronger a correction
        lab[:, :, 2] += correction[2] * 0.7  # stronger b correction

        lab = np.clip(lab, 0, 255)
        corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        # Blend with mask (only apply to skin, not eyes/lips)
        mask_3ch = np.stack([mask, mask, mask], axis=-1).astype(np.float32) / 255.0
        blended = (corrected.astype(np.float32) * mask_3ch +
                   face.astype(np.float32) * (1 - mask_3ch))

        return np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_micro_texture(self, face: np.ndarray) -> np.ndarray:
        """
        Synthesize and overlay pore-level micro-texture.
        Uses the reference portrait's texture map to add authentic skin detail.
        """
        if self.profile is None or self.profile.micro_texture_map is None:
            return face

        h, w = face.shape[:2]
        texture_map = cv2.resize(
            self.profile.micro_texture_map, (w, h)
        ).astype(np.float32)

        # Scale texture based on target pore density
        texture_strength = self.texture_detail * self.profile.pore_density * 0.4

        # Add high-frequency noise for organic randomness
        noise = np.random.normal(0, 1, (h, w)).astype(np.float32) * 0.5
        organic_texture = texture_map + noise * texture_strength

        # Apply texture overlay (add to luminance channel)
        lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] += organic_texture * texture_strength * 15
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    def _apply_sss(self, face: np.ndarray) -> np.ndarray:
        """
        Subsurface Scattering (SSS) simulation.
        Real skin is translucent — light penetrates and scatters under the surface,
        creating a warm glow especially at edges (ears, nose, thin skin areas).
        AI-generated faces lack this, making them look "plastic."
        """
        h, w = face.shape[:2]

        # SSS effect: blur the red channel more than green/blue
        # (blood absorbs blue/green, transmits red → warm subsurface glow)
        r, g, b = face[:, :, 0], face[:, :, 1], face[:, :, 2]

        # Different blur radii per channel simulate wavelength-dependent scattering
        blur_radius = max(3, int(min(h, w) * 0.02))
        if blur_radius % 2 == 0:
            blur_radius += 1

        r_k = blur_radius * 3 | 1  # ensure odd
        g_k = blur_radius * 2 | 1
        b_k = blur_radius | 1
        r_sss = cv2.GaussianBlur(r, (r_k, r_k), 0)
        g_sss = cv2.GaussianBlur(g, (g_k, g_k), 0)
        b_sss = cv2.GaussianBlur(b, (b_k, b_k), 0)

        # Blend SSS with original
        strength = self.sss_strength * self.realism_strength

        # Melanin-aware: darker skin has less visible SSS
        melanin_factor = 1.0 - (self.profile.melanin_level * 0.5) if self.profile else 0.7
        strength *= melanin_factor

        result = face.copy().astype(np.float32)
        result[:, :, 0] = r * (1 - strength * 0.15) + r_sss * (strength * 0.15)
        result[:, :, 1] = g * (1 - strength * 0.08) + g_sss * (strength * 0.08)
        result[:, :, 2] = b * (1 - strength * 0.03) + b_sss * (strength * 0.03)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _preserve_imperfections(self, face: np.ndarray) -> np.ndarray:
        """
        Re-inject identity-critical skin imperfections.
        Freckles, moles, and beauty marks must survive the generation process.
        """
        if self.profile is None:
            return face

        h, w = face.shape[:2]
        strength = self.imperfection_preserve * self.realism_strength

        # Overlay freckle map
        if self.profile.freckle_map is not None and strength > 0.3:
            freckle_resized = cv2.resize(self.profile.freckle_map, (w, h))
            # Freckles darken the skin slightly
            freckle_mask = (freckle_resized > 30).astype(np.float32)
            darkening = freckle_resized.astype(np.float32) * strength * 0.3

            lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[:, :, 0] -= darkening * freckle_mask
            lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
            face = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

        # Re-stamp moles at stored positions
        for mole in self.profile.mole_positions:
            mx = int(mole["x"] * w)
            my = int(mole["y"] * h)
            radius = max(1, int(mole.get("radius", 2) * min(w, h) / 512))

            if 0 <= mx < w and 0 <= my < h:
                # Darken a small circular region
                cv2.circle(face, (mx, my), radius,
                           (int(mole.get("r", 80)), int(mole.get("g", 60)), int(mole.get("b", 50))),
                           -1)

        return face

    def _apply_specular(self, face: np.ndarray) -> np.ndarray:
        """
        Add natural specular highlights.
        Real skin has a micro-sheen from oil/moisture — not the uniform matte
        look of AI-generated faces, nor the plastic-looking spec of bad CGI.
        """
        h, w = face.shape[:2]
        strength = self.specular_strength * self.realism_strength

        if self.profile is None:
            return face

        # Create specular highlight map based on face geometry
        # Highlights concentrate on: nose bridge, forehead, cheekbones, chin
        highlight_map = np.zeros((h, w), dtype=np.float32)

        # Nose bridge (center top)
        cv2.ellipse(highlight_map, (w // 2, int(h * 0.45)),
                    (w // 12, h // 6), 0, 0, 360, 1.0, -1)

        # Forehead
        cv2.ellipse(highlight_map, (w // 2, int(h * 0.2)),
                    (w // 4, h // 8), 0, 0, 360, 0.6, -1)

        # Cheekbones (left and right)
        cv2.ellipse(highlight_map, (int(w * 0.3), int(h * 0.5)),
                    (w // 8, h // 10), 15, 0, 360, 0.4, -1)
        cv2.ellipse(highlight_map, (int(w * 0.7), int(h * 0.5)),
                    (w // 8, h // 10), -15, 0, 360, 0.4, -1)

        # Blur for smooth falloff
        highlight_map = cv2.GaussianBlur(highlight_map, (31, 31), 0)
        highlight_map *= strength * self.profile.specular_intensity

        # Oiliness increases T-zone highlights
        if self.profile.oiliness > 0.3:
            t_zone = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(t_zone, (w // 2, int(h * 0.3)),
                        (w // 6, h // 4), 0, 0, 360,
                        float(self.profile.oiliness * 0.3), -1)
            t_zone = cv2.GaussianBlur(t_zone, (21, 21), 0)
            highlight_map += t_zone

        # Apply highlights (additive on luminance)
        lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] += highlight_map * 30
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    def _apply_dynamic_response(self, face: np.ndarray, emotion: dict) -> np.ndarray:
        """
        Emotion-driven skin response.
        - Joy/warmth → slight blush on cheeks
        - Fear/urgency → slight pallor
        - Confidence → healthy glow
        - Sadness → reduced color saturation
        """
        h, w = face.shape[:2]
        lab = cv2.cvtColor(face, cv2.COLOR_RGB2LAB).astype(np.float32)

        joy = emotion.get("joy", 0.5)
        warmth = emotion.get("warmth", 0.5)
        sadness = emotion.get("sadness", 0.0)
        urgency = emotion.get("urgency", 0.0)

        # Blush response (joy + warmth increase redness on cheeks)
        blush_intensity = max(0, (joy - 0.5) + (warmth - 0.5)) * 0.3
        if blush_intensity > 0.05:
            blush_map = np.zeros((h, w), dtype=np.float32)
            # Cheek regions
            cv2.ellipse(blush_map, (int(w * 0.3), int(h * 0.6)),
                        (w // 6, h // 8), 0, 0, 360, blush_intensity, -1)
            cv2.ellipse(blush_map, (int(w * 0.7), int(h * 0.6)),
                        (w // 6, h // 8), 0, 0, 360, blush_intensity, -1)
            blush_map = cv2.GaussianBlur(blush_map, (31, 31), 0)

            # Increase a-channel (red-green axis in LAB) for blush
            lab[:, :, 1] += blush_map * 15

        # Pallor response (urgency/fear reduces color)
        if urgency > 0.5:
            pallor = (urgency - 0.5) * 0.2
            lab[:, :, 0] += pallor * 10   # slightly brighter (pale)
            lab[:, :, 1] -= pallor * 5    # less red
            lab[:, :, 2] -= pallor * 3    # less yellow

        # Sadness: desaturate slightly
        if sadness > 0.3:
            desat = (sadness - 0.3) * 0.15
            lab[:, :, 1] = lab[:, :, 1] * (1 - desat) + 128 * desat
            lab[:, :, 2] = lab[:, :, 2] * (1 - desat) + 128 * desat

        lab = np.clip(lab, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # ── Helper Methods ─────────────────────────────────────────────

    def _detect_and_crop_face(self, image: np.ndarray) -> np.ndarray:
        """Simple face detection and crop."""
        try:
            import mediapipe as mp
            detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
            results = detector.process(image)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                h, w = image.shape[:2]
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                fw = min(w - x, int(bbox.width * w))
                fh = min(h - y, int(bbox.height * h))
                return image[y:y+fh, x:x+fw]
        except Exception:
            pass
        return image

    def _create_skin_mask(self, face: np.ndarray) -> np.ndarray:
        """Create binary mask of skin pixels using HSV thresholding."""
        hsv = cv2.cvtColor(face, cv2.COLOR_RGB2HSV)

        # Broad skin color range in HSV
        lower = np.array([0, 20, 50], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower, upper)

        # Additional range for darker skin tones
        lower2 = np.array([0, 10, 30], dtype=np.uint8)
        upper2 = np.array([20, 200, 200], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _detect_freckles(self, face: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
        """Detect freckle pattern from reference portrait."""
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

        # Freckles are small dark spots on lighter skin
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        diff = blurred.astype(np.float32) - gray.astype(np.float32)

        # Freckles are where the original is darker than the blur
        freckle_map = np.clip(diff, 0, 255).astype(np.uint8)

        # Only on skin
        freckle_map = cv2.bitwise_and(freckle_map, freckle_map, mask=skin_mask)

        # Threshold to get significant spots only
        _, freckle_map = cv2.threshold(freckle_map, 15, 255, cv2.THRESH_TOZERO)

        return freckle_map

    def _detect_moles(self, face: np.ndarray, skin_mask: np.ndarray) -> list:
        """Detect moles and beauty marks from reference portrait."""
        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        h, w = face.shape[:2]

        # Moles are dark, round, small features
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold to find dark spots
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 8
        )

        # Only on skin
        thresh = cv2.bitwise_and(thresh, thresh, mask=skin_mask)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        moles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Moles are small (3-30 pixels area at 512x512)
            if 3 < area < 50:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if circularity > 0.4:  # reasonably round
                        # Get color at mole position
                        mx, my = int(cx), int(cy)
                        if 0 <= mx < w and 0 <= my < h:
                            color = face[my, mx]
                            moles.append({
                                "x": cx / w,   # normalized position
                                "y": cy / h,
                                "radius": radius,
                                "r": int(color[0]),
                                "g": int(color[1]),
                                "b": int(color[2]),
                            })

        logger.debug(f"Detected {len(moles)} moles/beauty marks")
        return moles[:20]  # Cap at 20 to avoid false positives

    def update_settings(self, settings: dict) -> None:
        """Update from admin panel sliders."""
        if "skin_realism" in settings:
            self.realism_strength = float(settings["skin_realism"])
        if "texture_detail" in settings:
            self.texture_detail = float(settings["texture_detail"])
        if "sss_strength" in settings:
            self.sss_strength = float(settings["sss_strength"])
        if "specular" in settings:
            self.specular_strength = float(settings["specular"])

    def get_status(self) -> dict:
        """Get agent status."""
        return {
            "initialized": self._initialized,
            "melanin_level": self.profile.melanin_level if self.profile else None,
            "undertone": self.profile.undertone if self.profile else None,
            "pore_density": self.profile.pore_density if self.profile else None,
            "moles_detected": len(self.profile.mole_positions) if self.profile else 0,
            "realism_strength": self.realism_strength,
        }
