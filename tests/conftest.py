"""
EDEN OS — Test Suite Shared Fixtures
Provides synthetic test data for visual and vocal realism testing.
"""

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ═══════════════════════════════════════════════════════════════
# Synthetic Portrait Generation
# ═══════════════════════════════════════════════════════════════

def generate_skin_toned_portrait(
    size: int = 512,
    melanin: float = 0.5,
    add_features: bool = True,
) -> np.ndarray:
    """
    Generate a synthetic skin-toned portrait for testing.

    Args:
        size: Image size (square)
        melanin: 0.0 = very fair, 1.0 = very deep
        add_features: Add eyes, nose, mouth regions

    Returns:
        RGB uint8 numpy array (size, size, 3)
    """
    # Base skin color in LAB space (perceptually accurate)
    # L: 85 (fair) to 35 (deep), a: 8-15, b: 15-30
    L = int(85 - melanin * 50)
    a = int(128 + 8 + melanin * 7)  # slight red
    b = int(128 + 15 + melanin * 15)  # warm yellow

    # Create base face with slight gradient (forehead lighter, jaw darker)
    lab = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        gradient = (y / size) * 8  # subtle vertical gradient
        lab[y, :, 0] = np.clip(L - gradient, 0, 255)
        lab[y, :, 1] = a
        lab[y, :, 2] = b

    # Add skin texture noise
    noise = np.random.normal(0, 2, (size, size)).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0].astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Convert to RGB
    portrait = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    if add_features:
        # Add facial features as darker/lighter regions
        cx, cy = size // 2, size // 2

        # Eyes (darker oval regions)
        eye_color = np.clip(np.array([L - 40, a, b]), 0, 255).astype(np.uint8)
        eye_rgb = cv2.cvtColor(np.array([[eye_color]], dtype=np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
        cv2.ellipse(portrait, (cx - size // 6, cy - size // 10),
                    (size // 14, size // 20), 0, 0, 360, eye_rgb.tolist(), -1)
        cv2.ellipse(portrait, (cx + size // 6, cy - size // 10),
                    (size // 14, size // 20), 0, 0, 360, eye_rgb.tolist(), -1)

        # Nose (slightly darker triangle)
        nose_pts = np.array([
            [cx, cy - size // 20],
            [cx - size // 16, cy + size // 10],
            [cx + size // 16, cy + size // 10],
        ])
        nose_color = np.clip(np.array([max(0, int(portrait[cy, cx, 0]) - 10),
                                        max(0, int(portrait[cy, cx, 1]) - 5),
                                        max(0, int(portrait[cy, cx, 2]) - 5)]), 0, 255)
        cv2.fillPoly(portrait, [nose_pts], nose_color.tolist())

        # Mouth (reddish region)
        lip_color = [min(255, int(portrait[cy, cx, 0]) + 30),
                     max(0, int(portrait[cy, cx, 1]) - 20),
                     max(0, int(portrait[cy, cx, 2]) - 15)]
        cv2.ellipse(portrait, (cx, cy + size // 5),
                    (size // 8, size // 18), 0, 0, 360, lip_color, -1)

        # Eyebrows (darker arcs)
        brow_color = [max(0, int(portrait[cy, cx, c]) - 40) for c in range(3)]
        cv2.ellipse(portrait, (cx - size // 6, cy - size // 6),
                    (size // 10, size // 30), -10, 0, 180, brow_color, 2)
        cv2.ellipse(portrait, (cx + size // 6, cy - size // 6),
                    (size // 10, size // 30), 10, 0, 180, brow_color, 2)

    # Smooth to look more like a real face
    portrait = cv2.GaussianBlur(portrait, (3, 3), 0)

    return portrait


def generate_synthetic_audio(
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 220.0,
    noise_level: float = 0.01,
) -> np.ndarray:
    """Generate synthetic audio for testing (speech-like sine wave with harmonics)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)

    # Fundamental + harmonics (like a vowel)
    signal = (
        0.5 * np.sin(2 * np.pi * frequency * t) +
        0.25 * np.sin(2 * np.pi * frequency * 2 * t) +
        0.12 * np.sin(2 * np.pi * frequency * 3 * t) +
        0.06 * np.sin(2 * np.pi * frequency * 4 * t)
    )

    # Amplitude envelope (natural speech rise/fall)
    envelope = np.ones_like(t)
    attack = int(0.05 * sample_rate)
    release = int(0.1 * sample_rate)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if release > 0:
        envelope[-release:] = np.linspace(1, 0, release)

    signal *= envelope * 0.5

    # Add slight noise
    signal += np.random.normal(0, noise_level, len(t)).astype(np.float32)

    return np.clip(signal, -1.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# Pytest Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def fair_portrait():
    """Very fair skin tone portrait (melanin ~0.1)."""
    return generate_skin_toned_portrait(512, melanin=0.1)


@pytest.fixture
def medium_portrait():
    """Medium skin tone portrait (melanin ~0.5)."""
    return generate_skin_toned_portrait(512, melanin=0.5)


@pytest.fixture
def deep_portrait():
    """Deep skin tone portrait (melanin ~0.9)."""
    return generate_skin_toned_portrait(512, melanin=0.9)


@pytest.fixture
def portrait_set():
    """5 portraits spanning full melanin range."""
    return {
        "very_fair": generate_skin_toned_portrait(512, melanin=0.05),
        "fair": generate_skin_toned_portrait(512, melanin=0.25),
        "medium": generate_skin_toned_portrait(512, melanin=0.5),
        "dark": generate_skin_toned_portrait(512, melanin=0.75),
        "very_deep": generate_skin_toned_portrait(512, melanin=0.95),
    }


@pytest.fixture
def plastic_portrait():
    """Heavily smoothed portrait simulating AI 'plastic skin' artifact."""
    base = generate_skin_toned_portrait(512, melanin=0.4)
    # Heavy gaussian blur destroys micro-texture
    plastic = cv2.GaussianBlur(base, (31, 31), 10)
    return plastic


@pytest.fixture
def speech_audio():
    """1 second of speech-like audio at 16kHz."""
    return generate_synthetic_audio(1.0, 16000, 220.0)


@pytest.fixture
def silence_audio():
    """1 second of near-silence."""
    return np.zeros(16000, dtype=np.float32) + np.random.normal(0, 0.0001, 16000).astype(np.float32)


@pytest.fixture
def noise_audio():
    """1 second of white noise."""
    return (np.random.normal(0, 0.3, 16000)).astype(np.float32)


@pytest.fixture
def long_speech_audio():
    """5 seconds of speech-like audio with natural pauses."""
    segments = []
    for i in range(5):
        # Speech segment
        freq = 180 + np.random.randint(0, 80)
        seg = generate_synthetic_audio(0.6, 16000, freq)
        segments.append(seg)
        # Pause
        pause = np.zeros(int(0.2 * 16000), dtype=np.float32)
        segments.append(pause)
    return np.concatenate(segments)


# ═══════════════════════════════════════════════════════════════
# Visual Realism Fixtures
# ═══════════════════════════════════════════════════════════════

def make_face_with_dark_spots(
    size: int = 512,
    spot_positions: list | None = None,
    spot_radius: int = 6,
    melanin: float = 0.3,
) -> np.ndarray:
    """Generate a skin-toned face with synthetic dark spots (freckles)."""
    face = generate_skin_toned_portrait(size, melanin=melanin, add_features=False)
    if spot_positions is None:
        spot_positions = [
            (size // 3, size // 3),
            (2 * size // 3, size // 3),
            (size // 2, size // 2),
            (size // 4, size // 2),
            (3 * size // 4, 2 * size // 3),
        ]
    for (cx, cy) in spot_positions:
        cv2.circle(face, (cx, cy), spot_radius, (80, 60, 50), -1)
    return face


def make_face_with_moles(
    size: int = 512,
    mole_centers: list | None = None,
    melanin: float = 0.3,
) -> np.ndarray:
    """Generate a skin-toned face with 3 distinct synthetic moles."""
    face = generate_skin_toned_portrait(size, melanin=melanin, add_features=False)
    if mole_centers is None:
        mole_centers = [
            (size // 4, size // 3),
            (3 * size // 4, size // 2),
            (size // 2, 3 * size // 4),
        ]
    for (cx, cy) in mole_centers:
        cv2.circle(face, (cx, cy), 3, (50, 35, 30), -1)
    return face


def make_base_keypoints(num_kp: int = 21) -> np.ndarray:
    """Return neutral face keypoints matching LivePortraitDriver layout."""
    kp = np.zeros((num_kp, 3), dtype=np.float32)
    for i in range(5):
        angle = np.pi * (0.3 + 0.4 * i / 4)
        kp[i] = [np.cos(angle) * 0.4, np.sin(angle) * 0.4 + 0.1, 0.0]
    kp[5] = [-0.15, -0.1, 0.0]
    kp[6] = [-0.08, -0.12, 0.0]
    kp[7] = [-0.08, -0.08, 0.0]
    kp[8] = [0.15, -0.1, 0.0]
    kp[9] = [0.08, -0.12, 0.0]
    kp[10] = [0.08, -0.08, 0.0]
    kp[11] = [0.0, -0.05, 0.02]
    kp[12] = [-0.03, 0.03, 0.01]
    kp[13] = [0.03, 0.03, 0.01]
    kp[14] = [-0.08, 0.12, 0.0]
    kp[15] = [0.08, 0.12, 0.0]
    kp[16] = [0.0, 0.10, 0.0]
    kp[17] = [0.0, 0.14, 0.0]
    kp[18] = [0.0, 0.12, 0.0]
    kp[19] = [-0.12, -0.18, 0.0]
    kp[20] = [0.12, -0.18, 0.0]
    return kp


@pytest.fixture
def freckled_portrait():
    """Portrait with synthetic dark spots (freckles)."""
    return make_face_with_dark_spots(512)


@pytest.fixture
def mole_portrait():
    """Portrait with 3 synthetic moles at known positions."""
    return make_face_with_moles(512)


@pytest.fixture
def base_keypoints():
    """Neutral face keypoints matching LivePortraitDriver layout."""
    return make_base_keypoints()


@pytest.fixture
def skin_agent():
    """Pre-configured SkinRealismAgent instance."""
    from eden_os.genesis.skin_realism_agent import SkinRealismAgent
    return SkinRealismAgent()


@pytest.fixture
def eden_validator():
    """Pre-configured EdenProtocolValidator instance."""
    from eden_os.genesis.eden_protocol_validator import EdenProtocolValidator
    return EdenProtocolValidator()


@pytest.fixture
def idle_generator():
    """Pre-configured IdleGenerator at 30 fps."""
    from eden_os.animator.idle_generator import IdleGenerator
    return IdleGenerator(fps=30.0)


@pytest.fixture
def state_machine():
    """Pre-configured AvatarStateMachine."""
    from eden_os.animator.state_machine import AvatarStateMachine
    return AvatarStateMachine()


@pytest.fixture
def temporal_anchor():
    """Pre-configured EdenTemporalAnchor."""
    from eden_os.animator.eden_temporal_anchor import EdenTemporalAnchor
    return EdenTemporalAnchor()


@pytest.fixture
def liveportrait_driver():
    """LivePortraitDriver with a source image already set."""
    from eden_os.animator.liveportrait_driver import LivePortraitDriver
    driver = LivePortraitDriver()
    driver.is_loaded = True
    driver.set_source_image(generate_skin_toned_portrait(512, melanin=0.4))
    return driver
