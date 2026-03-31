"""
EDEN OS — Audio-to-Keypoints Bridge
Converts audio waveform features into LivePortrait-compatible implicit keypoint deltas.
This replaces the need for a driving video — audio becomes the driver.
"""

import numpy as np
from loguru import logger


class AudioToKeypoints:
    """Converts audio features to facial keypoint deltas for lip-sync animation."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._mel_bands = 80
        self._hop_length = 160  # 10ms at 16kHz
        self._win_length = 400  # 25ms at 16kHz

        # Viseme mapping: phoneme categories to mouth shape parameters
        # Each viseme defines: mouth_open, mouth_width, lip_round
        self._energy_history: list[float] = []
        self._pitch_history: list[float] = []
        self._smoothing_alpha = 0.3

    def extract_features(self, audio_chunk: np.ndarray) -> dict:
        """
        Extract audio features from a PCM audio chunk.

        Returns dict with:
            energy: float 0.0-1.0 (volume/loudness)
            pitch: float 0.0-1.0 (normalized fundamental frequency)
            spectral_centroid: float (brightness of sound)
            is_voiced: bool (speech detected vs silence)
        """
        if len(audio_chunk) == 0:
            return {
                "energy": 0.0,
                "pitch": 0.0,
                "spectral_centroid": 0.0,
                "is_voiced": False,
            }

        # Normalize audio
        audio = audio_chunk.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        # RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        energy = np.clip(rms * 10.0, 0.0, 1.0)

        # Smooth energy
        self._energy_history.append(energy)
        if len(self._energy_history) > 10:
            self._energy_history.pop(0)
        smoothed_energy = np.mean(self._energy_history[-5:])

        # Pitch estimation via autocorrelation
        pitch = self._estimate_pitch(audio)

        # Smooth pitch
        self._pitch_history.append(pitch)
        if len(self._pitch_history) > 10:
            self._pitch_history.pop(0)
        smoothed_pitch = np.mean(self._pitch_history[-3:])

        # Spectral centroid
        spectrum = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / self.sample_rate)
        if spectrum.sum() > 0:
            centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            centroid_norm = np.clip(centroid / 4000.0, 0.0, 1.0)
        else:
            centroid_norm = 0.0

        # Voice activity
        is_voiced = smoothed_energy > 0.02

        return {
            "energy": float(smoothed_energy),
            "pitch": float(smoothed_pitch),
            "spectral_centroid": float(centroid_norm),
            "is_voiced": bool(is_voiced),
        }

    def _estimate_pitch(self, audio: np.ndarray) -> float:
        """Estimate pitch via autocorrelation method."""
        if len(audio) < 512:
            return 0.0

        # Autocorrelation
        corr = np.correlate(audio[:1024], audio[:1024], mode="full")
        corr = corr[len(corr) // 2:]

        # Find first peak after initial decline
        # Min period: 50Hz = 320 samples at 16kHz
        # Max period: 400Hz = 40 samples at 16kHz
        min_lag = self.sample_rate // 400  # ~40
        max_lag = self.sample_rate // 50   # ~320

        if max_lag >= len(corr):
            max_lag = len(corr) - 1
        if min_lag >= max_lag:
            return 0.0

        search_region = corr[min_lag:max_lag]
        if len(search_region) == 0:
            return 0.0

        peak_idx = np.argmax(search_region) + min_lag

        if corr[0] > 0 and corr[peak_idx] / corr[0] > 0.3:
            freq = self.sample_rate / peak_idx
            # Normalize to 0-1 range (80Hz = 0.0, 400Hz = 1.0)
            pitch_norm = np.clip((freq - 80) / 320, 0.0, 1.0)
            return float(pitch_norm)

        return 0.0

    def features_to_keypoint_delta(self, features: dict) -> dict:
        """
        Convert extracted audio features to animation parameters.

        Returns dict compatible with LivePortraitDriver.apply_audio_keypoints()
        """
        if not features.get("is_voiced", False):
            return {"energy": 0.0, "pitch": 0.0}

        return {
            "energy": features["energy"],
            "pitch": features["pitch"],
            "spectral_centroid": features.get("spectral_centroid", 0.0),
        }

    def process_audio_chunk(self, audio_data: np.ndarray) -> dict:
        """
        Full pipeline: audio chunk → features → keypoint-ready parameters.
        """
        features = self.extract_features(audio_data)
        return self.features_to_keypoint_delta(features)

    def reset(self) -> None:
        """Reset state for new conversation turn."""
        self._energy_history.clear()
        self._pitch_history.clear()
