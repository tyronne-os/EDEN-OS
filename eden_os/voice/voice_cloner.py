"""
EDEN OS -- Voice Engine: Voice Cloner
Accepts reference audio, extracts mel-spectrogram features as voice embedding,
stores embeddings by voice_id, and returns voice_id string.
"""

import hashlib
import time
from typing import Dict, List, Optional

import numpy as np
from loguru import logger


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a Mel filterbank matrix (numpy-only, no librosa dependency)."""
    def _hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def _mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sr / 2.0)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, centre):
            if centre != left:
                fb[i, j] = (j - left) / (centre - left)
        for j in range(centre, right):
            if right != centre:
                fb[i, j] = (right - j) / (right - centre)
    return fb


def _extract_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> np.ndarray:
    """Compute log-mel spectrogram from raw audio using numpy FFT."""
    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 2.0:
        audio = audio / 32768.0

    # Pad to complete last frame
    pad_len = n_fft - (len(audio) % hop_length)
    audio = np.pad(audio, (0, pad_len))

    # STFT via numpy
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    window = np.hanning(n_fft).astype(np.float32)
    frames = np.stack(
        [audio[i * hop_length : i * hop_length + n_fft] * window for i in range(num_frames)]
    )
    spectrum = np.fft.rfft(frames, n=n_fft)
    power = np.abs(spectrum) ** 2

    # Apply mel filterbank
    fb = _mel_filterbank(sr, n_fft, n_mels)
    mel = power @ fb.T  # (frames, n_mels)

    # Log scale
    mel = np.log(np.maximum(mel, 1e-10))
    return mel  # (frames, n_mels)


class VoiceCloner:
    """Extracts and stores voice embeddings from reference audio.

    Embeddings are mean-pooled log-mel spectrogram feature vectors.
    """

    def __init__(self, n_mels: int = 80) -> None:
        self._n_mels = n_mels
        self._embeddings: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, dict] = {}
        logger.info("VoiceCloner initialised  n_mels={}", n_mels)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        reference_audio: np.ndarray,
        sample_rate: int = 16000,
        voice_id: Optional[str] = None,
    ) -> str:
        """Extract voice embedding from reference audio and store it.

        Args:
            reference_audio: Raw PCM samples (float32 or int16).
            sample_rate: Sample rate of the reference audio.
            voice_id: Optional custom id. Auto-generated if None.

        Returns:
            voice_id string used to recall this voice.
        """
        if reference_audio.size == 0:
            raise ValueError("Reference audio is empty")

        mel = _extract_mel_spectrogram(
            reference_audio, sr=sample_rate, n_mels=self._n_mels
        )
        # Mean-pool across time -> fixed-size embedding
        embedding = mel.mean(axis=0)  # (n_mels,)
        # L2 normalise
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if voice_id is None:
            # Deterministic id from audio content
            audio_hash = hashlib.sha256(reference_audio.tobytes()[:4096]).hexdigest()[:12]
            voice_id = f"voice_{audio_hash}"

        self._embeddings[voice_id] = embedding
        self._metadata[voice_id] = {
            "sample_rate": sample_rate,
            "duration_s": len(reference_audio) / sample_rate,
            "created": time.time(),
        }

        logger.info(
            "Voice cloned  id={}  duration={:.1f}s  embedding_dim={}",
            voice_id,
            self._metadata[voice_id]["duration_s"],
            embedding.shape[0],
        )
        return voice_id

    def get_voice_embedding(self, voice_id: str) -> np.ndarray:
        """Retrieve a previously stored voice embedding.

        Raises KeyError if voice_id not found.
        """
        if voice_id not in self._embeddings:
            raise KeyError(f"Voice id '{voice_id}' not found")
        return self._embeddings[voice_id].copy()

    def list_voices(self) -> List[Dict[str, object]]:
        """List all stored voice profiles."""
        result = []
        for vid, meta in self._metadata.items():
            result.append({"voice_id": vid, **meta})
        return result

    def has_voice(self, voice_id: str) -> bool:
        return voice_id in self._embeddings

    def remove_voice(self, voice_id: str) -> None:
        self._embeddings.pop(voice_id, None)
        self._metadata.pop(voice_id, None)
        logger.info("Voice removed  id={}", voice_id)
