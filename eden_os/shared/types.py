"""
EDEN OS — Shared Types
Dataclasses and enums used across all 7 engines.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AvatarState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class AudioChunk:
    """A chunk of PCM audio data."""
    data: np.ndarray          # PCM audio samples
    sample_rate: int          # 16000 or 22050
    duration_ms: float
    is_final: bool = False    # True if this is the last chunk


@dataclass
class VideoFrame:
    """A single rendered avatar frame."""
    pixels: np.ndarray        # RGB frame (H, W, 3)
    timestamp_ms: float
    state: AvatarState
    eden_score: float = 1.0   # Eden Protocol validation score (0.0–1.0)


@dataclass
class TextChunk:
    """A chunk of LLM-generated text."""
    text: str
    is_sentence_end: bool = False
    emotion: dict = field(default_factory=lambda: {
        "joy": 0.5, "sadness": 0.0, "confidence": 0.7,
        "urgency": 0.0, "warmth": 0.6,
    })


@dataclass
class EdenValidationResult:
    """Result of Eden Protocol skin-texture validation."""
    passed: bool
    score: float              # Deviation from reference (lower = better)
    feedback: str


@dataclass
class KnowledgeChunk:
    """A chunk of ingested knowledge with source attribution."""
    text: str
    source_type: str          # "youtube", "audiobook", "url"
    source_id: str            # URL or file path
    timestamp: Optional[str] = None   # e.g., "3:42" for YouTube
    chapter: Optional[str] = None     # For audiobooks
    metadata: dict = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of a Scholar media ingestion job."""
    job_id: str
    source_type: str
    chunks_created: int
    status: str = "completed"
    error: Optional[str] = None


@dataclass
class KnowledgeSummary:
    """Summary of all ingested knowledge."""
    total_chunks: int
    sources: dict = field(default_factory=dict)  # {source_type: count}
    status: str = "ready"


@dataclass
class SessionConfig:
    """Configuration for a conversation session."""
    session_id: str
    portrait_image: Optional[np.ndarray] = None
    template_name: str = "default"
    hardware_profile: str = "auto"
    settings: dict = field(default_factory=lambda: {
        "consistency": 0.7,
        "latency": 1.0,
        "expressiveness": 0.6,
        "voice_tone": 0.85,
        "eye_contact": 0.5,
        "flirtation": 0.15,
    })


@dataclass
class PipelineMetrics:
    """Real-time performance metrics."""
    asr_latency_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    animation_fps: float = 0.0
    total_latency_ms: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
