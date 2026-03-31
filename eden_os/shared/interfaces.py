"""
EDEN OS — Engine Interfaces
Abstract base classes for all 7 engines.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from eden_os.shared.types import (
    AudioChunk,
    AvatarState,
    EdenValidationResult,
    IngestionResult,
    KnowledgeChunk,
    KnowledgeSummary,
    PipelineMetrics,
    SessionConfig,
    TextChunk,
    VideoFrame,
)

import numpy as np


class IGenesisEngine(ABC):
    """Agent 1: Portrait-to-4D Engine."""

    @abstractmethod
    async def process_upload(self, image: np.ndarray) -> dict:
        """Process uploaded portrait: face detection, alignment, enhancement."""
        ...

    @abstractmethod
    async def validate_eden_protocol(
        self, generated: np.ndarray, reference: np.ndarray, threshold: float = 0.3
    ) -> EdenValidationResult:
        """Validate skin texture fidelity against reference."""
        ...

    @abstractmethod
    async def encode_to_latent(self, portrait: np.ndarray) -> np.ndarray:
        """Encode portrait to animation-engine-compatible latent."""
        ...

    @abstractmethod
    async def precompute_idle_cache(self, profile: dict) -> dict:
        """Pre-compute idle animation seed frames."""
        ...


class IVoiceEngine(ABC):
    """Agent 2: TTS + ASR + Voice Cloning."""

    @abstractmethod
    async def transcribe_stream(
        self, audio: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[str]:
        """Real-time speech-to-text from streaming audio."""
        ...

    @abstractmethod
    async def synthesize_stream(
        self, text: AsyncIterator[TextChunk]
    ) -> AsyncIterator[AudioChunk]:
        """Stream TTS audio from text chunks."""
        ...

    @abstractmethod
    async def detect_interruption(self, audio: AudioChunk) -> bool:
        """Detect if user has started speaking (interrupt)."""
        ...

    @abstractmethod
    async def clone_voice(self, reference_audio: np.ndarray) -> str:
        """Clone voice from reference audio, return voice_id."""
        ...


class IAnimatorEngine(ABC):
    """Agent 3: Lip-Sync + 4D Motion Engine."""

    @abstractmethod
    async def start_idle_loop(self, profile: dict) -> AsyncIterator[VideoFrame]:
        """Generate continuous idle animation (blinks, breathing, micro-movements)."""
        ...

    @abstractmethod
    async def drive_from_audio(
        self, audio: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[VideoFrame]:
        """Generate lip-synced animation driven by audio stream."""
        ...

    @abstractmethod
    async def transition_state(
        self, from_state: AvatarState, to_state: AvatarState
    ) -> None:
        """Smoothly transition between avatar states."""
        ...

    @abstractmethod
    async def get_current_frame(self) -> VideoFrame:
        """Get the most recent rendered frame."""
        ...


class IBrainEngine(ABC):
    """Agent 4: LLM Reasoning + Context Engine."""

    @abstractmethod
    async def reason_stream(
        self, user_input: str, context: dict
    ) -> AsyncIterator[TextChunk]:
        """Stream LLM response tokens as TextChunks."""
        ...

    @abstractmethod
    async def load_persona(self, template_path: str) -> None:
        """Load agent persona from YAML template."""
        ...

    @abstractmethod
    async def get_context(self) -> dict:
        """Get current conversation context and memory."""
        ...

    @abstractmethod
    async def process_user_input(self, text: str) -> None:
        """Process and store user input in conversation history."""
        ...


class IConductor(ABC):
    """Agent 5: Pipeline Orchestrator."""

    @abstractmethod
    async def create_session(self, config: SessionConfig) -> str:
        """Create a new conversation session, return session_id."""
        ...

    @abstractmethod
    async def start_conversation(self, session_id: str) -> None:
        """Begin the conversation loop for a session."""
        ...

    @abstractmethod
    async def end_conversation(self, session_id: str) -> None:
        """End session, cleanup resources."""
        ...

    @abstractmethod
    async def get_metrics(self, session_id: str) -> PipelineMetrics:
        """Get real-time pipeline metrics."""
        ...


class IGatewayServer(ABC):
    """Agent 6: WebRTC Server + API Layer."""

    @abstractmethod
    async def start(self, host: str, port: int) -> None:
        """Boot the API + WebRTC server."""
        ...


class IScholarEngine(ABC):
    """Agent 7: Knowledge Engine + RAG."""

    @abstractmethod
    async def ingest_youtube(self, url: str) -> IngestionResult:
        """Ingest YouTube video: download, transcribe, embed."""
        ...

    @abstractmethod
    async def ingest_audiobook(self, file_path: str) -> IngestionResult:
        """Ingest audiobook: transcribe, chunk, embed."""
        ...

    @abstractmethod
    async def ingest_url(self, url: str) -> IngestionResult:
        """Ingest web URL or PDF: extract, chunk, embed."""
        ...

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[KnowledgeChunk]:
        """Retrieve relevant knowledge chunks for a query."""
        ...

    @abstractmethod
    async def analyze_all(self) -> KnowledgeSummary:
        """Batch process all pending ingestion jobs."""
        ...

    @abstractmethod
    async def get_knowledge_summary(self) -> KnowledgeSummary:
        """Get summary of all ingested knowledge."""
        ...
