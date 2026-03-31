"""
EDEN OS — Scholar Engine: Audiobook Ingestor
Processes MP3/WAV/M4A files via Whisper with semantic chunking
by topic boundaries (long pauses and topic shifts).
"""

import os
import uuid
from pathlib import Path

from loguru import logger

from eden_os.shared.types import KnowledgeChunk, IngestionResult


class AudiobookIngestor:
    """Ingests audiobook/media files: transcribe in segments, semantic chunking."""

    # Supported audio formats
    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".wma"}

    # Segment duration in seconds for processing long audio
    SEGMENT_DURATION = 600  # 10 minutes per segment

    # Pause threshold in seconds to detect topic boundaries
    PAUSE_THRESHOLD = 2.0

    def __init__(self):
        self._whisper_model = None

    def _get_whisper_model(self):
        """Lazy-load whisper model."""
        if self._whisper_model is None:
            import whisper
            logger.info("Loading Whisper model (base) for audiobook transcription...")
            self._whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded.")
        return self._whisper_model

    def _validate_file(self, file_path: str) -> str | None:
        """Validate file exists and is a supported format. Returns error string or None."""
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        ext = Path(file_path).suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            return f"Unsupported format '{ext}'. Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"

        return None

    def _get_audio_duration(self, file_path: str) -> float:
        """Get audio duration in seconds using basic file inspection."""
        try:
            import wave

            if file_path.lower().endswith(".wav"):
                with wave.open(file_path, "r") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return frames / float(rate)
        except Exception:
            pass

        # Fallback: use whisper's audio loading to estimate
        try:
            import whisper
            audio = whisper.load_audio(file_path)
            return len(audio) / 16000.0  # whisper resamples to 16kHz
        except Exception:
            return 0.0

    def _transcribe_full(self, file_path: str) -> dict:
        """Transcribe audio file using Whisper. Handles long files via internal segmenting."""
        model = self._get_whisper_model()
        logger.info(f"Transcribing audiobook: {file_path}")

        # Whisper handles long audio internally with 30-second windows
        result = model.transcribe(
            file_path,
            verbose=False,
            condition_on_previous_text=True,
        )
        seg_count = len(result.get("segments", []))
        logger.info(f"Transcription complete: {seg_count} segments")
        return result

    def _detect_topic_boundaries(self, segments: list[dict]) -> list[int]:
        """
        Detect topic boundaries by finding long pauses between segments
        and significant vocabulary shifts.
        Returns list of segment indices where topic boundaries occur.
        """
        boundaries: list[int] = []

        for i in range(1, len(segments)):
            prev_end = segments[i - 1].get("end", 0.0)
            curr_start = segments[i].get("start", 0.0)
            gap = curr_start - prev_end

            # Long pause indicates topic boundary
            if gap >= self.PAUSE_THRESHOLD:
                boundaries.append(i)
                continue

            # Check for topic shift via keyword change
            prev_words = set(segments[i - 1].get("text", "").lower().split())
            curr_words = set(segments[i].get("text", "").lower().split())

            if prev_words and curr_words:
                # Low overlap suggests topic shift
                overlap = len(prev_words & curr_words)
                total = max(len(prev_words | curr_words), 1)
                similarity = overlap / total

                if similarity < 0.05 and len(curr_words) > 3:
                    boundaries.append(i)

        return boundaries

    def _semantic_chunk(
        self, segments: list[dict], file_path: str
    ) -> list[KnowledgeChunk]:
        """
        Group segments into semantic chunks based on topic boundaries.
        Falls back to size-based chunking if no boundaries detected.
        """
        if not segments:
            return []

        boundaries = self._detect_topic_boundaries(segments)
        logger.info(f"Detected {len(boundaries)} topic boundaries")

        # Add start and end boundaries
        split_points = [0] + boundaries + [len(segments)]

        chunks: list[KnowledgeChunk] = []
        chapter_num = 0

        for i in range(len(split_points) - 1):
            start_idx = split_points[i]
            end_idx = split_points[i + 1]

            group = segments[start_idx:end_idx]
            if not group:
                continue

            text = " ".join(seg.get("text", "").strip() for seg in group).strip()
            if not text:
                continue

            # If a chunk is too long (>1000 chars), split it further
            if len(text) > 1000:
                sub_chunks = self._split_long_chunk(text, group, file_path, chapter_num)
                chunks.extend(sub_chunks)
                chapter_num += len(sub_chunks)
            else:
                chapter_num += 1
                start_sec = group[0].get("start", 0.0)
                end_sec = group[-1].get("end", 0.0)

                chunks.append(
                    KnowledgeChunk(
                        text=text,
                        source_type="audiobook",
                        source_id=file_path,
                        timestamp=self._format_timestamp(start_sec),
                        chapter=f"Section {chapter_num}",
                        metadata={
                            "start_seconds": start_sec,
                            "end_seconds": end_sec,
                            "segment_count": len(group),
                        },
                    )
                )

        return chunks

    def _split_long_chunk(
        self, text: str, segments: list[dict], file_path: str, base_chapter: int
    ) -> list[KnowledgeChunk]:
        """Split a long text chunk into smaller pieces at sentence boundaries."""
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[KnowledgeChunk] = []
        current = ""
        sub_idx = 0

        # Estimate time per character for timestamp approximation
        total_chars = max(len(text), 1)
        start_sec = segments[0].get("start", 0.0) if segments else 0.0
        end_sec = segments[-1].get("end", 0.0) if segments else 0.0
        duration = end_sec - start_sec

        for sentence in sentences:
            current += " " + sentence
            if len(current) >= 400:
                sub_idx += 1
                char_ratio = max(len(current.strip()), 1) / total_chars
                chunk_start = start_sec + (duration * (1.0 - char_ratio))

                chunks.append(
                    KnowledgeChunk(
                        text=current.strip(),
                        source_type="audiobook",
                        source_id=file_path,
                        timestamp=self._format_timestamp(chunk_start),
                        chapter=f"Section {base_chapter + sub_idx}",
                        metadata={
                            "start_seconds": chunk_start,
                            "estimated_timestamp": True,
                        },
                    )
                )
                current = ""

        if current.strip():
            sub_idx += 1
            chunks.append(
                KnowledgeChunk(
                    text=current.strip(),
                    source_type="audiobook",
                    source_id=file_path,
                    timestamp=self._format_timestamp(end_sec),
                    chapter=f"Section {base_chapter + sub_idx}",
                    metadata={"start_seconds": end_sec, "estimated_timestamp": True},
                )
            )

        return chunks

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to HH:MM:SS or MM:SS format."""
        total = int(seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    async def ingest(
        self, file_path: str
    ) -> tuple[list[KnowledgeChunk], IngestionResult]:
        """
        Full audiobook ingestion pipeline:
        1. Validate file
        2. Transcribe via Whisper
        3. Detect topic boundaries
        4. Semantic chunking
        Returns (chunks, result).
        """
        job_id = uuid.uuid4().hex[:12]
        logger.info(f"[{job_id}] Starting audiobook ingestion: {file_path}")

        error = self._validate_file(file_path)
        if error:
            logger.warning(f"[{job_id}] Validation failed: {error}")
            return [], IngestionResult(
                job_id=job_id,
                source_type="audiobook",
                chunks_created=0,
                status="failed",
                error=error,
            )

        try:
            result = self._transcribe_full(file_path)
        except Exception as e:
            logger.error(f"[{job_id}] Transcription failed: {e}")
            return [], IngestionResult(
                job_id=job_id,
                source_type="audiobook",
                chunks_created=0,
                status="failed",
                error=f"Transcription failed: {str(e)}",
            )

        segments = result.get("segments", [])
        if not segments:
            logger.warning(f"[{job_id}] No speech detected")
            return [], IngestionResult(
                job_id=job_id,
                source_type="audiobook",
                chunks_created=0,
                status="completed",
                error="No speech detected in audio file",
            )

        chunks = self._semantic_chunk(segments, file_path)
        logger.info(f"[{job_id}] Created {len(chunks)} knowledge chunks from audiobook")

        return chunks, IngestionResult(
            job_id=job_id,
            source_type="audiobook",
            chunks_created=len(chunks),
            status="completed",
        )
