"""
EDEN OS — Scholar Engine: YouTube Ingestor
Downloads audio from YouTube via yt-dlp, transcribes via Whisper,
chunks transcript by sentences with timestamps.
"""

import os
import re
import tempfile
import uuid
from pathlib import Path

from loguru import logger

from eden_os.shared.types import KnowledgeChunk, IngestionResult


class YouTubeIngestor:
    """Ingests YouTube videos: download audio, transcribe, chunk."""

    def __init__(self, download_dir: str | None = None):
        self._download_dir = download_dir or tempfile.mkdtemp(prefix="eden_yt_")
        Path(self._download_dir).mkdir(parents=True, exist_ok=True)
        self._whisper_model = None

    def _get_whisper_model(self):
        """Lazy-load whisper model."""
        if self._whisper_model is None:
            import whisper
            logger.info("Loading Whisper model (base) for YouTube transcription...")
            self._whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded.")
        return self._whisper_model

    def _validate_youtube_url(self, url: str) -> bool:
        """Check if URL looks like a valid YouTube URL."""
        patterns = [
            r"(https?://)?(www\.)?youtube\.com/watch\?v=[\w-]+",
            r"(https?://)?(www\.)?youtu\.be/[\w-]+",
            r"(https?://)?(www\.)?youtube\.com/shorts/[\w-]+",
        ]
        return any(re.match(p, url) for p in patterns)

    def _download_audio(self, url: str) -> str:
        """Download audio from YouTube URL using yt-dlp. Returns path to audio file."""
        import yt_dlp

        audio_id = uuid.uuid4().hex[:8]
        output_path = os.path.join(self._download_dir, f"yt_{audio_id}.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        logger.info(f"Downloading audio from: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Unknown")
            logger.info(f"Downloaded: {title}")

        # Find the downloaded wav file
        wav_path = os.path.join(self._download_dir, f"yt_{audio_id}.wav")
        if not os.path.exists(wav_path):
            # Try to find any file with our ID
            for f in os.listdir(self._download_dir):
                if f.startswith(f"yt_{audio_id}"):
                    wav_path = os.path.join(self._download_dir, f)
                    break

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Downloaded audio not found at {wav_path}")

        return wav_path

    def _transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file using Whisper. Returns segments with timestamps."""
        model = self._get_whisper_model()
        logger.info(f"Transcribing: {audio_path}")
        result = model.transcribe(audio_path, verbose=False)
        logger.info(f"Transcription complete: {len(result.get('segments', []))} segments")
        return result

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to MM:SS or HH:MM:SS format."""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def _chunk_segments(self, segments: list[dict], url: str) -> list[KnowledgeChunk]:
        """Group whisper segments into sentence-based chunks with timestamps."""
        chunks: list[KnowledgeChunk] = []
        current_text = ""
        current_start = 0.0

        for seg in segments:
            text = seg.get("text", "").strip()
            start = seg.get("start", 0.0)

            if not current_text:
                current_start = start

            current_text += " " + text

            # Split on sentence boundaries
            sentence_ends = re.findall(r"[.!?]\s", current_text)
            if len(sentence_ends) >= 2 or len(current_text) > 500:
                chunk = KnowledgeChunk(
                    text=current_text.strip(),
                    source_type="youtube",
                    source_id=url,
                    timestamp=self._format_timestamp(current_start),
                    metadata={
                        "start_seconds": current_start,
                        "end_seconds": seg.get("end", start),
                    },
                )
                chunks.append(chunk)
                current_text = ""

        # Flush remaining text
        if current_text.strip():
            chunks.append(
                KnowledgeChunk(
                    text=current_text.strip(),
                    source_type="youtube",
                    source_id=url,
                    timestamp=self._format_timestamp(current_start),
                    metadata={"start_seconds": current_start},
                )
            )

        return chunks

    async def ingest(self, url: str) -> tuple[list[KnowledgeChunk], IngestionResult]:
        """
        Full YouTube ingestion pipeline:
        1. Validate URL
        2. Download audio via yt-dlp
        3. Transcribe via Whisper
        4. Chunk by sentences with timestamps
        Returns (chunks, result).
        """
        job_id = uuid.uuid4().hex[:12]
        logger.info(f"[{job_id}] Starting YouTube ingestion: {url}")

        if not self._validate_youtube_url(url):
            logger.warning(f"[{job_id}] Invalid YouTube URL: {url}")
            return [], IngestionResult(
                job_id=job_id,
                source_type="youtube",
                chunks_created=0,
                status="failed",
                error=f"Invalid YouTube URL: {url}",
            )

        try:
            audio_path = self._download_audio(url)
        except Exception as e:
            logger.error(f"[{job_id}] Download failed: {e}")
            return [], IngestionResult(
                job_id=job_id,
                source_type="youtube",
                chunks_created=0,
                status="failed",
                error=f"Download failed: {str(e)}",
            )

        try:
            result = self._transcribe(audio_path)
        except Exception as e:
            logger.error(f"[{job_id}] Transcription failed: {e}")
            return [], IngestionResult(
                job_id=job_id,
                source_type="youtube",
                chunks_created=0,
                status="failed",
                error=f"Transcription failed: {str(e)}",
            )
        finally:
            # Clean up downloaded audio
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except OSError:
                pass

        segments = result.get("segments", [])
        if not segments:
            logger.warning(f"[{job_id}] No segments found in transcription")
            return [], IngestionResult(
                job_id=job_id,
                source_type="youtube",
                chunks_created=0,
                status="completed",
                error="No speech detected in video",
            )

        chunks = self._chunk_segments(segments, url)
        logger.info(f"[{job_id}] Created {len(chunks)} knowledge chunks from YouTube")

        return chunks, IngestionResult(
            job_id=job_id,
            source_type="youtube",
            chunks_created=len(chunks),
            status="completed",
        )
