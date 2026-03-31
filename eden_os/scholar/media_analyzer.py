"""
EDEN OS — Scholar Engine: Media Analyzer
Batch processing controller for all ingestion jobs.
Tracks jobs, processes pending, returns summaries.
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from eden_os.shared.types import KnowledgeChunk, IngestionResult, KnowledgeSummary


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionJob:
    """Tracks a single ingestion job."""
    job_id: str
    source_type: str  # "youtube", "audiobook", "url"
    source: str  # URL or file path
    status: JobStatus = JobStatus.PENDING
    chunks: list[KnowledgeChunk] = field(default_factory=list)
    result: IngestionResult | None = None
    error: str | None = None


class MediaAnalyzer:
    """
    Batch processing controller for media ingestion.
    Tracks all jobs and provides analyze_all() for batch processing.
    """

    def __init__(self):
        self._jobs: dict[str, IngestionJob] = {}
        self._completed_results: list[IngestionResult] = []

    def add_job(self, source_type: str, source: str) -> str:
        """Register a new ingestion job. Returns job_id."""
        job_id = uuid.uuid4().hex[:12]
        self._jobs[job_id] = IngestionJob(
            job_id=job_id,
            source_type=source_type,
            source=source,
        )
        logger.info(f"Registered ingestion job {job_id}: {source_type} -> {source}")
        return job_id

    def get_pending_jobs(self) -> list[IngestionJob]:
        """Get all pending (unprocessed) jobs."""
        return [
            job for job in self._jobs.values() if job.status == JobStatus.PENDING
        ]

    def mark_processing(self, job_id: str) -> None:
        """Mark a job as currently processing."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.PROCESSING

    def mark_completed(
        self, job_id: str, chunks: list[KnowledgeChunk], result: IngestionResult
    ) -> None:
        """Mark a job as completed with results."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.COMPLETED
            self._jobs[job_id].chunks = chunks
            self._jobs[job_id].result = result
            self._completed_results.append(result)

    def mark_failed(self, job_id: str, error: str) -> None:
        """Mark a job as failed."""
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.FAILED
            self._jobs[job_id].error = error
            self._completed_results.append(
                IngestionResult(
                    job_id=job_id,
                    source_type=self._jobs[job_id].source_type,
                    chunks_created=0,
                    status="failed",
                    error=error,
                )
            )

    async def analyze_all(
        self,
        youtube_ingestor,
        audiobook_ingestor,
        url_ingestor,
        rag_retriever,
        knowledge_graph,
    ) -> KnowledgeSummary:
        """
        Process all pending ingestion jobs.
        Routes each job to the appropriate ingestor,
        adds chunks to RAG store and knowledge graph.
        Returns KnowledgeSummary with totals.
        """
        pending = self.get_pending_jobs()
        if not pending:
            logger.info("No pending jobs to process")
            return self.get_summary(rag_retriever)

        logger.info(f"Processing {len(pending)} pending ingestion jobs...")

        total_new_chunks = 0
        source_counts: dict[str, int] = {}

        for job in pending:
            self.mark_processing(job.job_id)
            logger.info(f"Processing job {job.job_id}: {job.source_type} -> {job.source}")

            try:
                chunks: list[KnowledgeChunk] = []
                result: IngestionResult | None = None

                if job.source_type == "youtube":
                    chunks, result = await youtube_ingestor.ingest(job.source)
                elif job.source_type == "audiobook":
                    chunks, result = await audiobook_ingestor.ingest(job.source)
                elif job.source_type == "url":
                    chunks, result = await url_ingestor.ingest(job.source)
                else:
                    raise ValueError(f"Unknown source type: {job.source_type}")

                if chunks:
                    # Add to RAG store
                    rag_retriever.add_chunks(chunks)
                    # Add to knowledge graph
                    knowledge_graph.add_entities(chunks)

                    count = len(chunks)
                    total_new_chunks += count
                    source_counts[job.source_type] = (
                        source_counts.get(job.source_type, 0) + count
                    )

                self.mark_completed(job.job_id, chunks, result)
                logger.info(
                    f"Job {job.job_id} completed: {len(chunks)} chunks created"
                )

            except Exception as e:
                error_msg = f"Job {job.job_id} failed: {str(e)}"
                logger.error(error_msg)
                self.mark_failed(job.job_id, str(e))

        logger.info(
            f"Batch processing complete: {total_new_chunks} new chunks "
            f"across {len(pending)} jobs"
        )

        return self.get_summary(rag_retriever)

    def get_summary(self, rag_retriever=None) -> KnowledgeSummary:
        """Get summary of all ingested knowledge."""
        if rag_retriever:
            total = rag_retriever.get_total_chunks()
            sources = rag_retriever.get_source_counts()
        else:
            # Fall back to counting from completed jobs
            total = sum(
                len(j.chunks) for j in self._jobs.values()
                if j.status == JobStatus.COMPLETED
            )
            sources: dict[str, int] = {}
            for j in self._jobs.values():
                if j.status == JobStatus.COMPLETED:
                    sources[j.source_type] = (
                        sources.get(j.source_type, 0) + len(j.chunks)
                    )

        pending_count = len(self.get_pending_jobs())
        status = "ready" if pending_count == 0 else f"processing ({pending_count} pending)"

        return KnowledgeSummary(
            total_chunks=total,
            sources=sources,
            status=status,
        )

    def get_job_status(self, job_id: str) -> dict | None:
        """Get status of a specific job."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "source_type": job.source_type,
            "source": job.source,
            "status": job.status.value,
            "chunks_created": len(job.chunks),
            "error": job.error,
        }
