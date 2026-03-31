"""
EDEN OS — Scholar Engine (Agent 7)
Knowledge Engine + Media Ingestion + RAG Retrieval.

Composes all sub-modules:
- YouTubeIngestor: YouTube video transcription pipeline
- AudiobookIngestor: Audiobook/media file processing
- URLIngestor: Web page and PDF ingestion
- KnowledgeGraph: Lightweight entity/relationship graph
- RAGRetriever: ChromaDB-backed semantic retrieval
- MediaAnalyzer: Batch processing controller
"""

from eden_os.shared.interfaces import IScholarEngine
from eden_os.shared.types import IngestionResult, KnowledgeChunk, KnowledgeSummary

from loguru import logger

from eden_os.scholar.youtube_ingestor import YouTubeIngestor
from eden_os.scholar.audiobook_ingestor import AudiobookIngestor
from eden_os.scholar.url_ingestor import URLIngestor
from eden_os.scholar.knowledge_graph import KnowledgeGraph
from eden_os.scholar.rag_retriever import RAGRetriever
from eden_os.scholar.media_analyzer import MediaAnalyzer


class ScholarEngine(IScholarEngine):
    """
    Agent 7: Knowledge Engine + RAG.

    Composes all scholar sub-modules into a single engine that
    implements the IScholarEngine interface. Handles YouTube videos,
    audiobooks, web pages, PDFs, and provides RAG retrieval for
    the Brain engine.
    """

    def __init__(self, chromadb_path: str | None = None):
        """
        Initialize the Scholar Engine with all sub-modules.

        Args:
            chromadb_path: Path for ChromaDB persistent storage.
                           Defaults to ~/EDEN-OS/data/chromadb.
        """
        logger.info("Initializing Scholar Engine (Agent 7)...")

        self.youtube = YouTubeIngestor()
        self.audiobook = AudiobookIngestor()
        self.url = URLIngestor()
        self.knowledge_graph = KnowledgeGraph()
        self.rag = RAGRetriever(db_path=chromadb_path)
        self.analyzer = MediaAnalyzer()

        logger.info("Scholar Engine initialized.")

    async def ingest_youtube(self, url: str) -> IngestionResult:
        """Ingest YouTube video: download, transcribe, chunk, embed."""
        logger.info(f"Scholar: ingesting YouTube video: {url}")

        chunks, result = await self.youtube.ingest(url)

        if chunks:
            self.rag.add_chunks(chunks)
            self.knowledge_graph.add_entities(chunks)
            logger.info(
                f"YouTube ingestion complete: {len(chunks)} chunks embedded"
            )

        return result

    async def ingest_audiobook(self, file_path: str) -> IngestionResult:
        """Ingest audiobook: transcribe, chunk by topic, embed."""
        logger.info(f"Scholar: ingesting audiobook: {file_path}")

        chunks, result = await self.audiobook.ingest(file_path)

        if chunks:
            self.rag.add_chunks(chunks)
            self.knowledge_graph.add_entities(chunks)
            logger.info(
                f"Audiobook ingestion complete: {len(chunks)} chunks embedded"
            )

        return result

    async def ingest_url(self, url: str) -> IngestionResult:
        """Ingest web URL or PDF: extract, chunk, embed."""
        logger.info(f"Scholar: ingesting URL: {url}")

        chunks, result = await self.url.ingest(url)

        if chunks:
            self.rag.add_chunks(chunks)
            self.knowledge_graph.add_entities(chunks)
            logger.info(
                f"URL ingestion complete: {len(chunks)} chunks embedded"
            )

        return result

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[KnowledgeChunk]:
        """Retrieve relevant knowledge chunks for a query using hybrid search."""
        logger.debug(f"Scholar: retrieving for query: '{query[:80]}...'")
        return self.rag.retrieve(query, top_k=top_k)

    async def analyze_all(self) -> KnowledgeSummary:
        """Batch process all pending ingestion jobs."""
        logger.info("Scholar: running batch analysis on all pending jobs...")
        return await self.analyzer.analyze_all(
            youtube_ingestor=self.youtube,
            audiobook_ingestor=self.audiobook,
            url_ingestor=self.url,
            rag_retriever=self.rag,
            knowledge_graph=self.knowledge_graph,
        )

    async def get_knowledge_summary(self) -> KnowledgeSummary:
        """Get summary of all ingested knowledge."""
        summary = self.analyzer.get_summary(rag_retriever=self.rag)
        graph_summary = self.knowledge_graph.get_summary()
        summary.sources["_knowledge_graph"] = {
            "entities": graph_summary["total_entities"],
            "relationships": graph_summary["total_relationships"],
        }
        return summary

    def queue_job(self, source_type: str, source: str) -> str:
        """Queue a job for batch processing via analyze_all(). Returns job_id."""
        return self.analyzer.add_job(source_type, source)


__all__ = ["ScholarEngine"]
