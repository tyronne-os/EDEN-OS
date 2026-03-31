"""
EDEN OS — Scholar Engine: URL Ingestor
Ingests web pages and PDF research papers.
Routes to appropriate extractor, chunks text with overlap.
"""

import re
import uuid
from urllib.parse import urlparse

from loguru import logger

from eden_os.shared.types import KnowledgeChunk, IngestionResult


class URLIngestor:
    """Ingests URLs: web pages via trafilatura, PDFs via pymupdf."""

    # Target chunk size in characters (~500 tokens * ~4 chars/token)
    CHUNK_SIZE = 2000
    # Overlap between chunks in characters
    CHUNK_OVERLAP = 200

    def _is_pdf_url(self, url: str) -> bool:
        """Determine if URL points to a PDF."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        if path.endswith(".pdf"):
            return True
        # arXiv PDFs
        if "arxiv.org" in parsed.netloc and "/pdf/" in path:
            return True
        return False

    def _extract_pdf_from_url(self, url: str) -> str:
        """Download and extract text from a PDF URL."""
        import urllib.request
        import tempfile
        import os

        # Download PDF to temp file
        logger.info(f"Downloading PDF from: {url}")
        tmp_path = os.path.join(tempfile.mkdtemp(prefix="eden_pdf_"), "doc.pdf")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (EDEN-OS Scholar Engine)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(tmp_path, "wb") as f:
                f.write(response.read())

        text = self._extract_pdf_from_file(tmp_path)

        # Clean up
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        return text

    @staticmethod
    def _extract_pdf_from_file(file_path: str) -> str:
        """Extract text from a local PDF file using pymupdf."""
        import pymupdf

        logger.info(f"Extracting text from PDF: {file_path}")
        doc = pymupdf.open(file_path)
        pages: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append(text.strip())

        doc.close()
        full_text = "\n\n".join(pages)
        logger.info(f"Extracted {len(full_text)} characters from {len(pages)} PDF pages")
        return full_text

    @staticmethod
    def _extract_web_page(url: str) -> str:
        """Extract clean text from a web page using trafilatura."""
        import trafilatura

        logger.info(f"Fetching web page: {url}")
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise ValueError(f"Failed to fetch URL: {url}")

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )

        if not text:
            raise ValueError(f"No extractable text content at: {url}")

        logger.info(f"Extracted {len(text)} characters from web page")
        return text

    def _chunk_text(self, text: str, url: str) -> list[KnowledgeChunk]:
        """
        Split text into ~500-token chunks with overlap.
        Splits at sentence boundaries when possible.
        """
        if not text.strip():
            return []

        # Clean text: normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        chunks: list[KnowledgeChunk] = []
        pos = 0
        text_len = len(text)

        while pos < text_len:
            end = min(pos + self.CHUNK_SIZE, text_len)

            # Try to find a sentence boundary near the end of the chunk
            if end < text_len:
                # Look for sentence-ending punctuation followed by space
                search_start = max(end - 300, pos)
                boundary_region = text[search_start:end]

                # Find the last sentence boundary in the region
                best_break = -1
                for match in re.finditer(r"[.!?]\s", boundary_region):
                    best_break = search_start + match.end()

                if best_break > pos:
                    end = best_break

            chunk_text = text[pos:end].strip()
            if chunk_text:
                chunks.append(
                    KnowledgeChunk(
                        text=chunk_text,
                        source_type="url",
                        source_id=url,
                        metadata={
                            "char_start": pos,
                            "char_end": end,
                            "chunk_index": len(chunks),
                        },
                    )
                )

            # Advance position with overlap
            pos = max(end - self.CHUNK_OVERLAP, pos + 1)
            if end >= text_len:
                break

        return chunks

    async def ingest(self, url: str) -> tuple[list[KnowledgeChunk], IngestionResult]:
        """
        Full URL ingestion pipeline:
        1. Detect content type (PDF vs web page)
        2. Extract text using appropriate tool
        3. Chunk with overlap
        Returns (chunks, result).
        """
        job_id = uuid.uuid4().hex[:12]
        logger.info(f"[{job_id}] Starting URL ingestion: {url}")

        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.warning(f"[{job_id}] Invalid URL: {url}")
            return [], IngestionResult(
                job_id=job_id,
                source_type="url",
                chunks_created=0,
                status="failed",
                error=f"Invalid URL: {url}",
            )

        try:
            if self._is_pdf_url(url):
                text = self._extract_pdf_from_url(url)
            else:
                text = self._extract_web_page(url)
        except Exception as e:
            logger.error(f"[{job_id}] Extraction failed: {e}")
            return [], IngestionResult(
                job_id=job_id,
                source_type="url",
                chunks_created=0,
                status="failed",
                error=f"Extraction failed: {str(e)}",
            )

        if not text.strip():
            return [], IngestionResult(
                job_id=job_id,
                source_type="url",
                chunks_created=0,
                status="completed",
                error="No text content extracted",
            )

        chunks = self._chunk_text(text, url)
        logger.info(f"[{job_id}] Created {len(chunks)} knowledge chunks from URL")

        return chunks, IngestionResult(
            job_id=job_id,
            source_type="url",
            chunks_created=len(chunks),
            status="completed",
        )
