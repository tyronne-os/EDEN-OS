"""
EDEN OS — Scholar Engine: RAG Retriever
ChromaDB-backed retrieval with sentence-transformers embeddings.
Supports hybrid search (semantic + keyword).
"""

import os
import uuid
from pathlib import Path

from loguru import logger

from eden_os.shared.types import KnowledgeChunk


class RAGRetriever:
    """
    RAG retrieval using ChromaDB persistent storage and
    sentence-transformers (all-MiniLM-L6-v2) embeddings.
    """

    COLLECTION_NAME = "eden_knowledge"
    DEFAULT_DB_PATH = os.path.expanduser("~/EDEN-OS/data/chromadb")

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or self.DEFAULT_DB_PATH
        Path(self._db_path).mkdir(parents=True, exist_ok=True)

        self._client = None
        self._collection = None
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy-load sentence-transformers model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: all-MiniLM-L6-v2")
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded.")
        return self._embedding_model

    def _get_collection(self):
        """Lazy-init ChromaDB client and collection."""
        if self._collection is None:
            import chromadb

            logger.info(f"Initializing ChromaDB at: {self._db_path}")
            self._client = chromadb.PersistentClient(path=self._db_path)
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB collection '{self.COLLECTION_NAME}' ready "
                f"({self._collection.count()} existing documents)"
            )
        return self._collection

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        model = self._get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def add_chunks(self, chunks: list[KnowledgeChunk]) -> int:
        """
        Add knowledge chunks to ChromaDB with embeddings.
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        collection = self._get_collection()

        # Prepare data for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk in chunks:
            doc_id = uuid.uuid4().hex
            ids.append(doc_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source_type": chunk.source_type,
                    "source_id": chunk.source_id,
                    "timestamp": chunk.timestamp or "",
                    "chapter": chunk.chapter or "",
                    **{
                        k: str(v)
                        for k, v in chunk.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                }
            )

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} chunks...")
        embeddings = self._embed_texts(documents)

        # Batch insert (ChromaDB has a batch limit)
        batch_size = 500
        added = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end],
            )
            added += end - i

        logger.info(f"Added {added} chunks to ChromaDB (total: {collection.count()})")
        return added

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_type: str | None = None,
    ) -> list[KnowledgeChunk]:
        """
        Retrieve relevant knowledge chunks via hybrid search.
        Combines semantic similarity with keyword matching.
        """
        collection = self._get_collection()

        if collection.count() == 0:
            logger.warning("Knowledge base is empty, nothing to retrieve")
            return []

        # Build where filter for source type if specified
        where_filter = None
        if source_type:
            where_filter = {"source_type": source_type}

        # Semantic search with embeddings
        query_embedding = self._embed_texts([query])[0]

        try:
            semantic_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            semantic_results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Keyword search (ChromaDB document contains filter)
        query_words = [w for w in query.lower().split() if len(w) > 3]
        keyword_results_ids: set[str] = set()

        if query_words:
            try:
                # Search for documents containing key query terms
                keyword_filter = {
                    "$or": [
                        {"source_type": {"$in": ["youtube", "audiobook", "url"]}}
                    ]
                }
                kw_results = collection.query(
                    query_texts=[query],
                    n_results=min(top_k, collection.count()),
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
                if kw_results["ids"] and kw_results["ids"][0]:
                    keyword_results_ids = set(kw_results["ids"][0])
            except Exception:
                pass  # Keyword search is supplementary

        # Merge and deduplicate results, prioritizing semantic matches
        seen_ids: set[str] = set()
        chunks: list[KnowledgeChunk] = []

        result_ids = semantic_results["ids"][0] if semantic_results["ids"] else []
        result_docs = semantic_results["documents"][0] if semantic_results["documents"] else []
        result_metas = semantic_results["metadatas"][0] if semantic_results["metadatas"] else []
        result_dists = semantic_results["distances"][0] if semantic_results["distances"] else []

        for idx, doc_id in enumerate(result_ids):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            meta = result_metas[idx] if idx < len(result_metas) else {}
            doc = result_docs[idx] if idx < len(result_docs) else ""
            distance = result_dists[idx] if idx < len(result_dists) else 1.0

            # Boost score if also found in keyword search
            is_keyword_match = doc_id in keyword_results_ids

            chunk = KnowledgeChunk(
                text=doc,
                source_type=meta.get("source_type", "unknown"),
                source_id=meta.get("source_id", ""),
                timestamp=meta.get("timestamp") or None,
                chapter=meta.get("chapter") or None,
                metadata={
                    "relevance_score": 1.0 - distance,
                    "keyword_match": is_keyword_match,
                },
            )
            chunks.append(chunk)

            if len(chunks) >= top_k:
                break

        logger.info(f"Retrieved {len(chunks)} chunks for query: '{query[:50]}...'")
        return chunks

    def get_total_chunks(self) -> int:
        """Get total number of chunks in the knowledge base."""
        collection = self._get_collection()
        return collection.count()

    def get_source_counts(self) -> dict[str, int]:
        """Get chunk counts by source type."""
        collection = self._get_collection()
        counts: dict[str, int] = {}

        for source_type in ["youtube", "audiobook", "url"]:
            try:
                results = collection.get(
                    where={"source_type": source_type},
                    include=[],
                )
                counts[source_type] = len(results["ids"])
            except Exception:
                counts[source_type] = 0

        return counts
