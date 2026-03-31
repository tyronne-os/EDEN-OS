"""
EDEN OS — Scholar Engine: Knowledge Graph
Lightweight knowledge graph using regex + keyword extraction.
Stores entities and relationships extracted from ingested content.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from loguru import logger

from eden_os.shared.types import KnowledgeChunk


@dataclass
class Entity:
    """A named entity in the knowledge graph."""
    name: str
    entity_type: str  # "person", "product", "concept", "organization", "technology"
    mentions: int = 0
    sources: list[str] = field(default_factory=list)
    context_snippets: list[str] = field(default_factory=list)


@dataclass
class Relationship:
    """A relationship between two entities."""
    source: str
    target: str
    relation_type: str  # "mentions", "uses", "related_to", "created_by", "part_of"
    weight: float = 1.0


class KnowledgeGraph:
    """
    Lightweight knowledge graph for extracted entities and relationships.
    Uses regex and keyword patterns for entity extraction.
    """

    # Patterns for entity extraction
    PERSON_INDICATORS = {
        "dr.", "dr ", "professor", "prof.", "prof ", "ceo", "founder",
        "author", "researcher", "scientist", "engineer", "mr.", "mrs.",
        "ms.", "said", "argues", "explains", "wrote", "discovered",
    }

    PRODUCT_INDICATORS = {
        "app", "platform", "software", "tool", "framework", "library",
        "product", "service", "api", "model", "system", "engine",
        "version", "v1", "v2", "release",
    }

    CONCEPT_INDICATORS = {
        "theory", "method", "approach", "technique", "algorithm",
        "protocol", "principle", "paradigm", "architecture", "pattern",
        "strategy", "process", "mechanism", "framework",
    }

    # Common stopwords to filter out
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "this", "that", "these", "those", "i", "you", "he", "she",
        "it", "we", "they", "me", "him", "her", "us", "them", "my",
        "your", "his", "its", "our", "their", "what", "which", "who",
        "whom", "when", "where", "why", "how", "all", "each", "every",
        "both", "few", "more", "most", "other", "some", "such", "no",
        "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "as", "until", "while", "of", "at", "by",
        "for", "with", "about", "against", "between", "through",
        "during", "before", "after", "above", "below", "to", "from",
        "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "and",
        "but", "or", "nor", "if", "also", "into", "however", "new",
    }

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []
        self._adjacency: dict[str, set[str]] = defaultdict(set)

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for consistent lookup."""
        return name.strip().lower()

    def _extract_capitalized_phrases(self, text: str) -> list[str]:
        """Extract multi-word capitalized phrases (likely proper nouns)."""
        # Match sequences of capitalized words (2+ words)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        matches = re.findall(pattern, text)
        return matches

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract significant keywords from text."""
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        # Filter stopwords and very common words
        keywords = [w for w in words if w not in self.STOPWORDS]
        # Count frequency
        freq: dict[str, int] = defaultdict(int)
        for w in keywords:
            freq[w] += 1
        # Return top keywords by frequency
        sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in sorted_kw[:20]]

    def _classify_entity(self, name: str, context: str) -> str:
        """Classify an entity based on surrounding context."""
        lower_context = context.lower()
        lower_name = name.lower()

        for indicator in self.PERSON_INDICATORS:
            if indicator in lower_context and lower_name in lower_context:
                return "person"

        for indicator in self.PRODUCT_INDICATORS:
            if indicator in lower_context and lower_name in lower_context:
                return "product"

        for indicator in self.CONCEPT_INDICATORS:
            if indicator in lower_context and lower_name in lower_context:
                return "concept"

        # Default: if it's a capitalized multi-word phrase, likely a person or org
        if len(name.split()) >= 2:
            return "person"
        return "concept"

    def add_entities(self, chunks: list[KnowledgeChunk]) -> int:
        """
        Extract and add entities from knowledge chunks.
        Returns count of new entities added.
        """
        new_count = 0

        for chunk in chunks:
            text = chunk.text
            source = chunk.source_id

            # Extract capitalized phrases as entity candidates
            phrases = self._extract_capitalized_phrases(text)

            for phrase in phrases:
                key = self._normalize_name(phrase)
                entity_type = self._classify_entity(phrase, text)

                if key in self._entities:
                    self._entities[key].mentions += 1
                    if source not in self._entities[key].sources:
                        self._entities[key].sources.append(source)
                    # Keep up to 5 context snippets
                    if len(self._entities[key].context_snippets) < 5:
                        snippet = text[:200]
                        self._entities[key].context_snippets.append(snippet)
                else:
                    self._entities[key] = Entity(
                        name=phrase,
                        entity_type=entity_type,
                        mentions=1,
                        sources=[source],
                        context_snippets=[text[:200]],
                    )
                    new_count += 1

            # Extract keyword-based concepts
            keywords = self._extract_keywords(text)
            for kw in keywords[:5]:  # Top 5 keywords per chunk
                key = self._normalize_name(kw)
                if key not in self._entities and len(kw) > 4:
                    self._entities[key] = Entity(
                        name=kw,
                        entity_type="concept",
                        mentions=1,
                        sources=[source],
                        context_snippets=[text[:200]],
                    )
                    new_count += 1
                elif key in self._entities:
                    self._entities[key].mentions += 1

            # Build relationships between co-occurring entities in same chunk
            chunk_entities = []
            for phrase in phrases:
                chunk_entities.append(self._normalize_name(phrase))

            for i, e1 in enumerate(chunk_entities):
                for e2 in chunk_entities[i + 1:]:
                    if e1 != e2:
                        self._add_relationship(e1, e2, "related_to")

        logger.info(
            f"Knowledge graph updated: {new_count} new entities, "
            f"total {self.entity_count} entities, {self.relationship_count} relationships"
        )
        return new_count

    def _add_relationship(
        self, source: str, target: str, relation_type: str
    ) -> None:
        """Add or strengthen a relationship between two entities."""
        # Check if relationship already exists
        for rel in self._relationships:
            if (
                (rel.source == source and rel.target == target)
                or (rel.source == target and rel.target == source)
            ) and rel.relation_type == relation_type:
                rel.weight += 1.0
                return

        self._relationships.append(
            Relationship(
                source=source,
                target=target,
                relation_type=relation_type,
            )
        )
        self._adjacency[source].add(target)
        self._adjacency[target].add(source)

    def query_related(
        self, query: str, max_depth: int = 2, max_results: int = 10
    ) -> list[dict]:
        """
        Find entities and relationships related to a query.
        Uses BFS traversal up to max_depth from matching entities.
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        # Find matching entities
        matches: list[str] = []
        for key, entity in self._entities.items():
            if query_lower in key or key in query_lower:
                matches.append(key)
            elif query_terms & set(key.split()):
                matches.append(key)

        if not matches:
            # Fuzzy: check if any query term is a substring of entity name
            for key in self._entities:
                for term in query_terms:
                    if len(term) > 3 and term in key:
                        matches.append(key)
                        break

        # BFS from matches
        visited: set[str] = set()
        results: list[dict] = []
        queue: list[tuple[str, int]] = [(m, 0) for m in matches]

        while queue and len(results) < max_results:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in self._entities:
                entity = self._entities[current]
                results.append(
                    {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "mentions": entity.mentions,
                        "sources": entity.sources,
                        "depth": depth,
                    }
                )

            if depth < max_depth:
                for neighbor in self._adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        return results

    def get_summary(self) -> dict:
        """Get a summary of the knowledge graph."""
        type_counts: dict[str, int] = defaultdict(int)
        for entity in self._entities.values():
            type_counts[entity.entity_type] += 1

        top_entities = sorted(
            self._entities.values(),
            key=lambda e: e.mentions,
            reverse=True,
        )[:10]

        return {
            "total_entities": self.entity_count,
            "total_relationships": self.relationship_count,
            "entity_types": dict(type_counts),
            "top_entities": [
                {"name": e.name, "type": e.entity_type, "mentions": e.mentions}
                for e in top_entities
            ],
        }
