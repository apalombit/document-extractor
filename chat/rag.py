"""Simple in-memory RAG for document chunks."""

import re
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter
import numpy as np
from chat.context import DocumentContext
from config import CONFIG


class RAGStore:
    """Simple in-memory RAG store for document retrieval."""

    def __init__(self, context: DocumentContext, chunk_size: int = None):
        """
        Initialize RAG store with document context.

        Args:
            context: DocumentContext containing OCR text and extracted fields
            chunk_size: Characters per chunk (defaults to config value)
        """
        self.context = context
        self.chunk_size = chunk_size or CONFIG["chat"]["rag_chunk_size"]
        self.chunks: List[str] = []
        self.chunk_keywords: List[set] = []  # Keywords for each chunk

        # Build chunks on initialization
        self._build_chunks()

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization: lowercase, split on non-alphanumeric.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter out very short tokens (< 3 chars) and numbers
        return [t for t in tokens if len(t) >= 3 and not t.isdigit()]

    def _build_chunks(self):
        """Split document into retrievable chunks."""
        text = self.context.ocr_text

        if not text or not text.strip():
            return

        # Try to split by paragraphs first (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is small enough, use as-is
            if len(para) <= self.chunk_size:
                self.chunks.append(para)
                self.chunk_keywords.append(set(self._tokenize(para)))
            else:
                # Split large paragraphs by sentences or fixed size
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            self.chunks.append(current_chunk.strip())
                            self.chunk_keywords.append(set(self._tokenize(current_chunk)))
                        current_chunk = sentence

                # Don't forget the last chunk
                if current_chunk:
                    self.chunks.append(current_chunk.strip())
                    self.chunk_keywords.append(set(self._tokenize(current_chunk)))

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Find most relevant chunks for a query using keyword matching.

        Args:
            query: User query string
            top_k: Number of top chunks to return (defaults to config)

        Returns:
            List of (chunk_text, score) tuples, sorted by relevance
        """
        top_k = top_k or CONFIG["chat"]["rag_top_k"]

        if not self.chunks:
            return []

        query_tokens = set(self._tokenize(query))

        if not query_tokens:
            return []

        # Score each chunk by keyword overlap
        scored_chunks: List[Tuple[str, float]] = []

        for chunk, chunk_kw in zip(self.chunks, self.chunk_keywords):
            if not chunk_kw:
                continue

            # Calculate Jaccard-like similarity
            intersection = len(query_tokens & chunk_kw)
            if intersection == 0:
                continue

            # Score: intersection / min(query_len, chunk_len)
            # This favors chunks that contain more query terms
            score = intersection / min(len(query_tokens), len(chunk_kw))
            scored_chunks.append((chunk, score))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:top_k]

    def get_augmented_context(self, query: str, top_k: int = None) -> str:
        """
        Get relevant chunks formatted for LLM context.

        Args:
            query: User query string
            top_k: Number of chunks to include

        Returns:
            Formatted string with relevant document excerpts
        """
        retrieved = self.retrieve(query, top_k)

        if not retrieved:
            # Fall back to full context if no matches
            return self.context.to_context_string()

        sections = []
        sections.append("=== EXTRACTED INFORMATION ===")

        # Add extracted metadata
        author_date = self.context.extracted_fields.get("author_date", {})
        if author_date:
            authors = author_date.get("authors")
            date = author_date.get("date")
            if authors:
                sections.append(f"Authors: {', '.join(authors)}")
            if date:
                sections.append(f"Date: {date}")

        keywords_data = self.context.extracted_fields.get("keywords", {})
        if keywords_data:
            keywords = keywords_data.get("keywords")
            if keywords:
                sections.append(f"Keywords: {', '.join(keywords)}")

        doc_type_data = self.context.extracted_fields.get("document_type", {})
        if doc_type_data:
            doc_type = doc_type_data.get("document_type")
            if doc_type:
                sections.append(f"Document Type: {doc_type}")

        # Add relevant excerpts
        sections.append("\n=== RELEVANT EXCERPTS ===")
        for i, (chunk, score) in enumerate(retrieved, 1):
            sections.append(f"\n[Excerpt {i}]")
            sections.append(chunk)

        return "\n".join(sections)

    def get_chunk_count(self) -> int:
        """Return number of chunks created."""
        return len(self.chunks)

    def get_all_chunks(self) -> List[str]:
        """Return all chunks (for debugging/testing)."""
        return self.chunks.copy()


class EmbeddingRAGStore:
    """Embedding-based RAG using SentenceTransformer and cosine similarity."""

    def __init__(self, context: DocumentContext, model_name: str = None, chunk_size: int = None):
        """
        Initialize embedding-based RAG store.

        Args:
            context: DocumentContext containing OCR text and extracted fields
            model_name: SentenceTransformer model name (defaults to config)
            chunk_size: Characters per chunk (defaults to config)
        """
        self.context = context
        self.model_name = model_name or CONFIG["chat"]["rag_embedding_model"]
        self.chunk_size = chunk_size or CONFIG["chat"]["rag_chunk_size"]
        self.threshold = CONFIG["chat"]["rag_similarity_threshold"]

        # Lazy load model (expensive)
        self._model = None
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

        self._build_chunks()
        self._build_embeddings()

    @property
    def model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _build_chunks(self):
        """Split document into retrievable chunks."""
        text = self.context.ocr_text

        if not text or not text.strip():
            return

        # Split by paragraphs first (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is small enough, use as-is
            if len(para) <= self.chunk_size:
                self.chunks.append(para)
            else:
                # Split large paragraphs by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            self.chunks.append(current_chunk.strip())
                        current_chunk = sentence

                if current_chunk:
                    self.chunks.append(current_chunk.strip())

    def _build_embeddings(self):
        """Create embeddings for all chunks."""
        if self.chunks:
            self.embeddings = self.model.encode(self.chunks)

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Find most relevant chunks using cosine similarity.

        Args:
            query: User query string
            top_k: Number of top chunks to return (defaults to config)

        Returns:
            List of (chunk_text, similarity_score) tuples, sorted by relevance
        """
        top_k = top_k or CONFIG["chat"]["rag_top_k"]

        if not self.chunks or self.embeddings is None:
            return []

        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Sort by similarity descending
        sorted_indices = np.argsort(similarities)[::-1]

        # Filter by threshold and return top_k
        results = []
        for idx in sorted_indices[:top_k]:
            if similarities[idx] >= self.threshold:
                results.append((self.chunks[idx], float(similarities[idx])))

        return results

    def get_augmented_context(self, query: str, top_k: int = None) -> str:
        """
        Get relevant chunks formatted for LLM context.

        Args:
            query: User query string
            top_k: Number of chunks to include

        Returns:
            Formatted string with relevant document excerpts
        """
        retrieved = self.retrieve(query, top_k)

        if not retrieved:
            # Fall back to full context if no matches
            return self.context.to_context_string()

        sections = []
        sections.append("=== EXTRACTED INFORMATION ===")

        # Add extracted metadata
        author_date = self.context.extracted_fields.get("author_date", {})
        if author_date:
            authors = author_date.get("authors")
            date = author_date.get("date")
            if authors:
                sections.append(f"Authors: {', '.join(authors)}")
            if date:
                sections.append(f"Date: {date}")

        keywords_data = self.context.extracted_fields.get("keywords", {})
        if keywords_data:
            keywords = keywords_data.get("keywords")
            if keywords:
                sections.append(f"Keywords: {', '.join(keywords)}")

        doc_type_data = self.context.extracted_fields.get("document_type", {})
        if doc_type_data:
            doc_type = doc_type_data.get("document_type")
            if doc_type:
                sections.append(f"Document Type: {doc_type}")

        # Add relevant excerpts with similarity scores
        sections.append("\n=== RELEVANT EXCERPTS ===")
        for i, (chunk, score) in enumerate(retrieved, 1):
            sections.append(f"\n[Excerpt {i}] (similarity: {score:.2f})")
            sections.append(chunk)

        return "\n".join(sections)

    def get_chunk_count(self) -> int:
        """Return number of chunks created."""
        return len(self.chunks)

    def get_all_chunks(self) -> List[str]:
        """Return all chunks (for debugging/testing)."""
        return self.chunks.copy()


def create_rag_store(context: DocumentContext) -> Union[RAGStore, EmbeddingRAGStore]:
    """
    Factory function to create RAG store based on config.

    Args:
        context: DocumentContext for the RAG store

    Returns:
        RAGStore or EmbeddingRAGStore based on config["chat"]["rag_method"]
    """
    method = CONFIG["chat"].get("rag_method", "keyword")
    if method == "embedding":
        return EmbeddingRAGStore(context)
    return RAGStore(context)
