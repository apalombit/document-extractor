"""Tests for RAG module"""
import pytest
from chat.context import DocumentContext
from chat.rag import RAGStore, EmbeddingRAGStore, create_rag_store
from config import CONFIG


@pytest.fixture
def sample_long_text():
    """Sample long document text for chunking tests"""
    return """Medical Report

Patient Name: John Smith
Date of Visit: March 15, 2024
Attending Physician: Dr. Sarah Johnson

Chief Complaint:
The patient presents with persistent headaches and fatigue over the past two weeks.

Medical History:
Patient has a history of hypertension, controlled with medication. No known allergies.
Previous surgeries include appendectomy in 2015.

Physical Examination:
Blood pressure: 130/85 mmHg
Heart rate: 72 bpm
Temperature: 98.6 F
General appearance: Patient appears tired but alert.

Laboratory Results:
Complete blood count shows normal values.
Thyroid function tests within normal limits.
Vitamin D levels slightly below normal at 25 ng/mL.

Diagnosis:
1. Tension headaches
2. Mild vitamin D deficiency

Treatment Plan:
1. Vitamin D supplementation 2000 IU daily
2. Stress management techniques recommended
3. Follow-up appointment in 4 weeks

Physician Signature: Dr. Sarah Johnson, MD"""


@pytest.fixture
def sample_context(sample_long_text):
    """Sample DocumentContext with long text"""
    return DocumentContext(
        ocr_text=sample_long_text,
        extracted_fields={
            "author_date": {"authors": ["Dr. Sarah Johnson"], "date": "2024-03-15"},
            "keywords": {"keywords": ["medical", "headaches", "vitamin"]},
            "document_type": {"document_type": "medical"}
        }
    )


@pytest.fixture
def empty_context():
    """Empty DocumentContext"""
    return DocumentContext(ocr_text="", extracted_fields={})


class TestRAGStore:
    def test_chunk_creation(self, sample_context):
        """Test that document is chunked"""
        rag = RAGStore(sample_context, chunk_size=200)

        assert rag.get_chunk_count() > 0
        chunks = rag.get_all_chunks()
        assert all(len(chunk) > 0 for chunk in chunks)

    def test_chunk_size_respected(self, sample_context):
        """Test that chunks respect size limit"""
        chunk_size = 150
        rag = RAGStore(sample_context, chunk_size=chunk_size)

        chunks = rag.get_all_chunks()
        # Most chunks should be under limit (some may slightly exceed)
        under_limit = sum(1 for c in chunks if len(c) <= chunk_size * 1.5)
        assert under_limit / len(chunks) > 0.8

    def test_empty_document(self, empty_context):
        """Test RAG with empty document"""
        rag = RAGStore(empty_context)

        assert rag.get_chunk_count() == 0
        assert rag.retrieve("any query") == []

    def test_retrieve_relevant_chunks(self, sample_context):
        """Test that retrieval finds relevant chunks"""
        rag = RAGStore(sample_context, chunk_size=200)

        results = rag.retrieve("blood pressure heart rate")

        assert len(results) > 0
        # First result should contain relevant terms
        top_chunk, score = results[0]
        assert score > 0
        # Check if blood pressure or heart rate is in result
        assert "blood" in top_chunk.lower() or "heart" in top_chunk.lower() or "pressure" in top_chunk.lower()

    def test_retrieve_respects_top_k(self, sample_context):
        """Test that retrieval respects top_k limit"""
        rag = RAGStore(sample_context, chunk_size=100)

        results = rag.retrieve("patient", top_k=2)
        assert len(results) <= 2

    def test_retrieve_no_matches(self, sample_context):
        """Test retrieval with query that won't match"""
        rag = RAGStore(sample_context)

        results = rag.retrieve("xyzabc123")
        assert len(results) == 0

    def test_retrieve_scores_sorted(self, sample_context):
        """Test that results are sorted by score descending"""
        rag = RAGStore(sample_context, chunk_size=100)

        results = rag.retrieve("patient medical")

        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)

    def test_augmented_context_includes_metadata(self, sample_context):
        """Test augmented context includes extracted metadata"""
        rag = RAGStore(sample_context, chunk_size=200)

        context = rag.get_augmented_context("headaches")

        assert "EXTRACTED INFORMATION" in context
        assert "Dr. Sarah Johnson" in context
        assert "RELEVANT EXCERPTS" in context

    def test_augmented_context_fallback(self, sample_context):
        """Test fallback to full context when no matches"""
        rag = RAGStore(sample_context)

        # Query that won't match
        context = rag.get_augmented_context("xyzabc123")

        # Should fall back to full context
        assert "DOCUMENT TEXT" in context

    def test_tokenize_filters_short_words(self, sample_context):
        """Test that tokenization filters short words"""
        rag = RAGStore(sample_context)

        tokens = rag._tokenize("I am a test of short words")
        # "I", "am", "a", "of" should be filtered (< 3 chars)
        assert "i" not in tokens
        assert "am" not in tokens
        assert "test" in tokens

    def test_tokenize_filters_numbers(self, sample_context):
        """Test that tokenization filters pure numbers"""
        rag = RAGStore(sample_context)

        tokens = rag._tokenize("Patient age is 45 years old in 2024")
        assert "45" not in tokens
        assert "2024" not in tokens
        assert "patient" in tokens


class TestChatHandlerWithRAG:
    def test_rag_disabled_by_default(self, sample_context):
        """Test that RAG is disabled by default"""
        from chat.chat_handler import ChatHandler
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider()
        )

        assert handler.use_rag == CONFIG["chat"]["use_rag"]
        if not CONFIG["chat"]["use_rag"]:
            assert handler.rag_store is None

    def test_rag_enabled_explicitly(self, sample_context):
        """Test enabling RAG explicitly"""
        from chat.chat_handler import ChatHandler
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=True
        )

        assert handler.use_rag is True
        assert handler.rag_store is not None

    def test_rag_retrieval_in_chat(self, sample_context):
        """Test that RAG retrieval affects system prompt"""
        from chat.chat_handler import ChatHandler
        from tests.test_chat import MockChatProvider

        mock_provider = MockChatProvider()
        handler = ChatHandler(
            context=sample_context,
            provider=mock_provider,
            use_rag=True
        )

        handler.chat("What is the blood pressure?")

        # System prompt should use RAG context (has RELEVANT EXCERPTS)
        system_prompt = mock_provider.system_prompts[0]
        assert "RELEVANT EXCERPTS" in system_prompt or "blood" in system_prompt.lower()


class TestEmbeddingRAGStore:
    """Tests for embedding-based RAG using SentenceTransformer."""

    def test_embedding_chunk_creation(self, sample_context):
        """Test that document is chunked and embeddings are created"""
        rag = EmbeddingRAGStore(sample_context, chunk_size=200)

        assert rag.get_chunk_count() > 0
        assert rag.embeddings is not None
        assert len(rag.embeddings) == rag.get_chunk_count()

    def test_embedding_dimensions(self, sample_context):
        """Test that embeddings have correct dimensions"""
        rag = EmbeddingRAGStore(sample_context, chunk_size=200)

        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert rag.embeddings.shape[1] == 384

    def test_cosine_similarity_retrieval(self, sample_context):
        """Test that retrieval uses cosine similarity"""
        rag = EmbeddingRAGStore(sample_context, chunk_size=200)

        results = rag.retrieve("blood pressure heart rate")

        assert len(results) > 0
        # Scores should be between 0 and 1 for cosine similarity
        for chunk, score in results:
            assert 0 <= score <= 1

    def test_semantic_similarity(self, sample_context):
        """Test semantic matching (not just keyword matching)"""
        rag = EmbeddingRAGStore(sample_context, chunk_size=200)

        # Query with synonyms/related terms
        results = rag.retrieve("patient health condition")

        assert len(results) > 0
        # Should find medically related chunks even without exact keyword match

    def test_threshold_filtering(self, sample_context):
        """Test that low-similarity results are filtered by threshold"""
        original_threshold = CONFIG["chat"]["rag_similarity_threshold"]
        CONFIG["chat"]["rag_similarity_threshold"] = 0.9  # Very high threshold

        try:
            rag = EmbeddingRAGStore(sample_context, chunk_size=200)
            results = rag.retrieve("random unrelated query xyz")

            # High threshold should filter out low-similarity results
            for chunk, score in results:
                assert score >= 0.9
        finally:
            CONFIG["chat"]["rag_similarity_threshold"] = original_threshold

    def test_empty_document(self, empty_context):
        """Test embedding RAG with empty document"""
        rag = EmbeddingRAGStore(empty_context)

        assert rag.get_chunk_count() == 0
        assert rag.embeddings is None
        assert rag.retrieve("any query") == []

    def test_augmented_context_includes_similarity(self, sample_context):
        """Test augmented context shows similarity scores"""
        rag = EmbeddingRAGStore(sample_context, chunk_size=200)

        context = rag.get_augmented_context("headaches treatment")

        assert "RELEVANT EXCERPTS" in context
        assert "similarity:" in context

    def test_retrieve_respects_top_k(self, sample_context):
        """Test that retrieval respects top_k limit"""
        rag = EmbeddingRAGStore(sample_context, chunk_size=100)

        results = rag.retrieve("patient", top_k=2)
        assert len(results) <= 2

    def test_results_sorted_by_similarity(self, sample_context):
        """Test that results are sorted by similarity descending"""
        rag = EmbeddingRAGStore(sample_context, chunk_size=100)

        results = rag.retrieve("medical diagnosis treatment")

        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)


class TestRAGFactory:
    """Tests for create_rag_store factory function."""

    def test_factory_creates_keyword_rag(self, sample_context):
        """Test factory creates keyword RAG when configured"""
        original_method = CONFIG["chat"]["rag_method"]
        CONFIG["chat"]["rag_method"] = "keyword"

        try:
            rag = create_rag_store(sample_context)
            assert isinstance(rag, RAGStore)
            assert not isinstance(rag, EmbeddingRAGStore)
        finally:
            CONFIG["chat"]["rag_method"] = original_method

    def test_factory_creates_embedding_rag(self, sample_context):
        """Test factory creates embedding RAG when configured"""
        original_method = CONFIG["chat"]["rag_method"]
        CONFIG["chat"]["rag_method"] = "embedding"

        try:
            rag = create_rag_store(sample_context)
            assert isinstance(rag, EmbeddingRAGStore)
        finally:
            CONFIG["chat"]["rag_method"] = original_method

    def test_factory_defaults_to_keyword(self, sample_context):
        """Test factory defaults to keyword RAG for unknown method"""
        original_method = CONFIG["chat"].get("rag_method")
        CONFIG["chat"]["rag_method"] = "unknown_method"

        try:
            rag = create_rag_store(sample_context)
            assert isinstance(rag, RAGStore)
            assert not isinstance(rag, EmbeddingRAGStore)
        finally:
            if original_method:
                CONFIG["chat"]["rag_method"] = original_method
