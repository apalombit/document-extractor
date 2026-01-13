"""Tests for hallucination detection guardrail."""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from chat.hallucination_guard import HallucinationGuard
from chat.context import DocumentContext
from chat.chat_handler import ChatHandler
from config import CONFIG


@pytest.fixture
def sample_context():
    """Sample DocumentContext for testing."""
    return DocumentContext(
        ocr_text="Medical Report\nPatient: John Smith\nDiagnosis: Headaches\nDate: 2024-03-15\nDoctor: Dr. Sarah Johnson\nTreatment: Rest and medication",
        extracted_fields={
            "author_date": {"authors": ["Dr. Sarah Johnson"], "date": "2024-03-15"},
            "keywords": {"keywords": ["medical", "headaches", "diagnosis"]},
            "document_type": {"document_type": "medical"}
        }
    )


@pytest.fixture
def source_chunks():
    """Sample source chunks for testing."""
    return [
        "Patient John Smith was diagnosed with headaches.",
        "The diagnosis was made on March 15, 2024.",
        "Dr. Sarah Johnson recommended rest and medication.",
        "Follow-up appointment scheduled in 4 weeks."
    ]


class TestHallucinationGuard:
    """Unit tests for HallucinationGuard class."""

    def test_init_with_defaults(self):
        """Test HallucinationGuard initializes with config defaults."""
        guard = HallucinationGuard()
        assert guard.embedding_model_name == CONFIG["chat"]["rag_embedding_model"]
        assert guard.nli_model_name == CONFIG["chat"]["hallucination_nli_model"]
        assert guard._sentence_model is None  # Lazy loaded
        assert guard._nli_classifier is None  # Lazy loaded

    def test_init_with_custom_models(self):
        """Test HallucinationGuard initializes with custom models."""
        guard = HallucinationGuard(
            embedding_model="custom/embedding",
            nli_model="custom/nli",
            entailment_threshold=0.7
        )
        assert guard.embedding_model_name == "custom/embedding"
        assert guard.nli_model_name == "custom/nli"
        assert guard.entailment_threshold == 0.7

    def test_check_grounded_empty_response(self):
        """Test empty response is considered grounded."""
        guard = HallucinationGuard()
        is_grounded, results = guard.check_grounded("", ["source chunk"])
        assert is_grounded is True
        assert results == []

    def test_check_grounded_whitespace_response(self):
        """Test whitespace-only response is considered grounded."""
        guard = HallucinationGuard()
        is_grounded, results = guard.check_grounded("   ", ["source chunk"])
        assert is_grounded is True
        assert results == []

    def test_check_grounded_no_sources(self):
        """Test handling when no sources available."""
        guard = HallucinationGuard()
        is_grounded, results = guard.check_grounded("Some response text.", [])
        assert is_grounded is True
        assert len(results) == 1
        assert results[0].get("warning") == "no_sources_available"

    def test_get_hallucinated_sentences(self):
        """Test extraction of hallucinated sentences."""
        guard = HallucinationGuard()
        results = [
            {"sentence": "Grounded sentence.", "is_grounded": True},
            {"sentence": "Hallucinated sentence.", "is_grounded": False},
            {"sentence": "Another grounded one.", "is_grounded": True},
            {"sentence": "Another hallucinated one.", "is_grounded": False},
        ]

        hallucinated = guard.get_hallucinated_sentences(results)

        assert len(hallucinated) == 2
        assert "Hallucinated sentence." in hallucinated
        assert "Another hallucinated one." in hallucinated

    def test_get_hallucinated_sentences_empty(self):
        """Test extraction when no hallucinations."""
        guard = HallucinationGuard()
        results = [
            {"sentence": "Grounded sentence.", "is_grounded": True},
            {"sentence": "Another grounded one.", "is_grounded": True},
        ]

        hallucinated = guard.get_hallucinated_sentences(results)

        assert len(hallucinated) == 0

    def test_get_hallucinated_sentences_with_warning(self):
        """Test extraction handles warning results gracefully."""
        guard = HallucinationGuard()
        results = [
            {"warning": "no_sources_available"},
            {"sentence": "Some sentence.", "is_grounded": False},
        ]

        hallucinated = guard.get_hallucinated_sentences(results)

        # Should skip the warning entry
        assert len(hallucinated) == 1
        assert "Some sentence." in hallucinated

    def test_get_debug_info(self):
        """Test get_debug_info returns expected structure."""
        guard = HallucinationGuard()
        debug = guard.get_debug_info()

        assert "embedding_model" in debug
        assert "nli_model" in debug
        assert "entailment_threshold" in debug
        assert "similarity_threshold" in debug
        assert "top_k_sources" in debug
        assert "last_response" in debug
        assert "last_sentences" in debug
        assert "last_has_hallucination" in debug
        assert "last_results_count" in debug


class TestHallucinationGuardIntegration:
    """Integration tests with real models (slow)."""

    @pytest.mark.slow
    def test_sentence_splitting(self):
        """Test sentence splitting works correctly."""
        guard = HallucinationGuard()
        guard._ensure_nltk()

        text = "First sentence. Second sentence! Third sentence?"
        sentences = guard._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence." in sentences
        assert "Second sentence!" in sentences
        assert "Third sentence?" in sentences

    @pytest.mark.slow
    def test_sentence_splitting_filters_short(self):
        """Test that short sentences are filtered."""
        guard = HallucinationGuard()
        guard._ensure_nltk()

        text = "OK. This is a longer sentence that should be kept."
        sentences = guard._split_sentences(text)

        # "OK." is too short (< 10 chars)
        assert len(sentences) == 1
        assert "This is a longer sentence" in sentences[0]

    @pytest.mark.slow
    def test_find_relevant_sources(self, source_chunks):
        """Test finding relevant sources for a sentence."""
        guard = HallucinationGuard()

        # Compute embeddings
        source_embeddings = guard.sentence_model.encode(source_chunks)

        # Find sources for a related sentence
        sentence = "John Smith has headaches."
        relevant = guard._find_relevant_sources(
            sentence, source_chunks, source_embeddings
        )

        # Should find at least one relevant source
        assert len(relevant) > 0
        # Results should be (text, similarity) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in relevant)

    @pytest.mark.slow
    def test_check_grounded_with_real_models(self, source_chunks):
        """Test full grounding check with real models."""
        guard = HallucinationGuard()

        # Grounded response (information from sources)
        response = "John Smith was diagnosed with headaches. Dr. Sarah Johnson is the doctor."
        is_grounded, results = guard.check_grounded(response, source_chunks)

        # Should have results for each sentence
        assert len(results) > 0
        assert guard.last_response == response
        assert guard.last_sentences is not None


class TestChatHandlerWithHallucinationGuard:
    """Tests for ChatHandler hallucination guard integration."""

    def test_hallucination_guard_follows_config(self, sample_context):
        """Test hallucination guard follows config setting."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False
        )

        assert handler.use_hallucination_guard == CONFIG["chat"].get("use_hallucination_guard", False)

    def test_hallucination_guard_enabled_explicitly(self, sample_context):
        """Test enabling hallucination guard explicitly."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False,
            use_hallucination_guard=True
        )

        assert handler.use_hallucination_guard is True
        assert handler.hallucination_guard is not None

    def test_hallucination_guard_disabled_explicitly(self, sample_context):
        """Test disabling hallucination guard explicitly."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False,
            use_hallucination_guard=False
        )

        assert handler.use_hallucination_guard is False
        assert handler.hallucination_guard is None

    def test_debug_info_includes_hallucination_guard(self, sample_context):
        """Test debug info includes hallucination guard status."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False,
            use_hallucination_guard=True
        )

        debug = handler.get_debug_info()

        assert "use_hallucination_guard" in debug
        assert debug["use_hallucination_guard"] is True
        assert "hallucination_guard" in debug
        assert debug["hallucination_guard"] is not None

    def test_get_source_chunks_without_rag(self, sample_context):
        """Test source chunk retrieval without RAG store."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False,
            use_hallucination_guard=False
        )

        chunks = handler._get_source_chunks()

        # Should have chunked the OCR text
        assert len(chunks) > 0
        # Chunks should contain document text
        assert any("John Smith" in chunk or "Headaches" in chunk for chunk in chunks)

    def test_split_into_chunks(self, sample_context):
        """Test internal chunking method."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False
        )

        text = "First paragraph with some content.\n\nSecond paragraph with more content.\n\nThird paragraph."
        chunks = handler._split_into_chunks(text)

        assert len(chunks) >= 1
        # All chunks should be non-empty
        assert all(len(c) > 0 for c in chunks)

    def test_hallucination_blocked_with_mock(self, sample_context):
        """Test that hallucinated response is blocked."""
        from tests.test_chat import MockChatProvider

        mock_provider = MockChatProvider(responses=["The patient underwent brain surgery."])
        handler = ChatHandler(
            context=sample_context,
            provider=mock_provider,
            use_rag=False,
            use_guardrail=False,
            use_jailbreak_guard=False,
            use_hallucination_guard=True
        )

        # Mock hallucination guard to detect hallucination
        handler.hallucination_guard._sentence_model = Mock()
        handler.hallucination_guard._sentence_model.encode = Mock(
            return_value=np.array([[0.1, 0.2, 0.3]])
        )
        handler.hallucination_guard._nli_classifier = Mock(
            return_value=[{"label": "contradiction", "score": 0.9}]
        )
        handler.hallucination_guard._nltk_initialized = True

        with patch('nltk.tokenize.sent_tokenize', return_value=["The patient underwent brain surgery."]):
            response = handler.chat("What happened to the patient?")

        assert "could not be verified" in response.lower() or "not directly supported" in response.lower()

    def test_grounded_response_passes_with_mock(self, sample_context):
        """Test that grounded response passes."""
        from tests.test_chat import MockChatProvider

        mock_provider = MockChatProvider(responses=["John Smith has headaches."])
        handler = ChatHandler(
            context=sample_context,
            provider=mock_provider,
            use_rag=False,
            use_guardrail=False,
            use_jailbreak_guard=False,
            use_hallucination_guard=True
        )

        # Mock hallucination guard to pass
        handler.hallucination_guard._sentence_model = Mock()
        handler.hallucination_guard._sentence_model.encode = Mock(
            return_value=np.array([[0.9, 0.8, 0.7]])
        )
        handler.hallucination_guard._nli_classifier = Mock(
            return_value=[{"label": "entailment", "score": 0.95}]
        )
        handler.hallucination_guard._nltk_initialized = True

        with patch('nltk.tokenize.sent_tokenize', return_value=["John Smith has headaches."]):
            response = handler.chat("What is the diagnosis?")

        # Response should pass through (not blocked)
        assert "could not verify" not in response.lower()
        assert "John Smith" in response or "headaches" in response

    @pytest.mark.slow
    def test_hallucination_guard_real_integration(self, sample_context):
        """Integration test with real models."""
        from tests.test_chat import MockChatProvider

        mock_provider = MockChatProvider(responses=["The patient John Smith has headaches."])
        handler = ChatHandler(
            context=sample_context,
            provider=mock_provider,
            use_rag=False,
            use_guardrail=False,
            use_jailbreak_guard=False,
            use_hallucination_guard=True
        )

        response = handler.chat("What is the patient's condition?")

        # Guard should have processed the response
        assert handler.hallucination_guard.last_response is not None
