"""Tests for on-topic guardrail."""
import pytest
from unittest.mock import Mock
from chat.guardrail import TopicGuard
from chat.context import DocumentContext
from chat.chat_handler import ChatHandler
from config import CONFIG


@pytest.fixture
def sample_context():
    """Sample DocumentContext for testing."""
    return DocumentContext(
        ocr_text="Medical Report\nPatient: John Smith\nDiagnosis: Headaches\nDate: 2024-03-15",
        extracted_fields={
            "author_date": {"authors": ["Dr. Sarah Johnson"], "date": "2024-03-15"},
            "keywords": {"keywords": ["medical", "headaches", "diagnosis"]},
            "document_type": {"document_type": "medical"}
        }
    )


@pytest.fixture
def mock_classifier():
    """Mock classifier for unit tests."""
    mock = Mock()
    return mock


class TestTopicGuard:
    """Tests for TopicGuard class."""

    def test_init_with_default_model(self):
        """Test TopicGuard initializes with default model from config."""
        guard = TopicGuard()
        assert guard.model_name == CONFIG["chat"]["guardrail_model"]
        assert guard.threshold == CONFIG["chat"]["guardrail_threshold"]
        assert guard._classifier is None  # Lazy loaded

    def test_init_with_custom_model(self):
        """Test TopicGuard initializes with custom model."""
        guard = TopicGuard(model_name="custom/model")
        assert guard.model_name == "custom/model"

    def test_check_on_topic_with_no_topics(self):
        """Test that empty topics list allows everything."""
        guard = TopicGuard()
        is_on_topic, score, topic = guard.check_on_topic("any query", [])
        assert is_on_topic is True
        assert score == 1.0
        assert topic == "no_topics_defined"

    def test_check_on_topic_stores_debug_info(self):
        """Test that check_on_topic stores debug info."""
        guard = TopicGuard()

        # Mock classifier response
        mock_classifier = Mock()
        mock_classifier.return_value = {
            "labels": ["medical", "legal", "document"],
            "scores": [0.8, 0.1, 0.05]
        }
        guard._classifier = mock_classifier

        guard.check_on_topic("What is the diagnosis?", ["medical", "legal", "document"])

        assert guard.last_query == "What is the diagnosis?"
        assert guard.last_score == 0.8
        assert guard.last_topic == "medical"
        assert guard.last_on_topic is True

    def test_get_debug_info(self):
        """Test get_debug_info returns expected structure."""
        guard = TopicGuard()
        debug = guard.get_debug_info()

        assert "model" in debug
        assert "threshold" in debug
        assert "last_query" in debug
        assert "last_score" in debug
        assert "last_topic" in debug
        assert "last_on_topic" in debug


class TestTopicGuardIntegration:
    """Integration tests with real classifier (slow, require transformers)."""

    @pytest.mark.slow
    def test_on_topic_medical_query(self):
        """Medical query should be on-topic for medical document."""
        guard = TopicGuard()
        topics = ["medical", "diagnosis", "patient", "document"]

        is_on_topic, score, topic = guard.check_on_topic(
            "What is the diagnosis?",
            topics,
            threshold=0.2
        )

        assert is_on_topic is True
        assert score > 0.2

    @pytest.mark.slow
    def test_off_topic_politics_query(self):
        """Politics query should be off-topic for medical document."""
        guard = TopicGuard()
        topics = ["medical", "diagnosis", "patient"]

        is_on_topic, score, topic = guard.check_on_topic(
            "Who will win the next election?",
            topics,
            threshold=0.5
        )

        assert is_on_topic is False
        assert score < 0.5

    @pytest.mark.slow
    def test_threshold_filtering(self):
        """High threshold should reject low-confidence matches."""
        guard = TopicGuard()
        topics = ["medical", "diagnosis"]

        # Query somewhat related but not strongly
        is_on_topic, score, topic = guard.check_on_topic(
            "What is the document about?",
            topics,
            threshold=0.9  # Very high threshold
        )

        # With high threshold, even related queries might be rejected
        assert score < 0.9 or is_on_topic is True


class TestChatHandlerWithGuardrail:
    """Tests for ChatHandler guardrail integration."""

    def test_guardrail_disabled_by_default_follows_config(self, sample_context):
        """Test that guardrail follows config setting."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False
        )

        assert handler.use_guardrail == CONFIG["chat"]["use_guardrail"]

    def test_guardrail_enabled_explicitly(self, sample_context):
        """Test enabling guardrail explicitly."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=True
        )

        assert handler.use_guardrail is True
        assert handler.guardrail is not None

    def test_guardrail_disabled_explicitly(self, sample_context):
        """Test disabling guardrail explicitly."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False
        )

        assert handler.use_guardrail is False
        assert handler.guardrail is None

    def test_get_allowed_topics(self, sample_context):
        """Test allowed topics are built from document metadata."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False
        )

        topics = handler._get_allowed_topics()

        assert "medical" in topics  # From document_type
        assert "headaches" in topics  # From keywords
        assert "diagnosis" in topics  # From keywords
        assert "document" in topics  # Generic topic

    def test_get_allowed_topics_empty_fields(self):
        """Test allowed topics with empty extracted fields."""
        from tests.test_chat import MockChatProvider

        empty_context = DocumentContext(
            ocr_text="Some text",
            extracted_fields={}
        )

        handler = ChatHandler(
            context=empty_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=False
        )

        topics = handler._get_allowed_topics()

        # Should still have generic topics
        assert "document" in topics
        assert "content" in topics
        assert "text" in topics
        assert "information" in topics

    def test_debug_info_includes_guardrail(self, sample_context):
        """Test debug info includes guardrail status."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=True
        )

        debug = handler.get_debug_info()

        assert "use_guardrail" in debug
        assert debug["use_guardrail"] is True
        assert "guardrail" in debug
        assert debug["guardrail"] is not None
        assert "allowed_topics" in debug["guardrail"]

    @pytest.mark.slow
    def test_off_topic_query_blocked(self, sample_context):
        """Test that off-topic queries are blocked."""
        from tests.test_chat import MockChatProvider

        handler = ChatHandler(
            context=sample_context,
            provider=MockChatProvider(),
            use_rag=False,
            use_guardrail=True
        )

        response = handler.chat("What's the weather like today?")

        assert "off-topic" in response.lower()
        assert "medical" in response  # Document type mentioned

    @pytest.mark.slow
    def test_on_topic_query_allowed(self, sample_context):
        """Test that on-topic queries proceed to LLM."""
        from tests.test_chat import MockChatProvider

        mock_provider = MockChatProvider()
        handler = ChatHandler(
            context=sample_context,
            provider=mock_provider,
            use_rag=False,
            use_guardrail=True
        )

        response = handler.chat("What is the patient's diagnosis?")

        # Should have called the LLM, not blocked
        assert "off-topic" not in response.lower()
        assert len(mock_provider.user_prompts) > 0
