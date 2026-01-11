"""Tests for chat module"""
import pytest
from typing import Dict, List, Optional
from chat.context import DocumentContext
from chat.chat_handler import ChatHandler
from llm.provider import LLMProvider
from config import CONFIG


class MockChatProvider(LLMProvider):
    """Mock LLM provider for chat testing"""

    def __init__(self, responses: List[str] = None):
        """
        Args:
            responses: List of response strings to return in order
        """
        self.responses = responses or ["This is a mock response about the document."]
        self.call_index = 0
        self.messages = []
        self.system_prompts = []
        self.user_prompts = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List] = None
    ) -> Dict:
        """Return mock response"""
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)

        response = self.responses[self.call_index % len(self.responses)]
        self.call_index += 1

        return {"type": "text", "content": response}

    def reset_conversation(self):
        """Clear conversation history"""
        self.messages = []

    def add_tool_result(self, tool_name: str, result: Dict):
        """Add tool result to conversation (not used in chat)"""
        pass


# Test fixtures
@pytest.fixture
def sample_workflow_results():
    """Sample workflow results for testing"""
    return {
        "ocr_text": "Medical Report\nPatient: John Doe\nDate: 2024-03-15\nDiagnosis: Healthy",
        "author_date": {"authors": ["Dr. Smith"], "date": "2024-03-15"},
        "keywords": {"keywords": ["medical", "diagnosis", "patient"]},
        "document_type": {"document_type": "medical"},
    }


@pytest.fixture
def sample_context(sample_workflow_results):
    """Sample DocumentContext for testing"""
    return DocumentContext.from_workflow_results(sample_workflow_results)


# DocumentContext tests
class TestDocumentContext:
    def test_context_creation(self):
        """Test basic context creation"""
        context = DocumentContext(
            ocr_text="Test document content",
            extracted_fields={"author_date": {"authors": ["Test Author"]}}
        )
        assert context.ocr_text == "Test document content"
        assert context.extracted_fields["author_date"]["authors"] == ["Test Author"]

    def test_context_string_format(self, sample_context):
        """Test context string includes OCR text and metadata"""
        context_str = sample_context.to_context_string()

        assert "EXTRACTED INFORMATION" in context_str
        assert "DOCUMENT TEXT" in context_str
        assert "Dr. Smith" in context_str
        assert "2024-03-15" in context_str
        assert "medical" in context_str
        assert "Medical Report" in context_str

    def test_from_workflow_results(self, sample_workflow_results):
        """Test factory method creates context correctly"""
        context = DocumentContext.from_workflow_results(sample_workflow_results)

        assert context.ocr_text == sample_workflow_results["ocr_text"]
        assert context.extracted_fields["author_date"] == sample_workflow_results["author_date"]
        assert context.extracted_fields["keywords"] == sample_workflow_results["keywords"]
        assert context.extracted_fields["document_type"] == sample_workflow_results["document_type"]

    def test_is_empty_with_content(self, sample_context):
        """Test is_empty returns False when content exists"""
        assert sample_context.is_empty() is False

    def test_is_empty_without_content(self):
        """Test is_empty returns True for empty OCR text"""
        context = DocumentContext(ocr_text="", extracted_fields={})
        assert context.is_empty() is True

    def test_context_with_missing_fields(self):
        """Test context handles missing extracted fields gracefully"""
        context = DocumentContext(
            ocr_text="Some text",
            extracted_fields={}
        )
        context_str = context.to_context_string()

        # Should not raise errors
        assert "Some text" in context_str


# ChatHandler tests
class TestChatHandler:
    def test_chat_returns_response(self, sample_context):
        """Test chat returns a response string"""
        mock_provider = MockChatProvider(["The patient is John Doe."])
        handler = ChatHandler(context=sample_context, provider=mock_provider)

        response = handler.chat("Who is the patient?")

        assert isinstance(response, str)
        assert len(response) > 0
        assert response == "The patient is John Doe."

    def test_chat_grounded_in_context(self, sample_context):
        """Test system prompt includes document context"""
        mock_provider = MockChatProvider()
        handler = ChatHandler(context=sample_context, provider=mock_provider)

        handler.chat("What is the date?")

        # Check system prompt was sent with document context
        assert len(mock_provider.system_prompts) > 0
        system_prompt = mock_provider.system_prompts[0]
        assert "Medical Report" in system_prompt
        assert "Dr. Smith" in system_prompt

    def test_chat_history_maintained(self, sample_context):
        """Test conversation history is maintained"""
        mock_provider = MockChatProvider(["Response 1", "Response 2", "Response 3"])
        handler = ChatHandler(context=sample_context, provider=mock_provider)

        handler.chat("Question 1")
        handler.chat("Question 2")
        handler.chat("Question 3")

        history = handler.get_history()
        assert len(history) == 6  # 3 user + 3 assistant messages
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_chat_history_truncation(self, sample_context):
        """Test old messages are removed when limit exceeded"""
        original_max = CONFIG["chat"]["max_history_turns"]
        CONFIG["chat"]["max_history_turns"] = 2  # Allow only 2 turns

        try:
            mock_provider = MockChatProvider(["R1", "R2", "R3", "R4"])
            handler = ChatHandler(context=sample_context, provider=mock_provider)

            handler.chat("Q1")
            handler.chat("Q2")
            handler.chat("Q3")
            handler.chat("Q4")

            history = handler.get_history()
            # Should only keep last 2 turns (4 messages)
            assert len(history) == 4
            # First message should be Q3, not Q1
            assert history[0]["content"] == "Q3"
        finally:
            CONFIG["chat"]["max_history_turns"] = original_max

    def test_chat_empty_context(self):
        """Test chat with empty document context"""
        empty_context = DocumentContext(ocr_text="", extracted_fields={})
        mock_provider = MockChatProvider()
        handler = ChatHandler(context=empty_context, provider=mock_provider)

        response = handler.chat("What is this document about?")

        assert "No document" in response or "Please analyze" in response

    def test_clear_history(self, sample_context):
        """Test clearing conversation history"""
        mock_provider = MockChatProvider()
        handler = ChatHandler(context=sample_context, provider=mock_provider)

        handler.chat("Question 1")
        handler.chat("Question 2")
        assert len(handler.get_history()) == 4

        handler.clear_history()

        assert len(handler.get_history()) == 0

    def test_chat_uses_config_settings(self, sample_context):
        """Test chat uses configuration for temperature and tokens"""
        mock_provider = MockChatProvider()
        handler = ChatHandler(context=sample_context, provider=mock_provider)

        assert handler.temperature == CONFIG["chat"]["temperature"]
        assert handler.max_tokens == CONFIG["chat"]["max_tokens"]
        assert handler.max_history_turns == CONFIG["chat"]["max_history_turns"]


# E2E tests (require Ollama)
@pytest.mark.e2e
class TestChatE2E:
    def test_chat_with_real_llm(self, sample_context):
        """Test chat with actual Ollama provider"""
        handler = ChatHandler(context=sample_context)

        response = handler.chat("What type of document is this?")

        assert isinstance(response, str)
        assert len(response) > 0
        # Response should mention medical since that's the document type
        assert "medical" in response.lower() or "report" in response.lower()

    def test_chat_refuses_unknown_info(self, sample_context):
        """Test chat refuses to answer questions not in document"""
        handler = ChatHandler(context=sample_context)

        response = handler.chat("What is the weather forecast for tomorrow?")

        assert isinstance(response, str)
        # Should indicate it can't find the info
        assert "cannot find" in response.lower() or "not in the document" in response.lower() or "don't" in response.lower()
