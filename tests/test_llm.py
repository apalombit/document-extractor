"""Tests for LLM extraction module"""
import pytest
from llm.extractor import LLMExtractor
from llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        """Return mock response based on user prompt"""
        self.call_count += 1

        # Determine task from prompt
        if "author" in user_prompt.lower():
            return self.responses.get("author_date", '{"authors": ["Dr. Example"], "date": "2009-07-09"}')
        elif "keyword" in user_prompt.lower():
            return self.responses.get("keywords", '{"keywords": ["medical", "hospital", "report"]}')
        elif "classify" in user_prompt.lower() or "document type" in user_prompt.lower():
            return self.responses.get("document_type", '{"document_type": "medical"}')
        else:
            return '{}'


def test_extract_author_date(sample_ocr_text):
    """Test extraction of author names and dates"""
    mock_provider = MockLLMProvider()
    extractor = LLMExtractor(provider=mock_provider)

    result, validation = extractor.extract_field(sample_ocr_text, "author_date")

    assert isinstance(result, dict)
    assert "authors" in result or "date" in result
    assert isinstance(validation, dict)
    assert "valid_json" in validation
    assert "confidence" in validation


def test_extract_keywords(sample_ocr_text):
    """Test keyword extraction"""
    mock_provider = MockLLMProvider()
    extractor = LLMExtractor(provider=mock_provider)

    result, validation = extractor.extract_field(sample_ocr_text, "keywords")

    assert isinstance(result, dict)
    assert "keywords" in result
    assert isinstance(validation, dict)
    assert validation["valid_json"] is True


def test_extract_document_type(sample_ocr_text):
    """Test document type classification"""
    mock_provider = MockLLMProvider()
    extractor = LLMExtractor(provider=mock_provider)

    result, validation = extractor.extract_field(sample_ocr_text, "document_type")

    assert isinstance(result, dict)
    assert "document_type" in result
    assert isinstance(validation, dict)
    assert validation["valid_json"] is True


def test_json_output_format(sample_ocr_text):
    """Test LLM returns valid JSON format"""
    mock_provider = MockLLMProvider()
    extractor = LLMExtractor(provider=mock_provider)

    result, validation = extractor.extract_field(sample_ocr_text, "keywords")

    # Verify it's valid JSON (dict)
    assert isinstance(result, dict)
    assert validation["valid_json"] is True


def test_null_when_info_missing():
    """Test LLM returns null when information not present"""
    # Mock provider that returns null values
    mock_provider = MockLLMProvider(responses={
        "author_date": '{"authors": null, "date": null}'
    })
    extractor = LLMExtractor(provider=mock_provider)

    empty_text = "This is a blank document with no useful information."
    result, validation = extractor.extract_field(empty_text, "author_date")

    # Should handle null values gracefully
    assert isinstance(result, dict)
    assert validation["valid_json"] is True


def test_invalid_json_handling():
    """Test handling of invalid JSON from LLM"""
    # Mock provider that returns invalid JSON
    mock_provider = MockLLMProvider(responses={
        "keywords": 'This is not JSON!'
    })
    extractor = LLMExtractor(provider=mock_provider)

    result, validation = extractor.extract_field("Some text", "keywords")

    # Should handle gracefully
    assert validation["valid_json"] is False
    assert len(validation["grounding_issues"]) > 0
    assert validation["confidence"] == "low"


def test_unknown_task_error():
    """Test error handling for unknown task"""
    mock_provider = MockLLMProvider()
    extractor = LLMExtractor(provider=mock_provider)

    with pytest.raises(ValueError, match="Unknown task"):
        extractor.extract_field("Some text", "invalid_task")
