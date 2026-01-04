"""Tests for LLM extraction module"""
import pytest
from typing import Dict, List, Optional
from llm.extractor import LLMExtractor
from llm.provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing multi-turn conversations"""

    def __init__(self, responses=None, simulate_tool_call=False):
        """
        Args:
            responses: Dict mapping task names to JSON response strings
            simulate_tool_call: If True, first response is tool_call, then text
        """
        self.responses = responses or {}
        self.call_count = 0
        self.messages = []
        self.simulate_tool_call = simulate_tool_call
        self.tool_call_count = 0

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, tools: Optional[List] = None) -> Dict:
        """Return mock response based on user prompt"""
        self.call_count += 1

        # Simulate tool call on first turn if requested
        if self.simulate_tool_call and self.tool_call_count == 0 and tools:
            self.tool_call_count += 1
            return {
                "type": "tool_call",
                "tool_calls": [{
                    "function": {
                        "name": "validate_date",
                        "arguments": '{"date_string": "2024-03-15"}'
                    }
                }]
            }

        # Determine task from prompt
        content = ""
        if "author" in user_prompt.lower():
            content = self.responses.get("author_date", '{"authors": ["Dr. Example"], "date": "2009-07-09"}')
        elif "keyword" in user_prompt.lower():
            content = self.responses.get("keywords", '{"keywords": ["medical", "hospital", "report"]}')
        elif "classify" in user_prompt.lower() or "document type" in user_prompt.lower():
            content = self.responses.get("document_type", '{"document_type": "medical"}')
        else:
            content = '{}'

        return {
            "type": "text",
            "content": content
        }

    def reset_conversation(self):
        """Clear conversation history"""
        self.messages = []
        self.tool_call_count = 0

    def add_tool_result(self, tool_name: str, result: Dict):
        """Add tool result to conversation"""
        self.messages.append({
            "role": "tool",
            "name": tool_name,
            "result": result
        })


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


# Multi-turn and tool support tests

def test_multi_turn_with_tool_call(sample_ocr_text):
    """Test multi-turn conversation with tool call"""
    mock_provider = MockLLMProvider(simulate_tool_call=True)
    extractor = LLMExtractor(provider=mock_provider)

    result, validation = extractor.extract_field(sample_ocr_text, "author_date")

    # Should have made 2 calls: first for tool call, second for final answer
    assert mock_provider.call_count == 2
    assert isinstance(result, dict)
    assert validation["valid_json"] is True


def test_conversation_reset_between_tasks():
    """Test that conversation is reset between different extraction tasks"""
    mock_provider = MockLLMProvider()
    extractor = LLMExtractor(provider=mock_provider)

    # Extract first field
    extractor.extract_field("Some text", "keywords")
    first_call_count = mock_provider.call_count

    # Extract second field - conversation should be reset
    extractor.extract_field("Some text", "document_type")

    # Conversation should have been reset
    assert len(mock_provider.messages) == 0 or mock_provider.tool_call_count == 0


def test_tool_execution_for_author_date():
    """Test that tools are provided for author_date task"""
    mock_provider = MockLLMProvider(simulate_tool_call=True)
    extractor = LLMExtractor(provider=mock_provider)

    # For author_date, tools should be used
    result, validation = extractor.extract_field("Report from Dr. Smith dated 2024-03-15", "author_date")

    # Should have tool result in messages
    assert len(mock_provider.messages) > 0


def test_no_tools_for_other_tasks(sample_ocr_text):
    """Test that tools are not provided for non-author_date tasks"""
    mock_provider = MockLLMProvider()
    extractor = LLMExtractor(provider=mock_provider)

    # For keywords task, should complete in single turn
    result, validation = extractor.extract_field(sample_ocr_text, "keywords")

    # Should be single-turn (no tool calls)
    assert mock_provider.call_count == 1
    assert validation["valid_json"] is True


def test_max_turns_limit():
    """Test that tools are disabled after first call to prevent infinite loops"""
    # Create provider that always returns tool calls if tools are available
    class InfiniteToolCallProvider(MockLLMProvider):
        def generate(self, system_prompt, user_prompt, temperature, max_tokens, tools=None):
            self.call_count += 1
            if tools:
                return {
                    "type": "tool_call",
                    "tool_calls": [{
                        "function": {
                            "name": "validate_date",
                            "arguments": '{"date_string": "2024-03-15"}'
                        }
                    }]
                }
            return {"type": "text", "content": '{"authors": null, "date": null}'}

    mock_provider = InfiniteToolCallProvider()
    extractor = LLMExtractor(provider=mock_provider)

    result, validation = extractor.extract_field("Some text", "author_date")

    # Should make exactly 2 calls: one tool call, then final response with tools disabled
    assert mock_provider.call_count == 2
    # Should succeed because tools are disabled after first call
    assert validation["valid_json"] is True
    assert result == {"authors": None, "date": None}
