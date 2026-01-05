"""Tests for self-critique functionality"""
import pytest
from typing import Dict, List, Optional
from llm.extractor import LLMExtractor
from llm.provider import LLMProvider
from config import CONFIG


class MockLLMProviderWithCritique(LLMProvider):
    """Mock provider that simulates critique behavior"""

    def __init__(self, initial_response: str, critique_response: str = None):
        """
        Args:
            initial_response: JSON response for first extraction
            critique_response: JSON response after critique (if different from initial)
        """
        self.initial_response = initial_response
        self.critique_response = critique_response or initial_response
        self.call_count = 0
        self.messages = []

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, tools: Optional[List] = None) -> Dict:
        """Return mock response, alternating between initial and critique"""
        self.call_count += 1

        # First call returns initial response
        if self.call_count == 1:
            content = self.initial_response
        # Second call (critique) returns improved or same response
        else:
            content = self.critique_response

        return {
            "type": "text",
            "content": content
        }

    def reset_conversation(self):
        """Clear conversation history"""
        self.messages = []

    def add_tool_result(self, tool_name: str, result: Dict):
        """Add tool result (not used in these tests)"""
        pass


def test_critique_disabled_by_default():
    """Test that critique is disabled by default (enable_critique=False)"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Ensure critique is disabled
        CONFIG["llm"]["enable_critique"] = False

        mock_provider = MockLLMProviderWithCritique(
            initial_response='{"authors": ["Dr. Smith"], "date": "2024-03-15"}',
            critique_response='{"authors": ["Dr. Jones"], "date": "2024-03-20"}'  # Different response
        )
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "author_date")

        # Should only make 1 call (no critique)
        assert mock_provider.call_count == 1
        # Should return initial response, not critique
        assert result["authors"] == ["Dr. Smith"]
        assert result["date"] == "2024-03-15"
        assert validation["valid_json"] is True

    finally:
        # Restore original config
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_enabled_improves_answer():
    """Test that critique step can improve the initial answer"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        mock_provider = MockLLMProviderWithCritique(
            initial_response='{"authors": ["Generic Person"], "date": "2024-03-15"}',  # Flawed answer
            critique_response='{"authors": ["Dr. Smith"], "date": "2024-03-15"}'  # Improved answer
        )
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "author_date")

        # Should make 2 calls (initial + critique)
        assert mock_provider.call_count == 2
        # Should return improved response from critique
        assert result["authors"] == ["Dr. Smith"]
        assert result["date"] == "2024-03-15"
        assert validation["valid_json"] is True

    finally:
        # Restore original config
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_unchanged_when_correct():
    """Test that critique returns same answer when initial is already correct"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        correct_answer = '{"authors": ["Dr. Smith"], "date": "2024-03-15"}'
        mock_provider = MockLLMProviderWithCritique(
            initial_response=correct_answer,
            critique_response=correct_answer  # Same answer after critique
        )
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "author_date")

        # Should make 2 calls (initial + critique)
        assert mock_provider.call_count == 2
        # Should return same answer
        assert result["authors"] == ["Dr. Smith"]
        assert result["date"] == "2024-03-15"
        assert validation["valid_json"] is True

    finally:
        # Restore original config
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_handles_invalid_json():
    """Test that critique gracefully handles invalid JSON from critique step"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        mock_provider = MockLLMProviderWithCritique(
            initial_response='{"authors": ["Dr. Smith"], "date": "2024-03-15"}',
            critique_response='This is not valid JSON!'  # Invalid critique response
        )
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "author_date")

        # Should make 2 calls
        assert mock_provider.call_count == 2
        # Should fall back to original answer when critique fails
        assert result["authors"] == ["Dr. Smith"]
        assert result["date"] == "2024-03-15"

    finally:
        # Restore original config
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_with_null_values():
    """Test critique handles null values correctly"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        mock_provider = MockLLMProviderWithCritique(
            initial_response='{"authors": null, "date": null}',
            critique_response='{"authors": null, "date": null}'  # Still null after critique
        )
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "author_date")

        # Should make 2 calls
        assert mock_provider.call_count == 2
        # Should handle null values
        assert result["authors"] is None
        assert result["date"] is None
        assert validation["valid_json"] is True

    finally:
        # Restore original config
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_for_keywords_task():
    """Test critique works for keywords task"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        mock_provider = MockLLMProviderWithCritique(
            initial_response='{"keywords": ["generic", "words"]}',
            critique_response='{"keywords": ["specific", "technical", "terms"]}'  # Improved
        )
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "keywords")

        # Should make 2 calls
        assert mock_provider.call_count == 2
        # Should return improved keywords
        assert result["keywords"] == ["specific", "technical", "terms"]
        assert validation["valid_json"] is True

    finally:
        # Restore original config
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_for_document_type_task():
    """Test critique works for document_type task"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        mock_provider = MockLLMProviderWithCritique(
            initial_response='{"document_type": "other"}',  # Generic classification
            critique_response='{"document_type": "medical"}'  # More specific
        )
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "document_type")

        # Should make 2 calls
        assert mock_provider.call_count == 2
        # Should return improved classification
        assert result["document_type"] == "medical"
        assert validation["valid_json"] is True

    finally:
        # Restore original config
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_skips_empty_results():
    """Test critique skips processing when initial result is empty"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        mock_provider = MockLLMProviderWithCritique(
            initial_response='{}',  # Empty result
            critique_response='{"authors": ["Dr. Smith"], "date": "2024-03-15"}'
        )
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "author_date")

        # Should make 1 call only (critique skipped for empty result)
        # Note: The critique step checks if result is empty and skips if so
        # Empty results get returned early without critique
        assert result == {}

    finally:
        # Restore original config
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_with_tools_enabled():
    """Test that critique can use validate_date tool when enabled"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        # Mock provider that simulates tool call during critique
        class MockProviderWithToolsInCritique(LLMProvider):
            def __init__(self):
                self.call_count = 0
                self.messages = []
                self.in_critique = False

            def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, tools: Optional[List] = None) -> Dict:
                self.call_count += 1

                # Check if we're in critique (user prompt contains "Review your previous")
                if "Review your previous" in user_prompt:
                    self.in_critique = True

                # Initial extraction - return with European date format
                if not self.in_critique:
                    return {
                        "type": "text",
                        "content": '{"authors": ["Dr. Smith"], "date": "09/07/2009"}'
                    }
                else:
                    # Critique phase - first call tool to validate, second return normalized
                    if self.call_count == 2:
                        # First critique call - use tool
                        return {
                            "type": "tool_call",
                            "tool_calls": [{
                                "function": {
                                    "name": "validate_date",
                                    "arguments": {"date_string": "09/07/2009"}
                                }
                            }]
                        }
                    else:
                        # After tool result - return normalized date
                        return {
                            "type": "text",
                            "content": '{"authors": ["Dr. Smith"], "date": "2009-07-09"}'
                        }

            def reset_conversation(self):
                self.messages = []
                self.in_critique = False

            def add_tool_result(self, tool_name: str, result: Dict):
                self.messages.append({"tool": tool_name, "result": result})

        mock_provider = MockProviderWithToolsInCritique()
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "author_date")

        # Should make multiple calls (initial + critique with tool + final)
        assert mock_provider.call_count >= 2
        # Critique should normalize date via tool
        assert result["date"] == "2009-07-09"
        assert validation["valid_json"] is True

    finally:
        CONFIG["llm"]["enable_critique"] = original_critique


def test_critique_tool_call_handling():
    """Test critique properly handles tool execution and results"""
    # Save original config
    original_critique = CONFIG["llm"]["enable_critique"]

    try:
        # Enable critique
        CONFIG["llm"]["enable_critique"] = True

        # Mock provider to track tool usage
        class MockProviderTrackingTools(LLMProvider):
            def __init__(self):
                self.call_count = 0
                self.tool_results_added = []
                self.messages = []

            def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, tools: Optional[List] = None) -> Dict:
                self.call_count += 1

                # Initial extraction
                if self.call_count == 1:
                    return {
                        "type": "text",
                        "content": '{"authors": ["Dr. Smith"], "date": "2099-12-31"}'  # Future date (invalid)
                    }
                # Critique - call tool
                elif self.call_count == 2:
                    return {
                        "type": "tool_call",
                        "tool_calls": [{
                            "function": {
                                "name": "validate_date",
                                "arguments": '{"date_string": "2099-12-31"}'
                            }
                        }]
                    }
                # After tool - fix the date based on tool feedback
                else:
                    return {
                        "type": "text",
                        "content": '{"authors": ["Dr. Smith"], "date": null}'  # Set to null due to invalid
                    }

            def reset_conversation(self):
                self.messages = []

            def add_tool_result(self, tool_name: str, result: Dict):
                self.tool_results_added.append({"tool": tool_name, "result": result})

        mock_provider = MockProviderTrackingTools()
        extractor = LLMExtractor(provider=mock_provider)

        result, validation, tool_calls = extractor.extract_field("Some text", "author_date")

        # Verify tool was called during critique
        assert len(mock_provider.tool_results_added) > 0
        assert mock_provider.tool_results_added[0]["tool"] == "validate_date"

        # Verify tool result indicated invalid date
        tool_result = mock_provider.tool_results_added[0]["result"]
        assert tool_result["valid"] is False
        assert "future" in tool_result["reason"].lower()

        # Verify critique corrected the date to null
        assert result["date"] is None

    finally:
        CONFIG["llm"]["enable_critique"] = original_critique
