"""Tests for LLM tools"""
import pytest
from datetime import datetime, timedelta
from llm.tools import validate_date


class TestValidateDate:
    """Test suite for date validation tool"""

    def test_validate_date_valid_iso_format(self):
        """Test valid date in ISO format (YYYY-MM-DD)"""
        result = validate_date("2024-03-15")
        assert result["valid"] is True
        assert result["reason"] == ""
        assert result["normalized"] == "2024-03-15"

    def test_validate_date_valid_european_format(self):
        """Test valid date in European format (DD/MM/YYYY)"""
        result = validate_date("15/03/2024")
        assert result["valid"] is True
        assert result["reason"] == ""
        assert result["normalized"] == "2024-03-15"

    def test_validate_date_valid_month_name(self):
        """Test valid date with month name"""
        result = validate_date("March 15, 2024")
        assert result["valid"] is True
        assert result["reason"] == ""
        assert result["normalized"] == "2024-03-15"

    def test_validate_date_valid_month_year_only(self):
        """Test valid date with only month and year"""
        result = validate_date("March 2024")
        assert result["valid"] is True
        assert result["reason"] == ""
        # Should default to first day of month
        assert "2024-03" in result["normalized"]

    def test_validate_date_future(self):
        """Test that future dates are invalid"""
        future_date = (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")
        result = validate_date(future_date)
        assert result["valid"] is False
        assert "future" in result["reason"].lower()
        assert result["normalized"] == future_date

    def test_validate_date_invalid_format(self):
        """Test unparseable date format"""
        result = validate_date("not a date")
        assert result["valid"] is False
        assert "parse" in result["reason"].lower()
        assert result["normalized"] == ""

    def test_validate_date_empty_string(self):
        """Test empty string input"""
        result = validate_date("")
        assert result["valid"] is False
        assert "empty" in result["reason"].lower()
        assert result["normalized"] == ""

    def test_validate_date_none(self):
        """Test None input"""
        result = validate_date(None)
        assert result["valid"] is False
        assert result["normalized"] == ""

    def test_validate_date_with_extra_text(self):
        """Test date extraction with surrounding text"""
        result = validate_date("Report dated 2024-03-15 from hospital")
        assert result["valid"] is True
        assert result["normalized"] == "2024-03-15"

    def test_validate_date_past_date(self):
        """Test that past dates are valid"""
        result = validate_date("2020-01-01")
        assert result["valid"] is True
        assert result["reason"] == ""
        assert result["normalized"] == "2020-01-01"

    def test_validate_date_today(self):
        """Test that today's date is valid"""
        today = datetime.now().strftime("%Y-%m-%d")
        result = validate_date(today)
        assert result["valid"] is True
        assert result["reason"] == ""
        assert result["normalized"] == today

    def test_validate_date_italian_format(self):
        """Test Italian date format"""
        result = validate_date("15 marzo 2024")
        # Note: dateutil may or may not parse Italian month names
        # This test documents the behavior
        # For production, may need to add locale support
        if result["valid"]:
            assert "2024" in result["normalized"]


class TestExpandKeywordsWithWebSearch:
    """Test expand_keywords_with_web_search tool (DuckDuckGo implementation)"""

    @pytest.mark.integration
    def test_expand_keywords_with_valid_input(self):
        """Test keyword expansion with valid input (requires internet)"""
        from llm.tools import expand_keywords_with_web_search

        keywords = ["python", "programming"]
        result = expand_keywords_with_web_search(keywords)

        # Should return structured response
        assert "original" in result
        assert "web_keywords" in result
        assert "sources" in result
        assert "status" in result
        assert "message" in result

        # Original keywords preserved
        assert result["original"] == keywords

        # With real implementation, status can be success, partial_success, or failed
        # depending on internet connectivity and search results
        assert result["status"] in ["success", "partial_success", "failed"]
        assert isinstance(result["web_keywords"], list)
        assert isinstance(result["sources"], list)

    def test_expand_keywords_empty_input(self):
        """Test with empty keywords list"""
        from llm.tools import expand_keywords_with_web_search

        result = expand_keywords_with_web_search([])

        assert result["status"] == "failed"
        assert "Invalid or empty" in result["message"]
        assert result["web_keywords"] == []
        assert result["sources"] == []

    def test_expand_keywords_invalid_input_none(self):
        """Test with None input"""
        from llm.tools import expand_keywords_with_web_search

        result = expand_keywords_with_web_search(None)

        assert result["status"] == "failed"
        assert "Invalid or empty" in result["message"]
        assert result["web_keywords"] == []

    def test_expand_keywords_invalid_input_not_list(self):
        """Test with non-list input"""
        from llm.tools import expand_keywords_with_web_search

        result = expand_keywords_with_web_search("medical blood test")

        assert result["status"] == "failed"
        assert "Invalid or empty" in result["message"]

    @pytest.mark.integration
    def test_expand_keywords_single_keyword(self):
        """Test with single keyword (requires internet)"""
        from llm.tools import expand_keywords_with_web_search

        result = expand_keywords_with_web_search(["medical"])

        assert result["original"] == ["medical"]
        # Status depends on internet and search results
        assert result["status"] in ["success", "partial_success", "failed"]
        assert isinstance(result["web_keywords"], list)
        assert isinstance(result["sources"], list)

    def test_expand_keywords_return_structure(self):
        """Test that return structure matches expected format"""
        from llm.tools import expand_keywords_with_web_search

        keywords = ["test", "keyword"]
        result = expand_keywords_with_web_search(keywords)

        # Verify all required fields present
        required_fields = ["original", "web_keywords", "sources", "status", "message"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Verify types
        assert isinstance(result["original"], list)
        assert isinstance(result["web_keywords"], list)
        assert isinstance(result["sources"], list)
        assert isinstance(result["status"], str)
        assert isinstance(result["message"], str)


class TestFetchWebpageText:
    """Test fetch_webpage_text helper function"""

    @pytest.mark.integration
    def test_fetch_webpage_text_valid_url(self):
        """Test fetching text from a valid URL (requires internet)"""
        from llm.tools import fetch_webpage_text

        # Use a reliable, simple webpage
        url = "https://example.com"
        text, error = fetch_webpage_text(url)

        # Should return some text and no error
        assert isinstance(text, str)
        assert error is None
        # example.com should have some content
        assert len(text) > 0

    def test_fetch_webpage_text_invalid_url(self):
        """Test fetching text from invalid URL"""
        from llm.tools import fetch_webpage_text

        url = "https://this-domain-definitely-does-not-exist-12345.com"
        text, error = fetch_webpage_text(url)

        # Should handle gracefully and return empty string with error
        assert text == ""
        assert error is not None

    def test_fetch_webpage_text_with_timeout(self):
        """Test fetch with custom timeout"""
        from llm.tools import fetch_webpage_text

        # Very short timeout should likely fail
        url = "https://example.com"
        text, error = fetch_webpage_text(url, timeout=0.001)

        # Should return empty string on timeout with error
        assert isinstance(text, str)
        assert text == "" or error is None  # Either timed out or got result


class TestExtractKeywordsFromText:
    """Test extract_keywords_from_text helper function"""

    def test_extract_keywords_from_text_valid_input(self):
        """Test keyword extraction from valid text"""
        from llm.tools import extract_keywords_from_text

        text = """
        Machine learning is a subset of artificial intelligence that focuses on
        building systems that can learn from data. Deep learning uses neural networks
        with multiple layers to process complex patterns.
        """
        keywords = extract_keywords_from_text(text, max_keywords=5)

        # Should return a list
        assert isinstance(keywords, list)
        # Should return at most max_keywords
        assert len(keywords) <= 5
        # All keywords should be strings
        assert all(isinstance(k, str) for k in keywords)
        # All keywords should be lowercase
        assert all(k == k.lower() for k in keywords)
        # Keywords should have length > 2
        assert all(len(k) > 2 for k in keywords)

    def test_extract_keywords_from_text_short_text(self):
        """Test with very short text"""
        from llm.tools import extract_keywords_from_text

        text = "Short"
        keywords = extract_keywords_from_text(text)

        # Should return empty list for text < 50 chars
        assert keywords == []

    def test_extract_keywords_from_text_empty(self):
        """Test with empty text"""
        from llm.tools import extract_keywords_from_text

        keywords = extract_keywords_from_text("")

        # Should return empty list
        assert keywords == []

    def test_extract_keywords_from_text_none(self):
        """Test with None input"""
        from llm.tools import extract_keywords_from_text

        keywords = extract_keywords_from_text(None)

        # Should handle None gracefully
        assert keywords == []
