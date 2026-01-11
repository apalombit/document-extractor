"""Tests for PII guard module."""
import pytest
from unittest.mock import Mock, patch
from llm.pii_guard import PIIGuard
from config import CONFIG


class TestPIIGuard:
    """Tests for PIIGuard class."""

    def test_init_with_default_entities(self):
        """Test PIIGuard initializes with default entity types."""
        guard = PIIGuard()
        assert "PERSON" in guard.entities
        assert "PHONE_NUMBER" in guard.entities
        assert "EMAIL_ADDRESS" in guard.entities

    def test_init_with_custom_entities(self):
        """Test PIIGuard initializes with custom entity types."""
        guard = PIIGuard(entities=["PERSON", "EMAIL_ADDRESS"])
        assert guard.entities == ["PERSON", "EMAIL_ADDRESS"]

    def test_detect_pii_empty_text(self):
        """Test detect_pii with empty text returns empty list."""
        guard = PIIGuard()
        assert guard.detect_pii("") == []
        assert guard.detect_pii("   ") == []
        assert guard.detect_pii(None) == []

    def test_detect_pii_no_pii(self):
        """Test detect_pii with clean text."""
        guard = PIIGuard()
        result = guard.detect_pii("medical diagnosis treatment")
        assert len(result) == 0

    def test_detect_pii_with_person_name(self):
        """Test detect_pii detects person names."""
        guard = PIIGuard()
        # Use a clear two-word name pattern
        result = guard.detect_pii("Contact John Smith for details")

        assert len(result) > 0
        assert any(r["entity_type"] == "PERSON" for r in result)

    def test_detect_pii_with_phone_number(self):
        """Test detect_pii detects phone numbers."""
        guard = PIIGuard()
        result = guard.detect_pii("Call 555-123-4567")

        assert len(result) > 0
        assert any(r["entity_type"] == "PHONE_NUMBER" for r in result)

    def test_detect_pii_with_email(self):
        """Test detect_pii detects email addresses."""
        guard = PIIGuard()
        result = guard.detect_pii("Email john.smith@example.com")

        assert len(result) > 0
        assert any(r["entity_type"] == "EMAIL_ADDRESS" for r in result)

    def test_contains_pii_true(self):
        """Test contains_pii returns True for PII text."""
        guard = PIIGuard()
        # Use clear name pattern
        assert guard.contains_pii("Contact John Smith") is True

    def test_contains_pii_false(self):
        """Test contains_pii returns False for clean text."""
        guard = PIIGuard()
        assert guard.contains_pii("medical diagnosis") is False

    def test_contains_pii_with_email(self):
        """Test contains_pii detects email."""
        guard = PIIGuard()
        assert guard.contains_pii("email@example.com") is True

    def test_filter_keywords_no_pii(self):
        """Test filter_keywords with clean keywords."""
        guard = PIIGuard()
        keywords = ["medical", "diagnosis", "treatment", "healthcare"]

        safe, filtered, details = guard.filter_keywords(keywords)

        assert safe == keywords
        assert filtered == []
        assert details == []

    def test_filter_keywords_with_pii(self):
        """Test filter_keywords removes PII-containing keywords."""
        guard = PIIGuard()
        keywords = ["medical", "John Smith", "diagnosis", "555-123-4567"]

        safe, filtered, details = guard.filter_keywords(keywords)

        assert "medical" in safe
        assert "diagnosis" in safe
        assert "John Smith" in filtered
        assert "555-123-4567" in filtered
        assert len(details) >= 2

    def test_filter_keywords_with_email(self):
        """Test filter_keywords removes email addresses."""
        guard = PIIGuard()
        keywords = ["medical", "jane.doe@email.com", "diagnosis"]

        safe, filtered, details = guard.filter_keywords(keywords)

        assert "medical" in safe
        assert "diagnosis" in safe
        assert "jane.doe@email.com" in filtered

    def test_filter_keywords_stores_debug_info(self):
        """Test filter_keywords stores debug info."""
        guard = PIIGuard()
        keywords = ["medical", "John Smith"]

        guard.filter_keywords(keywords)

        assert guard.last_input == keywords
        assert "John Smith" in guard.last_filtered
        assert guard.last_pii_found is not None

    def test_filter_keywords_empty_list(self):
        """Test filter_keywords with empty list."""
        guard = PIIGuard()
        safe, filtered, details = guard.filter_keywords([])

        assert safe == []
        assert filtered == []
        assert details == []

    def test_filter_keywords_skips_empty_strings(self):
        """Test filter_keywords skips empty strings."""
        guard = PIIGuard()
        keywords = ["medical", "", "  ", "diagnosis"]

        safe, filtered, details = guard.filter_keywords(keywords)

        assert "" not in safe
        assert "  " not in safe
        assert "medical" in safe
        assert "diagnosis" in safe

    def test_get_debug_info(self):
        """Test get_debug_info returns expected structure."""
        guard = PIIGuard()
        guard.filter_keywords(["medical", "John Smith"])

        debug = guard.get_debug_info()

        assert "entities_checked" in debug
        assert "last_input" in debug
        assert "last_filtered" in debug
        assert "last_pii_found" in debug


class TestPIIGuardIntegration:
    """Integration tests for PII guard with web search tool."""

    def test_web_search_filters_pii_keywords(self):
        """Test that web search tool filters PII from keywords."""
        from llm.tools import expand_keywords_with_web_search

        # Temporarily enable PII guard
        original = CONFIG["llm"].get("pii_guard_enabled")
        CONFIG["llm"]["pii_guard_enabled"] = True

        try:
            # Keywords with PII
            result = expand_keywords_with_web_search([
                "medical",
                "John Smith",  # Person name - should be filtered
                "diagnosis"
            ])

            # Should have filtered the name
            if "pii_filtered" in result:
                assert "John Smith" in result["pii_filtered"]

            # Query should not contain the name
            if result.get("debug_info", {}).get("query_used"):
                assert "John Smith" not in result["debug_info"]["query_used"]

        finally:
            if original is not None:
                CONFIG["llm"]["pii_guard_enabled"] = original

    def test_web_search_blocks_all_pii_keywords(self):
        """Test that web search blocks when all keywords are PII."""
        from llm.tools import expand_keywords_with_web_search

        # Temporarily enable PII guard
        original = CONFIG["llm"].get("pii_guard_enabled")
        CONFIG["llm"]["pii_guard_enabled"] = True

        try:
            # All keywords are PII
            result = expand_keywords_with_web_search([
                "John Smith",
                "jane.doe@email.com"
            ])

            assert result["status"] == "blocked"
            assert "PII" in result["message"]

        finally:
            if original is not None:
                CONFIG["llm"]["pii_guard_enabled"] = original

    def test_web_search_respects_pii_guard_disabled(self):
        """Test that web search skips PII check when disabled."""
        from llm.tools import expand_keywords_with_web_search

        # Temporarily disable PII guard
        original = CONFIG["llm"].get("pii_guard_enabled")
        CONFIG["llm"]["pii_guard_enabled"] = False

        try:
            result = expand_keywords_with_web_search(["John Smith"])

            # Should not have pii_filtered key when guard is disabled
            assert "pii_filtered" not in result or result.get("pii_filtered") == []

        finally:
            if original is not None:
                CONFIG["llm"]["pii_guard_enabled"] = original


class TestPIIEntityTypes:
    """Tests for different PII entity types."""

    @pytest.mark.parametrize("text,entity_type", [
        ("Contact John Smith please", "PERSON"),
        ("Dr Sarah Johnson called", "PERSON"),
        ("Call 555-123-4567", "PHONE_NUMBER"),
        ("Phone: 555.123.4567", "PHONE_NUMBER"),
        ("Email john@example.com", "EMAIL_ADDRESS"),
        ("Send to test.user@company.org", "EMAIL_ADDRESS"),
    ])
    def test_detect_common_pii_types(self, text, entity_type):
        """Test detection of common PII types."""
        guard = PIIGuard()
        result = guard.detect_pii(text)

        detected_types = [r["entity_type"] for r in result]
        assert entity_type in detected_types, f"Expected {entity_type} in '{text}', got {detected_types}"

    def test_detect_credit_card(self):
        """Test credit card detection."""
        guard = PIIGuard()
        result = guard.detect_pii("Card: 4111-1111-1111-1111")

        assert len(result) > 0
        assert any(r["entity_type"] == "CREDIT_CARD" for r in result)

    def test_detect_ip_address(self):
        """Test IP address detection."""
        guard = PIIGuard()
        result = guard.detect_pii("Server at 192.168.1.1")

        assert len(result) > 0
        assert any(r["entity_type"] == "IP_ADDRESS" for r in result)
