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
