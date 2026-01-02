"""Tests for output validation"""
import pytest
from llm.extractor import LLMExtractor


def test_flag_hallucinated_content(sample_ocr_text):
    """Test detection of hallucinated content not in source text"""
    extractor = LLMExtractor()

    # Create fake result with hallucinated content
    hallucinated_result = {
        "authors": ["Dr. Nonexistent Person"],
        "date": "2025-01-01"
    }

    validation = extractor._validate_grounding(hallucinated_result, sample_ocr_text)

    # Should flag hallucinated content
    assert validation["valid_json"] is True
    assert len(validation["grounding_issues"]) > 0
    assert validation["confidence"] == "low"
    assert any("not found" in issue.lower() for issue in validation["grounding_issues"])


def test_valid_extraction_passes(sample_ocr_text):
    """Test that valid extractions pass grounding check"""
    extractor = LLMExtractor()

    # Create result with content that exists in OCR text
    # sample_ocr_text contains "UNIVERSITA' CATTOLICA", "GEMELLI", "Fernando", "2009"
    valid_result = {
        "keywords": ["Cattolica", "Gemelli", "Fernando"]
    }

    validation = extractor._validate_grounding(valid_result, sample_ocr_text)

    # Should pass with high confidence
    assert validation["valid_json"] is True
    assert len(validation["grounding_issues"]) == 0
    assert validation["confidence"] == "high"


def test_confidence_scoring(sample_ocr_text):
    """Test confidence scoring mechanism"""
    extractor = LLMExtractor()

    # Test high confidence (all values found)
    valid_result = {"keywords": ["Gemelli", "Fernando"]}
    validation_high = extractor._validate_grounding(valid_result, sample_ocr_text)
    assert validation_high["confidence"] == "high"

    # Test low confidence (values not found)
    invalid_result = {"keywords": ["NonexistentKeyword1", "NonexistentKeyword2"]}
    validation_low = extractor._validate_grounding(invalid_result, sample_ocr_text)
    assert validation_low["confidence"] == "low"


def test_grounding_with_null_values():
    """Test grounding validation handles null values correctly"""
    extractor = LLMExtractor()

    # Result with null values should pass
    null_result = {"authors": None, "date": None}
    validation = extractor._validate_grounding(null_result, "Some text")

    assert validation["valid_json"] is True
    assert len(validation["grounding_issues"]) == 0
    assert validation["confidence"] == "high"


def test_grounding_case_insensitive():
    """Test grounding validation is case-insensitive"""
    extractor = LLMExtractor()

    # Mixed case should still match
    text = "This is a Medical Report from Hospital"
    result = {"keywords": ["medical", "HOSPITAL"]}

    validation = extractor._validate_grounding(result, text)

    assert len(validation["grounding_issues"]) == 0
    assert validation["confidence"] == "high"


def test_grounding_with_nested_values():
    """Test grounding validation handles nested structures"""
    extractor = LLMExtractor()

    text = "Dr. Smith and Dr. Jones worked on this project in 2023"
    result = {
        "metadata": {
            "authors": ["Dr. Smith", "Dr. Jones"],
            "year": "2023"
        }
    }

    validation = extractor._validate_grounding(result, text)

    # All nested values should be checked
    assert validation["valid_json"] is True
    assert len(validation["grounding_issues"]) == 0
    assert validation["confidence"] == "high"
