"""Tests for output validation"""
import pytest
from llm.extractor import LLMExtractor


# TODO: Test grounding validation
def test_flag_hallucinated_content(sample_ocr_text):
    """Test detection of hallucinated content not in source text"""
    pass


def test_valid_extraction_passes(sample_ocr_text):
    """Test that valid extractions pass grounding check"""
    pass


def test_confidence_scoring(sample_ocr_text):
    """Test confidence scoring mechanism"""
    pass
