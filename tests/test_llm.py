"""Tests for LLM extraction module"""
import pytest
from llm.extractor import LLMExtractor


# TODO: Test LLM extraction
def test_extract_author_date(sample_ocr_text):
    """Test extraction of author names and dates"""
    pass


def test_extract_keywords(sample_ocr_text):
    """Test keyword extraction"""
    pass


def test_extract_document_type(sample_ocr_text):
    """Test document type classification"""
    pass


def test_json_output_format(sample_ocr_text):
    """Test LLM returns valid JSON format"""
    pass


def test_null_when_info_missing():
    """Test LLM returns null when information not present"""
    pass
