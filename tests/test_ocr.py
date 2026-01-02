"""Tests for OCR module"""
import pytest
from ocr.engine import SelectedOCR


# TODO: Test OCR engine
def test_extract_text_success(sample_image_path):
    """Test successful text extraction from valid image"""
    pass


def test_extract_text_invalid_file():
    """Test error handling for invalid file format"""
    pass


def test_ocr_accuracy_threshold(sample_image_path, expected_outputs):
    """Test OCR accuracy meets minimum threshold"""
    pass
