"""Tests for OCR module"""
import pytest
import tempfile
from pathlib import Path
from PIL import Image
from ocr.engine import SelectedOCR
from ocr.base import OCREngine
from config import CONFIG


def test_extract_text_success(sample_image_path):
    """Test successful text extraction from valid image"""
    ocr = SelectedOCR()
    text = ocr.extract_text(sample_image_path)

    assert isinstance(text, str)
    assert len(text) > 0
    # Verify some expected content from medical report
    assert any(keyword in text.lower() for keyword in ['universita', 'cattolica', 'gemelli', 'fernando'])


def test_extract_text_invalid_file():
    """Test error handling for invalid file format"""
    ocr = SelectedOCR()

    # Create temp non-image file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test")
        temp_path = f.name

    try:
        with pytest.raises(Exception):  # Should raise error for invalid file
            ocr.extract_text(temp_path)
    finally:
        Path(temp_path).unlink()


def test_ocr_accuracy_threshold(sample_image_path, sample_ocr_text):
    """Test OCR accuracy meets minimum threshold"""
    ocr = SelectedOCR()
    extracted_text = ocr.extract_text(sample_image_path)

    # Check that key terms from expected output appear in extracted text
    key_terms = ['cattolica', 'gemelli', 'fernando', '2009']

    # Count how many key terms are found (case-insensitive)
    found_terms = sum(1 for term in key_terms if term in extracted_text.lower())

    # At least 50% of key terms should be found
    accuracy_ratio = found_terms / len(key_terms)
    assert accuracy_ratio >= 0.5, f"OCR accuracy too low: {accuracy_ratio:.1%} (found {found_terms}/{len(key_terms)} key terms)"


def test_config_integration():
    """Test OCR uses config for engine and languages"""
    ocr = SelectedOCR()

    # Verify config values are loaded
    assert ocr.engine == CONFIG["ocr"]["engine"]
    assert ocr.languages == CONFIG["ocr"]["languages"]
    assert ocr.engine == "easyocr"
    assert "en" in ocr.languages
    assert "it" in ocr.languages


def test_unsupported_engine():
    """Test error handling for unsupported OCR engine"""
    # Temporarily modify config
    original_engine = CONFIG["ocr"]["engine"]

    try:
        CONFIG["ocr"]["engine"] = "tesseract"
        with pytest.raises(ValueError, match="not implemented"):
            SelectedOCR()
    finally:
        CONFIG["ocr"]["engine"] = original_engine


def test_ocr_engine_interface():
    """Test SelectedOCR implements OCREngine abstract base class"""
    ocr = SelectedOCR()

    # Verify it's an instance of abstract base class
    assert isinstance(ocr, OCREngine)

    # Verify abstract method is implemented
    assert hasattr(ocr, 'extract_text')
    assert callable(ocr.extract_text)


def test_empty_image():
    """Test OCR handling of blank/empty image"""
    ocr = SelectedOCR()

    # Create blank white image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        blank_image = Image.new('RGB', (200, 200), color='white')
        blank_image.save(f, format='PNG')
        temp_path = f.name

    try:
        text = ocr.extract_text(temp_path)
        # Should return empty string or minimal content for blank image
        assert isinstance(text, str)
        assert len(text) < 10  # Very minimal or no text
    finally:
        Path(temp_path).unlink()
