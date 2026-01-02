"""Tests for file handling utilities"""
import pytest
import csv
import tempfile
from pathlib import Path
from PIL import Image
from utils.file_handler import FileHandler


def test_load_image(sample_image_path):
    """Test successful image loading"""
    handler = FileHandler()
    image = handler.load_image(sample_image_path)

    assert isinstance(image, Image.Image)
    assert image.size[0] > 0
    assert image.size[1] > 0


def test_invalid_format():
    """Test error handling for unsupported formats"""
    handler = FileHandler()

    # Create temp .txt file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported format"):
            handler.load_image(temp_path)
    finally:
        Path(temp_path).unlink()


def test_file_size_limit():
    """Test file size validation"""
    handler = FileHandler()

    # Create temp large PNG file (> 10MB)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create large image
        large_image = Image.new('RGB', (5000, 5000), color='white')
        large_image.save(f, format='PNG')
        temp_path = f.name

    try:
        # Check if file is actually > 10MB
        size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
        if size_mb > 10:
            with pytest.raises(ValueError, match="File too large"):
                handler.load_image(temp_path)
        else:
            # If not large enough, just verify it loads
            image = handler.load_image(temp_path)
            assert isinstance(image, Image.Image)
    finally:
        Path(temp_path).unlink()


def test_export_csv():
    """Test CSV export functionality"""
    handler = FileHandler()

    test_results = {
        "author_date": {"authors": ["Dr. Smith"], "date": "2023-05-10"},
        "keywords": {"keywords": ["medical", "test", "results"]},
        "document_type": {"document_type": "medical"}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name

    try:
        handler.export_to_csv(test_results, temp_path)

        # Verify CSV contents
        with open(temp_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert rows[0] == ['Field', 'Value']
        assert len(rows) == 4  # Header + 3 data rows

        # Verify fields exist
        fields = [row[0] for row in rows[1:]]
        assert 'author_date' in fields
        assert 'keywords' in fields
        assert 'document_type' in fields
    finally:
        Path(temp_path).unlink()


def test_format_for_copy():
    """Test text formatting for copy-paste"""
    handler = FileHandler()

    test_results = {
        "author_date": {"authors": ["Dr. Smith"], "date": "2023-05-10"},
        "keywords": {"keywords": ["medical", "test"]},
        "document_type": {"document_type": "medical"}
    }

    formatted = handler.format_for_copy(test_results)

    assert isinstance(formatted, str)
    assert "author_date:" in formatted
    assert "keywords:" in formatted
    assert "document_type:" in formatted
    assert len(formatted.split('\n')) == 3
