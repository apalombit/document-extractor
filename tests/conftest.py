"""Pytest configuration and shared fixtures"""
import pytest
from pathlib import Path


@pytest.fixture
def sample_image_path():
    """Path to sample document for testing"""
    return "tests/fixtures/sample_documents/medical_report.png"


@pytest.fixture
def expected_outputs():
    """Load expected outputs for test documents"""
    # TODO: Load from expected_outputs.json
    return {
        "medical_report": {
            "authors": ["Dr. Example"],
            "date": "2009-07-09",
            "keywords": ["blood test", "hemoglobin", "medical"],
            "document_type": "medical"
        }
    }


@pytest.fixture
def sample_ocr_text():
    """Sample OCR text for unit testing"""
    return """UNIVERSITA' CATTOLICA DEL SACRO CUORE
POLICLINICO UNIVERSITARIO 'A. GEMELLI'
Paziente: Fernando
Data di nascita: 1969
Reparto: Spedalità Roma
Data referto: 09/07/2009"""
