"""Pytest configuration and shared fixtures"""
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


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


@pytest.fixture
def e2e_test_cases():
    """
    Dynamically discover all PNG/TXT test case pairs in fixtures.

    Returns:
        List of tuples (png_path, txt_path) for parameterized tests
    """
    fixtures_dir = Path("tests/fixtures/sample_documents")
    test_cases = []

    for png_file in sorted(fixtures_dir.glob("*.png")):
        txt_file = png_file.with_suffix(".txt")
        if txt_file.exists():
            test_cases.append((str(png_file), str(txt_file)))

    return test_cases
