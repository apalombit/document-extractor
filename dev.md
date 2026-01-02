# Document Extractor MVP - Development Plan

## Project Overview
**Goal:** Automated extraction of structured information from document images (PNG) to reduce manual transcription work.

**User:** Non-technical professionals (admins, doctors) processing 100+ documents/week.

**Core Flow:** Upload PNG → OCR → LLM extraction → Display/Export results

**Target Accuracy:** 99% (validated via test cases, not runtime detection)

---

## Tech Stack
```python
CONFIG = {
    "llm": {
        "model": "llama3",  # Via Ollama
        "temperature": 0.1,
        "max_tokens": 500
    },
    "ocr": {
        "engine": "easyocr",  # Modular: easyocr, tesseract, paddleocr
        "languages": ["en", "it"],
        "confidence_threshold": 0.5
    },
    "file": {
        "supported_formats": [".png"],
        "max_size_mb": 10
    },
    "output": {
        "format": "json",
        "validation_enabled": True
    }
}
```

**Framework:** Streamlit (Python), pytest (testing), local hosting only

---

## Project Structure
```
document-extractor/
├── app.py                      # Streamlit UI
├── config.py                   # Settings dict
├── requirements.txt
├── ocr/
│   ├── __init__.py
│   ├── base.py                # Abstract OCR interface
│   └── engine.py              # OCR implementation
├── llm/
│   ├── __init__.py
│   └── extractor.py           # Ollama + prompt logic
├── utils/
│   ├── __init__.py
│   └── file_handler.py        # I/O operations
└── tests/
    ├── conftest.py
    ├── test_ocr.py
    ├── test_llm.py
    ├── test_file_handler.py
    ├── test_validation.py
    └── fixtures/
        └── sample_documents/
```

---

## Development Phases

### Phase 1: Configuration & File Handling
**Module:** `config.py`, `utils/file_handler.py`

**Signatures:**
```python
# config.py
CONFIG: Dict  # Nested dict with llm, ocr, file, output settings

# utils/file_handler.py
class FileHandler:
    @staticmethod
    def load_image(file_path: str) -> Image:
        """Load and validate PNG. Raises ValueError for invalid format/size."""
        
    @staticmethod
    def export_to_csv(results: dict, output_path: str) -> None:
        """Flatten dict to CSV with Field, Value columns."""
        
    @staticmethod
    def format_for_copy(results: dict) -> str:
        """Format as 'key: value' text lines."""
```

**Tests:**
- `test_file_handler.py`: test_load_image, test_invalid_format, test_file_size_limit, test_export_csv

---

### Phase 2: OCR Interface
**Module:** `ocr/base.py`, `ocr/engine.py`

**Signatures:**
```python
# ocr/base.py
from abc import ABC, abstractmethod

class OCREngine(ABC):
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image. Returns raw text string."""

# ocr/engine.py
class SelectedOCR(OCREngine):
    def __init__(self):
        """Initialize chosen OCR library (EasyOCR/Tesseract/PaddleOCR)."""
        
    def extract_text(self, image_path: str) -> str:
        """Implementation of text extraction."""
```

**Assumptions:**
- Modular design allows swapping OCR engines
- Returns raw text, no preprocessing

**Tests:**
- `test_ocr.py`: test_extract_text_success, test_extract_text_invalid_file, test_ocr_accuracy_threshold

---

### Phase 3: LLM Extraction
**Module:** `llm/extractor.py`

**Signatures:**
```python
class LLMExtractor:
    def __init__(self):
        """Load model config and prompt templates."""
        self.model: str
        self.temperature: float
        self.system_prompt: str
        self.task_prompts: Dict[str, str]
    
    def extract_field(self, ocr_text: str, task: str) -> Tuple[Dict, Dict]:
        """
        Extract field from OCR text using LLM.
        
        Args:
            ocr_text: Raw OCR output
            task: "author_date" | "keywords" | "document_type"
        
        Returns:
            (extraction_result, validation_flags)
            - extraction_result: {"authors": [...], "date": "..."} or null
            - validation_flags: {"valid_json": bool, "grounding_issues": [...], "confidence": "high"|"low"}
        """
    
    def _validate_grounding(self, result: Dict, ocr_text: str) -> Dict:
        """Check extracted values exist in source text. Returns validation flags dict."""
```

**Prompt Strategy:**
- Single question at a time (optimal for 3-8B models)
- System prompt: OCR text + strict rules (only use provided text, output JSON, null if uncertain)
- User prompts: Task-specific with examples and JSON schema
- Few-shot examples per task

**Extraction Tasks:**
1. **author_date**: `{"authors": ["string"], "date": "YYYY-MM-DD"}`
2. **keywords**: `{"keywords": ["string"]}` (3-5 keywords)
3. **document_type**: `{"document_type": "medical|legal|invoice|..."}` 
4. **html** (placeholder): `{"html": null, "status": "not_implemented"}`

**Validation:**
- Flag only, don't reject output
- Check: valid JSON, values exist in OCR text, confidence scoring

**Tool Use (Placeholder):**
```python
tools = [
    {"name": "validate_date", "parameters": {"date_string": "string"}},
    {"name": "compare_html_to_image", "parameters": {"html": "string", "image_path": "string"}}
]
```

**Tests:**
- `test_llm.py`: test_extract_author_date, test_extract_keywords, test_extract_document_type, test_json_output_format, test_null_when_info_missing
- `test_validation.py`: test_flag_hallucinated_content, test_valid_extraction_passes, test_confidence_scoring

---

### Phase 4: Streamlit UI
**Module:** `app.py`

**Flow:**
1. File uploader (PNG only)
2. Image preview thumbnail
3. "Analyze Document" button
4. OCR processing with spinner
5. Show raw OCR text (expandable)
6. Progressive LLM extraction (3 tasks, each with spinner)
7. Display results as JSON cards with validation checkboxes (disabled, show confidence)
8. Export: "Copy Text" and "Download CSV" buttons
9. Error tab (expandable) for debugging

**Key Features:**
- Single column sequential layout
- Results appear progressively as LLM completes each task
- Validation shown as disabled checkbox per field
- Session state for error tracking

**Error Handling:**
- Try/catch per extraction task
- All errors collected in `st.session_state.errors`
- Display in expandable "⚠️ Errors" section

**Tests:**
- Manual UI testing (Streamlit limitation)

---

## Key Assumptions

1. **MVP Scope:**
   - Single PNG only (no batch)
   - No manual validation interface
   - Local deployment only
   - Trust automation (validation flags informational)

2. **OCR:**
   - Modular design, implementation TBD
   - Returns raw text string

3. **LLM:**
   - Local Ollama with Llama3 (3-8B)
   - One question at a time (better for small models)
   - JSON output only
   - Null for missing information (no hallucination)

4. **Validation:**
   - Flags grounding issues but doesn't reject
   - 99% accuracy target validated via test cases, not runtime

5. **HTML Generation:**
   - Stretch goal, placeholder only in MVP
   - Tool use structure prepared for future

---

## Test Data Requirements
- `tests/fixtures/sample_documents/`: PNG files (medical report, invoice, etc.)
- `expected_outputs.json`: Ground truth for each test document

---

## Next Steps for Implementation
1. Set up `requirements.txt` (streamlit, pillow, ollama, easyocr, pytest)
2. Implement Phase 1 → run tests
3. Implement Phase 2 → run tests
4. Implement Phase 3 → run tests
5. Implement Phase 4 → manual testing
6. Integration testing with real documents
