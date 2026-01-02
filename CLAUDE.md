# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Document Extractor MVP - automated extraction of structured information from document images (PNG). Built for non-technical users to process 100+ documents/week with 99% accuracy target.

**Core Flow:** Upload PNG → OCR → LLM extraction → Display/Export results

**Tech Stack:** Python/Streamlit UI, EasyOCR for text extraction, local Ollama/Llama3 for field extraction

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install and start Ollama with Llama3 (required for LLM extraction)
ollama pull llama3
ollama serve
```

### Running
```bash
# Start Streamlit app
streamlit run app.py
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ocr.py

# Run specific test
pytest tests/test_llm.py::test_extract_author_date
```

## Architecture

### Modular Pipeline Design
The application follows a strict separation of concerns with three main processing stages:

1. **File Handling** (`utils/file_handler.py`)
   - Validates PNG format and size limits (10MB max)
   - Exports to CSV or copyable text format

2. **OCR Layer** (`ocr/`)
   - `base.py`: Abstract `OCREngine` interface for swappable implementations
   - `engine.py`: Concrete implementation using EasyOCR (configurable: tesseract, paddleocr)
   - Returns raw text string with no preprocessing

3. **LLM Extraction** (`llm/extractor.py`)
   - Single-question-at-a-time approach (optimal for 3-8B models)
   - Three extraction tasks run sequentially:
     - `author_date`: Returns `{"authors": [...], "date": "YYYY-MM-DD"}`
     - `keywords`: Returns `{"keywords": [...]}` (3-5 keywords)
     - `document_type`: Returns `{"document_type": "medical|legal|invoice|..."}`
   - **Grounding validation**: `_validate_grounding()` checks extracted values exist in OCR text
   - Flags issues but doesn't reject output (informational only)
   - Returns null for missing information (never hallucinates)

### Configuration
All settings centralized in `config.py` as nested dict:
- LLM: model (llama3), temperature (0.1), max_tokens (500)
- OCR: engine selection, languages ([en, it]), confidence threshold
- File: supported formats, size limits
- Output: format, validation toggle

### Streamlit UI Flow (`app.py`)
Sequential single-column layout:
1. File upload → image preview
2. "Analyze Document" button triggers OCR
3. Raw OCR text shown in expandable section
4. Progressive LLM extraction (3 spinners, results appear as completed)
5. Validation checkboxes (disabled, show confidence level)
6. Export: "Copy Text" and "Download CSV" buttons
7. Error tracking: All exceptions collected in `st.session_state.errors`, displayed in expandable "⚠️ Errors" section

## Key Implementation Details

### LLM Prompting Strategy
- **System prompt** (in `_load_system_prompt()`): Contains OCR text + strict rules ("use only provided text", "output JSON", "null if uncertain")
- **Task prompts** (in `_load_task_prompts()`): Task-specific with few-shot examples and JSON schemas
- Each task runs independently with full system context

### Validation Approach
Runtime validation flags grounding issues but doesn't reject output. 99% accuracy target validated through test cases (see `tests/fixtures/sample_documents/` and `expected_outputs.json`), not runtime detection.

### Stretch Goal: HTML Generation
Placeholder in `extract_html()` with tool use structure prepared:
- `validate_date`: Date format validation/normalization
- `compare_html_to_image`: Visual similarity scoring for generated HTML

## Testing Strategy
- Phase 1: `test_file_handler.py` - format validation, size limits, CSV export
- Phase 2: `test_ocr.py` - text extraction, accuracy thresholds
- Phase 3: `test_llm.py` + `test_validation.py` - field extraction, JSON output, grounding validation, confidence scoring
- Phase 4: Manual UI testing (Streamlit limitation)

Integration tests require real documents in `tests/fixtures/sample_documents/` with ground truth in `expected_outputs.json`.

## MVP Scope Constraints
- Single PNG only (no batch processing)
- No manual validation interface
- Local deployment only (no web hosting)
- Trust automation (validation flags informational)
- HTML generation is post-MVP stretch goal
