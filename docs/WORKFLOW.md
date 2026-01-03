# Document Extraction Workflow

## Overview

The document extraction workflow is a 4-stage pipeline that processes document images and extracts structured information using OCR and LLM technology.

## Architecture

```
┌─────────────┐
│ Input Image │ (PNG file)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  File Validation    │  → Validates format, size limits
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  OCR Extraction     │  → Extracts raw text from image
└──────┬──────────────┘    (EasyOCR/Tesseract/PaddleOCR)
       │
       ▼
┌─────────────────────┐
│  LLM Extraction     │  → Extracts structured fields:
└──────┬──────────────┘    • author_date
       │                   • keywords
       │                   • document_type
       ▼
┌─────────────────────┐
│  Validation         │  → Checks grounding, confidence
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Structured Output  │  → JSON results + validation flags
└─────────────────────┘
```

## Pipeline Stages

### 1. File Validation
- **Input**: Image file path (string or file-like object)
- **Process**: `FileHandler.load_image()`
- **Validates**:
  - File format (PNG only in MVP)
  - File size (max 10MB)
  - Image can be loaded as PIL Image
- **Output**: PIL Image object
- **Error Handling**: Raises `ValueError` on invalid format/size

### 2. OCR Text Extraction
- **Input**: Image file path
- **Process**: `SelectedOCR.extract_text()`
- **Engine**: Configurable (EasyOCR by default)
- **Output**: Raw text string
- **Languages**: Supports English and Italian (configurable in `config.py`)
- **Error Handling**: Returns empty string on OCR failure

### 3. LLM Field Extraction
- **Input**: OCR text string
- **Process**: `LLMExtractor.extract_field()` called 3 times
- **LLM Model**: llama3:8b via Ollama (local)
- **Temperature**: 0.0 (deterministic)
- **Max Tokens**: 500

#### Extraction Tasks

**a) Author & Date (`author_date`)**
- Extracts document authors (persons or institutions)
- Extracts emission/writing date
- Returns: `{"authors": ["name1", "name2"], "date": "YYYY-MM-DD"}`
- Validation: Grounding check (values must appear in OCR text)

**b) Keywords (`keywords`)**
- Extracts 2-4 main content keywords
- Prioritizes technical/domain-specific terms
- Returns: `{"keywords": ["term1", "term2", ...]}`
- Validation: Grounding check (values must appear in OCR text)

**c) Document Type (`document_type`)**
- Classifies document into predefined types
- Valid types: medical, legal, invoice, receipt, contract, report, letter, form, other
- Returns: `{"document_type": "medical"}`
- Validation: Type validation (must be in VALID_DOCUMENT_TYPES)

### 4. Validation & Confidence Scoring
- **Process**: `LLMExtractor._validate_grounding()`
- **Strategy**:
  - **Extraction tasks** (author_date, keywords): Check if values exist in OCR text
  - **Classification tasks** (document_type): Check if value is in valid types set
- **Output**:
  ```python
  {
      "valid_json": True,
      "grounding_issues": ["issue1", "issue2"],
      "confidence": "high"|"low"
  }
  ```

## Using the Workflow Programmatically

### Basic Usage

```python
from pipeline.extraction_workflow import ExtractionWorkflow

# Initialize workflow
workflow = ExtractionWorkflow()

# Process a document
results = workflow.process_document("path/to/document.png")

# Access extraction results
print(results["author_date"])      # {"authors": [...], "date": "..."}
print(results["keywords"])         # {"keywords": [...]}
print(results["document_type"])    # {"document_type": "..."}

# Check validation
print(results["validation"]["author_date"]["confidence"])  # "high" or "low"

# Check for errors
if results["errors"]:
    print("Errors occurred:", results["errors"])
```

### Output Structure

```python
{
    "author_date": {
        "authors": ["Author Name"] or null,
        "date": "YYYY-MM-DD" or null
    },
    "keywords": {
        "keywords": ["keyword1", "keyword2", ...] or null
    },
    "document_type": {
        "document_type": "medical|legal|invoice|..." or null
    },
    "validation": {
        "author_date": {
            "valid_json": True,
            "grounding_issues": [],
            "confidence": "high"
        },
        "keywords": {...},
        "document_type": {...}
    },
    "ocr_text": "extracted text content...",
    "errors": []  # List of error messages if any occurred
}
```

### Error Handling

The workflow is designed to fail gracefully:

```python
# Non-existent file
results = workflow.process_document("missing.png")
assert len(results["errors"]) > 0
assert "File validation error" in results["errors"][0]

# Invalid format
results = workflow.process_document("document.txt")
assert len(results["errors"]) > 0

# Even on errors, structure is complete
assert all(key in results for key in ["author_date", "keywords", "document_type", "validation", "ocr_text", "errors"])
```

### Batch Processing

```python
from pathlib import Path

workflow = ExtractionWorkflow()
results_list = []

for image_path in Path("documents/").glob("*.png"):
    results = workflow.process_document(str(image_path))
    results_list.append({
        "file": image_path.name,
        "results": results
    })

# Save to JSON
import json
with open("batch_results.json", "w") as f:
    json.dump(results_list, f, indent=2)
```

## Configuration

Workflow behavior is controlled by `config.py`:

```python
CONFIG = {
    "llm": {
        "model": "llama3:8b",      # Ollama model
        "temperature": 0.0,         # Deterministic output
        "max_tokens": 500
    },
    "ocr": {
        "engine": "easyocr",        # easyocr|tesseract|paddleocr
        "languages": ["en", "it"],
        "confidence_threshold": 0.5
    },
    "file": {
        "supported_formats": [".png"],
        "max_size_mb": 10
    }
}
```

## Integration with Streamlit UI

The workflow is designed to be framework-agnostic. In the Streamlit UI (`app.py`), it can be used as:

```python
import streamlit as st
from pipeline.extraction_workflow import ExtractionWorkflow

if st.button("Analyze Document"):
    workflow = ExtractionWorkflow()

    with st.spinner("Processing..."):
        results = workflow.process_document(uploaded_file)

    # Display results
    st.json(results["author_date"])
    st.json(results["keywords"])
    st.json(results["document_type"])

    # Show validation
    for task in ["author_date", "keywords", "document_type"]:
        is_valid = results["validation"][task]["confidence"] == "high"
        st.checkbox("Validated", value=is_valid, disabled=True)
```

## Testing

### End-to-End Tests

The workflow is tested with automated e2e tests:

```bash
# Run e2e tests
pytest -m e2e

# Run specific test
pytest tests/test_e2e_pipeline.py::test_extraction_pipeline_accuracy -v

# Skip e2e tests (run unit tests only)
pytest -m "not e2e"
```

### Test Case Format

E2E tests use image/text pairs:
- `tests/fixtures/sample_documents/medical_report.png` (input image)
- `tests/fixtures/sample_documents/medical_report.txt` (expected output)

Expected output format:
```
author_date: {"authors": ["Name"], "date": "YYYY-MM-DD"}
keywords: {"keywords": ["term1", "term2"]}
document_type: {"document_type": "medical"}
```

### Adding New Test Cases

1. Add PNG image to `tests/fixtures/sample_documents/`
2. Create matching `.txt` file with expected outputs
3. Tests auto-discover new cases via `e2e_test_cases` fixture

## Performance

- **OCR**: ~2-10 seconds (depends on image size/complexity)
- **LLM Extraction**: ~15-30 seconds per task (3 tasks total)
- **Total Pipeline**: ~60-120 seconds per document
- **Throughput**: ~30-60 documents/hour (single thread)

## Limitations (MVP)

- Single document processing only (no batch API)
- PNG format only
- English and Italian languages only
- Local LLM only (Ollama required)
- No manual validation interface
- No retry logic on LLM failures

## Future Enhancements

- Batch processing API
- PDF/JPG support
- Multi-language expansion
- Cloud LLM providers (OpenAI, Anthropic)
- Async/parallel processing
- Retry logic with exponential backoff
- Caching for repeated documents
