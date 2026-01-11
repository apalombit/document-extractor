# Document Extractor

Automated extraction of structured information from document images (PNG).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install and start Ollama with Llama3:
```bash
ollama pull llama3:8b
ollama serve
```

3. Run the app:
```bash
streamlit run app.py
```

## Testing

### Run All Tests
```bash
# Run all tests (unit + e2e)
pytest tests/

# Run with verbose output
pytest tests/ -v
```

### Run Specific Test Types
```bash
# Run only unit tests (fast)
pytest -m "not e2e"

# Run only e2e tests (slow, requires Ollama)
pytest -m e2e

# Run specific test file
pytest tests/test_e2e_pipeline.py -v
```

### End-to-End Testing

E2E tests validate the complete extraction workflow:
```bash
pytest tests/test_e2e_pipeline.py -v
```

**Test Coverage:**
- ✅ Full pipeline accuracy (image → OCR → LLM → validation)
- ✅ Validation confidence scoring
- ✅ Error handling (invalid files, missing data)
- ✅ OCR text extraction
- ✅ All extraction tasks completion

**Adding New Test Cases:**
1. Add PNG image to `tests/fixtures/sample_documents/yourfile.png`
2. Create `tests/fixtures/sample_documents/yourfile.txt` with expected outputs:
   ```
   author_date: {"authors": ["Name"], "date": "YYYY-MM-DD"}
   keywords: {"keywords": ["term1", "term2"]}
   document_type: {"document_type": "medical"}
   ```
3. Tests auto-discover new cases automatically

**Note:** E2E tests require Ollama running (`ollama serve`) and take ~3-5 minutes to complete.

## Project Structure

- `app.py` - Streamlit UI
- `config.py` - Configuration settings
- `pipeline/` - **Extraction workflow orchestrator**
- `ocr/` - OCR engine interface and implementation
- `llm/` - LLM-based extraction logic
- `utils/` - File handling utilities
- `tests/` - Test suite (unit + e2e)
- `docs/` - Documentation

## Programmatic Usage

The extraction workflow can be used independently of the UI:

```python
from pipeline.extraction_workflow import ExtractionWorkflow

# Initialize workflow
workflow = ExtractionWorkflow()

# Process document
results = workflow.process_document("path/to/document.png")

# Access results
print(results["author_date"])      # {"authors": [...], "date": "..."}
print(results["keywords"])         # {"keywords": [...]}
print(results["document_type"])    # {"document_type": "..."}

# Check validation
print(results["validation"]["author_date"]["confidence"])  # "high" or "low"
```

See `docs/WORKFLOW.md` for detailed documentation.

## Features

- Upload PNG document
- OCR text extraction (EasyOCR)
- LLM-based field extraction (Llama3):
  - Author(s) and date
  - Content keywords
  - Document type classification
- Validation & confidence scoring
- Export to CSV or copyable text
- Programmatic API for automation
- Chat with document (RAG-powered Q&A)
- On-topic guardrail for chat queries
