# Document Extractor MVP

Automated extraction of structured information from document images (PNG).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install and start Ollama with Llama3:
```bash
ollama pull llama3
ollama serve
```

3. Run the app:
```bash
streamlit run app.py
```

## Development

See `dev.md` for complete development plan and implementation phases.

Run tests:
```bash
pytest tests/
```

## Project Structure

- `app.py` - Streamlit UI
- `config.py` - Configuration settings
- `ocr/` - OCR engine interface and implementation
- `llm/` - LLM-based extraction logic
- `utils/` - File handling utilities
- `tests/` - Test suite

## MVP Features

- Upload PNG document
- OCR text extraction
- LLM-based field extraction:
  - Author(s) and date
  - Content keywords
  - Document type classification
- Export to CSV or copyable text
- Validation flagging
