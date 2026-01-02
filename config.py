"""Configuration settings for Document Extractor MVP"""

CONFIG = {
    "llm": {
        "model": "llama3",
        "temperature": 0.1,
        "max_tokens": 500
    },
    "ocr": {
        "engine": "easyocr",  # Options: easyocr, tesseract, paddleocr
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
