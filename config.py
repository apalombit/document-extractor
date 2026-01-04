"""Configuration settings for Document Extractor MVP"""

CONFIG = {
    "llm": { 
        "model": "llama3.2:3b",
        "temperature": 0.0,
        "max_tokens": 500,
        "max_conversation_turns": 3,  # Limit for multi-turn loops
        "enable_critique": False       # Enable self-critique step
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
