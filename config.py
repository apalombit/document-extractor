"""Configuration settings for Document Extractor MVP"""

CONFIG = {
    "llm": {
        "model": "llama3.2:3b",
        "temperature": 0.0,
        "max_tokens": 500,
        "max_conversation_turns": 5,   # Limit for multi-turn loops
        "enable_critique": True,       # Enable self-critique step
        "enable_web_keyword_expansion": True,  # Enable web-based keyword expansion
        "web_search_timeout": 10,        # Timeout in seconds per URL fetch
        "web_search_max_results": 5,    # Maximum number of search results to process
        "web_search_max_text_length": 2000,  # Maximum characters to extract from webpage
        "web_search_region": "it-it",   # DuckDuckGo region (it-it for Italian, us-en for US English)
        "web_search_blocked_domains": ["zhihu.com", "baidu.com", "weibo.com", "qq.com", "csdn.net"]
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
