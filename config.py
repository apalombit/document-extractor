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
        "web_search_blocked_domains": ["zhihu.com", "baidu.com", "weibo.com", "qq.com", "csdn.net"],
        # PII guard settings for web search
        "pii_guard_enabled": True,           # Filter PII from keywords before web search
        "pii_guard_threshold": 0.5,          # Minimum confidence to flag as PII
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
    },
    "chat": {
        "model": "llama3.2:3b",
        "temperature": 0.3,          # Slightly more creative than extraction
        "max_tokens": 1000,          # Longer responses for explanations
        "max_history_turns": 10,     # Limit conversation memory
        "use_rag": True,             # Enable RAG retrieval
        "rag_method": "embedding",   # "keyword" or "embedding"
        "rag_chunk_size": 200,       # Characters per chunk
        "rag_top_k": 3,              # Number of chunks to retrieve
        "rag_embedding_model": "all-MiniLM-L6-v2",  # SentenceTransformer model
        "rag_similarity_threshold": 0.3,  # Minimum cosine similarity for results
        # Guardrail settings
        "use_guardrail": True,           # Enable on-topic guardrail
        "guardrail_model": "facebook/bart-large-mnli",  # Zero-shot classifier
        "guardrail_threshold": 0.3,      # Minimum score to be considered on-topic
        # Jailbreak guard settings
        "use_jailbreak_guard": True,
        "jailbreak_model": "deepset/deberta-v3-base-injection",
        "jailbreak_threshold": 0.7,      # Higher threshold to reduce false positives
        # Hallucination guard settings
        "use_hallucination_guard": True,
        "hallucination_nli_model": "GuardrailsAI/finetuned_nli_provenance",
        "hallucination_threshold": 0.4,  # Minimum entailment score (lowered for leniency)
        "hallucination_top_k": 3,        # Number of source chunks to check per sentence
        "hallucination_similarity_threshold": 0.25,  # Minimum similarity to consider source relevant
        "hallucination_lenient": True,   # If True, show response with warnings instead of blocking
    }
}
