"""LLM tools for multi-turn conversations"""
from datetime import datetime
from typing import Dict
from dateutil import parser
from dateutil.parser import ParserError


def validate_date(date_string: str) -> Dict:
    """
    Validate that a date exists and is not in the future.

    This tool parses common date formats and verifies the date is valid
    and not in the future. Used by LLM during extraction to validate dates.

    Args:
        date_string: Date string to validate (e.g., "2024-03-15", "March 2023", "15/03/2024")

    Returns:
        Dict with validation result:
        - valid (bool): True if date is valid and not in future
        - reason (str): Explanation if invalid, empty string if valid
        - normalized (str): Standardized date format (YYYY-MM-DD) if parseable, empty if not

    Examples:
        >>> validate_date("2024-03-15")
        {"valid": True, "reason": "", "normalized": "2024-03-15"}

        >>> validate_date("2099-01-01")
        {"valid": False, "reason": "Date is in the future", "normalized": "2099-01-01"}

        >>> validate_date("February 30, 2024")
        {"valid": False, "reason": "Could not parse date format", "normalized": ""}
    """
    if not date_string or not isinstance(date_string, str):
        return {
            "valid": False,
            "reason": "Date string is empty or invalid",
            "normalized": ""
        }

    # Strip whitespace
    date_string = date_string.strip()

    try:
        # Detect if date is already in ISO format (YYYY-MM-DD)
        # If so, don't use dayfirst to avoid misinterpretation
        import re
        iso_format = re.match(r'^\d{4}-\d{2}-\d{2}$', date_string)

        if iso_format:
            # Already in ISO format, parse without dayfirst
            parsed_date = parser.parse(date_string, fuzzy=True)
        else:
            # Parse with dayfirst=True for European/Italian DD/MM/YYYY format
            parsed_date = parser.parse(date_string, fuzzy=True, dayfirst=True)

        # Normalize to YYYY-MM-DD format
        normalized = parsed_date.strftime("%Y-%m-%d")

        # Get current date/time
        now = datetime.now()

        # Check if date is in the future
        if parsed_date > now:
            return {
                "valid": False,
                "reason": "Date is in the future",
                "normalized": normalized
            }

        # Date is valid
        return {
            "valid": True,
            "reason": "",
            "normalized": normalized
        }

    except ParserError:
        # Could not parse the date
        return {
            "valid": False,
            "reason": "Could not parse date format",
            "normalized": ""
        }
    except Exception as e:
        # Other errors (e.g., invalid date like Feb 30)
        return {
            "valid": False,
            "reason": f"Invalid date: {str(e)}",
            "normalized": ""
        }


def extract_urls_from_search_results(search_results: str, limit: int = 5) -> list:
    """
    Parse markdown links from WebSearch results.

    Args:
        search_results: Search results string containing markdown links
        limit: Maximum number of URLs to return

    Returns:
        List of URLs extracted from search results

    Examples:
        >>> extract_urls_from_search_results("[Title](https://example.com)")
        ['https://example.com']
    """
    import re
    # Match markdown links: [text](url)
    pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    matches = re.findall(pattern, search_results)
    urls = [url for _, url in matches]
    return urls[:limit]


def fetch_webpage_text(url: str, timeout: int = None) -> str:
    """
    Fetch and extract text from a webpage.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds (uses config if not specified)

    Returns:
        Cleaned text content (max length from config)

    Examples:
        >>> text = fetch_webpage_text("https://example.com")
        >>> isinstance(text, str)
        True
    """
    import logging
    import requests
    from bs4 import BeautifulSoup
    from config import CONFIG

    if timeout is None:
        timeout = CONFIG["llm"]["web_search_timeout"]

    max_length = CONFIG["llm"]["web_search_max_text_length"]

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; DocumentExtractor/1.0)'}
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator=' ', strip=True)

        # Clean and limit length
        text = ' '.join(text.split())  # Normalize whitespace
        return text[:max_length]

    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {str(e)}")
        return ""


def extract_keywords_from_text(text: str, max_keywords: int = 5) -> list:
    """
    Extract keywords from text using Ollama LLM.

    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords to extract

    Returns:
        List of extracted keywords

    Examples:
        >>> keywords = extract_keywords_from_text("This is about machine learning and AI")
        >>> isinstance(keywords, list)
        True
    """
    import logging
    import ollama
    from config import CONFIG

    if not text or len(text) < 50:
        return []

    try:
        prompt = f"""Extract {max_keywords} main keywords or key concepts from this text.
Focus on specific, meaningful terms.
Return ONLY a comma-separated list of keywords.

Text: {text[:1000]}

Keywords:"""

        response = ollama.generate(
            model=CONFIG["llm"]["model"],
            prompt=prompt,
            options={"temperature": 0.0}
        )

        # Parse response
        keywords_text = response['response'].strip()
        keywords = [k.strip().lower() for k in keywords_text.split(',')]

        # Clean and filter
        keywords = [k for k in keywords if k and len(k) > 2]

        return keywords[:max_keywords]

    except Exception as e:
        logging.warning(f"Failed to extract keywords from text: {str(e)}")
        return []


def expand_keywords_with_web_search(keywords: list) -> Dict:
    """
    Expand keywords by searching web with DuckDuckGo and analyzing related content.

    This tool performs a web search using DuckDuckGo, fetches the top search results,
    and extracts keywords from each page using Ollama LLM analysis. The extracted
    keywords are consolidated and returned to help validate and expand the original
    keyword list.

    Args:
        keywords: Initial list of keywords from document

    Returns:
        Dict with:
            - original (list): List of original input keywords
            - web_keywords (list): Keywords found from web sources (deduplicated)
            - sources (list): List of URLs successfully analyzed
            - status (str): "success", "partial_success", or "failed"
            - message (str): Explanation if partial/failed

    Examples:
        >>> expand_keywords_with_web_search(["medical", "blood", "test"])
        {
            "original": ["medical", "blood", "test"],
            "web_keywords": ["healthcare", "laboratory", "diagnosis", ...],
            "sources": ["https://...", ...],
            "status": "success",
            "message": "Found 15 keywords from 5 sources"
        }
    """
    import logging
    from duckduckgo_search import DDGS
    from config import CONFIG

    # Input validation
    if not keywords or not isinstance(keywords, list) or len(keywords) == 0:
        return {
            "original": keywords or [],
            "web_keywords": [],
            "sources": [],
            "status": "failed",
            "message": "Invalid or empty keywords list"
        }

    # Get config values
    max_results = CONFIG["llm"]["web_search_max_results"]

    # Combine keywords into search query
    query = " ".join(keywords)
    logging.info(f"Searching web for: {query}")

    try:
        # Perform DuckDuckGo search
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        logging.info(f"Found {len(results)} search results")

        if not results:
            return {
                "original": keywords,
                "web_keywords": [],
                "sources": [],
                "status": "failed",
                "message": "No search results found"
            }

        # Extract keywords from each result
        web_keywords = []
        sources = []

        for result in results:
            url = result.get('href') or result.get('link')
            if not url:
                continue

            logging.info(f"Fetching: {url}")

            # Fetch webpage text (uses config timeout)
            text = fetch_webpage_text(url)

            if not text:
                continue

            # Extract keywords from text
            page_keywords = extract_keywords_from_text(text, max_keywords=5)

            if page_keywords:
                web_keywords.extend(page_keywords)
                sources.append(url)

        # Deduplicate and clean
        web_keywords = list(set([k.lower().strip() for k in web_keywords if k.strip()]))

        # Determine status
        if len(web_keywords) > 0:
            status = "success"
            message = f"Found {len(web_keywords)} keywords from {len(sources)} sources"
        elif len(sources) > 0:
            status = "partial_success"
            message = "Sources fetched but no keywords extracted"
        else:
            status = "failed"
            message = "No sources could be fetched"

        logging.info(f"Web search complete: {status} - {message}")

        return {
            "original": keywords,
            "web_keywords": web_keywords,
            "sources": sources,
            "status": status,
            "message": message
        }

    except Exception as e:
        logging.error(f"Web search failed: {str(e)}")
        return {
            "original": keywords,
            "web_keywords": [],
            "sources": [],
            "status": "failed",
            "message": f"Web search error: {str(e)}"
        }