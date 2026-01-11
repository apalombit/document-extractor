"""PII (Personally Identifiable Information) guard for web search queries."""

import re
from typing import List, Tuple, Optional, Set
from config import CONFIG


class PIIGuard:
    """Guard to detect and filter PII from keywords before web searches.

    Uses regex-based detection for common PII patterns. Falls back to
    Presidio if available and Python version is compatible.
    """

    # Default PII entity types to detect
    DEFAULT_ENTITIES = [
        "PERSON",           # Names
        "PHONE_NUMBER",     # Phone numbers
        "EMAIL_ADDRESS",    # Email addresses
        "CREDIT_CARD",      # Credit card numbers
        "IP_ADDRESS",       # IP addresses
    ]

    # Regex patterns for PII detection
    PII_PATTERNS = {
        "EMAIL_ADDRESS": re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        ),
        "PHONE_NUMBER": re.compile(
            r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        ),
        "CREDIT_CARD": re.compile(
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        ),
        "IP_ADDRESS": re.compile(
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ),
        # Simple name patterns - capitalized words that look like names
        "PERSON": re.compile(
            r'\b(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+\b'
        ),
    }

    # Common non-name capitalized words to exclude from PERSON detection
    COMMON_WORDS = {
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "medical", "diagnosis", "treatment", "healthcare", "document",
        "report", "analysis", "summary", "patient", "doctor", "hospital",
    }

    def __init__(self, entities: List[str] = None, use_presidio: bool = False):
        """
        Initialize PII guard.

        Args:
            entities: List of entity types to detect.
                     Defaults to DEFAULT_ENTITIES if not specified.
            use_presidio: Try to use Presidio if available (may fail on Python 3.14+)
        """
        self.entities = entities or self.DEFAULT_ENTITIES
        self.use_presidio = use_presidio
        self._analyzer = None
        self._presidio_available = None

        # Track last check for debugging
        self.last_input: Optional[List[str]] = None
        self.last_filtered: Optional[List[str]] = None
        self.last_pii_found: Optional[List[dict]] = None

    def _check_presidio_available(self) -> bool:
        """Check if Presidio is available and working."""
        if self._presidio_available is not None:
            return self._presidio_available

        try:
            from presidio_analyzer import AnalyzerEngine
            self._analyzer = AnalyzerEngine()
            # Test with simple text
            self._analyzer.analyze("test", language='en', entities=["EMAIL_ADDRESS"])
            self._presidio_available = True
        except Exception:
            self._presidio_available = False

        return self._presidio_available

    def detect_pii(self, text: str) -> List[dict]:
        """
        Detect PII entities in text.

        Uses regex-based detection, falls back to Presidio if available.

        Args:
            text: Text to analyze for PII

        Returns:
            List of detected PII entities with type, value, and confidence
        """
        if not text or not text.strip():
            return []

        # Try Presidio first if enabled and available
        if self.use_presidio and self._check_presidio_available():
            return self._detect_with_presidio(text)

        # Use regex-based detection
        return self._detect_with_regex(text)

    def _detect_with_presidio(self, text: str) -> List[dict]:
        """Detect PII using Presidio analyzer."""
        results = self._analyzer.analyze(
            text=text,
            language='en',
            entities=self.entities
        )

        return [
            {
                "entity_type": r.entity_type,
                "text": text[r.start:r.end],
                "start": r.start,
                "end": r.end,
                "confidence": r.score
            }
            for r in results
        ]

    def _detect_with_regex(self, text: str) -> List[dict]:
        """Detect PII using regex patterns."""
        results = []

        for entity_type in self.entities:
            pattern = self.PII_PATTERNS.get(entity_type)
            if not pattern:
                continue

            for match in pattern.finditer(text):
                matched_text = match.group()

                # For PERSON, filter out common non-name words
                if entity_type == "PERSON":
                    words = matched_text.lower().split()
                    if any(w in self.COMMON_WORDS for w in words):
                        continue
                    # Skip if it's just one short word
                    if len(words) == 1 and len(matched_text) < 5:
                        continue

                results.append({
                    "entity_type": entity_type,
                    "text": matched_text,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.85  # Regex matches get fixed confidence
                })

        return results

    def contains_pii(self, text: str, min_confidence: float = 0.5) -> bool:
        """
        Check if text contains PII above confidence threshold.

        Args:
            text: Text to check
            min_confidence: Minimum confidence score to consider a match

        Returns:
            True if PII detected, False otherwise
        """
        pii_found = self.detect_pii(text)
        return any(p["confidence"] >= min_confidence for p in pii_found)

    def filter_keywords(
        self,
        keywords: List[str],
        min_confidence: float = None
    ) -> Tuple[List[str], List[str], List[dict]]:
        """
        Filter PII from a list of keywords.

        Args:
            keywords: List of keywords to filter
            min_confidence: Minimum confidence to flag as PII (defaults to config)

        Returns:
            Tuple of:
            - safe_keywords: Keywords with no PII detected
            - filtered_keywords: Keywords that were removed (contained PII)
            - pii_details: Details of all PII found
        """
        if min_confidence is None:
            min_confidence = CONFIG["llm"].get("pii_guard_threshold", 0.5)

        safe_keywords = []
        filtered_keywords = []
        pii_details = []

        for keyword in keywords:
            if not keyword or not keyword.strip():
                continue

            pii_found = self.detect_pii(keyword)

            # Check if any PII found meets confidence threshold
            high_confidence_pii = [
                p for p in pii_found
                if p["confidence"] >= min_confidence
            ]

            if high_confidence_pii:
                filtered_keywords.append(keyword)
                for p in high_confidence_pii:
                    pii_details.append({
                        "keyword": keyword,
                        "entity_type": p["entity_type"],
                        "matched_text": p["text"],
                        "confidence": p["confidence"]
                    })
            else:
                safe_keywords.append(keyword)

        # Store for debugging
        self.last_input = keywords
        self.last_filtered = filtered_keywords
        self.last_pii_found = pii_details

        return safe_keywords, filtered_keywords, pii_details

    def get_debug_info(self) -> dict:
        """Get debug information about last filter operation."""
        return {
            "entities_checked": self.entities,
            "last_input": self.last_input,
            "last_filtered": self.last_filtered,
            "last_pii_found": self.last_pii_found,
        }
