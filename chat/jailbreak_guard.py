"""Jailbreak/prompt injection detection guardrail."""

from typing import Tuple, Optional
from config import CONFIG


class JailbreakGuard:
    """Guardrail to detect prompt injection and jailbreak attempts."""

    def __init__(self, model_name: str = None, threshold: float = None):
        """
        Initialize jailbreak guardrail.

        Args:
            model_name: HuggingFace model for injection detection
                       (defaults to config value)
            threshold: Minimum confidence to flag as jailbreak (defaults to config)
        """
        self.model_name = model_name or CONFIG["chat"]["jailbreak_model"]
        self.threshold = threshold if threshold is not None else CONFIG["chat"]["jailbreak_threshold"]
        self._classifier = None

        # Track last check for debugging
        self.last_query: Optional[str] = None
        self.last_score: Optional[float] = None
        self.last_label: Optional[str] = None
        self.last_is_safe: Optional[bool] = None

    @property
    def classifier(self):
        """Lazy load classifier (expensive)."""
        if self._classifier is None:
            from transformers import pipeline
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
            )
        return self._classifier

    def check_safe(self, query: str, threshold: float = None) -> Tuple[bool, float, str]:
        """
        Check if query is safe (not a jailbreak attempt).

        Args:
            query: User query to check
            threshold: Minimum score to flag as injection (defaults to config)

        Returns:
            Tuple of (is_safe, injection_score, label)
        """
        threshold = threshold if threshold is not None else self.threshold

        if not query or not query.strip():
            return (True, 0.0, "empty_query")

        result = self.classifier(query, truncation=True, max_length=512)[0]
        label = result["label"]
        score = result["score"]

        # Model outputs "INJECTION" or "SAFE" labels (case may vary by model)
        # If label indicates injection and score > threshold, it's not safe
        is_injection_label = label.upper() in ("INJECTION", "LABEL_1", "POSITIVE")
        is_safe = not (is_injection_label and score >= threshold)

        # Store for debugging
        self.last_query = query
        self.last_score = score
        self.last_label = label
        self.last_is_safe = is_safe

        return (is_safe, score, label)

    def get_debug_info(self) -> dict:
        """Get debug information about last check."""
        return {
            "model": self.model_name,
            "threshold": self.threshold,
            "last_query": self.last_query,
            "last_score": self.last_score,
            "last_label": self.last_label,
            "last_is_safe": self.last_is_safe,
        }
