"""On-topic guardrail using zero-shot classification."""

from typing import List, Tuple, Optional
from config import CONFIG


class TopicGuard:
    """Guardrail to check if user query is on-topic with the document."""

    def __init__(self, model_name: str = None):
        """
        Initialize topic guardrail.

        Args:
            model_name: HuggingFace model for zero-shot classification
                       (defaults to config value)
        """
        self.model_name = model_name or CONFIG["chat"]["guardrail_model"]
        self.threshold = CONFIG["chat"]["guardrail_threshold"]
        self._classifier = None

        # Track last check for debugging
        self.last_query: Optional[str] = None
        self.last_score: Optional[float] = None
        self.last_topic: Optional[str] = None
        self.last_on_topic: Optional[bool] = None

    @property
    def classifier(self):
        """Lazy load classifier (expensive)."""
        if self._classifier is None:
            from transformers import pipeline
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                multi_label=True,
            )
        return self._classifier

    def check_on_topic(
        self,
        query: str,
        allowed_topics: List[str],
        threshold: float = None
    ) -> Tuple[bool, float, str]:
        """
        Check if query is on-topic with allowed topics.

        Args:
            query: User query to check
            allowed_topics: List of topics the query should relate to
            threshold: Minimum score to be considered on-topic (defaults to config)

        Returns:
            Tuple of (is_on_topic, max_score, best_matching_topic)
        """
        threshold = threshold if threshold is not None else self.threshold

        if not allowed_topics:
            # No topics to check against, allow everything
            return (True, 1.0, "no_topics_defined")

        result = self.classifier(query, allowed_topics)

        # Get highest scoring topic
        max_score = max(result["scores"])
        best_topic = result["labels"][0]  # Already sorted by score

        is_on_topic = max_score >= threshold

        # Store for debugging
        self.last_query = query
        self.last_score = max_score
        self.last_topic = best_topic
        self.last_on_topic = is_on_topic

        return (is_on_topic, max_score, best_topic)

    def get_debug_info(self) -> dict:
        """Get debug information about last check."""
        return {
            "model": self.model_name,
            "threshold": self.threshold,
            "last_query": self.last_query,
            "last_score": self.last_score,
            "last_topic": self.last_topic,
            "last_on_topic": self.last_on_topic,
        }
