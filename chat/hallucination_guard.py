"""Hallucination detection guardrail using NLI-based grounding check."""

from typing import List, Tuple, Dict, Optional
import numpy as np
from config import CONFIG


class HallucinationGuard:
    """Guardrail to detect hallucinated content in LLM responses.

    Uses an NLI-based approach:
    1. Split response into sentences
    2. Find relevant source chunks using embeddings
    3. Check entailment using NLI model
    4. Flag sentences not entailed by any source
    """

    def __init__(
        self,
        embedding_model: str = None,
        nli_model: str = None,
        entailment_threshold: float = None
    ):
        """
        Initialize hallucination guardrail.

        Args:
            embedding_model: SentenceTransformer model for finding relevant sources
                            (defaults to config, reuses RAG model)
            nli_model: NLI model for entailment checking (defaults to config)
            entailment_threshold: Minimum entailment score (defaults to config)
        """
        self.embedding_model_name = embedding_model or CONFIG["chat"]["rag_embedding_model"]
        self.nli_model_name = nli_model or CONFIG["chat"]["hallucination_nli_model"]
        self.entailment_threshold = (
            entailment_threshold if entailment_threshold is not None
            else CONFIG["chat"]["hallucination_threshold"]
        )
        self.top_k_sources = CONFIG["chat"].get("hallucination_top_k", 3)
        self.similarity_threshold = CONFIG["chat"].get("hallucination_similarity_threshold", 0.3)

        # Lazy loaded models
        self._sentence_model = None
        self._nli_classifier = None
        self._nltk_initialized = False

        # Track last check for debugging
        self.last_response: Optional[str] = None
        self.last_sentences: Optional[List[str]] = None
        self.last_results: Optional[List[Dict]] = None
        self.last_has_hallucination: Optional[bool] = None

    @property
    def sentence_model(self):
        """Lazy load SentenceTransformer model."""
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer(self.embedding_model_name)
        return self._sentence_model

    @property
    def nli_classifier(self):
        """Lazy load NLI classifier."""
        if self._nli_classifier is None:
            from transformers import pipeline
            self._nli_classifier = pipeline(
                "text-classification",
                model=self.nli_model_name,
            )
        return self._nli_classifier

    def _ensure_nltk(self):
        """Ensure NLTK sentence tokenizer is available."""
        if not self._nltk_initialized:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
            self._nltk_initialized = True

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        self._ensure_nltk()
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        # Filter out very short sentences (< 10 chars) as they're likely noise
        return [s.strip() for s in sentences if len(s.strip()) >= 10]

    def _find_relevant_sources(
        self,
        sentence: str,
        source_chunks: List[str],
        source_embeddings: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Find most relevant source chunks for a sentence.

        Args:
            sentence: Sentence to find sources for
            source_chunks: List of source text chunks
            source_embeddings: Pre-computed embeddings for chunks

        Returns:
            List of (chunk_text, similarity_score) tuples
        """
        if not source_chunks or source_embeddings is None or len(source_embeddings) == 0:
            return []

        # Encode sentence
        sentence_embedding = self.sentence_model.encode([sentence])[0]

        # Calculate cosine similarities
        norms = np.linalg.norm(source_embeddings, axis=1) * np.linalg.norm(sentence_embedding)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        similarities = np.dot(source_embeddings, sentence_embedding) / norms

        # Get top-k sources above similarity threshold
        sorted_indices = np.argsort(similarities)[::-1][:self.top_k_sources]

        results = []
        for i in sorted_indices:
            if similarities[i] >= self.similarity_threshold:
                results.append((source_chunks[i], float(similarities[i])))

        return results

    def _extract_key_terms(self, sentence: str) -> List[str]:
        """
        Extract key terms (dates, numbers, proper nouns) from a sentence.

        These are factual elements that can be verified via exact match.
        """
        import re
        terms = []

        # Date patterns (various formats)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 28/11/2019, 11-28-2019
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',     # 2019-11-28
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # November 28, 2019
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',     # 28 November 2019
        ]
        for pattern in date_patterns:
            terms.extend(re.findall(pattern, sentence, re.IGNORECASE))

        # Numbers with context (e.g., "$500", "42%", "100 pages")
        number_patterns = [
            r'[$€£]\s?\d+(?:,\d{3})*(?:\.\d{2})?',  # Currency
            r'\b\d+(?:,\d{3})*(?:\.\d+)?%',          # Percentages
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',         # Plain numbers
        ]
        for pattern in number_patterns:
            terms.extend(re.findall(pattern, sentence))

        return terms

    def _check_exact_match(self, sentence: str, source_chunks: List[str]) -> bool:
        """
        Check if key factual terms from sentence appear in source chunks.

        This is a fast path for verifying factual claims (dates, numbers)
        without needing NLI inference.

        Returns:
            True if key terms are found in sources (grounded via exact match)
        """
        key_terms = self._extract_key_terms(sentence)

        if not key_terms:
            return False  # No key terms to verify, fall back to NLI

        # Combine all source text for matching
        combined_sources = " ".join(source_chunks).lower()

        # Check if key terms appear in sources
        matches_found = 0
        for term in key_terms:
            if term.lower() in combined_sources:
                matches_found += 1

        # If at least one key term matches, consider it grounded
        return matches_found > 0

    def _check_entailment(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """
        Check if premise entails hypothesis using NLI model.

        Args:
            premise: Source text (the document content)
            hypothesis: Claim to verify (sentence from LLM response)

        Returns:
            Tuple of (label, score) where label is 'entailment', 'neutral', or 'contradiction'
        """
        # NLI input format varies by model
        # GuardrailsAI/finetuned_nli_provenance expects: {"text": premise, "text_pair": hypothesis}
        try:
            result = self.nli_classifier(
                {"text": premise, "text_pair": hypothesis},
                truncation=True,
                max_length=512
            )
            # Handle both list and dict responses
            if isinstance(result, list):
                result = result[0]
            return (result["label"].lower(), result["score"])
        except Exception:
            # Fallback format for other NLI models
            nli_input = f"{premise} </s></s> {hypothesis}"
            result = self.nli_classifier(nli_input, truncation=True, max_length=512)
            if isinstance(result, list):
                result = result[0]
            return (result["label"].lower(), result["score"])

    def check_grounded(
        self,
        response: str,
        source_chunks: List[str],
        source_embeddings: np.ndarray = None
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if LLM response is grounded in source documents.

        Args:
            response: LLM response to check
            source_chunks: Document chunks to check against
            source_embeddings: Pre-computed embeddings (computed if None)

        Returns:
            Tuple of:
            - is_grounded: True if all sentences are entailed
            - results: List of per-sentence results with grounding details
        """
        if not response or not response.strip():
            return (True, [])

        if not source_chunks:
            # No sources to check against - can't verify
            return (True, [{"warning": "no_sources_available"}])

        # Compute source embeddings if not provided
        if source_embeddings is None:
            source_embeddings = self.sentence_model.encode(source_chunks)

        # Split response into sentences
        sentences = self._split_sentences(response)

        if not sentences:
            return (True, [])

        results = []
        has_hallucination = False

        for sentence in sentences:
            # Check entailment against each relevant source
            sentence_result = {
                "sentence": sentence,
                "is_grounded": False,
                "best_entailment_score": 0.0,
                "best_source": None,
                "checked_sources": [],
                "exact_match": False
            }

            # Fast path: check exact match for factual terms (dates, numbers)
            if self._check_exact_match(sentence, source_chunks):
                sentence_result["is_grounded"] = True
                sentence_result["exact_match"] = True
                sentence_result["checked_sources"].append({
                    "note": "Verified via exact term match in source"
                })
                results.append(sentence_result)
                continue

            # Find relevant sources for this sentence
            relevant_sources = self._find_relevant_sources(
                sentence, source_chunks, source_embeddings
            )

            # If no relevant sources found, mark as potentially hallucinated
            if not relevant_sources:
                sentence_result["checked_sources"].append({
                    "note": "No relevant sources found above similarity threshold"
                })
                has_hallucination = True
                results.append(sentence_result)
                continue

            for source_text, similarity in relevant_sources:
                label, score = self._check_entailment(source_text, sentence)

                sentence_result["checked_sources"].append({
                    "source_preview": source_text[:100] + "..." if len(source_text) > 100 else source_text,
                    "similarity": similarity,
                    "entailment_label": label,
                    "entailment_score": score
                })

                if label == "entailment" and score >= self.entailment_threshold:
                    sentence_result["is_grounded"] = True
                    if score > sentence_result["best_entailment_score"]:
                        sentence_result["best_entailment_score"] = score
                        sentence_result["best_source"] = source_text[:100]

            if not sentence_result["is_grounded"]:
                has_hallucination = True

            results.append(sentence_result)

        # Store for debugging
        self.last_response = response
        self.last_sentences = sentences
        self.last_results = results
        self.last_has_hallucination = has_hallucination

        return (not has_hallucination, results)

    def get_hallucinated_sentences(self, results: List[Dict]) -> List[str]:
        """Extract list of sentences flagged as hallucinated."""
        return [r["sentence"] for r in results if not r.get("is_grounded", True) and "sentence" in r]

    def get_debug_info(self) -> dict:
        """Get debug information about last check."""
        return {
            "embedding_model": self.embedding_model_name,
            "nli_model": self.nli_model_name,
            "entailment_threshold": self.entailment_threshold,
            "similarity_threshold": self.similarity_threshold,
            "top_k_sources": self.top_k_sources,
            "last_response": self.last_response,
            "last_sentences": self.last_sentences,
            "last_has_hallucination": self.last_has_hallucination,
            "last_results_count": len(self.last_results) if self.last_results else 0,
        }
