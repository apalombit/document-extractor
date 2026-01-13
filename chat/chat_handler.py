"""Chat handler for document-grounded conversations."""

from typing import Dict, List, Optional
from chat.context import DocumentContext
from chat.rag import create_rag_store
from chat.guardrail import TopicGuard
from chat.jailbreak_guard import JailbreakGuard
from chat.hallucination_guard import HallucinationGuard
from llm.provider import LLMProvider
from llm.ollama_provider import OllamaProvider
from config import CONFIG


CHAT_SYSTEM_PROMPT = """You are a document assistant. Answer questions based ONLY on the document content provided below.

{document_context}

RULES:
- Answer ONLY based on information explicitly present in the document
- If the answer is not in the document, say "I cannot find this information in the document"
- Be concise but complete in your answers
- Reference specific parts of the document when possible
- Do not make assumptions or add information not present in the document
"""


class ChatHandler:
    """Handles chat conversations grounded in document context."""

    def __init__(
        self,
        context: DocumentContext,
        provider: Optional[LLMProvider] = None,
        use_rag: Optional[bool] = None,
        use_guardrail: Optional[bool] = None,
        use_jailbreak_guard: Optional[bool] = None,
        use_hallucination_guard: Optional[bool] = None
    ):
        """
        Initialize chat handler.

        Args:
            context: DocumentContext with OCR text and extracted fields
            provider: LLM provider (defaults to OllamaProvider with chat config)
            use_rag: Enable RAG retrieval (defaults to config value)
            use_guardrail: Enable on-topic guardrail (defaults to config value)
            use_jailbreak_guard: Enable jailbreak/injection detection (defaults to config value)
            use_hallucination_guard: Enable hallucination detection (defaults to config value)
        """
        self.context = context
        self.model = CONFIG["chat"]["model"]
        self.temperature = CONFIG["chat"]["temperature"]
        self.max_tokens = CONFIG["chat"]["max_tokens"]
        self.max_history_turns = CONFIG["chat"]["max_history_turns"]

        # RAG settings
        self.use_rag = use_rag if use_rag is not None else CONFIG["chat"]["use_rag"]
        self.rag_store = None
        if self.use_rag:
            self.rag_store = create_rag_store(context)

        # Guardrail settings
        self.use_guardrail = use_guardrail if use_guardrail is not None else CONFIG["chat"]["use_guardrail"]
        self.guardrail = None
        if self.use_guardrail:
            self.guardrail = TopicGuard()

        # Jailbreak guard settings
        self.use_jailbreak_guard = (
            use_jailbreak_guard if use_jailbreak_guard is not None
            else CONFIG["chat"].get("use_jailbreak_guard", False)
        )
        self.jailbreak_guard = None
        if self.use_jailbreak_guard:
            self.jailbreak_guard = JailbreakGuard()

        # Hallucination guard settings
        self.use_hallucination_guard = (
            use_hallucination_guard if use_hallucination_guard is not None
            else CONFIG["chat"].get("use_hallucination_guard", False)
        )
        self.hallucination_guard = None
        if self.use_hallucination_guard:
            self.hallucination_guard = HallucinationGuard()

        self.provider = provider or OllamaProvider(self.model)
        self.history: List[Dict[str, str]] = []  # For UI display
        self._initialized = False

    def _get_system_prompt(self, user_message: str = None) -> str:
        """
        Build system prompt with document context.

        Args:
            user_message: User query (used for RAG retrieval)

        Returns:
            System prompt string with document content embedded
        """
        if self.use_rag and self.rag_store and user_message:
            # Use RAG to get relevant context
            document_context = self.rag_store.get_augmented_context(user_message)
        else:
            # Use full document context
            document_context = self.context.to_context_string()

        return CHAT_SYSTEM_PROMPT.format(document_context=document_context)

    def _truncate_history(self):
        """Remove oldest messages if history exceeds max turns."""
        max_messages = self.max_history_turns * 2  # Each turn = user + assistant
        if len(self.history) > max_messages:
            # Keep most recent messages
            self.history = self.history[-max_messages:]

    def _get_allowed_topics(self) -> List[str]:
        """
        Build list of allowed topics from document metadata.

        Returns:
            List of topic strings for guardrail classification
        """
        topics = []

        # Add document type (handle both dict and string formats)
        doc_type = self.context.extracted_fields.get("document_type")
        if doc_type:
            if isinstance(doc_type, dict) and doc_type.get("document_type"):
                topics.append(doc_type["document_type"])
            elif isinstance(doc_type, str):
                topics.append(doc_type)

        # Add keywords (handle both dict and list formats)
        keywords = self.context.extracted_fields.get("keywords")
        if keywords:
            if isinstance(keywords, dict) and keywords.get("keywords"):
                topics.extend(keywords["keywords"])
            elif isinstance(keywords, list):
                topics.extend(keywords)

        # Add generic document-related topics
        topics.extend(["document", "content", "text", "information"])

        return topics

    def _get_source_chunks(self) -> List[str]:
        """Get document chunks for hallucination checking."""
        if self.rag_store:
            return self.rag_store.get_all_chunks()
        # Fallback: split OCR text into chunks
        return self._split_into_chunks(self.context.ocr_text)

    def _get_source_embeddings(self):
        """Get pre-computed embeddings if available from RAG store."""
        if self.rag_store and hasattr(self.rag_store, 'embeddings'):
            return self.rag_store.embeddings
        return None  # HallucinationGuard will compute them

    def _split_into_chunks(self, text: str, chunk_size: int = None) -> List[str]:
        """Simple chunking fallback when RAG is disabled."""
        import re
        chunk_size = chunk_size or CONFIG["chat"].get("rag_chunk_size", 200)

        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) <= chunk_size:
                        current += (" " if current else "") + sent
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = sent
                if current:
                    chunks.append(current.strip())
        return chunks

    def _is_uncertainty_response(self, response: str) -> bool:
        """
        Check if response indicates model uncertainty or inability to answer.

        These responses should bypass hallucination checking since expressing
        uncertainty is the correct behavior, not a hallucination.

        Args:
            response: LLM response text

        Returns:
            True if response expresses uncertainty/inability to answer
        """
        response_lower = response.lower().strip()

        # Common uncertainty phrases (model admitting it cannot answer)
        uncertainty_phrases = [
            "cannot find",
            "could not find",
            "can't find",
            "couldn't find",
            "not mentioned",
            "not specified",
            "not stated",
            "not included",
            "not provided",
            "no information",
            "no mention",
            "does not mention",
            "doesn't mention",
            "does not contain",
            "doesn't contain",
            "not in the document",
            "not available in",
            "unable to find",
            "unable to determine",
            "unclear from",
            "not clear from",
            "i don't have",
            "i do not have",
            "cannot determine",
            "can't determine",
            "not enough information",
            "insufficient information",
        ]

        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                return True

        return False

    def chat(self, user_message: str) -> str:
        """
        Process user message and return grounded response.

        Args:
            user_message: User's question about the document

        Returns:
            LLM response grounded in document context
        """
        if self.context.is_empty():
            return "No document has been loaded. Please analyze a document first."

        # Check jailbreak guard FIRST (pre-generation)
        if self.use_jailbreak_guard and self.jailbreak_guard:
            is_safe, score, label = self.jailbreak_guard.check_safe(user_message)
            if not is_safe:
                return (
                    f"I cannot process this request as it appears to be a prompt injection attempt "
                    f"(detected: {label}, confidence: {score:.2f}). "
                    f"Please rephrase your question about the document."
                )

        # Check topic guardrail
        if self.use_guardrail and self.guardrail:
            allowed_topics = self._get_allowed_topics()
            is_on_topic, score, topic = self.guardrail.check_on_topic(
                user_message,
                allowed_topics
            )
            if not is_on_topic:
                # Get document type for personalized response
                doc_type = self.context.extracted_fields.get("document_type", {}).get("document_type", "document")
                return f"I can only answer questions about this {doc_type} document. Your question seems off-topic (best match: '{topic}' with score {score:.2f}). Please ask something related to the document content."

        # Initialize provider with system prompt on first message
        if not self._initialized:
            self.provider.reset_conversation()
            self._initialized = True

        # Add user message to display history
        self.history.append({"role": "user", "content": user_message})

        # Generate response (pass user_message for RAG retrieval)
        system_prompt = self._get_system_prompt(user_message)
        response = self.provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=None
        )

        # Extract response content
        if response["type"] == "text":
            assistant_message = response["content"]
        else:
            # Shouldn't happen without tools, but handle gracefully
            assistant_message = "I encountered an issue processing your question. Please try again."

        # Check hallucination guard (post-generation)
        if self.use_hallucination_guard and self.hallucination_guard:
            # Skip hallucination check if model expressed uncertainty (not a hallucination)
            if not self._is_uncertainty_response(assistant_message):
                source_chunks = self._get_source_chunks()
                source_embeddings = self._get_source_embeddings()

                is_grounded, grounding_results = self.hallucination_guard.check_grounded(
                    assistant_message,
                    source_chunks,
                    source_embeddings
                )

                if not is_grounded:
                    hallucinated = self.hallucination_guard.get_hallucinated_sentences(grounding_results)
                    lenient_mode = CONFIG["chat"].get("hallucination_lenient", False)

                    if lenient_mode:
                        # Lenient mode: show response with debug info appended
                        if hallucinated:
                            flagged_preview = "; ".join(s[:50] + "..." if len(s) > 50 else s for s in hallucinated[:3])
                            assistant_message = (
                                f"{assistant_message}\n\n"
                                f"(Grounding warning: Some parts could not be verified against document. "
                                f"Flagged: {flagged_preview})"
                            )
                    else:
                        # Strict mode: reject response entirely
                        assistant_message = (
                            "I apologize, but I could not verify my response against the document content. "
                            "Some parts of my answer may not be directly supported by the document. "
                            "Please ask a more specific question about the document content."
                        )

        # Add assistant message to display history
        self.history.append({"role": "assistant", "content": assistant_message})

        # Truncate history if needed
        self._truncate_history()

        return assistant_message

    def clear_history(self):
        """Reset conversation for new chat session."""
        self.history = []
        self.provider.reset_conversation()
        self._initialized = False

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history for display.

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        return self.history.copy()

    def get_provider_messages(self) -> List[Dict]:
        """
        Get raw provider messages for debugging.

        Returns:
            List of messages sent to LLM (system, user, assistant)
        """
        return self.provider.messages.copy()

    def get_debug_info(self) -> Dict:
        """
        Get debug information about chat state.

        Returns:
            Dict with provider messages count, RAG status, guardrail status, etc.
        """
        rag_method = None
        rag_model = None
        if self.rag_store:
            from chat.rag import EmbeddingRAGStore
            if isinstance(self.rag_store, EmbeddingRAGStore):
                rag_method = "embedding"
                rag_model = self.rag_store.model_name
            else:
                rag_method = "keyword"

        # Guardrail info
        guardrail_info = None
        if self.guardrail:
            guardrail_info = self.guardrail.get_debug_info()
            guardrail_info["allowed_topics"] = self._get_allowed_topics()

        # Jailbreak guard info
        jailbreak_info = None
        if self.jailbreak_guard:
            jailbreak_info = self.jailbreak_guard.get_debug_info()

        # Hallucination guard info
        hallucination_info = None
        if self.hallucination_guard:
            hallucination_info = self.hallucination_guard.get_debug_info()

        return {
            "provider_message_count": len(self.provider.messages),
            "display_history_count": len(self.history),
            "use_rag": self.use_rag,
            "rag_method": rag_method,
            "rag_model": rag_model,
            "rag_chunks": self.rag_store.get_chunk_count() if self.rag_store else 0,
            "initialized": self._initialized,
            "use_guardrail": self.use_guardrail,
            "guardrail": guardrail_info,
            "use_jailbreak_guard": self.use_jailbreak_guard,
            "jailbreak_guard": jailbreak_info,
            "use_hallucination_guard": self.use_hallucination_guard,
            "hallucination_guard": hallucination_info,
        }
