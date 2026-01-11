"""Chat handler for document-grounded conversations."""

from typing import Dict, List, Optional
from chat.context import DocumentContext
from chat.rag import create_rag_store
from chat.guardrail import TopicGuard
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
        use_guardrail: Optional[bool] = None
    ):
        """
        Initialize chat handler.

        Args:
            context: DocumentContext with OCR text and extracted fields
            provider: LLM provider (defaults to OllamaProvider with chat config)
            use_rag: Enable RAG retrieval (defaults to config value)
            use_guardrail: Enable on-topic guardrail (defaults to config value)
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

        # Add document type
        doc_type = self.context.extracted_fields.get("document_type", {})
        if doc_type.get("document_type"):
            topics.append(doc_type["document_type"])

        # Add keywords
        keywords = self.context.extracted_fields.get("keywords", {})
        if keywords.get("keywords"):
            topics.extend(keywords["keywords"])

        # Add generic document-related topics
        topics.extend(["document", "content", "text", "information"])

        return topics

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

        # Check guardrail first
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
        }
