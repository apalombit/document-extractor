"""Ollama LLM provider implementation"""
import ollama
from llm.provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "llama3")
        """
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generate response from Ollama.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Raw response string from LLM
        """
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        return response['message']['content']
