"""Abstract base class for LLM providers"""
from abc import ABC, abstractmethod
from typing import Dict


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Generate response from LLM.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Raw response string from LLM
        """
        pass
