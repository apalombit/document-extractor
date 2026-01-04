"""Abstract base class for LLM providers"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, tools: Optional[List] = None) -> Dict:
        """
        Generate response from LLM with optional tool support.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool functions available to LLM

        Returns:
            Dict with response type and content:
            - {"type": "text", "content": str} for regular text responses
            - {"type": "tool_call", "tool_calls": list} for tool invocations
        """
        pass

    @abstractmethod
    def reset_conversation(self):
        """
        Clear conversation history to start fresh extraction task.
        """
        pass

    @abstractmethod
    def add_tool_result(self, tool_name: str, result: Dict):
        """
        Add tool execution result to conversation history.

        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result dictionary
        """
        pass
