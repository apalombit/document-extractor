"""Ollama LLM provider implementation"""
import json
import ollama
from json_repair import repair_json
from typing import Dict, List, Optional
from llm.provider import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str):
        """
        Initialize Ollama provider.

        Args:
            model: Model name (e.g., "llama3")
        """
        self.model = model
        self.messages = []  # Conversation history for multi-turn

    def generate(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, tools: Optional[List] = None) -> Dict:
        """
        Generate response from Ollama with multi-turn and tool support.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool functions

        Returns:
            Dict with response type:
            - {"type": "text", "content": str} for regular responses
            - {"type": "tool_call", "tool_calls": list} for tool invocations
        """
        # Initialize conversation with system prompt if first turn
        if not self.messages:
            self.messages.append({"role": "system", "content": system_prompt})

        # Add user prompt if provided (empty on subsequent turns)
        if user_prompt:
            self.messages.append({"role": "user", "content": user_prompt})

        # Call Ollama with conversation history and optional tools
        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            tools=tools if tools else None,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        # Add assistant response to conversation history
        self.messages.append(response['message'])

        # Check if response contains tool calls
        if response['message'].get('tool_calls'):
            return {
                "type": "tool_call",
                "tool_calls": response['message']['tool_calls']
            }
        else:
            content = response['message']['content']

            # Check if the text response is actually a tool call formatted as JSON
            # (some smaller models output tool calls as text instead of using tool_calls)
            if content.strip().startswith('{') and tools:
                try:
                    parsed = json.loads(repair_json(content))
                    # Check if it looks like a tool call (has "name" and "parameters" keys)
                    if isinstance(parsed, dict) and "name" in parsed and "parameters" in parsed:
                        # Convert text tool call to proper format
                        return {
                            "type": "tool_call",
                            "tool_calls": [{
                                "function": {
                                    "name": parsed["name"],
                                    "arguments": parsed["parameters"]
                                }
                            }]
                        }
                except json.JSONDecodeError:
                    pass  # Not valid JSON, treat as regular text

            return {
                "type": "text",
                "content": content
            }

    def add_tool_result(self, tool_name: str, result: Dict):
        """
        Add tool execution result to conversation history.

        Args:
            tool_name: Name of the executed tool
            result: Tool result dictionary
        """
        # Add tool result as user message to prompt for final response
        # Note: Some smaller models struggle with tools, so we prompt explicitly
        self.messages.append({
            "role": "user",
            "content": f"Tool result: {json.dumps(result)}. Now provide your final JSON response."
        })

    def reset_conversation(self):
        """
        Clear conversation history for new extraction task.
        """
        self.messages = []
