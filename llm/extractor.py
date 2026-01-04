"""LLM-based information extraction from OCR text"""
import json
from typing import Tuple, Dict
from config import CONFIG
from llm.ollama_provider import OllamaProvider


# Valid document types for classification validation
VALID_DOCUMENT_TYPES = {"medical", "legal", "invoice", "receipt", "contract", "report", "letter", "form", "other"}

# Critique prompt for self-review step
CRITIQUE_PROMPT = """Review your previous extraction answer for potential flaws or inconsistencies.

YOUR PREVIOUS ANSWER:
{previous_answer}

ORIGINAL TASK:
{task_description}

CRITIQUE INSTRUCTIONS:
1. Check if extracted values truly exist in the source document text
2. Verify dates are realistic and properly formatted
3. Ensure authors/keywords are specific entities (not generic terms)
4. Confirm document type classification makes sense for the content
5. Look for any logical inconsistencies

RESPONSE OPTIONS:
A) If you find issues, provide an IMPROVED answer in the same JSON format
B) If your answer is already correct, return it unchanged

Output ONLY the JSON (improved or unchanged) with NO OTHER COMMENT.
"""


class LLMExtractor:
    def __init__(self, provider=None):
        """
        Initialize LLM extractor with prompts and config.

        Args:
            provider: LLM provider instance (defaults to OllamaProvider)
        """
        self.model = CONFIG["llm"]["model"]
        self.temperature = CONFIG["llm"]["temperature"]
        self.max_tokens = CONFIG["llm"]["max_tokens"]

        # Use provided provider or default to Ollama
        self.provider = provider or OllamaProvider(self.model)

        self.system_prompt = self._load_system_prompt()
        self.task_prompts = self._load_task_prompts()
    
    def _load_system_prompt(self) -> str:
        """Load system prompt template"""
        return """You are a document analysis assistant. Extract information ONLY from the provided DOCUMENT TEXT only to determine the REQUESTED INFORMATION.

DOCUMENT TEXT:
{ocr_text}

RULES:
- Answer ONLY based on information explicitly present in the text
- If information is not clearly present or you cannot answer based on text, respond with null
- Never infer, assume, or add information not in the text
- Your answer should ONLY be a JSON following provided schema in RESPONSE FORMAT with NO OTHER COMMENT added
"""

    def _get_system_prompt_with_tools(self, ocr_text: str) -> str:
        """
        Create system prompt with tool availability information.

        Args:
            ocr_text: OCR text to include in prompt

        Returns:
            System prompt string with tool instructions
        """
        base = self.system_prompt.format(ocr_text=ocr_text)

        tool_info = """

AVAILABLE TOOLS:
- validate_date(date_string): Validates that a date exists and is not in the future.
  Use this to verify any dates you extract from the document.
  Returns: {"valid": bool, "reason": str, "normalized": str}

TOOL USAGE INSTRUCTIONS:
- After extracting a date, call validate_date to verify it
- If validation fails (valid=False), reconsider your extraction or set date to null
- Use the normalized format from the tool if provided
"""

        return base + tool_info

    def _load_task_prompts(self) -> Dict[str, str]:
        """Load task-specific prompts"""
        return {
"author_date": """REQUESTED INFORMATION: document author(s) and date.

NOTE: author name(s) is the author (person or institution) that wrote or emitted the document from which the text was extracted, if not clear leave null.
NOTE: date is the writing or emission date associated with the document from which the text was extracted, if not clear leave null.
CRITICAL: both identified author names and date should be reported VERBATIM as in provided input text.

EXAMPLES:
Input: "Dr. Smith wrote a report for exam referred to 2023 10th May"
Output: {"authors": ["Dr. Smith"], "date": "2023 10th May"}

Input: "Report by Dr. Jones and Hospital San Raffaele dated 15/03/2024"
Output: {"authors": ["Dr. Jones", "Hospital San Raffaele"], "date": "15/03/2024"}

Input: "Medical record from Gemelli Hospital - March 2023"
Output: {"authors": ["Gemelli Hospital"], "date": "March 2023"}

Input: "Patient John Doe underwent blood test. No physician signature. Date not specified."
Output: {"authors": null, "date": null}

Input: "Test results for patient Maria Rossi"
Output: {"authors": null, "date": null}

RESPONSE FORMAT (JSON):
{
  "authors": ["document authors found"] or null,
  "date": "date" found or null
}
Answer ONLY as in RESPONSE FORMAT with NO OTHER COMMENT added.
""",

"keywords": """REQUESTED INFORMATION: 2-4 content keywords representing main topics of this text.

SELECTION CRITERIA:
- Choose the most significant/specific nouns or concepts
- Prioritize technical/domain-specific terms over common words
- Extract keywords EXACTLY as they appear in text
- Aim for 2-4 keywords; if text is too short/unclear, return null

EXAMPLES:
Input: "Medical blood test with hemoglobin, glucose readings indicate normal situation as of 2023 10th May"
Output: {"keywords": ["blood", "hemoglobin", "glucose"]}

Input: "Legal contract for property transfer between parties. Confidential agreement signed on 2024."
Output: {"keywords": ["legal", "contract", "property", "transfer", "agreement"]}

Input: "Hello"
Output: {"keywords": null}

Input: "This is a document with some text and other things here."
Output: {"keywords": null}

RESPONSE FORMAT (JSON):
{
  "keywords": ["2-4 specific terms from text"] or null
}
Answer ONLY as in RESPONSE FORMAT with NO OTHER COMMENT added.
""",

"document_type": """REQUESTED INFORMATION: document type.

VALID TYPES: medical, legal, invoice, receipt, contract, report, letter, form, other

EXAMPLES:
Input: "Blood test results from hospital"
Output: {"document_type": "medical"}

Input: "Agreement between parties for service delivery signed on March 2024"
Output: {"document_type": "contract"}

Input: "Invoice #12345 - Payment due: €500.00 - Services rendered"
Output: {"document_type": "invoice"}

Input: "Some random text fragments without clear purpose or structure"
Output: {"document_type": null}

Input: "Unclear content"
Output: {"document_type": null}

RESPONSE FORMAT (JSON):
{
  "document_type": "one of the VALID TYPES" or null
}
Answer ONLY as in RESPONSE FORMAT with NO OTHER COMMENT added.
"""
}

    def extract_field(self, ocr_text: str, task: str) -> Tuple[Dict, Dict]:
        """
        Extract specific field from document text using multi-turn conversation with tool support.

        Args:
            ocr_text: Raw text from OCR
            task: Field to extract ("author_date", "keywords", "document_type")

        Returns:
            Tuple of (extraction_result, validation_flags)
            - extraction_result: Dict with extracted data or null values
            - validation_flags: Dict with confidence and grounding issues
        """
        if task not in self.task_prompts:
            raise ValueError(f"Unknown task: {task}")

        # Reset conversation for fresh extraction
        self.provider.reset_conversation()

        # Format prompts (use tool-aware system prompt for author_date task)
        if task == "author_date":
            system = self._get_system_prompt_with_tools(ocr_text)
        else:
            system = self.system_prompt.format(ocr_text=ocr_text)

        user = self.task_prompts[task]

        # Import tools for author_date task
        tools = None
        use_tools = task == "author_date"
        if use_tools:
            from llm.tools import validate_date
            tools = [validate_date]

        # Multi-turn conversation loop
        max_turns = CONFIG["llm"]["max_conversation_turns"]
        turn = 0
        tool_support_error = False

        try:
            while turn < max_turns:
                # Generate response from LLM
                try:
                    response = self.provider.generate(
                        system_prompt=system,
                        user_prompt=user if turn == 0 else "",  # Only send user prompt on first turn
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        tools=tools if not tool_support_error else None  # Disable tools if not supported
                    )
                except Exception as e:
                    # Check if error is due to tool support
                    error_msg = str(e).lower()
                    if "does not support tools" in error_msg or "tool" in error_msg and "400" in error_msg:
                        # Model doesn't support tools - retry without tools
                        tool_support_error = True
                        tools = None
                        # Reset conversation and use non-tool system prompt
                        self.provider.reset_conversation()
                        system = self.system_prompt.format(ocr_text=ocr_text)
                        turn = 0
                        continue
                    else:
                        # Re-raise other errors
                        raise

                if response["type"] == "tool_call":
                    # Execute tools and add results to conversation
                    for tool_call in response["tool_calls"]:
                        if tool_call['function']['name'] == "validate_date":
                            # Parse arguments (handle both string and dict formats)
                            arguments = tool_call['function']['arguments']
                            if isinstance(arguments, str):
                                args = json.loads(arguments)
                            else:
                                args = arguments  # Already a dict

                            from llm.tools import validate_date
                            result = validate_date(args['date_string'])
                            self.provider.add_tool_result("validate_date", result)
                            break  # Only execute first tool call to prevent loops

                    # Disable tools after first call to prevent infinite loops
                    tools = None
                    turn += 1
                else:
                    # Got final text response
                    # Parse JSON response
                    result = json.loads(response["content"])

                    # Apply optional critique step
                    result = self._apply_critique_step(result, ocr_text, task)

                    # Validate grounding
                    validation = self._validate_grounding(result, ocr_text, task)

                    return result, validation

            # Max turns exceeded
            return {}, {
                "valid_json": False,
                "grounding_issues": ["Max conversation turns exceeded"],
                "confidence": "low"
            }

        except json.JSONDecodeError as e:
            # Return error state with low confidence
            return {}, {
                "valid_json": False,
                "grounding_issues": [f"JSON parse error: {str(e)}"],
                "confidence": "low"
            }
        except Exception as e:
            # Catch other errors
            return {}, {
                "valid_json": False,
                "grounding_issues": [f"Error: {str(e)}"],
                "confidence": "low"
            }
    
    def _validate_grounding(self, result: Dict, ocr_text: str, task: str) -> Dict:
        """
        Check if extracted values are valid based on task type.

        Args:
            result: Extraction result dictionary
            ocr_text: Original OCR text
            task: Task name ("author_date", "keywords", "document_type")

        Returns:
            Validation flags dictionary with confidence and grounding issues
        """
        flags = {
            "valid_json": True,
            "grounding_issues": [],
            "confidence": "high"
        }

        issues = []

        # Task-specific validation
        if task == "document_type":
            # For classification: validate against valid types
            if "document_type" in result:
                doc_type = result["document_type"]
                if doc_type is not None and doc_type not in VALID_DOCUMENT_TYPES:
                    issues.append(f"'{doc_type}' is not a valid document type. Valid types: {', '.join(sorted(VALID_DOCUMENT_TYPES))}")
        else:
            # For extraction tasks (author_date, keywords): validate grounding in OCR text
            ocr_text_lower = ocr_text.lower()

            # Extract all string values from result recursively
            def extract_strings(obj):
                strings = []
                if isinstance(obj, str):
                    strings.append(obj)
                elif isinstance(obj, list):
                    for item in obj:
                        strings.extend(extract_strings(item))
                elif isinstance(obj, dict):
                    for value in obj.values():
                        strings.extend(extract_strings(value))
                return strings

            extracted_values = extract_strings(result)

            # Check each extracted value exists in OCR text
            for value in extracted_values:
                if value and isinstance(value, str) and len(value) > 2:
                    # Skip very short strings, check if value appears in text
                    if value.lower() not in ocr_text_lower:
                        issues.append(f"'{value}' not found in source text")

        # Update flags based on issues
        if issues:
            flags["grounding_issues"] = issues
            flags["confidence"] = "low"

        return flags

    def _apply_critique_step(self, result: Dict, ocr_text: str, task: str) -> Dict:
        """
        Optional self-critique step where LLM reviews its own answer.

        Args:
            result: Initial extraction result
            ocr_text: Original document text
            task: Task name

        Returns:
            Improved result or original if no changes
        """
        if not CONFIG["llm"]["enable_critique"]:
            return result

        # Skip critique if result is empty or invalid
        if not result or not isinstance(result, dict):
            return result

        try:
            # Format critique prompt
            critique_prompt = CRITIQUE_PROMPT.format(
                previous_answer=json.dumps(result, indent=2),
                task_description=self.task_prompts[task]
            )

            # Reset conversation for critique step
            self.provider.reset_conversation()

            # Get LLM's critique and improved response
            response = self.provider.generate(
                system_prompt=self.system_prompt.format(ocr_text=ocr_text),
                user_prompt=critique_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=None  # No tools in critique step
            )

            # Parse improved result
            if response["type"] == "text" and response.get("content"):
                improved_result = json.loads(response["content"])
                return improved_result
            else:
                # If critique didn't return text, keep original
                return result

        except (json.JSONDecodeError, KeyError, Exception):
            # If critique fails for any reason, return original result
            return result

    def extract_html(self, ocr_text: str) -> Dict:
        """
        Placeholder for HTML generation (stretch goal).
        
        TODO: HTML Generation (Stretch Goal - Not MVP)
        
        GOAL: Convert document to HTML preserving layout and content
        
        STEPS TO IMPLEMENT:
        1. LLM generates HTML from OCR text + layout hints
        2. Use tool "compare_html_to_image" to validate visual similarity
        3. Iterate if comparison score < threshold
        
        SCHEMA:
        {
          "html": "string",
          "layout_preserved": boolean,
          "similarity_score": float
        }
        """
        return {
            "html": None,
            "status": "not_implemented",
            "note": "Stretch goal - post MVP"
        }
