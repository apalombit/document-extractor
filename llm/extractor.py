"""LLM-based information extraction from OCR text"""
import json
from typing import Tuple, Dict
from config import CONFIG
from llm.ollama_provider import OllamaProvider


# Valid document types for classification validation
VALID_DOCUMENT_TYPES = {"medical", "legal", "invoice", "receipt", "contract", "report", "letter", "form", "other"}


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
        Extract specific field from document text.

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

        # Format prompts
        system = self.system_prompt.format(ocr_text=ocr_text)
        user = self.task_prompts[task]

        # Generate response from LLM
        try:
            response = self.provider.generate(
                system_prompt=system,
                user_prompt=user,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Parse JSON response
            result = json.loads(response)

            # Validate grounding
            validation = self._validate_grounding(result, ocr_text, task)

            return result, validation

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


# Tool placeholders for future implementation
TOOLS = [
    {
        "name": "validate_date",
        "description": "Validate and normalize date format",
        "parameters": {"date_string": "string"}
    },
    {
        "name": "compare_html_to_image",
        "description": "Compare rendered HTML to original image",
        "parameters": {"html": "string", "image_path": "string"}
    }
]
