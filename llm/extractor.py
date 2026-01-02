"""LLM-based information extraction from OCR text"""
import json
from typing import Tuple, Dict
from config import CONFIG
from llm.ollama_provider import OllamaProvider


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
        return """You are a document analysis assistant. Extract information ONLY from the provided text.

EXTRACTED DOCUMENT TEXT:
{ocr_text}

RULES:
- Answer ONLY using information explicitly present in the text
- If information is not clearly present, respond with null
- Output must be valid JSON matching the provided schema
- Never infer, assume, or add information not in the text"""
    
    def _load_task_prompts(self) -> Dict[str, str]:
        """Load task-specific prompts"""
        return {
            "author_date": """Extract document author(s) and date.

EXAMPLES:
Input: "Dr. Smith, 2023-05-10"
Output: {"authors": ["Dr. Smith"], "date": "2023-05-10"}

SCHEMA:
{
  "authors": ["string"] or null,
  "date": "YYYY-MM-DD" or null
}

Extract from the document text above.""",
            
            "keywords": """Extract 3-5 content keywords representing main topics.

EXAMPLES:
Input: Medical blood test with hemoglobin, glucose readings
Output: {"keywords": ["blood test", "hemoglobin", "glucose", "medical results"]}

SCHEMA:
{
  "keywords": ["string"] or null
}

Extract from the document text above.""",
            
            "document_type": """Classify document type.

VALID TYPES: medical, legal, invoice, receipt, contract, report, letter, form, other

EXAMPLES:
Input: Blood test results from hospital
Output: {"document_type": "medical"}

SCHEMA:
{
  "document_type": "string" or null
}

Classify the document text above."""
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
            validation = self._validate_grounding(result, ocr_text)

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
    
    def _validate_grounding(self, result: Dict, ocr_text: str) -> Dict:
        """
        Check if extracted values exist in source text.

        Args:
            result: Extraction result dictionary
            ocr_text: Original OCR text

        Returns:
            Validation flags dictionary with confidence and grounding issues
        """
        flags = {
            "valid_json": True,
            "grounding_issues": [],
            "confidence": "high"
        }

        ocr_text_lower = ocr_text.lower()
        issues = []

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
