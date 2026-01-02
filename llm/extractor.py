"""LLM-based information extraction from OCR text"""
from typing import Tuple, Dict
# import ollama  # Uncomment when implementing
from config import CONFIG


class LLMExtractor:
    def __init__(self):
        """Initialize LLM extractor with prompts and config"""
        self.model = CONFIG["llm"]["model"]
        self.temperature = CONFIG["llm"]["temperature"]
        self.max_tokens = CONFIG["llm"]["max_tokens"]
        
        # TODO: Load prompt templates
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
        # TODO: Implement Ollama call
        # prompt = {
        #     "system": self.system_prompt.format(ocr_text=ocr_text),
        #     "user": self.task_prompts[task]
        # }
        # response = ollama.generate(
        #     model=self.model,
        #     prompt=prompt,
        #     temperature=self.temperature
        # )
        # result = json.loads(response)
        # validation = self._validate_grounding(result, ocr_text)
        # return result, validation
        pass
    
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
        
        # TODO: Implement grounding validation
        # Check if extracted values appear in ocr_text
        # Flag any values not found in source
        
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
