"""Extraction workflow orchestrator for document processing"""
import json
import tempfile
from pathlib import Path
from typing import Dict
from utils.file_handler import FileHandler
from ocr.engine import SelectedOCR
from llm.extractor import LLMExtractor


class ExtractionWorkflow:
    """
    Orchestrates the complete document extraction pipeline.

    Pipeline stages:
    1. File validation and loading
    2. OCR text extraction
    3. LLM-based field extraction (author_date, keywords, document_type)
    4. Validation and confidence scoring

    Usage:
        workflow = ExtractionWorkflow()
        results = workflow.process_document("path/to/document.png")
    """

    def __init__(self):
        """Initialize workflow with required components"""
        self.file_handler = FileHandler()
        self.ocr = SelectedOCR()
        self.extractor = LLMExtractor()

    def process_document(self, image_path: str) -> Dict:
        """
        Run complete extraction pipeline on a document image.

        Args:
            image_path: Path to PNG image file

        Returns:
            Dictionary containing:
            {
                "author_date": {"authors": [...], "date": "..."},
                "keywords": {"keywords": [...]},
                "document_type": {"document_type": "..."},
                "validation": {
                    "author_date": {"valid_json": bool, "grounding_issues": [...], "confidence": "high|low"},
                    "keywords": {...},
                    "document_type": {...}
                },
                "ocr_text": "extracted text...",
                "errors": []
            }
        """
        results = {
            "author_date": {},
            "keywords": {},
            "document_type": {},
            "validation": {
                "author_date": {},
                "keywords": {},
                "document_type": {}
            },
            "ocr_text": "",
            "errors": []
        }

        try:
            # Step 1: Validate and load image
            try:
                image = self.file_handler.load_image(image_path)
            except Exception as e:
                error_msg = f"File validation error: {str(e)}"
                results["errors"].append(error_msg)
                return results

            # Step 2: Extract OCR text
            try:
                # For file-like objects, we need to save to temp file
                if hasattr(image_path, 'read'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        # Assuming image_path is a file-like object
                        image_path.seek(0)
                        tmp_file.write(image_path.read())
                        tmp_path = tmp_file.name

                    try:
                        ocr_text = self.ocr.extract_text(tmp_path)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                else:
                    # image_path is a string path
                    ocr_text = self.ocr.extract_text(image_path)

                results["ocr_text"] = ocr_text

            except Exception as e:
                error_msg = f"OCR extraction error: {str(e)}"
                results["errors"].append(error_msg)
                return results

            # Step 3: Extract fields with LLM
            tasks = ["author_date", "keywords", "document_type"]

            for task in tasks:
                try:
                    extraction_result, validation_flags = self.extractor.extract_field(ocr_text, task)
                    results[task] = extraction_result
                    results["validation"][task] = validation_flags

                except Exception as e:
                    error_msg = f"LLM extraction error for {task}: {str(e)}"
                    results["errors"].append(error_msg)
                    # Set empty result and low confidence for failed task
                    results[task] = {}
                    results["validation"][task] = {
                        "valid_json": False,
                        "grounding_issues": [error_msg],
                        "confidence": "low"
                    }

            return results

        except Exception as e:
            # Catch-all for unexpected errors
            error_msg = f"Unexpected workflow error: {str(e)}"
            results["errors"].append(error_msg)
            return results

    def export_results(self, results: Dict, format: str = "text") -> str:
        """
        Export extraction results in specified format.

        Args:
            results: Results dictionary from process_document()
            format: "text" or "csv"

        Returns:
            Formatted string representation
        """
        if format == "text":
            return self.file_handler.format_for_copy(results)
        elif format == "csv":
            # CSV export handled by file_handler
            # This would need a file path, so return formatted dict instead
            return json.dumps(results, indent=2)
        else:
            raise ValueError(f"Unknown export format: {format}")
