"""Document context container for chat grounding."""

from typing import Dict, Optional
import json


class DocumentContext:
    """Container for document data that grounds chat responses."""

    def __init__(self, ocr_text: str, extracted_fields: Dict):
        """
        Initialize document context.

        Args:
            ocr_text: Raw OCR text from document
            extracted_fields: Dict with author_date, keywords, document_type
        """
        self.ocr_text = ocr_text
        self.extracted_fields = extracted_fields

    def to_context_string(self) -> str:
        """
        Format document data for LLM system prompt.

        Returns:
            Formatted string with OCR text and extracted metadata
        """
        sections = []

        # Add extracted metadata section
        sections.append("=== EXTRACTED INFORMATION ===")

        # Author and date
        author_date = self.extracted_fields.get("author_date", {})
        if author_date:
            authors = author_date.get("authors")
            date = author_date.get("date")
            if authors:
                sections.append(f"Authors: {', '.join(authors)}")
            if date:
                sections.append(f"Date: {date}")

        # Keywords
        keywords_data = self.extracted_fields.get("keywords", {})
        if keywords_data:
            keywords = keywords_data.get("keywords")
            if keywords:
                sections.append(f"Keywords: {', '.join(keywords)}")

        # Document type
        doc_type_data = self.extracted_fields.get("document_type", {})
        if doc_type_data:
            doc_type = doc_type_data.get("document_type")
            if doc_type:
                sections.append(f"Document Type: {doc_type}")

        # Add OCR text section
        sections.append("\n=== DOCUMENT TEXT ===")
        sections.append(self.ocr_text)

        return "\n".join(sections)

    @classmethod
    def from_workflow_results(cls, results: Dict) -> 'DocumentContext':
        """
        Create DocumentContext from ExtractionWorkflow.process_document() results.

        Args:
            results: Dict returned by ExtractionWorkflow.process_document()
                    Expected keys: ocr_text, author_date, keywords, document_type

        Returns:
            DocumentContext instance
        """
        ocr_text = results.get("ocr_text", "")

        extracted_fields = {
            "author_date": results.get("author_date", {}),
            "keywords": results.get("keywords", {}),
            "document_type": results.get("document_type", {}),
        }

        return cls(ocr_text=ocr_text, extracted_fields=extracted_fields)

    def is_empty(self) -> bool:
        """Check if context has any meaningful content."""
        return not self.ocr_text.strip()
