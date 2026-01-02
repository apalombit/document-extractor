"""Abstract base class for OCR engines"""
from abc import ABC, abstractmethod


class OCREngine(ABC):
    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Raw text string extracted from image
        """
        pass
