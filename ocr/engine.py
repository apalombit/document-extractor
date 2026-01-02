"""OCR engine implementation"""
from ocr.base import OCREngine
from config import CONFIG


class SelectedOCR(OCREngine):
    def __init__(self):
        """Initialize chosen OCR library based on config"""
        self.engine = CONFIG["ocr"]["engine"]
        self.languages = CONFIG["ocr"]["languages"]
        # TODO: Initialize actual OCR library (EasyOCR/Tesseract/PaddleOCR)
        pass
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using configured OCR engine.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Raw text string
        """
        # TODO: Implement OCR extraction
        # Example for EasyOCR:
        # reader = easyocr.Reader(self.languages)
        # result = reader.readtext(image_path)
        # return ' '.join([detection[1] for detection in result])
        pass
