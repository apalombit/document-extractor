"""OCR engine implementation"""
import easyocr
from ocr.base import OCREngine
from config import CONFIG


class SelectedOCR(OCREngine):
    def __init__(self):
        """Initialize chosen OCR library based on config"""
        self.engine = CONFIG["ocr"]["engine"]
        self.languages = CONFIG["ocr"]["languages"]

        if self.engine == "easyocr":
            self.reader = easyocr.Reader(self.languages, gpu=False)
        else:
            raise ValueError(f"OCR engine '{self.engine}' not implemented")

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using configured OCR engine.

        Args:
            image_path: Path to image file

        Returns:
            Raw text string
        """
        if self.engine == "easyocr":
            result = self.reader.readtext(image_path)
            # Extract text from results: [(bbox, text, confidence), ...]
            text_parts = [detection[1] for detection in result]
            return ' '.join(text_parts)
        else:
            raise ValueError(f"OCR engine '{self.engine}' not implemented")
