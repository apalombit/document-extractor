"""File handling utilities for document loading and export"""
from PIL import Image
import csv
import json
from pathlib import Path
from config import CONFIG


class FileHandler:
    
    @staticmethod
    def load_image(file_path):
        """
        Load and validate image file.

        Args:
            file_path: Path to image file (str or Streamlit UploadedFile)

        Returns:
            PIL Image object

        Raises:
            ValueError: If format unsupported or file too large
        """
        # Handle Streamlit UploadedFile
        if hasattr(file_path, 'name'):
            # Streamlit UploadedFile object
            file_ext = Path(file_path.name).suffix.lower()
            if file_ext not in CONFIG["file"]["supported_formats"]:
                raise ValueError(f"Unsupported format: {file_ext}")

            # Check size
            file_path.seek(0, 2)  # Seek to end
            size_mb = file_path.tell() / (1024 * 1024)
            file_path.seek(0)  # Reset to beginning
            if size_mb > CONFIG["file"]["max_size_mb"]:
                raise ValueError(f"File too large: {size_mb:.1f}MB")

            return Image.open(file_path)

        # Handle file path string
        path = Path(file_path)

        # Validate format
        if path.suffix.lower() not in CONFIG["file"]["supported_formats"]:
            raise ValueError(f"Unsupported format: {path.suffix}")

        # Validate size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > CONFIG["file"]["max_size_mb"]:
            raise ValueError(f"File too large: {size_mb:.1f}MB")

        return Image.open(file_path)
    
    @staticmethod
    def export_to_csv(results: dict, output_path: str):
        """
        Export extraction results to CSV.
        
        Args:
            results: Dictionary of extraction results
            output_path: Path for CSV output file
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Field', 'Value'])
            
            # Flatten nested dict
            for key, value in results.items():
                if isinstance(value, dict):
                    writer.writerow([key, json.dumps(value)])
                elif isinstance(value, list):
                    writer.writerow([key, ', '.join(map(str, value))])
                else:
                    writer.writerow([key, value])
    
    @staticmethod
    def format_for_copy(results: dict) -> str:
        """
        Format results as copyable text.
        
        Args:
            results: Dictionary of extraction results
            
        Returns:
            Formatted string with 'key: value' lines
        """
        lines = []
        for key, value in results.items():
            if value:
                lines.append(f"{key}: {value}")
        return '\n'.join(lines)
