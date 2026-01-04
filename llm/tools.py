"""LLM tools for multi-turn conversations"""
from datetime import datetime
from typing import Dict
from dateutil import parser
from dateutil.parser import ParserError


def validate_date(date_string: str) -> Dict:
    """
    Validate that a date exists and is not in the future.

    This tool parses common date formats and verifies the date is valid
    and not in the future. Used by LLM during extraction to validate dates.

    Args:
        date_string: Date string to validate (e.g., "2024-03-15", "March 2023", "15/03/2024")

    Returns:
        Dict with validation result:
        - valid (bool): True if date is valid and not in future
        - reason (str): Explanation if invalid, empty string if valid
        - normalized (str): Standardized date format (YYYY-MM-DD) if parseable, empty if not

    Examples:
        >>> validate_date("2024-03-15")
        {"valid": True, "reason": "", "normalized": "2024-03-15"}

        >>> validate_date("2099-01-01")
        {"valid": False, "reason": "Date is in the future", "normalized": "2099-01-01"}

        >>> validate_date("February 30, 2024")
        {"valid": False, "reason": "Could not parse date format", "normalized": ""}
    """
    if not date_string or not isinstance(date_string, str):
        return {
            "valid": False,
            "reason": "Date string is empty or invalid",
            "normalized": ""
        }

    # Strip whitespace
    date_string = date_string.strip()

    try:
        # Detect if date is already in ISO format (YYYY-MM-DD)
        # If so, don't use dayfirst to avoid misinterpretation
        import re
        iso_format = re.match(r'^\d{4}-\d{2}-\d{2}$', date_string)

        if iso_format:
            # Already in ISO format, parse without dayfirst
            parsed_date = parser.parse(date_string, fuzzy=True)
        else:
            # Parse with dayfirst=True for European/Italian DD/MM/YYYY format
            parsed_date = parser.parse(date_string, fuzzy=True, dayfirst=True)

        # Normalize to YYYY-MM-DD format
        normalized = parsed_date.strftime("%Y-%m-%d")

        # Get current date/time
        now = datetime.now()

        # Check if date is in the future
        if parsed_date > now:
            return {
                "valid": False,
                "reason": "Date is in the future",
                "normalized": normalized
            }

        # Date is valid
        return {
            "valid": True,
            "reason": "",
            "normalized": normalized
        }

    except ParserError:
        # Could not parse the date
        return {
            "valid": False,
            "reason": "Could not parse date format",
            "normalized": ""
        }
    except Exception as e:
        # Other errors (e.g., invalid date like Feb 30)
        return {
            "valid": False,
            "reason": f"Invalid date: {str(e)}",
            "normalized": ""
        }
