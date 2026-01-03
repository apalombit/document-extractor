"""Test helper utilities for e2e testing"""
import json
from typing import Dict, Tuple, List
from pathlib import Path


def load_expected_outputs(txt_path: str) -> Dict:
    """
    Parse expected outputs from .txt file.

    Format: Each line contains "task_name: {json_result}"
    Example:
        author_date: {'authors': ['Dr. Smith'], 'date': '2023-01-01'}
        keywords: {'keywords': ['medical', 'test']}
        document_type: {'document_type': 'medical'}

    Args:
        txt_path: Path to .txt file with expected outputs

    Returns:
        Dictionary mapping task names to expected results:
        {
            "author_date": {"authors": [...], "date": "..."},
            "keywords": {"keywords": [...]},
            "document_type": {"document_type": "..."}
        }
    """
    expected = {}
    txt_file = Path(txt_path)

    if not txt_file.exists():
        raise FileNotFoundError(f"Expected output file not found: {txt_path}")

    with open(txt_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                # Skip empty lines and comments
                continue

            # Parse format: "task_name: {json_content}"
            if ':' not in line:
                raise ValueError(f"Invalid format in {txt_path} line {line_num}: missing colon")

            task_name, json_str = line.split(':', 1)
            task_name = task_name.strip()
            json_str = json_str.strip()

            try:
                # Parse JSON (handle Python dict syntax with single quotes)
                # Convert single quotes to double quotes for JSON compatibility
                json_str = json_str.replace("'", '"')
                result = json.loads(json_str)
                expected[task_name] = result
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {txt_path} line {line_num}: {json_str}\n"
                    f"Error: {str(e)}"
                )

    return expected


def compare_extraction_results(actual: Dict, expected: Dict) -> Tuple[bool, List[str]]:
    """
    Compare actual extraction results against expected ground truth.

    Comparison strategy per task:
    - author_date: Exact match for authors list (order matters) and date string
    - keywords: Set comparison (order doesn't matter)
    - document_type: Exact string match
    - Null values: Exact match (null == null passes)

    Args:
        actual: Actual extraction results from workflow
        expected: Expected results from load_expected_outputs()

    Returns:
        Tuple of (all_passed: bool, differences: List[str])
        - all_passed: True if all comparisons passed
        - differences: List of human-readable difference descriptions
    """
    differences = []
    all_passed = True

    # Check all expected tasks are present in actual
    for task in expected.keys():
        if task not in actual:
            differences.append(f"Task '{task}' missing from actual results")
            all_passed = False
            continue

        # Compare based on task type
        if task == "keywords":
            # Keywords: set comparison (order independent)
            passed, diff = _compare_keywords(actual[task], expected[task], task)
        elif task == "author_date":
            # Author/Date: exact match (order matters for authors list)
            passed, diff = _compare_author_date(actual[task], expected[task], task)
        elif task == "document_type":
            # Document type: exact string match
            passed, diff = _compare_document_type(actual[task], expected[task], task)
        else:
            # Generic exact match for unknown tasks
            passed, diff = _compare_exact(actual[task], expected[task], task)

        if not passed:
            all_passed = False
            differences.extend(diff)

    return all_passed, differences


def _compare_keywords(actual: Dict, expected: Dict, task_name: str) -> Tuple[bool, List[str]]:
    """Compare keywords using set comparison (order independent)"""
    differences = []

    actual_keywords = actual.get("keywords")
    expected_keywords = expected.get("keywords")

    # Handle null cases
    if actual_keywords is None and expected_keywords is None:
        return True, []
    if actual_keywords is None:
        differences.append(f"{task_name}: Expected keywords {expected_keywords}, got null")
        return False, differences
    if expected_keywords is None:
        differences.append(f"{task_name}: Expected null, got keywords {actual_keywords}")
        return False, differences

    # Convert to sets for comparison
    actual_set = set(actual_keywords)
    expected_set = set(expected_keywords)

    if actual_set != expected_set:
        missing = expected_set - actual_set
        extra = actual_set - expected_set

        msg = f"{task_name} mismatch:"
        if missing:
            msg += f"\n  Missing keywords: {sorted(missing)}"
        if extra:
            msg += f"\n  Extra keywords: {sorted(extra)}"
        msg += f"\n  Expected: {sorted(expected_keywords)}"
        msg += f"\n  Got: {sorted(actual_keywords)}"

        differences.append(msg)
        return False, differences

    return True, []


def _compare_author_date(actual: Dict, expected: Dict, task_name: str) -> Tuple[bool, List[str]]:
    """Compare author_date with exact matching"""
    differences = []

    # Compare authors
    actual_authors = actual.get("authors")
    expected_authors = expected.get("authors")

    if actual_authors != expected_authors:
        differences.append(
            f"{task_name}.authors mismatch:\n"
            f"  Expected: {expected_authors}\n"
            f"  Got: {actual_authors}"
        )

    # Compare date
    actual_date = actual.get("date")
    expected_date = expected.get("date")

    if actual_date != expected_date:
        differences.append(
            f"{task_name}.date mismatch:\n"
            f"  Expected: {expected_date}\n"
            f"  Got: {actual_date}"
        )

    return len(differences) == 0, differences


def _compare_document_type(actual: Dict, expected: Dict, task_name: str) -> Tuple[bool, List[str]]:
    """Compare document_type with exact string matching"""
    differences = []

    actual_type = actual.get("document_type")
    expected_type = expected.get("document_type")

    if actual_type != expected_type:
        differences.append(
            f"{task_name} mismatch:\n"
            f"  Expected: {expected_type}\n"
            f"  Got: {actual_type}"
        )
        return False, differences

    return True, []


def _compare_exact(actual: Dict, expected: Dict, task_name: str) -> Tuple[bool, List[str]]:
    """Generic exact match comparison"""
    differences = []

    if actual != expected:
        differences.append(
            f"{task_name} mismatch:\n"
            f"  Expected: {expected}\n"
            f"  Got: {actual}"
        )
        return False, differences

    return True, []
