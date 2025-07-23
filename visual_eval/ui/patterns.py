"""
Common patterns and utilities following frontend design principles.
"""

from typing import Dict, Any, Tuple, List


def extract_test_case_data(
    test_case: Dict[str, Any]
) -> Tuple[str, List[str], List[str]]:
    """
    Extract data from test case with consistent pattern.

    Args:
        test_case: Test case data dictionary

    Returns:
        Tuple of (diff, actual_concern_types, shas)
    """
    diff = test_case.get("tangleChange", "")
    atomic_changes = test_case.get("atomicChanges", [])
    actual_concern_types = [change.get("label", "") for change in atomic_changes]
    shas = test_case.get("shas", [])
    return diff, actual_concern_types, shas
