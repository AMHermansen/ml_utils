from typing import Any


def exists(value: Any) -> bool:
    """Check if a value is not None.

    Args:
        value: The value to check.

    Returns:
        bool: True if the value is not None, False otherwise.
    """
    return value is not None
