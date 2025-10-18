from typing import Any


def exists(value: Any) -> bool:
    """Check if a value is not None.

    Args:
        value: The value to check.

    Returns:
        bool: True if the value is not None, False otherwise.
    """
    return value is not None


def maybe_add(a: Any, b: Any) -> Any:
    """Add two values if both exist (are not None).

    Args:
        a: The first value.
        b: The second value.

    Returns:
        The sum of a and b if both exist, otherwise returns the existing value or None.
    """
    if exists(a) and exists(b):
        return a + b
    return None


def maybe_subtract(a: Any, b: Any) -> Any:
    """Subtract two values if both exist (are not None).

    Args:
        a: The first value.
        b: The second value.

    Returns:
        The difference of a and b if both exist, otherwise returns the existing value
        or None.
    """
    if exists(a) and exists(b):
        return a - b
    return None


def maybe_multiply(a: Any, b: Any) -> Any:
    """Multiply two values if both exist (are not None).

    Args:
        a: The first value.
        b: The second value.

    Returns:
        The product of a and b if both exist, otherwise returns the existing value
        or None.
    """
    if exists(a) and exists(b):
        return a * b
    return None
