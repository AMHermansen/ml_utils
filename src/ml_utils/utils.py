from collections.abc import Callable
from functools import wraps
from typing import Any, TypeGuard, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def exists(value: T | None) -> TypeGuard[T]:
    """Check if a value is not None.

    Args:
        value: The value to check.

    Returns:
        bool: True if the value is not None, False otherwise.
    """
    return value is not None


def default(value: T | None, default_value: T) -> T:
    """Return the value if it exists, otherwise return the default value.

    Args:
        value: The value to check.
        default_value: The default value to return if value is None.

    Returns:
        The value if it exists, otherwise the default value.
    """
    if exists(value):
        return value
    return default_value


def maybe_apply(func: Callable[[T], R | None], value: T | None) -> R | None:
    """Apply a function to a value if it exists (is not None).

    Args:
        func: The function to apply.
        value: The value to check and potentially apply the function to.

    Returns:
        The result of func(value) if value exists, otherwise None.
    """
    if exists(value):
        return func(value)
    return None


def maybe(func: Callable[[T], R | None]) -> Callable[[T | None], R | None]:
    """Decorator around maybe_apply.

    Args:
        func: The function to apply.

    Returns:
        A new function that applies func to its argument if it exists.
    """
    @wraps(func)
    def wrapper(value: T | None) -> R | None:
        return maybe_apply(func, value)

    return wrapper


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


def identity(x: T) -> T:
    """Return the input value unchanged."""
    return x
