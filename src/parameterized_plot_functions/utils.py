"""Small shared helpers for normalizing input values."""

from collections.abc import Sequence
from typing import Any


def ensure_list(value: Any) -> list[Any]:
    """
    Function purpose:
        Convert an optional value into a list.

    Args:
        value: Value to normalize. ``None`` becomes an empty list, lists are
            returned unchanged, and all other values are wrapped in a list.

    Outputs:
        A list representation of ``value``.
    """
    # Treat missing values as an empty collection.
    if value is None:
        return []

    # Preserve existing list objects without copying.
    if isinstance(value, list):
        return value

    # Wrap scalar or non-list values so callers can iterate consistently.
    return [value]


def is_sequence_like(value: Any) -> bool:
    """
    Function purpose:
        Check whether a value behaves like a sequence without treating strings
        as data sequences.

    Args:
        value: Value to inspect.

    Outputs:
        ``True`` when ``value`` is a non-string sequence, otherwise ``False``.
    """
    # Strings and bytes are sequences in Python, but they should behave as scalars here.
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))
