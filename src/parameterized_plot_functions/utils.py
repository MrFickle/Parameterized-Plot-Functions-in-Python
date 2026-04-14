from collections.abc import Sequence


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def is_sequence_like(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))