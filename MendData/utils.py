"""Shared input-validation helpers for MendData.

These are internal helpers: every public method checks its inputs with them so
mistakes produce a clear, plain-English error. They are deliberately not
exported from ``MendData/__init__.py``.

Each helper returns the value it checked, so it can be used inline::

    self._weights = validate_fraction(weight, 'weight')
"""

from __future__ import annotations

import numbers
from typing import Any, Iterable

import pandas as pd

__all__ = ["validate_dataframe", "validate_choice", "validate_fraction"]


def validate_dataframe(data: Any, name: str = "data") -> pd.DataFrame:
    """Check that ``data`` is a pandas DataFrame.

    Args:
        data: The object to check.
        name: How to refer to the argument in the error message.

    Returns:
        ``data`` itself, unchanged.

    Raises:
        TypeError: If ``data`` is not a ``pandas.DataFrame``.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame, got {type(data).__name__}."
        )
    return data


def validate_choice(value: Any, allowed: Iterable[Any], name: str) -> Any:
    """Check that ``value`` is one of the allowed choices.

    Args:
        value: The value to check.
        allowed: The values that are accepted.
        name: How to refer to the argument in the error message.

    Returns:
        ``value`` itself, unchanged.

    Raises:
        ValueError: If ``value`` is not one of ``allowed``.
    """
    options = tuple(allowed)
    if value not in options:
        listed = ", ".join(repr(option) for option in options)
        raise ValueError(f"{name} must be one of {listed}; got {value!r}.")
    return value


def validate_fraction(value: Any, name: str) -> float:
    """Check that ``value`` is a number from 0 to 1 (inclusive).

    Args:
        value: The value to check. NumPy numbers are accepted.
        name: How to refer to the argument in the error message.

    Returns:
        ``value`` as a plain ``float``.

    Raises:
        TypeError: If ``value`` is not a number (booleans do not count).
        ValueError: If ``value`` is outside the range 0 to 1.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {number}.")
    return number