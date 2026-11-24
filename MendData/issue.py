"""The ``Issue`` class: the shared contract between detectors and fixers.

Detectors create ``Issue`` objects, fixers read them, and ``CleaningSession``
passes them around. This module depends only on the standard library, so every
teammate can import it from day one.

Kinds of issue
    ``Issue.MISSING``    missing values in one column
    ``Issue.DUPLICATE``  repeated rows (row-level, so ``column`` is ``None``)
    ``Issue.OUTLIER``    extreme numeric values in one column

Severity
    A float from 0 to 1: the share of what was checked that is affected.
    Missing: missing cells / rows. Duplicate: duplicate rows / rows.
    Outlier: outliers / non-missing values in the column. It is used for
    ranking only and is not weighted across kinds.

Details (filled in by detectors, read by fixers)
    missing    no keys needed
    duplicate  ``subset``: list of column names, or ``None`` for all columns
    outlier    ``method`` ('iqr' or 'zscore'), plus the ``lower`` and
               ``upper`` bounds that were used to flag outliers
"""

from __future__ import annotations

import numbers
from collections.abc import Hashable, Mapping
from typing import Any, Optional

__all__ = ["Issue"]


def _plural(count: int, word: str) -> str:
    """Return ``word`` with an ``s`` added unless ``count`` is exactly 1."""
    return word if count == 1 else word + "s"


class Issue:
    """One data-quality problem found in a dataset.

    An ``Issue`` is immutable: its values are stored in private attributes
    and exposed through read-only properties, so nothing can change it after
    it is created. Detectors build issues, ``CleaningSession`` ranks them,
    and fixers read ``column``, ``kind`` and ``details`` to decide what to do.

    Args:
        column: The column label (for example ``'age'``), or ``None`` for a
            row-level issue. Only ``DUPLICATE`` issues may use ``None``.
        kind: One of ``Issue.MISSING``, ``Issue.DUPLICATE`` or
            ``Issue.OUTLIER``.
        count: How many values or rows are affected (0 or more). NumPy
            integers are accepted and converted to ``int``.
        severity: The share of what was checked that is affected, from 0 to
            1. NumPy floats are accepted and converted to ``float``.
        details: Optional mapping with extra facts for the fixer, such as
            ``{'method': 'iqr', 'lower': 1.0, 'upper': 9.0}``. It is copied.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``kind`` is unknown, ``column`` is ``None`` for a kind
            that needs one, ``count`` is negative, or ``severity`` is outside
            the range 0 to 1.

    Example:
        >>> issue = Issue('age', Issue.MISSING, count=12, severity=0.06)
        >>> issue.describe()
        "Column 'age' has 12 missing values (6.0% of its rows)."
        >>> issue
        Issue(column='age', kind='missing', count=12, severity=0.060)
        >>> outlier = Issue('pay', Issue.OUTLIER, 3, 0.2,
        ...                 details={'method': 'iqr', 'lower': 1.0, 'upper': 9.0})
        >>> [i.kind for i in sorted([issue, outlier], reverse=True)]
        ['outlier', 'missing']
    """

    MISSING = "missing"
    DUPLICATE = "duplicate"
    OUTLIER = "outlier"
    KINDS = (MISSING, DUPLICATE, OUTLIER)

    def __init__(
        self,
        column: Optional[Hashable],
        kind: str,
        count: int,
        severity: float,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Validate the arguments and store them in private attributes.

        See the class docstring for the meaning of each argument and for the
        errors that can be raised.
        """
        if not isinstance(kind, str):
            raise TypeError(f"kind must be a string, got {type(kind).__name__}.")
        if kind not in self.KINDS:
            raise ValueError(
                f"kind must be one of {', '.join(self.KINDS)}; got {kind!r}."
            )

        if column is None:
            if kind != self.DUPLICATE:
                raise ValueError(
                    f"column is required for {kind!r} issues; "
                    "only 'duplicate' issues may use None."
                )
        elif not isinstance(column, Hashable):
            raise TypeError("column must be a column label (such as a string) or None.")

        if isinstance(count, bool) or not isinstance(count, numbers.Integral):
            raise TypeError(f"count must be an integer, got {type(count).__name__}.")
        if count < 0:
            raise ValueError(f"count cannot be negative, got {count}.")

        if isinstance(severity, bool) or not isinstance(severity, numbers.Real):
            raise TypeError(
                f"severity must be a number, got {type(severity).__name__}."
            )
        severity = float(severity)
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be between 0 and 1, got {severity}.")

        if details is None:
            details = {}
        elif not isinstance(details, Mapping):
            raise TypeError(
                f"details must be a mapping or None, got {type(details).__name__}."
            )

        self._column = column
        self._kind = kind
        self._count = int(count)
        self._severity = severity
        self._details = dict(details)

    # ------------------------------------------------------------------
    # Read-only properties (encapsulation)
    # ------------------------------------------------------------------
    @property
    def column(self) -> Optional[Hashable]:
        """The column label, or ``None`` for a row-level issue."""
        return self._column

    @property
    def kind(self) -> str:
        """The kind of issue: ``'missing'``, ``'duplicate'`` or ``'outlier'``."""
        return self._kind

    @property
    def count(self) -> int:
        """How many values or rows are affected."""
        return self._count

    @property
    def severity(self) -> float:
        """The share of what was checked that is affected (0 to 1)."""
        return self._severity

    @property
    def details(self) -> dict:
        """A copy of the extra facts the detector saved for the fixer."""
        return dict(self._details)

    # ------------------------------------------------------------------
    # Plain-English description
    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Describe the issue in one plain-English sentence.

        Returns:
            A sentence such as ``"Column 'age' has 12 missing values (6.0% of
            its rows)."``.
        """
        pct = f"{self._severity:.1%}"
        n = self._count

        if self._kind == self.MISSING:
            return (
                f"Column '{self._column}' has {n} missing "
                f"{_plural(n, 'value')} ({pct} of its rows)."
            )

        if self._kind == self.DUPLICATE:
            sentence = f"Found {n} duplicate {_plural(n, 'row')} ({pct} of all rows)"
            subset = self._details.get("subset")
            if subset:
                sentence += ", comparing only the columns " + ", ".join(
                    str(name) for name in subset
                )
            return sentence + "."

        return (
            f"Column '{self._column}' has {n} {_plural(n, 'outlier')} "
            f"({pct} of its values)."
        )

    # ------------------------------------------------------------------
    # Dunder methods (polymorphism through operator overloading)
    # ------------------------------------------------------------------
    def _key(self) -> tuple:
        """The fields that define equality and hashing (details are ignored)."""
        return (self._column, self._kind, self._count, self._severity)

    def __repr__(self) -> str:
        """Return a developer-friendly string (severity rounded to 3 places)."""
        return (
            f"Issue(column={self._column!r}, kind={self._kind!r}, "
            f"count={self._count!r}, severity={self._severity:.3f})"
        )

    def __str__(self) -> str:
        """Return the plain-English description, so ``print(issue)`` is readable."""
        return self.describe()

    def __eq__(self, other: object) -> bool:
        """Two issues are equal if column, kind, count and severity match."""
        if not isinstance(other, Issue):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        """Hash the same fields as ``__eq__`` so issues work in sets and dicts."""
        return hash(self._key())

    def __lt__(self, other: object) -> bool:
        """Compare by severity only, so issues can be ranked.

        Python also uses this method for ``>``. ``<=`` and ``>=`` are
        intentionally not supported: two different issues can share a
        severity without being equal. Rank with
        ``sorted(issues, reverse=True)`` to put the worst issue first.
        """
        if not isinstance(other, Issue):
            return NotImplemented
        return self._severity < other._severity