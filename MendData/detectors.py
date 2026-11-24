"""Detectors for finding data-quality issues in pandas DataFrames.

This module defines ``BaseDetector`` and three concrete subclasses:
``MissingValueDetector``, ``DuplicateDetector``, and ``OutlierDetector``.
Each detector reads a DataFrame without mutating it and returns ``Issue``
objects whose severity is at least ``min_severity``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
import pandas as pd

from .issue import Issue

__all__ = [
    "BaseDetector",
    "MissingValueDetector",
    "DuplicateDetector",
    "OutlierDetector",
]


def _validate_dataframe(data: Any) -> pd.DataFrame:
    """Validate that ``data`` is a pandas DataFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"data must be a pandas DataFrame, got {type(data).__name__}."
        )
    return data


def _validate_severity(value: Any, name: str = "min_severity") -> float:
    """Validate that ``value`` is a real number (not bool) in [0.0, 1.0]."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(
            f"{name} must be a number between 0 and 1, got {type(value).__name__}."
        )
    number = float(value)
    if np.isnan(number) or not (0.0 <= number <= 1.0):
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")
    return number


def _validate_positive_threshold(value: Any, name: str = "threshold") -> float:
    """Validate that ``value`` is a positive real number (not bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(
            f"{name} must be a positive number, got {type(value).__name__}."
        )
    number = float(value)
    if np.isnan(number) or np.isinf(number) or number <= 0.0:
        raise ValueError(f"{name} must be greater than 0, got {value}.")
    return number


class BaseDetector(ABC):
    """Abstract base class for DataFrame issue detectors.

    Args:
        min_severity: Minimum severity threshold (from 0.0 to 1.0) required
            for a detected ``Issue`` to be kept. Defaults to ``0.0``.

    Raises:
        TypeError: If ``min_severity`` is not a number.
        ValueError: If ``min_severity`` is outside 0.0 to 1.0.
    """

    def __init__(self, min_severity: float = 0.0) -> None:
        """Initialize the detector with a minimum severity threshold."""
        self._min_severity = _validate_severity(min_severity, "min_severity")

    @property
    def min_severity(self) -> float:
        """Minimum severity (from 0.0 to 1.0) for returned issues."""
        return self._min_severity

    def _prepare_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Validate ``data`` and return ``None`` if it is empty."""
        _validate_dataframe(data)
        if data.empty:
            return None
        return data

    def _filter_by_severity(self, issues: list[Issue]) -> list[Issue]:
        """Return only issues whose severity is at least ``min_severity``."""
        return [
            issue for issue in issues if issue.severity >= self._min_severity
        ]

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> list[Issue]:
        """Scan ``data`` and return issues with severity >= ``min_severity``.

        Args:
            data: The pandas DataFrame to inspect.

        Returns:
            A list of ``Issue`` objects meeting ``min_severity``. Returns an
            empty list if ``data`` is empty or no qualifying issues are found.

        Raises:
            TypeError: If ``data`` is not a pandas DataFrame.
        """


class MissingValueDetector(BaseDetector):
    """Detect missing values in each column of a DataFrame.

    Emits at most one ``Issue`` per column that contains missing values,
    with severity equal to ``missing_count / len(data)``.

    Args:
        min_severity: Minimum severity threshold (from 0.0 to 1.0).
            Defaults to ``0.0``.
    """

    def __init__(self, min_severity: float = 0.0) -> None:
        """Initialize the missing-value detector."""
        super().__init__(min_severity=min_severity)

    def detect(self, data: pd.DataFrame) -> list[Issue]:
        """Find columns with missing values and return qualifying issues.

        Args:
            data: The pandas DataFrame to inspect.

        Returns:
            A list of ``Issue.MISSING`` items with severity >= ``min_severity``.

        Raises:
            TypeError: If ``data`` is not a pandas DataFrame.
        """
        prepared = self._prepare_data(data)
        if prepared is None:
            return []

        total_rows = len(prepared)
        issues: list[Issue] = []

        for col in prepared.columns:
            missing_count = int(prepared[col].isna().sum())
            if missing_count > 0:
                severity = float(missing_count) / float(total_rows)
                issues.append(
                    Issue(
                        column=col,
                        kind=Issue.MISSING,
                        count=missing_count,
                        severity=severity,
                        details=None,
                    )
                )

        return self._filter_by_severity(issues)


class DuplicateDetector(BaseDetector):
    """Detect duplicate rows in a DataFrame.

    Emits at most one row-level ``Issue`` (``column=None``) when duplicate
    rows exist, with severity equal to ``duplicate_count / len(data)`` and
    ``details={'subset': ...}``.

    Args:
        subset: Optional iterable of column names to check for duplicates.
            If ``None``, all columns are checked.
        min_severity: Minimum severity threshold (from 0.0 to 1.0).
            Defaults to ``0.0``.

    Raises:
        TypeError: If ``subset`` is neither ``None`` nor an iterable of
            column labels, or if ``min_severity`` is invalid.
        ValueError: If ``min_severity`` is out of range, or if a column in
            ``subset`` is missing from ``data.columns`` when ``detect()`` runs.
    """

    def __init__(
        self,
        subset: Optional[Iterable[Any]] = None,
        min_severity: float = 0.0,
    ) -> None:
        """Initialize the duplicate-row detector."""
        super().__init__(min_severity=min_severity)
        if subset is None:
            self._subset: Optional[list[Any]] = None
        else:
            if isinstance(subset, (str, bytes)) or not isinstance(subset, Iterable):
                raise TypeError(
                    "subset must be None or an iterable of column names, "
                    f"got {type(subset).__name__}."
                )
            self._subset = list(subset)

    @property
    def subset(self) -> Optional[list[Any]]:
        """A copy of the column subset used for duplicate checks, or ``None``."""
        if self._subset is None:
            return None
        return list(self._subset)

    def detect(self, data: pd.DataFrame) -> list[Issue]:
        """Find duplicate rows and return a qualifying ``Issue.DUPLICATE``.

        Args:
            data: The pandas DataFrame to inspect.

        Returns:
            A list containing at most one ``Issue.DUPLICATE`` item meeting
            ``min_severity``.

        Raises:
            TypeError: If ``data`` is not a pandas DataFrame.
            ValueError: If any column in ``subset`` is not in ``data.columns``.
        """
        prepared = self._prepare_data(data)
        if prepared is None:
            return []

        subset_copy = self.subset
        if subset_copy is not None:
            missing_cols = [
                col for col in subset_copy if col not in prepared.columns
            ]
            if missing_cols:
                raise ValueError(
                    "subset contains unknown column(s): "
                    + ", ".join(repr(col) for col in missing_cols)
                    + "."
                )

        dup_count = int(prepared.duplicated(subset=subset_copy).sum())
        if dup_count == 0:
            return []

        severity = float(dup_count) / float(len(prepared))
        issues = [
            Issue(
                column=None,
                kind=Issue.DUPLICATE,
                count=dup_count,
                severity=severity,
                details={"subset": subset_copy},
            )
        ]
        return self._filter_by_severity(issues)


class OutlierDetector(BaseDetector):
    """Detect numeric outliers in each column using IQR or z-score fences.

    Inspects numeric columns only (excluding booleans), ignores missing
    values, and skips columns with fewer than 4 non-missing values or zero
    spread. Emits at most one ``Issue.OUTLIER`` per qualifying column with
    ``details={'method': ..., 'lower': float, 'upper': float}``.

    Args:
        method: Outlier detection method: ``'iqr'`` or ``'zscore'``.
            Defaults to ``'iqr'``.
        threshold: Multiplier ``k`` for bounds. Defaults to ``1.5`` for
            ``'iqr'`` and ``3.0`` for ``'zscore'``.
        min_severity: Minimum severity threshold (from 0.0 to 1.0).
            Defaults to ``0.0``.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If ``method`` is unsupported, ``threshold`` is not
            positive, or ``min_severity`` is outside 0.0 to 1.0.
    """

    VALID_METHODS = ("iqr", "zscore")
    DEFAULT_THRESHOLDS = {"iqr": 1.5, "zscore": 3.0}
    MIN_NON_MISSING = 4

    def __init__(
        self,
        method: str = "iqr",
        threshold: Optional[float] = None,
        min_severity: float = 0.0,
    ) -> None:
        """Initialize the outlier detector."""
        super().__init__(min_severity=min_severity)
        if not isinstance(method, str):
            raise TypeError(f"method must be a string, got {type(method).__name__}.")
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of 'iqr' or 'zscore', got {method!r}."
            )

        resolved_threshold = (
            self.DEFAULT_THRESHOLDS[method] if threshold is None else threshold
        )
        self._method = method
        self._threshold = _validate_positive_threshold(resolved_threshold, "threshold")

    @property
    def method(self) -> str:
        """Outlier detection method (``'iqr'`` or ``'zscore'``)."""
        return self._method

    @property
    def threshold(self) -> float:
        """Positive multiplier ``k`` used to compute lower and upper bounds."""
        return self._threshold

    def _compute_bounds(self, series: pd.Series) -> Optional[tuple[float, float]]:
        """Compute ``(lower, upper)`` bounds for a non-empty numeric Series.

        Returns ``None`` if the column has fewer than 4 non-missing values
        or zero spread.
        """
        clean = series.dropna().astype(float)
        if len(clean) < self.MIN_NON_MISSING:
            return None

        if self._method == "iqr":
            q1 = float(np.percentile(clean, 25))
            q3 = float(np.percentile(clean, 75))
            iqr = q3 - q1
            if iqr == 0.0:
                return None
            lower = q1 - self._threshold * iqr
            upper = q3 + self._threshold * iqr
            return (float(lower), float(upper))

        mean_val = float(clean.mean())
        std_val = float(clean.std(ddof=1))
        if std_val == 0.0 or np.isnan(std_val):
            return None
        lower = mean_val - self._threshold * std_val
        upper = mean_val + self._threshold * std_val
        return (float(lower), float(upper))

    def detect(self, data: pd.DataFrame) -> list[Issue]:
        """Find numeric column outliers and return qualifying issues.

        Args:
            data: The pandas DataFrame to inspect.

        Returns:
            A list of ``Issue.OUTLIER`` items with severity >= ``min_severity``.

        Raises:
            TypeError: If ``data`` is not a pandas DataFrame.
        """
        prepared = self._prepare_data(data)
        if prepared is None:
            return []

        numeric_cols = prepared.select_dtypes(
            include=[np.number], exclude=[bool, "boolean"]
        ).columns
        issues: list[Issue] = []

        for col in numeric_cols:
            bounds = self._compute_bounds(prepared[col])
            if bounds is None:
                continue

            lower, upper = bounds
            non_null = prepared[col].dropna().astype(float)
            non_null_count = len(non_null)
            outlier_mask = (non_null < lower) | (non_null > upper)
            outlier_count = int(outlier_mask.sum())

            if outlier_count > 0:
                severity = float(outlier_count) / float(non_null_count)
                issues.append(
                    Issue(
                        column=col,
                        kind=Issue.OUTLIER,
                        count=outlier_count,
                        severity=severity,
                        details={
                            "method": self._method,
                            "lower": lower,
                            "upper": upper,
                        },
                    )
                )

        return self._filter_by_severity(issues)