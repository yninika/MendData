"""Data-cleaning fixer classes for MendData."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from .issue import Issue
from .utils import validate_choice, validate_dataframe

__all__ = [
    "BaseFixer",
    "ImputeFixer",
    "DropFixer",
    "ClipFixer",
]


def _is_true_numeric_dtype(series: pd.Series) -> bool:
    """Return True if a Series is numeric and not boolean.

    Args:
        series: Series to inspect.

    Returns:
        True when the Series has a numeric dtype excluding booleans,
        otherwise False.
    """
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _validate_issue(issue: Any) -> Issue:
    """Validate that an object is an Issue instance.

    Args:
        issue: Candidate issue object.

    Returns:
        The validated Issue instance.

    Raises:
        TypeError: If issue is not an Issue instance.
    """
    if not isinstance(issue, Issue):
        raise TypeError(
            f"Expected an Issue instance, got {type(issue).__name__}."
        )
    return issue


def _validate_column_present(data: pd.DataFrame, issue: Issue) -> Any:
    """Validate and return the column name for a column-level Issue.

    Args:
        data: Target DataFrame.
        issue: Issue referencing a column.

    Returns:
        The validated column label from issue.

    Raises:
        ValueError: If issue.column is None or missing from data.columns.
    """
    if issue.column is None:
        raise ValueError(
            f"Issue of kind '{issue.kind}' requires a non-None column name."
        )
    if issue.column not in data.columns:
        raise ValueError(
            f"Column '{issue.column}' is not present in the DataFrame."
        )
    return issue.column


class BaseFixer(ABC):
    """Abstract base class for fixers that resolve data quality issues.

    Attributes:
        HANDLES: Tuple of Issue.kind values that this fixer can resolve.
    """

    HANDLES: tuple[str, ...] = ()

    def __init__(self, strategy: str) -> None:
        """Initialize the fixer with a strategy name.

        Args:
            strategy: Name of the fixing strategy shown in audit logs.

        Raises:
            TypeError: If strategy is not a string.
            ValueError: If strategy is empty or whitespace-only.
        """
        if not isinstance(strategy, str):
            raise TypeError(
                f"Expected strategy to be a str, got {type(strategy).__name__}."
            )
        if not strategy.strip():
            raise ValueError("Strategy name must not be empty.")
        self._strategy = strategy

    @property
    def strategy(self) -> str:
        """Return the fixer strategy name.

        Returns:
            Strategy identifier string.
        """
        return self._strategy

    def can_fix(self, issue: Issue) -> bool:
        """Check whether this fixer supports the given issue kind.

        Args:
            issue: Issue instance to inspect.

        Returns:
            True if issue.kind is in HANDLES, otherwise False.

        Raises:
            TypeError: If issue is not an Issue instance.
        """
        validated_issue = _validate_issue(issue)
        return validated_issue.kind in self.HANDLES

    @abstractmethod
    def fix(self, data: pd.DataFrame, issue: Issue) -> pd.DataFrame:
        """Apply the fix for an issue to a copy of the DataFrame.

        Args:
            data: Input DataFrame to clean without mutating the original.
            issue: Issue describing the problem to resolve.

        Returns:
            A new DataFrame with the fix applied.

        Raises:
            TypeError: If data is not a DataFrame or issue is not an Issue.
            ValueError: If the fixer cannot handle the issue or inputs are invalid.
        """

    @abstractmethod
    def explain(self, issue: Issue) -> str:
        """Return a plain-English present-tense explanation of the fix.

        Args:
            issue: Issue being addressed.

        Returns:
            One present-tense sentence describing the action taken.

        Raises:
            TypeError: If issue is not an Issue instance.
            ValueError: If the fixer cannot handle the issue.
        """


class ImputeFixer(BaseFixer):
    """Fixer that fills missing values in a column using a chosen strategy.

    Attributes:
        HANDLES: Tuple containing only Issue.MISSING.
        VALID_STRATEGIES: Supported imputation strategies ('mean', 'median', 'mode').
    """

    HANDLES: tuple[str, ...] = (Issue.MISSING,)
    VALID_STRATEGIES: tuple[str, ...] = ("mean", "median", "mode")

    def __init__(self, strategy: str = "median") -> None:
        """Initialize ImputeFixer with an imputation strategy.

        Args:
            strategy: Imputation strategy ('mean', 'median', or 'mode').
                Defaults to 'median'.

        Raises:
            TypeError: If strategy is not a string.
            ValueError: If strategy is not one of 'mean', 'median', or 'mode'.
        """
        if not isinstance(strategy, str):
            raise TypeError(
                f"Expected strategy to be a str, got {type(strategy).__name__}."
            )
        validate_choice(strategy, self.VALID_STRATEGIES, "strategy")
        super().__init__(strategy)

    def _compute_fill_value(self, series: pd.Series) -> tuple[bool, Any]:
        """Compute the replacement value for missing entries in a Series.

        Args:
            series: Target column Series.

        Returns:
            A tuple (has_value, fill_value) where has_value indicates whether
            a valid non-missing replacement value was found.
        """
        if series.dropna().empty:
            return False, None

        effective_strategy = self._strategy
        if effective_strategy in ("mean", "median") and not _is_true_numeric_dtype(
            series
        ):
            effective_strategy = "mode"

        if effective_strategy == "mean":
            val = series.mean()
            if pd.isna(val):
                return False, None
            return True, val

        if effective_strategy == "median":
            val = series.median()
            if pd.isna(val):
                return False, None
            return True, val

        modes = series.mode(dropna=True)
        if modes.empty:
            return False, None
        return True, modes.iloc[0]

    def fix(self, data: pd.DataFrame, issue: Issue) -> pd.DataFrame:
        """Fill missing values in issue.column and return a new DataFrame.

        Args:
            data: Input DataFrame.
            issue: Missing-value Issue to fix.

        Returns:
            A new DataFrame with missing values in issue.column imputed when
            a usable replacement value exists.

        Raises:
            TypeError: If data is not a DataFrame or issue is not an Issue.
            ValueError: If issue cannot be fixed by ImputeFixer or column is missing.
        """
        validate_dataframe(data)
        validated_issue = _validate_issue(issue)
        if not self.can_fix(validated_issue):
            raise ValueError(
                f"ImputeFixer cannot fix issue of kind '{validated_issue.kind}'."
            )

        col = _validate_column_present(data, validated_issue)
        result = data.copy()

        has_value, fill_value = self._compute_fill_value(result[col])
        if not has_value:
            return result

        result[col] = result[col].fillna(fill_value)
        return result

    def explain(self, issue: Issue) -> str:
        """Explain how ImputeFixer addresses the given issue in plain English.

        Args:
            issue: Missing-value Issue to describe.

        Returns:
            A single present-tense sentence describing the imputation.

        Raises:
            TypeError: If issue is not an Issue instance.
            ValueError: If issue cannot be fixed by ImputeFixer or column is None.
        """
        validated_issue = _validate_issue(issue)
        if not self.can_fix(validated_issue):
            raise ValueError(
                f"ImputeFixer cannot explain issue of kind '{validated_issue.kind}'."
            )
        if validated_issue.column is None:
            raise ValueError("Missing issue requires a column name.")

        return (
            f"Replace {validated_issue.count} missing values in "
            f"'{validated_issue.column}' using the column {self._strategy}."
        )


class DropFixer(BaseFixer):
    """Fixer that removes rows affected by missing values or duplicate records.

    Attributes:
        HANDLES: Tuple of supported issue kinds (Issue.MISSING, Issue.DUPLICATE).
    """

    HANDLES: tuple[str, ...] = (Issue.MISSING, Issue.DUPLICATE)

    def __init__(self) -> None:
        """Initialize DropFixer with strategy name 'drop'."""
        super().__init__("drop")

    def _resolve_duplicate_subset(
        self, data: pd.DataFrame, issue: Issue
    ) -> list[str] | None:
        """Validate and normalize the subset column list for duplicate issues.

        Args:
            data: Target DataFrame.
            issue: Duplicate Issue containing optional details['subset'].

        Returns:
            A list of column names or None to evaluate all columns.

        Raises:
            ValueError: If details['subset'] is invalid or references unknown columns.
        """
        details = issue.details or {}
        if "subset" not in details or details["subset"] is None:
            return None

        raw_subset = details["subset"]
        if isinstance(raw_subset, (str, bytes)) or not isinstance(
            raw_subset, (list, tuple)
        ):
            raise ValueError(
                "Duplicate issue 'subset' detail must be a list/tuple of column names or None."
            )

        subset_list = list(raw_subset)
        missing_cols = [c for c in subset_list if c not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Subset columns not found in DataFrame: {missing_cols}."
            )
        return subset_list

    def fix(self, data: pd.DataFrame, issue: Issue) -> pd.DataFrame:
        """Drop rows with missing values or duplicate rows and return a new DataFrame.

        Args:
            data: Input DataFrame.
            issue: Issue of kind Issue.MISSING or Issue.DUPLICATE.

        Returns:
            A new DataFrame with matching rows dropped.

        Raises:
            TypeError: If data is not a DataFrame or issue is not an Issue.
            ValueError: If issue cannot be fixed by DropFixer or column/subset is invalid.
        """
        validate_dataframe(data)
        validated_issue = _validate_issue(issue)
        if not self.can_fix(validated_issue):
            raise ValueError(
                f"DropFixer cannot fix issue of kind '{validated_issue.kind}'."
            )

        result = data.copy()
        if validated_issue.kind == Issue.MISSING:
            col = _validate_column_present(result, validated_issue)
            return result.dropna(subset=[col])

        subset = self._resolve_duplicate_subset(result, validated_issue)
        return result.drop_duplicates(subset=subset)

    def explain(self, issue: Issue) -> str:
        """Explain how DropFixer resolves the given issue in plain English.

        Args:
            issue: Issue of kind Issue.MISSING or Issue.DUPLICATE.

        Returns:
            A single present-tense sentence describing the row removal.

        Raises:
            TypeError: If issue is not an Issue instance.
            ValueError: If issue cannot be fixed by DropFixer or column is invalid.
        """
        validated_issue = _validate_issue(issue)
        if not self.can_fix(validated_issue):
            raise ValueError(
                f"DropFixer cannot explain issue of kind '{validated_issue.kind}'."
            )

        if validated_issue.kind == Issue.MISSING:
            if validated_issue.column is None:
                raise ValueError("Missing issue requires a column name.")
            return (
                f"Drop rows with missing values in '{validated_issue.column}'."
            )

        return f"Remove {validated_issue.count} duplicate rows from the dataset."


class ClipFixer(BaseFixer):
    """Fixer that clips numeric column values to detector-provided bounds.

    Attributes:
        HANDLES: Tuple containing only Issue.OUTLIER.
    """

    HANDLES: tuple[str, ...] = (Issue.OUTLIER,)

    def __init__(self) -> None:
        """Initialize ClipFixer with strategy name 'clip'."""
        super().__init__("clip")

    def _validate_bound(self, bound: Any, name: str) -> float:
        """Validate that an outlier bound is a finite real number.

        Args:
            bound: Bound value from issue.details.
            name: Bound label ('lower' or 'upper') for error messages.

        Returns:
            The bound converted to float.

        Raises:
            ValueError: If bound is missing, boolean, non-numeric, or non-finite.
        """
        if bound is None or isinstance(bound, bool) or not isinstance(bound, (int, float, np.number)):
            raise ValueError(
                f"Outlier issue requires a numeric '{name}' bound in details."
            )
        value = float(bound)
        if not np.isfinite(value):
            raise ValueError(
                f"Outlier issue '{name}' bound must be finite, got {bound}."
            )
        return value

    def _extract_bounds(self, issue: Issue) -> tuple[float, float]:
        """Extract and validate lower and upper clipping bounds from issue.details.

        Args:
            issue: Outlier Issue containing 'lower' and 'upper' in details.

        Returns:
            Tuple of (lower, upper) bounds as floats.

        Raises:
            ValueError: If bounds are missing, invalid, or lower exceeds upper.
        """
        details = issue.details
        if not isinstance(details, dict) or "lower" not in details or "upper" not in details:
            raise ValueError(
                "Outlier issue details must include both 'lower' and 'upper' bounds."
            )

        lower = self._validate_bound(details["lower"], "lower")
        upper = self._validate_bound(details["upper"], "upper")
        if lower > upper:
            raise ValueError(
                f"Lower bound ({lower}) cannot be greater than upper bound ({upper})."
            )
        return lower, upper

    def fix(self, data: pd.DataFrame, issue: Issue) -> pd.DataFrame:
        """Clip values in issue.column to lower and upper bounds on a new DataFrame.

        Args:
            data: Input DataFrame.
            issue: Outlier Issue with lower and upper bounds in details.

        Returns:
            A new DataFrame with issue.column clipped to [lower, upper].

        Raises:
            TypeError: If data is not a DataFrame or issue is not an Issue.
            ValueError: If issue is unsupported, column is missing, or bounds are invalid.
        """
        validate_dataframe(data)
        validated_issue = _validate_issue(issue)
        if not self.can_fix(validated_issue):
            raise ValueError(
                f"ClipFixer cannot fix issue of kind '{validated_issue.kind}'."
            )

        col = _validate_column_present(data, validated_issue)
        lower, upper = self._extract_bounds(validated_issue)

        result = data.copy()
        if not _is_true_numeric_dtype(result[col]):
            raise ValueError(
                f"Column '{col}' must be numeric to apply clipping."
            )

        result[col] = result[col].clip(lower=lower, upper=upper)
        return result

    def explain(self, issue: Issue) -> str:
        """Explain how ClipFixer clips values for the given outlier issue.

        Args:
            issue: Outlier Issue with lower and upper bounds in details.

        Returns:
            A single present-tense sentence describing the clipping bounds.

        Raises:
            TypeError: If issue is not an Issue instance.
            ValueError: If issue is unsupported, column is None, or bounds are invalid.
        """
        validated_issue = _validate_issue(issue)
        if not self.can_fix(validated_issue):
            raise ValueError(
                f"ClipFixer cannot explain issue of kind '{validated_issue.kind}'."
            )
        if validated_issue.column is None:
            raise ValueError("Outlier issue requires a column name.")

        lower, upper = self._extract_bounds(validated_issue)
        return (
            f"Clip values in '{validated_issue.column}' between {lower} and {upper}."
        )
