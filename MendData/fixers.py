
"""Cleaning fixer classes for the MendData library."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFixer(ABC):
    """Abstract parent class for all MendData fixers."""

    HANDLES = ()

    def __init__(self, strategy):
        """Initialize the fixer."""
        self._strategy = strategy

    @property
    def strategy(self):
        """Return the cleaning strategy."""
        return self._strategy

    def can_fix(self, issue):
        """Check if this fixer supports the issue."""
        return issue.kind in self.HANDLES

    @abstractmethod
    def fix(self, data, issue):
        """Apply the cleaning operation."""

    @abstractmethod
    def explain(self, issue):
        """Explain the cleaning operation."""


class ImputeFixer(BaseFixer):
    """Fill missing values using mean, median, or mode."""

    HANDLES = ("missing",)

    def __init__(self, strategy="median"):
        """Initialize the imputation strategy."""
        allowed = ("mean", "median", "mode")

        if strategy not in allowed:
            raise ValueError("Invalid imputation strategy.")

        super().__init__(strategy)

    def fix(self, data, issue):
        """Fill missing values in the selected column."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")

        if not self.can_fix(issue):
            raise ValueError("Unsupported issue type.")

        if issue.column not in data.columns:
            raise ValueError("Column does not exist.")

        result = data.copy()
        series = result[issue.column]

        if self.strategy == "mean":
            value = (
                series.mean()
                if pd.api.types.is_numeric_dtype(series)
                else None
            )
        elif self.strategy == "median":
            value = (
                series.median()
                if pd.api.types.is_numeric_dtype(series)
                else None
            )
        else:
            value = None

        if value is None or pd.isna(value):
            modes = series.mode(dropna=True)

            if modes.empty:
                return result

            value = modes.iloc[0]

        result[issue.column] = series.fillna(value)

        return result

    def explain(self, issue):
        """Explain the imputation operation."""
        if not self.can_fix(issue):
            raise ValueError("Unsupported issue type.")

        return (
            f"Replace {issue.count} missing values in "
            f"'{issue.column}' using {self.strategy}."
        )


class DropFixer(BaseFixer):
    """Remove rows with missing values or duplicates."""

    HANDLES = ("missing", "duplicate")

    def __init__(self):
        """Initialize the drop fixer."""
        super().__init__("drop")

    def fix(self, data, issue):
        """Remove rows based on the issue type."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")

        if not self.can_fix(issue):
            raise ValueError("Unsupported issue type.")

        result = data.copy()

        if issue.kind == "missing":
            if issue.column not in result.columns:
                raise ValueError("Column does not exist.")

            return result.dropna(subset=[issue.column])

        details = issue.details or {}
        subset = details.get("subset")

        if subset is not None:
            if any(column not in result.columns for column in subset):
                raise ValueError("Unknown duplicate columns.")

        return result.drop_duplicates(subset=subset)

    def explain(self, issue):
        """Explain the row removal operation."""
        if not self.can_fix(issue):
            raise ValueError("Unsupported issue type.")

        if issue.kind == "missing":
            return f"Drop rows with missing values in {issue.column}."

        return f"Remove {issue.count} duplicate rows."


class ClipFixer(BaseFixer):
    """Handle outliers by clipping values to specified bounds."""

    HANDLES = ("outlier",)

    def __init__(self):
        """Initialize the clipping fixer."""
        super().__init__("clip")

    def fix(self, data, issue):
        """Clip outliers using the detected boundaries."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame.")

        if not self.can_fix(issue):
            raise ValueError("Unsupported issue type.")

        if issue.column not in data.columns:
            raise ValueError("Column does not exist.")

        details = issue.details or {}
        lower = details.get("lower")
        upper = details.get("upper")

        if lower is None or upper is None:
            raise ValueError("Outlier bounds are required.")

        if lower > upper:
            raise ValueError("Invalid outlier bounds.")

        result = data.copy()

        result[issue.column] = result[issue.column].clip(
            lower=lower,
            upper=upper
        )

        return result

    def explain(self, issue):
        """Explain the outlier clipping operation."""
        if not self.can_fix(issue):
            raise ValueError("Unsupported issue type.")

        details = issue.details or {}
        lower = details.get("lower")
        upper = details.get("upper")

        if lower is None or upper is None:
            raise ValueError("Outlier bounds are required.")

        return (
            f"Clip outliers in '{issue.column}' "
            f"between {lower} and {upper}."
        )
        