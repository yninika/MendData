"""Unit tests for ``MendData.detectors``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from MendData.detectors import (
    BaseDetector,
    DuplicateDetector,
    MissingValueDetector,
    OutlierDetector,
)
from MendData.issue import Issue


class TestBaseDetector:
    """Tests for ``BaseDetector`` contract and validation."""

    def test_cannot_instantiate_abstract_base_class(self) -> None:
        """``BaseDetector`` cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDetector()  # type: ignore[abstract]

    def test_min_severity_property_and_validation(self) -> None:
        """``min_severity`` stores valid floats and rejects invalid inputs."""
        detector = MissingValueDetector(min_severity=0.25)
        assert detector.min_severity == 0.25

        with pytest.raises(AttributeError):
            detector.min_severity = 0.5  # type: ignore[misc]

        for bad_type in (True, False, "0.5", [0.5], None):
            with pytest.raises(TypeError):
                MissingValueDetector(min_severity=bad_type)  # type: ignore[arg-type]

        for bad_val in (-0.01, 1.01, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                MissingValueDetector(min_severity=bad_val)

    def test_non_dataframe_raises_type_error(self) -> None:
        """Passing a non-DataFrame to ``detect`` raises ``TypeError``."""
        detector = MissingValueDetector()
        with pytest.raises(TypeError):
            detector.detect([{"a": 1}])  # type: ignore[arg-type]

    def test_empty_dataframe_returns_empty_list(self) -> None:
        """An empty DataFrame returns ``[]`` across all detectors."""
        empty_df = pd.DataFrame({"a": pd.Series(dtype="float64"), "b": pd.Series(dtype="object")})
        assert MissingValueDetector().detect(empty_df) == []
        assert DuplicateDetector().detect(empty_df) == []
        assert OutlierDetector().detect(empty_df) == []


class TestMissingValueDetector:
    """Tests for ``MissingValueDetector``."""

    def test_detects_missing_values_and_computes_severity(self) -> None:
        """Emits one ``Issue.MISSING`` per column with missing cells."""
        df = pd.DataFrame(
            {
                "age": [10.0, np.nan, 30.0, np.nan],
                "city": ["CDO", "Manila", "Cebu", "Davao"],
                "score": [1.0, 2.0, 3.0, 4.0],
            }
        )
        snapshot = df.copy(deep=True)

        issues = MissingValueDetector().detect(df)

        assert df.equals(snapshot)
        assert len(issues) == 1
        issue = issues[0]
        assert isinstance(issue, Issue)
        assert issue.column == "age"
        assert issue.kind == Issue.MISSING
        assert isinstance(issue.count, int)
        assert issue.count == 2
        assert isinstance(issue.severity, float)
        assert issue.severity == pytest.approx(0.5)
        assert issue.details == {}

    def test_min_severity_filters_low_severity_columns(self) -> None:
        """Issues below ``min_severity`` are excluded."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
        detector = MissingValueDetector(min_severity=0.3)

        assert detector.detect(df) == []


class TestDuplicateDetector:
    """Tests for ``DuplicateDetector``."""

    def test_detects_all_column_duplicates(self) -> None:
        """Emits one row-level ``Issue.DUPLICATE`` when full rows repeat."""
        df = pd.DataFrame(
            {
                "name": ["Ana", "Ana", "Joner", "Joner"],
                "age": [20, 20, 21, 22],
            }
        )
        snapshot = df.copy(deep=True)

        detector = DuplicateDetector()
        issues = detector.detect(df)

        assert df.equals(snapshot)
        assert detector.subset is None
        assert len(issues) == 1
        issue = issues[0]
        assert issue.column is None
        assert issue.kind == Issue.DUPLICATE
        assert issue.count == 1
        assert issue.severity == pytest.approx(0.25)
        assert issue.details == {"subset": None}

    def test_subset_property_returns_copy_and_checks_subset(self) -> None:
        """``subset`` returns a copy and checks duplicates on chosen columns."""
        df = pd.DataFrame(
            {
                "id": [1, 1, 2],
                "val": ["a", "b", "c"],
            }
        )
        detector = DuplicateDetector(subset=["id"])
        returned_subset = detector.subset
        assert returned_subset == ["id"]

        # Mutating the returned list must not mutate the detector's internal state.
        assert returned_subset is not None
        returned_subset.append("mutated")
        assert detector.subset == ["id"]

        issues = detector.detect(df)
        assert len(issues) == 1
        assert issues[0].count == 1
        assert issues[0].severity == pytest.approx(1 / 3)
        assert issues[0].details == {"subset": ["id"]}

    def test_unknown_subset_column_raises_value_error(self) -> None:
        """Unknown column names in ``subset`` raise ``ValueError``."""
        df = pd.DataFrame({"a": [1, 2]})
        detector = DuplicateDetector(subset=["missing_col"])
        with pytest.raises(ValueError):
            detector.detect(df)

    def test_invalid_subset_type_raises_type_error(self) -> None:
        """Non-iterable or string ``subset`` raises ``TypeError``."""
        with pytest.raises(TypeError):
            DuplicateDetector(subset="col_a")
        with pytest.raises(TypeError):
            DuplicateDetector(subset=123)  # type: ignore[arg-type]


class TestOutlierDetector:
    """Tests for ``OutlierDetector``."""

    def test_iqr_defaults_and_bounds(self) -> None:
        """Default IQR (k=1.5) flags extreme numeric values with float bounds."""
        df = pd.DataFrame({"values": [10.0, 12.0, 11.0, 13.0, 12.0, 100.0]})
        snapshot = df.copy(deep=True)

        detector = OutlierDetector()
        assert detector.method == "iqr"
        assert detector.threshold == 1.5

        issues = detector.detect(df)

        assert df.equals(snapshot)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.column == "values"
        assert issue.kind == Issue.OUTLIER
        assert issue.count == 1
        assert issue.severity == pytest.approx(1 / 6)
        assert issue.details["method"] == "iqr"
        assert isinstance(issue.details["lower"], float)
        assert isinstance(issue.details["upper"], float)
        assert issue.details["lower"] < issue.details["upper"]

    def test_zscore_method_and_default_threshold(self) -> None:
        """Z-score method defaults to threshold 3.0 and flags extreme values."""
        df = pd.DataFrame({"score": [50.0, 51.0, 49.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 500.0]})
        detector = OutlierDetector(method="zscore")
        assert detector.method == "zscore"
        assert detector.threshold == 3.0

        issues = detector.detect(df)
        assert len(issues) == 1
        assert issues[0].column == "score"
        assert issues[0].count == 1
        assert issues[0].details["method"] == "zscore"

    def test_skips_booleans_few_non_missing_and_zero_spread(self) -> None:
        """Ignores booleans, columns with < 4 non-nulls, and zero-spread columns."""
        df = pd.DataFrame(
            {
                "flag": [False, False, False, True],  # bool excluded
                "too_few": [1.0, 100.0, np.nan, np.nan],  # only 2 non-missing
                "flat": [5.0, 5.0, 5.0, 5.0],  # zero spread
            }
        )
        assert OutlierDetector().detect(df) == []

    def test_invalid_method_or_threshold_raises(self) -> None:
        """Invalid method or non-positive threshold raises appropriate errors."""
        with pytest.raises(ValueError):
            OutlierDetector(method="mad")
        with pytest.raises(TypeError):
            OutlierDetector(method=123)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            OutlierDetector(threshold=0)
        with pytest.raises(ValueError):
            OutlierDetector(threshold=-1.5)
        with pytest.raises(TypeError):
            OutlierDetector(threshold=True)  # type: ignore[arg-type]
