"""Unit tests for Issue, the shared contract between detectors and fixers.

The tests follow Section 5.2 of Narrative Report 1: what an Issue stores, what
it refuses, that it cannot be changed, how it describes itself, and how it is
compared, hashed and ranked.

Run from the repository root with:  python -m pytest
"""

import doctest
from typing import Any

import numpy as np
import pytest

import MendData.issue
from MendData.issue import Issue



def make_missing(**changes):
    """A valid 'missing' issue; keyword arguments replace its fields."""
    fields = dict(column="age", kind=Issue.MISSING, count=12, severity=0.06)
    fields.update(changes)
    return Issue(**fields)


def make_outlier(**changes):
    """A valid 'outlier' issue with the details a ClipFixer needs."""
    fields = dict(
        column="pay", kind=Issue.OUTLIER, count=3, severity=0.2,
        details={"method": "iqr", "lower": 1.0, "upper": 9.0},
    )
    fields.update(changes)
    return Issue(**fields)


def make_duplicate(**changes):
    """A valid row-level 'duplicate' issue."""
    fields = dict(
        column=None, kind=Issue.DUPLICATE, count=5, severity=0.025,
        details={"subset": None},
    )
    fields.update(changes)
    return Issue(**fields)


# ----------------------------------------------------------------------
# Kinds and construction
# ----------------------------------------------------------------------
def test_kind_constants_have_the_contract_values():
    assert Issue.MISSING == "missing"
    assert Issue.DUPLICATE == "duplicate"
    assert Issue.OUTLIER == "outlier"
    assert Issue.KINDS == ("missing", "duplicate", "outlier")


def test_properties_return_the_values_given():
    details = {"method": "iqr", "lower": 1.0, "upper": 9.0}
    issue = Issue("pay", Issue.OUTLIER, count=3, severity=0.2, details=details)
    assert issue.column == "pay"
    assert issue.kind == "outlier"
    assert issue.count == 3
    assert issue.severity == 0.2
    assert issue.details == details


def test_details_default_to_an_empty_dict():
    assert make_missing().details == {}
    assert Issue("age", Issue.MISSING, 1, 0.5, details=None).details == {}


def test_arguments_can_be_given_by_position():
    issue = Issue("age", Issue.MISSING, 12, 0.06)
    assert (issue.column, issue.kind, issue.count, issue.severity) == (
        "age", "missing", 12, 0.06,
    )


def test_only_duplicate_issues_may_have_no_column():
    assert make_duplicate().column is None
    assert make_duplicate(details=None).details == {}


def test_column_can_be_any_hashable_label():
    assert make_missing(column=3).column == 3
    assert make_missing(column=("a", "b")).column == ("a", "b")


def test_limits_of_the_allowed_ranges_are_accepted():
    assert make_missing(count=0, severity=0.0).count == 0
    assert make_missing(count=1, severity=1.0).severity == 1.0


def test_numpy_numbers_are_converted_to_plain_python_numbers():
    issue = make_missing(count=np.int64(3), severity=np.float64(0.25))
    assert type(issue.count) is int
    assert type(issue.severity) is float
    assert (issue.count, issue.severity) == (3, 0.25)
    assert type(make_missing(severity=np.float32(0.5)).severity) is float
    assert type(make_missing(severity=1).severity) is float  # int is a valid number


def test_details_are_copied_when_the_issue_is_created():
    details: dict[str, Any] = {"subset": ["id"]}
    issue = make_duplicate(details=details)
    details["subset"] = ["something else"]
    details["extra"] = 1
    assert issue.details == {"subset": ["id"]}



@pytest.mark.parametrize("bad_kind", ["nope", "", "Missing", "MISSING"])
def test_unknown_kind_is_rejected(bad_kind):
    with pytest.raises(ValueError, match="kind must be one of"):
        make_missing(kind=bad_kind)


@pytest.mark.parametrize("bad_kind", [None, 1, b"missing", ["missing"]])
def test_kind_must_be_a_string(bad_kind):
    with pytest.raises(TypeError, match="kind must be a string"):
        make_missing(kind=bad_kind)


@pytest.mark.parametrize("kind", [Issue.MISSING, Issue.OUTLIER])
def test_missing_and_outlier_issues_require_a_column(kind):
    with pytest.raises(ValueError, match="column is required"):
        Issue(None, kind, 1, 0.5)


@pytest.mark.parametrize("bad_column", [["age"], {"a": 1}, {"age"}])
def test_column_must_be_a_hashable_label(bad_column):
    with pytest.raises(TypeError, match="column"):
        make_missing(column=bad_column)


@pytest.mark.parametrize("bad_count", [-1, -100])
def test_negative_count_is_rejected(bad_count):
    with pytest.raises(ValueError, match="negative"):
        make_missing(count=bad_count)


@pytest.mark.parametrize("bad_count", [1.0, 2.5, "3", None, True, False])
def test_count_must_be_an_integer(bad_count):
    with pytest.raises(TypeError, match="count must be an integer"):
        make_missing(count=bad_count)


@pytest.mark.parametrize("bad_severity", [-0.01, 1.01, 2, -1, float("nan"), float("inf")])
def test_severity_must_be_between_zero_and_one(bad_severity):
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_missing(severity=bad_severity)


@pytest.mark.parametrize("bad_severity", ["0.5", None, True, [0.5]])
def test_severity_must_be_a_number(bad_severity):
    with pytest.raises(TypeError, match="severity must be a number"):
        make_missing(severity=bad_severity)


@pytest.mark.parametrize("bad_details", [[("a", 1)], "abc", 5, ("a", 1)])
def test_details_must_be_a_mapping_or_none(bad_details):
    with pytest.raises(TypeError, match="details must be a mapping"):
        make_missing(details=bad_details)



@pytest.mark.parametrize("name", ["column", "kind", "count", "severity", "details"])
def test_properties_are_read_only(name):
    issue = make_outlier()
    with pytest.raises(AttributeError):
        setattr(issue, name, "new value")


def test_details_property_returns_a_copy():
    issue = make_outlier()
    issue.details["upper"] = 999.0
    issue.details["extra"] = True
    assert issue.details == {"method": "iqr", "lower": 1.0, "upper": 9.0}


def test_values_are_stored_in_private_attributes():
    issue = make_missing()
    for name in ("_column", "_kind", "_count", "_severity", "_details"):
        assert hasattr(issue, name)



def test_describe_matches_the_example_in_the_contract():
    assert make_missing().describe() == (
        "Column 'age' has 12 missing values (6.0% of its rows)."
    )


def test_describe_uses_singular_and_plural_correctly():
    assert make_missing(count=1, severity=0.5).describe() == (
        "Column 'age' has 1 missing value (50.0% of its rows)."
    )
    assert make_outlier(count=1, severity=0.01).describe() == (
        "Column 'pay' has 1 outlier (1.0% of its values)."
    )
    assert make_outlier().describe() == (
        "Column 'pay' has 3 outliers (20.0% of its values)."
    )
    assert make_duplicate(count=1, severity=0.1).describe() == (
        "Found 1 duplicate row (10.0% of all rows)."
    )
    assert make_missing(count=0, severity=0.0).describe() == (
        "Column 'age' has 0 missing values (0.0% of its rows)."
    )


def test_describe_a_duplicate_issue():
    assert make_duplicate().describe() == "Found 5 duplicate rows (2.5% of all rows)."


def test_describe_a_duplicate_issue_that_compares_only_some_columns():
    issue = make_duplicate(details={"subset": ["id", "name"]})
    assert issue.describe() == (
        "Found 5 duplicate rows (2.5% of all rows), "
        "comparing only the columns id, name."
    )


@pytest.mark.parametrize("subset", [None, []])
def test_describe_ignores_an_empty_or_missing_subset(subset):
    issue = make_duplicate(details={"subset": subset})
    assert issue.describe() == "Found 5 duplicate rows (2.5% of all rows)."


def test_str_is_the_plain_english_description():
    for issue in (make_missing(), make_outlier(), make_duplicate()):
        assert str(issue) == issue.describe()


def test_repr_shows_the_fields_with_severity_rounded_to_three_places():
    assert repr(make_missing()) == (
        "Issue(column='age', kind='missing', count=12, severity=0.060)"
    )
    assert repr(make_duplicate()) == (
        "Issue(column=None, kind='duplicate', count=5, severity=0.025)"
    )
    assert repr(make_missing(severity=1 / 3)) == (
        "Issue(column='age', kind='missing', count=12, severity=0.333)"
    )



def test_issues_with_the_same_fields_are_equal():
    assert make_missing() == make_missing()
    assert not (make_missing() != make_missing())


def test_equality_ignores_details():
    first = make_outlier()
    second = make_outlier(details={"method": "zscore", "lower": 0.0, "upper": 5.0})
    assert first == second
    assert hash(first) == hash(second)


@pytest.mark.parametrize(
    "different",
    [
        make_missing(column="pay"),
        make_missing(kind=Issue.OUTLIER),
        make_missing(count=13),
        make_missing(severity=0.07),
    ],
)
def test_issues_that_differ_in_any_field_are_not_equal(different):
    assert make_missing() != different


@pytest.mark.parametrize("other", ["a string", 5, None, ("age", "missing", 12, 0.06)])
def test_an_issue_is_never_equal_to_a_non_issue(other):
    assert make_missing() != other
    assert not (make_missing() == other)


def test_equal_issues_have_equal_hashes():
    assert hash(make_missing()) == hash(make_missing())


def test_issues_can_be_used_in_sets_and_as_dict_keys():
    issues = {make_missing(), make_missing(), make_outlier()}
    assert len(issues) == 2
    lookup = {make_missing(): "fill"}
    assert lookup[make_missing()] == "fill"



def test_less_than_compares_severity_only():
    mild, severe = make_missing(severity=0.1), make_outlier(severity=0.9)
    assert mild < severe
    assert not (severe < mild)


def test_greater_than_works_through_the_reflected_comparison():
    mild, severe = make_missing(severity=0.1), make_outlier(severity=0.9)
    assert severe > mild
    assert not (mild > severe)


def test_sorted_with_reverse_puts_the_worst_issue_first():
    low = make_missing(severity=0.1)
    mid = make_outlier(severity=0.5)
    high = make_duplicate(severity=0.9)
    assert sorted([mid, high, low], reverse=True) == [high, mid, low]
    assert sorted([mid, high, low]) == [low, mid, high]


def test_max_and_min_use_severity():
    low, high = make_missing(severity=0.1), make_outlier(severity=0.9)
    assert max([low, high]) is high
    assert min([low, high]) is low


def test_issues_with_equal_severity_keep_their_order_when_sorted():
    first = make_missing(column="a", severity=0.5)
    second = make_missing(column="b", severity=0.5)
    top = make_outlier(severity=0.9)
    assert sorted([first, second, top], reverse=True) == [top, first, second]


def test_issues_with_equal_severity_are_neither_smaller_larger_nor_equal():
    first = make_missing(column="a", severity=0.5)
    second = make_missing(column="b", severity=0.5)
    assert not (first < second)
    assert not (first > second)
    assert first != second


@pytest.mark.parametrize("compare", [
    lambda a, b: a <= b,
    lambda a, b: a >= b,
])
def test_less_or_equal_and_greater_or_equal_are_not_supported(compare):
    with pytest.raises(TypeError):
        compare(make_missing(severity=0.1), make_outlier(severity=0.9))


@pytest.mark.parametrize("compare", [
    lambda issue: issue < 0.5,
    lambda issue: issue > 0.5,
    lambda issue: issue < "x",
])
def test_ordering_an_issue_against_a_non_issue_raises_type_error(compare):
    with pytest.raises(TypeError):
        compare(make_missing())



def test_the_examples_in_the_docstrings_are_correct():
    results = doctest.testmod(MendData.issue)
    assert results.attempted > 0
    assert results.failed == 0