"""Unit tests for CleaningSession.

Most tests use small fake collaborators (detector, fixer, scorer, log) that
follow the team contract. That tests the session on its own, so it works
before detectors.py, fixers.py, scoring.py and audit.py are finished. The last
test is an integration check that runs only once those modules exist.

Run from the repository root with:  python -m pytest
"""

import pandas as pd
import pytest

from MendData.issue import Issue
from MendData.session import CleaningSession


# ----------------------------------------------------------------------
# Fakes that follow the contract
# ----------------------------------------------------------------------
class FakeDetector:
    """Returns the issues it was given."""

    def __init__(self, *issues):
        self._issues = list(issues)

    def detect(self, data):
        return list(self._issues)


class FakeFixer:
    """Handles some kinds of issue and applies ``action(data, issue)``."""

    def __init__(self, handles, action, calls=None):
        self.HANDLES = tuple(handles)
        self._action = action
        self._calls = calls

    def can_fix(self, issue):
        return issue.kind in self.HANDLES

    def fix(self, data, issue):
        if self._calls is not None:
            self._calls.append(issue.kind)
        return self._action(data, issue)

    def explain(self, issue):
        return f"Fix the {issue.kind} issue in {issue.column}"


class FakeScorer:
    """Scores 100 minus 10 per missing cell."""

    @staticmethod
    def _score(data):
        return int(100 - 10 * data.isna().sum().sum())

    def compare(self, before, after):
        old, new = self._score(before), self._score(after)
        return {"before": old, "after": new, "change": new - old}


class FakeLog:
    """Keeps (step, reason, affected) tuples."""

    def __init__(self):
        self.entries = []

    def record(self, step, reason, affected=0):
        self.entries.append((step, reason, affected))

    def export(self):
        return "\n".join(f"{n}. {e[0]}" for n, e in enumerate(self.entries, 1))

    def __len__(self):
        return len(self.entries)


# ----------------------------------------------------------------------
# Test data and helpers
# ----------------------------------------------------------------------
def make_df():
    return pd.DataFrame(
        {"age": [20.0, None, 30.0, 400.0, None], "name": list("abcde")}
    )


MISSING = Issue("age", Issue.MISSING, count=2, severity=0.4)
OUTLIER = Issue(
    "age", Issue.OUTLIER, count=1, severity=0.3,
    details={"method": "iqr", "lower": 0.0, "upper": 100.0},
)
DUPLICATE = Issue(None, Issue.DUPLICATE, count=1, severity=0.1, details={"subset": None})


def fill_age(data, issue):
    out = data.copy()
    out["age"] = out["age"].fillna(25.0)
    return out


def clip_age(data, issue):
    out = data.copy()
    out["age"] = out["age"].clip(upper=issue.details["upper"])
    return out


def drop_first_row(data, issue):
    return data.iloc[1:].copy()


def make_fixers(calls=None):
    return {
        Issue.MISSING: FakeFixer([Issue.MISSING], fill_age, calls),
        Issue.OUTLIER: FakeFixer([Issue.OUTLIER], clip_age, calls),
        Issue.DUPLICATE: FakeFixer([Issue.DUPLICATE], drop_first_row, calls),
    }


def make_session(*issues, df=None, calls=None, log=None, **kwargs):
    return CleaningSession(
        make_df() if df is None else df,
        detectors=[FakeDetector(*issues)],
        fixers=make_fixers(calls),
        scorer=FakeScorer(),
        log=FakeLog() if log is None else log,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Construction and encapsulation
# ----------------------------------------------------------------------
def test_session_never_changes_the_dataframe_it_is_given():
    df = make_df()
    original = df.copy()
    session = make_session(MISSING, OUTLIER, DUPLICATE, df=df)
    session.scan()
    session.fix_all()
    pd.testing.assert_frame_equal(df, original)


def test_data_property_returns_a_copy():
    session = make_session(MISSING)
    session.data.loc[0, "age"] = 999
    assert session.data.loc[0, "age"] == 20.0


def test_issues_property_returns_a_copy():
    session = make_session(MISSING, OUTLIER)
    session.scan()
    session.issues.clear()
    assert len(session.issues) == 2


def test_constructor_rejects_bad_arguments():
    with pytest.raises(TypeError, match="DataFrame"):
        CleaningSession([1, 2, 3])
    with pytest.raises(ValueError, match="at least 1"):
        make_session(max_history=0)
    with pytest.raises(TypeError, match="integer"):
        make_session(max_history=2.5)
    with pytest.raises(ValueError, match="empty"):
        CleaningSession(make_df(), detectors=[], fixers=make_fixers(),
                        scorer=FakeScorer(), log=FakeLog())
    with pytest.raises(TypeError, match="detect"):
        CleaningSession(make_df(), detectors=[object()], fixers=make_fixers(),
                        scorer=FakeScorer(), log=FakeLog())
    with pytest.raises(ValueError, match="fixer kind"):
        CleaningSession(make_df(), detectors=[FakeDetector()],
                        fixers={"nonsense": make_fixers()[Issue.MISSING]},
                        scorer=FakeScorer(), log=FakeLog())
    with pytest.raises(TypeError, match="compare"):
        CleaningSession(make_df(), detectors=[FakeDetector()], fixers=make_fixers(),
                        scorer=object(), log=FakeLog())


def test_an_empty_injected_log_is_used_not_replaced():
    log = FakeLog()  # empty, so len(log) == 0 and it is falsy
    session = make_session(MISSING, log=log)
    session.scan()
    assert len(log) == 1


# ----------------------------------------------------------------------
# scan
# ----------------------------------------------------------------------
def test_scan_ranks_issues_worst_first_and_logs_one_entry():
    log = FakeLog()
    session = make_session(DUPLICATE, MISSING, OUTLIER, log=log)
    found = session.scan()
    assert [i.kind for i in found] == [Issue.MISSING, Issue.OUTLIER, Issue.DUPLICATE]
    assert session.issues == found
    assert len(log) == 1
    assert log.entries[0][0] == "Scanned the data and found 3 issues."


def test_scan_rejects_a_detector_that_returns_non_issues():
    session = CleaningSession(
        make_df(), detectors=[FakeDetector("not an issue")],
        fixers=make_fixers(), scorer=FakeScorer(), log=FakeLog(),
    )
    with pytest.raises(TypeError, match="Issue"):
        session.scan()


# ----------------------------------------------------------------------
# fix and fix_all
# ----------------------------------------------------------------------
def test_fix_applies_the_default_fixer_and_logs_the_step():
    log = FakeLog()
    session = make_session(MISSING, log=log)
    result = session.fix(MISSING)
    assert result["age"].isna().sum() == 0
    step, reason, affected = log.entries[-1]
    assert step == "Fix the missing issue in age"
    assert reason == MISSING.describe()
    assert affected == 2  # two missing cells were filled


def test_fix_counts_dropped_rows_as_affected():
    log = FakeLog()
    session = make_session(DUPLICATE, log=log)
    result = session.fix(DUPLICATE)
    assert len(result) == 4
    assert log.entries[-1][2] == 1


def test_fix_uses_the_fixer_it_is_given():
    session = make_session(MISSING)
    other = FakeFixer([Issue.MISSING], lambda data, issue: data.fillna(0))
    assert session.fix(MISSING, other)["age"].eq(0).sum() == 2


def test_fix_rejects_a_fixer_that_cannot_handle_the_issue():
    session = make_session(MISSING)
    with pytest.raises(ValueError, match="cannot fix"):
        session.fix(MISSING, make_fixers()[Issue.OUTLIER])
    with pytest.raises(TypeError, match="Issue"):
        session.fix("not an issue")


def test_fix_all_fixes_every_issue_in_ranked_order():
    calls = []
    session = make_session(DUPLICATE, OUTLIER, MISSING, calls=calls)
    result = session.fix_all()
    assert calls == [Issue.MISSING, Issue.OUTLIER, Issue.DUPLICATE]
    assert session.issues == []
    assert result["age"].isna().sum() == 0
    assert len(result) == 4


def test_fix_all_scans_first_when_needed_but_not_twice():
    log = FakeLog()
    session = make_session(MISSING, log=log)
    session.fix_all()
    scans = [e for e in log.entries if e[0].startswith("Scanned")]
    assert len(scans) == 1


def test_fix_all_lets_the_caller_override_a_fixer_for_one_kind():
    session = make_session(MISSING, OUTLIER)
    use_zero = FakeFixer([Issue.MISSING], lambda data, issue: data.fillna(0))
    result = session.fix_all(fixers={Issue.MISSING: use_zero})
    assert result["age"].eq(0).sum() == 2  # the override ran, not fill_age
    assert result["age"].max() == 100.0     # the default clip still ran


def test_fix_all_checks_every_issue_before_changing_anything():
    session = make_session(MISSING, OUTLIER)
    wrong = FakeFixer([Issue.DUPLICATE], drop_first_row)
    before = session.data
    with pytest.raises(ValueError, match="cannot fix"):
        session.fix_all(fixers={Issue.OUTLIER: wrong})
    pd.testing.assert_frame_equal(session.data, before)
    assert "steps=0" in repr(session)


def test_a_failing_fixer_leaves_the_session_untouched():
    def boom(data, issue):
        raise RuntimeError("boom")

    log = FakeLog()
    session = make_session(MISSING, log=log)
    broken = FakeFixer([Issue.MISSING], boom)
    before = session.data
    with pytest.raises(RuntimeError):
        session.fix(MISSING, broken)
    pd.testing.assert_frame_equal(session.data, before)
    assert len(log) == 0


def test_a_fixer_that_returns_a_non_dataframe_is_rejected():
    session = make_session(MISSING)
    bad = FakeFixer([Issue.MISSING], lambda data, issue: [1, 2, 3])
    with pytest.raises(TypeError, match="DataFrame"):
        session.fix(MISSING, bad)


# ----------------------------------------------------------------------
# undo, reset, history
# ----------------------------------------------------------------------
def test_undo_restores_data_reopens_the_issue_and_logs_it():
    log = FakeLog()
    session = make_session(MISSING, OUTLIER, log=log)
    session.scan()
    before = session.data
    session.fix(MISSING)
    assert [i.kind for i in session.issues] == [Issue.OUTLIER]

    restored = session.undo()

    pd.testing.assert_frame_equal(restored, before)
    assert [i.kind for i in session.issues] == [Issue.MISSING, Issue.OUTLIER]
    assert log.entries[-1][0].startswith("Undid: ")
    assert len(log) == 3  # scan, fix, undo: nothing was deleted from the log


def test_undo_with_nothing_to_undo_raises():
    with pytest.raises(RuntimeError, match="Nothing to undo"):
        make_session(MISSING).undo()


def test_undo_goes_back_one_step_at_a_time():
    session = make_session(MISSING, OUTLIER)
    start = session.data
    session.fix_all()
    session.undo()  # undoes the outlier fix only
    assert session.data["age"].isna().sum() == 0
    assert session.data["age"].max() == 400.0
    session.undo()  # undoes the missing-value fix
    pd.testing.assert_frame_equal(session.data, start)


def test_history_is_capped_at_max_history():
    session = make_session(MISSING, OUTLIER, DUPLICATE, max_history=2)
    session.fix_all()
    assert "steps=2" in repr(session)
    session.undo()
    session.undo()
    with pytest.raises(RuntimeError):
        session.undo()


def test_a_fixer_that_changes_its_input_cannot_break_undo():
    def sloppy(data, issue):  # breaks the "never change the input" rule
        data["age"] = data["age"].fillna(0)
        return data

    session = make_session(MISSING)
    before = session.data
    session.fix(MISSING, FakeFixer([Issue.MISSING], sloppy))
    pd.testing.assert_frame_equal(session.undo(), before)


def test_reset_returns_to_the_original_and_clears_state():
    log = FakeLog()
    session = make_session(MISSING, OUTLIER, log=log)
    start = session.data
    session.fix_all()
    restored = session.reset()
    pd.testing.assert_frame_equal(restored, start)
    assert session.issues == []
    assert "steps=0" in repr(session)
    assert log.entries[-1][0] == "Reset the data to its original state."
    with pytest.raises(RuntimeError):
        session.undo()


# ----------------------------------------------------------------------
# score, report, repr
# ----------------------------------------------------------------------
def test_score_compares_the_original_with_the_current_data():
    session = make_session(MISSING)
    assert session.score() == {"before": 80, "after": 80, "change": 0}
    session.fix_all()
    assert session.score() == {"before": 80, "after": 100, "change": 20}


def test_report_covers_size_score_open_issues_and_steps():
    session = make_session(MISSING, OUTLIER)
    assert "no scan has been run yet" in session.report()
    assert "Nothing yet." in session.report()

    session.scan()
    session.fix(MISSING)
    text = session.report()
    assert "Quality score: 80 -> 100 (+20)" in text
    assert "5 rows x 2 columns" in text
    assert "Issues still open from the last scan:" in text
    assert OUTLIER.describe() in text
    assert "Fix the missing issue in age" in text


def test_repr_summarises_the_session():
    session = make_session(MISSING, OUTLIER)
    assert repr(session) == "CleaningSession(rows=5, columns=2, issues=0, steps=0)"
    session.scan()
    session.fix(MISSING)
    assert repr(session) == "CleaningSession(rows=5, columns=2, issues=1, steps=1)"


# ----------------------------------------------------------------------
# Integration: runs only when the real modules exist
# ----------------------------------------------------------------------
def test_default_collaborators_work_together():
    for module in ("detectors", "fixers", "scoring", "audit"):
        pytest.importorskip(f"MendData.{module}")
    df = pd.DataFrame(
        {
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, None, None, 400, 20],
            "city": list("abcdefghijklmn"),
        }
    )
    session = CleaningSession(df)
    assert all(isinstance(issue, Issue) for issue in session.scan())
    session.fix_all()
    scores = session.score()
    assert scores["after"] >= scores["before"]
    assert isinstance(session.report(), str)

    
