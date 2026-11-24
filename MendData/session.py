"""The ``CleaningSession`` class: the main entry point of MendData.

A session wraps a DataFrame and runs the library's order of work: scan for
problems, fix each one, score the quality before and after, and explain every
change in plain English. It is written against the team contract (Section 5 of
Narrative Report 1), so it only relies on these methods of its collaborators:

    detector   ``detect(data) -> list[Issue]``
    fixer      ``can_fix(issue)``, ``fix(data, issue) -> DataFrame``,
               ``explain(issue) -> str``
    scorer     ``compare(before, after) -> {'before', 'after', 'change'}``
    log        ``record(step, reason, affected)``, ``export() -> str``,
               ``len(log)``

Any object with those methods works, which is what makes the session easy to
test with simple stand-ins before the real classes exist.
"""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import pandas as pd

from . import utils
from .issue import Issue

if TYPE_CHECKING:  # pragma: no cover - only for editors and type checkers
    from .audit import AuditLog  # noqa: F401
    from .detectors import BaseDetector  # noqa: F401
    from .fixers import BaseFixer  # noqa: F401
    from .scoring import QualityScorer  # noqa: F401

__all__ = ["CleaningSession"]


# ----------------------------------------------------------------------
# Defaults. They are imported inside these functions, not at the top of the
# file, so this module can be imported and tested before detectors.py,
# fixers.py, scoring.py and audit.py exist. Move the imports to the top at
# Phase 4 if you prefer.
# ----------------------------------------------------------------------
def _default_detectors() -> list:
    """Build the default detectors: missing values, duplicates, outliers."""
    from .detectors import DuplicateDetector, MissingValueDetector, OutlierDetector

    return [MissingValueDetector(), DuplicateDetector(), OutlierDetector()]


def _default_fixers() -> Dict[str, Any]:
    """Build the default fixer for each kind of issue."""
    from .fixers import ClipFixer, DropFixer, ImputeFixer

    return {
        Issue.MISSING: ImputeFixer("median"),
        Issue.DUPLICATE: DropFixer(),
        Issue.OUTLIER: ClipFixer(),
    }


def _default_scorer() -> Any:
    """Build the default quality scorer."""
    from .scoring import QualityScorer

    return QualityScorer()


def _default_log() -> Any:
    """Build an empty audit log."""
    from .audit import AuditLog

    return AuditLog()


# ----------------------------------------------------------------------
# Small private helpers
# ----------------------------------------------------------------------
def _plural(count: int, word: str) -> str:
    """Return ``word`` with an ``s`` added unless ``count`` is exactly 1.

    Args:
        count: The number the word describes.
        word: A singular noun, such as ``'issue'``.

    Returns:
        ``word`` when ``count`` is 1, otherwise ``word`` followed by ``s``.
    """
    return word if count == 1 else word + "s"


def _require_methods(obj: Any, names: Iterable[str], role: str) -> None:
    """Check that ``obj`` has every method in ``names``.

    Args:
        obj: The collaborator to check.
        names: The method names it must provide.
        role: How to refer to the object in the error message.

    Raises:
        TypeError: If a method is missing or not callable.
    """
    for name in names:
        if not callable(getattr(obj, name, None)):
            raise TypeError(
                f"{role} must have a {name}() method; got {type(obj).__name__}."
            )


def _validate_fixer_map(fixers: Any) -> Dict[str, Any]:
    """Check a dict that maps an issue kind to a fixer.

    Args:
        fixers: The mapping to check, for example
            ``{'missing': ImputeFixer('mean')}``.

    Returns:
        A new dict with the same entries.

    Raises:
        TypeError: If ``fixers`` is not a mapping, or a fixer lacks the
            fixer methods.
        ValueError: If a key is not one of ``Issue.KINDS``.
    """
    if not isinstance(fixers, Mapping):
        raise TypeError(
            "fixers must be a dict that maps an issue kind to a fixer, "
            f"got {type(fixers).__name__}."
        )
    checked: Dict[str, Any] = {}
    for kind, fixer in fixers.items():
        utils.validate_choice(kind, Issue.KINDS, "fixer kind")
        _require_methods(fixer, ("can_fix", "fix", "explain"), f"The {kind!r} fixer")
        checked[kind] = fixer
    return checked


def _count_changes(before: pd.DataFrame, after: pd.DataFrame) -> int:
    """Count how much a cleaning step changed: rows dropped or cells changed.

    Args:
        before: The data before the step.
        after: The data after the step.

    Returns:
        The number of rows removed or added when the row count differs.
        Otherwise the number of cells whose value changed (a missing value
        that became a number counts as a change). If the columns differ, the
        whole of ``after`` counts as changed.
    """
    if len(before) != len(after):
        return abs(len(before) - len(after))
    old = before.reset_index(drop=True)
    new = after.reset_index(drop=True)
    if list(old.columns) != list(new.columns):
        return int(new.size)
    old = old.astype(object)
    new = new.astype(object)
    changed = old.ne(new) & ~(old.isna() & new.isna())
    return int(changed.to_numpy().sum())


class CleaningSession:
    """Scan, fix, score and explain the cleaning of one DataFrame.

    The session owns a working copy of the data, the detectors that find
    problems, a fixer for each kind of problem, a quality scorer and an audit
    log. It never changes the DataFrame you give it. Methods return values
    and never print.

    Usage::

        session = CleaningSession(df)
        session.scan()
        session.fix_all()
        print(session.score())
        print(session.report())

    Args:
        data: The DataFrame to clean. It is copied.
        detectors: Objects with a ``detect(data)`` method. Defaults to a
            missing-value, a duplicate and an outlier detector.
        fixers: A dict that maps an issue kind (``'missing'``,
            ``'duplicate'`` or ``'outlier'``) to a fixer. Kinds you leave out
            use the defaults: median fill, drop duplicates, clip outliers.
        scorer: Object with a ``compare(before, after)`` method. Defaults to
            ``QualityScorer()``.
        log: Object with ``record``, ``export`` and ``len()``. Defaults to an
            empty ``AuditLog()``.
        max_history: How many cleaning steps ``undo()`` can go back through.
            The oldest step is forgotten first. ``reset()`` always works.

    Raises:
        TypeError: If ``data`` is not a DataFrame, or a collaborator lacks
            the methods it needs.
        ValueError: If ``detectors`` is empty, a key of ``fixers`` is not an
            issue kind, or ``max_history`` is less than 1.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        detectors: Optional[Iterable["BaseDetector"]] = None,
        fixers: Optional[Mapping] = None,
        scorer: Optional["QualityScorer"] = None,
        log: Optional["AuditLog"] = None,
        max_history: int = 20,
    ) -> None:
        """Validate the arguments and set up the session.

        See the class docstring for the meaning of each argument and for the
        errors that can be raised.
        """
        utils.validate_dataframe(data)
        if isinstance(max_history, bool) or not isinstance(max_history, numbers.Integral):
            raise TypeError(
                f"max_history must be an integer, got {type(max_history).__name__}."
            )
        if max_history < 1:
            raise ValueError(f"max_history must be at least 1, got {max_history}.")

        if detectors is None:
            detector_list = _default_detectors()
        else:
            detector_list = list(detectors)
            if not detector_list:
                raise ValueError(
                    "detectors cannot be empty; pass None to use the defaults."
                )
        for detector in detector_list:
            _require_methods(detector, ("detect",), "A detector")

        fixer_map = {} if fixers is None else _validate_fixer_map(fixers)
        if any(kind not in fixer_map for kind in Issue.KINDS):
            fixer_map = {**_default_fixers(), **fixer_map}

        if scorer is None:
            scorer = _default_scorer()
        _require_methods(scorer, ("compare",), "The scorer")

        if log is None:  # never write `log or ...`: an empty log is falsy
            log = _default_log()
        _require_methods(log, ("record", "export"), "The log")

        self._original = data.copy()
        self._data = data.copy()
        # Each entry is (data before the step, step description, issue to restore).
        self._history: List[tuple] = []
        # None until scan() runs; afterwards the issues that are still open.
        self._issues: Optional[List[Issue]] = None
        self._max_history = int(max_history)
        self._detectors = detector_list
        self._fixers = fixer_map
        self._scorer = scorer
        self._log = log

    # ------------------------------------------------------------------
    # Read-only properties (encapsulation)
    # ------------------------------------------------------------------
    @property
    def data(self) -> pd.DataFrame:
        """A copy of the current data, so callers cannot change the session."""
        return self._data.copy()

    @property
    def issues(self) -> List[Issue]:
        """The issues found by the last scan that are still open (a copy)."""
        return [] if self._issues is None else list(self._issues)

    # ------------------------------------------------------------------
    # The main workflow
    # ------------------------------------------------------------------
    def scan(self) -> List[Issue]:
        """Run every detector and rank the issues, worst first.

        Replaces the issues from any earlier scan and logs one entry.

        Returns:
            The issues found, highest severity first (a copy).

        Raises:
            TypeError: If a detector returns something that is not an
                ``Issue``.
        """
        found: List[Issue] = []
        for detector in self._detectors:
            for item in detector.detect(self._data):
                if not isinstance(item, Issue):
                    raise TypeError(
                        f"{type(detector).__name__}.detect() must return Issue "
                        f"objects, got {type(item).__name__}."
                    )
                found.append(item)
        found.sort(reverse=True)  # highest severity first; ties keep their order
        self._issues = found
        count = len(found)
        self._log.record(
            step=f"Scanned the data and found {count} {_plural(count, 'issue')}.",
            reason="To find problems before cleaning.",
            affected=0,
        )
        return self.issues

    def fix(self, issue: Issue, fixer: Optional["BaseFixer"] = None) -> pd.DataFrame:
        """Apply one cleaning step: one fixer to one issue.

        Args:
            issue: The issue to resolve.
            fixer: The fixer to use. Defaults to the fixer set for
                ``issue.kind``.

        Returns:
            A copy of the cleaned data.

        Raises:
            TypeError: If ``issue`` is not an ``Issue``, ``fixer`` lacks the
                fixer methods, or the fixer does not return a DataFrame.
            ValueError: If the fixer cannot handle this kind of issue.
        """
        if not isinstance(issue, Issue):
            raise TypeError(f"issue must be an Issue, got {type(issue).__name__}.")
        if fixer is None:
            fixer = self._fixers[issue.kind]
        else:
            _require_methods(fixer, ("can_fix", "fix", "explain"), "The fixer")
        if not fixer.can_fix(issue):
            raise ValueError(
                f"{type(fixer).__name__} cannot fix {issue.kind!r} issues."
            )
        self._apply(issue, fixer)
        return self.data

    def fix_all(self, fixers: Optional[Mapping] = None) -> pd.DataFrame:
        """Fix every open issue, worst first.

        Scans first if no scan has been run. Every issue is checked against
        its fixer before anything changes, so a mismatch stops the whole
        call instead of leaving the data half-cleaned.

        Args:
            fixers: Optional dict that maps an issue kind to a fixer for this
                call only, for example ``{'missing': ImputeFixer('mean')}``.
                Kinds you leave out use the session's fixers.

        Returns:
            A copy of the cleaned data.

        Raises:
            TypeError: If ``fixers`` is not a dict of fixers.
            ValueError: If a key of ``fixers`` is not an issue kind, or a
                fixer cannot handle one of the issues.
        """
        overrides = {} if fixers is None else _validate_fixer_map(fixers)
        if self._issues is None:
            self.scan()

        plan = []
        for issue in list(self._issues):
            fixer = overrides.get(issue.kind, self._fixers[issue.kind])
            if not fixer.can_fix(issue):
                raise ValueError(
                    f"{type(fixer).__name__} cannot fix {issue.kind!r} issues."
                )
            plan.append((issue, fixer))
        for issue, fixer in plan:
            self._apply(issue, fixer)
        return self.data

    def score(self) -> Dict[str, Any]:
        """Compare the quality of the original data with the current data.

        Returns:
            The scorer's result: a dict with ``'before'``, ``'after'`` and
            ``'change'``.
        """
        return self._scorer.compare(self._original, self._data)

    def report(self) -> str:
        """Summarise the cleaning in plain English.

        Returns:
            A text report with the size of the data, the quality score
            before and after, the issues still open, and every step in the
            audit log.
        """
        rows_then, cols_then = self._original.shape
        rows_now, cols_now = self._data.shape
        scores = self.score()

        lines = [
            "MendData cleaning report",
            "========================",
            f"Data: {rows_now} rows x {cols_now} columns now "
            f"(started with {rows_then} rows x {cols_then} columns).",
            f"Quality score: {scores['before']} -> {scores['after']} "
            f"({scores['change']:+})",
            "",
        ]
        if self._issues is None:
            lines.append("Issues: no scan has been run yet.")
        elif not self._issues:
            lines.append("Issues: none left from the last scan.")
        else:
            lines.append("Issues still open from the last scan:")
            for number, issue in enumerate(self._issues, start=1):
                lines.append(f"  {number}. {issue.describe()}")
        lines.append("")
        lines.append("What was done:")
        if len(self._log) == 0:
            lines.append("  Nothing yet.")
        else:
            lines.append(self._log.export())
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Undo and reset
    # ------------------------------------------------------------------
    def undo(self) -> pd.DataFrame:
        """Revert the last cleaning step (one fixer applied to one issue).

        The data goes back to how it was, the issue becomes open again, and
        an 'Undid ...' entry is added to the log. Old log entries are never
        deleted, so the log always tells the truth.

        Returns:
            A copy of the restored data.

        Raises:
            RuntimeError: If there is nothing to undo.
        """
        if not self._history:
            raise RuntimeError("Nothing to undo.")
        snapshot, step, restore = self._history.pop()
        affected = _count_changes(self._data, snapshot)
        self._data = snapshot
        if restore is not None and self._issues is not None:
            if restore not in self._issues:
                self._issues.append(restore)
                self._issues.sort(reverse=True)
        self._log.record(
            step=f"Undid: {step}",
            reason="The last cleaning step was reverted.",
            affected=affected,
        )
        return self.data

    def reset(self) -> pd.DataFrame:
        """Go back to the original data and clear the step history.

        The issues from the last scan are cleared too, because they no longer
        match the data. The reset is logged.

        Returns:
            A copy of the original data.
        """
        affected = _count_changes(self._data, self._original)
        self._data = self._original.copy()
        self._history.clear()
        self._issues = None
        self._log.record(
            step="Reset the data to its original state.",
            reason="All cleaning steps were reverted.",
            affected=affected,
        )
        return self.data

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply(self, issue: Issue, fixer: Any) -> None:
        """Run one fixer on one issue and record the step.

        The data is only replaced after the fixer succeeds, so a failing
        fixer leaves the session untouched. A copy of the data is saved
        first, which protects undo even from a fixer that changes its input.

        Args:
            issue: The issue to resolve.
            fixer: A fixer that can handle ``issue``.

        Raises:
            TypeError: If the fixer does not return a DataFrame.
        """
        snapshot = self._data.copy()
        result = fixer.fix(self._data, issue)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"{type(fixer).__name__}.fix() must return a DataFrame, "
                f"got {type(result).__name__}."
            )
        step = fixer.explain(issue)
        affected = _count_changes(snapshot, result)

        restore = None
        if self._issues is not None and issue in self._issues:
            self._issues.remove(issue)
            restore = issue
        self._history.append((snapshot, step, restore))
        if len(self._history) > self._max_history:
            del self._history[0]
        self._data = result
        self._log.record(step=step, reason=issue.describe(), affected=affected)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return a short summary of the session.

        Returns:
            A string such as ``CleaningSession(rows=200, columns=5, issues=3,
            steps=2)``, where ``issues`` counts open issues and ``steps``
            counts the cleaning steps ``undo()`` can still revert.
        """
        rows, columns = self._data.shape
        open_issues = 0 if self._issues is None else len(self._issues)
        return (
            f"CleaningSession(rows={rows}, columns={columns}, "
            f"issues={open_issues}, steps={len(self._history)})"
        )