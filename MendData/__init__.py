"""MendData: a beginner-friendly data cleaning library for pandas.

MendData follows one simple order of work: scan a DataFrame for problems,
fix each one with a chosen strategy, score the quality before and after, and
explain every change in plain English.

Quick start
-----------
Wrap a DataFrame in a ``CleaningSession`` and call its methods. The methods
return values and never print, so wrap them in ``print()`` to see them::

    import pandas as pd
    from MendData import CleaningSession, ImputeFixer

    df = pd.read_csv("messy_data.csv")

    session = CleaningSession(df)   # wrap a pandas DataFrame
    session.scan()                  # find issues, ranked by severity
    session.fix_all()               # apply a default fixer to each issue
    print(session.score())          # quality score before and after
    print(session.report())         # plain-English summary of changes
    session.undo()                  # revert the last cleaning step
    session.reset()                 # go back to the original data

    clean_df = session.data         # a copy of the current data

To choose a strategy for one kind of issue, pass your own fixer::

    session.fix_all(fixers={"missing": ImputeFixer("mean")})

Public classes
--------------
Issue
    A problem found in the data (column, kind, count, severity, details).
BaseDetector, MissingValueDetector, DuplicateDetector, OutlierDetector
    Find problems and return them as ``Issue`` objects.
BaseFixer, ImputeFixer, DropFixer, ClipFixer
    Resolve an ``Issue`` using a chosen strategy.
QualityScorer
    Rates a dataset from 0 to 100 (completeness, uniqueness, validity).
AuditLog
    Records what each cleaning step did and why.
CleaningSession
    The main entry point that ties all of the above together.
"""

import importlib
from typing import TYPE_CHECKING

# Package metadata. Keep __version__ and __license__ in sync with setup.py
# and the LICENSE file. TestPyPI/PyPI reject re-uploads of the same version,
# so bump __version__ before every new upload.
__version__ = "0.1.0"
__author__ = (
    "Group 4 (DS2C): Ana Lynica B. Ferolino, Angeline B. Coniendo, "
    "Joner D. Jusol, Janel P. Obrador"
)
__license__ = "MIT"
__url__ = "https://github.com/yninika/MendData"

# Every public name and the submodule that defines it. This is the single
# source of truth: to export a new class, add it here (and to the
# TYPE_CHECKING block below so editors can see it).
_EXPORTS = {
    # Data model
    "Issue": "issue",
    # Detectors
    "BaseDetector": "detectors",
    "MissingValueDetector": "detectors",
    "DuplicateDetector": "detectors",
    "OutlierDetector": "detectors",
    # Fixers
    "BaseFixer": "fixers",
    "ImputeFixer": "fixers",
    "DropFixer": "fixers",
    "ClipFixer": "fixers",
    # Scoring and audit
    "QualityScorer": "scoring",
    "AuditLog": "audit",
    # Main entry point
    "CleaningSession": "session",
}

__all__ = ["__version__"] + list(_EXPORTS)

# Never executed at runtime. It only lets editors, type checkers and
# documentation tools see where each public name comes from.
if TYPE_CHECKING:  # pragma: no cover
    from .issue import Issue  # noqa: F401
    from .detectors import (  # noqa: F401
        BaseDetector,
        MissingValueDetector,
        DuplicateDetector,
        OutlierDetector,
    )
    from .fixers import BaseFixer, ImputeFixer, DropFixer, ClipFixer  # noqa: F401
    from .scoring import QualityScorer  # noqa: F401
    from .audit import AuditLog  # noqa: F401
    from .session import CleaningSession  # noqa: F401


def __getattr__(name: str):
    """Import a public class the first time it is asked for (PEP 562).

    Because each submodule is only loaded when its classes are used,
    ``from MendData.detectors import MissingValueDetector`` works even while
    teammates are still writing the other modules, and
    ``from MendData import CleaningSession`` works as soon as ``session.py``
    and the modules it needs exist.

    Args:
        name: The attribute being looked up on the package.

    Returns:
        The class called ``name``.

    Raises:
        AttributeError: If ``name`` is not a public MendData name.
    """
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = getattr(importlib.import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # cache it so this function runs once per name
    return value


def __dir__() -> list:
    """List the public names so ``dir(MendData)`` and tab completion work."""
    return sorted(set(globals()) | set(__all__))
