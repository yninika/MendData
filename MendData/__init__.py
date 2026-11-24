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

# Package metadata. Keep __version__ and __license__ in sync with setup.py
# and the LICENSE file. TestPyPI/PyPI reject re-uploads of the same version,
# so bump __version__ before every new upload.
__version__ = "0.1.0"
__author__ = ("Group 4 (DS2C): Ana Lynica B. Ferolino, Angeline B. Coniendo, "
    "Joner D. Jusol, Janel P. Obrador")
__license__ = "MIT"
__url__ = "https://github.com/yninika/MendData"

# Imports are listed in dependency order: Issue first, CleaningSession last.
# utils.py holds internal helpers (input validation), so it is not exported.
from .issue import Issue
from .detectors import (
    BaseDetector,
    MissingValueDetector,
    DuplicateDetector,
    OutlierDetector,
)
from .fixers import (
    BaseFixer,
    ImputeFixer,
    DropFixer,
    ClipFixer,
)
from .scoring import QualityScorer
from .audit import AuditLog
from .session import CleaningSession

__all__ = [
    "__version__",
    # Data model
    "Issue",
    # Detectors
    "BaseDetector",
    "MissingValueDetector",
    "DuplicateDetector",
    "OutlierDetector",
    # Fixers
    "BaseFixer",
    "ImputeFixer",
    "DropFixer",
    "ClipFixer",
    # Scoring and audit
    "QualityScorer",
    "AuditLog",
    # Main entry point
    "CleaningSession",
]
