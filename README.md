# MendData

A beginner-friendly data cleaning library for pandas. MendData scans a
dataset for problems, fixes each one with a strategy you choose, scores the
quality before and after, and explains every change in plain English.

> **Status:** in development (Phase 1, planning). Not yet published on PyPI.

Built for the OOP Final Project (Section DS2C, Group 4).

## Planned features

| # | Feature | What it does |
|---|---------|--------------|
| 1 | Issue scanning | Detects missing values, duplicate rows and outliers (IQR or z-score), ranked by severity |
| 2 | Strategy-based fixing | Resolves each issue by imputing (mean, median, mode), dropping, or clipping |
| 3 | Data quality score | Rates the dataset from 0 to 100 before and after cleaning |
| 4 | Plain-English audit log | Records what each step changed and why, and exports it as readable text |
| 5 | Undo and step history | Reverts the dataset one cleaning step at a time |

## Planned usage

Names may change as development continues.

```python
import pandas as pd
from menddata import CleaningSession

df = pd.read_csv("data.csv")
session = CleaningSession(df)   # wrap a pandas DataFrame
session.scan()                  # find issues, ranked by severity
session.fix_all()               # apply a fixer to each issue
session.score()                 # quality score before and after
session.report()                # plain-English summary of changes
session.undo()                  # revert the last cleaning step
```

## Installation

Once published on PyPI:

```
pip install menddata
```

From source (for contributors):

```
git clone https://github.com/yninika/MendData.git
cd MendData
pip install -r requirements.txt
pip install -e .
```

## Project structure

```
MendData/
├── menddata/
│   ├── __init__.py    version info and public imports
│   ├── issue.py       Issue
│   ├── detectors.py   BaseDetector, MissingValueDetector, DuplicateDetector, OutlierDetector
│   ├── fixers.py      BaseFixer, ImputeFixer, DropFixer, ClipFixer
│   ├── scoring.py     QualityScorer
│   ├── audit.py       AuditLog
│   ├── session.py     CleaningSession
│   └── utils.py       shared helpers and input validation
├── tests/
├── demo.ipynb         (coming soon)
├── README.md
├── setup.py
├── LICENSE
├── requirements.txt
└── .gitignore
```

## Contributors

| Member | Responsibility |
|--------|----------------|
| Joner D. Jusol | Detectors (`issue.py`, `detectors.py`) and repository maintenance |
| Angeline B. Coniendo | Fixers (`fixers.py`) and the UML diagram |
| Janel P. Obrador | Scoring and audit (`scoring.py`, `audit.py`), narrative reports, README |
| Ana Lynica B. Ferolino | Group leader; session (`session.py`, `utils.py`, `__init__.py`), tests, demo notebook |

## Workflow

Work happens on feature branches and is merged into `main` through pull
requests reviewed by a teammate. Please do not commit directly to `main`.

## License

Released under the MIT License. See [LICENSE](LICENSE).
