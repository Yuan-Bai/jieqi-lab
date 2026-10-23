"""Test-suite path setup.

The project is run from its source tree, so tests add the repository root to
``sys.path``.  Test modules themselves stay non-package modules; several tests
import helpers from sibling ``test_*.py`` files.
"""
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
