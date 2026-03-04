"""Package test configuration for lobster-transcriptomics.

Adds repo root to sys.path so shared test fixtures (tests.mock_data)
are importable when running from the repo root with importlib mode.
"""
import sys
from pathlib import Path

# __file__ = .../lobster/packages/lobster-transcriptomics/tests/conftest.py
# parents[3] = .../lobster/ (repo root)
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
