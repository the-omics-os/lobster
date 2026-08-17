"""Package test configuration for lobster-proteomics.

Adds repo root to sys.path so shared test fixtures (tests.mock_data)
are importable when running from the repo root with importlib mode.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# __file__ = .../lobster/packages/lobster-proteomics/tests/conftest.py
# parents[3] = .../lobster/ (repo root)
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


@pytest.fixture(params=[False, True], ids=["infer_string_off", "infer_string_on"])
def pandas_infer_string(request):
    """Run a test under both pandas string-inference regimes.

    The suite otherwise pins ``pd.options.future.infer_string = False`` so the
    default string fixtures produce ``object`` dtype — which is exactly why the
    ``StringDtype`` / pandas-3 ``str`` corruption paths never got exercised.
    A test that must see both regimes requests this fixture: it runs once with
    the option off (``object``) and once on (``string``/``str``), restoring the
    previous value on teardown so no other test is affected.
    """
    previous = pd.options.future.infer_string
    pd.options.future.infer_string = request.param
    try:
        yield request.param
    finally:
        pd.options.future.infer_string = previous
