"""Keep the process-wide UI language from leaking between tests."""

import pytest

from lobster.i18n import current_lang, set_lang


@pytest.fixture(autouse=True)
def _restore_ui_language():
    before = current_lang()
    yield
    set_lang(before)
