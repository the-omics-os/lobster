"""Keep the process-wide UI language from leaking between tests."""

import pytest

import lobster.i18n as i18n


@pytest.fixture(autouse=True)
def _restore_ui_language():
    before = (i18n._LANG, i18n._SOURCE)
    yield
    i18n._LANG, i18n._SOURCE = before
