import builtins

import pytest

from lobster.cli_internal.commands.light import vector_search_commands


def test_vector_search_reports_backend_unavailable(monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "lobster.services.vector.service":
            raise ImportError("No module named 'lobster.services.vector.service'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr(
        "lobster.core.component_registry.get_install_command",
        lambda package, is_extra=False: "uv pip install 'lobster-ai[vector-search]'",
    )

    with pytest.raises(ImportError) as exc_info:
        vector_search_commands.vector_search_all_collections("glioblastoma")

    error_text = str(exc_info.value)
    assert "Vector search backend is not available in this install." in error_text
    assert "lobster-metadata development package" in error_text
    assert "uv pip install 'lobster-ai[vector-search]'" in error_text
