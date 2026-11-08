"""Exported notebook code for PRIDE and MassIVE downloads runs its setup cell."""

from pathlib import Path
from unittest.mock import Mock

import pytest


def _service(kind, dm):
    if kind == "pride":
        from lobster.services.data_access.pride_download_service import (
            PRIDEDownloadService as cls,
        )
    else:
        from lobster.services.data_access.massive_download_service import (
            MassIVEDownloadService as cls,
        )
    return cls(dm)


def _setup_lines(code: str) -> str:
    """The template up to and including the ``service = ...`` line (no network)."""
    lines = []
    for line in code.splitlines():
        lines.append(line)
        if line.startswith("service = "):
            return "\n".join(lines)
    raise AssertionError("template has no 'service = ...' line")


@pytest.mark.parametrize(
    "kind,dataset_id",
    [("pride", "PXD000001"), ("massive", "MSV000000001")],
)
def test_rendered_template_compiles_and_constructs(
    tmp_path, monkeypatch, kind, dataset_id
):
    dm = Mock()
    dm.workspace_path = tmp_path / "unused"
    dm.cache_dir = tmp_path / "unused" / "cache"
    service = _service(kind, dm)

    code = service._create_download_ir(dataset_id, "RESULT_FIRST", {}).render()
    compile(code, f"<{kind} template>", "exec")
    assert "workspace_dir" not in code

    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(
        compile(_setup_lines(code), f"<{kind} setup>", "exec"), namespace
    )  # nosec B102

    from lobster.core.runtime.data_manager import DataManagerV2

    assert isinstance(namespace["dm"], DataManagerV2)
    assert (
        Path(namespace["dm"].workspace_path).resolve()
        == (tmp_path / "workspace").resolve()
    )
    assert type(namespace["service"]).__name__ == type(service).__name__
