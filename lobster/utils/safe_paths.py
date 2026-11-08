"""Local paths for files whose names come from a remote listing."""

from pathlib import Path, PurePosixPath, PureWindowsPath


def safe_download_path(directory: Path, name: str) -> Path:
    """Return ``directory / <base name of name>``, always inside ``directory``.

    Remote listings (FTP, EBI, GEO) supply file names. Only the last path
    component is kept, so ``../../x`` and ``/abs/x`` both become ``x``.

    Raises:
        ValueError: if no usable file name is left (empty, ``.`` or ``..``).
    """
    base = PureWindowsPath(PurePosixPath(str(name)).name).name.strip()
    if base in ("", ".", ".."):
        raise ValueError(f"Unusable file name from remote listing: {name!r}")
    root = Path(directory).resolve()
    candidate = (root / base).resolve()
    if candidate.parent != root:
        raise ValueError(f"File name escapes the download directory: {name!r}")
    return root / base
