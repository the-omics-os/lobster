"""Deploy gate: the pandas-3 cross-version artifact test (Assignment 3).

Writes proteomics artifacts INSIDE the live production image
(pandas 3.0.5 / anndata 0.13.2) via the raw ``write_h5ad`` bypass path, then
reads them back under the candidate stack (this venv, pandas < 3 / anndata <
0.13) and runs the hardened contaminant/reverse guard and the adapter
metadata/intensity classifier over them.

Two things are proven:
  1. On artifacts carrying the pandas-3 string representation, the hardened
     guard retains the correct features (and leaves the numeric matrix intact),
     regardless of whether the downgrade-read surfaces the flag as
     ``str`` / ``string`` / ``category`` / ``object``.
  2. In the production runtime itself, the code currently on ``main`` empties
     the matrix for ``'1'/'0'`` and ``'True'/'False'`` flags while the hardened
     guard does not (``in_image_compare.py``).

Requires Docker and the production image locally. Resolution order:
``$LOBSTER_PROD_IMAGE`` → the task-def-439 digest tag that was live when this
was authored. Skips (never fails) when neither is present, so ordinary CI —
which has no production image — is unaffected. This is a manual pre-deploy gate,
run on the build host that holds the image.

Regenerate/inspect manually::

    docker run --rm --network none --entrypoint /opt/venv/bin/python \\
      -v "$PWD/tests/integration/pandas3_gate":/a3 -v /tmp/a3out:/artifacts \\
      <prod-image> /a3/write_artifacts.py
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from pandas.api import types as pdt

from lobster.core.utils.dtype_guards import boolean_flag_mask
from lobster.core.adapters.proteomics_adapter import ProteomicsAdapter

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# task-def-439 image digest (live production when authored, 2026-08-15).
_DEFAULT_IMAGE = (
    "cdkasset-8961c8639f13856dd4f6a1b9d9101399996c0689b97e8d01360ddc663a2980fa:latest"
)
_GATE_DIR = Path(__file__).parent / "pandas3_gate"


def _resolve_image():
    if not shutil.which("docker"):
        return None
    for ref in (os.environ.get("LOBSTER_PROD_IMAGE"), _DEFAULT_IMAGE):
        if not ref:
            continue
        probe = subprocess.run(
            ["docker", "image", "inspect", ref],
            capture_output=True, text=True,
        )
        if probe.returncode == 0:
            return ref
    return None


_IMAGE = _resolve_image()
_reason = "Docker + production image required (set $LOBSTER_PROD_IMAGE); this is a manual pre-deploy gate."
pytestmark.append(pytest.mark.skipif(_IMAGE is None, reason=_reason))


def _docker_python(script_name, out_dir=None):
    mounts = ["-v", f"{_GATE_DIR}:/a3"]
    if out_dir is not None:
        mounts += ["-v", f"{out_dir}:/artifacts"]
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "--entrypoint", "/opt/venv/bin/python",
        *mounts, _IMAGE, f"/a3/{script_name}",
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def written_artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("pandas3_artifacts")
    proc = _docker_python("write_artifacts.py", out_dir=str(out))
    assert proc.returncode == 0, f"writer failed:\n{proc.stdout}\n{proc.stderr}"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stack"]["pandas"].startswith("3."), (
        f"expected a pandas-3 image, got {manifest['stack']}"
    )
    return out, manifest


def test_hardened_guard_retains_features_on_prod_written_h5ad(written_artifacts):
    out, manifest = written_artifacts
    adapter = ProteomicsAdapter.__new__(ProteomicsAdapter)
    expected = manifest["expected_kept"]

    for entry in manifest["files"]:
        adata = ad.read_h5ad(out / entry["h5ad"])
        original = np.asarray(
            adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        )

        for flag_col in ("is_contaminant", "is_reverse"):
            mask = boolean_flag_mask(adata.var[flag_col])
            kept = adata[:, ~mask].copy()
            assert list(kept.var_names) == expected, (
                f"{entry['encoding']}/{flag_col}: kept {list(kept.var_names)} "
                f"(persisted dtype {adata.var[flag_col].dtype})"
            )
            keptX = np.asarray(
                kept.X.toarray() if hasattr(kept.X, "toarray") else kept.X
            )
            assert np.allclose(keptX, original[:, [1, 3]]), (
                f"{entry['encoding']}/{flag_col}: numeric matrix altered"
            )

        # A persisted text metadata column must classify as metadata, and a
        # numeric-encoded text column must stay intensity (never blanked).
        assert not adapter._is_numeric_string_column(adata.var["extra_label"])
        assert pdt.is_string_dtype(adata.var["extra_label"])
        assert adapter._is_numeric_string_column(adata.var["num_as_text"])


def test_hardened_guard_retains_features_on_prod_written_zarr(written_artifacts):
    out, manifest = written_artifacts
    zarr_entries = [e for e in manifest["files"] if "zarr" in e]
    if not zarr_entries:
        pytest.skip("writer produced no zarr artifacts in this image")
    for entry in zarr_entries:
        adata = ad.read_zarr(out / entry["zarr"])
        mask = boolean_flag_mask(adata.var["is_contaminant"])
        assert list(adata[:, ~mask].var_names) == manifest["expected_kept"], (
            f"{entry['encoding']} (zarr)"
        )


def test_current_code_empties_matrix_in_prod_runtime():
    """The bug is real in the prod runtime and the hardened guard fixes it."""
    proc = _docker_python("in_image_compare.py")
    assert proc.returncode == 0, (
        f"hardened guard wrong in prod image:\n{proc.stdout}\n{proc.stderr}"
    )
    summary_line = next(
        ln for ln in proc.stdout.splitlines() if ln.startswith("SUMMARY_JSON ")
    )
    summary = json.loads(summary_line[len("SUMMARY_JSON "):])
    assert summary["pandas"].startswith("3.")
    assert summary["new_wrong"] == 0, summary
    # '1'/'0' and 'True'/'False' both empty the matrix under the current guard.
    assert summary["old_emptied"] >= 2, (
        f"expected the current guard to empty >=2 encodings, got {summary}"
    )
