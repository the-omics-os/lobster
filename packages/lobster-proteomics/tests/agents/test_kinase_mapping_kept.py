"""An export never overwrites the custom kinase mapping KSEA read.

run_kinase_enrichment(custom_mapping_path=...) reads a user CSV but did not note
it as a session input, so write_file (or an export) could replace it. It is now
noted like every import tool's file, in this process and on the logged step, so a
restored session still refuses.
"""

import gc
import weakref

import anndata as ad
import numpy as np
import pandas as pd

from lobster.agents.proteomics.de_analysis_expert import de_analysis_expert
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.testing.provenance_replay import factory_tools
from lobster.tools.filesystem_tools import create_filesystem_tools

DE = "lobster.agents.proteomics.de_analysis_expert"
REFUSAL = "Error: " + (
    "this export path is inside the managed figure folder (plots/<figure>/...), "
    "or it is a file this session loaded or imported (or a copy of one). "
    "Exports cannot write there; choose a new file name under exports/ instead."
)
SITES = [f"P{i}_S{i}" for i in range(6)]


def _manager(ws):
    dm = DataManagerV2(workspace_path=ws, session_dir=ws / "session")
    var = pd.DataFrame(
        {"log2_fold_change": [2.0, 1.5, 1.8, -1.0, -1.2, -0.8]}, index=SITES
    )
    dm.store_modality("sites", ad.AnnData(np.ones((2, 6)), var=var))
    return dm


def _enrich(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "project"
    dm = _manager(project / ".lobster_workspace")
    mapping = project / "kinase_map.csv"
    pd.DataFrame({"kinase": ["K1"] * 3 + ["K2"] * 3, "substrate_site": SITES}).to_csv(
        mapping, index=False
    )
    original = mapping.read_bytes()
    reply = factory_tools(de_analysis_expert, DE, dm)["run_kinase_enrichment"].invoke(
        {
            "modality_name": "sites",
            "custom_mapping_path": str(mapping),
            "min_substrates": 3,
        }
    )
    assert "KSEA" in reply and "Error" not in reply, reply
    write = {t.name: t for t in create_filesystem_tools(workspace_path=project)}
    return dm, mapping, original, write["write_file"]


def test_write_file_onto_the_kinase_mapping_is_refused(tmp_path, monkeypatch):
    dm, mapping, original, write_file = _enrich(tmp_path, monkeypatch)

    reply = write_file.invoke({"path": "kinase_map.csv", "content": "x\n"})

    assert reply == REFUSAL, reply
    assert mapping.read_bytes() == original


def test_a_restored_session_still_refuses(tmp_path, monkeypatch):
    dm, mapping, original, write_file = _enrich(tmp_path, monkeypatch)
    ws = dm.workspace_path
    gone = weakref.ref(dm)
    del dm
    gc.collect()
    assert gone() is None  # the earlier process's in-memory record is gone

    restored = _manager(ws)  # a new process restores the session record
    reply = write_file.invoke({"path": "kinase_map.csv", "content": "x\n"})

    assert restored is not None
    assert reply == REFUSAL, reply
    assert mapping.read_bytes() == original
