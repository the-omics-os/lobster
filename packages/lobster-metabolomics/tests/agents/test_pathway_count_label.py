"""The metabolomics pathway reply says its significant pathways are counted at an
uncorrected p-value: they were counted at raw p < 0.05 under the bare label
"Significant Pathways"."""

from unittest.mock import MagicMock

import anndata as ad
import numpy as np
import pandas as pd

from lobster.agents.metabolomics.shared_tools import create_shared_tools
from lobster.testing.provenance_replay import RecordingDataManager


def test_the_pathway_count_says_raw_p_not_corrected(tmp_path, monkeypatch):
    from lobster.core.component_registry import component_registry

    service = MagicMock()
    service.overrepresentation_analysis.return_value = {
        "glycolysis": {"p_value": 0.001},
        "tca cycle": {"p_value": 0.03},
        "urea cycle": {"p_value": 0.4},
    }
    monkeypatch.setattr(component_registry, "get_service", lambda name: service)
    adata = ad.AnnData(
        np.ones((4, 2)),
        var=pd.DataFrame(
            {"fdr": [0.01, 0.01], "annotation_name": ["glucose", "citrate"]},
            index=["m1", "m2"],
        ),
    )
    dm = RecordingDataManager({"met": adata}, workspace_path=tmp_path / "ws")
    tools = create_shared_tools(dm, MagicMock(), MagicMock(), MagicMock(), MagicMock())
    tool = {t.name: t for t in tools}["run_pathway_enrichment"]

    reply = tool.invoke({"modality_name": "met"})

    assert "**Significant Pathways (raw p < 0.05, not corrected): 2**" in reply, reply
