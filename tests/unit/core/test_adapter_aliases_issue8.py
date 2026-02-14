import pytest

from lobster.core.data_manager_v2 import DataManagerV2


@pytest.fixture
def lightweight_data_manager():
    dm = DataManagerV2.__new__(DataManagerV2)
    dm.adapters = {"transcriptomics_single_cell": object()}
    return dm


@pytest.mark.unit
@pytest.mark.parametrize(
    "adapter_input",
    [
        "single_cell_rna_seq",
        "single-cell rna-seq",
        "scrna",
        "scrna_seq",
        "scRNA-seq",
    ],
)
def test_single_cell_aliases_resolve_to_canonical(
    lightweight_data_manager, adapter_input
):
    resolved = lightweight_data_manager._resolve_adapter_name(adapter_input)
    assert resolved == "transcriptomics_single_cell"


@pytest.mark.unit
def test_canonical_adapter_resolves_as_is(lightweight_data_manager):
    resolved = lightweight_data_manager._resolve_adapter_name("transcriptomics_single_cell")
    assert resolved == "transcriptomics_single_cell"


@pytest.mark.unit
def test_invalid_adapter_still_raises_value_error(lightweight_data_manager):
    with pytest.raises(ValueError, match="Adapter 'not_a_real_adapter' not registered"):
        lightweight_data_manager._resolve_adapter_name("not_a_real_adapter")
