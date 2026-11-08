"""GEO platform gate: array platforms are refused by title or GEO technology,
methylation arrays are named as such, and unknown platforms still get the
modality check. Titles and technologies are copied from the GEO platform records."""

from unittest.mock import Mock, patch

import pytest

from lobster.core.exceptions import UnsupportedPlatformError
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_access.geo_service import GEOService


@pytest.fixture
def geo_service(tmp_path):
    return GEOService(data_manager=DataManagerV2(workspace_path=tmp_path / "ws"))


def _metadata(gpl: str, title: str, technology: str, soft_keys: bool = False) -> dict:
    platform = (
        {"platform_title": title, "platform_technology": technology}
        if soft_keys
        else {"title": title, "technology": technology}
    )
    return {
        "title": "test series",
        "platforms": {gpl: platform},
        "samples": {"GSM1": {"platform_id": gpl, "title": "s1"}},
    }


def _supported_rna():
    result = Mock()
    result.modality = "bulk_rna"
    result.is_supported = True
    result.confidence = 0.9
    result.detected_signals = ["RNA-Seq"]
    result.compatibility_reason = "RNA-seq"
    result.suggestions = []
    return result


@pytest.mark.parametrize(
    "gpl,title,technology,platform_type",
    [
        # Neither title matches the old keyword list.
        (
            "GPL3255",
            "VUMC MACF human 19K oligo v33",
            "spotted oligonucleotide",
            "microarray",
        ),
        (
            "GPL21145",
            "Infinium MethylationEPIC",
            "oligonucleotide beads",
            "methylation_array",
        ),
        (
            "GPL13534",
            "Illumina HumanMethylation450 BeadChip (HumanMethylation450_15017482)",
            "oligonucleotide beads",
            "methylation_array",
        ),
    ],
)
def test_array_platforms_are_refused_and_labelled(
    geo_service, gpl, title, technology, platform_type
):
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        with pytest.raises(UnsupportedPlatformError) as err:
            geo_service._check_platform_compatibility(
                "GSE1", _metadata(gpl, title, technology)
            )
    assert err.value.details["platform_type"] == platform_type
    assert err.value.details["modality"] == platform_type
    llm.detect_modality.assert_not_called()


def test_methylation_message_does_not_call_it_a_microarray(geo_service):
    with pytest.raises(UnsupportedPlatformError) as err:
        geo_service._check_platform_compatibility(
            "GSE1",
            _metadata("GPL21145", "Infinium MethylationEPIC", "oligonucleotide beads"),
        )
    explanation = err.value.details["explanation"]
    assert "methylation" in explanation.lower()
    assert "microarray" not in explanation.lower()


def test_array_technology_is_read_from_soft_keys(geo_service):
    with pytest.raises(UnsupportedPlatformError):
        geo_service._check_platform_compatibility(
            "GSE1",
            _metadata(
                "GPL3255",
                "VUMC MACF human 19K oligo v33",
                "spotted oligonucleotide",
                soft_keys=True,
            ),
        )


def test_unknown_platform_still_gets_the_modality_check(geo_service):
    metadata = _metadata("GPL99999", "In-house sequencer v2", "")
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        llm.detect_modality.return_value = _supported_rna()
        ok, message = geo_service._check_platform_compatibility("GSE1", metadata)
    assert ok is True
    assert message != "Unknown platform - will attempt loading"
    llm.detect_modality.assert_called_once()


def test_sequencing_platform_is_unchanged(geo_service):
    metadata = _metadata(
        "GPL24676", "Illumina NovaSeq 6000 (Homo sapiens)", "high-throughput sequencing"
    )
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        llm.detect_modality.return_value = _supported_rna()
        ok, _ = geo_service._check_platform_compatibility("GSE1", metadata)
    assert ok is True
    llm.detect_modality.assert_called_once()


def test_oligo_title_keyword_refuses_without_technology(geo_service):
    with pytest.raises(UnsupportedPlatformError):
        geo_service._check_platform_compatibility(
            "GSE1", _metadata("GPL3255", "VUMC MACF human 19K oligo v33", "")
        )


@pytest.mark.parametrize(
    "title", ["oligodendrocyte", "oligodendroglioma", "oligo-dT primed RNA-seq"]
)
def test_oligo_is_matched_as_a_word(geo_service, title):
    metadata = _metadata("GPL99999", title, "")
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        llm.detect_modality.return_value = _supported_rna()
        ok, _ = geo_service._check_platform_compatibility("GSE1", metadata)
    assert ok is True
