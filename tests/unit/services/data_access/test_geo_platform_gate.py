"""GEO platform gate: array platforms are refused by title or GEO technology,
methylation arrays are named as such, and unknown platforms still get the
modality check. Titles and technologies are copied from the GEO platform records."""

from unittest.mock import Mock, patch

import pytest

# GEOService imports scanpy (the transcriptomics extra); skip, not error, without it.
pytest.importorskip("scanpy")

from lobster.core.exceptions import UnsupportedPlatformError
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.data_access.geo.metadata_fetch import MetadataFetcher
from lobster.services.data_access.geo_service import GEOService


@pytest.fixture
def geo_service(tmp_path):
    return GEOService(data_manager=DataManagerV2(workspace_path=tmp_path / "ws"))


def _metadata(gpl: str, title: str, technology: str) -> dict:
    return {
        "title": "test series",
        "platforms": {gpl: {"title": title, "technology": technology}},
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
        # Not registered; the GEO technology decides. The title matches no
        # keyword.
        (
            "GPL99998",
            "Custom spotted cDNA v1",
            "spotted dna/cdna",
            "microarray",
        ),
        # Not registered; refused by technology (and by the "oligo" word).
        (
            "GPL3255",
            "VUMC MACF human 19K oligo v33",
            "spotted oligonucleotide",
            "microarray",
        ),
        # Registered as methylation arrays.
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
    "title",
    [
        "oligodendrocyte",
        "oligodendroglioma",
        "oligo-dT primed RNA-seq",
        "oligoclonal B cells",
    ],
)
def test_oligo_is_matched_as_a_word(geo_service, title):
    metadata = _metadata("GPL99999", title, "")
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        llm.detect_modality.return_value = _supported_rna()
        ok, _ = geo_service._check_platform_compatibility("GSE1", metadata)
    assert ok is True


@pytest.mark.parametrize(
    "title", ["Custom oligo pool sequencing", "In-house array capture panel"]
)
def test_sequencing_technology_overrides_array_title_keywords(geo_service, title):
    metadata = _metadata("GPL99999", title, "high-throughput sequencing")
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        llm.detect_modality.return_value = _supported_rna()
        ok, _ = geo_service._check_platform_compatibility("GSE1", metadata)
    assert ok is True
    llm.detect_modality.assert_called_once()


@pytest.mark.parametrize("gpl", ["GPL8490", "GPL13534", "GPL21145"])
def test_methylation_gpl_is_named_without_a_title(geo_service, gpl):
    # Entrez-only metadata: placeholder title, technology defaulted to sequencing.
    metadata = _metadata(gpl, f"Platform {gpl}", "high throughput sequencing")
    with pytest.raises(UnsupportedPlatformError) as err:
        geo_service._check_platform_compatibility("GSE1", metadata)
    assert err.value.details["platform_type"] == "methylation_array"


def test_unrecognised_technology_falls_back_to_title_keywords(geo_service):
    metadata = _metadata(
        "GPL99999",
        "Affymetrix custom array",
        "high-throughput sequencing; in situ oligonucleotide",
    )
    with pytest.raises(UnsupportedPlatformError):
        geo_service._check_platform_compatibility("GSE1", metadata)


def test_entrez_platforms_do_not_invent_a_technology():
    platforms = MetadataFetcher(Mock())._extract_platform_info_from_entrez(
        {"taxon": "Homo sapiens", "ptechtype": ""}, ["GPL570", "GPL99999"]
    )
    assert {p["technology"] for p in platforms.values()} == {""}


def test_entrez_style_array_series_is_not_waved_through_as_mixed(geo_service):
    # A registered array plus an unregistered GPL, both from Entrez metadata
    # with no technology: must stay refused, not become a "mixed" series.
    platforms = MetadataFetcher(Mock())._extract_platform_info_from_entrez(
        {"taxon": "Homo sapiens", "ptechtype": ""}, ["GPL570", "GPL99999"]
    )
    metadata = {"title": "test series", "platforms": platforms, "samples": {}}
    with patch.object(geo_service, "_data_expert_assistant", create=True):
        with pytest.raises(UnsupportedPlatformError):
            geo_service._check_platform_compatibility("GSE1", metadata)


def test_sequencing_keyword_wins_a_tie_without_technology(geo_service):
    metadata = _metadata("GPL99997", "Illumina HiSeq 2500 array-capture panel", "")
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        llm.detect_modality.return_value = _supported_rna()
        ok, _ = geo_service._check_platform_compatibility("GSE1", metadata)
    assert ok is True
    llm.detect_modality.assert_called_once()


@pytest.mark.parametrize(
    "title",
    [
        "Affymetrix Human Mitochondrial Resequencing Array 2.0",
        "GeneChip resequencing array",
    ],
)
def test_resequencing_array_is_not_a_sequencing_keyword(geo_service, title):
    with pytest.raises(UnsupportedPlatformError):
        geo_service._check_platform_compatibility(
            "GSE1", _metadata("GPL99996", title, "")
        )


@pytest.mark.parametrize(
    "title", ["In-house scRNA-seq", "Custom snRNA-seq", "In-house mRNA-seq"]
)
def test_rna_seq_inside_a_word_is_a_sequencing_keyword(geo_service, title):
    # Next to an array platform, a sequencing platform makes the series mixed
    # instead of array-only, so it is not refused.
    metadata = {
        "title": "test series",
        "platforms": {
            "GPL3255": {
                "title": "VUMC MACF human 19K oligo v33",
                "technology": "spotted oligonucleotide",
            },
            "GPL99995": {"title": title, "technology": ""},
        },
        "samples": {},
    }
    ok, message = geo_service._check_platform_compatibility("GSE1", metadata)
    assert ok is True
    assert message.startswith("Mixed platform dataset")


def test_rna_seq_inside_a_word_wins_over_an_array_keyword(geo_service):
    metadata = _metadata("GPL99995", "snRNA-seq capture array", "")
    with patch.object(geo_service, "_data_expert_assistant", create=True) as llm:
        llm.detect_modality.return_value = _supported_rna()
        ok, _ = geo_service._check_platform_compatibility("GSE1", metadata)
    assert ok is True
    llm.detect_modality.assert_called_once()
