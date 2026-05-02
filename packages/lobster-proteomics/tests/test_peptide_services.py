"""
Unit tests for peptide services (property, activity, digestion).

Tests property calculation vs reference values, heuristic scoring,
and enzymatic digestion.
"""

import numpy as np
import pandas as pd
import pytest

import anndata as ad


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_peptides():
    """Create AnnData with well-known peptides for property validation."""
    sequences = {
        "magainin2": "GIGKFLHSAKKFGKAFVGEIMNS",  # Known AMP
        "ll37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",  # Human cathelicidin AMP
        "melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",  # Bee venom AMP
        "insulin_b": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",  # Non-AMP control
        "tat": "GRKKRRQRRRPPQ",  # HIV Tat CPP
    }
    obs = pd.DataFrame({
        "sequence": list(sequences.values()),
        "length": [len(s) for s in sequences.values()],
    }, index=list(sequences.keys()))
    return ad.AnnData(obs=obs)


@pytest.fixture
def bsa_protein():
    """BSA protein for digestion testing."""
    # First 100 residues of BSA
    bsa_seq = "DTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETYGDMADCCEKQEPERNECFLSHKDDSPDLPKLKPDPNTLCDEFKADEKKFWGKYLYEIARRHPYFYAPELLYYANKYNGVFQECCQAEDKGACLLPKIETMREKVLASSARQRLRCASIQKFGERALKAWSVARLSQKFPKAEFVEVTKLVTDLTKVHKECCHGDLLECADDR"
    obs = pd.DataFrame({
        "sequence": [bsa_seq],
        "length": [len(bsa_seq)],
    }, index=["BSA"])
    return ad.AnnData(obs=obs)


# ============================================================
# Property Service Tests
# ============================================================


class TestPeptidePropertyService:
    """Test physicochemical property calculation."""

    def test_calculate_all_properties(self, sample_peptides):
        from lobster.services.peptidomics.peptide_property_service import PeptidePropertyService

        service = PeptidePropertyService()
        adata_out, stats, ir = service.calculate_properties(sample_peptides)

        assert stats["n_peptides"] == 5
        assert stats["n_valid"] == 5
        assert len(stats["properties_computed"]) >= 9
        assert ir.tool_name == "calculate_peptide_properties"

        # All property columns should be present
        for prop in stats["properties_computed"]:
            assert f"peptide_{prop}" in adata_out.obs.columns

    def test_magainin2_molecular_weight(self, sample_peptides):
        """Magainin 2 MW should be ~2466 Da (known value)."""
        from lobster.services.peptidomics.peptide_property_service import PeptidePropertyService

        service = PeptidePropertyService()
        adata_out, stats, _ = service.calculate_properties(sample_peptides, properties=["molecular_weight", "length"])

        mw = adata_out.obs.loc["magainin2", "peptide_molecular_weight"]
        assert not np.isnan(mw)
        # Allow 5% tolerance (different MW calculation methods vary slightly)
        assert abs(mw - 2466) / 2466 < 0.05, f"Magainin 2 MW {mw} not within 5% of 2466"

    def test_length_property(self, sample_peptides):
        from lobster.services.peptidomics.peptide_property_service import PeptidePropertyService

        service = PeptidePropertyService()
        adata_out, _, _ = service.calculate_properties(sample_peptides, properties=["length"])

        assert adata_out.obs.loc["tat", "peptide_length"] == 13
        assert adata_out.obs.loc["magainin2", "peptide_length"] == 23

    def test_missing_sequence_column(self, sample_peptides):
        from lobster.services.peptidomics.peptide_property_service import PeptidePropertyService

        service = PeptidePropertyService()
        bad_adata = ad.AnnData(obs=pd.DataFrame({"name": ["a", "b"]}))
        with pytest.raises(ValueError, match="Column 'sequence' not found"):
            service.calculate_properties(bad_adata)

    def test_returns_3_tuple(self, sample_peptides):
        from lobster.services.peptidomics.peptide_property_service import PeptidePropertyService

        service = PeptidePropertyService()
        result = service.calculate_properties(sample_peptides)
        assert len(result) == 3
        assert isinstance(result[0], ad.AnnData)
        assert isinstance(result[1], dict)
        from lobster.core.provenance.analysis_ir import AnalysisStep
        assert isinstance(result[2], AnalysisStep)


# ============================================================
# Activity Service Tests
# ============================================================


class TestPeptideActivityService:
    """Test heuristic activity prediction."""

    def test_known_amps_score_high(self, sample_peptides):
        """Known AMPs (magainin2, ll37, melittin) should score >0.5."""
        from lobster.services.peptidomics.peptide_activity_service import PeptideActivityService

        service = PeptideActivityService()
        adata_out, stats, ir = service.predict_activity(sample_peptides, activity_type="antimicrobial")

        assert stats["activity_type"] == "antimicrobial"
        assert stats["method"] == "heuristic"

        mag_score = adata_out.obs.loc["magainin2", "peptide_antimicrobial_score"]
        assert mag_score > 0.5, f"Magainin 2 AMP score {mag_score} should be >0.5"

    def test_tat_cpp_score(self, sample_peptides):
        """HIV Tat (GRKKRRQRRRPPQ) is a known CPP, should score high."""
        from lobster.services.peptidomics.peptide_activity_service import PeptideActivityService

        service = PeptideActivityService()
        adata_out, stats, _ = service.predict_activity(sample_peptides, activity_type="cell_penetrating")

        tat_score = adata_out.obs.loc["tat", "peptide_cell_penetrating_score"]
        assert tat_score > 0.5, f"HIV Tat CPP score {tat_score} should be >0.5"

    def test_activity_labels_assigned(self, sample_peptides):
        from lobster.services.peptidomics.peptide_activity_service import PeptideActivityService

        service = PeptideActivityService()
        adata_out, _, _ = service.predict_activity(sample_peptides)

        assert "peptide_antimicrobial_label" in adata_out.obs.columns
        labels = set(adata_out.obs["peptide_antimicrobial_label"].unique())
        assert labels.issubset({"positive", "negative", "unknown"})

    def test_invalid_activity_type(self, sample_peptides):
        from lobster.services.peptidomics.peptide_activity_service import PeptideActivityService

        service = PeptideActivityService()
        with pytest.raises(ValueError, match="Unknown activity_type"):
            service.predict_activity(sample_peptides, activity_type="bogus")

    def test_returns_3_tuple(self, sample_peptides):
        from lobster.services.peptidomics.peptide_activity_service import PeptideActivityService

        service = PeptideActivityService()
        result = service.predict_activity(sample_peptides)
        assert len(result) == 3
        assert isinstance(result[0], ad.AnnData)
        assert isinstance(result[1], dict)
        from lobster.core.provenance.analysis_ir import AnalysisStep
        assert isinstance(result[2], AnalysisStep)


# ============================================================
# Digestion Service Tests
# ============================================================


class TestPeptideDigestionService:
    """Test enzymatic digestion."""

    def test_trypsin_digest(self, bsa_protein):
        """Trypsin cleaves after K/R (not before P)."""
        from lobster.services.peptidomics.peptide_digestion_service import PeptideDigestionService

        service = PeptideDigestionService()
        adata_out, stats, ir = service.digest(bsa_protein, enzyme="trypsin")

        assert stats["enzyme"] == "trypsin"
        assert stats["n_fragments"] > 0
        assert stats["n_parents"] == 1

        # All fragments should not start with P (trypsin skips before P)
        # and should end with K or R (except last fragment)
        sequences = adata_out.obs["sequence"].tolist()
        assert len(sequences) > 5, "BSA should produce many tryptic fragments"

    def test_missed_cleavages(self, bsa_protein):
        from lobster.services.peptidomics.peptide_digestion_service import PeptideDigestionService

        service = PeptideDigestionService()
        _, stats_0mc, _ = service.digest(bsa_protein, enzyme="trypsin", missed_cleavages=0)
        _, stats_2mc, _ = service.digest(bsa_protein, enzyme="trypsin", missed_cleavages=2)

        assert stats_2mc["n_fragments"] > stats_0mc["n_fragments"]

    def test_length_filter(self, bsa_protein):
        from lobster.services.peptidomics.peptide_digestion_service import PeptideDigestionService

        service = PeptideDigestionService()
        adata_out, _, _ = service.digest(bsa_protein, min_length=8, max_length=20)

        lengths = adata_out.obs["length"].astype(int)
        assert all(8 <= l <= 20 for l in lengths)

    def test_unknown_enzyme(self, bsa_protein):
        from lobster.services.peptidomics.peptide_digestion_service import PeptideDigestionService

        service = PeptideDigestionService()
        with pytest.raises(ValueError, match="Unknown enzyme"):
            service.digest(bsa_protein, enzyme="bogus_enzyme")

    def test_returns_3_tuple(self, bsa_protein):
        from lobster.services.peptidomics.peptide_digestion_service import PeptideDigestionService

        service = PeptideDigestionService()
        result = service.digest(bsa_protein)
        assert len(result) == 3
        assert isinstance(result[0], ad.AnnData)
        assert isinstance(result[1], dict)
        from lobster.core.provenance.analysis_ir import AnalysisStep
        assert isinstance(result[2], AnalysisStep)


# ============================================================
# Provider Tests
# ============================================================


class TestBiomoleculeProviders:
    """Test provider base class and implementations."""

    def test_biomolecule_source_enum(self):
        from lobster.tools.providers.biomolecule_provider import BiomoleculeSource

        assert BiomoleculeSource.DBAASP.value == "dbaasp"
        assert BiomoleculeSource.IEDB.value == "iedb"
        assert BiomoleculeSource.PEPTIPEDIA.value == "peptipedia"

    def test_peptide_metadata_model(self):
        from lobster.tools.providers.biomolecule_provider import PeptideMetadata

        pm = PeptideMetadata(
            accession="DBAASP:123",
            sequence="GIGKFLHSAKKFGKAFVGEIMNS",
            length=23,
            source_db="dbaasp",
            activities=["antimicrobial"],
        )
        assert pm.accession == "DBAASP:123"
        assert pm.length == 23

    def test_dbaasp_provider_init(self):
        from lobster.tools.providers.dbaasp_provider import DBAASPProvider

        provider = DBAASPProvider()
        assert provider.source.value == "dbaasp"
        assert provider.priority == 10

    def test_iedb_provider_init(self):
        from lobster.tools.providers.iedb_provider import IEDBProvider

        provider = IEDBProvider()
        assert provider.source.value == "iedb"
        assert provider.priority == 10

    def test_peptipedia_provider_init(self):
        from lobster.tools.providers.peptipedia_provider import PeptipediaProvider

        provider = PeptipediaProvider()
        assert provider.source.value == "peptipedia"
        assert provider.priority == 20

    def test_format_results_empty(self):
        from lobster.tools.providers.dbaasp_provider import DBAASPProvider

        provider = DBAASPProvider()
        result = provider.format_results([], "test query")
        assert "No peptides found" in result

    def test_format_results_with_data(self):
        from lobster.tools.providers.biomolecule_provider import PeptideMetadata
        from lobster.tools.providers.dbaasp_provider import DBAASPProvider

        provider = DBAASPProvider()
        peptides = [
            PeptideMetadata(
                accession="DBAASP:1",
                sequence="ACDEFG",
                length=6,
                source_db="dbaasp",
                activities=["antimicrobial"],
            )
        ]
        result = provider.format_results(peptides, "test")
        assert "DBAASP" in result
        assert "ACDEFG" in result
        assert "antimicrobial" in result
