"""
Unit tests for knowledgebase tool factories.

Tests the three factory functions that produce LangChain @tool-decorated closures:
- create_cross_database_id_mapping_tool -> map_cross_database_ids
- create_variant_consequence_tool -> predict_variant_consequences
- create_sequence_retrieval_tool -> get_ensembl_sequence

Each factory wraps lazy-loaded services (UniProtService, EnsemblService) with
markdown formatting and provenance logging via data_manager.log_tool_usage().
"""

from unittest.mock import MagicMock, patch

import pytest

from lobster.tools.knowledgebase_tools import (
    _map_to_ensembl_external_db,
    create_cross_database_id_mapping_tool,
    create_sequence_retrieval_tool,
    create_variant_consequence_tool,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManagerV2 with log_tool_usage as MagicMock."""
    dm = MagicMock()
    dm.log_tool_usage = MagicMock()
    return dm


@pytest.fixture
def mock_uniprot_service():
    """Create a mock UniProtService."""
    return MagicMock()


@pytest.fixture
def mock_ensembl_service():
    """Create a mock EnsemblService."""
    return MagicMock()


# =============================================================================
# Helper: _map_to_ensembl_external_db
# =============================================================================


class TestMapToEnsemblExternalDb:
    """Tests for the internal _map_to_ensembl_external_db helper."""

    def test_known_mappings(self):
        assert _map_to_ensembl_external_db("UniProtKB_AC-ID") == "UniProt/SWISSPROT"
        assert _map_to_ensembl_external_db("UniProtKB") == "UniProt/SWISSPROT"
        assert _map_to_ensembl_external_db("PDB") == "PDB"
        assert _map_to_ensembl_external_db("GeneID") == "EntrezGene"
        assert _map_to_ensembl_external_db("HGNC") == "HGNC"
        assert _map_to_ensembl_external_db("RefSeq_mRNA") == "RefSeq_mRNA"
        assert _map_to_ensembl_external_db("RefSeq_Protein") == "RefSeq_peptide"
        assert _map_to_ensembl_external_db("MIM_GENE") == "MIM_GENE"

    def test_unknown_mapping_returns_none(self):
        assert _map_to_ensembl_external_db("SomeUnknownDB") is None


# =============================================================================
# Tool 1: Cross-Database ID Mapping
# =============================================================================


class TestCrossDatabaseIdMappingTool:
    """Tests for the map_cross_database_ids tool."""

    def test_factory_returns_langchain_tool(self, mock_data_manager):
        """Factory returns a LangChain StructuredTool with invoke interface."""
        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        assert hasattr(tool, "invoke")
        assert hasattr(tool, "name")
        assert tool.name == "map_cross_database_ids"

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_uniprot_id_mapping_path(
        self, MockUniProtClass, mock_data_manager
    ):
        """UniProt ID mapping is used for non-Ensembl source databases."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service

        mock_service.map_ids.return_value = {
            "results": [
                {"from": "TP53", "to": {"primaryAccession": "P04637"}},
                {"from": "BRCA1", "to": {"primaryAccession": "P38398"}},
            ],
            "failedIds": [],
        }

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "TP53,BRCA1",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })

        # Verify service was called correctly
        mock_service.map_ids.assert_called_once_with(
            from_db="Gene_Name", to_db="UniProtKB_AC-ID", ids=["TP53", "BRCA1"]
        )

        # Verify output formatting
        assert "Cross-Database Mapping: Gene_Name" in result
        assert "Mapped 2 identifier(s)" in result
        assert "TP53" in result
        assert "P04637" in result
        assert "BRCA1" in result
        assert "P38398" in result

        # Verify provenance logging
        mock_data_manager.log_tool_usage.assert_called_once()
        call_kwargs = mock_data_manager.log_tool_usage.call_args
        assert call_kwargs.kwargs["tool_name"] == "map_cross_database_ids"
        assert call_kwargs.kwargs["parameters"]["backend"] == "uniprot_idmapping"
        assert "2 IDs via UniProt" in call_kwargs.kwargs["description"]

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_uniprot_path_with_failed_ids(
        self, MockUniProtClass, mock_data_manager
    ):
        """UniProt path reports failed IDs when present."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service

        mock_service.map_ids.return_value = {
            "results": [
                {"from": "TP53", "to": {"primaryAccession": "P04637"}},
            ],
            "failedIds": ["FAKEGENEX"],
        }

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "TP53,FAKEGENEX",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })

        assert "P04637" in result
        assert "Failed IDs: FAKEGENEX" in result

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_uniprot_path_no_results(
        self, MockUniProtClass, mock_data_manager
    ):
        """UniProt path returns informative message when no mappings found."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service

        mock_service.map_ids.return_value = {
            "results": [],
            "failedIds": ["UNKNOWNGENE"],
        }

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "UNKNOWNGENE",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })

        assert "No mappings found" in result
        assert "UNKNOWNGENE" in result
        # log_tool_usage should NOT be called when there are no results
        mock_data_manager.log_tool_usage.assert_not_called()

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_uniprot_path_target_as_string(
        self, MockUniProtClass, mock_data_manager
    ):
        """UniProt path handles to-value as plain string instead of dict."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service

        mock_service.map_ids.return_value = {
            "results": [
                {"from": "P04637", "to": "4HJE"},
            ],
            "failedIds": [],
        }

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "P04637",
            "from_db": "UniProtKB_AC-ID",
            "to_db": "PDB",
        })

        assert "P04637" in result
        assert "4HJE" in result

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_uniprot_path_caps_at_50_results(
        self, MockUniProtClass, mock_data_manager
    ):
        """UniProt path caps output at 50 results and shows overflow message."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service

        # Build 60 results
        results = [
            {"from": f"GENE{i}", "to": {"primaryAccession": f"P{i:05d}"}}
            for i in range(60)
        ]
        mock_service.map_ids.return_value = {"results": results, "failedIds": []}

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": ",".join(f"GENE{i}" for i in range(60)),
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })

        assert "Mapped 60 identifier(s)" in result
        assert "... and 10 more" in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_ensembl_xrefs_path(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Ensembl xrefs path is used for single Ensembl ID lookups."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        mock_service.get_xrefs.return_value = [
            {
                "primary_id": "P04637",
                "display_id": "P53_HUMAN",
                "dbname": "UniProt/SWISSPROT",
                "description": "Cellular tumor antigen p53",
            },
        ]

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "ENSG00000141510",
            "from_db": "Ensembl",
            "to_db": "UniProtKB_AC-ID",
        })

        # Verify Ensembl service was used
        mock_service.get_xrefs.assert_called_once_with(
            "ENSG00000141510", external_db="UniProt/SWISSPROT"
        )

        # Verify output
        assert "Cross-Database Mapping: ENSG00000141510" in result
        assert "1 reference(s)" in result
        assert "P04637" in result
        assert "P53_HUMAN" in result
        assert "UniProt/SWISSPROT" in result
        assert "Cellular tumor antigen p53" in result

        # Verify provenance logging with ensembl_xrefs backend
        mock_data_manager.log_tool_usage.assert_called_once()
        call_kwargs = mock_data_manager.log_tool_usage.call_args
        assert call_kwargs.kwargs["parameters"]["backend"] == "ensembl_xrefs"

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_ensembl_xrefs_no_results(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Ensembl xrefs path with no cross-references found."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_xrefs.return_value = []

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "ENSG99999999999",
            "from_db": "Ensembl",
            "to_db": "PDB",
        })

        assert "No cross-references found" in result
        assert "ENSG99999999999" in result
        mock_data_manager.log_tool_usage.assert_not_called()

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_ensembl_xrefs_minimal_fields(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Ensembl xrefs with only primary_id (no display, dbname, desc)."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        mock_service.get_xrefs.return_value = [
            {"primary_id": "12345"},
        ]

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "ENSG00000141510",
            "from_db": "Ensembl",
            "to_db": "GeneID",
        })

        assert "12345" in result
        assert "1 reference(s)" in result

    def test_empty_ids_error(self, mock_data_manager):
        """Empty IDs string returns error message without calling any service."""
        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })

        assert "Error" in result
        assert "No identifiers provided" in result
        mock_data_manager.log_tool_usage.assert_not_called()

    def test_whitespace_only_ids_error(self, mock_data_manager):
        """Whitespace-only IDs string returns error message."""
        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "  , ,  ",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })

        assert "No identifiers provided" in result

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_service_error_handling(
        self, MockUniProtClass, mock_data_manager
    ):
        """Service exceptions are caught and returned as error strings."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service
        mock_service.map_ids.side_effect = RuntimeError("API timeout")

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "TP53",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })

        assert "Error mapping IDs" in result
        assert "API timeout" in result
        mock_data_manager.log_tool_usage.assert_not_called()

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_ensembl_service_error_handling(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Ensembl xrefs errors are caught and returned as error strings."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_xrefs.side_effect = ConnectionError("Ensembl down")

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "ENSG00000141510",
            "from_db": "Ensembl",
            "to_db": "UniProtKB_AC-ID",
        })

        assert "Error mapping IDs" in result
        assert "Ensembl down" in result

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_ensembl_from_db_multiple_ids_uses_uniprot(
        self, MockUniProtClass, mock_data_manager
    ):
        """Multiple Ensembl IDs fall through to UniProt path (xrefs only for single IDs)."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service

        mock_service.map_ids.return_value = {
            "results": [
                {"from": "ENSG00000141510", "to": {"primaryAccession": "P04637"}},
                {"from": "ENSG00000012048", "to": {"primaryAccession": "P38398"}},
            ],
            "failedIds": [],
        }

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "ENSG00000141510,ENSG00000012048",
            "from_db": "Ensembl",
            "to_db": "UniProtKB_AC-ID",
        })

        # Should use UniProt (not Ensembl xrefs) because multiple IDs
        mock_service.map_ids.assert_called_once()
        assert "P04637" in result
        assert "P38398" in result

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_uniprot_path_uniprotkb_id_in_target(
        self, MockUniProtClass, mock_data_manager
    ):
        """UniProt target dict with uniProtkbId key (not primaryAccession)."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service

        mock_service.map_ids.return_value = {
            "results": [
                {"from": "TP53", "to": {"uniProtkbId": "P53_HUMAN"}},
            ],
            "failedIds": [],
        }

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        result = tool.invoke({
            "ids": "TP53",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })

        assert "P53_HUMAN" in result


# =============================================================================
# Tool 2: Variant Consequence Prediction
# =============================================================================


class TestVariantConsequenceTool:
    """Tests for the predict_variant_consequences tool."""

    def test_factory_returns_langchain_tool(self, mock_data_manager):
        """Factory returns a LangChain StructuredTool with invoke interface."""
        tool = create_variant_consequence_tool(mock_data_manager)
        assert hasattr(tool, "invoke")
        assert hasattr(tool, "name")
        assert tool.name == "predict_variant_consequences"

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_normal_vep_response(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Normal VEP response with transcript consequences, SIFT, and PolyPhen."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        mock_service.get_variant_consequences.return_value = [
            {
                "most_severe_consequence": "missense_variant",
                "allele_string": "G/C",
                "colocated_variants": [{"id": "rs1042522"}],
                "transcript_consequences": [
                    {
                        "gene_symbol": "TP53",
                        "transcript_id": "ENST00000269305",
                        "consequence_terms": ["missense_variant"],
                        "impact": "MODERATE",
                        "biotype": "protein_coding",
                        "amino_acids": "R/P",
                        "protein_start": 72,
                        "sift_prediction": "tolerated",
                        "sift_score": 0.27,
                        "polyphen_prediction": "benign",
                        "polyphen_score": 0.001,
                    },
                ],
            }
        ]

        tool = create_variant_consequence_tool(mock_data_manager)
        result = tool.invoke({
            "notation": "9:g.22125503G>C",
            "species": "human",
            "notation_type": "hgvs",
        })

        # Verify service was called correctly
        mock_service.get_variant_consequences.assert_called_once_with(
            notation="9:g.22125503G>C",
            species="human",
            notation_type="hgvs",
        )

        # Verify output formatting
        assert "Variant Effect Prediction: 9:g.22125503G>C" in result
        assert "missense_variant" in result
        assert "G/C" in result
        assert "rs1042522" in result
        assert "TP53" in result
        assert "ENST00000269305" in result
        assert "MODERATE" in result
        assert "protein_coding" in result
        assert "p.R/P at pos 72" in result
        assert "SIFT: tolerated(0.27)" in result
        assert "PolyPhen: benign(0.001)" in result

        # Verify provenance logging
        mock_data_manager.log_tool_usage.assert_called_once()
        call_kwargs = mock_data_manager.log_tool_usage.call_args
        assert call_kwargs.kwargs["tool_name"] == "predict_variant_consequences"
        assert call_kwargs.kwargs["parameters"]["notation"] == "9:g.22125503G>C"
        assert call_kwargs.kwargs["parameters"]["species"] == "human"
        assert "missense_variant" in call_kwargs.kwargs["description"]

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_vep_single_result_dict(
        self, MockEnsemblClass, mock_data_manager
    ):
        """VEP response as a single dict (not a list) is handled correctly."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        mock_service.get_variant_consequences.return_value = {
            "most_severe_consequence": "synonymous_variant",
            "allele_string": "A/G",
            "transcript_consequences": [
                {
                    "gene_symbol": "EGFR",
                    "transcript_id": "ENST00000275493",
                    "consequence_terms": ["synonymous_variant"],
                    "impact": "LOW",
                    "biotype": "protein_coding",
                },
            ],
        }

        tool = create_variant_consequence_tool(mock_data_manager)
        result = tool.invoke({
            "notation": "rs12345",
            "notation_type": "id",
        })

        assert "synonymous_variant" in result
        assert "EGFR" in result
        assert "LOW" in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_vep_empty_results(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Empty VEP results return informative message."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_variant_consequences.return_value = []

        tool = create_variant_consequence_tool(mock_data_manager)
        result = tool.invoke({
            "notation": "1:g.999999999A>T",
        })

        assert "No consequences predicted" in result
        assert "1:g.999999999A>T" in result
        mock_data_manager.log_tool_usage.assert_not_called()

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_vep_none_results(
        self, MockEnsemblClass, mock_data_manager
    ):
        """None VEP results return informative message."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_variant_consequences.return_value = None

        tool = create_variant_consequence_tool(mock_data_manager)
        result = tool.invoke({
            "notation": "invalid_notation",
        })

        assert "No consequences predicted" in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_vep_transcript_consequences_capped_at_10(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Output is capped at 10 transcript consequences with overflow message."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        consequences = [
            {
                "gene_symbol": f"GENE{i}",
                "transcript_id": f"ENST{i:011d}",
                "consequence_terms": ["missense_variant"],
                "impact": "MODERATE",
                "biotype": "protein_coding",
            }
            for i in range(15)
        ]
        mock_service.get_variant_consequences.return_value = [
            {
                "most_severe_consequence": "missense_variant",
                "transcript_consequences": consequences,
            }
        ]

        tool = create_variant_consequence_tool(mock_data_manager)
        result = tool.invoke({"notation": "9:g.22125503G>C"})

        # First 10 should be present
        assert "GENE0" in result
        assert "GENE9" in result
        # 11th should NOT be present
        assert "GENE10" not in result
        assert "... and 5 more transcripts" in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_vep_minimal_transcript_fields(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Transcript consequences with only required fields (no optional ones)."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        mock_service.get_variant_consequences.return_value = [
            {
                "most_severe_consequence": "intron_variant",
                "transcript_consequences": [
                    {
                        "gene_id": "ENSG00000141510",
                        "transcript_id": "ENST00000269305",
                        "consequence_terms": ["intron_variant"],
                    },
                ],
            }
        ]

        tool = create_variant_consequence_tool(mock_data_manager)
        result = tool.invoke({"notation": "9:g.22125504A>G"})

        assert "intron_variant" in result
        # Falls back to gene_id when gene_symbol is missing
        assert "ENSG00000141510" in result
        # No SIFT/PolyPhen lines should appear
        assert "SIFT" not in result
        assert "PolyPhen" not in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_vep_no_transcript_consequences(
        self, MockEnsemblClass, mock_data_manager
    ):
        """VEP response with no transcript_consequences key."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        mock_service.get_variant_consequences.return_value = [
            {
                "most_severe_consequence": "intergenic_variant",
                "allele_string": "C/T",
            }
        ]

        tool = create_variant_consequence_tool(mock_data_manager)
        result = tool.invoke({"notation": "1:g.100000C>T"})

        assert "intergenic_variant" in result
        assert "C/T" in result
        # No transcript section
        assert "Transcript consequences" not in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_vep_error_handling(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Service exceptions are caught and returned as error strings."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_variant_consequences.side_effect = RuntimeError(
            "VEP service unavailable"
        )

        tool = create_variant_consequence_tool(mock_data_manager)
        result = tool.invoke({"notation": "9:g.22125503G>C"})

        assert "Error predicting variant consequences" in result
        assert "VEP service unavailable" in result
        mock_data_manager.log_tool_usage.assert_not_called()

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_vep_default_parameters(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Default species and notation_type values are passed correctly."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_variant_consequences.return_value = []

        tool = create_variant_consequence_tool(mock_data_manager)
        tool.invoke({"notation": "rs1042522"})

        mock_service.get_variant_consequences.assert_called_once_with(
            notation="rs1042522",
            species="human",
            notation_type="hgvs",
        )


# =============================================================================
# Tool 3: Sequence Retrieval
# =============================================================================


class TestSequenceRetrievalTool:
    """Tests for the get_ensembl_sequence tool."""

    def test_factory_returns_langchain_tool(self, mock_data_manager):
        """Factory returns a LangChain StructuredTool with invoke interface."""
        tool = create_sequence_retrieval_tool(mock_data_manager)
        assert hasattr(tool, "invoke")
        assert hasattr(tool, "name")
        assert tool.name == "get_ensembl_sequence"

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_short_sequence_shown_fully(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Short sequences (<=500 chars) are shown in full."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        short_seq = "ATGCGATCGATCGATCG"
        mock_service.get_sequence.return_value = {
            "seq": short_seq,
            "id": "ENST00000269305",
            "desc": "tumor protein p53",
            "molecule": "cdna",
        }

        tool = create_sequence_retrieval_tool(mock_data_manager)
        result = tool.invoke({
            "ensembl_id": "ENST00000269305",
            "seq_type": "cdna",
        })

        # Verify service call
        mock_service.get_sequence.assert_called_once_with(
            ensembl_id="ENST00000269305",
            seq_type="cdna",
        )

        # Verify output
        assert "Sequence: ENST00000269305" in result
        assert "cdna" in result
        assert f"{len(short_seq):,} bases" in result
        assert "tumor protein p53" in result
        assert "Full sequence" in result
        assert short_seq in result
        assert "preview" not in result.lower()

        # Verify provenance logging
        mock_data_manager.log_tool_usage.assert_called_once()
        call_kwargs = mock_data_manager.log_tool_usage.call_args
        assert call_kwargs.kwargs["tool_name"] == "get_ensembl_sequence"
        assert call_kwargs.kwargs["parameters"]["ensembl_id"] == "ENST00000269305"
        assert call_kwargs.kwargs["parameters"]["seq_type"] == "cdna"
        assert "bases" in call_kwargs.kwargs["description"]

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_long_sequence_truncated(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Long sequences (>500 chars) are truncated with preview."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        long_seq = "A" * 1000
        mock_service.get_sequence.return_value = {
            "seq": long_seq,
            "id": "ENSG00000141510",
            "desc": "",
            "molecule": "dna",
        }

        tool = create_sequence_retrieval_tool(mock_data_manager)
        result = tool.invoke({
            "ensembl_id": "ENSG00000141510",
            "seq_type": "genomic",
        })

        assert "1,000 bases" in result
        assert "Sequence preview" in result
        assert "first 500 of 1,000" in result
        # Full sequence should NOT appear
        assert "Full sequence" not in result
        # Only first 500 chars + ellipsis
        assert "A" * 500 + "..." in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_protein_sequence_uses_residues(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Protein sequences show 'residues' instead of 'bases'."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        protein_seq = "MEEPQSDPSVEPPLSQETFSDLWK"
        mock_service.get_sequence.return_value = {
            "seq": protein_seq,
            "id": "ENSP00000269305",
            "desc": "p53 protein",
            "molecule": "protein",
        }

        tool = create_sequence_retrieval_tool(mock_data_manager)
        result = tool.invoke({
            "ensembl_id": "ENSP00000269305",
            "seq_type": "protein",
        })

        assert "residues" in result
        assert "bases" not in result
        assert protein_seq in result
        assert "protein" in result

        # Verify provenance description uses 'residues'
        call_kwargs = mock_data_manager.log_tool_usage.call_args
        assert "residues" in call_kwargs.kwargs["description"]

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_sequence_no_description(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Sequence without description field omits Description line."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        mock_service.get_sequence.return_value = {
            "seq": "ATGCGATCG",
            "id": "ENST00000269305",
            "desc": "",
            "molecule": "cdna",
        }

        tool = create_sequence_retrieval_tool(mock_data_manager)
        result = tool.invoke({"ensembl_id": "ENST00000269305"})

        assert "Description" not in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_sequence_fallback_id(
        self, MockEnsemblClass, mock_data_manager
    ):
        """When result has no 'id' key, falls back to the input ensembl_id."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        mock_service.get_sequence.return_value = {
            "seq": "ATGCGA",
            "molecule": "cdna",
        }

        tool = create_sequence_retrieval_tool(mock_data_manager)
        result = tool.invoke({"ensembl_id": "ENST00000269305"})

        # Should fall back to ensembl_id
        assert "ENST00000269305" in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_sequence_exactly_500_chars(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Boundary: sequence of exactly 500 chars is shown fully (not truncated)."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        seq_500 = "G" * 500
        mock_service.get_sequence.return_value = {
            "seq": seq_500,
            "id": "ENST00000000001",
            "molecule": "cdna",
        }

        tool = create_sequence_retrieval_tool(mock_data_manager)
        result = tool.invoke({"ensembl_id": "ENST00000000001"})

        assert "Full sequence" in result
        assert "preview" not in result.lower()

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_sequence_501_chars_is_truncated(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Boundary: sequence of 501 chars triggers truncation."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service

        seq_501 = "C" * 501
        mock_service.get_sequence.return_value = {
            "seq": seq_501,
            "id": "ENST00000000002",
            "molecule": "cdna",
        }

        tool = create_sequence_retrieval_tool(mock_data_manager)
        result = tool.invoke({"ensembl_id": "ENST00000000002"})

        assert "Sequence preview" in result
        assert "Full sequence" not in result

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_sequence_error_handling(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Service exceptions are caught and returned as error strings."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_sequence.side_effect = ValueError(
            "Invalid Ensembl ID format"
        )

        tool = create_sequence_retrieval_tool(mock_data_manager)
        result = tool.invoke({"ensembl_id": "INVALID_ID"})

        assert "Error retrieving sequence" in result
        assert "Invalid Ensembl ID format" in result
        mock_data_manager.log_tool_usage.assert_not_called()

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_sequence_default_seq_type(
        self, MockEnsemblClass, mock_data_manager
    ):
        """Default seq_type is 'cdna'."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_sequence.return_value = {
            "seq": "ATGC",
            "id": "ENST00000269305",
            "molecule": "cdna",
        }

        tool = create_sequence_retrieval_tool(mock_data_manager)
        tool.invoke({"ensembl_id": "ENST00000269305"})

        mock_service.get_sequence.assert_called_once_with(
            ensembl_id="ENST00000269305",
            seq_type="cdna",
        )


# =============================================================================
# Lazy Service Loading
# =============================================================================


class TestLazyServiceLoading:
    """Tests that service instances are lazily created and cached."""

    @patch("lobster.services.data_access.ensembl_service.EnsemblService")
    def test_ensembl_service_created_on_first_invoke(
        self, MockEnsemblClass, mock_data_manager
    ):
        """EnsemblService is not instantiated at factory time, only on first invoke."""
        mock_service = MagicMock()
        MockEnsemblClass.return_value = mock_service
        mock_service.get_sequence.return_value = {
            "seq": "ATG",
            "id": "TEST",
            "molecule": "cdna",
        }

        # Factory call should NOT create service
        tool = create_sequence_retrieval_tool(mock_data_manager)
        MockEnsemblClass.assert_not_called()

        # First invoke creates service
        tool.invoke({"ensembl_id": "ENST00000000001"})
        assert MockEnsemblClass.call_count == 1

        # Second invoke reuses cached service
        tool.invoke({"ensembl_id": "ENST00000000002"})
        assert MockEnsemblClass.call_count == 1

    @patch("lobster.services.data_access.uniprot_service.UniProtService")
    def test_uniprot_service_created_on_first_invoke(
        self, MockUniProtClass, mock_data_manager
    ):
        """UniProtService is not instantiated at factory time, only on first invoke."""
        mock_service = MagicMock()
        MockUniProtClass.return_value = mock_service
        mock_service.map_ids.return_value = {"results": [], "failedIds": []}

        tool = create_cross_database_id_mapping_tool(mock_data_manager)
        MockUniProtClass.assert_not_called()

        tool.invoke({
            "ids": "TP53",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })
        assert MockUniProtClass.call_count == 1

        tool.invoke({
            "ids": "BRCA1",
            "from_db": "Gene_Name",
            "to_db": "UniProtKB_AC-ID",
        })
        assert MockUniProtClass.call_count == 1
