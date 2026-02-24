"""
Step 6: Provenance Chain Validation

Verify that multi-step workflows produce complete, valid W3C-PROV chains.
Every service method must return a valid AnalysisStep with populated fields.
"""

import ast
import re

import pytest

from lobster.core.analysis_ir import AnalysisStep

from .conftest import RDKIT_AVAILABLE

pytestmark = [pytest.mark.integration]


class TestAllServicesReturnValidIR:
    """Every service method must return a valid AnalysisStep."""

    def test_target_scoring_ir(self, scorer):
        """TargetScoringService.score_target produces valid IR."""
        _, stats, ir = scorer.score_target({"genetic_association": 0.5})
        self._assert_valid_ir(ir, "TargetScoring")

    def test_target_ranking_ir(self, scorer):
        """TargetScoringService.rank_targets produces valid IR."""
        _, stats, ir = scorer.rank_targets([
            ("GENE_A", {"genetic_association": 0.5}),
            ("GENE_B", {"known_drug": 0.3}),
        ])
        self._assert_valid_ir(ir, "TargetRanking")

    def test_bliss_synergy_ir(self, syn):
        """SynergyScoringService.bliss_independence produces valid IR."""
        _, stats, ir = syn.bliss_independence(0.3, 0.5, 0.7)
        self._assert_valid_ir(ir, "Synergy-Bliss")

    def test_hsa_synergy_ir(self, syn):
        """SynergyScoringService.hsa_model produces valid IR."""
        _, stats, ir = syn.hsa_model(0.3, 0.5, 0.7)
        self._assert_valid_ir(ir, "Synergy-HSA")

    def test_loewe_synergy_ir(self, syn):
        """SynergyScoringService.loewe_additivity produces valid IR."""
        _, stats, ir = syn.loewe_additivity(5, 10, 10, 20, 0.5)
        self._assert_valid_ir(ir, "Synergy-Loewe")

    @pytest.mark.real_api
    def test_pubchem_ir(self, pubchem):
        """PubChemService.get_compound_properties produces valid IR."""
        _, stats, ir = pubchem.get_compound_properties("aspirin", id_type="name")
        # IR is produced even when API fails
        self._assert_valid_ir(ir, "PubChem")

    @pytest.mark.real_api
    def test_opentargets_evidence_ir(self, ot):
        """OpenTargetsService.get_target_disease_evidence produces valid IR."""
        _, stats, ir = ot.get_target_disease_evidence("ENSG00000146648", limit=3)
        self._assert_valid_ir(ir, "OpenTargets-Evidence")

    @pytest.mark.real_api
    def test_opentargets_score_ir(self, ot):
        """OpenTargetsService.score_target produces valid IR."""
        _, stats, ir = ot.score_target("ENSG00000146648")
        self._assert_valid_ir(ir, "OpenTargets-Score")

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
    def test_molecular_descriptors_ir(self, mol_svc):
        """MolecularAnalysisService.calculate_descriptors produces valid IR."""
        _, stats, ir = mol_svc.calculate_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        self._assert_valid_ir(ir, "MolecularDescriptors")

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
    def test_lipinski_ir(self, mol_svc):
        """MolecularAnalysisService.lipinski_check produces valid IR."""
        _, stats, ir = mol_svc.lipinski_check("CC(=O)OC1=CC=CC=C1C(=O)O")
        self._assert_valid_ir(ir, "Lipinski")

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
    def test_fingerprint_similarity_ir(self, mol_svc):
        """MolecularAnalysisService.fingerprint_similarity produces valid IR."""
        _, stats, ir = mol_svc.fingerprint_similarity(
            ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN(C)C(=N)NC(=N)N"],
            fingerprint="morgan",
        )
        self._assert_valid_ir(ir, "FingerprintSimilarity")

    @staticmethod
    def _assert_valid_ir(ir, service_name: str):
        """Assert that an AnalysisStep has all required fields populated."""
        assert ir is not None, f"{service_name} returned None IR"
        assert isinstance(ir, AnalysisStep), (
            f"{service_name} IR is not AnalysisStep, got {type(ir)}"
        )
        assert ir.operation, f"{service_name} IR missing operation"
        assert ir.tool_name, f"{service_name} IR missing tool_name"
        assert ir.description, f"{service_name} IR missing description"
        assert ir.code_template, f"{service_name} IR missing code_template"
        # imports can be a list or string — just check it exists
        assert ir.imports is not None, f"{service_name} IR missing imports"


class TestIRCodeTemplateValidity:
    """Verify code_templates are syntactically valid Python (after Jinja2 stripping)."""

    @staticmethod
    def _strip_jinja2(code: str) -> str:
        """Replace {{ var }} Jinja2 tokens with valid Python identifiers.

        Uses an unquoted identifier (``_PH``) so it works inside f-strings,
        format strings, and plain code.  Not all replacements will produce
        *semantically* correct code, but the goal is syntactic validity.
        """
        return re.sub(r"\{\{.*?\}\}", "_PH", code)

    def test_target_scoring_code_template(self, scorer):
        _, _, ir = scorer.score_target({"genetic_association": 0.5})
        code = self._strip_jinja2(ir.code_template)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"TargetScoring code_template invalid Python: {e}\n{code}")

    def test_bliss_code_template(self, syn):
        _, _, ir = syn.bliss_independence(0.3, 0.5, 0.7)
        code = self._strip_jinja2(ir.code_template)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Bliss code_template invalid Python: {e}\n{code}")

    def test_loewe_code_template(self, syn):
        _, _, ir = syn.loewe_additivity(5, 10, 10, 20, 0.5)
        code = self._strip_jinja2(ir.code_template)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Loewe code_template invalid Python: {e}\n{code}")

    def test_hsa_code_template(self, syn):
        _, _, ir = syn.hsa_model(0.3, 0.5, 0.7)
        code = self._strip_jinja2(ir.code_template)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"HSA code_template invalid Python: {e}\n{code}")

    @pytest.mark.real_api
    def test_pubchem_code_template(self, pubchem):
        _, _, ir = pubchem.get_compound_properties("aspirin", id_type="name")
        code = self._strip_jinja2(ir.code_template)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"PubChem code_template invalid Python: {e}\n{code}")

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
    def test_molecular_descriptors_code_template(self, mol_svc):
        _, _, ir = mol_svc.calculate_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        code = self._strip_jinja2(ir.code_template)
        try:
            ast.parse(code)
        except SyntaxError as e:
            pytest.fail(f"Descriptors code_template invalid Python: {e}\n{code}")


class TestProvenanceChainCompleteness:
    """Verify a multi-step workflow produces a complete IR chain."""

    @pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
    def test_three_step_chain_all_irs_valid(self, scorer, syn, mol_svc):
        """Score target → score synergy → calculate descriptors: all produce IR."""
        irs = []

        # Step 1: Score a target
        _, _, ir1 = scorer.score_target({
            "genetic_association": 0.8,
            "known_drug": 0.6,
        })
        irs.append(("score_target", ir1))

        # Step 2: Score a synergy combination
        _, _, ir2 = syn.bliss_independence(0.3, 0.5, 0.8)
        irs.append(("bliss_independence", ir2))

        # Step 3: Calculate molecular descriptors
        _, _, ir3 = mol_svc.calculate_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        irs.append(("calculate_descriptors", ir3))

        # Validate each IR in the chain
        for step_name, ir in irs:
            assert isinstance(ir, AnalysisStep), f"{step_name} IR not AnalysisStep"
            assert ir.tool_name == step_name, (
                f"IR tool_name should be '{step_name}', got '{ir.tool_name}'"
            )
            assert ir.operation, f"{step_name} IR missing operation"
            assert ir.code_template, f"{step_name} IR missing code_template"

        # All IRs should have distinct operations
        operations = [ir.operation for _, ir in irs]
        assert len(set(operations)) == len(operations), (
            f"Duplicate operations in chain: {operations}"
        )
