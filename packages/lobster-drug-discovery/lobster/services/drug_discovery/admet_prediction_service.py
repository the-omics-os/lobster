"""
ADMET prediction service for drug-likeness and pharmacokinetic estimation.

Provides rule-based absorption, distribution, metabolism, excretion, and toxicity
(ADMET) predictions using RDKit molecular descriptors and structural alert
filters. RDKit is optional -- the service degrades gracefully when unavailable.

All methods return 3-tuples (None, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Optional RDKit dependency
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

_RDKIT_MISSING_MSG = (
    "RDKit not installed. Install with: pip install lobster-drug-discovery[chemistry]"
)

# ---------------------------------------------------------------------------
# Structural alert SMARTS definitions
# ---------------------------------------------------------------------------

# PAINS (Pan-Assay Interference Compounds) -- representative subset
# These SMARTS flag frequent hitters in HTS assays.
_PAINS_SMARTS = {
    "quinone": "[#6]1(=[O,N])[#6]=,:[#6][#6](=[O,N])[#6]=,:[#6]1",
    "catechol": "c1(O)c(O)cccc1",
    "hydroxyphenyl_hydrazone": "c1ccc(O)cc1[NX2]=[NX2]",
    "rhodanine": "O=C1CSC(=S)N1",
    "azo_compound": "[N;!R]=[N;!R]",
    "michael_acceptor_1": "[#6]=CC(=O)[!N]",
}

# Brenk structural alerts -- reactive or metabolically labile groups
_BRENK_SMARTS = {
    "aldehyde": "[CH1](=O)",
    "acyl_halide": "[CX3](=[OX1])[FX1,ClX1,BrX1,IX1]",
    "sulfonyl_halide": "[SX4](=[OX1])(=[OX1])[FX1,ClX1,BrX1,IX1]",
    "epoxide": "C1OC1",
    "michael_acceptor": "C=CC(=O)",
    "peroxide": "OO",
    "thiol": "[SH]",
    "nitro": "[NX3+](=O)[O-]",
}

# CYP substrate likelihood -- substructures associated with CYP metabolism
_CYP_SUBSTRATE_SMARTS = {
    "CYP3A4_substrate": "[cR1]1[cR1][cR1][cR1]([N,O])[cR1][cR1]1",
    "CYP2D6_substrate": "c1ccc2c(c1)[nH]cc2",
    "CYP2C9_substrate": "c1ccc(-c2ccccn2)cc1",
}


class ADMETPredictionError(Exception):
    """Base exception for ADMET prediction operations."""

    pass


class ADMETPredictionService:
    """
    Stateless ADMET prediction service using RDKit-based heuristics.

    Applies rule-based models for absorption (TPSA, MW), distribution (LogP,
    plasma protein binding estimate), metabolism (CYP substrate alerts),
    excretion (MW-based renal clearance), and toxicity (PAINS, Brenk filters).

    When the SwissADME or pkCSM APIs are unavailable, this service provides
    a reasonable first-pass filter for lead optimization campaigns.
    """

    def __init__(self):
        """Initialize the ADMET prediction service (stateless)."""
        logger.debug(
            "Initializing ADMETPredictionService (RDKit available: %s)",
            RDKIT_AVAILABLE,
        )
        # Pre-compile SMARTS patterns for fast matching
        self._pains_patterns: Dict[str, Any] = {}
        self._brenk_patterns: Dict[str, Any] = {}
        self._cyp_patterns: Dict[str, Any] = {}

        if RDKIT_AVAILABLE:
            for name, smarts in _PAINS_SMARTS.items():
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None:
                    self._pains_patterns[name] = pat
            for name, smarts in _BRENK_SMARTS.items():
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None:
                    self._brenk_patterns[name] = pat
            for name, smarts in _CYP_SUBSTRATE_SMARTS.items():
                pat = Chem.MolFromSmarts(smarts)
                if pat is not None:
                    self._cyp_patterns[name] = pat

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _rdkit_guard(self) -> Optional[Dict[str, Any]]:
        """Return an error dict when RDKit is not available, else None."""
        if not RDKIT_AVAILABLE:
            return {"error": _RDKIT_MISSING_MSG}
        return None

    def _validate_smiles(self, smiles: str) -> "Chem.Mol":
        """Parse and validate a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ADMETPredictionError(
                f"Invalid SMILES string: '{smiles}'. "
                "Ensure the SMILES follows valid chemical notation."
            )
        return mol

    def _predict_absorption(
        self, mol: "Chem.Mol", tpsa: float, mw: float, logp: float
    ) -> Dict[str, Any]:
        """Predict absorption properties using rule-based heuristics."""
        from lobster.agents.drug_discovery.config import ADMET_THRESHOLDS

        # Oral absorption: TPSA < 140 and MW < 500
        oral_absorption = tpsa < ADMET_THRESHOLDS["tpsa_max_oral"] and mw < 500.0

        # Caco-2 permeability estimate (simplified log scale)
        # High permeability: TPSA < 90, MW < 450
        if tpsa < 60 and mw < 400:
            caco2_class = "high"
        elif tpsa < 90 and mw < 450:
            caco2_class = "moderate"
        else:
            caco2_class = "low"

        # P-glycoprotein substrate heuristic: MW > 400, HBD > 2
        hbd = Lipinski.NumHDonors(mol)
        pgp_substrate_likely = mw > 400 and hbd > 2

        # Water solubility estimate from LogP (Delaney model approximation)
        # log S = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP
        rot_bonds = Lipinski.NumRotatableBonds(mol)
        aromatic_proportion = (
            rdMolDescriptors.CalcNumAromaticRings(mol)
            / max(mol.GetNumHeavyAtoms(), 1)
        )
        log_s = (
            0.16
            - 0.63 * logp
            - 0.0062 * mw
            + 0.066 * rot_bonds
            - 0.74 * aromatic_proportion
        )

        if log_s > -1:
            solubility_class = "highly_soluble"
        elif log_s > -3:
            solubility_class = "soluble"
        elif log_s > -5:
            solubility_class = "moderately_soluble"
        else:
            solubility_class = "poorly_soluble"

        return {
            "oral_absorption_likely": oral_absorption,
            "caco2_permeability_class": caco2_class,
            "pgp_substrate_likely": pgp_substrate_likely,
            "estimated_log_solubility": round(log_s, 2),
            "solubility_class": solubility_class,
            "tpsa": round(tpsa, 2),
        }

    def _predict_distribution(
        self, mol: "Chem.Mol", logp: float, mw: float, tpsa: float
    ) -> Dict[str, Any]:
        """Predict distribution properties."""
        # Volume of distribution estimate (Vd)
        # Simplified: lipophilic compounds have higher Vd
        if logp > 3:
            vd_class = "high"
        elif logp > 1:
            vd_class = "moderate"
        else:
            vd_class = "low"

        # BBB permeability (Egan egg model): TPSA < 79 and LogP in (0, 3)
        bbb_permeable = tpsa < 79 and 0 < logp < 3

        # Plasma protein binding estimate
        # Higher LogP -> higher PPB
        if logp > 4:
            ppb_class = "high"
            ppb_estimate = ">90%"
        elif logp > 2:
            ppb_class = "moderate"
            ppb_estimate = "70-90%"
        else:
            ppb_class = "low"
            ppb_estimate = "<70%"

        return {
            "vd_class": vd_class,
            "bbb_permeable_likely": bbb_permeable,
            "ppb_class": ppb_class,
            "ppb_estimate": ppb_estimate,
        }

    def _predict_metabolism(self, mol: "Chem.Mol") -> Dict[str, Any]:
        """Predict metabolism properties using CYP substrate SMARTS."""
        cyp_substrates = []
        for name, pat in self._cyp_patterns.items():
            if mol.HasSubstructMatch(pat):
                cyp_substrates.append(name)

        # Count metabolically labile groups
        labile_groups = 0
        # Ester hydrolysis
        ester = Chem.MolFromSmarts("[CX3](=O)[OX2H0]")
        if ester is not None and mol.HasSubstructMatch(ester):
            labile_groups += len(mol.GetSubstructMatches(ester))
        # N-demethylation
        n_methyl = Chem.MolFromSmarts("[NX3][CH3]")
        if n_methyl is not None and mol.HasSubstructMatch(n_methyl):
            labile_groups += len(mol.GetSubstructMatches(n_methyl))
        # O-demethylation
        o_methyl = Chem.MolFromSmarts("[OX2][CH3]")
        if o_methyl is not None and mol.HasSubstructMatch(o_methyl):
            labile_groups += len(mol.GetSubstructMatches(o_methyl))

        if labile_groups > 3:
            metabolic_stability = "low"
        elif labile_groups > 1:
            metabolic_stability = "moderate"
        else:
            metabolic_stability = "high"

        return {
            "cyp_substrate_alerts": cyp_substrates,
            "n_cyp_alerts": len(cyp_substrates),
            "metabolically_labile_groups": labile_groups,
            "metabolic_stability_estimate": metabolic_stability,
        }

    def _predict_excretion(self, mw: float, logp: float) -> Dict[str, Any]:
        """Predict excretion properties based on MW and LogP."""
        # Renal clearance: smaller, hydrophilic molecules cleared renally
        if mw < 350 and logp < 1.5:
            renal_clearance = "high"
            primary_route = "renal"
        elif mw < 500 and logp < 3:
            renal_clearance = "moderate"
            primary_route = "mixed"
        else:
            renal_clearance = "low"
            primary_route = "hepatic"

        # Half-life estimate class (rough heuristic)
        if logp > 4 and mw > 500:
            halflife_class = "long"
        elif logp > 2:
            halflife_class = "moderate"
        else:
            halflife_class = "short"

        return {
            "renal_clearance_class": renal_clearance,
            "primary_excretion_route": primary_route,
            "halflife_class": halflife_class,
        }

    def _predict_toxicity(self, mol: "Chem.Mol") -> Dict[str, Any]:
        """Predict toxicity flags using PAINS and Brenk structural alerts."""
        pains_hits = []
        for name, pat in self._pains_patterns.items():
            if mol.HasSubstructMatch(pat):
                pains_hits.append(name)

        brenk_hits = []
        for name, pat in self._brenk_patterns.items():
            if mol.HasSubstructMatch(pat):
                brenk_hits.append(name)

        # Overall toxicity risk classification
        n_alerts = len(pains_hits) + len(brenk_hits)
        if n_alerts == 0:
            toxicity_risk = "low"
        elif n_alerts <= 2:
            toxicity_risk = "moderate"
        else:
            toxicity_risk = "high"

        return {
            "pains_alerts": pains_hits,
            "n_pains_alerts": len(pains_hits),
            "brenk_alerts": brenk_hits,
            "n_brenk_alerts": len(brenk_hits),
            "total_structural_alerts": n_alerts,
            "toxicity_risk_class": toxicity_risk,
        }

    # -------------------------------------------------------------------------
    # IR builder
    # -------------------------------------------------------------------------

    def _create_ir_predict_admet(self, smiles: str) -> AnalysisStep:
        return AnalysisStep(
            operation="lobster.services.drug_discovery.admet.predict",
            tool_name="predict_admet",
            description=(
                "Predict ADMET properties using RDKit-based heuristics: "
                "absorption (TPSA, Caco-2), distribution (BBB, PPB), "
                "metabolism (CYP alerts), excretion (renal clearance), "
                "toxicity (PAINS, Brenk filters)"
            ),
            library="rdkit",
            code_template="""from lobster.services.drug_discovery.admet_prediction_service import ADMETPredictionService

service = ADMETPredictionService()
_, admet, _ = service.predict_admet({{ smiles | tojson }})
for category, props in admet['predictions'].items():
    print(f"\\n{category.upper()}:")
    for k, v in props.items():
        print(f"  {k}: {v}")""",
            imports=[
                "from lobster.services.drug_discovery.admet_prediction_service import ADMETPredictionService",
            ],
            parameters={"smiles": smiles},
            parameter_schema={
                "smiles": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="SMILES string of the molecule",
                ),
            },
            input_entities=["smiles"],
            output_entities=["admet_predictions"],
        )

    # -------------------------------------------------------------------------
    # Public method
    # -------------------------------------------------------------------------

    def predict_admet(
        self, smiles: str
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Predict ADMET properties using RDKit-based heuristic models.

        Evaluates five pharmacokinetic categories:
        - **Absorption**: Oral bioavailability (TPSA, MW), Caco-2 permeability
          class, P-gp substrate likelihood, aqueous solubility (Delaney model).
        - **Distribution**: Volume of distribution class, BBB permeability
          (Egan model), plasma protein binding estimate.
        - **Metabolism**: CYP substrate alerts (3A4, 2D6, 2C9), metabolically
          labile group count, metabolic stability estimate.
        - **Excretion**: Renal clearance class, primary excretion route,
          half-life class estimate.
        - **Toxicity**: PAINS filter hits, Brenk structural alerts, overall
          toxicity risk classification.

        Args:
            smiles: SMILES representation of the molecule.

        Returns:
            Tuple of (None, ADMET predictions dict, AnalysisStep).
        """
        guard = self._rdkit_guard()
        if guard is not None:
            return None, guard, self._create_ir_predict_admet(smiles)

        logger.info("Predicting ADMET for SMILES: %s", smiles[:60])

        try:
            mol = self._validate_smiles(smiles)

            # Core descriptors shared across categories
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            absorption = self._predict_absorption(mol, tpsa, mw, logp)
            distribution = self._predict_distribution(mol, logp, mw, tpsa)
            metabolism = self._predict_metabolism(mol)
            excretion = self._predict_excretion(mw, logp)
            toxicity = self._predict_toxicity(mol)

            # Overall drugability summary
            overall_flags = []
            if not absorption["oral_absorption_likely"]:
                overall_flags.append("poor_oral_absorption")
            if toxicity["total_structural_alerts"] > 0:
                overall_flags.append("structural_alerts_present")
            if metabolism["metabolic_stability_estimate"] == "low":
                overall_flags.append("low_metabolic_stability")
            if not distribution["bbb_permeable_likely"]:
                overall_flags.append("limited_bbb_permeability")

            result = {
                "smiles": smiles,
                "molecular_weight": round(mw, 2),
                "logp": round(logp, 2),
                "tpsa": round(tpsa, 2),
                "predictions": {
                    "absorption": absorption,
                    "distribution": distribution,
                    "metabolism": metabolism,
                    "excretion": excretion,
                    "toxicity": toxicity,
                },
                "overall_flags": overall_flags,
                "n_flags": len(overall_flags),
                "overall_assessment": (
                    "favorable"
                    if len(overall_flags) == 0
                    else "acceptable" if len(overall_flags) <= 2
                    else "unfavorable"
                ),
            }

            ir = self._create_ir_predict_admet(smiles)
            logger.info(
                "ADMET prediction complete: %s (%d flags)",
                result["overall_assessment"],
                len(overall_flags),
            )
            return None, result, ir

        except ADMETPredictionError:
            raise
        except Exception as e:
            raise ADMETPredictionError(
                f"ADMET prediction failed for '{smiles}': {e}"
            ) from e
