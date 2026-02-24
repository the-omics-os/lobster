"""
Molecular analysis service for cheminformatics computations.

Provides descriptor calculation, Lipinski Rule of 5 checks, fingerprint similarity,
3D conformation generation, and molecule comparison. RDKit is optional -- all methods
degrade gracefully when it is not installed.

All methods return 3-tuples (None, Dict, AnalysisStep) for provenance tracking and
reproducible notebook export via /pipeline export.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Optional RDKit dependency
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

_RDKIT_MISSING_MSG = (
    "RDKit not installed. Install with: pip install lobster-drug-discovery[chemistry]"
)


class MolecularAnalysisError(Exception):
    """Base exception for molecular analysis operations."""

    pass


class MolecularAnalysisService:
    """
    Stateless cheminformatics service for molecular property computation.

    Provides descriptor calculation, Lipinski compliance checks, fingerprint
    similarity matrices, 3D conformation generation, and side-by-side molecule
    comparison. All heavy computation is performed via RDKit; methods return
    informative error dicts when RDKit is unavailable.
    """

    def __init__(self):
        """Initialize the molecular analysis service (stateless)."""
        logger.debug(
            "Initializing MolecularAnalysisService (RDKit available: %s)",
            RDKIT_AVAILABLE,
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _validate_smiles(self, smiles: str) -> "Chem.Mol":
        """Parse and validate a SMILES string.

        Args:
            smiles: SMILES representation of a molecule.

        Returns:
            RDKit Mol object.

        Raises:
            MolecularAnalysisError: If the SMILES cannot be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise MolecularAnalysisError(
                f"Invalid SMILES string: '{smiles}'. "
                "Ensure the SMILES follows valid chemical notation."
            )
        return mol

    def _rdkit_guard(self) -> Optional[Dict[str, Any]]:
        """Return an error dict when RDKit is not available, else None."""
        if not RDKIT_AVAILABLE:
            return {"error": _RDKIT_MISSING_MSG}
        return None

    # -------------------------------------------------------------------------
    # IR builders
    # -------------------------------------------------------------------------

    def _create_ir_calculate_descriptors(self, smiles: str) -> AnalysisStep:
        return AnalysisStep(
            operation="rdkit.descriptors.calculate",
            tool_name="calculate_descriptors",
            description=(
                "Calculate molecular descriptors (MW, LogP, TPSA, HBD, HBA, "
                "rotatable bonds, stereocenters, aromatic rings) from SMILES"
            ),
            library="rdkit",
            code_template="""from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

mol = Chem.MolFromSmiles({{ smiles | tojson }})
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)
hbd = Lipinski.NumHDonors(mol)
hba = Lipinski.NumHAcceptors(mol)
rotatable = Lipinski.NumRotatableBonds(mol)
stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
print(f"MW={mw:.2f}, LogP={logp:.2f}, TPSA={tpsa:.2f}")""",
            imports=[
                "from rdkit import Chem",
                "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors",
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
            output_entities=["descriptors_dict"],
        )

    def _create_ir_lipinski_check(self, smiles: str) -> AnalysisStep:
        return AnalysisStep(
            operation="rdkit.lipinski.rule_of_five",
            tool_name="lipinski_check",
            description="Check Lipinski Rule of 5 compliance for oral bioavailability",
            library="rdkit",
            code_template="""from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

mol = Chem.MolFromSmiles({{ smiles | tojson }})
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
hbd = Lipinski.NumHDonors(mol)
hba = Lipinski.NumHAcceptors(mol)
violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
print(f"Lipinski: {'PASS' if violations <= 1 else 'FAIL'} ({violations} violations)")""",
            imports=[
                "from rdkit import Chem",
                "from rdkit.Chem import Descriptors, Lipinski",
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
            output_entities=["lipinski_result"],
        )

    def _create_ir_fingerprint_similarity(
        self,
        smiles_list: List[str],
        fingerprint: str,
        radius: int,
        n_bits: int,
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="rdkit.fingerprints.tanimoto_similarity",
            tool_name="fingerprint_similarity",
            description=(
                f"Compute pairwise Tanimoto similarity using {fingerprint} "
                f"fingerprints (radius={radius}, n_bits={n_bits})"
            ),
            library="rdkit",
            code_template="""from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

smiles_list = {{ smiles_list | tojson }}
gen = rdFingerprintGenerator.GetMorganGenerator(radius={{ radius }}, fpSize={{ n_bits }})
fps = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    fps.append(gen.GetFingerprint(mol))

n = len(fps)
sim_matrix = [[0.0] * n for _ in range(n)]
for i in range(n):
    for j in range(n):
        sim_matrix[i][j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
print(f"Computed {n}x{n} similarity matrix")""",
            imports=[
                "from rdkit import Chem, DataStructs",
                "from rdkit.Chem import rdFingerprintGenerator",
            ],
            parameters={
                "smiles_list": smiles_list,
                "fingerprint": fingerprint,
                "radius": radius,
                "n_bits": n_bits,
            },
            parameter_schema={
                "smiles_list": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=[],
                    required=True,
                    description="List of SMILES strings to compare",
                ),
                "fingerprint": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="morgan",
                    required=False,
                    description="Fingerprint type (morgan)",
                ),
                "radius": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=2,
                    required=False,
                    validation_rule="radius >= 1",
                    description="Morgan fingerprint radius",
                ),
                "n_bits": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=2048,
                    required=False,
                    description="Fingerprint bit vector length",
                ),
            },
            input_entities=["smiles_list"],
            output_entities=["similarity_matrix"],
        )

    def _create_ir_prepare_molecule_3d(
        self, smiles: str, n_conformers: int
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="rdkit.allchem.embed_molecule",
            tool_name="prepare_molecule_3d",
            description=(
                "Generate 3D conformation from SMILES using ETKDG embedding "
                "and MMFF94 force field optimization"
            ),
            library="rdkit",
            code_template="""from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles({{ smiles | tojson }})
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol)
mol_block = Chem.MolToMolBlock(mol)
print(f"Generated 3D conformation ({mol.GetNumAtoms()} atoms)")""",
            imports=[
                "from rdkit import Chem",
                "from rdkit.Chem import AllChem",
            ],
            parameters={"smiles": smiles, "n_conformers": n_conformers},
            parameter_schema={
                "smiles": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="SMILES string of the molecule",
                ),
                "n_conformers": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=1,
                    required=False,
                    validation_rule="n_conformers >= 1",
                    description="Number of 3D conformers to generate",
                ),
            },
            input_entities=["smiles"],
            output_entities=["mol_block"],
        )

    def _create_ir_compare_molecules(
        self, smiles_a: str, smiles_b: str
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="rdkit.descriptors.compare",
            tool_name="compare_molecules",
            description="Side-by-side descriptor comparison of two molecules",
            library="rdkit",
            code_template="""from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

for label, smi in [("A", {{ smiles_a | tojson }}), ("B", {{ smiles_b | tojson }})]:
    mol = Chem.MolFromSmiles(smi)
    print(f"Molecule {label}: MW={Descriptors.MolWt(mol):.2f}, LogP={Descriptors.MolLogP(mol):.2f}")""",
            imports=[
                "from rdkit import Chem",
                "from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors",
            ],
            parameters={"smiles_a": smiles_a, "smiles_b": smiles_b},
            parameter_schema={
                "smiles_a": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="SMILES of first molecule",
                ),
                "smiles_b": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="SMILES of second molecule",
                ),
            },
            input_entities=["smiles_a", "smiles_b"],
            output_entities=["comparison_dict"],
        )

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def calculate_descriptors(
        self, smiles: str
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Calculate molecular descriptors from a SMILES string.

        Computes molecular weight, LogP, TPSA, hydrogen bond donors/acceptors,
        rotatable bonds, stereocenters, and aromatic ring count.

        Args:
            smiles: SMILES representation of the molecule.

        Returns:
            Tuple of (None, descriptors dict, AnalysisStep).
        """
        guard = self._rdkit_guard()
        if guard is not None:
            return None, guard, self._create_ir_calculate_descriptors(smiles)

        logger.info("Calculating descriptors for SMILES: %s", smiles[:60])

        try:
            mol = self._validate_smiles(smiles)

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rotatable_bonds = Lipinski.NumRotatableBonds(mol)
            stereocenters = len(
                Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            )
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

            descriptors = {
                "smiles": smiles,
                "molecular_weight": round(mw, 3),
                "logp": round(logp, 3),
                "tpsa": round(tpsa, 2),
                "hbd": hbd,
                "hba": hba,
                "rotatable_bonds": rotatable_bonds,
                "stereocenters": stereocenters,
                "aromatic_rings": aromatic_rings,
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            }

            ir = self._create_ir_calculate_descriptors(smiles)
            logger.info(
                "Descriptors computed: MW=%.2f, LogP=%.2f, TPSA=%.2f",
                mw,
                logp,
                tpsa,
            )
            return None, descriptors, ir

        except MolecularAnalysisError:
            raise
        except Exception as e:
            raise MolecularAnalysisError(
                f"Failed to calculate descriptors for '{smiles}': {e}"
            ) from e

    def lipinski_check(
        self, smiles: str
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Check Lipinski Rule of 5 compliance for oral bioavailability.

        A molecule passes if it violates at most 1 of the 4 rules:
        MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10.

        Args:
            smiles: SMILES representation of the molecule.

        Returns:
            Tuple of (None, dict with pass/fail and individual checks, AnalysisStep).
        """
        from lobster.agents.drug_discovery.config import LIPINSKI_RULES

        guard = self._rdkit_guard()
        if guard is not None:
            return None, guard, self._create_ir_lipinski_check(smiles)

        logger.info("Running Lipinski check for SMILES: %s", smiles[:60])

        try:
            mol = self._validate_smiles(smiles)

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)

            checks = {
                "molecular_weight": {
                    "value": round(mw, 2),
                    "threshold": LIPINSKI_RULES["mw_max"],
                    "pass": mw <= LIPINSKI_RULES["mw_max"],
                },
                "logp": {
                    "value": round(logp, 2),
                    "threshold": LIPINSKI_RULES["logp_max"],
                    "pass": logp <= LIPINSKI_RULES["logp_max"],
                },
                "hbd": {
                    "value": hbd,
                    "threshold": LIPINSKI_RULES["hbd_max"],
                    "pass": hbd <= LIPINSKI_RULES["hbd_max"],
                },
                "hba": {
                    "value": hba,
                    "threshold": LIPINSKI_RULES["hba_max"],
                    "pass": hba <= LIPINSKI_RULES["hba_max"],
                },
            }

            violations = sum(1 for c in checks.values() if not c["pass"])
            # Lipinski allows at most 1 violation
            overall_pass = violations <= 1

            result = {
                "smiles": smiles,
                "overall_pass": overall_pass,
                "violations": violations,
                "checks": checks,
                "classification": "drug-like" if overall_pass else "non-drug-like",
            }

            ir = self._create_ir_lipinski_check(smiles)
            logger.info(
                "Lipinski result: %s (%d violations)",
                "PASS" if overall_pass else "FAIL",
                violations,
            )
            return None, result, ir

        except MolecularAnalysisError:
            raise
        except Exception as e:
            raise MolecularAnalysisError(
                f"Lipinski check failed for '{smiles}': {e}"
            ) from e

    def fingerprint_similarity(
        self,
        smiles_list: List[str],
        fingerprint: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Compute pairwise Tanimoto similarity using Morgan/ECFP fingerprints.

        Args:
            smiles_list: List of SMILES strings to compare.
            fingerprint: Fingerprint type. Currently only 'morgan' is supported.
            radius: Morgan fingerprint radius (default 2 = ECFP4).
            n_bits: Bit vector length for the fingerprint.

        Returns:
            Tuple of (None, dict with similarity_matrix and compound_ids, AnalysisStep).
        """
        guard = self._rdkit_guard()
        if guard is not None:
            return (
                None,
                guard,
                self._create_ir_fingerprint_similarity(
                    smiles_list, fingerprint, radius, n_bits
                ),
            )

        if len(smiles_list) < 2:
            raise MolecularAnalysisError(
                "At least 2 SMILES strings are required for similarity comparison."
            )

        logger.info(
            "Computing %s fingerprint similarity for %d molecules",
            fingerprint,
            len(smiles_list),
        )

        try:
            # Parse molecules
            mols = []
            compound_ids = []
            for idx, smi in enumerate(smiles_list):
                mol = self._validate_smiles(smi)
                mols.append(mol)
                compound_ids.append(f"compound_{idx}")

            # Generate fingerprints
            fps = []
            if fingerprint == "morgan":
                gen = rdFingerprintGenerator.GetMorganGenerator(
                    radius=radius, fpSize=n_bits
                )
            else:
                raise MolecularAnalysisError(
                    f"Unsupported fingerprint type: '{fingerprint}'. "
                    "Supported: 'morgan'."
                )
            for mol in mols:
                fps.append(gen.GetFingerprint(mol))

            # Compute pairwise Tanimoto similarity
            n = len(fps)
            similarity_matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                similarity_matrix[i][i] = 1.0
                for j in range(i + 1, n):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarity_matrix[i][j] = round(sim, 4)
                    similarity_matrix[j][i] = round(sim, 4)

            # Summary statistics
            off_diag = [
                similarity_matrix[i][j]
                for i in range(n)
                for j in range(i + 1, n)
            ]
            mean_sim = sum(off_diag) / len(off_diag) if off_diag else 0.0
            max_sim = max(off_diag) if off_diag else 0.0
            min_sim = min(off_diag) if off_diag else 0.0

            result = {
                "similarity_matrix": similarity_matrix,
                "compound_ids": compound_ids,
                "smiles_list": smiles_list,
                "fingerprint_type": fingerprint,
                "radius": radius,
                "n_bits": n_bits,
                "n_compounds": n,
                "mean_similarity": round(mean_sim, 4),
                "max_similarity": round(max_sim, 4),
                "min_similarity": round(min_sim, 4),
            }

            ir = self._create_ir_fingerprint_similarity(
                smiles_list, fingerprint, radius, n_bits
            )
            logger.info(
                "Similarity matrix: %dx%d, mean=%.3f", n, n, mean_sim
            )
            return None, result, ir

        except MolecularAnalysisError:
            raise
        except Exception as e:
            raise MolecularAnalysisError(
                f"Fingerprint similarity failed: {e}"
            ) from e

    def prepare_molecule_3d(
        self, smiles: str, n_conformers: int = 1
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Generate 3D conformation from SMILES using ETKDG + MMFF94 optimization.

        Args:
            smiles: SMILES representation of the molecule.
            n_conformers: Number of conformers to generate (best is returned).

        Returns:
            Tuple of (None, dict with mol_block string and energy, AnalysisStep).
        """
        guard = self._rdkit_guard()
        if guard is not None:
            return (
                None,
                guard,
                self._create_ir_prepare_molecule_3d(smiles, n_conformers),
            )

        logger.info(
            "Preparing 3D molecule (n_conformers=%d) for SMILES: %s",
            n_conformers,
            smiles[:60],
        )

        try:
            mol = self._validate_smiles(smiles)
            mol = Chem.AddHs(mol)

            # Generate conformer(s)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 1

            if n_conformers == 1:
                embed_result = AllChem.EmbedMolecule(mol, params)
                if embed_result == -1:
                    raise MolecularAnalysisError(
                        f"Failed to embed molecule: '{smiles}'. "
                        "The molecule may be too constrained for 3D embedding."
                    )
                conf_ids = [0]
            else:
                conf_ids = list(
                    AllChem.EmbedMultipleConfs(mol, n_conformers, params)
                )
                if not conf_ids:
                    raise MolecularAnalysisError(
                        f"Failed to generate any conformers for '{smiles}'."
                    )

            # MMFF94 optimization -- pick lowest energy conformer
            best_energy = float("inf")
            best_conf_id = conf_ids[0]

            for cid in conf_ids:
                ff_result = AllChem.MMFFOptimizeMolecule(mol, confId=cid)
                if ff_result == -1:
                    logger.warning(
                        "MMFF optimization did not converge for conformer %d",
                        cid,
                    )

                ff_props = AllChem.MMFFGetMoleculeProperties(mol)
                if ff_props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(
                        mol, ff_props, confId=cid
                    )
                    if ff is not None:
                        energy = ff.CalcEnergy()
                        if energy < best_energy:
                            best_energy = energy
                            best_conf_id = cid

            mol_block = Chem.MolToMolBlock(mol, confId=best_conf_id)

            result = {
                "smiles": smiles,
                "mol_block": mol_block,
                "energy_kcal_mol": (
                    round(best_energy, 3)
                    if best_energy != float("inf")
                    else None
                ),
                "n_conformers_generated": len(conf_ids),
                "best_conformer_id": best_conf_id,
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            }

            ir = self._create_ir_prepare_molecule_3d(smiles, n_conformers)
            logger.info(
                "3D molecule prepared: %d atoms, energy=%.2f kcal/mol",
                mol.GetNumAtoms(),
                best_energy if best_energy != float("inf") else 0.0,
            )
            return None, result, ir

        except MolecularAnalysisError:
            raise
        except Exception as e:
            raise MolecularAnalysisError(
                f"3D preparation failed for '{smiles}': {e}"
            ) from e

    def compare_molecules(
        self, smiles_a: str, smiles_b: str
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Side-by-side comparison of two molecules' properties.

        Computes descriptors, Lipinski compliance, and Tanimoto similarity for
        both molecules and returns a unified comparison table.

        Args:
            smiles_a: SMILES of the first molecule.
            smiles_b: SMILES of the second molecule.

        Returns:
            Tuple of (None, comparison dict, AnalysisStep).
        """
        guard = self._rdkit_guard()
        if guard is not None:
            return (
                None,
                guard,
                self._create_ir_compare_molecules(smiles_a, smiles_b),
            )

        logger.info("Comparing molecules: %s vs %s", smiles_a[:40], smiles_b[:40])

        try:
            mol_a = self._validate_smiles(smiles_a)
            mol_b = self._validate_smiles(smiles_b)

            def _props(mol, smi):
                return {
                    "smiles": smi,
                    "molecular_weight": round(Descriptors.MolWt(mol), 3),
                    "logp": round(Descriptors.MolLogP(mol), 3),
                    "tpsa": round(Descriptors.TPSA(mol), 2),
                    "hbd": Lipinski.NumHDonors(mol),
                    "hba": Lipinski.NumHAcceptors(mol),
                    "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
                    "stereocenters": len(
                        Chem.FindMolChiralCenters(mol, includeUnassigned=True)
                    ),
                    "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                    "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                    "formula": rdMolDescriptors.CalcMolFormula(mol),
                }

            props_a = _props(mol_a, smiles_a)
            props_b = _props(mol_b, smiles_b)

            # Tanimoto similarity between the two
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp_a = gen.GetFingerprint(mol_a)
            fp_b = gen.GetFingerprint(mol_b)
            tanimoto = round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 4)

            # Build comparison table
            properties = [
                "molecular_weight",
                "logp",
                "tpsa",
                "hbd",
                "hba",
                "rotatable_bonds",
                "stereocenters",
                "aromatic_rings",
                "num_heavy_atoms",
            ]
            comparison_table = []
            for prop in properties:
                val_a = props_a[prop]
                val_b = props_b[prop]
                diff = round(val_a - val_b, 3) if isinstance(val_a, (int, float)) else None
                comparison_table.append(
                    {
                        "property": prop,
                        "molecule_a": val_a,
                        "molecule_b": val_b,
                        "difference": diff,
                    }
                )

            result = {
                "molecule_a": props_a,
                "molecule_b": props_b,
                "tanimoto_similarity": tanimoto,
                "comparison_table": comparison_table,
            }

            ir = self._create_ir_compare_molecules(smiles_a, smiles_b)
            logger.info(
                "Comparison complete: Tanimoto=%.3f, MW diff=%.1f",
                tanimoto,
                props_a["molecular_weight"] - props_b["molecular_weight"],
            )
            return None, result, ir

        except MolecularAnalysisError:
            raise
        except Exception as e:
            raise MolecularAnalysisError(
                f"Molecule comparison failed: {e}"
            ) from e
