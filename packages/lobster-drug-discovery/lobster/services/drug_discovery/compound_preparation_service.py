"""
Compound preparation service for ligand and binding site processing.

Provides SMILES-to-3D conversion, CAS-to-SMILES lookup via PubChem, and
binding site identification from PDB content. Adapted from G-ReInCATALiZE
thesis components for enzyme engineering docking workflows.

RDKit is optional -- methods that require it degrade gracefully. PubChem
lookups use httpx for synchronous HTTP requests.

All methods return 3-tuples (None, Dict, AnalysisStep) for provenance tracking
and reproducible notebook export via /pipeline export.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Optional RDKit dependency
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

_RDKIT_MISSING_MSG = (
    "RDKit not installed. Install with: pip install lobster-drug-discovery[chemistry]"
)


class CompoundPreparationError(Exception):
    """Base exception for compound preparation operations."""

    pass


class CompoundPreparationService:
    """
    Stateless compound preparation service for docking and structure analysis.

    Provides three core capabilities:
    - Ligand preparation: SMILES to 3D with hydrogen addition and MMFF
      optimization (requires RDKit).
    - CAS-to-SMILES conversion: Batch lookup via PubChem REST API.
    - Binding site identification: Parse PDB content and find residues near
      a center coordinate (pure Python, no external dependencies).
    """

    def __init__(self):
        """Initialize the compound preparation service (stateless)."""
        logger.debug(
            "Initializing CompoundPreparationService (RDKit available: %s)",
            RDKIT_AVAILABLE,
        )

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
            raise CompoundPreparationError(
                f"Invalid SMILES string: '{smiles}'. "
                "Ensure the SMILES follows valid chemical notation."
            )
        return mol

    def _parse_pdb_atoms(
        self, pdb_content: str
    ) -> List[Dict[str, Any]]:
        """Parse ATOM/HETATM records from PDB content.

        Extracts atom coordinates, residue names, chain IDs, and residue
        sequence numbers following the PDB fixed-column format.

        Args:
            pdb_content: Raw PDB file content as a string.

        Returns:
            List of atom dicts with keys: record_type, atom_name, res_name,
            chain_id, res_seq, x, y, z.
        """
        atoms = []
        for line in pdb_content.splitlines():
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue

            try:
                atom = {
                    "record_type": record,
                    "atom_name": line[12:16].strip(),
                    "res_name": line[17:20].strip(),
                    "chain_id": line[21:22].strip(),
                    "res_seq": int(line[22:26].strip()),
                    "x": float(line[30:38].strip()),
                    "y": float(line[38:46].strip()),
                    "z": float(line[46:54].strip()),
                }
                atoms.append(atom)
            except (ValueError, IndexError):
                # Skip malformed lines
                continue

        return atoms

    def _distance_3d(
        self,
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
    ) -> float:
        """Euclidean distance between two 3D points."""
        return math.sqrt(
            (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
        )

    # -------------------------------------------------------------------------
    # IR builders
    # -------------------------------------------------------------------------

    def _create_ir_prepare_ligand(
        self, smiles: str, output_format: str
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="rdkit.allchem.prepare_ligand",
            tool_name="prepare_ligand",
            description=(
                "Prepare ligand: SMILES to 3D structure with hydrogen "
                "addition, ETKDG embedding, and MMFF94 force field optimization"
            ),
            library="rdkit",
            code_template="""from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles({{ smiles | tojson }})
mol = Chem.AddHs(mol)
params = AllChem.ETKDGv3()
params.randomSeed = 42
AllChem.EmbedMolecule(mol, params)
AllChem.MMFFOptimizeMolecule(mol)
mol_block = Chem.MolToMolBlock(mol)
print(f"Ligand prepared: {mol.GetNumAtoms()} atoms")""",
            imports=[
                "from rdkit import Chem",
                "from rdkit.Chem import AllChem",
            ],
            parameters={"smiles": smiles, "output_format": output_format},
            parameter_schema={
                "smiles": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="SMILES string of the ligand molecule",
                ),
                "output_format": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="mol",
                    required=False,
                    description="Output format: 'mol' (MDL Molfile) or 'pdb'",
                ),
            },
            input_entities=["smiles"],
            output_entities=["mol_block"],
        )

    def _create_ir_cas_to_smiles(
        self, cas_numbers: List[str]
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="pubchem.compound.cas_to_smiles",
            tool_name="cas_to_smiles",
            description=(
                "Convert CAS registry numbers to canonical SMILES via "
                "PubChem REST API (PUG REST)"
            ),
            library="httpx",
            code_template="""import httpx

cas_numbers = {{ cas_numbers | tojson }}
results = {}
for cas in cas_numbers:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/CanonicalSMILES/JSON"
    resp = httpx.get(url, timeout=30.0)
    if resp.status_code == 200:
        data = resp.json()
        smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        results[cas] = smiles
    else:
        results[cas] = None
print(f"Resolved {sum(1 for v in results.values() if v)} of {len(cas_numbers)} CAS numbers")""",
            imports=["import httpx"],
            parameters={"cas_numbers": cas_numbers},
            parameter_schema={
                "cas_numbers": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=[],
                    required=True,
                    description="List of CAS registry numbers to resolve",
                ),
            },
            input_entities=["cas_numbers"],
            output_entities=["cas_to_smiles_mapping"],
        )

    def _create_ir_identify_binding_site(
        self,
        center_coords: Optional[List[float]],
        center_mode: str,
        radius: float,
    ) -> AnalysisStep:
        return AnalysisStep(
            operation="lobster.services.drug_discovery.compound_prep.binding_site",
            tool_name="identify_binding_site",
            description=(
                f"Identify binding site residues within {radius}A of center "
                f"(mode: {center_mode}) from PDB content"
            ),
            library="lobster",
            code_template="""from lobster.services.drug_discovery.compound_preparation_service import CompoundPreparationService

service = CompoundPreparationService()
_, result, _ = service.identify_binding_site(
    pdb_content=pdb_content,
    center_coords={{ center_coords | tojson }},
    center_mode={{ center_mode | tojson }},
    radius={{ radius }},
)
print(f"Found {result['n_residues']} residues within {result['radius']}A of center")
for res in result['residues'][:10]:
    print(f"  {res['chain_id']}:{res['res_name']}{res['res_seq']} (dist={res['distance_to_center']:.1f}A)")""",
            imports=[
                "from lobster.services.drug_discovery.compound_preparation_service import CompoundPreparationService",
            ],
            parameters={
                "center_coords": center_coords,
                "center_mode": center_mode,
                "radius": radius,
            },
            parameter_schema={
                "center_coords": ParameterSpec(
                    param_type="Optional[List[float]]",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="[x, y, z] center coordinates; computed if None",
                ),
                "center_mode": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="geometric",
                    required=False,
                    description="Center calculation mode: 'geometric' (mean of all atoms) or 'ligand' (mean of HETATM)",
                ),
                "radius": ParameterSpec(
                    param_type="float",
                    papermill_injectable=True,
                    default_value=8.0,
                    required=False,
                    validation_rule="radius > 0",
                    description="Search radius in Angstroms around the center",
                ),
            },
            input_entities=["pdb_content"],
            output_entities=["binding_site_result"],
        )

    # -------------------------------------------------------------------------
    # Public methods
    # -------------------------------------------------------------------------

    def prepare_ligand(
        self, smiles: str, output_format: str = "mol"
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Prepare a ligand from SMILES to a 3D structure.

        Workflow:
        1. Parse SMILES to RDKit Mol object.
        2. Add explicit hydrogens (Chem.AddHs).
        3. Generate 3D conformation via ETKDG v3 embedding.
        4. Optimize geometry with MMFF94 force field.
        5. Output as MDL Molfile or PDB block.

        Args:
            smiles: SMILES string of the ligand molecule.
            output_format: Output format -- 'mol' for MDL Molfile V2000 block
                or 'pdb' for PDB format block.

        Returns:
            Tuple of (None, dict with mol_block/pdb_block + energy, AnalysisStep).
        """
        guard = self._rdkit_guard()
        if guard is not None:
            return None, guard, self._create_ir_prepare_ligand(smiles, output_format)

        if output_format not in ("mol", "pdb"):
            raise CompoundPreparationError(
                f"Unsupported output format: '{output_format}'. Use 'mol' or 'pdb'."
            )

        logger.info(
            "Preparing ligand (format=%s) for SMILES: %s",
            output_format,
            smiles[:60],
        )

        try:
            mol = self._validate_smiles(smiles)
            mol = Chem.AddHs(mol)

            # 3D embedding
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.numThreads = 1

            embed_result = AllChem.EmbedMolecule(mol, params)
            if embed_result == -1:
                raise CompoundPreparationError(
                    f"Failed to embed molecule '{smiles}' into 3D. "
                    "The molecule may be too constrained or contain unsupported features."
                )

            # MMFF94 optimization
            converged = AllChem.MMFFOptimizeMolecule(mol)
            if converged == -1:
                logger.warning(
                    "MMFF optimization setup failed for '%s'; returning unoptimized structure",
                    smiles[:40],
                )

            # Calculate energy
            energy = None
            ff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if ff_props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, ff_props)
                if ff is not None:
                    energy = round(ff.CalcEnergy(), 3)

            # Generate output block
            if output_format == "mol":
                structure_block = Chem.MolToMolBlock(mol)
            else:
                structure_block = Chem.MolToPDBBlock(mol)

            result = {
                "smiles": smiles,
                "output_format": output_format,
                "structure_block": structure_block,
                "energy_kcal_mol": energy,
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "optimization_converged": converged != -1,
            }

            ir = self._create_ir_prepare_ligand(smiles, output_format)
            logger.info(
                "Ligand prepared: %d atoms, energy=%s kcal/mol",
                mol.GetNumAtoms(),
                energy if energy is not None else "N/A",
            )
            return None, result, ir

        except CompoundPreparationError:
            raise
        except Exception as e:
            raise CompoundPreparationError(
                f"Ligand preparation failed for '{smiles}': {e}"
            ) from e

    def cas_to_smiles(
        self, cas_numbers: List[str]
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Convert CAS registry numbers to canonical SMILES via PubChem REST API.

        Uses the PubChem PUG REST service to look up compounds by CAS number
        and retrieve their canonical SMILES representations. Failed lookups
        are reported with error messages rather than raising exceptions.

        Args:
            cas_numbers: List of CAS registry number strings (e.g. ['50-78-2', '64-17-5']).

        Returns:
            Tuple of (None, dict with cas_to_smiles mapping + stats, AnalysisStep).

        Raises:
            CompoundPreparationError: If the HTTP client fails entirely.
        """
        if not cas_numbers:
            raise CompoundPreparationError(
                "Empty CAS number list. Provide at least one CAS number."
            )

        # Validate CAS format (digits-digits-digit)
        cas_pattern = re.compile(r"^\d{2,7}-\d{2}-\d$")
        for cas in cas_numbers:
            if not cas_pattern.match(cas):
                logger.warning(
                    "CAS number '%s' does not match standard format (XXXXXXX-XX-X); "
                    "attempting lookup anyway",
                    cas,
                )

        logger.info("Converting %d CAS numbers to SMILES via PubChem", len(cas_numbers))

        try:
            import httpx
        except ImportError:
            raise CompoundPreparationError(
                "httpx not installed. Install with: pip install httpx"
            )

        try:
            from lobster.agents.drug_discovery.config import (
                DEFAULT_HTTP_TIMEOUT,
                PUBCHEM_API_BASE,
            )

            mapping = {}
            errors = {}

            with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                for cas in cas_numbers:
                    url = (
                        f"{PUBCHEM_API_BASE}/compound/name/{cas}"
                        f"/property/CanonicalSMILES,MolecularFormula,"
                        f"MolecularWeight,IUPACName/JSON"
                    )

                    try:
                        response = client.get(url)

                        if response.status_code == 200:
                            data = response.json()
                            props = data["PropertyTable"]["Properties"][0]
                            mapping[cas] = {
                                "smiles": props.get("CanonicalSMILES"),
                                "formula": props.get("MolecularFormula"),
                                "molecular_weight": props.get("MolecularWeight"),
                                "iupac_name": props.get("IUPACName"),
                            }
                            logger.debug("Resolved CAS %s -> %s", cas, props.get("CanonicalSMILES"))
                        elif response.status_code == 404:
                            errors[cas] = "Not found in PubChem"
                            logger.warning("CAS %s not found in PubChem", cas)
                        else:
                            errors[cas] = f"HTTP {response.status_code}: {response.text[:200]}"
                            logger.warning(
                                "PubChem lookup for CAS %s failed: HTTP %d",
                                cas,
                                response.status_code,
                            )
                    except httpx.TimeoutException:
                        errors[cas] = "Request timed out"
                        logger.warning("PubChem lookup for CAS %s timed out", cas)
                    except httpx.HTTPError as exc:
                        errors[cas] = f"HTTP error: {exc}"
                        logger.warning("PubChem lookup for CAS %s HTTP error: %s", cas, exc)

            # Build simple smiles-only mapping for convenience
            cas_to_smiles_simple = {
                cas: info["smiles"]
                for cas, info in mapping.items()
                if info.get("smiles")
            }

            result = {
                "cas_to_smiles": cas_to_smiles_simple,
                "detailed_results": mapping,
                "errors": errors,
                "n_requested": len(cas_numbers),
                "n_resolved": len(cas_to_smiles_simple),
                "n_failed": len(errors),
                "success_rate": (
                    round(len(cas_to_smiles_simple) / len(cas_numbers), 3)
                    if cas_numbers
                    else 0.0
                ),
            }

            ir = self._create_ir_cas_to_smiles(cas_numbers)
            logger.info(
                "CAS lookup complete: %d/%d resolved",
                len(cas_to_smiles_simple),
                len(cas_numbers),
            )
            return None, result, ir

        except CompoundPreparationError:
            raise
        except Exception as e:
            raise CompoundPreparationError(
                f"CAS-to-SMILES conversion failed: {e}"
            ) from e

    def identify_binding_site(
        self,
        pdb_content: str,
        center_coords: Optional[List[float]] = None,
        center_mode: str = "geometric",
        radius: float = 8.0,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Identify binding site residues from PDB content.

        Parses PDB ATOM/HETATM records, determines a center point, then
        finds all residues with at least one alpha carbon (CA) within the
        specified radius of the center.

        Center calculation modes:
        - **geometric**: Mean coordinate of all ATOM alpha carbons.
        - **ligand**: Mean coordinate of all HETATM atoms (useful when a
          co-crystallized ligand defines the binding pocket).

        If center_coords is provided explicitly, center_mode is ignored.

        Args:
            pdb_content: Raw PDB file content as a string.
            center_coords: Optional [x, y, z] center coordinates. If None,
                the center is computed from the PDB content.
            center_mode: How to compute the center when center_coords is None.
                'geometric' = mean of all CA atoms; 'ligand' = mean of HETATM atoms.
            radius: Search radius in Angstroms around the center.

        Returns:
            Tuple of (None, dict with residues list + center + radius, AnalysisStep).

        Raises:
            CompoundPreparationError: If PDB content cannot be parsed or no
            atoms are found for the selected center mode.
        """
        if not pdb_content or not pdb_content.strip():
            raise CompoundPreparationError(
                "Empty PDB content. Provide valid PDB file content."
            )

        if radius <= 0:
            raise CompoundPreparationError(
                f"Radius must be positive, got {radius}"
            )

        if center_mode not in ("geometric", "ligand"):
            raise CompoundPreparationError(
                f"Unknown center_mode: '{center_mode}'. Use 'geometric' or 'ligand'."
            )

        logger.info(
            "Identifying binding site (radius=%.1fA, mode=%s)",
            radius,
            center_mode if center_coords is None else "explicit",
        )

        try:
            atoms = self._parse_pdb_atoms(pdb_content)
            if not atoms:
                raise CompoundPreparationError(
                    "No ATOM/HETATM records found in PDB content."
                )

            # Determine center
            if center_coords is not None:
                if len(center_coords) != 3:
                    raise CompoundPreparationError(
                        f"center_coords must have 3 elements [x, y, z], "
                        f"got {len(center_coords)}"
                    )
                cx, cy, cz = center_coords
            elif center_mode == "geometric":
                # Mean of all alpha carbons
                ca_atoms = [
                    a for a in atoms
                    if a["record_type"] == "ATOM" and a["atom_name"] == "CA"
                ]
                if not ca_atoms:
                    # Fallback to all ATOM records
                    ca_atoms = [a for a in atoms if a["record_type"] == "ATOM"]
                if not ca_atoms:
                    raise CompoundPreparationError(
                        "No ATOM records found for geometric center calculation."
                    )
                cx = sum(a["x"] for a in ca_atoms) / len(ca_atoms)
                cy = sum(a["y"] for a in ca_atoms) / len(ca_atoms)
                cz = sum(a["z"] for a in ca_atoms) / len(ca_atoms)
            else:  # ligand
                hetatm_atoms = [
                    a for a in atoms
                    if a["record_type"] == "HETATM"
                    and a["res_name"] not in ("HOH", "WAT", "DOD")
                ]
                if not hetatm_atoms:
                    raise CompoundPreparationError(
                        "No HETATM records (excluding water) found for "
                        "ligand center calculation. Use 'geometric' mode or "
                        "provide explicit center_coords."
                    )
                cx = sum(a["x"] for a in hetatm_atoms) / len(hetatm_atoms)
                cy = sum(a["y"] for a in hetatm_atoms) / len(hetatm_atoms)
                cz = sum(a["z"] for a in hetatm_atoms) / len(hetatm_atoms)

            # Find residues within radius (using CA atoms for distance)
            ca_atoms = [
                a for a in atoms
                if a["record_type"] == "ATOM" and a["atom_name"] == "CA"
            ]

            # Track unique residues (chain_id + res_seq + res_name)
            residue_set = set()
            residues = []

            for ca in ca_atoms:
                dist = self._distance_3d(cx, cy, cz, ca["x"], ca["y"], ca["z"])
                if dist <= radius:
                    key = (ca["chain_id"], ca["res_seq"], ca["res_name"])
                    if key not in residue_set:
                        residue_set.add(key)
                        residues.append(
                            {
                                "chain_id": ca["chain_id"],
                                "res_seq": ca["res_seq"],
                                "res_name": ca["res_name"],
                                "ca_x": round(ca["x"], 3),
                                "ca_y": round(ca["y"], 3),
                                "ca_z": round(ca["z"], 3),
                                "distance_to_center": round(dist, 2),
                            }
                        )

            # Sort by distance
            residues.sort(key=lambda r: r["distance_to_center"])

            # Summary by residue type
            res_type_counts = {}
            for r in residues:
                rn = r["res_name"]
                res_type_counts[rn] = res_type_counts.get(rn, 0) + 1

            # Chain distribution
            chain_counts = {}
            for r in residues:
                ch = r["chain_id"] or "?"
                chain_counts[ch] = chain_counts.get(ch, 0) + 1

            result = {
                "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
                "center_mode": center_mode if center_coords is None else "explicit",
                "radius": radius,
                "n_residues": len(residues),
                "residues": residues,
                "residue_type_counts": res_type_counts,
                "chain_distribution": chain_counts,
                "n_total_atoms_parsed": len(atoms),
            }

            ir = self._create_ir_identify_binding_site(
                center_coords, center_mode, radius
            )
            logger.info(
                "Binding site identified: %d residues within %.1fA of center [%.1f, %.1f, %.1f]",
                len(residues),
                radius,
                cx,
                cy,
                cz,
            )
            return None, result, ir

        except CompoundPreparationError:
            raise
        except Exception as e:
            raise CompoundPreparationError(
                f"Binding site identification failed: {e}"
            ) from e
