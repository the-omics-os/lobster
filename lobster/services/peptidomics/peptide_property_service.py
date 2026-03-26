"""
Peptide physicochemical property calculation service.

Stateless service returning 3-tuple (AnnData, Dict, AnalysisStep).
Uses the `peptides` PyPI package for property calculations.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PeptidePropertyService:
    """Calculate physicochemical properties for peptide sequences."""

    # Standard amino acid molecular weights (monoisotopic)
    AA_MW = {
        "A": 71.03711,
        "R": 156.10111,
        "N": 114.04293,
        "D": 115.02694,
        "C": 103.00919,
        "E": 129.04259,
        "Q": 128.05858,
        "G": 57.02146,
        "H": 137.05891,
        "I": 113.08406,
        "L": 113.08406,
        "K": 128.09496,
        "M": 131.04049,
        "F": 147.06841,
        "P": 97.05276,
        "S": 87.03203,
        "T": 101.04768,
        "W": 186.07931,
        "Y": 163.06333,
        "V": 99.06841,
    }

    HYDROPHOBIC_AA = set("AILMFWV")

    def calculate_properties(
        self,
        adata: ad.AnnData,
        sequence_col: str = "sequence",
        properties: Optional[list] = None,
    ) -> Tuple[ad.AnnData, Dict[str, Any], Dict[str, Any]]:
        """
        Batch compute physicochemical properties for all peptide sequences.

        Properties: MW, pI, charge, GRAVY, Boman index, aliphatic index,
        instability index, hydrophobic moment, hydrophobic ratio.

        Args:
            adata: AnnData with sequences in obs[sequence_col]
            sequence_col: Column name containing peptide sequences
            properties: Subset of properties to compute (None = all)

        Returns:
            (AnnData, stats_dict, analysis_step_ir)
        """
        if sequence_col not in adata.obs.columns:
            raise ValueError(
                f"Column '{sequence_col}' not found in obs. "
                f"Available: {list(adata.obs.columns)}"
            )

        sequences = adata.obs[sequence_col].astype(str).tolist()
        n_seqs = len(sequences)

        all_props = properties or [
            "molecular_weight",
            "isoelectric_point",
            "charge",
            "gravy",
            "boman_index",
            "aliphatic_index",
            "instability_index",
            "hydrophobic_moment",
            "hydrophobic_ratio",
            "length",
        ]

        results = {prop: [] for prop in all_props}

        try:
            from peptides import Peptide as PeptideCalc

            use_peptides_lib = True
        except ImportError:
            use_peptides_lib = False
            logger.info("peptides package not available, using built-in calculations")

        for seq in sequences:
            clean_seq = "".join(c for c in seq.upper() if c.isalpha())
            if not clean_seq:
                for prop in all_props:
                    results[prop].append(np.nan)
                continue

            if use_peptides_lib:
                try:
                    p = PeptideCalc(clean_seq)
                    prop_map = {
                        "molecular_weight": lambda: p.molecular_weight(),
                        "isoelectric_point": lambda: p.isoelectric_point(),
                        "charge": lambda: p.charge(pH=7.4),
                        "gravy": lambda: p.hydrophobicity("KyteDoolittle"),
                        "boman_index": lambda: p.boman(),
                        "aliphatic_index": lambda: p.aliphatic_index(),
                        "instability_index": lambda: p.instability_index(),
                        "hydrophobic_moment": lambda: p.hydrophobic_moment(),
                        "hydrophobic_ratio": lambda: sum(
                            1 for aa in clean_seq if aa in self.HYDROPHOBIC_AA
                        )
                        / len(clean_seq),
                        "length": lambda: len(clean_seq),
                    }
                    for prop in all_props:
                        if prop in prop_map:
                            try:
                                results[prop].append(float(prop_map[prop]()))
                            except Exception:
                                results[prop].append(np.nan)
                        else:
                            results[prop].append(np.nan)
                except Exception as e:
                    logger.warning(
                        f"Property calculation failed for {clean_seq[:20]}...: {e}"
                    )
                    for prop in all_props:
                        results[prop].append(np.nan)
            else:
                self._calculate_builtin(clean_seq, all_props, results)

        # Store properties in obs
        adata_out = adata.copy()
        for prop in all_props:
            adata_out.obs[f"peptide_{prop}"] = results[prop]

        # Build property matrix in X if no existing X data
        if adata_out.X is None or (
            hasattr(adata_out.X, "shape") and adata_out.X.shape[1] == 0
        ):
            prop_df = pd.DataFrame(results, index=adata_out.obs_names)
            # Preserve existing metadata (.uns, .obsm, .obsp, .layers)
            prev_uns = adata_out.uns.copy() if adata_out.uns else {}
            prev_obsm = dict(adata_out.obsm) if adata_out.obsm else {}
            adata_out = ad.AnnData(
                X=prop_df.values.astype(np.float32),
                obs=adata_out.obs,
                var=pd.DataFrame(index=prop_df.columns),
                uns=prev_uns,
                obsm=prev_obsm,
            )

        # Statistics
        # Count valid sequences using first available property
        first_prop = all_props[0]
        n_valid = int(sum(1 for v in results[first_prop] if not np.isnan(v)))

        stats = {
            "n_peptides": n_seqs,
            "properties_computed": all_props,
            "n_valid": n_valid,
            "method": "peptides_pypi" if use_peptides_lib else "builtin",
        }
        for prop in all_props:
            valid_vals = [v for v in results[prop] if not np.isnan(v)]
            if valid_vals:
                stats[f"{prop}_mean"] = float(np.mean(valid_vals))
                stats[f"{prop}_std"] = float(np.std(valid_vals))

        from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec

        ir = AnalysisStep(
            operation="peptidomics.calculate_properties",
            tool_name="calculate_peptide_properties",
            description=f"Computed {len(all_props)} physicochemical properties for {n_seqs} peptides",
            library="peptides",
            imports=["from peptides import Peptide"],
            code_template=(
                "from peptides import Peptide\n"
                "props = [Peptide(seq).molecular_weight() for seq in adata.obs['{{ sequence_col }}']]\n"
                "adata.obs['peptide_molecular_weight'] = props"
            ),
            parameters={
                "sequence_col": sequence_col,
                "properties": all_props,
            },
            parameter_schema={
                "sequence_col": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="sequence",
                    required=True,
                    description="Column containing peptide sequences",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata"],
            execution_context={
                "method": "peptides_pypi" if use_peptides_lib else "builtin",
                "n_peptides": n_seqs,
            },
        )

        return adata_out, stats, ir

    def _calculate_builtin(self, seq: str, props: list, results: dict):
        """Fallback property calculation without peptides package."""
        for prop in props:
            if prop == "length":
                results[prop].append(float(len(seq)))
            elif prop == "molecular_weight":
                mw = sum(self.AA_MW.get(aa, 0) for aa in seq) + 18.01524  # water
                results[prop].append(mw)
            elif prop == "hydrophobic_ratio":
                ratio = sum(1 for aa in seq if aa in self.HYDROPHOBIC_AA) / max(
                    len(seq), 1
                )
                results[prop].append(ratio)
            elif prop == "charge":
                pos = sum(1 for aa in seq if aa in "RK") + sum(
                    0.1 for aa in seq if aa == "H"
                )
                neg = sum(1 for aa in seq if aa in "DE")
                results[prop].append(float(pos - neg))
            else:
                results[prop].append(np.nan)
