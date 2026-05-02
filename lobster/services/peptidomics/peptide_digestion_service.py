"""
In silico enzymatic digestion service.

Supports trypsin, chymotrypsin, pepsin, and custom regex patterns.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import pandas as pd

logger = logging.getLogger(__name__)

# Enzyme cleavage rules (regex pattern, position: 'C-term' or 'N-term')
ENZYME_RULES = {
    "trypsin": {
        "pattern": r"(?<=[KR])(?!P)",  # After K/R, not before P
        "description": "Cleaves after K/R (not before P)",
    },
    "chymotrypsin": {
        "pattern": r"(?<=[FWYL])(?!P)",  # After F/W/Y/L, not before P
        "description": "Cleaves after F/W/Y/L (not before P)",
    },
    "pepsin": {
        "pattern": r"(?<=[FL])",  # After F/L
        "description": "Cleaves after F/L",
    },
    "lys_c": {
        "pattern": r"(?<=K)",  # After K
        "description": "Cleaves after K",
    },
    "asp_n": {
        "pattern": r"(?=[D])",  # Before D
        "description": "Cleaves before D",
    },
    "glu_c": {
        "pattern": r"(?<=[DE])",  # After D/E
        "description": "Cleaves after D/E",
    },
}


class PeptideDigestionService:
    """In silico enzymatic digestion of protein/peptide sequences."""

    def digest(
        self,
        adata: ad.AnnData,
        enzyme: str = "trypsin",
        custom_pattern: Optional[str] = None,
        missed_cleavages: int = 0,
        min_length: int = 6,
        max_length: int = 50,
        sequence_col: str = "sequence",
    ) -> Tuple[ad.AnnData, Dict[str, Any], Dict[str, Any]]:
        """
        Perform in silico enzymatic digestion.

        Args:
            adata: AnnData with sequences in obs[sequence_col]
            enzyme: Enzyme name (trypsin, chymotrypsin, pepsin, lys_c, asp_n, glu_c)
            custom_pattern: Custom regex cleavage pattern (overrides enzyme)
            missed_cleavages: Number of allowed missed cleavages (0-3)
            min_length: Minimum fragment length to keep
            max_length: Maximum fragment length to keep
            sequence_col: Column containing parent sequences

        Returns:
            (AnnData with fragments, stats_dict, analysis_step_ir)
        """
        if sequence_col not in adata.obs.columns:
            raise ValueError(
                f"Column '{sequence_col}' not found. Available: {list(adata.obs.columns)}"
            )

        if custom_pattern:
            # Validate regex to prevent ReDoS
            if len(custom_pattern) > 200:
                raise ValueError("Custom regex pattern too long (max 200 chars)")
            try:
                re.compile(custom_pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
            pattern = custom_pattern
            enzyme_name = "custom"
        elif enzyme.lower() in ENZYME_RULES:
            pattern = ENZYME_RULES[enzyme.lower()]["pattern"]
            enzyme_name = enzyme.lower()
        else:
            raise ValueError(
                f"Unknown enzyme '{enzyme}'. Available: {', '.join(ENZYME_RULES.keys())} or use custom_pattern"
            )

        sequences = adata.obs[sequence_col].astype(str).tolist()
        parent_ids = adata.obs_names.tolist()

        all_fragments: List[Dict[str, Any]] = []

        for parent_id, seq in zip(parent_ids, sequences):
            clean_seq = "".join(c for c in seq.upper() if c.isalpha())
            if not clean_seq:
                continue

            # Split at cleavage sites
            base_fragments = re.split(pattern, clean_seq)
            base_fragments = [f for f in base_fragments if f]

            # Compute cumulative start positions for each base fragment
            frag_starts = []
            pos = 0
            for bf in base_fragments:
                frag_starts.append(pos)
                pos += len(bf)

            # Generate fragments with missed cleavages
            fragments = []
            for i in range(len(base_fragments)):
                for mc in range(missed_cleavages + 1):
                    end = i + mc + 1
                    if end > len(base_fragments):
                        break
                    frag = "".join(base_fragments[i:end])
                    if min_length <= len(frag) <= max_length:
                        fragments.append(
                            {
                                "sequence": frag,
                                "length": len(frag),
                                "parent_id": parent_id,
                                "parent_sequence": seq,
                                "start_pos": frag_starts[i] + 1,
                                "missed_cleavages": mc,
                                "enzyme": enzyme_name,
                            }
                        )

            all_fragments.extend(fragments)

        if not all_fragments:
            # Return empty AnnData with proper structure
            empty_obs = pd.DataFrame(
                columns=[
                    "sequence",
                    "length",
                    "parent_id",
                    "parent_sequence",
                    "start_pos",
                    "missed_cleavages",
                    "enzyme",
                ]
            )
            adata_out = ad.AnnData(obs=empty_obs)
            stats = {
                "n_parents": len(sequences),
                "n_fragments": 0,
                "enzyme": enzyme_name,
                "min_length": min_length,
                "max_length": max_length,
            }
            from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec

            ir = AnalysisStep(
                operation="peptidomics.enzymatic_digestion",
                tool_name="simulate_enzymatic_digestion",
                description=f"{enzyme_name} digest: 0 fragments",
                library="re",
                imports=["import re"],
                code_template="# No fragments produced by {{ enzyme }} digest",
                parameters={
                    "enzyme": enzyme_name,
                    "missed_cleavages": missed_cleavages,
                },
                parameter_schema={
                    "enzyme": ParameterSpec(
                        param_type="str",
                        papermill_injectable=True,
                        default_value="trypsin",
                        required=True,
                        description="Enzyme for digestion",
                    ),
                },
                input_entities=["adata"],
                output_entities=["adata"],
            )
            return adata_out, stats, ir

        frag_df = pd.DataFrame(all_fragments)
        frag_df.index = [f"frag_{i}" for i in range(len(frag_df))]

        adata_out = ad.AnnData(obs=frag_df)

        # Deduplicate count
        unique_seqs = frag_df["sequence"].nunique()

        stats = {
            "n_parents": len(sequences),
            "n_fragments": len(all_fragments),
            "n_unique_fragments": unique_seqs,
            "enzyme": enzyme_name,
            "missed_cleavages": missed_cleavages,
            "min_length": min_length,
            "max_length": max_length,
            "mean_fragment_length": float(frag_df["length"].mean()),
            "length_range": f"{int(frag_df['length'].min())}-{int(frag_df['length'].max())} aa",
        }

        from lobster.core.provenance.analysis_ir import AnalysisStep, ParameterSpec

        ir = AnalysisStep(
            operation="peptidomics.enzymatic_digestion",
            tool_name="simulate_enzymatic_digestion",
            description=f"{enzyme_name} digest: {len(all_fragments)} fragments from {len(sequences)} sequences",
            library="re",
            imports=["import re"],
            code_template=(
                "import re\n"
                "fragments = re.split(r'{{ pattern }}', sequence)\n"
                "# Filter by length: {{ min_length }}-{{ max_length }} aa"
            ),
            parameters={
                "enzyme": enzyme_name,
                "missed_cleavages": missed_cleavages,
                "min_length": min_length,
                "max_length": max_length,
            },
            parameter_schema={
                "enzyme": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="trypsin",
                    required=True,
                    description="Enzyme for digestion",
                ),
                "missed_cleavages": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=0,
                    required=False,
                    description="Allowed missed cleavages",
                ),
            },
            input_entities=["adata"],
            output_entities=["adata_fragments"],
            execution_context={
                "n_fragments": len(all_fragments),
                "n_unique": unique_seqs,
            },
        )

        return adata_out, stats, ir
