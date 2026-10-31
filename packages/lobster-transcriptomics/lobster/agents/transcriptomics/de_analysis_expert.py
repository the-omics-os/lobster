"""
Differential Expression Analysis Sub-Agent for specialized DE workflows.

This sub-agent handles all differential expression analysis tools including:
- Pseudobulk aggregation from single-cell data
- Formula-based experimental design
- pyDESeq2 differential expression analysis
- Iterative DE analysis with comparison
- Pathway enrichment analysis

CRITICAL SCIENTIFIC FIXES:
1. DESeq2 requires RAW INTEGER COUNTS from adata.raw.X (not normalized adata.X)
2. Minimum replicate threshold changed from 2 to 3 for stable variance estimation
3. Warning when any condition has fewer than 4 replicates (low statistical power)
"""

# Agent configuration for entry point discovery (must be at top, before heavy imports)
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="de_analysis_expert",
    display_name="DE Analysis Expert",
    description="Differential expression sub-agent: pseudobulk, pyDESeq2, formula-based DE, pathway enrichment",
    factory_function="lobster.agents.transcriptomics.de_analysis_expert.de_analysis_expert",
    handoff_tool_name=None,  # Not directly accessible
    handoff_tool_description=None,
    supervisor_accessible=False,  # Only via transcriptomics_expert
)

# === Heavy imports below ===
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.agents.transcriptomics.prompts import create_de_analysis_expert_prompt
from lobster.agents.transcriptomics.state import DEAnalysisExpertState
from lobster.config.inference_context import (
    BoundModelFactory,
    InferenceContext,
    validate_inference_binding,
)
from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core import (
    AggregationError,
    DesignMatrixError,
    FormulaError,
    InsufficientCellsError,
    PseudobulkError,
)
from lobster.core.export_scope import (
    WORKSPACE_EXPORT_MESSAGE,
    InvalidModalityNameError,
    WorkspaceExportError,
    assert_inside_workspace,
    check_modality_name,
    outside_workspace,
    plain_name_part,
)
from lobster.core.figures.errors import ManagedPathError
from lobster.core.figures.export_dirs import assert_dirs_allowed
from lobster.core.figures.export_guard import (
    MANAGED_EXPORT_MESSAGE,
    assert_export_allowed,
    export_refused,
)
from lobster.core.provenance.analysis_ir import AnalysisStep
from lobster.core.runtime.data_manager import DataManagerV2
from lobster.services.analysis.bulk_rnaseq_service import (
    BulkRNASeqError,
    BulkRNASeqService,
)
from lobster.services.analysis.differential_formula_service import (
    DifferentialFormulaService,
)
from lobster.services.analysis.pseudobulk_service import PseudobulkService
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


# -------------------------
# PROVENANCE IR BUILDERS
# -------------------------
# The formula-based DE tools keep their logic in BulkRNASeqService /
# DifferentialFormulaService. Each IR below replays the same service calls on a
# notebook's ``adata`` so an exported notebook reproduces the tool's result.

_BULK_SERVICE_SETUP = (
    "from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService\n"
    "\n"
    "bulk_service = BulkRNASeqService(results_dir='bulk_results')\n"
)


def _de_result_name(modality_name: str, group1, group2) -> str:
    """The DE result modality name: group labels are data values, so each is encoded
    as part of one plain name (two labels never collide); refused (fixed reason)
    before any analysis if the name is still not plain, for example too long."""
    return check_modality_name(
        f"{modality_name}_de_{plain_name_part(group1)}_vs_{plain_name_part(group2)}"
    )


def _is_pseudobulk(adata) -> bool:
    """True when the modality carries a pseudobulk or formula design (``uns`` only)."""
    return (
        "pseudobulk_design" in adata.uns
        or "is_pseudobulk" in adata.uns
        or "formula_design" in adata.uns
    )


def _empty_bulk_de_refusal(adata_de, de_stats):
    """An empty filtered table is not a successful biological DE result."""
    if adata_de.n_vars:
        return None
    filters = de_stats["filtering_params"]
    before = de_stats["pre_filter_gene_count"]
    after = de_stats["post_filter_gene_count"]
    return (
        "**DE refused: no genes remain after filtering.**\n\n"
        f"Genes after marker-style post-hoc filters: {before} -> {after}. "
        f"All {before} genes were removed by the combined filters: "
        f"min_fold_change={filters['min_fold_change']}, "
        f"min_pct_expressed={filters['min_pct_expressed']}, "
        f"max_out_pct_expressed={filters['max_out_pct_expressed']}.\n"
        "These require sufficient fold change and expression in group1, and cap "
        "the fraction of group2 samples with expression > 1. The out-group cap "
        "is a marker-style filter and can remove genes expressed across bulk samples.\n\n"
        "This is not a zero-DE biological result. Use run_de_with_formula for "
        "bulk DE without these marker-style post-hoc filters. No result modality was stored."
    )


def _plain(value: Any) -> Any:
    """Convert numpy scalars/containers into plain Python for IR literals."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


# Column names that identify an independent biological unit. Matched after
# lower-casing, mapping spaces/dashes to "_" and removing an "_id"/"id"/"_name" suffix.
_BIOLOGICAL_UNIT_NAMES = (
    "donor",
    "patient",
    "subject",
    "individual",
    "participant",
    "animal",
    "mouse",
)


def _unit_name_root(column: Any) -> str:
    name = str(column).strip().lower().replace(" ", "_").replace("-", "_")
    for suffix in ("_id", "_name", "id"):
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)].rstrip("_")
            break
    return name


def _find_biological_unit_column(metadata: pd.DataFrame, groupby: str) -> Optional[str]:
    """Return the column naming the biological unit (donor, patient, ...), if any.

    If several candidates exist, the one with the fewest distinct values is used:
    counting fewer units can only make the replicate check stricter.
    """
    candidates = [
        col
        for col in metadata.columns
        if col != groupby and _unit_name_root(col) in _BIOLOGICAL_UNIT_NAMES
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda col: metadata[col].nunique(dropna=True))


def _count_biological_replicates(
    metadata: pd.DataFrame,
    groupby: str,
    min_replicates: int = 3,
) -> Dict[str, Any]:
    """Count replicates per group in independent biological units, not rows.

    Rows are samples or sequencing libraries. Several libraries from one donor are
    technical replicates: counting them as biological replicates inflates n and
    the confidence of every downstream test (pseudoreplication). When a donor-like
    column exists, n is the number of distinct units per group.
    """
    unit_col = _find_biological_unit_column(metadata, groupby)
    row_counts = metadata[groupby].value_counts().to_dict()
    if unit_col is not None:
        unit_counts = (
            metadata.groupby(groupby, observed=True)[unit_col]
            .nunique(dropna=True)
            .to_dict()
        )
    else:
        unit_counts = dict(row_counts)

    validation: Dict[str, Any] = {
        "valid": True,
        "group_counts": unit_counts,
        "row_counts": row_counts,
        "unit_col": unit_col,
        "summary": [],
        "warnings": [],
        "errors": [],
    }
    unit_label = f"donor(s) in column '{unit_col}'" if unit_col else "samples"
    kind = "biological replicates" if unit_col else "replicates"

    if unit_col is None:
        validation["warnings"].append(
            "No biological-unit column (donor/patient/subject/individual/animal) "
            "found, so replicates are counted as rows. Confirm each row is an "
            "independent biological unit; technical libraries from one donor are "
            "not biological replicates."
        )

    for group, n_units in unit_counts.items():
        n_rows = int(row_counts.get(group, 0))
        n_units = int(n_units)
        if unit_col is not None:
            validation["summary"].append(
                f"Group '{group}': {n_rows} libraries/samples -> {n_units} {unit_label}"
            )
        if n_units < min_replicates:
            validation["valid"] = False
            detail = (
                f" ({n_rows} libraries/samples -> {n_units} {unit_label}; "
                "libraries from the same donor are technical, not biological, "
                "replicates)"
                if unit_col is not None and n_rows != n_units
                else ""
            )
            if unit_col is not None and n_units == 1:
                detail += (
                    "; a single donor cannot support a statistical comparison "
                    "(pseudoreplication)"
                )
            validation["errors"].append(
                f"Group '{group}' has only {n_units} {kind}{detail} "
                f"(minimum {min_replicates} required for stable variance estimation)"
            )
            continue
        if unit_col is not None and n_rows > n_units:
            validation["warnings"].append(
                f"Group '{group}': {n_rows} rows come from {n_units} donor(s). Sum "
                f"technical libraries per donor first (create_pseudobulk_matrix "
                f"with sample_col='{unit_col}'), otherwise the test treats "
                "technical replicates as independent."
            )
        if n_units < 4:
            validation["warnings"].append(
                f"Group '{group}' has {n_units} {kind}. "
                f"Statistical power may be limited. 4+ replicates recommended."
            )

    return validation


def _replicate_summary_text(validation: Dict[str, Any]) -> str:
    """Markdown lines reporting libraries/samples -> donors per group, if known."""
    if not validation.get("summary"):
        return ""
    lines = "".join(f"\n- {line}" for line in validation["summary"])
    return f"\n\n**Biological replicates (n counted in donors):**{lines}"


def _de_result_keys(adata) -> List[str]:
    """Include explicit historical choices, but never infer their chronology."""
    return sorted(
        key
        for key, value in adata.uns.items()
        if (key.startswith("de_results_") or key.startswith("de_iteration_results_"))
        and isinstance(value, dict)
    )


def _resolve_de_results_key(adata, requested: Optional[str] = None) -> str:
    """Use recorded identity; legacy data is safe only with one possible result."""
    key = (
        requested if requested is not None else adata.uns.get("de_current_results_key")
    )
    available = _de_result_keys(adata)
    if key is not None:
        # Caller-supplied identity is not limited by automatic discovery prefixes.
        if key not in adata.uns or (requested is None and key not in available):
            raise ValueError(
                f"DE results key {key!r} is unavailable. Choose an explicit "
                f"de_results_key from {available}."
            )
        result = adata.uns[key]
        if not isinstance(result, dict) or not isinstance(
            result.get("results_df"), pd.DataFrame
        ):
            raise ValueError(
                f"Invalid DE results format at {key!r}. "
                "Expected dict with 'results_df' DataFrame."
            )
        return key
    if len(available) == 1:
        return available[0]
    if not available:
        raise ValueError(
            "No DE results found. Run differential expression analysis first."
        )
    raise ValueError(
        "Current DE result identity is unavailable; choose an explicit "
        f"de_results_key from {available}. Dictionary order is not analysis time."
    )


def _read_de_iterations(adata) -> Dict[str, Any]:
    """Decode versioned H5AD-safe history, or migrate legacy input on next write."""
    if "de_iterations_json" in adata.uns:
        tracker = json.loads(adata.uns["de_iterations_json"])
        if not isinstance(tracker, dict) or tracker.get("version") != 1:
            raise ValueError(
                "Unsupported DE iteration ledger version; history was not changed."
            )
    else:
        tracker = _plain(
            adata.uns.get(
                "de_iterations",
                {
                    "iterations": [],
                    "current_iteration": 0,
                },
            )
        )
    if (
        not isinstance(tracker, dict)
        or not isinstance(tracker.get("iterations"), list)
        or any(not isinstance(i, dict) for i in tracker["iterations"])
        or not isinstance(tracker.get("current_iteration"), int)
        or tracker["current_iteration"] < 0
    ):
        raise ValueError("Invalid DE iteration ledger; history was not changed.")
    return tracker


def _write_de_iterations(adata, tracker: Dict[str, Any]) -> None:
    """Persist a JSON scalar, not a heterogeneous list that H5AD cannot write."""

    def json_value(value):
        value = _plain(value)
        if isinstance(value, dict):
            return {k: json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_value(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    encoded = json.dumps(json_value({**tracker, "version": 1}), allow_nan=False)
    adata.uns["de_iterations_json"] = encoded
    adata.uns.pop("de_iterations", None)


def _snapshot_ids(adata) -> List[int]:
    ids = []
    for key in adata.uns:
        for prefix in ("de_iteration_results_", "de_results_iteration_"):
            if key.startswith(prefix) and key[len(prefix) :].isdigit():
                ids.append(int(key[len(prefix) :]))
    return ids


def _next_de_iteration(adata, tracker: Dict[str, Any]) -> int:
    """Existing snapshots reserve IDs even when the ledger was lost."""
    return max([tracker["current_iteration"], *_snapshot_ids(adata)]) + 1


def _record_de_iteration(
    adata, iteration_info: Dict[str, Any], current_iter: int
) -> None:
    """Shared runtime/replay snapshot writer: immutable history, durable ledger."""
    key = iteration_info["results_key"]
    if key in adata.uns or current_iter in _snapshot_ids(adata):
        raise ValueError(
            f"Refusing to overwrite existing DE iteration snapshot {key!r}."
        )
    tracker = _read_de_iterations(adata)
    source = adata.uns[f"de_results_formula_{iteration_info['contrast_name']}"]
    tracker["iterations"].append(iteration_info)
    tracker["current_iteration"] = current_iter
    snapshot = {
        "results_df": source["results_df"].copy(),
        "analysis_stats": dict(source["analysis_stats"]),
    }
    # Serialize first: unsupported metadata must not leave a partial snapshot.
    _write_de_iterations(adata, tracker)
    adata.uns[key] = snapshot
    adata.uns["de_current_results_key"] = key


def _with_current_de_ir(
    ir: AnalysisStep, results_key: str, target: str = "adata"
) -> AnalysisStep:
    """Add identity bookkeeping to an existing DE calculation, without changing it."""
    ir.code_template += f"\n{target}.uns['de_current_results_key'] = {{{{ current_de_results_key | repr }}}}"
    ir.parameters["current_de_results_key"] = results_key
    return ir


def _de_design_ir(
    formula: str, contrast: List[str], reference_condition: Optional[str]
) -> AnalysisStep:
    """IR for prepare_de_design: validate the design and store it in uns."""
    return AnalysisStep(
        operation="lobster.prepare_de_design",
        tool_name="prepare_de_design",
        description=f"Validate and store DE design {formula} for contrast {contrast}",
        library="lobster",
        code_template=(
            _BULK_SERVICE_SETUP
            + "design_validation = bulk_service.validate_experimental_design(\n"
            "    metadata=adata.obs, formula={{ formula | repr }}, min_replicates=3\n"
            ")\n"
            "if not design_validation['valid']:\n"
            "    raise ValueError(\n"
            "        'Invalid experimental design: '\n"
            "        + '; '.join(design_validation['errors'])\n"
            "    )\n"
            "design_result = bulk_service.create_formula_design(\n"
            "    metadata=adata.obs,\n"
            "    condition_col={{ contrast[0] | repr }},\n"
            "    reference_condition={{ reference_condition | repr }},\n"
            ")\n"
            "adata.uns['formula_design'] = {\n"
            "    'formula': {{ formula | repr }},\n"
            "    'contrast': {{ contrast | repr }},\n"
            "    'design_matrix_info': design_result,\n"
            "    'validation_results': design_validation,\n"
            "}"
        ),
        imports=[],
        parameters={
            "formula": formula,
            "contrast": list(contrast),
            "reference_condition": reference_condition,
        },
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata"],
    )


def _design_validation_ir(formula: str) -> AnalysisStep:
    """IR for validate_experimental_design: a read-only check, not replayed."""
    return AnalysisStep(
        operation="lobster.validate_experimental_design",
        tool_name="validate_experimental_design",
        description=f"Validate experimental design {formula} (min 3 replicates)",
        library="lobster",
        code_template=(
            "from lobster.services.analysis.differential_formula_service import (\n"
            "    DifferentialFormulaService,\n"
            ")\n"
            "\n"
            "design_check = DifferentialFormulaService().validate_experimental_design(\n"
            "    adata.obs, {{ formula | repr }}, min_replicates=3\n"
            ")\n"
            "print('Design valid:', design_check['valid'])\n"
            "for message in design_check.get('errors', []) + design_check.get('warnings', []):\n"
            "    print('-', message)"
        ),
        imports=[],
        parameters={"formula": formula},
        parameter_schema={},
        input_entities=["adata"],
        output_entities=[],
        replayable=False,
    )


def _formula_de_ir(
    formula: str,
    contrast: List[str],
    reference_levels: Optional[Dict[str, str]],
    alpha: float,
    lfc_threshold: float,
    count_layer: Optional[str] = None,
) -> AnalysisStep:
    """IR for run_de_with_formula: design checks, pyDESeq2 run, results in uns."""
    contrast = list(contrast)
    reference_condition = (
        reference_levels.get(contrast[0]) if reference_levels else None
    )
    results_key = f"de_results_formula_{contrast[0]}_{contrast[1]}_vs_{contrast[2]}"
    count_source = "layers['counts']" if count_layer else "X"
    return AnalysisStep(
        operation="pydeseq2.formula_differential_expression",
        tool_name="run_de_with_formula",
        description=(
            f"pyDESeq2 DE with formula {formula}: {contrast[1]} vs {contrast[2]}; "
            f"whole-number counts from {count_source}"
        ),
        library="pydeseq2",
        code_template=(
            _BULK_SERVICE_SETUP
            + "design_validation = bulk_service.validate_experimental_design(\n"
            "    metadata=adata.obs, formula={{ formula | repr }}, min_replicates=3\n"
            ")\n"
            "if not design_validation['valid']:\n"
            "    raise ValueError(\n"
            "        'Invalid experimental design: '\n"
            "        + '; '.join(design_validation['errors'])\n"
            "    )\n"
            "design_result = bulk_service.create_formula_design(\n"
            "    metadata=adata.obs,\n"
            "    condition_col={{ contrast[0] | repr }},\n"
            "    reference_condition={{ reference_condition | repr }},\n"
            ")\n"
            "adata.uns['de_formula_design'] = {\n"
            "    'formula': {{ formula | repr }},\n"
            "    'contrast': {{ contrast | repr }},\n"
            "    'design_matrix_info': design_result,\n"
            "    'validation_results': design_validation,\n"
            "    'reference_levels': {{ reference_levels | repr }},\n"
            "}\n"
            "results_df, de_stats = bulk_service.run_pydeseq2_from_pseudobulk(\n"
            "    pseudobulk_adata=adata,\n"
            "    formula={{ formula | repr }},\n"
            "    contrast={{ contrast | repr }},\n"
            "    count_layer={{ count_layer | repr }},\n"
            "    alpha={{ alpha | repr }},\n"
            "    shrink_lfc=True,\n"
            "    n_cpus=1,\n"
            ")\n"
            "adata.uns[{{ results_key | repr }}] = {\n"
            "    'results_df': results_df,\n"
            "    'analysis_stats': de_stats,\n"
            "    'parameters': {\n"
            "        'formula': {{ formula | repr }},\n"
            "        'contrast': {{ contrast | repr }},\n"
            "        'alpha': {{ alpha | repr }},\n"
            "        'lfc_threshold': {{ lfc_threshold | repr }},\n"
            "        'reference_levels': {{ reference_levels | repr }},\n"
            "        'count_layer': {{ count_layer | repr }},\n"
            "        'count_source': {{ count_source | repr }},\n"
            "    },\n"
            "}\n"
            "adata.uns['de_current_results_key'] = {{ results_key | repr }}"
        ),
        imports=[],
        parameters={
            "formula": formula,
            "contrast": contrast,
            "reference_levels": reference_levels,
            "reference_condition": reference_condition,
            "alpha": alpha,
            "lfc_threshold": lfc_threshold,
            "results_key": results_key,
            "count_layer": count_layer,
            "count_source": count_source,
        },
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata", "results_df"],
    )


def _iteration_results_key(iteration_info: Dict[str, Any]) -> str:
    """uns key holding an iteration's own results (older records: by contrast)."""
    return iteration_info.get("results_key") or (
        f"de_results_formula_{iteration_info['contrast_name']}"
    )


def _iteration_record_ir(
    iteration_info: Dict[str, Any], current_iter: int
) -> AnalysisStep:
    """IR for iterate_de_analysis: snapshot the results and record the iteration.

    The DE run itself is recorded separately by run_de_with_formula's IR, which
    replays just before this step. This step copies that result to the
    iteration's own key and restores the iteration bookkeeping as recorded.
    """
    return AnalysisStep(
        operation="lobster.record_de_iteration",
        tool_name="iterate_de_analysis",
        description=f"Record DE iteration '{iteration_info.get('name')}'",
        library="lobster",
        code_template=(
            "from lobster.agents.transcriptomics.de_analysis_expert import _record_de_iteration\n"
            "_record_de_iteration(adata, {{ iteration_info | repr }}, {{ current_iter | repr }})"
        ),
        imports=[],
        parameters={
            "iteration_info": _plain(iteration_info),
            "current_iter": int(current_iter),
        },
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata"],
    )


def _iteration_comparison_ir(
    iter1_info: Dict[str, Any], iter2_info: Dict[str, Any], save_comparison: bool
) -> AnalysisStep:
    """IR for compare_de_iterations: recompute the comparison from stored results.

    Iterations are found by their unique results_key, not their display name:
    names may repeat, and a name lookup could pick the same record twice.
    """
    iteration_1, iteration_2 = iter1_info["name"], iter2_info["name"]
    return AnalysisStep(
        operation="lobster.compare_de_iterations",
        tool_name="compare_de_iterations",
        description=f"Compare DE iterations '{iteration_1}' and '{iteration_2}'",
        library="pandas",
        code_template=(
            "from lobster.agents.transcriptomics.de_analysis_expert import _read_de_iterations\n"
            "de_iterations = _read_de_iterations(adata)['iterations']\n"
            "def find_iteration(name, results_key):\n"
            "    if results_key:\n"
            "        return next(i for i in de_iterations if i.get('results_key') == results_key)\n"
            "    return next(i for i in de_iterations if i['name'] == name)\n"
            "iter1_info = find_iteration({{ iteration_1 | repr }}, {{ results_key_1 | repr }})\n"
            "iter2_info = find_iteration({{ iteration_2 | repr }}, {{ results_key_2 | repr }})\n"
            "def iteration_results(info):\n"
            "    key = info.get('results_key') or f\"de_results_formula_{info['contrast_name']}\"\n"
            "    return adata.uns[key]['results_df']\n"
            "results1 = iteration_results(iter1_info)\n"
            "results2 = iteration_results(iter2_info)\n"
            "sig1 = set(results1[results1['padj'] < 0.05].index)\n"
            "sig2 = set(results2[results2['padj'] < 0.05].index)\n"
            "overlap, unique1, unique2 = sig1 & sig2, sig1 - sig2, sig2 - sig1\n"
            "correlation = None\n"
            "if len(overlap) > 3:\n"
            "    overlap_genes = list(overlap)\n"
            "    correlation = results1.loc[overlap_genes, 'log2FoldChange'].corr(\n"
            "        results2.loc[overlap_genes, 'log2FoldChange']\n"
            "    )\n"
            "overlap_percent = len(overlap) / max(len(sig1), len(sig2)) * 100\n"
            "print(f'Overlap: {len(overlap)} genes ({overlap_percent:.1f}%), correlation: {correlation}')\n"
            "{% if save_comparison %}"
            "adata.uns.setdefault('iteration_comparisons', {})[\n"
            "    {{ ('iteration_comparison_' ~ iteration_1 ~ '_vs_' ~ iteration_2) | repr }}\n"
            "] = {\n"
            "    'iteration_1': iter1_info,\n"
            "    'iteration_2': iter2_info,\n"
            "    'overlap_genes': list(overlap),\n"
            "    'unique_to_1': list(unique1),\n"
            "    'unique_to_2': list(unique2),\n"
            "    'correlation': correlation,\n"
            "    'summary_stats': {\n"
            "        'n_sig_1': len(sig1),\n"
            "        'n_sig_2': len(sig2),\n"
            "        'n_overlap': len(overlap),\n"
            "        'overlap_percent': overlap_percent,\n"
            "    },\n"
            "}\n"
            "{% endif %}"
        ),
        imports=[],
        parameters={
            "iteration_1": iteration_1,
            "iteration_2": iteration_2,
            "results_key_1": iter1_info.get("results_key"),
            "results_key_2": iter2_info.get("results_key"),
            "save_comparison": save_comparison,
        },
        parameter_schema={},
        input_entities=["adata"],
        output_entities=["adata"] if save_comparison else [],
        replayable=save_comparison,
    )


class DEAnalysisError(Exception):
    """Base exception for differential expression analysis errors."""

    pass


class ModalityNotFoundError(DEAnalysisError):
    """Raised when requested modality doesn't exist."""

    pass


class InsufficientReplicatesError(DEAnalysisError):
    """Raised when there are insufficient replicates for stable variance estimation."""

    pass


def de_analysis_expert(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "de_analysis_expert",
    delegation_tools: List = None,
    workspace_path: Optional[Path] = None,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    inference_context: Optional[InferenceContext] = None,
    model_factory: Optional[BoundModelFactory] = None,
):
    """
    Create differential expression analysis sub-agent.

    This agent handles all DE-related tasks for transcriptomics workflows.

    Args:
        data_manager: DataManagerV2 instance for modality management
        callback_handler: Optional LangChain callback handler
        agent_name: Name for the agent
        delegation_tools: List of delegation tools from parent agent

    Returns:
        LangGraph react agent configured for DE analysis
    """
    strict_binding = validate_inference_binding(inference_context, model_factory)
    model_params = {}
    if not strict_binding:
        settings = get_settings()
        model_params = settings.get_agent_llm_params("de_analysis_expert")
    llm = create_llm(
        "de_analysis_expert",
        model_params,
        provider_override=provider_override,
        model_override=model_override,
        workspace_path=workspace_path,
        **(
            {"inference_context": inference_context, "model_factory": model_factory}
            if strict_binding
            else {}
        ),
    )

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = (
            callback_handler
            if isinstance(callback_handler, list)
            else [callback_handler]
        )
        llm = llm.with_config(callbacks=callbacks)

    # Initialize stateless services
    pseudobulk_service = PseudobulkService()
    bulk_rnaseq_service = BulkRNASeqService(data_manager=data_manager)
    formula_service = DifferentialFormulaService()

    analysis_results = {"summary": "", "details": {}}

    # -------------------------
    # HELPER FUNCTIONS
    # -------------------------
    def _deseq2_count_source(adata) -> Tuple[Optional[str], Optional[str]]:
        """Choose a finite non-negative whole-number matrix for pyDESeq2.

        This numeric check does not establish raw-count provenance.
        pyDESeq2 casts its input to int, so fractional normalized or
        log-transformed values would be silently truncated. Returns
        (count_layer, None), where count_layer None means adata.X, or
        (None, refusal) when the available matrix fails validation.
        """
        from lobster.core.utils.count_matrix import is_whole_number_matrix as is_counts

        hint = (
            "DESeq2 needs raw integer counts. Was this pseudobulk matrix "
            "aggregated from normalized values? Rebuild it from raw counts, "
            "using layer 'counts' (or load the count matrix), and run again."
        )
        if "counts" in adata.layers:
            if is_counts(adata.layers["counts"]):
                return "counts", None
            return None, (
                "**Raw counts required**: layer 'counts' contains non-finite, "
                f"negative or non-integer values. {hint}"
            )
        if is_counts(adata.X):
            return None, None
        return None, (
            "**Raw counts required**: the pseudobulk expression matrix does not "
            "hold raw integer counts (non-finite, negative, normalized or "
            "log-transformed values), and there is no 'counts' layer. "
            f"{hint}"
        )

    def _extract_raw_counts(adata) -> Tuple[pd.DataFrame, bool]:
        """
        Extract raw counts from AnnData, preferring adata.raw.X.

        CRITICAL: DESeq2 requires raw integer counts, not normalized data.

        Args:
            adata: AnnData object

        Returns:
            Tuple of (count_matrix DataFrame, used_raw_flag)
        """
        used_raw = False

        # CRITICAL FIX: Use raw counts for DESeq2
        if adata.raw is not None:
            raw_data = adata.raw.X
            if hasattr(raw_data, "toarray"):
                raw_data = raw_data.toarray()
            count_matrix = pd.DataFrame(
                raw_data.T, index=adata.raw.var_names, columns=adata.obs_names
            )
            used_raw = True
            logger.info("Using adata.raw.X for count matrix (recommended for DESeq2)")
        else:
            # Fallback to adata.X with warning
            logger.warning(
                "No adata.raw found - using adata.X which may be normalized. "
                "DESeq2 requires raw counts for accurate results."
            )
            data = adata.X
            if hasattr(data, "toarray"):
                data = data.toarray()
            count_matrix = pd.DataFrame(
                data.T, index=adata.var_names, columns=adata.obs_names
            )

        return count_matrix, used_raw

    def _validate_replicate_counts(
        metadata: pd.DataFrame,
        groupby: str,
        min_replicates: int = 3,  # SCIENTIFIC FIX: Changed from 2 to 3
    ) -> Dict[str, Any]:
        """Validate replicate counts per condition, in biological units.

        See ``_count_biological_replicates``: n is the number of distinct donors
        (or patients, subjects, animals) per group when such a column exists.
        """
        return _count_biological_replicates(metadata, groupby, min_replicates)

    # -------------------------
    # PSEUDOBULK TOOLS
    # -------------------------
    @tool
    def create_pseudobulk_matrix(
        modality_name: str,
        sample_col: str,
        celltype_col: str,
        layer: str = None,
        min_cells: int = 10,
        aggregation_method: str = "sum",
        min_genes: int = 200,
        filter_zeros: bool = True,
        save_result: bool = True,
    ) -> str:
        """
        Aggregate single-cell data to pseudobulk matrix for differential expression analysis.

        IMPORTANT: This tool extracts RAW COUNTS from adata.raw.X for DESeq2 compatibility.

        Args:
            modality_name: Name of single-cell modality to aggregate
            sample_col: Column identifying the BIOLOGICAL replicate (donor,
                patient, animal), NOT the sequencing library or run. All cells
                and libraries from the same donor are summed into one pseudobulk
                sample. If no such column exists or it is unclear which one it
                is, stop and ask (or, when you cannot ask, stop and explain);
                never pass a library or run ID here.
            celltype_col: Column containing cell type identifiers
            layer: Layer to use for aggregation (default: None, uses raw counts)
            min_cells: Minimum cells per sample-celltype combination
            aggregation_method: Aggregation method ('sum' for DESeq2, 'mean', 'median')
            min_genes: Minimum genes detected per pseudobulk sample
            filter_zeros: Remove genes with all zeros after aggregation
            save_result: Whether to save the pseudobulk modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the single-cell modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Creating pseudobulk matrix from '{modality_name}': "
                f"{adata.n_obs} cells x {adata.n_vars} genes"
            )

            # Validate required columns exist
            if sample_col not in adata.obs.columns:
                available_cols = list(adata.obs.columns)[:10]
                raise PseudobulkError(
                    f"Sample column '{sample_col}' not found. "
                    f"Available columns: {available_cols}..."
                )

            if celltype_col not in adata.obs.columns:
                available_cols = list(adata.obs.columns)[:10]
                raise PseudobulkError(
                    f"Cell type column '{celltype_col}' not found. "
                    f"Available columns: {available_cols}..."
                )

            # Use pseudobulk service (it uses raw counts internally)
            pseudobulk_adata, _stats, ir = pseudobulk_service.aggregate_to_pseudobulk(
                adata=adata,
                sample_col=sample_col,
                celltype_col=celltype_col,
                layer=layer,
                min_cells=min_cells,
                aggregation_method=aggregation_method,
                min_genes=min_genes,
                filter_zeros=filter_zeros,
            )

            # Save as new modality
            pseudobulk_modality_name = f"{modality_name}_pseudobulk"
            data_manager.store_modality(
                name=pseudobulk_modality_name,
                adata=pseudobulk_adata,
                parent_name=modality_name,
                step_summary=f"Created pseudobulk: {pseudobulk_adata.n_obs} samples x {pseudobulk_adata.n_vars} genes",
            )

            # Save to file if requested
            if save_result:
                save_path = f"{modality_name}_pseudobulk.h5ad"
                data_manager.save_modality(pseudobulk_modality_name, save_path)

            # Get aggregation statistics
            agg_stats = pseudobulk_adata.uns.get("aggregation_stats", {})

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="create_pseudobulk_matrix",
                parameters={
                    "modality_name": modality_name,
                    "sample_col": sample_col,
                    "celltype_col": celltype_col,
                    "aggregation_method": aggregation_method,
                    "min_cells": min_cells,
                },
                description=f"Created pseudobulk matrix: {pseudobulk_adata.n_obs} samples x {pseudobulk_adata.n_vars} genes",
                ir=ir,
                agent=agent_name,
            )

            # Format response
            response = f"""Pseudobulk matrix created from single-cell data '{modality_name}'!

**Aggregation Results:**
- Original: {adata.n_obs:,} single cells -> {pseudobulk_adata.n_obs} pseudobulk samples
- Genes retained: {pseudobulk_adata.n_vars:,} / {adata.n_vars:,} ({pseudobulk_adata.n_vars / adata.n_vars * 100:.1f}%)
- Aggregation method: {aggregation_method}
- Min cells threshold: {min_cells}

**Sample & Cell Type Distribution:**
- Unique samples: {agg_stats.get("n_samples", "N/A")}
- Cell types: {agg_stats.get("n_cell_types", "N/A")}
- Total cells aggregated: {agg_stats.get("total_cells_aggregated", "N/A"):,}
- Mean cells per pseudobulk: {agg_stats.get("mean_cells_per_pseudobulk", 0):.1f}

**New modality created**: '{pseudobulk_modality_name}'"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += "\n\nNext step: Use 'prepare_de_design' to set up statistical design for DE analysis."

            analysis_results["details"]["pseudobulk_aggregation"] = response
            return response

        except (
            PseudobulkError,
            AggregationError,
            InsufficientCellsError,
            ModalityNotFoundError,
        ) as e:
            logger.error(f"Error creating pseudobulk matrix: {e}")
            return f"Error creating pseudobulk matrix: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pseudobulk creation: {e}")
            return f"Unexpected error: {str(e)}"

    create_pseudobulk_matrix.metadata = {
        "categories": ["PREPROCESS"],
        "provenance": True,
    }
    create_pseudobulk_matrix.tags = ["PREPROCESS"]

    @tool
    def prepare_de_design(
        modality_name: str,
        formula: str,
        contrast: List[str],
        reference_condition: str = None,
    ) -> str:
        """
        Prepare design matrix and validate experimental design for differential expression analysis.

        CRITICAL: Validates minimum replicate requirements (3+ per condition).

        Args:
            modality_name: Name of pseudobulk modality
            formula: R-style formula (e.g., "~condition + batch")
            contrast: Contrast specification [factor, level1, level2]
            reference_condition: Reference level for main condition
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the pseudobulk modality
            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Preparing DE design for '{modality_name}': "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            # SCIENTIFIC FIX: Validate with min_replicates=3
            design_validation = bulk_rnaseq_service.validate_experimental_design(
                metadata=adata.obs, formula=formula, min_replicates=3
            )

            if not design_validation["valid"]:
                error_msg = "; ".join(design_validation["errors"])
                raise DesignMatrixError(f"Invalid experimental design: {error_msg}")

            # Additional replicate validation
            condition_col = contrast[0]
            replicate_validation = _validate_replicate_counts(
                adata.obs, condition_col, min_replicates=3
            )

            if not replicate_validation["valid"]:
                error_msg = "; ".join(replicate_validation["errors"])
                raise InsufficientReplicatesError(error_msg)

            # Create design matrix
            design_result = bulk_rnaseq_service.create_formula_design(
                metadata=adata.obs,
                condition_col=contrast[0],
                reference_condition=reference_condition,
            )

            # Store design information in modality
            adata.uns["formula_design"] = {
                "formula": formula,
                "contrast": contrast,
                "design_matrix_info": design_result,
                "validation_results": design_validation,
            }

            # Update modality with design info
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                step_summary=f"Prepared DE design: formula={formula}",
            )

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="prepare_de_design",
                parameters={
                    "modality_name": modality_name,
                    "formula": formula,
                    "contrast": contrast,
                },
                description=f"Prepared DE design for {adata.n_obs} pseudobulk samples",
                ir=_de_design_ir(formula, contrast, reference_condition),
                agent=agent_name,
            )

            # Format response with warnings
            response = f"""Differential expression design prepared for '{modality_name}'!

**Experimental Design:**
- Formula: {formula}
- Contrast: {contrast[1]} vs {contrast[2]} in {contrast[0]}
- Design matrix: {design_result["design_matrix"].shape[0]} samples x {design_result["design_matrix"].shape[1]} coefficients
- Matrix rank: {design_result["rank"]} (full rank: {"Yes" if design_result["rank"] == design_result["n_coefficients"] else "No"})

**Design Validation:**
- Valid: {"Yes" if design_validation["valid"] else "No"}
- Warnings: {len(design_validation["warnings"])} ({", ".join(design_validation["warnings"][:2]) if design_validation["warnings"] else "None"})

**Replicate Counts:**"""

            for group, count in replicate_validation["group_counts"].items():
                status = (
                    "OK"
                    if count >= 4
                    else "LOW POWER" if count >= 3 else "INSUFFICIENT"
                )
                response += f"\n- {group}: {count} replicates ({status})"
            response += _replicate_summary_text(replicate_validation)

            if replicate_validation["warnings"]:
                response += "\n\n**Warnings:**"
                for warning in replicate_validation["warnings"]:
                    response += f"\n- {warning}"

            response += (
                "\n\n**Design information stored in**: adata.uns['formula_design']"
            )
            response += "\n\nNext step: Run 'run_differential_expression' to perform pyDESeq2 analysis."

            analysis_results["details"]["de_design"] = response
            return response

        except (
            DesignMatrixError,
            FormulaError,
            ModalityNotFoundError,
            InsufficientReplicatesError,
        ) as e:
            logger.error(f"Error preparing DE design: {e}")
            return f"Error preparing differential expression design: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in DE design preparation: {e}")
            return f"Unexpected error: {str(e)}"

    prepare_de_design.metadata = {"categories": ["PREPROCESS"], "provenance": True}
    prepare_de_design.tags = ["PREPROCESS"]

    @tool
    def run_differential_expression(
        modality_name: str,
        groupby: str,
        group1: str,
        group2: str,
        method: str = "deseq2",
        alpha: float = 0.05,
        shrink_lfc: bool = True,
        save_result: bool = True,
    ) -> str:
        """
        Run differential expression analysis (simple 2-group comparison).

        Works for both pseudobulk (from single-cell) and direct bulk RNA-seq data.
        Auto-detects whether the data is pseudobulk (via adata.uns['pseudobulk_design']
        or adata.uns['formula_design']) and routes to the appropriate analysis method.

        CRITICAL: For pyDESeq2, require finite non-negative whole-number counts
        from layer 'counts', or X if no counts layer is present; refuse other
        values. The numeric check does not establish raw-count provenance.
        Validates minimum replicate requirements (3+ per group).

        For complex multi-factor designs with covariates, use run_de_with_formula instead.

        Args:
            modality_name: Name of the bulk RNA-seq or pseudobulk modality
            groupby: Column name for grouping (e.g., 'condition', 'treatment')
            group1: First group for comparison (e.g., 'control')
            group2: Second group for comparison (e.g., 'treatment')
            method: Analysis method ('deseq2', 'wilcoxon', 't_test')
            alpha: Significance threshold for adjusted p-values
            shrink_lfc: Whether to apply log fold change shrinkage (DESeq2 only)
            save_result: Whether to save the results
        """
        try:
            # Ensure group IDs are strings (cluster IDs from Leiden are often integers)
            group1 = str(group1)
            group2 = str(group2)

            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get the modality
            adata = data_manager.get_modality(modality_name)

            # The result name carries the group labels (data values, which may hold
            # '/'): encoded into one plain name, checked before any analysis.
            de_modality_name = None
            if not (_is_pseudobulk(adata) and method == "deseq2"):
                de_modality_name = _de_result_name(modality_name, group1, group2)

            # Every save target of the branch that will run is checked before the
            # data changes or any analysis runs.
            if save_result:
                if _is_pseudobulk(adata) and method == "deseq2":
                    results_csv = (
                        data_manager.exports_dir / f"{modality_name}_de_results.csv"
                    )
                    assert_inside_workspace(data_manager.workspace_path, results_csv)
                    assert_export_allowed(data_manager.workspace_path, results_csv)
                    data_manager._assert_save_allowed(
                        f"{modality_name}_with_de_results.h5ad"
                    )
                else:
                    data_manager._assert_save_allowed(f"{de_modality_name}.h5ad")

            logger.info(
                f"Running DE analysis on '{modality_name}': "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            # Validate groupby column exists
            if groupby not in adata.obs.columns:
                available_columns = [
                    col
                    for col in adata.obs.columns
                    if col.lower() in ["condition", "treatment", "group", "batch"]
                ]
                return f"Grouping column '{groupby}' not found. Available experimental design columns: {available_columns}"

            # Check if groups exist - match type of column values
            available_groups = list(adata.obs[groupby].unique())
            available_groups_str = [str(g) for g in available_groups]

            if group1 not in available_groups and group1 in available_groups_str:
                adata.obs[groupby] = adata.obs[groupby].astype(str)
                available_groups = list(adata.obs[groupby].unique())
            elif group1 not in available_groups:
                return f"Group '{group1}' not found in column '{groupby}'. Available groups: {available_groups}"
            if group2 not in available_groups:
                return f"Group '{group2}' not found in column '{groupby}'. Available groups: {available_groups}"

            # Validate replicate counts
            replicate_validation = _validate_replicate_counts(
                adata.obs, groupby, min_replicates=3
            )

            if not replicate_validation["valid"]:
                error_msg = "; ".join(replicate_validation["errors"])
                return f"**Insufficient replicates**: {error_msg}\n\nMinimum 3 replicates per group required for stable variance estimation."

            # Get group counts for response
            group1_count = (adata.obs[groupby] == group1).sum()
            group2_count = (adata.obs[groupby] == group2).sum()

            # Auto-detect pseudobulk vs direct bulk
            is_pseudobulk = _is_pseudobulk(adata)

            if is_pseudobulk and method == "deseq2":
                # Pseudobulk path: use pyDESeq2 via formula design
                formula = f"~{groupby}"
                contrast = [groupby, group2, group1]

                # CRITICAL: DESeq2 must run on raw integer counts
                count_layer, refusal = _deseq2_count_source(adata)
                if refusal:
                    return refusal
                count_source = "layers['counts']" if count_layer else "X"

                results_df, analysis_stats = (
                    bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                        pseudobulk_adata=adata,
                        formula=formula,
                        contrast=contrast,
                        count_layer=count_layer,
                        alpha=alpha,
                        shrink_lfc=shrink_lfc,
                        n_cpus=1,
                    )
                )

                # Store results
                contrast_name = f"{groupby}_{group2}_vs_{group1}"
                adata.uns[f"de_results_{contrast_name}"] = {
                    "results_df": results_df,
                    "analysis_stats": analysis_stats,
                    "parameters": {
                        "alpha": alpha,
                        "shrink_lfc": shrink_lfc,
                        "formula": formula,
                        "contrast": contrast,
                        "used_integer_counts": True,
                        "count_layer": count_layer,
                        "count_source": count_source,
                        "method": "pydeseq2",
                    },
                }

                adata.uns["de_current_results_key"] = f"de_results_{contrast_name}"
                data_manager.store_modality(
                    name=modality_name,
                    adata=adata,
                    step_summary=f"DE analysis complete: {contrast_name}",
                )

                if save_result:
                    results_path = (
                        data_manager.exports_dir / f"{modality_name}_de_results.csv"
                    )
                    assert_export_allowed(data_manager.workspace_path, results_path)
                    results_df.to_csv(results_path)
                    save_path = f"{modality_name}_with_de_results.h5ad"
                    data_manager.save_modality(modality_name, save_path)

                # Create IR for provenance
                from lobster.core.provenance import AnalysisStep

                ir = AnalysisStep(
                    operation="pydeseq2.differential_expression",
                    tool_name="run_differential_expression",
                    description=(
                        f"pyDESeq2 DE: {group2} vs {group1} in {groupby}; "
                        f"whole-number counts from {count_source}"
                    ),
                    library="pydeseq2",
                    parameters={
                        "modality_name": modality_name,
                        "groupby": groupby,
                        "group1": group1,
                        "group2": group2,
                        "alpha": alpha,
                        "shrink_lfc": shrink_lfc,
                        "count_layer": count_layer,
                        "count_source": count_source,
                        "formula": formula,
                        "contrast": contrast,
                    },
                    parameter_schema={},
                    code_template=(
                        _BULK_SERVICE_SETUP
                        + "results_df, de_stats = bulk_service.run_pydeseq2_from_pseudobulk(\n"
                        "    pseudobulk_adata=adata,\n"
                        "    formula={{ formula | repr }},\n"
                        "    contrast={{ contrast | repr }},\n"
                        "    count_layer={{ count_layer | repr }},\n"
                        "    alpha={{ alpha | repr }},\n"
                        "    shrink_lfc={{ shrink_lfc | repr }},\n"
                        "    n_cpus=1,\n"
                        ")"
                    ),
                    imports=[],
                )

                ir = _with_current_de_ir(ir, f"de_results_{contrast_name}")
                data_manager.log_tool_usage(
                    tool_name="run_differential_expression",
                    parameters={
                        "modality_name": modality_name,
                        "groupby": groupby,
                        "group1": group1,
                        "group2": group2,
                        "method": "pydeseq2",
                        "alpha": alpha,
                        "shrink_lfc": shrink_lfc,
                        "count_layer": count_layer,
                        "count_source": count_source,
                    },
                    description=(
                        f"pyDESeq2 analysis: {analysis_stats['n_significant_genes']} significant genes found; "
                        f"whole-number counts from {count_source}"
                    ),
                    ir=ir,
                    agent=agent_name,
                )

                response = f"""## Differential Expression Analysis Complete for '{modality_name}'

**pyDESeq2 Analysis Results (pseudobulk detected):**
- Contrast: {group2} vs {group1} in {groupby}
- Genes tested: {analysis_stats["n_genes_tested"]:,}
- Significant genes: {analysis_stats["n_significant_genes"]:,} (alpha={alpha})
- Upregulated: {analysis_stats["n_upregulated"]:,}
- Downregulated: {analysis_stats["n_downregulated"]:,}

**Top Differentially Expressed Genes:**
**Upregulated ({group2} > {group1}):**
{chr(10).join([f"- {gene}" for gene in analysis_stats["top_upregulated"][:5]])}

**Downregulated ({group2} < {group1}):**
{chr(10).join([f"- {gene}" for gene in analysis_stats["top_downregulated"][:5]])}

**Analysis Parameters:**
- Method: pyDESeq2 (pseudobulk)
- LFC shrinkage: {"Yes" if shrink_lfc else "No"}
- Significance threshold: {alpha}
- Used whole-number counts from: {count_source}

**Results stored in**: adata.uns['de_results_{contrast_name}']"""

                if save_result:
                    response += f"\n**Saved to**: {results_path} & {save_path}"

            else:
                # Direct bulk / simple 2-group DE path
                method_map = {
                    "deseq2": "deseq2_like",
                    "wilcoxon": "wilcoxon",
                    "t_test": "t_test",
                }
                internal_method = method_map.get(method, method)

                adata_de, de_stats, ir = (
                    bulk_rnaseq_service.run_differential_expression_analysis(
                        adata=adata,
                        groupby=groupby,
                        group1=group1,
                        group2=group2,
                        method=internal_method,
                    )
                )

                refusal = _empty_bulk_de_refusal(adata_de, de_stats)
                if refusal:
                    return refusal
                adata_de.uns["de_current_results_key"] = de_stats["de_results_key"]
                ir = _with_current_de_ir(
                    ir,
                    de_stats["de_results_key"],
                    "adata_de" if internal_method == "deseq2_like" else "adata",
                )
                # Save as new modality (name derived and checked before the analysis)
                data_manager.store_modality(
                    name=de_modality_name,
                    adata=adata_de,
                    parent_name=modality_name,
                    step_summary=f"DE analysis: {group1} vs {group2}",
                )

                if save_result:
                    save_path = f"{de_modality_name}.h5ad"
                    data_manager.save_modality(de_modality_name, save_path)

                data_manager.log_tool_usage(
                    tool_name="run_differential_expression",
                    parameters={
                        "modality_name": modality_name,
                        "groupby": groupby,
                        "group1": group1,
                        "group2": group2,
                        "method": internal_method,
                    },
                    description=f"DE analysis: {de_stats['n_significant_genes']} significant genes found",
                    ir=ir,
                    agent=agent_name,
                )

                analysis_stats = de_stats

                response = f"""## Differential Expression Analysis Complete for '{modality_name}'

**Analysis Results:**
- Comparison: {group1} ({group1_count} samples) vs {group2} ({group2_count} samples)
- Method: {de_stats['method']}
- Genes tested: {de_stats['n_genes_tested']:,}
- Significant genes (padj < {alpha}): {de_stats['n_significant_genes']:,}

**Differential Expression Summary:**
- Upregulated in {group2}: {de_stats['n_upregulated']} genes
- Downregulated in {group2}: {de_stats['n_downregulated']} genes

**Top Upregulated Genes:**
{chr(10).join([f"- {gene}" for gene in de_stats["top_upregulated"][:5]])}

**Top Downregulated Genes:**
{chr(10).join([f"- {gene}" for gene in de_stats["top_downregulated"][:5]])}

**New modality created**: '{de_modality_name}'"""

                if save_result:
                    response += f"\n**Saved to**: {save_path}"

                response += f"\n**Access detailed results**: adata.uns['{de_stats['de_results_key']}']\n"

            response += _replicate_summary_text(replicate_validation)

            # Add replicate warnings if any
            if replicate_validation["warnings"]:
                response += "\n\n**Statistical Power Warnings:**\n"
                for warning in replicate_validation["warnings"]:
                    response += f"- {warning}\n"

            response += "\n\nNext steps: Visualize with volcano plots, use filter_de_results to filter, or run_pathway_enrichment for pathway analysis."

            analysis_results["details"]["differential_expression"] = response
            return response

        except WorkspaceExportError:
            # A save target is outside the workspace: nothing was saved.
            return WORKSPACE_EXPORT_MESSAGE
        except InvalidModalityNameError as e:
            # The derived result name is not one plain name (fixed reason).
            return f"Error: {e}"
        except ManagedPathError:
            # A save target is inside the managed figure folder: nothing was saved.
            return MANAGED_EXPORT_MESSAGE
        except (BulkRNASeqError, PseudobulkError, ModalityNotFoundError) as e:
            logger.error(f"Error in differential expression analysis: {e}")
            return f"Error in differential expression analysis: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in differential expression: {e}")
            return f"Unexpected error: {str(e)}"

    run_differential_expression.metadata = {
        "categories": ["ANALYZE"],
        "provenance": True,
    }
    run_differential_expression.tags = ["ANALYZE"]

    @tool
    def validate_experimental_design(
        modality_name: str,
        formula: str,
    ) -> str:
        """
        Validate experimental design for statistical power and balance.

        CRITICAL: Requires minimum 3 replicates per condition (changed from 2).
        Warns when any condition has fewer than 4 replicates.

        Args:
            modality_name: Name of bulk RNA-seq or pseudobulk modality
            formula: R-style formula to validate
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            # Get modality
            adata = data_manager.get_modality(modality_name)
            metadata = adata.obs

            # SCIENTIFIC FIX: Validate design with min_replicates=3
            validation_result = formula_service.validate_experimental_design(
                metadata, formula, min_replicates=3
            )

            # Format response
            response = f"## Experimental Design Validation: `{formula}`\n\n"
            response += f"**Modality**: `{modality_name}`\n"
            response += f"**Samples**: {len(metadata)}\n\n"

            # Overall status
            if validation_result["valid"]:
                response += "**Overall Status**: PASSED - Design is valid for DESeq2 analysis\n\n"
            else:
                response += (
                    "**Overall Status**: FAILED - Design has issues (see below)\n\n"
                )

            # Design summary
            if validation_result.get("design_summary"):
                response += "### Group Balance\n"
                for factor, counts in validation_result["design_summary"].items():
                    response += f"**{factor}**:\n"
                    for level, count in counts.items():
                        # SCIENTIFIC FIX: Indicate status based on replicate count
                        if count < 3:
                            status = "INSUFFICIENT"
                        elif count < 4:
                            status = "LOW POWER"
                        else:
                            status = "OK"
                        response += f"  - {level}: {count} samples ({status})\n"
                response += "\n"

            # Warnings
            if validation_result.get("warnings"):
                response += "### Warnings\n"
                for warning in validation_result["warnings"]:
                    response += f"- {warning}\n"
                response += "\n"

            # Errors
            if validation_result.get("errors"):
                response += "### Errors\n"
                for error in validation_result["errors"]:
                    response += f"- {error}\n"
                response += "\n"

            # Replicate requirements note
            response += "### Replicate Requirements\n"
            response += "- **Minimum required**: 3 replicates per condition (for stable variance estimation)\n"
            response += "- **Recommended**: 4+ replicates per condition (for adequate statistical power)\n"
            response += "- **Ideal**: 6+ replicates per condition (for publication-quality results)\n\n"

            if validation_result["valid"]:
                response += "**Conclusion**: Design is ready for pyDESeq2 analysis\n"
            else:
                response += (
                    "**Conclusion**: Please address issues before running analysis\n"
                )

            data_manager.log_tool_usage(
                tool_name="validate_experimental_design",
                parameters={"modality_name": modality_name, "formula": formula},
                description=f"Validated experimental design: {'valid' if validation_result['valid'] else 'invalid'}",
                ir=_design_validation_ir(formula),
                agent=agent_name,
            )

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Modality not found error: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Error validating experimental design: {e}")
            return f"Error: {str(e)}"

    validate_experimental_design.metadata = {
        "categories": ["QUALITY"],
        "provenance": True,
    }
    validate_experimental_design.tags = ["QUALITY"]

    @tool
    def suggest_de_formula(
        modality_name: str,
        groupby: str = None,
        covariates: Optional[List[str]] = None,
        include_interaction: bool = False,
    ) -> str:
        """
        Analyze metadata, suggest DE formula, construct it, and validate the design.

        Combines metadata analysis, formula construction, and validation in one tool.
        Examines the data to suggest appropriate formulas, then builds and validates
        the chosen formula. Returns ready-to-use formula for run_de_with_formula.

        Args:
            modality_name: Name of pseudobulk or bulk RNA-seq modality
            groupby: Main variable of interest (auto-detected if None)
            covariates: Optional list of covariate variables to include
            include_interaction: Whether to include interaction terms
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                return f"Modality '{modality_name}' not found. Available: {data_manager.list_modalities()}"

            # Get the data
            adata = data_manager.get_modality(modality_name)
            metadata = adata.obs
            n_samples = len(metadata)

            # Step 1: Analyze metadata columns
            variable_analysis = {}
            for col in metadata.columns:
                if col.startswith("_") or col in ["n_cells", "total_counts"]:
                    continue

                series = metadata[col]
                if pd.api.types.is_numeric_dtype(series):
                    var_type = "continuous"
                else:
                    var_type = "categorical"

                unique_vals = len(series.unique())
                missing = series.isna().sum()

                variable_analysis[col] = {
                    "type": var_type,
                    "unique_values": unique_vals,
                    "missing_count": missing,
                    "sample_values": list(series.unique())[:5],
                }

            categorical_vars = [
                col
                for col, info in variable_analysis.items()
                if info["type"] == "categorical"
                and info["unique_values"] > 1
                and info["unique_values"] < n_samples / 2
            ]
            continuous_vars = [
                col
                for col, info in variable_analysis.items()
                if info["type"] == "continuous"
                and info["missing_count"] < n_samples / 2
            ]

            # Auto-detect groupby if not provided
            if groupby is None:
                for col in categorical_vars:
                    if variable_analysis[col]["unique_values"] == 2:
                        groupby = col
                        break
                if groupby is None and categorical_vars:
                    groupby = categorical_vars[0]

            if groupby is None:
                return (
                    "**No suitable grouping variable found.**\n"
                    "Please ensure your data has at least one categorical variable with 2+ levels.\n"
                    f"Available variables: {list(variable_analysis.keys())}"
                )

            # Validate groupby exists
            if groupby not in metadata.columns:
                available_vars = [
                    col for col in metadata.columns if not col.startswith("_")
                ]
                return f"Variable '{groupby}' not found. Available: {available_vars}"

            # Validate covariates if provided
            if covariates:
                missing_covariates = [
                    c for c in covariates if c not in metadata.columns
                ]
                if missing_covariates:
                    return f"Covariates not found: {missing_covariates}"

            # Step 2: Construct formula
            formula_terms = [groupby]
            if covariates:
                formula_terms.extend(covariates)

            if include_interaction and covariates:
                interaction_term = f"{groupby}*{covariates[0]}"
                formula = f"~{interaction_term}"
                if len(covariates) > 1:
                    formula += f" + {' + '.join(covariates[1:])}"
            else:
                formula = f"~{' + '.join(formula_terms)}"

            # Step 3: Validate formula
            validation_warnings = []
            validation_errors = []

            # Check all terms exist
            for term in formula_terms:
                if term not in metadata.columns:
                    validation_errors.append(f"Variable '{term}' not found in metadata")

            # Check for confounding (single-level factors)
            for term in formula_terms:
                if (
                    term in variable_analysis
                    and variable_analysis[term]["unique_values"] < 2
                ):
                    validation_errors.append(
                        f"Variable '{term}' has only 1 level - cannot be used in formula"
                    )

            # Check replicate counts
            replicate_validation = _validate_replicate_counts(
                metadata, groupby, min_replicates=3
            )
            if not replicate_validation["valid"]:
                for err in replicate_validation["errors"]:
                    validation_errors.append(err)
            if replicate_validation["warnings"]:
                for warn in replicate_validation["warnings"]:
                    validation_warnings.append(warn)

            # Validate with formula service if no errors
            formula_valid = len(validation_errors) == 0
            design_result = None
            if formula_valid:
                try:
                    formula_components = formula_service.parse_formula(
                        formula, metadata
                    )
                    design_result = formula_service.construct_design_matrix(
                        formula_components, metadata
                    )
                    svc_validation = formula_service.validate_experimental_design(
                        metadata, formula, min_replicates=3
                    )
                    if not svc_validation["valid"]:
                        for err in svc_validation.get("errors", []):
                            validation_errors.append(err)
                        formula_valid = False
                    for warn in svc_validation.get("warnings", []):
                        if warn not in validation_warnings:
                            validation_warnings.append(warn)
                except (FormulaError, DesignMatrixError) as e:
                    validation_errors.append(f"Formula validation error: {str(e)}")
                    formula_valid = False

            # Generate alternative suggestions
            suggestions = []
            batch_vars = [
                col
                for col in categorical_vars
                if col.lower() in ["batch", "sample", "donor", "patient", "subject"]
                or (
                    variable_analysis[col]["unique_values"] > 2
                    and variable_analysis[col]["unique_values"] <= 6
                )
            ]

            if batch_vars and not covariates:
                primary_batch = batch_vars[0]
                suggestions.append(
                    {
                        "formula": f"~{groupby} + {primary_batch}",
                        "description": f"Add batch correction for {primary_batch}",
                        "min_samples": 8,
                    }
                )

            if len(batch_vars) > 1 or continuous_vars:
                extra_covs = batch_vars[:2] + continuous_vars[:1]
                all_terms = [groupby] + extra_covs
                suggestions.append(
                    {
                        "formula": f"~{' + '.join(all_terms)}",
                        "description": f"Multi-factor model with {len(extra_covs)} covariates",
                        "min_samples": max(12, len(all_terms) * 4),
                    }
                )

            # Store formula in modality
            adata.uns["constructed_formula"] = {
                "formula": formula,
                "main_variable": groupby,
                "covariates": covariates,
                "include_interactions": include_interaction,
                "validated": formula_valid,
            }
            data_manager.store_modality(
                name=modality_name,
                adata=adata,
                step_summary=f"Formula suggested and validated: {formula}",
            )

            # Create IR
            from lobster.core.provenance import AnalysisStep

            ir = AnalysisStep(
                operation="de_formula.suggest_and_validate",
                tool_name="suggest_de_formula",
                description=f"Analyzed metadata and constructed formula: {formula}",
                library="lobster",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "covariates": covariates,
                    "include_interaction": include_interaction,
                    "formula": formula,
                },
                parameter_schema={},
                code_template=(
                    "# Formula: {{ formula | string | repr }}\n"
                    "# Main variable: {{ groupby | string | repr }}\n"
                    "# Covariates: {{ covariates | string | repr }}"
                ),
                imports=[],
            )

            data_manager.log_tool_usage(
                tool_name="suggest_de_formula",
                parameters={
                    "modality_name": modality_name,
                    "groupby": groupby,
                    "covariates": covariates,
                    "formula": formula,
                },
                description=f"Suggested and validated formula: {formula}",
                ir=ir,
                agent=agent_name,
            )

            # Build response
            response = f"## DE Formula Analysis for '{modality_name}'\n\n"

            response += "**Metadata Summary:**\n"
            response += f"- Samples: {n_samples}\n"
            response += f"- Categorical variables: {len(categorical_vars)}\n"
            response += f"- Continuous variables: {len(continuous_vars)}\n\n"

            response += "**Key Variables:**\n"
            for col, info in list(variable_analysis.items())[:6]:
                if col in categorical_vars + continuous_vars:
                    response += (
                        f"- **{col}**: {info['type']}, {info['unique_values']} levels"
                    )
                    if info["type"] == "categorical":
                        response += f" ({', '.join(map(str, info['sample_values']))})"
                    response += "\n"
            response += "\n"

            response += f"**Constructed Formula**: `{formula}`\n"
            response += f"- Main variable: {groupby}\n"
            if covariates:
                response += f"- Covariates: {', '.join(covariates)}\n"
            if include_interaction:
                response += f"- Interaction: {groupby} x {covariates[0] if covariates else 'N/A'}\n"
            response += "\n"

            if design_result:
                response += "**Design Matrix:**\n"
                response += f"- Dimensions: {design_result['design_matrix'].shape[0]} samples x {design_result['design_matrix'].shape[1]} coefficients\n"
                response += f"- Matrix rank: {design_result['rank']} (full rank: {'Yes' if design_result['rank'] == design_result['n_coefficients'] else 'WARNING'})\n\n"

            response += "**Validation:**\n"
            if formula_valid:
                response += "- Status: PASSED\n"
            else:
                response += "- Status: FAILED\n"

            if validation_errors:
                for err in validation_errors:
                    response += f"- ERROR: {err}\n"
            if validation_warnings:
                for warn in validation_warnings:
                    response += f"- WARNING: {warn}\n"

            response += "\n**Replicate Counts:**\n"
            for group, count in replicate_validation["group_counts"].items():
                status = (
                    "OK"
                    if count >= 4
                    else "LOW POWER" if count >= 3 else "INSUFFICIENT"
                )
                response += f"- {group}: {count} replicates ({status})\n"
            response += _replicate_summary_text(replicate_validation)

            if suggestions:
                response += "\n**Alternative Formulas:**\n"
                for i, s in enumerate(suggestions, 1):
                    response += f"  {i}. `{s['formula']}` - {s['description']} (min {s['min_samples']} samples)\n"

            if formula_valid:
                response += f"\n**Next step**: Use `run_de_with_formula` with formula='{formula}' to execute the analysis."
            else:
                response += "\n**Action needed**: Address validation errors before running DE analysis."

            return response

        except Exception as e:
            logger.error(f"Error in suggest_de_formula: {e}")
            return f"Error analyzing design for formula suggestions: {str(e)}"

    suggest_de_formula.metadata = {"categories": ["ANALYZE"], "provenance": True}
    suggest_de_formula.tags = ["ANALYZE"]

    @tool
    def construct_de_formula_interactive(
        pseudobulk_modality: str,
        main_variable: str,
        covariates: Optional[List[str]] = None,
        include_interactions: bool = False,
        validate_design: bool = True,
    ) -> str:
        """
        DEPRECATED: Use suggest_de_formula instead.

        This tool is deprecated and will be removed in a future version.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            main_variable: Primary variable of interest
            covariates: List of covariate variables
            include_interactions: Whether to include interaction terms
            validate_design: Whether to validate the design
        """
        logger.warning(
            "construct_de_formula_interactive is deprecated. Use suggest_de_formula instead."
        )
        return "DEPRECATED: Use suggest_de_formula for formula construction and validation."

    construct_de_formula_interactive.metadata = {
        "categories": ["UTILITY"],
        "provenance": False,
    }
    construct_de_formula_interactive.tags = ["UTILITY"]

    @tool
    def run_de_with_formula(
        pseudobulk_modality: str,
        formula: Optional[str] = None,
        contrast: Optional[List[str]] = None,
        reference_levels: Optional[Dict[str, str]] = None,
        alpha: float = 0.05,
        lfc_threshold: float = 0.0,
        save_results: bool = True,
    ) -> str:
        """
        Run DE with a custom formula for complex multi-factor designs (covariates, interactions, batch correction).

        Uses pyDESeq2 on finite non-negative whole-number counts: layer 'counts'
        when present, otherwise the expression matrix. Other values are refused.
        This numeric check does not establish raw-count provenance.
        For simple 2-group comparisons, use run_differential_expression instead.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            formula: R-style formula (uses stored formula if None)
            contrast: Contrast specification [factor, level1, level2]
            reference_levels: Reference levels for categorical variables
            alpha: Significance threshold for adjusted p-values
            lfc_threshold: Log fold change threshold
            save_results: Whether to save results to files
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            # Both save targets are checked before any analysis or store.
            if save_results:
                formula_csv = (
                    data_manager.exports_dir
                    / f"{pseudobulk_modality}_formula_de_results.csv"
                )
                assert_inside_workspace(data_manager.workspace_path, formula_csv)
                assert_export_allowed(data_manager.workspace_path, formula_csv)
                data_manager._assert_save_allowed(
                    f"{pseudobulk_modality}_with_formula_results.h5ad"
                )

            count_layer, refusal = _deseq2_count_source(adata)
            if refusal:
                return refusal
            count_source = "layers['counts']" if count_layer else "X"

            # Use stored formula if none provided
            if formula is None:
                if "constructed_formula" in adata.uns:
                    formula = adata.uns["constructed_formula"]["formula"]
                    stored_info = adata.uns["constructed_formula"]
                    response_prefix = (
                        "Using stored formula from interactive construction:\n"
                    )
                else:
                    return "No formula provided and no stored formula found. Use `suggest_de_formula` first or provide a formula."
            else:
                response_prefix = "Using provided formula:\n"
                stored_info = None

            # Auto-detect contrast if not provided
            if contrast is None and stored_info:
                main_var = stored_info["main_variable"]
                levels = list(adata.obs[main_var].unique())
                if len(levels) == 2:
                    contrast = [main_var, str(levels[1]), str(levels[0])]
                    response_prefix += (
                        f"Auto-detected contrast: {contrast[1]} vs {contrast[2]}\n"
                    )
                else:
                    return f"Multiple levels found for {main_var}: {levels}. Please specify contrast as [factor, level1, level2]."
            elif contrast is None:
                return "No contrast specified. Please provide contrast as [factor, level1, level2]."

            logger.info(
                f"Running DE analysis on '{pseudobulk_modality}' with formula: {formula}"
            )

            # SCIENTIFIC FIX: Validate design with min_replicates=3
            design_validation = bulk_rnaseq_service.validate_experimental_design(
                metadata=adata.obs, formula=formula, min_replicates=3
            )

            if not design_validation["valid"]:
                error_msgs = "; ".join(design_validation["errors"])
                return f"**Invalid experimental design**: {error_msgs}\n\nUse `suggest_de_formula` to debug the design."

            # Create design matrix
            condition_col = contrast[0]
            reference_condition = (
                reference_levels.get(condition_col) if reference_levels else None
            )

            design_result = bulk_rnaseq_service.create_formula_design(
                metadata=adata.obs,
                condition_col=condition_col,
                reference_condition=reference_condition,
            )

            # Store design information
            adata.uns["de_formula_design"] = {
                "formula": formula,
                "contrast": contrast,
                "design_matrix_info": design_result,
                "validation_results": design_validation,
                "reference_levels": reference_levels,
            }

            # Run pyDESeq2 analysis
            results_df, analysis_stats = (
                bulk_rnaseq_service.run_pydeseq2_from_pseudobulk(
                    pseudobulk_adata=adata,
                    formula=formula,
                    contrast=contrast,
                    count_layer=count_layer,
                    alpha=alpha,
                    shrink_lfc=True,
                    n_cpus=1,
                )
            )

            # Filter by LFC threshold if specified
            if lfc_threshold > 0:
                significant_mask = (results_df["padj"] < alpha) & (
                    abs(results_df["log2FoldChange"]) >= lfc_threshold
                )
                n_lfc_filtered = significant_mask.sum()
            else:
                n_lfc_filtered = analysis_stats["n_significant_genes"]

            # Store results in modality
            contrast_name = f"{contrast[0]}_{contrast[1]}_vs_{contrast[2]}"
            adata.uns[f"de_results_formula_{contrast_name}"] = {
                "results_df": results_df,
                "analysis_stats": analysis_stats,
                "parameters": {
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "lfc_threshold": lfc_threshold,
                    "reference_levels": reference_levels,
                    "count_layer": count_layer,
                    "count_source": count_source,
                },
            }

            adata.uns["de_current_results_key"] = f"de_results_formula_{contrast_name}"
            # Update modality
            data_manager.store_modality(
                name=pseudobulk_modality,
                adata=adata,
                step_summary=f"DE with formula complete: {contrast_name}",
            )

            # Save results if requested
            if save_results:
                results_path = (
                    data_manager.exports_dir
                    / f"{pseudobulk_modality}_formula_de_results.csv"
                )
                assert_export_allowed(data_manager.workspace_path, results_path)
                results_df.to_csv(results_path)

                modality_path = f"{pseudobulk_modality}_with_formula_results.h5ad"
                data_manager.save_modality(pseudobulk_modality, modality_path)

            # Format response
            response = "## Differential Expression Analysis Complete\n\n"
            response += response_prefix
            response += f"**Formula**: `{formula}`\n"
            response += f"**Count matrix**: {count_source} (whole-number counts)\n"
            response += (
                f"**Contrast**: {contrast[1]} vs {contrast[2]} (in {contrast[0]})\n\n"
            )

            response += "**Results Summary**:\n"
            response += f"- Genes tested: {analysis_stats['n_genes_tested']:,}\n"
            response += f"- Significant genes (FDR < {alpha}): {analysis_stats['n_significant_genes']:,}\n"
            if lfc_threshold > 0:
                response += (
                    f"- Significant + |LFC| >= {lfc_threshold}: {n_lfc_filtered:,}\n"
                )
            response += f"- Upregulated ({contrast[1]} > {contrast[2]}): {analysis_stats['n_upregulated']:,}\n"
            response += f"- Downregulated ({contrast[1]} < {contrast[2]}): {analysis_stats['n_downregulated']:,}\n\n"

            response += "**Top Differentially Expressed Genes**:\n"
            response += "**Most Upregulated**:\n"
            for gene in analysis_stats["top_upregulated"][:5]:
                gene_data = results_df.loc[gene]
                response += f"- {gene}: LFC = {gene_data['log2FoldChange']:.2f}, FDR = {gene_data['padj']:.2e}\n"

            response += "\n**Most Downregulated**:\n"
            for gene in analysis_stats["top_downregulated"][:5]:
                gene_data = results_df.loc[gene]
                response += f"- {gene}: LFC = {gene_data['log2FoldChange']:.2f}, FDR = {gene_data['padj']:.2e}\n"

            response += "\n**Results Storage**:\n"
            response += (
                f"- Stored in: adata.uns['de_results_formula_{contrast_name}']\n"
            )
            if save_results:
                response += f"- CSV file: {results_path}\n"
                response += f"- H5AD file: {modality_path}\n"

            response += "\n**Next steps**: Use `iterate_de_analysis` to try different formulas or `compare_de_iterations` to compare results."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="run_de_with_formula",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "formula": formula,
                    "contrast": contrast,
                    "alpha": alpha,
                    "lfc_threshold": lfc_threshold,
                    "count_layer": count_layer,
                    "count_source": count_source,
                },
                description=(
                    f"Formula-based DE analysis: {analysis_stats['n_significant_genes']} significant genes; "
                    f"whole-number counts from {count_source}"
                ),
                ir=_formula_de_ir(
                    formula,
                    contrast,
                    reference_levels,
                    alpha,
                    lfc_threshold,
                    count_layer=count_layer,
                ),
                agent=agent_name,
            )

            return response

        except WorkspaceExportError:
            # A save target is outside the workspace: nothing was saved.
            return WORKSPACE_EXPORT_MESSAGE
        except ManagedPathError:
            # A save target is inside the managed figure folder: nothing was saved.
            return MANAGED_EXPORT_MESSAGE
        except Exception as e:
            logger.error(f"Error in formula-based DE analysis: {e}")
            return f"Error running differential expression with formula: {str(e)}"

    run_de_with_formula.metadata = {"categories": ["ANALYZE"], "provenance": True}
    run_de_with_formula.tags = ["ANALYZE"]

    # NOTE: run_differential_expression_analysis has been merged into run_differential_expression above.
    # The merged tool auto-detects pseudobulk vs direct bulk data.

    @tool
    def iterate_de_analysis(
        pseudobulk_modality: str,
        new_formula: Optional[str] = None,
        new_contrast: Optional[List[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        compare_to_previous: bool = True,
        iteration_name: Optional[str] = None,
    ) -> str:
        """
        Support iterative analysis with formula/filter changes.

        Enables trying different formulas or filters, tracking iterations,
        and comparing results between analyses.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            new_formula: New formula to try (if None, modify existing)
            new_contrast: New contrast to test
            filter_criteria: Additional filtering criteria (e.g., {'min_lfc': 0.5})
            compare_to_previous: Whether to compare with previous iteration
            iteration_name: Custom name for this iteration
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            iteration_tracker = _read_de_iterations(adata)
            current_iter = _next_de_iteration(adata, iteration_tracker)

            # Determine iteration name
            if iteration_name is None:
                iteration_name = f"iteration_{current_iter}"

            # Get previous results for comparison
            previous_results = None
            previous_iteration = None
            if compare_to_previous and iteration_tracker["iterations"]:
                previous_iteration = iteration_tracker["iterations"][-1]
                prev_key = _iteration_results_key(previous_iteration)
                if prev_key in adata.uns:
                    previous_results = adata.uns[prev_key]["results_df"]

            # Use existing formula/contrast if not provided
            if new_formula is None or new_contrast is None:
                if "de_formula_design" in adata.uns:
                    if new_formula is None:
                        new_formula = adata.uns["de_formula_design"]["formula"]
                    if new_contrast is None:
                        new_contrast = adata.uns["de_formula_design"]["contrast"]
                else:
                    return "No previous formula/contrast found and none provided. Run a DE analysis first."

            logger.info(
                f"Starting DE iteration '{iteration_name}' on '{pseudobulk_modality}'"
            )

            contrast_name = f"{new_contrast[0]}_{new_contrast[1]}_vs_{new_contrast[2]}"
            current_key = f"de_results_formula_{contrast_name}"
            # Identity of any earlier result for this contrast: the run below
            # must replace it, or it did not succeed (e.g. a rejected design
            # returns a message rather than raising).
            stale_results = adata.uns.get(current_key, {}).get("results_df")

            # Run the analysis with new parameters
            # run_de_with_formula is a LangChain tool object, so it must be
            # invoked rather than called. It logs its own provenance.
            run_result = run_de_with_formula.invoke(
                {
                    "pseudobulk_modality": pseudobulk_modality,
                    "formula": new_formula,
                    "contrast": new_contrast,
                    "alpha": 0.05,
                    "lfc_threshold": (
                        filter_criteria.get("min_lfc", 0.0) if filter_criteria else 0.0
                    ),
                    "save_results": False,
                }
            )

            if "Error" in run_result:
                return f"Error in iteration '{iteration_name}': {run_result}"

            new_results = adata.uns.get(current_key, {}).get("results_df")
            if new_results is None or new_results is stale_results:
                return (
                    f"Error in iteration '{iteration_name}': the DE run produced no "
                    f"new results, so no iteration was recorded.\n\n{run_result}"
                )

            # Snapshot this iteration's results under its own key. The formula
            # result key depends only on the contrast, so the next iteration of
            # the same contrast would overwrite it and comparisons would then
            # compare the latest result with itself.
            results_key = f"de_iteration_results_{current_iter}"
            current_results = adata.uns[current_key]["results_df"]
            current_stats = adata.uns[current_key]["analysis_stats"]

            # Store iteration information
            iteration_info = {
                "name": iteration_name,
                "formula": new_formula,
                "contrast": new_contrast,
                "contrast_name": contrast_name,
                "results_key": results_key,
                "n_significant": current_stats["n_significant_genes"],
                "timestamp": pd.Timestamp.now().isoformat(),
                "filter_criteria": filter_criteria or {},
            }

            # Compare with previous if requested
            comparison_results = None
            if compare_to_previous and previous_results is not None:
                current_sig = set(current_results[current_results["padj"] < 0.05].index)
                previous_sig = set(
                    previous_results[previous_results["padj"] < 0.05].index
                )

                overlap = len(current_sig & previous_sig)
                current_only = len(current_sig - previous_sig)
                previous_only = len(previous_sig - current_sig)

                common_genes = list(current_sig & previous_sig)
                if len(common_genes) > 3:
                    current_lfc = current_results.loc[common_genes, "log2FoldChange"]
                    previous_lfc = previous_results.loc[common_genes, "log2FoldChange"]
                    correlation = current_lfc.corr(previous_lfc)
                else:
                    correlation = None

                comparison_results = {
                    "overlap_genes": overlap,
                    "current_only": current_only,
                    "previous_only": previous_only,
                    "correlation": correlation,
                }

                iteration_info["comparison"] = comparison_results

            # Update iteration tracking
            _record_de_iteration(adata, iteration_info, current_iter)
            iteration_tracker = _read_de_iterations(adata)

            # Update modality
            data_manager.store_modality(
                name=pseudobulk_modality,
                adata=adata,
                step_summary=f"DE iteration '{iteration_name}' complete",
            )

            # Format response
            response = f"## DE Analysis Iteration '{iteration_name}' Complete\n\n"
            response += f"**Formula**: `{new_formula}`\n"
            response += f"**Contrast**: {new_contrast[1]} vs {new_contrast[2]} (in {new_contrast[0]})\n\n"

            response += "**Current Results**:\n"
            response += (
                f"- Significant genes: {current_stats['n_significant_genes']:,}\n"
            )
            response += f"- Upregulated: {current_stats['n_upregulated']:,}\n"
            response += f"- Downregulated: {current_stats['n_downregulated']:,}\n"

            if comparison_results:
                response += "\n**Comparison with Previous Iteration**:\n"
                response += f"- Overlapping significant genes: {comparison_results['overlap_genes']:,}\n"
                response += (
                    f"- New in current: {comparison_results['current_only']:,}\n"
                )
                response += (
                    f"- Lost from previous: {comparison_results['previous_only']:,}\n"
                )
                if comparison_results["correlation"] is not None:
                    response += f"- Fold change correlation: {comparison_results['correlation']:.3f}\n"

            response += "\n**Iteration Summary**:\n"
            response += f"- Total iterations: {len(iteration_tracker['iterations'])}\n"
            response += f"- Current iteration: {current_iter}\n"

            response += f"\n**Results stored in**: adata.uns['de_results_formula_{contrast_name}']\n"
            response += "**Iteration tracking**: adata.uns['de_iterations_json']\n"

            response += "\n**Next steps**: Use `compare_de_iterations` to compare all iterations or continue iterating with different parameters."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="iterate_de_analysis",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "iteration_name": iteration_name,
                    "formula": new_formula,
                    "contrast": new_contrast,
                    "compare_to_previous": compare_to_previous,
                },
                description=f"DE iteration {current_iter}: {current_stats['n_significant_genes']} significant genes",
                ir=_iteration_record_ir(iteration_info, current_iter),
                agent=agent_name,
            )

            return response

        except Exception as e:
            logger.error(f"Error in DE iteration: {e}")
            return f"Error in DE analysis iteration: {str(e)}"

    iterate_de_analysis.metadata = {"categories": ["ANALYZE"], "provenance": True}
    iterate_de_analysis.tags = ["ANALYZE"]

    @tool
    def compare_de_iterations(
        pseudobulk_modality: str,
        iteration_1: Optional[str] = None,
        iteration_2: Optional[str] = None,
        show_overlap: bool = True,
        show_unique: bool = True,
        save_comparison: bool = True,
    ) -> str:
        """
        Compare results between different DE analysis iterations.

        Shows overlapping and unique DEGs, correlation of fold changes, and helps
        users understand impact of formula changes.

        Args:
            pseudobulk_modality: Name of pseudobulk modality
            iteration_1: Name of first iteration (latest if None)
            iteration_2: Name of second iteration (second latest if None)
            show_overlap: Whether to show overlapping genes
            show_unique: Whether to show unique genes per iteration
            save_comparison: Whether to save comparison results
        """
        try:
            # Validate modality exists
            if pseudobulk_modality not in data_manager.list_modalities():
                return f"Modality '{pseudobulk_modality}' not found. Available: {data_manager.list_modalities()}"

            # Get the pseudobulk data
            adata = data_manager.get_modality(pseudobulk_modality)

            iteration_tracker = _read_de_iterations(adata)
            iterations = iteration_tracker["iterations"]
            if not iterations:
                return "No iteration tracking found. Run `iterate_de_analysis` first to create iterations."

            if len(iterations) < 2:
                return f"Only {len(iterations)} iteration(s) available. Need at least 2 for comparison."

            # Select iterations to compare
            if iteration_1 is None:
                iter1_info = iterations[-1]
            else:
                iter1_info = next(
                    (i for i in iterations if i["name"] == iteration_1), None
                )
                if not iter1_info:
                    available = [i["name"] for i in iterations]
                    return (
                        f"Iteration '{iteration_1}' not found. Available: {available}"
                    )

            if iteration_2 is None:
                iter2_info = iterations[-2] if len(iterations) >= 2 else iterations[0]
            else:
                iter2_info = next(
                    (i for i in iterations if i["name"] == iteration_2), None
                )
                if not iter2_info:
                    available = [i["name"] for i in iterations]
                    return (
                        f"Iteration '{iteration_2}' not found. Available: {available}"
                    )

            # Get results DataFrames
            iter1_key = _iteration_results_key(iter1_info)
            iter2_key = _iteration_results_key(iter2_info)

            if iter1_key not in adata.uns or iter2_key not in adata.uns:
                return "Results not found for one or both iterations."

            results1 = adata.uns[iter1_key]["results_df"]
            results2 = adata.uns[iter2_key]["results_df"]

            # Get significant genes (FDR < 0.05)
            sig1 = set(results1[results1["padj"] < 0.05].index)
            sig2 = set(results2[results2["padj"] < 0.05].index)

            # Calculate overlaps
            overlap = sig1 & sig2
            unique1 = sig1 - sig2
            unique2 = sig2 - sig1

            # Calculate fold change correlation for overlapping genes
            if len(overlap) > 3:
                overlap_genes = list(overlap)
                lfc1 = results1.loc[overlap_genes, "log2FoldChange"]
                lfc2 = results2.loc[overlap_genes, "log2FoldChange"]
                correlation = lfc1.corr(lfc2)
            else:
                correlation = None

            # Format response
            response = "## DE Iteration Comparison\n\n"
            response += "**Comparing:**\n"
            response += (
                f"- Iteration 1: '{iter1_info['name']}' - {iter1_info['formula']}\n"
            )
            response += (
                f"- Iteration 2: '{iter2_info['name']}' - {iter2_info['formula']}\n\n"
            )

            response += "**Results Summary:**\n"
            response += f"- Iteration 1 significant genes: {len(sig1):,}\n"
            response += f"- Iteration 2 significant genes: {len(sig2):,}\n"
            response += f"- Overlapping genes: {len(overlap):,} ({len(overlap) / max(len(sig1), len(sig2)) * 100:.1f}%)\n"
            response += f"- Unique to iteration 1: {len(unique1):,}\n"
            response += f"- Unique to iteration 2: {len(unique2):,}\n"

            if correlation is not None:
                response += f"- Fold change correlation: {correlation:.3f}\n"

            if show_overlap and len(overlap) > 0:
                response += "\n**Top Overlapping Genes:**\n"
                overlap_df = results1.loc[list(overlap)]
                overlap_df = overlap_df.reindex(overlap_df["padj"].sort_values().index)

                for gene in list(overlap_df.index)[:10]:
                    lfc1 = results1.loc[gene, "log2FoldChange"]
                    lfc2 = results2.loc[gene, "log2FoldChange"]
                    response += f"- {gene}: LFC1={lfc1:.2f}, LFC2={lfc2:.2f}\n"

            if show_unique and (len(unique1) > 0 or len(unique2) > 0):
                response += "\n**Unique Significant Genes:**\n"

                if len(unique1) > 0:
                    response += (
                        f"**Only in '{iter1_info['name']}'** ({len(unique1)} genes):\n"
                    )
                    unique1_sorted = results1.loc[list(unique1)].sort_values("padj")
                    for gene in unique1_sorted.index[:8]:
                        lfc = results1.loc[gene, "log2FoldChange"]
                        fdr = results1.loc[gene, "padj"]
                        response += f"- {gene}: LFC={lfc:.2f}, FDR={fdr:.2e}\n"

                if len(unique2) > 0:
                    response += f"\n**Only in '{iter2_info['name']}'** ({len(unique2)} genes):\n"
                    unique2_sorted = results2.loc[list(unique2)].sort_values("padj")
                    for gene in unique2_sorted.index[:8]:
                        lfc = results2.loc[gene, "log2FoldChange"]
                        fdr = results2.loc[gene, "padj"]
                        response += f"- {gene}: LFC={lfc:.2f}, FDR={fdr:.2e}\n"

            # Analysis interpretation
            response += "\n**Interpretation:**\n"
            if correlation is not None:
                if correlation > 0.8:
                    response += f"- High correlation ({correlation:.3f}) suggests similar biological effects\n"
                elif correlation > 0.5:
                    response += f"- Moderate correlation ({correlation:.3f}) - some consistency but notable differences\n"
                else:
                    response += f"- Low correlation ({correlation:.3f}) - formulas capture different effects\n"

            overlap_percent = len(overlap) / max(len(sig1), len(sig2)) * 100
            if overlap_percent > 70:
                response += f"- High overlap ({overlap_percent:.1f}%) - formulas yield similar gene sets\n"
            elif overlap_percent > 40:
                response += f"- Moderate overlap ({overlap_percent:.1f}%) - some formula-specific effects\n"
            else:
                response += f"- Low overlap ({overlap_percent:.1f}%) - formulas capture different biology\n"

            # Save comparison if requested
            if save_comparison:
                comparison_data = {
                    "iteration_1": iter1_info,
                    "iteration_2": iter2_info,
                    "overlap_genes": list(overlap),
                    "unique_to_1": list(unique1),
                    "unique_to_2": list(unique2),
                    "correlation": correlation,
                    "summary_stats": {
                        "n_sig_1": len(sig1),
                        "n_sig_2": len(sig2),
                        "n_overlap": len(overlap),
                        "overlap_percent": overlap_percent,
                    },
                }

                comparison_key = (
                    f"iteration_comparison_{iter1_info['name']}_vs_{iter2_info['name']}"
                )
                if "iteration_comparisons" not in adata.uns:
                    adata.uns["iteration_comparisons"] = {}
                adata.uns["iteration_comparisons"][comparison_key] = comparison_data

                data_manager.store_modality(
                    name=pseudobulk_modality,
                    adata=adata,
                    step_summary=f"Compared iterations: {iter1_info['name']} vs {iter2_info['name']}",
                )
                response += f"\n**Comparison saved**: adata.uns['iteration_comparisons']['{comparison_key}']\n"

            response += "\n**Next steps**: Choose the most appropriate formula based on biological interpretation and statistical robustness."

            # Log the operation
            data_manager.log_tool_usage(
                tool_name="compare_de_iterations",
                parameters={
                    "pseudobulk_modality": pseudobulk_modality,
                    "iteration_1": iter1_info["name"],
                    "iteration_2": iter2_info["name"],
                    "show_overlap": show_overlap,
                    "show_unique": show_unique,
                },
                description=f"Compared iterations: {len(overlap)} overlapping, {len(unique1)}+{len(unique2)} unique genes",
                ir=_iteration_comparison_ir(iter1_info, iter2_info, save_comparison),
                agent=agent_name,
            )

            return response

        except Exception as e:
            logger.error(f"Error comparing DE iterations: {e}")
            return f"Error comparing DE iterations: {str(e)}"

    compare_de_iterations.metadata = {"categories": ["ANALYZE"], "provenance": True}
    compare_de_iterations.tags = ["ANALYZE"]

    @tool
    def run_pathway_enrichment(
        gene_list: List[str],
        analysis_type: str = "GO",
        modality_name: str = None,
        save_result: bool = True,
        de_results_key: str = None,
    ) -> str:
        """
        Run pathway enrichment analysis on gene lists from differential expression results.

        Args:
            gene_list: List of genes for enrichment analysis
            analysis_type: Type of analysis ("GO" or "KEGG")
            modality_name: Optional modality name to extract genes from DE results
            save_result: Whether to save enrichment results
            de_results_key: Explicit DE result choice if current identity is unavailable
        """
        try:
            # If modality name provided, extract significant genes from it
            if modality_name and modality_name in data_manager.list_modalities():
                adata = data_manager.get_modality(modality_name)

                # A supplied gene list can be used when this modality has no DE.
                # Ambiguous or stale result identity must never select by order.
                if (
                    _de_result_keys(adata)
                    or de_results_key is not None
                    or "de_current_results_key" in adata.uns
                ):
                    selected_key = _resolve_de_results_key(adata, de_results_key)
                    de_results = adata.uns[selected_key]
                    if isinstance(de_results, dict) and "results_df" in de_results:
                        de_df = de_results["results_df"]
                        if "padj" in de_df.columns:
                            significant_genes = de_df[
                                de_df["padj"] < 0.05
                            ].index.tolist()
                            if significant_genes:
                                gene_list = significant_genes[:500]
                                logger.info(
                                    f"Extracted {len(gene_list)} significant genes from {modality_name}"
                                )

            if not gene_list or len(gene_list) == 0:
                return "No genes provided for enrichment analysis. Please provide a gene list or run differential expression analysis first."

            logger.info(f"Running pathway enrichment on {len(gene_list)} genes")

            # Use bulk service for pathway enrichment
            enrichment_df, enrichment_stats, ir = (
                bulk_rnaseq_service.run_pathway_enrichment(
                    gene_list=gene_list, analysis_type=analysis_type
                )
            )

            # Log the operation with IR for provenance tracking
            data_manager.log_tool_usage(
                tool_name="run_pathway_enrichment",
                parameters={
                    "gene_list_size": len(gene_list),
                    "analysis_type": analysis_type,
                    "modality_name": modality_name,
                },
                description=f"{analysis_type} enrichment: {enrichment_stats['n_significant_terms']} significant terms",
                ir=ir,
                agent=agent_name,
            )

            # Format response
            response = f"## {analysis_type} Pathway Enrichment Analysis Complete\n\n"
            response += "**Enrichment Results:**\n"
            response += f"- Genes analyzed: {enrichment_stats['n_genes_input']:,}\n"
            response += f"- Database: {enrichment_stats['enrichment_database']}\n"
            response += f"- Terms found: {enrichment_stats['n_terms_total']}\n"
            response += f"- Significant terms (p.adj < 0.05): {enrichment_stats['n_significant_terms']}\n\n"

            response += "**Top Enriched Pathways:**\n"
            for term in enrichment_stats["top_terms"][:8]:
                response += f"- {term}\n"

            if len(enrichment_stats["top_terms"]) > 8:
                remaining = len(enrichment_stats["top_terms"]) - 8
                response += f"... and {remaining} more pathways\n"

            response += "\nPathway enrichment reveals biological processes and pathways associated with differential expression."

            analysis_results["details"]["pathway_enrichment"] = response
            return response

        except BulkRNASeqError as e:
            logger.error(f"Error in pathway enrichment: {e}")
            return f"Error running pathway enrichment: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pathway enrichment: {e}")
            return f"Unexpected error: {str(e)}"

    run_pathway_enrichment.metadata = {"categories": ["ANALYZE"], "provenance": True}
    run_pathway_enrichment.tags = ["ANALYZE"]

    # -------------------------
    # RESULT TOOLS
    # -------------------------
    @tool
    def filter_de_results(
        modality_name: str,
        de_results_key: str = None,
        padj_threshold: float = 0.05,
        lfc_threshold: float = 1.0,
        min_base_mean: float = 10.0,
        save_result: bool = True,
    ) -> str:
        """Filter differential expression results by significance and effect size thresholds.

        Filters DE results stored in adata.uns by adjusted p-value, log fold change,
        and base mean expression. Handles column name variations across DE methods
        (pyDESeq2, simple DE, etc.).

        Args:
            modality_name: Name of modality containing DE results
            de_results_key: Specific DE results key in adata.uns (auto-detected if None)
            padj_threshold: Maximum adjusted p-value (default: 0.05)
            lfc_threshold: Minimum absolute log2 fold change (default: 1.0)
            min_base_mean: Minimum base mean expression (default: 10.0)
            save_result: Whether to store filtered results back in adata.uns
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            de_results_key = _resolve_de_results_key(adata, de_results_key)
            de_data = adata.uns[de_results_key]
            if not isinstance(de_data, dict) or "results_df" not in de_data:
                return f"Invalid DE results format at '{de_results_key}'. Expected dict with 'results_df'."

            results_df = de_data["results_df"]

            # Handle column name variations
            # Adjusted p-value columns
            padj_col = None
            for col_name in ["padj", "pvalue_adj", "FDR", "p_adj", "adjusted_pvalue"]:
                if col_name in results_df.columns:
                    padj_col = col_name
                    break

            # Log fold change columns
            lfc_col = None
            for col_name in ["log2FoldChange", "logFC", "mean_log2FC", "log2FC", "lfc"]:
                if col_name in results_df.columns:
                    lfc_col = col_name
                    break

            # Base mean columns
            basemean_col = None
            for col_name in ["baseMean", "base_mean", "AveExpr", "mean_expression"]:
                if col_name in results_df.columns:
                    basemean_col = col_name
                    break

            if padj_col is None:
                return f"No adjusted p-value column found. Available columns: {list(results_df.columns)}"
            if lfc_col is None:
                return f"No log fold change column found. Available columns: {list(results_df.columns)}"

            # Apply filters
            mask = results_df[padj_col] < padj_threshold
            mask = mask & (abs(results_df[lfc_col]) > lfc_threshold)
            if basemean_col is not None:
                mask = mask & (results_df[basemean_col] > min_base_mean)

            filtered_df = results_df[mask].copy()

            # Calculate up/down split
            n_up = (filtered_df[lfc_col] > 0).sum()
            n_down = (filtered_df[lfc_col] < 0).sum()

            # Store filtered results
            filtered_key = f"filtered_{de_results_key}"
            if save_result:
                adata.uns[filtered_key] = {
                    "results_df": filtered_df,
                    "filter_stats": {
                        "padj_threshold": padj_threshold,
                        "lfc_threshold": lfc_threshold,
                        "min_base_mean": min_base_mean,
                        "n_input": len(results_df),
                        "n_filtered": len(filtered_df),
                        "n_upregulated": int(n_up),
                        "n_downregulated": int(n_down),
                    },
                }

                data_manager.store_modality(
                    name=modality_name,
                    adata=adata,
                    step_summary=f"Filtered DE results: {len(filtered_df)} significant genes",
                )

            # Create IR
            from lobster.core.provenance import AnalysisStep

            ir = AnalysisStep(
                operation="de_results.filter",
                tool_name="filter_de_results",
                description=f"Filtered DE results: {len(filtered_df)}/{len(results_df)} genes pass thresholds",
                library="pandas",
                parameters={
                    "modality_name": modality_name,
                    "de_results_key": de_results_key,
                    "padj_col": padj_col,
                    "lfc_col": lfc_col,
                    "basemean_col": basemean_col,
                    "save_result": save_result,
                    "padj_threshold": padj_threshold,
                    "lfc_threshold": lfc_threshold,
                    "min_base_mean": min_base_mean,
                },
                parameter_schema={},
                code_template=(
                    "# Filter DE results\n"
                    "results_df = adata.uns[{{ de_results_key | repr }}]['results_df']\n"
                    "mask = (results_df[{{ padj_col | repr }}] < {{ padj_threshold | repr }}) & "
                    "(abs(results_df[{{ lfc_col | repr }}]) > {{ lfc_threshold | repr }})"
                    "{% if basemean_col %} & (results_df[{{ basemean_col | repr }}] > {{ min_base_mean | repr }}){% endif %}\n"
                    "filtered_df = results_df[mask].copy()\n"
                    "{% if save_result %}"
                    "adata.uns[{{ ('filtered_' ~ de_results_key) | repr }}] = {\n"
                    "    'results_df': filtered_df,\n"
                    "    'filter_stats': {\n"
                    "        'padj_threshold': {{ padj_threshold | repr }},\n"
                    "        'lfc_threshold': {{ lfc_threshold | repr }},\n"
                    "        'min_base_mean': {{ min_base_mean | repr }},\n"
                    "        'n_input': len(results_df), 'n_filtered': len(filtered_df),\n"
                    "        'n_upregulated': int((filtered_df[{{ lfc_col | repr }}] > 0).sum()),\n"
                    "        'n_downregulated': int((filtered_df[{{ lfc_col | repr }}] < 0).sum()),\n"
                    "    },\n"
                    "}\n"
                    "{% endif %}"
                ),
                imports=[],
            )

            data_manager.log_tool_usage(
                tool_name="filter_de_results",
                parameters={
                    "modality_name": modality_name,
                    "de_results_key": de_results_key,
                    "padj_threshold": padj_threshold,
                    "lfc_threshold": lfc_threshold,
                    "min_base_mean": min_base_mean,
                },
                description=f"Filtered DE results: {len(filtered_df)} significant genes",
                ir=ir,
                agent=agent_name,
            )

            # Get top genes for response
            sorted_df = filtered_df.sort_values(padj_col)
            top_genes = []
            for gene in sorted_df.index[:10]:
                gene_lfc = sorted_df.loc[gene, lfc_col]
                gene_padj = sorted_df.loc[gene, padj_col]
                direction = "UP" if gene_lfc > 0 else "DOWN"
                top_genes.append(
                    f"- {gene}: LFC={gene_lfc:.2f}, padj={gene_padj:.2e} ({direction})"
                )

            response = f"""## DE Results Filtered for '{modality_name}'

**Filter Thresholds:**
- Adjusted p-value < {padj_threshold}
- |log2FoldChange| > {lfc_threshold}
- Base mean > {min_base_mean}

**Results:**
- Input genes: {len(results_df):,}
- Passing filters: {len(filtered_df):,} ({len(filtered_df) / len(results_df) * 100:.1f}%)
- Upregulated: {n_up:,}
- Downregulated: {n_down:,}

**Top Significant Genes:**
{chr(10).join(top_genes)}

**Filtered results stored in**: adata.uns['{filtered_key}']"""

            response += "\n\nNext steps: Use export_de_results to save as CSV/Excel, or run_pathway_enrichment for pathway analysis."

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error filtering DE results: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error filtering DE results: {e}")
            return f"Unexpected error: {str(e)}"

    filter_de_results.metadata = {"categories": ["FILTER"], "provenance": True}
    filter_de_results.tags = ["FILTER"]

    @tool
    def export_de_results(
        modality_name: str,
        de_results_key: str = None,
        output_format: str = "csv",
        output_path: str = None,
        filtered_only: bool = False,
        sort_by: str = "padj",
    ) -> str:
        """Export DE results as publication-ready CSV or Excel file.

        Exports differential expression results with standardized column names
        suitable for publication supplementary tables.

        Args:
            modality_name: Name of modality containing DE results
            de_results_key: Specific DE results key in adata.uns (auto-detected if None)
            output_format: Export format ('csv' or 'xlsx')
            output_path: Custom output path (auto-generated if None)
            filtered_only: Whether to export only filtered results
            sort_by: Column to sort by (default: 'padj')
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            de_results_key = _resolve_de_results_key(adata, de_results_key)

            # If filtered_only, look for filtered version first
            actual_key = de_results_key
            if filtered_only:
                filtered_key = f"filtered_{de_results_key}"
                if filtered_key in adata.uns:
                    actual_key = filtered_key
                else:
                    return f"No filtered results found at '{filtered_key}'. Run filter_de_results first."

            if actual_key not in adata.uns:
                return f"DE results key '{actual_key}' not found."

            de_data = adata.uns[actual_key]
            if not isinstance(de_data, dict) or "results_df" not in de_data:
                return f"Invalid DE results format at '{actual_key}'."

            results_df = de_data["results_df"].copy()

            # Standardize column names for publication
            column_renames = {
                "log2FoldChange": "log2FoldChange",
                "logFC": "log2FoldChange",
                "mean_log2FC": "log2FoldChange",
                "padj": "padj",
                "pvalue_adj": "padj",
                "FDR": "padj",
                "p_adj": "padj",
                "pvalue": "pvalue",
                "baseMean": "baseMean",
                "base_mean": "baseMean",
                "AveExpr": "baseMean",
                "lfcSE": "lfcSE",
                "stat": "stat",
            }

            for old_name, new_name in column_renames.items():
                if old_name in results_df.columns and old_name != new_name:
                    results_df = results_df.rename(columns={old_name: new_name})

            # Sort results
            if sort_by in results_df.columns:
                results_df = results_df.sort_values(sort_by)

            # Ensure gene names are the index (named 'gene')
            if results_df.index.name is None:
                results_df.index.name = "gene"

            # Determine output path
            if output_path is None:
                export_dir = data_manager.exports_dir

                suffix = "_filtered" if filtered_only else ""
                clean_key = de_results_key.replace("de_results_", "")
                output_path = str(
                    export_dir / f"de_results_{clean_key}{suffix}.{output_format}"
                )

            # Export (relative paths are rooted in the session exports directory)
            output_path_obj = Path(output_path)
            if not output_path_obj.is_absolute():
                output_path_obj = data_manager.exports_dir / output_path_obj
            # The destination stays inside the workspace (or a listed export root).
            if outside_workspace(data_manager.workspace_path, output_path_obj):
                return WORKSPACE_EXPORT_MESSAGE
            # Figures are saved only through the trusted save step: check the
            # caller's (unresolved) destination, and every folder its recursive
            # creation can make, before anything is created.
            if export_refused(data_manager.workspace_path, output_path_obj):
                return MANAGED_EXPORT_MESSAGE
            try:
                assert_dirs_allowed(data_manager.workspace_path, output_path_obj.parent)
            except ManagedPathError:
                return MANAGED_EXPORT_MESSAGE
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_path_obj.resolve())
            if outside_workspace(data_manager.workspace_path, output_path):
                return WORKSPACE_EXPORT_MESSAGE
            if export_refused(
                data_manager.workspace_path, output_path_obj, output_path
            ):
                return MANAGED_EXPORT_MESSAGE

            if output_format == "xlsx":
                try:
                    results_df.to_excel(output_path, index=True)
                except ImportError:
                    return "Excel export requires openpyxl. Falling back to CSV."
            else:
                results_df.to_csv(output_path, index=True)

            # Create IR
            from lobster.core.provenance import AnalysisStep

            ir = AnalysisStep(
                operation="de_results.export",
                tool_name="export_de_results",
                description=f"Exported {len(results_df)} DE results to {output_format}",
                library="pandas",
                parameters={
                    "modality_name": modality_name,
                    "de_results_key": de_results_key,
                    "output_format": output_format,
                    "output_path": output_path,
                    "filtered_only": filtered_only,
                    "sort_by": sort_by,
                },
                parameter_schema={},
                code_template=(
                    "results_df = adata.uns[{{ de_results_key | string | repr }}]['results_df']\n"
                    "results_df = results_df.sort_values({{ sort_by | string | repr }})\n"
                    "results_df.to_{{ output_format }}({{ output_path | string | repr }}, index=True)"
                ),
                imports=["pandas"],
            )

            data_manager.log_tool_usage(
                tool_name="export_de_results",
                parameters={
                    "modality_name": modality_name,
                    "de_results_key": actual_key,
                    "output_format": output_format,
                    "output_path": output_path,
                    "filtered_only": filtered_only,
                },
                description=f"Exported {len(results_df)} DE results to {output_format}",
                ir=ir,
                agent=agent_name,
            )

            response = f"""## DE Results Exported

**Export Details:**
- Modality: {modality_name}
- Source: {actual_key}
- Genes exported: {len(results_df):,}
- Format: {output_format.upper()}
- Sorted by: {sort_by}
- Output: {output_path}

**Columns:** {', '.join(results_df.columns.tolist()[:8])}{'...' if len(results_df.columns) > 8 else ''}
**Filtered only:** {'Yes' if filtered_only else 'No'}"""

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error exporting DE results: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error exporting DE results: {e}")
            return f"Unexpected error: {str(e)}"

    export_de_results.metadata = {"categories": ["UTILITY"], "provenance": False}
    export_de_results.tags = ["UTILITY"]

    # -------------------------
    # BULK DE PIPELINE TOOLS
    # -------------------------
    @tool
    def run_bulk_de_direct(
        modality_name: str,
        group_key: str,
        group1: str,
        group2: str,
        min_replicates: int = 3,
        alpha: float = 0.05,
        save_result: bool = True,
    ) -> str:
        """One-shot DE for simple bulk RNA-seq comparisons. Use when bulk data is already imported
        and you want quick 2-group DE without pseudobulk aggregation or formula complexity.

        Args:
            modality_name: Name of bulk RNA-seq modality
            group_key: Column in obs containing group labels (e.g., 'condition')
            group1: First group (reference/control)
            group2: Second group (treatment/case)
            min_replicates: Minimum replicates per group (default: 3)
            alpha: Significance threshold for adjusted p-values (default: 0.05)
            save_result: Whether to save result modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)
            logger.info(
                f"Running bulk DE direct on '{modality_name}': "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            # Verify raw counts availability
            has_counts = "counts" in adata.layers if adata.layers else False
            has_raw = adata.raw is not None
            count_warning = ""
            if not has_counts and not has_raw:
                count_warning = (
                    "\n\n**WARNING**: No raw counts layer ('counts') or adata.raw found. "
                    "DESeq2 requires raw integer counts for accurate results."
                )

            # Validate group_key exists
            if group_key not in adata.obs.columns:
                available_cols = [
                    col for col in adata.obs.columns if not col.startswith("_")
                ]
                return f"Group key '{group_key}' not found in obs columns. Available: {available_cols}"

            # Validate groups exist
            available_groups = list(adata.obs[group_key].unique())
            available_groups_str = [str(g) for g in available_groups]
            group1 = str(group1)
            group2 = str(group2)

            if group1 not in available_groups_str:
                return f"Group '{group1}' not found in '{group_key}'. Available: {available_groups}"
            if group2 not in available_groups_str:
                return f"Group '{group2}' not found in '{group_key}'. Available: {available_groups}"

            # The result name carries the group labels: encoded and checked first.
            de_modality_name = _de_result_name(modality_name, group1, group2)

            # Ensure string type matching
            if group1 not in available_groups:
                adata.obs[group_key] = adata.obs[group_key].astype(str)

            # Check replicate counts
            replicate_validation = _validate_replicate_counts(
                adata.obs, group_key, min_replicates=min_replicates
            )

            if not replicate_validation["valid"]:
                error_msg = "; ".join(replicate_validation["errors"])
                return (
                    f"**Insufficient replicates**: {error_msg}\n\n"
                    f"Minimum {min_replicates} replicates per group required."
                )

            group1_count = (adata.obs[group_key] == group1).sum()
            group2_count = (adata.obs[group_key] == group2).sum()

            # Run DE via BulkRNASeqService
            adata_de, de_stats, ir_from_service = (
                bulk_rnaseq_service.run_differential_expression_analysis(
                    adata=adata,
                    groupby=group_key,
                    group1=group1,
                    group2=group2,
                    method="deseq2_like",
                )
            )

            refusal = _empty_bulk_de_refusal(adata_de, de_stats)
            if refusal:
                return refusal
            adata_de.uns["de_current_results_key"] = de_stats["de_results_key"]
            # Store result modality (name derived and checked before the analysis)
            data_manager.store_modality(
                name=de_modality_name,
                adata=adata_de,
                parent_name=modality_name,
                step_summary=f"Bulk DE: {group1} vs {group2}",
            )

            if save_result:
                save_path = f"{de_modality_name}.h5ad"
                data_manager.save_modality(de_modality_name, save_path)

            # Create IR
            from lobster.core.provenance import AnalysisStep as ProvAnalysisStep

            ir = ProvAnalysisStep(
                operation="pydeseq2.differential_expression",
                tool_name="run_bulk_de_direct",
                description=f"Direct bulk DE: {group2} vs {group1} in {group_key}",
                library="pydeseq2",
                parameters={
                    "modality_name": modality_name,
                    "group_key": group_key,
                    "group1": group1,
                    "group2": group2,
                    "alpha": alpha,
                },
                parameter_schema={},
                code_template=(
                    "from pydeseq2.dds import DeseqDataSet\n"
                    "from pydeseq2.ds import DeseqStats\n"
                    "dds = DeseqDataSet(adata=adata, design_factors={{ group_key | string | repr }})\n"
                    "dds.deseq2()\n"
                    "stat_res = DeseqStats(dds, contrast=[{{ group_key | string | repr }}, {{ group2 | string | repr }}, {{ group1 | string | repr }}])\n"
                    "stat_res.summary()\n"
                    "results_df = stat_res.results_df"
                ),
                imports=["pydeseq2"],
            )

            ir = _with_current_de_ir(ir, de_stats["de_results_key"])
            data_manager.log_tool_usage(
                tool_name="run_bulk_de_direct",
                parameters={
                    "modality_name": modality_name,
                    "group_key": group_key,
                    "group1": group1,
                    "group2": group2,
                    "alpha": alpha,
                },
                description=f"Bulk DE: {de_stats['n_significant_genes']} significant genes",
                ir=ir,
                agent=agent_name,
            )

            # Get DEG counts at alpha threshold
            n_sig = de_stats.get("n_significant_genes", 0)
            n_up = de_stats.get("n_upregulated", 0)
            n_down = de_stats.get("n_downregulated", 0)
            top_up = de_stats.get("top_upregulated", [])[:10]
            top_down = de_stats.get("top_downregulated", [])[:10]

            response = f"""## Bulk DE Analysis Complete for '{modality_name}'

**Comparison:** {group2} ({group2_count} samples) vs {group1} ({group1_count} samples) in {group_key}
**Method:** DESeq2-like bulk differential expression

**Results:**
- Genes tested: {de_stats.get('n_genes_tested', 'N/A'):,}
- Significant genes (padj < {alpha}): {n_sig:,}
- Upregulated in {group2}: {n_up:,}
- Downregulated in {group2}: {n_down:,}

**Top 10 Upregulated Genes ({group2} > {group1}):**
{chr(10).join([f"- {gene}" for gene in top_up]) if top_up else "- None found"}

**Top 10 Downregulated Genes ({group2} < {group1}):**
{chr(10).join([f"- {gene}" for gene in top_down]) if top_down else "- None found"}

**New modality created**: '{de_modality_name}'{count_warning}"""

            if save_result:
                response += f"\n**Saved to**: {save_path}"

            response += _replicate_summary_text(replicate_validation)

            # Add replicate warnings
            if replicate_validation["warnings"]:
                response += "\n\n**Statistical Power Warnings:**"
                for warning in replicate_validation["warnings"]:
                    response += f"\n- {warning}"

            response += "\n\n**Next steps**: filter_de_results, run_gsea_analysis, or extract_and_export_de_results"

            return response

        except InvalidModalityNameError as e:
            # The derived result name is not one plain name (fixed reason).
            return f"Error: {e}"
        except (BulkRNASeqError, ModalityNotFoundError) as e:
            logger.error(f"Error in bulk DE direct: {e}")
            return f"Error in bulk DE analysis: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in bulk DE direct: {e}")
            return f"Unexpected error: {str(e)}"

    run_bulk_de_direct.metadata = {"categories": ["ANALYZE"], "provenance": True}
    run_bulk_de_direct.tags = ["ANALYZE"]

    # -------------------------
    # GSEA TOOL
    # -------------------------
    @tool
    def run_gsea_analysis(
        modality_name: str,
        de_results_key: str = None,
        ranking_metric: str = "log2fc",
        databases: list[str] | None = None,
        organism: str = "human",
        min_size: int = 15,
        max_size: int = 500,
        save_result: bool = True,
    ) -> str:
        """Run Gene Set Enrichment Analysis on DE results using ranked gene lists.

        Extracts ranked genes from differential expression results and runs GSEA
        via PathwayEnrichmentService. Supports ranking by log2 fold change or
        signed p-value metric.

        Args:
            modality_name: Name of modality containing DE results
            de_results_key: Specific DE results key in adata.uns (auto-detected if None)
            ranking_metric: Ranking method - 'log2fc' or 'signed_pvalue' (default: 'log2fc')
            databases: List of gene set databases (default: GO_Biological_Process_2023, KEGG_2021_Human)
            organism: Organism for pathway mapping (default: 'human')
            min_size: Minimum gene set size (default: 15)
            max_size: Maximum gene set size (default: 500)
            save_result: Whether to store GSEA results as new modality
        """
        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            de_results_key = _resolve_de_results_key(adata, de_results_key)
            de_data = adata.uns[de_results_key]
            if not isinstance(de_data, dict) or "results_df" not in de_data:
                return f"Invalid DE results format at '{de_results_key}'. Expected dict with 'results_df'."

            results_df = de_data["results_df"]

            # Identify fold change column
            lfc_col = None
            for col_name in ["log2FoldChange", "logFC", "mean_log2FC", "log2FC", "lfc"]:
                if col_name in results_df.columns:
                    lfc_col = col_name
                    break

            if lfc_col is None:
                return f"No log fold change column found in DE results. Available columns: {list(results_df.columns)}"

            # Identify p-value column
            pval_col = None
            for col_name in ["padj", "pvalue_adj", "FDR", "pvalue", "p_adj"]:
                if col_name in results_df.columns:
                    pval_col = col_name
                    break

            # Build ranked gene DataFrame with columns ['gene', 'score']
            if ranking_metric == "signed_pvalue" and pval_col is not None:
                # score = -log10(pvalue) * sign(log2FoldChange)
                pvals = results_df[pval_col].fillna(1.0).clip(lower=1e-300)
                scores = -np.log10(pvals) * np.sign(results_df[lfc_col].fillna(0))
            else:
                # Default: score = log2FoldChange
                scores = results_df[lfc_col].fillna(0)

            ranked_df = pd.DataFrame(
                {
                    "gene": results_df.index.tolist(),
                    "score": scores.values,
                }
            )

            # Drop NaN scores and sort descending
            ranked_df = ranked_df.dropna(subset=["score"])
            ranked_df = ranked_df.sort_values("score", ascending=False).reset_index(
                drop=True
            )

            if len(ranked_df) < 15:
                return f"Only {len(ranked_df)} ranked genes available. GSEA requires at least {min_size} genes per set."

            logger.info(
                f"Running GSEA on {len(ranked_df)} ranked genes from '{de_results_key}' "
                f"using {ranking_metric} ranking"
            )

            # Initialize PathwayEnrichmentService
            from lobster.services.analysis.pathway_enrichment_service import (
                PathwayEnrichmentService,
            )

            pathway_service = PathwayEnrichmentService()

            # Run GSEA
            db_list = databases or ["GO_Biological_Process_2023", "KEGG_2021_Human"]
            adata_gsea, gsea_stats, ir_from_service = (
                pathway_service.gene_set_enrichment_analysis(
                    adata=adata,
                    ranked_genes=ranked_df,
                    databases=db_list,
                    organism=organism,
                    min_size=min_size,
                    max_size=max_size,
                )
            )

            # Store GSEA result as new modality if requested
            if save_result:
                gsea_modality_name = f"{modality_name}_gsea"
                data_manager.store_modality(
                    name=gsea_modality_name,
                    adata=adata_gsea,
                    parent_name=modality_name,
                    step_summary=f"GSEA: {gsea_stats.get('n_significant_gene_sets', 0)} significant gene sets",
                )

            # Log tool usage with IR from service
            data_manager.log_tool_usage(
                tool_name="run_gsea_analysis",
                parameters={
                    "modality_name": modality_name,
                    "de_results_key": de_results_key,
                    "ranking_metric": ranking_metric,
                    "databases": db_list,
                    "organism": organism,
                    "min_size": min_size,
                    "max_size": max_size,
                },
                description=f"GSEA: {gsea_stats.get('n_significant_gene_sets', 0)} significant gene sets",
                ir=ir_from_service,
                agent=agent_name,
            )

            # Parse results for response
            n_total = gsea_stats.get("n_gene_sets_tested", 0)
            n_sig = gsea_stats.get("n_significant_gene_sets", 0)

            # Get top enriched pathways from adata.uns
            gsea_results = adata_gsea.uns.get("gsea_results", {})
            results_list = gsea_results.get("results", [])

            top_up = []
            top_down = []
            for r in results_list[:20]:
                nes = r.get("NES", r.get("nes", 0))
                term = r.get("Term", r.get("term", "Unknown"))
                fdr = r.get("FDR q-val", r.get("fdr", 1.0))
                if isinstance(fdr, (int, float)) and fdr < 0.25:
                    if isinstance(nes, (int, float)):
                        if nes > 0:
                            top_up.append(f"{term} (NES={nes:.2f}, FDR={fdr:.3f})")
                        else:
                            top_down.append(f"{term} (NES={nes:.2f}, FDR={fdr:.3f})")

            response = f"""## GSEA Results for '{modality_name}'

**Analysis Parameters:**
- DE results: {de_results_key}
- Ranking metric: {ranking_metric}
- Genes ranked: {len(ranked_df):,}
- Databases: {', '.join(db_list)}
- Organism: {organism}

**Results:**
- Gene sets tested: {n_total}
- Significant gene sets (FDR < 0.25): {n_sig}

**Top Enriched (NES > 0, upregulated):**
{chr(10).join([f"- {p}" for p in top_up[:5]]) if top_up else "- None found at FDR < 0.25"}

**Top Enriched (NES < 0, downregulated):**
{chr(10).join([f"- {p}" for p in top_down[:5]]) if top_down else "- None found at FDR < 0.25"}

**GSEA results stored in**: adata.uns['gsea_results']"""

            if save_result:
                response += f"\n**New modality created**: '{gsea_modality_name}'"

            response += "\n\n**Next steps**: extract_and_export_de_results for publication tables, or visualize with the visualization expert."

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in GSEA: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in GSEA: {e}")
            return f"Unexpected error in GSEA: {str(e)}"

    run_gsea_analysis.metadata = {"categories": ["ANALYZE"], "provenance": True}
    run_gsea_analysis.tags = ["ANALYZE"]

    # -------------------------
    # PUBLICATION EXPORT TOOL
    # -------------------------
    @tool
    def extract_and_export_de_results(
        modality_name: str,
        de_results_key: str = None,
        shrink_lfc: bool = True,
        output_format: str = "csv",
        output_path: str = None,
        padj_threshold: float = 0.05,
        lfc_threshold: float = 0.0,
    ) -> str:
        """Extract DE results with optional LFC shrinkage and export publication-ready tables.

        Combines result extraction, shrinkage, filtering, and export in one operation.
        For simple export without shrinkage, use export_de_results instead.

        Args:
            modality_name: Name of modality containing DE results
            de_results_key: Specific DE results key in adata.uns (auto-detected if None)
            shrink_lfc: Whether to apply LFC shrinkage (requires DeseqDataSet, default: True)
            output_format: Export format - 'csv' or 'xlsx' (default: 'csv')
            output_path: Custom output path (auto-generated if None)
            padj_threshold: Maximum adjusted p-value for filtering (default: 0.05)
            lfc_threshold: Minimum absolute log2 fold change for filtering (default: 0.0)
        """

        def _resolved_csv_fallback(path: Path) -> Optional[Path]:
            """The resolved CSV name the XLSX fallback writes, or None when the path
            cannot be resolved (the caller refuses with the fixed message)."""
            try:
                return Path(str(path.resolve()).replace(".xlsx", ".csv"))
            except (OSError, RuntimeError, ValueError):
                return None

        try:
            # Validate modality exists
            if modality_name not in data_manager.list_modalities():
                raise ModalityNotFoundError(
                    f"Modality '{modality_name}' not found. "
                    f"Available: {data_manager.list_modalities()}"
                )

            adata = data_manager.get_modality(modality_name)

            de_results_key = _resolve_de_results_key(adata, de_results_key)
            de_data = adata.uns[de_results_key]
            if not isinstance(de_data, dict) or "results_df" not in de_data:
                return f"Invalid DE results format at '{de_results_key}'."

            results_df = de_data["results_df"].copy()

            # Attempt LFC shrinkage if requested
            shrinkage_applied = False
            shrinkage_note = ""
            if shrink_lfc:
                # Check if a DeseqDataSet is stored (required for pyDESeq2 shrinkage)
                dds_stored = de_data.get("parameters", {}).get("method", "") in [
                    "pydeseq2",
                    "deseq2",
                ]
                if dds_stored and "log2FoldChange" in results_df.columns:
                    # pyDESeq2 shrinkage is typically applied during the DE step.
                    # Check if shrinkage was already applied
                    already_shrunk = de_data.get("parameters", {}).get(
                        "shrink_lfc", False
                    )
                    if already_shrunk:
                        shrinkage_applied = True
                        shrinkage_note = (
                            "LFC shrinkage was already applied during DE analysis."
                        )
                    else:
                        # Try to apply shrinkage via BulkRNASeqService if possible
                        # Note: pyDESeq2 shrinkage requires the DeseqDataSet object which
                        # may not be serializable in adata.uns. Warn if unavailable.
                        shrinkage_note = (
                            "LFC shrinkage requested but DeseqDataSet object not available "
                            "in stored results. Exporting unshrunk results. "
                            "To get shrinkage, re-run DE with shrink_lfc=True in run_differential_expression."
                        )
                        logger.warning(shrinkage_note)
                else:
                    shrinkage_note = (
                        "LFC shrinkage not applicable for this DE method. "
                        "Shrinkage requires pyDESeq2-based analysis."
                    )

            # Standardize columns for publication format
            column_renames = {
                "logFC": "log2FoldChange",
                "mean_log2FC": "log2FoldChange",
                "pvalue_adj": "padj",
                "FDR": "padj",
                "p_adj": "padj",
                "base_mean": "baseMean",
                "AveExpr": "baseMean",
            }

            for old_name, new_name in column_renames.items():
                if old_name in results_df.columns and old_name != new_name:
                    results_df = results_df.rename(columns={old_name: new_name})

            # Apply optional filtering
            total_genes = len(results_df)
            if padj_threshold < 1.0 and "padj" in results_df.columns:
                results_df = results_df[results_df["padj"] <= padj_threshold]
            if lfc_threshold > 0 and "log2FoldChange" in results_df.columns:
                results_df = results_df[
                    abs(results_df["log2FoldChange"]) >= lfc_threshold
                ]

            sig_genes = len(results_df)

            # Sort by padj ascending
            if "padj" in results_df.columns:
                results_df = results_df.sort_values("padj")

            # Ensure gene names are the index
            if results_df.index.name is None:
                results_df.index.name = "gene"

            # Select publication columns (in standard order)
            pub_cols = []
            for col in [
                "baseMean",
                "log2FoldChange",
                "lfcSE",
                "stat",
                "pvalue",
                "padj",
            ]:
                if col in results_df.columns:
                    pub_cols.append(col)
            # Include any remaining columns
            extra_cols = [c for c in results_df.columns if c not in pub_cols]
            results_df = results_df[pub_cols + extra_cols]

            # Determine output path
            if output_path is None:
                export_dir = data_manager.exports_dir

                clean_key = de_results_key.replace("de_results_", "")
                output_path = str(
                    export_dir / f"de_publication_{clean_key}.{output_format}"
                )

            # Export (relative paths are rooted in the session exports directory)
            output_path_obj = Path(output_path)
            if not output_path_obj.is_absolute():
                output_path_obj = data_manager.exports_dir / output_path_obj
            # Figures are saved only through the trusted save step: check the
            # caller's (unresolved) destination, and the CSV fallback name derived
            # from it, before anything is created.
            targets = [output_path_obj]
            if output_format == "xlsx":
                targets.append(Path(str(output_path_obj).replace(".xlsx", ".csv")))
            # Every target stays inside the workspace (or a listed export root).
            if outside_workspace(data_manager.workspace_path, *targets):
                return WORKSPACE_EXPORT_MESSAGE
            if export_refused(data_manager.workspace_path, *targets):
                return MANAGED_EXPORT_MESSAGE
            if output_format == "xlsx":
                # ...and the name exactly as the late fallback derives it; a path
                # that cannot be resolved is refused with the same fixed message.
                fallback = _resolved_csv_fallback(output_path_obj)
                if fallback is not None and outside_workspace(
                    data_manager.workspace_path, fallback
                ):
                    return WORKSPACE_EXPORT_MESSAGE
                if fallback is None or export_refused(
                    data_manager.workspace_path, fallback
                ):
                    return MANAGED_EXPORT_MESSAGE
                targets.append(fallback)
            try:
                assert_dirs_allowed(data_manager.workspace_path, output_path_obj.parent)
            except ManagedPathError:
                return MANAGED_EXPORT_MESSAGE
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            output_path = str(output_path_obj.resolve())
            if outside_workspace(data_manager.workspace_path, output_path):
                return WORKSPACE_EXPORT_MESSAGE
            if export_refused(data_manager.workspace_path, *targets, output_path):
                return MANAGED_EXPORT_MESSAGE

            if output_format == "xlsx":
                try:
                    results_df.to_excel(output_path, index=True)
                except ImportError:
                    output_format = "csv"
                    output_path = output_path.replace(".xlsx", ".csv")
                    if outside_workspace(data_manager.workspace_path, output_path):
                        return WORKSPACE_EXPORT_MESSAGE
                    if export_refused(data_manager.workspace_path, output_path):
                        return MANAGED_EXPORT_MESSAGE
                    results_df.to_csv(output_path, index=True)
                    logger.warning("openpyxl not available, falling back to CSV.")
            else:
                results_df.to_csv(output_path, index=True)

            # Create IR
            from lobster.core.provenance import AnalysisStep as ProvAnalysisStep

            ir = ProvAnalysisStep(
                operation="de_results.extract_and_export",
                tool_name="extract_and_export_de_results",
                description=f"Extracted and exported {sig_genes} DE results with publication formatting",
                library="pandas",
                parameters={
                    "modality_name": modality_name,
                    "de_results_key": de_results_key,
                    "shrink_lfc": shrink_lfc,
                    "shrinkage_applied": shrinkage_applied,
                    "output_format": output_format,
                    "output_path": output_path,
                    "padj_threshold": padj_threshold,
                    "lfc_threshold": lfc_threshold,
                },
                parameter_schema={},
                code_template=(
                    "# Extract and export DE results\n"
                    "results_df = adata.uns[{{ de_results_key | string | repr }}]['results_df'].copy()\n"
                    "{% if padj_threshold < 1.0 %}results_df = results_df[results_df['padj'] <= {{ padj_threshold }}]\n{% endif %}"
                    "{% if lfc_threshold > 0 %}results_df = results_df[abs(results_df['log2FoldChange']) >= {{ lfc_threshold }}]\n{% endif %}"
                    "results_df = results_df.sort_values('padj')\n"
                    "results_df.to_{{ output_format }}({{ output_path | string | repr }}, index=True)"
                ),
                imports=["pandas"],
            )

            data_manager.log_tool_usage(
                tool_name="extract_and_export_de_results",
                parameters={
                    "modality_name": modality_name,
                    "de_results_key": de_results_key,
                    "shrink_lfc": shrink_lfc,
                    "output_format": output_format,
                    "output_path": output_path,
                    "padj_threshold": padj_threshold,
                    "lfc_threshold": lfc_threshold,
                },
                description=f"Exported {sig_genes} DE results as publication table",
                ir=ir,
                agent=agent_name,
            )

            response = f"""## Publication DE Results Exported

**Source:** {de_results_key}
**Modality:** {modality_name}

**Export Details:**
- Total genes in DE results: {total_genes:,}
- Genes after filtering: {sig_genes:,}
- padj threshold: {padj_threshold}
- |LFC| threshold: {lfc_threshold}
- Format: {output_format.upper()}
- Output: {output_path}

**LFC Shrinkage:** {'Applied' if shrinkage_applied else 'Not applied'}
{f'- Note: {shrinkage_note}' if shrinkage_note else ''}

**Publication Columns:** {', '.join(pub_cols)}

**File ready for publication supplementary tables.**"""

            return response

        except ModalityNotFoundError as e:
            logger.error(f"Error in extract_and_export_de_results: {e}")
            return f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in extract_and_export_de_results: {e}")
            return f"Unexpected error: {str(e)}"

    extract_and_export_de_results.metadata = {
        "categories": ["UTILITY"],
        "provenance": False,
    }
    extract_and_export_de_results.tags = ["UTILITY"]

    # -------------------------
    # TOOL REGISTRY
    # -------------------------
    base_tools = [
        # Pseudobulk preparation
        create_pseudobulk_matrix,
        prepare_de_design,
        # Design validation
        validate_experimental_design,
        # Formula tools
        suggest_de_formula,
        # DE analysis (2 clear tools)
        run_differential_expression,
        run_de_with_formula,
        # Bulk DE pipeline
        run_bulk_de_direct,
        # Result tools
        filter_de_results,
        export_de_results,
        # GSEA
        run_gsea_analysis,
        # Publication export
        extract_and_export_de_results,
        # Iteration tools
        iterate_de_analysis,
        compare_de_iterations,
        # Pathway analysis (ORA)
        run_pathway_enrichment,
    ]

    tools = base_tools + (delegation_tools or [])

    # -------------------------
    # CREATE AGENT
    # -------------------------
    system_prompt = create_de_analysis_expert_prompt()

    from lobster.tools.code_tool_switch import attach_code_tool

    tools = attach_code_tool(
        tools,
        data_manager,
        AGENT_CONFIG.name,
        default=AGENT_CONFIG.code_tool_default,
        tool_agent_name=agent_name,
    )

    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        name=agent_name,
        state_schema=DEAnalysisExpertState,
    )
