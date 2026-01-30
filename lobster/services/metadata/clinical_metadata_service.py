"""
Clinical Metadata Service for processing clinical trial metadata.

This service provides functionality for:
- Processing and validating clinical trial sample metadata
- Creating responder vs non-responder groups
- Filtering samples by clinical trial timepoints
- RECIST 1.1 response normalization

Used by proteomics_expert agent for the Biognosys pilot (SAKK17/18).

Follows Lobster architecture:
- Receives DataManagerV2 in __init__
- Stateless operations
- Returns 3-tuples: (result, stats, AnalysisStep)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.schemas.clinical_schema import (
    ClinicalSample,
    classify_response_group,
    normalize_response,
    parse_timepoint,
)

logger = logging.getLogger(__name__)


class ClinicalMetadataService:
    """Service for processing and validating clinical trial metadata.

    This service enables natural language queries like "compare responders vs
    non-responders at C2D1" by providing:
    - RECIST 1.1 response normalization
    - Responder/non-responder group classification
    - Clinical trial timepoint parsing and filtering

    Follows Lobster architecture:
    - Receives DataManagerV2 in __init__
    - Stateless operations
    - Returns 3-tuples: (result, stats, AnalysisStep)

    Example:
        >>> service = ClinicalMetadataService(data_manager, cycle_length_days=21)
        >>> groups, stats, ir = service.create_responder_groups(metadata_df)
        >>> print(groups)
        {'responder': ['S001', 'S002'], 'non_responder': ['S003'], 'unknown': ['S004']}
    """

    def __init__(self, data_manager: DataManagerV2, cycle_length_days: int = 21) -> None:
        """Initialize ClinicalMetadataService.

        Args:
            data_manager: DataManagerV2 instance for modality access
            cycle_length_days: Days per treatment cycle for absolute day calculation
                              (default 21 for standard 3-week cycles)
        """
        self.data_manager = data_manager
        self.cycle_length_days = cycle_length_days
        logger.info(
            f"ClinicalMetadataService initialized (cycle_length={cycle_length_days} days)"
        )

    def _create_ir(
        self,
        operation: str,
        tool_name: str,
        description: str,
        parameters: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
    ) -> AnalysisStep:
        """Create lightweight IR for clinical metadata operations.

        Clinical metadata operations are tracked for provenance but excluded
        from notebook export (exportable=False) since they are preprocessing
        steps that don't need to appear in reproducible notebooks.

        Args:
            operation: Operation name (e.g., "process_sample_metadata")
            tool_name: Tool name for provenance
            description: Human-readable description
            parameters: Parameters used in operation
            stats: Optional statistics dictionary

        Returns:
            AnalysisStep with exportable=False
        """
        # Build parameter schema for provenance tracking
        parameter_schema = {}
        for param_name, param_value in parameters.items():
            param_type = type(param_value).__name__
            if param_type == "NoneType":
                param_type = "Optional[Any]"
            elif isinstance(param_value, list):
                param_type = "List"
            elif isinstance(param_value, dict):
                param_type = "Dict"
            elif isinstance(param_value, pd.DataFrame):
                param_type = "DataFrame"
                # Don't store DataFrame in params (not JSON-serializable)
                parameters = {
                    k: v for k, v in parameters.items() if not isinstance(v, pd.DataFrame)
                }

            parameter_schema[param_name] = ParameterSpec(
                param_type=param_type,
                papermill_injectable=False,
                default_value=None,
                required=False,
                description=f"Parameter for {operation}",
            )

        return AnalysisStep(
            operation=f"clinical_metadata.{operation}",
            tool_name=tool_name,
            description=description,
            library="lobster",
            code_template="# Clinical metadata operation - not included in notebook export",
            imports=[],
            parameters=parameters,
            parameter_schema=parameter_schema,
            input_entities=["metadata"],
            output_entities=["processed_metadata"],
            execution_context={
                "timestamp": datetime.now().isoformat(),
                "service": "ClinicalMetadataService",
                "cycle_length_days": self.cycle_length_days,
                "statistics": stats or {},
            },
            validates_on_export=False,
            requires_validation=False,
            exportable=False,  # Exclude from notebook export
        )

    def process_sample_metadata(
        self,
        metadata_df: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None,
        validate: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], AnalysisStep]:
        """Process and validate clinical sample metadata.

        Normalizes response codes to RECIST 1.1 standards, parses timepoints,
        derives response groups, and optionally validates each row via
        ClinicalSample schema.

        Args:
            metadata_df: Input DataFrame with clinical metadata.
                Expected columns (flexible via column_mapping):
                - sample_id: Unique sample identifier (required)
                - response_status: RECIST response code or synonym
                - timepoint: Clinical trial timepoint (e.g., "C2D1")
                - pfs_days, pfs_event, os_days, os_event: Survival data
                - age, sex: Demographics
            column_mapping: Optional mapping of input columns to standard names.
                Example: {"BOR": "response_status", "Sample": "sample_id"}
            validate: Whether to validate each row via ClinicalSample schema.
                If False, only normalization is performed without strict validation.

        Returns:
            Tuple of (processed_df, stats_dict, analysis_step_ir):
                - processed_df: DataFrame with normalized columns added:
                    - response_status_normalized: Canonical RECIST code
                    - response_group: responder/non_responder/None
                    - cycle, day: Parsed timepoint components
                    - absolute_day: Days since treatment start
                - stats_dict: Processing statistics
                - analysis_step_ir: Provenance IR (exportable=False)

        Example:
            >>> df = pd.DataFrame({
            ...     'sample_id': ['S001', 'S002'],
            ...     'BOR': ['Complete Response', 'SD'],
            ...     'timepoint': ['C1D1', 'C2D1']
            ... })
            >>> processed, stats, ir = service.process_sample_metadata(
            ...     df, column_mapping={'BOR': 'response_status'}
            ... )
        """
        logger.info(
            f"Processing clinical metadata: {len(metadata_df)} samples, "
            f"validate={validate}"
        )

        # Apply column mapping if provided
        df = metadata_df.copy()
        if column_mapping:
            df = df.rename(columns=column_mapping)
            logger.debug(f"Applied column mapping: {column_mapping}")

        # Ensure sample_id column exists
        if "sample_id" not in df.columns:
            if df.index.name:
                df["sample_id"] = df.index.astype(str)
            else:
                df["sample_id"] = df.index.astype(str)
            logger.debug("Created sample_id from DataFrame index")

        # Initialize result columns
        df["response_status_normalized"] = None
        df["response_group"] = None
        df["cycle"] = None
        df["day"] = None
        df["absolute_day"] = None

        validation_errors = {}
        valid_count = 0
        invalid_count = 0

        for idx, row in df.iterrows():
            # Normalize response status if present
            if "response_status" in df.columns and pd.notna(row.get("response_status")):
                normalized = normalize_response(row["response_status"])
                df.at[idx, "response_status_normalized"] = normalized
                if normalized:
                    df.at[idx, "response_group"] = classify_response_group(normalized)

            # Parse timepoint if present
            if "timepoint" in df.columns and pd.notna(row.get("timepoint")):
                cycle, day = parse_timepoint(str(row["timepoint"]))
                df.at[idx, "cycle"] = cycle
                df.at[idx, "day"] = day

                # Calculate absolute day
                if cycle is not None and day is not None:
                    if cycle == 0 and day == 0:
                        df.at[idx, "absolute_day"] = 0
                    else:
                        df.at[idx, "absolute_day"] = (
                            (cycle - 1) * self.cycle_length_days + day
                        )

            # Optionally validate via ClinicalSample schema
            if validate:
                try:
                    sample_data = row.dropna().to_dict()
                    sample_data["sample_id"] = str(row.get("sample_id", idx))
                    ClinicalSample.from_dict(sample_data)
                    valid_count += 1
                except ValidationError as e:
                    error_msg = "; ".join(
                        [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
                    )
                    validation_errors[str(idx)] = error_msg
                    invalid_count += 1
                except Exception as e:
                    validation_errors[str(idx)] = str(e)
                    invalid_count += 1

        # Collect statistics
        stats = {
            "total_samples": len(df),
            "valid_samples": valid_count if validate else len(df),
            "invalid_samples": invalid_count if validate else 0,
            "validation_errors": validation_errors,
            "columns_mapped": column_mapping or {},
            "response_distribution": (
                df["response_status_normalized"].value_counts().to_dict()
                if "response_status_normalized" in df.columns
                else {}
            ),
            "response_group_distribution": (
                df["response_group"].value_counts().to_dict()
                if "response_group" in df.columns
                else {}
            ),
            "timepoints_parsed": df["cycle"].notna().sum(),
        }

        logger.info(
            f"Processing complete: {stats['valid_samples']} valid, "
            f"{stats['invalid_samples']} invalid"
        )

        # Create IR
        ir = self._create_ir(
            operation="process_sample_metadata",
            tool_name="process_sample_metadata",
            description=f"Process clinical metadata for {len(df)} samples",
            parameters={
                "column_mapping": column_mapping,
                "validate": validate,
                "n_samples": len(df),
            },
            stats=stats,
        )

        return df, stats, ir

    def create_responder_groups(
        self,
        metadata_df: pd.DataFrame,
        response_column: str = "response_status",
        sample_id_column: str = "sample_id",
        grouping_strategy: str = "orr",
    ) -> Tuple[Dict[str, List[str]], Dict[str, Any], AnalysisStep]:
        """Create response-based sample groups.

        Supports two grouping strategies for different clinical endpoints:

        **ORR (Objective Response Rate)** - default:
        - responder: CR, PR (tumor shrinkage)
        - non_responder: SD, PD (no shrinkage)
        - unknown: NE or invalid values

        **DCR (Disease Control Rate)** - for immunotherapy trials:
        - disease_control: CR, PR, SD (tumor controlled)
        - progressive: PD only
        - unknown: NE or invalid values

        The DCR strategy is often preferred for immunotherapy trials (like NSCLC
        checkpoint inhibitors) where Stable Disease represents meaningful clinical
        benefit and the FDA accepts DCR as a valid endpoint.

        Args:
            metadata_df: DataFrame with response data
            response_column: Column containing RECIST/iRECIST response codes.
                Can contain canonical codes (CR, PR, SD, PD, NE, iCR, etc.) or
                synonyms (complete response, stable disease, etc.)
            sample_id_column: Column containing sample identifiers
            grouping_strategy: Clinical endpoint strategy:
                - "orr" (default): Standard Objective Response Rate
                - "dcr": Disease Control Rate (SD grouped with responders)

        Returns:
            Tuple of (groups_dict, stats_dict, analysis_step_ir):
                - groups_dict: Dictionary with group names as keys, sample ID lists as values
                    ORR: {'responder', 'non_responder', 'unknown'}
                    DCR: {'disease_control', 'progressive', 'unknown'}
                - stats_dict: Group statistics
                - analysis_step_ir: Provenance IR (exportable=False)

        Example:
            >>> # Standard ORR grouping
            >>> groups, stats, ir = service.create_responder_groups(df)
            >>> print(f"Responders: {len(groups['responder'])}")

            >>> # DCR grouping for immunotherapy
            >>> groups, stats, ir = service.create_responder_groups(df, grouping_strategy="dcr")
            >>> print(f"Disease control: {len(groups['disease_control'])}")
        """
        # Validate grouping strategy
        valid_strategies = ("orr", "dcr")
        if grouping_strategy.lower() not in valid_strategies:
            raise ValueError(
                f"Invalid grouping_strategy '{grouping_strategy}'. "
                f"Must be one of: {valid_strategies}"
            )
        grouping_strategy = grouping_strategy.lower()

        logger.info(
            f"Creating response groups (strategy={grouping_strategy}) from column "
            f"'{response_column}', sample_id='{sample_id_column}'"
        )

        # Validate required columns exist
        if response_column not in metadata_df.columns:
            raise ValueError(
                f"Response column '{response_column}' not found in DataFrame. "
                f"Available columns: {list(metadata_df.columns)}"
            )

        if sample_id_column not in metadata_df.columns:
            # Try using index as sample_id (works for named index or non-trivial index values)
            metadata_df = metadata_df.copy()
            metadata_df[sample_id_column] = metadata_df.index.astype(str)

        # Initialize groups based on strategy
        if grouping_strategy == "orr":
            groups: Dict[str, List[str]] = {
                "responder": [],
                "non_responder": [],
                "unknown": [],
            }
            # ORR: CR/PR = responder, SD/PD = non_responder
            positive_codes = {"CR", "PR"}
            negative_codes = {"SD", "PD"}
            positive_key = "responder"
            negative_key = "non_responder"
        else:  # dcr
            groups = {
                "disease_control": [],
                "progressive": [],
                "unknown": [],
            }
            # DCR: CR/PR/SD = disease_control, PD = progressive
            positive_codes = {"CR", "PR", "SD"}
            negative_codes = {"PD"}
            positive_key = "disease_control"
            negative_key = "progressive"

        # Classify each sample
        for _, row in metadata_df.iterrows():
            sample_id = str(row[sample_id_column])
            response_value = row[response_column]

            # Normalize response to canonical code
            normalized = normalize_response(response_value)

            if normalized is None:
                groups["unknown"].append(sample_id)
                continue

            # Classify based on strategy
            if normalized in positive_codes:
                groups[positive_key].append(sample_id)
            elif normalized in negative_codes:
                groups[negative_key].append(sample_id)
            else:
                groups["unknown"].append(sample_id)

        # Collect statistics
        total = sum(len(g) for g in groups.values())

        if grouping_strategy == "orr":
            stats = {
                "grouping_strategy": "orr",
                "total_samples": total,
                "responder_count": len(groups["responder"]),
                "non_responder_count": len(groups["non_responder"]),
                "unknown_count": len(groups["unknown"]),
                "responder_percentage": (
                    round(len(groups["responder"]) / total * 100, 1) if total > 0 else 0
                ),
                "non_responder_percentage": (
                    round(len(groups["non_responder"]) / total * 100, 1) if total > 0 else 0
                ),
            }
            summary = (
                f"{stats['responder_count']} responders, "
                f"{stats['non_responder_count']} non-responders"
            )
        else:  # dcr
            stats = {
                "grouping_strategy": "dcr",
                "total_samples": total,
                "disease_control_count": len(groups["disease_control"]),
                "progressive_count": len(groups["progressive"]),
                "unknown_count": len(groups["unknown"]),
                "disease_control_rate": (
                    round(len(groups["disease_control"]) / total * 100, 1) if total > 0 else 0
                ),
                "progression_rate": (
                    round(len(groups["progressive"]) / total * 100, 1) if total > 0 else 0
                ),
            }
            summary = (
                f"{stats['disease_control_count']} disease control, "
                f"{stats['progressive_count']} progressive"
            )

        logger.info(f"Created groups ({grouping_strategy}): {summary}, {groups['unknown']} unknown")

        # Create IR
        ir = self._create_ir(
            operation="create_responder_groups",
            tool_name="create_responder_groups",
            description=f"Create response groups ({grouping_strategy}): {summary}",
            parameters={
                "response_column": response_column,
                "sample_id_column": sample_id_column,
                "grouping_strategy": grouping_strategy,
            },
            stats=stats,
        )

        return groups, stats, ir

    def get_timepoint_samples(
        self,
        metadata_df: pd.DataFrame,
        timepoint: str,
        timepoint_column: str = "timepoint",
        sample_id_column: str = "sample_id",
    ) -> Tuple[List[str], Dict[str, Any], AnalysisStep]:
        """Get sample IDs for a specific clinical trial timepoint.

        Supports flexible timepoint matching:
        - Exact match (case-insensitive): "C2D1" matches "c2d1"
        - Parsed match: "C2D1" matches "Cycle 2 Day 1"
        - Special timepoints: "baseline", "screening", "eot"

        Args:
            metadata_df: DataFrame with timepoint data
            timepoint: Timepoint to filter.
                Examples: "C1D1", "C2D1", "C2D8", "Baseline", "EOT"
            timepoint_column: Column containing timepoint strings
            sample_id_column: Column containing sample identifiers

        Returns:
            Tuple of (sample_ids, stats_dict, analysis_step_ir):
                - sample_ids: List of sample IDs at the specified timepoint
                - stats_dict: Filtering statistics
                - analysis_step_ir: Provenance IR (exportable=False)

        Example:
            >>> samples, stats, ir = service.get_timepoint_samples(df, "C2D1")
            >>> print(f"Found {len(samples)} samples at C2D1")
        """
        logger.info(f"Filtering samples for timepoint '{timepoint}'")

        # Validate required columns exist
        if timepoint_column not in metadata_df.columns:
            raise ValueError(
                f"Timepoint column '{timepoint_column}' not found in DataFrame. "
                f"Available columns: {list(metadata_df.columns)}"
            )

        if sample_id_column not in metadata_df.columns:
            # Try using index as sample_id
            metadata_df = metadata_df.copy()
            metadata_df[sample_id_column] = metadata_df.index.astype(str)

        # Parse target timepoint
        target_cycle, target_day = parse_timepoint(timepoint)

        matching_samples = []
        total_samples = len(metadata_df)

        for _, row in metadata_df.iterrows():
            sample_id = str(row[sample_id_column])
            row_timepoint = row[timepoint_column]

            if pd.isna(row_timepoint):
                continue

            row_timepoint_str = str(row_timepoint)

            # Try exact match first (case-insensitive)
            if row_timepoint_str.lower().strip() == timepoint.lower().strip():
                matching_samples.append(sample_id)
                continue

            # Try parsed match
            row_cycle, row_day = parse_timepoint(row_timepoint_str)

            if row_cycle is not None and row_day is not None:
                if row_cycle == target_cycle and row_day == target_day:
                    matching_samples.append(sample_id)

        # Collect statistics
        stats = {
            "target_timepoint": timepoint,
            "parsed_cycle": target_cycle,
            "parsed_day": target_day,
            "total_samples": total_samples,
            "matching_samples": len(matching_samples),
            "match_percentage": (
                round(len(matching_samples) / total_samples * 100, 1)
                if total_samples > 0
                else 0
            ),
        }

        logger.info(
            f"Found {len(matching_samples)} samples at timepoint '{timepoint}' "
            f"({stats['match_percentage']}% of total)"
        )

        # Create IR
        ir = self._create_ir(
            operation="get_timepoint_samples",
            tool_name="get_timepoint_samples",
            description=f"Filter {len(matching_samples)} samples at timepoint '{timepoint}'",
            parameters={
                "timepoint": timepoint,
                "timepoint_column": timepoint_column,
                "sample_id_column": sample_id_column,
            },
            stats=stats,
        )

        return matching_samples, stats, ir

    def filter_by_response_and_timepoint(
        self,
        metadata_df: pd.DataFrame,
        response_group: Optional[str] = None,
        timepoint: Optional[str] = None,
        response_column: str = "response_status",
        timepoint_column: str = "timepoint",
        sample_id_column: str = "sample_id",
    ) -> Tuple[List[str], Dict[str, Any], AnalysisStep]:
        """Filter samples by response group and/or timepoint.

        Convenience method combining responder grouping and timepoint filtering.
        Useful for queries like "compare responders vs non-responders at C2D1".

        Args:
            metadata_df: DataFrame with response and timepoint data
            response_group: Filter by response group.
                Options: "responder", "non_responder", None (no filter)
            timepoint: Filter by timepoint.
                Examples: "C1D1", "C2D1", "Baseline", None (no filter)
            response_column: Column containing RECIST response codes
            timepoint_column: Column containing timepoint strings
            sample_id_column: Column containing sample identifiers

        Returns:
            Tuple of (sample_ids, stats_dict, analysis_step_ir):
                - sample_ids: List of sample IDs matching criteria
                - stats_dict: Filtering statistics
                - analysis_step_ir: Provenance IR (exportable=False)

        Example:
            >>> # Get responders at C2D1
            >>> samples, stats, ir = service.filter_by_response_and_timepoint(
            ...     df, response_group="responder", timepoint="C2D1"
            ... )
        """
        logger.info(
            f"Filtering samples: response_group={response_group}, timepoint={timepoint}"
        )

        # Start with all samples
        if sample_id_column not in metadata_df.columns:
            metadata_df = metadata_df.copy()
            metadata_df[sample_id_column] = metadata_df.index.astype(str)

        candidate_samples = set(metadata_df[sample_id_column].astype(str).tolist())
        initial_count = len(candidate_samples)

        # Filter by response group if specified
        if response_group is not None:
            groups, _, _ = self.create_responder_groups(
                metadata_df, response_column, sample_id_column
            )
            if response_group not in groups:
                raise ValueError(
                    f"Invalid response_group '{response_group}'. "
                    f"Must be one of: {list(groups.keys())}"
                )
            response_samples = set(groups[response_group])
            candidate_samples = candidate_samples.intersection(response_samples)
            logger.debug(
                f"After response filter: {len(candidate_samples)} samples "
                f"({response_group})"
            )

        # Filter by timepoint if specified
        if timepoint is not None:
            timepoint_samples, _, _ = self.get_timepoint_samples(
                metadata_df, timepoint, timepoint_column, sample_id_column
            )
            timepoint_sample_set = set(timepoint_samples)
            candidate_samples = candidate_samples.intersection(timepoint_sample_set)
            logger.debug(
                f"After timepoint filter: {len(candidate_samples)} samples "
                f"({timepoint})"
            )

        # Convert to sorted list
        result_samples = sorted(list(candidate_samples))

        # Collect statistics
        stats = {
            "initial_samples": initial_count,
            "response_group_filter": response_group,
            "timepoint_filter": timepoint,
            "final_samples": len(result_samples),
            "filter_retention": (
                round(len(result_samples) / initial_count * 100, 1)
                if initial_count > 0
                else 0
            ),
        }

        logger.info(
            f"Filter complete: {len(result_samples)}/{initial_count} samples "
            f"({stats['filter_retention']}% retained)"
        )

        # Create IR
        ir = self._create_ir(
            operation="filter_by_response_and_timepoint",
            tool_name="filter_by_response_and_timepoint",
            description=(
                f"Filter to {len(result_samples)} samples "
                f"(response={response_group}, timepoint={timepoint})"
            ),
            parameters={
                "response_group": response_group,
                "timepoint": timepoint,
                "response_column": response_column,
                "timepoint_column": timepoint_column,
                "sample_id_column": sample_id_column,
            },
            stats=stats,
        )

        return result_samples, stats, ir
