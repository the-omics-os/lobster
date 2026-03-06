"""
GEO download coordination and pipeline execution.

Extracted from geo_service.py as part of Phase 4 GEO Service Decomposition.
Contains 10 methods that coordinate download strategies, execute pipeline
steps, and manage the layered download approach.
"""

import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anndata
import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.core.exceptions import (
    UnsupportedPlatformError,
)
from lobster.services.data_access.geo.constants import (
    GEODataSource,
    GEOResult,
    GEOServiceError,
)
from lobster.services.data_access.geo.helpers import (
    _is_data_valid,
    _score_expression_file,
)
from lobster.services.data_access.geo.soft_download import pre_download_soft_file
from lobster.services.data_access.geo.parser import ParseResult
from lobster.services.data_access.geo.strategy import (
    PipelineType,
    _is_null_value,
    create_pipeline_context,
)
from lobster.utils.logger import get_logger
from lobster.utils.ssl_utils import create_ssl_context, handle_ssl_error

logger = get_logger(__name__)


class DownloadExecutor:
    """GEO download coordination and pipeline step execution.

    Handles all download-related operations including:
    - Strategic download with pipeline selection
    - Pipeline step execution (processed matrix, raw matrix, H5, supplementary)
    - Emergency fallback and GEOparse download
    - Download dataset entry point with adapter selection
    """

    def __init__(self, service):
        """Initialize with reference to parent GEOService.

        Args:
            service: Parent GEOService instance providing shared state
        """
        self.service = service

    def download_dataset(self, geo_id: str, adapter: str = None, **kwargs) -> str:
        """
        Download and process a dataset using modular strategy with fallbacks (Scenarios 2 & 3).

        Args:
            geo_id: GEO accession ID
            adapter: Optional adapter name override

        Returns:
            str: Status message with detailed information
        """
        try:
            logger.info(f"Processing GEO query with modular strategy: {geo_id}")

            # Clean the GEO ID
            clean_geo_id = geo_id.strip().upper()
            if not clean_geo_id.startswith("GSE"):
                return f"Invalid GEO ID format: {geo_id}. Must be a GSE accession."

            # Check if metadata already exists (should be fetched first)
            if clean_geo_id not in self.service.data_manager.metadata_store:
                logger.debug(f"Metadata not found, fetching first for {clean_geo_id}")
                metadata, validation_result = self.service.fetch_metadata_only(clean_geo_id)
                if metadata is None:
                    return f"Failed to fetch metadata for {clean_geo_id}"

            # Safety check: Verify platform compatibility (Phase 2: Early Validation)
            stored_metadata = self.service.data_manager.metadata_store.get(clean_geo_id)
            if stored_metadata:
                # Check if platform error was previously detected
                if "platform_error" in stored_metadata:
                    logger.error(
                        f"Cannot download {clean_geo_id} - platform validation failed previously"
                    )
                    raise UnsupportedPlatformError(
                        message=stored_metadata["platform_error"],
                        details=stored_metadata["platform_details"],
                    )

                # Validate platform compatibility if not done yet
                metadata_dict = stored_metadata.get("metadata", {})
                if metadata_dict:
                    try:
                        is_compatible, compat_message = (
                            self.service._check_platform_compatibility(
                                clean_geo_id, metadata_dict
                            )
                        )
                        logger.info(
                            f"Platform re-validation for {clean_geo_id}: {compat_message}"
                        )
                    except UnsupportedPlatformError:
                        # Already logged and stored by _check_platform_compatibility
                        raise

            # Use the strategic download approach
            geo_result = self.download_with_strategy(geo_id=clean_geo_id, **kwargs)

            if not geo_result.success:
                raise GEOServiceError(
                    f"Failed to download {clean_geo_id} using all available methods. "
                    f"Last error: {geo_result.error_message}"
                )

            # Store as modality in DataManagerV2
            enhanced_metadata = {
                "dataset_id": clean_geo_id,
                "dataset_type": "GEO",
                "source_metadata": geo_result.metadata,
                "processing_date": pd.Timestamp.now().isoformat(),
                "download_source": geo_result.source.value,
                "processing_method": geo_result.processing_info.get(
                    "method", "unknown"
                ),
                "data_type": geo_result.processing_info.get("data_type", "unknown"),
            }

            # Determine appropriate adapter based on data characteristics and metadata
            if not enhanced_metadata.get("data_type", None):
                cached_metadata = self.service.data_manager.metadata_store[clean_geo_id][
                    "metadata"
                ]
                self.service._determine_data_type_from_metadata(cached_metadata)

            n_obs, n_vars = geo_result.data.shape

            # if no adapter name is given find out from data downloading step
            if not adapter:
                # FIXED Bug #3: Use metadata-based modality detection for TAR archives
                try:
                    from lobster.agents.data_expert.assistant import DataExpertAssistant

                    assistant = DataExpertAssistant()

                    # Get metadata for detection
                    # First try metadata from the GEO result, then fallback to cached metadata
                    metadata_for_detection = (
                        geo_result.metadata if geo_result.metadata else {}
                    )
                    if (
                        not metadata_for_detection
                        and clean_geo_id in self.service.data_manager.metadata_store
                    ):
                        metadata_for_detection = self.service.data_manager.metadata_store[
                            clean_geo_id
                        ].get("metadata", {})

                    # Use LLM-based modality detection
                    modality_result = assistant.detect_modality(
                        metadata_for_detection, clean_geo_id
                    )

                    if modality_result and modality_result.modality == "bulk_rna":
                        adapter_name = "transcriptomics_bulk"
                        logger.info(
                            f"{clean_geo_id}: Detected bulk RNA-seq via metadata analysis (confidence: {modality_result.confidence:.2f})"
                        )
                        enhanced_metadata["data_type"] = "bulk_rna_seq"
                    elif modality_result and modality_result.modality in [
                        "scrna_10x",
                        "scrna_smartseq",
                    ]:
                        adapter_name = "transcriptomics_single_cell"
                        logger.info(
                            f"{clean_geo_id}: Detected single-cell RNA-seq via metadata analysis (confidence: {modality_result.confidence:.2f})"
                        )
                        enhanced_metadata["data_type"] = "single_cell_rna_seq"
                    else:
                        # Fallback: Use sample count + cell count heuristic
                        n_samples = len(metadata_for_detection.get("samples", {}))
                        if n_samples == 0 and geo_result.data is not None:
                            # If no sample metadata, use data shape as hint
                            n_samples = (
                                n_obs if n_obs < n_vars else 1000
                            )  # Conservative estimate

                        # Cell count override: if the loaded matrix has >10K obs,
                        # it's single-cell regardless of GSM sample count.
                        if n_obs > 10000:
                            adapter_name = "transcriptomics_single_cell"
                            logger.debug(
                                f"{clean_geo_id}: Cell count override - {n_obs} cells (>{10000}) indicates single-cell despite {n_samples} GSM samples"
                            )
                            enhanced_metadata["data_type"] = "single_cell_rna_seq"
                        elif n_samples < 500:
                            adapter_name = "transcriptomics_bulk"
                            logger.debug(
                                f"{clean_geo_id}: Using sample count heuristic - {n_samples} samples suggests bulk RNA-seq"
                            )
                            enhanced_metadata["data_type"] = "bulk_rna_seq"
                        else:
                            adapter_name = "transcriptomics_single_cell"
                            logger.debug(
                                f"{clean_geo_id}: Using sample count heuristic - {n_samples} samples suggests single-cell"
                            )
                            enhanced_metadata["data_type"] = "single_cell_rna_seq"
                except Exception as e:
                    # Ultimate fallback if modality detection fails completely
                    logger.error(f"Modality detection failed for {clean_geo_id}: {e}")
                    # Conservative default: Use the old logic
                    if enhanced_metadata.get("data_type") == "single_cell_rna_seq":
                        adapter_name = "transcriptomics_single_cell"
                    elif enhanced_metadata.get("data_type") == "bulk_rna_seq":
                        adapter_name = "transcriptomics_bulk"
                    else:
                        # Default to single-cell for GEO datasets (more common)
                        adapter_name = "transcriptomics_single_cell"
            else:
                adapter_name = adapter

            logger.debug(
                f"Using adapter '{adapter_name}' based on predicted type '{enhanced_metadata.get('data_type', None)}' and data shape {geo_result.data.shape}"
            )

            # Add transpose tracking for bulk RNA-seq data
            if (
                adapter_name == "transcriptomics_bulk"
                and geo_result.source == GEODataSource.TAR_ARCHIVE
            ):
                enhanced_metadata["transpose_info"] = {
                    "transpose_applied": False,
                    "transpose_reason": "TAR archive data typically in correct orientation (samples x genes)",
                    "format_specific": False,
                }
            elif adapter_name == "transcriptomics_bulk":
                enhanced_metadata["transpose_info"] = {
                    "transpose_applied": True,
                    "transpose_reason": "Source format may store as genes x samples",
                    "format_specific": True,
                }

            # Construct modality name with correct adapter
            modality_name = f"geo_{clean_geo_id.lower()}_{adapter_name}"

            # Check if modality already exists in DataManagerV2
            existing_modalities = self.service.data_manager.list_modalities()
            if modality_name in existing_modalities:
                return f"Dataset {clean_geo_id} already loaded as modality '{modality_name}'. Use data_manager.get_modality('{modality_name}') to access it."

            # Load as modality in DataManagerV2
            should_transpose = False
            if adapter_name == "transcriptomics_bulk":
                if "transpose_info" in enhanced_metadata:
                    should_transpose = enhanced_metadata["transpose_info"].get(
                        "transpose_applied", False
                    )
                else:
                    should_transpose = geo_result.source not in [
                        GEODataSource.TAR_ARCHIVE
                    ]

            # DEFENSIVE VALIDATION: Check if data appears to be metadata file, not expression matrix
            if (
                isinstance(geo_result.data, pd.DataFrame)
                and geo_result.data.shape[1] <= 3
            ):
                logger.error(
                    f"Data appears to be metadata (only {geo_result.data.shape[1]} columns), "
                    f"not expression matrix. Shape: {geo_result.data.shape}"
                )
                raise ValueError(
                    f"Detected metadata file instead of expression matrix for {clean_geo_id}. "
                    f"Data has only {geo_result.data.shape[1]} columns (likely barcode/gene file). "
                    f"This suggests incorrect file selection during download."
                )

            adata = self.service.data_manager.load_modality(
                name=modality_name,
                source=geo_result.data,
                adapter=adapter_name,
                validate=True,
                transpose=should_transpose,
                **enhanced_metadata,
            )

            # Inject clinical metadata from GEO characteristics_ch1 into adata.obs
            self.service._inject_clinical_metadata(adata, clean_geo_id)

            # Save to workspace
            save_path = f"{modality_name}_raw.h5ad"
            saved_file = self.service.data_manager.save_modality(modality_name, save_path)

            # Check if this was a multi-modal dataset and log exclusions
            multimodal_info = None
            if clean_geo_id in self.service.data_manager.metadata_store:
                stored_entry = self.service.data_manager._get_geo_metadata(clean_geo_id)
                if stored_entry:
                    multimodal_info = stored_entry.get("multimodal_info")

            # Log successful download and save (with multi-modal info if applicable)
            log_params = {
                "geo_id": clean_geo_id,
                "download_source": geo_result.source.value,
                "processing_method": geo_result.processing_info.get(
                    "method", "unknown"
                ),
            }

            log_description = f"Downloaded GEO dataset {clean_geo_id} using strategic approach ({geo_result.source.value}), saved to {saved_file}"

            if multimodal_info and multimodal_info.get("is_multimodal"):
                log_params["is_multimodal"] = True
                log_params["loaded_modalities"] = multimodal_info.get(
                    "supported_types", []
                )
                log_params["excluded_modalities"] = multimodal_info.get(
                    "unsupported_types", []
                )

                sample_types = multimodal_info.get("sample_types", {})
                excluded_count = sum(
                    len(samples)
                    for modality, samples in sample_types.items()
                    if modality != "rna"
                )

                log_description += f" | Multi-modal dataset: loaded {len(sample_types.get('rna', []))} RNA samples, excluded {excluded_count} unsupported samples"

            self.service.data_manager.log_tool_usage(
                tool_name="download_geo_dataset_strategic",
                parameters=log_params,
                description=log_description,
            )

            # Auto-save current state
            self.service.data_manager.auto_save_state()

            # Generate success message (enhanced for multi-modal)
            success_msg = f"""Successfully downloaded and loaded GEO dataset {clean_geo_id}!

Modality: '{modality_name}' ({adata.n_obs} obs x {adata.n_vars} vars)
Adapter: {adapter_name} (predicted: {enhanced_metadata.get("data_type", None)})
Saved to: {save_path}
Source: {geo_result.source.value} ({geo_result.processing_info.get("method", "unknown")})
Ready for quality control and downstream analysis!"""

            if multimodal_info and multimodal_info.get("is_multimodal"):
                sample_types = multimodal_info.get("sample_types", {})
                excluded_summary = ", ".join(
                    [
                        f"{modality.upper()}: {len(samples)}"
                        for modality, samples in sample_types.items()
                        if modality != "rna"
                    ]
                )
                success_msg += f"""

Multi-Modal Dataset Detected:
   Loaded: RNA ({len(sample_types.get("rna", []))} samples)
   Skipped: {excluded_summary} (support coming in v2.6+)

   Note: Only RNA samples were downloaded. Unsupported modalities were excluded to save bandwidth.
   When protein/VDJ support is added, you can re-download to get all modalities."""

            success_msg += f"\n\nThe dataset is now available as modality '{modality_name}' for other agents to use."

            return success_msg

        except GEOServiceError:
            raise
        except UnsupportedPlatformError:
            raise
        except Exception as e:
            logger.exception(f"Error downloading dataset: {e}")
            raise GEOServiceError(
                f"Error downloading dataset {geo_id}: {str(e)}"
            ) from e

    def download_with_strategy(
        self,
        geo_id: str,
        manual_strategy_override: PipelineType = None,
        use_intersecting_genes_only: bool = None,
    ) -> GEOResult:
        """
        Master function implementing layered download approach using dynamic pipeline strategy.

        Args:
            geo_id: GEO accession ID
            manual_strategy_override: Optional manual pipeline override
            use_intersecting_genes_only: Concatenation strategy (None=auto, True=inner, False=outer)

        Returns:
            GEOResult: Comprehensive result with data and metadata
        """
        # Store concatenation strategy for use in pipeline functions
        # Also store on service for backward compat with cross-module access
        self.service._use_intersecting_genes_only = use_intersecting_genes_only
        clean_geo_id = geo_id.strip().upper()

        logger.debug(f"Starting strategic download for {clean_geo_id}")

        try:
            # Step 1: Ensure metadata exists
            if clean_geo_id not in self.service.data_manager.metadata_store:
                metadata, validation_result = self.service.fetch_metadata_only(clean_geo_id)
                if metadata is None:
                    return GEOResult(
                        success=False,
                        error_message=f"Failed to fetch metadata for {clean_geo_id}",
                        source=GEODataSource.GEOPARSE,
                    )

            # Step 2: Get metadata and strategy config using validated retrieval
            stored_metadata_info = self.service.data_manager._get_geo_metadata(clean_geo_id)
            if not stored_metadata_info:
                raise ValueError(
                    f"Metadata for {clean_geo_id} not found or malformed in metadata_store. "
                    f"This indicates a storage/retrieval bug."
                )
            cached_metadata = stored_metadata_info["metadata"]
            strategy_config = stored_metadata_info.get("strategy_config", {})

            if not strategy_config:
                logger.debug(
                    f"No strategy config found for {clean_geo_id}, using defaults"
                )
                strategy_config = {
                    "raw_data_available": True,
                    "summary_file_name": "",
                    "processed_matrix_name": "",
                    "raw_UMI_like_matrix_name": "",
                    "cell_annotation_name": "",
                }

            # Step 3: IF USER DECIDES WHICH APPROACH TO CHOOSE MANUALLY OVERRIDE THE AUTOMATED APPROACH
            if manual_strategy_override:
                pipeline = self.service.pipeline_engine.get_pipeline_functions(
                    manual_strategy_override, self.service
                )
            else:
                pipeline = self._get_processing_pipeline(
                    clean_geo_id, cached_metadata, strategy_config
                )

            logger.debug(f"Using dynamic pipeline with {len(pipeline)} steps")

            # Step 4: Execute pipeline with retries
            for i, pipeline_func in enumerate(pipeline):
                logger.debug(
                    f"Executing pipeline step {i + 1}: {pipeline_func.__name__}"
                )

                try:
                    result = pipeline_func(clean_geo_id, cached_metadata)
                    if result.success:
                        logger.debug(f"Success via {pipeline_func.__name__}")
                        return result
                    else:
                        logger.debug(f"Step failed: {result.error_message}")
                except Exception as e:
                    logger.debug(
                        f"Pipeline step {pipeline_func.__name__} failed: {e}"
                    )
                    continue

            return GEOResult(
                success=False,
                error_message="All pipeline steps failed after enough attempts",
                metadata=cached_metadata,
                source=GEODataSource.GEOPARSE,
            )

        except Exception as e:
            logger.exception(f"Error in strategic download: {e}")
            return GEOResult(
                success=False, error_message=str(e), source=GEODataSource.GEOPARSE
            )

    def _get_processing_pipeline(
        self, geo_id: str, metadata: Dict[str, Any], strategy_config: Dict[str, Any]
    ) -> List[Callable]:
        """
        Get the appropriate processing pipeline using the strategy engine.

        Args:
            geo_id: GEO accession ID
            metadata: GEO metadata
            strategy_config: Extracted strategy configuration

        Returns:
            List[Callable]: Pipeline functions to execute in order
        """
        # Prefer LLM modality detection (high confidence) over keyword heuristic
        data_type = None
        stored = self.service.data_manager._get_geo_metadata(geo_id)
        if stored and isinstance(stored, dict):
            modality_info = stored.get("modality_detection")
            if isinstance(modality_info, dict) and modality_info.get("confidence", 0) >= 0.8:
                llm_modality = modality_info["modality"]
                modality_to_data_type = {
                    "bulk_rna": "bulk_rna_seq",
                    "scrna_10x": "single_cell_rna_seq",
                    "scrna_smartseq": "single_cell_rna_seq",
                    "cite_seq": "single_cell_rna_seq",
                }
                data_type = modality_to_data_type.get(llm_modality)
                if data_type:
                    logger.info(
                        f"Using LLM modality signal: {llm_modality} → data_type={data_type}"
                    )

        # Fall back to heuristic if LLM signal unavailable or unmapped
        if not data_type:
            data_type = self.service._determine_data_type_from_metadata(metadata)

        # Create pipeline context
        context = create_pipeline_context(
            geo_id=geo_id,
            strategy_config=strategy_config,
            metadata=metadata,
            data_type=data_type,
        )

        # Determine best pipeline type
        pipeline_type, description = self.service.pipeline_engine.determine_pipeline(context)

        logger.info(f"Pipeline selection for {geo_id}: {pipeline_type.name}")
        logger.info(f"Reason: {description}")

        # Get the actual processing functions
        pipeline_functions = self.service.pipeline_engine.get_pipeline_functions(
            pipeline_type, self.service
        )

        return pipeline_functions

    def _try_processed_matrix_first(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Try to directly download and use processed matrix files based on LLM strategy config."""
        try:
            logger.debug(f"Attempting to use processed matrix for {geo_id}")

            stored_metadata = self.service.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get("strategy_config", {})

            matrix_name = strategy_config.get("processed_matrix_name", "")
            matrix_type = strategy_config.get("processed_matrix_filetype", "")

            if _is_null_value(matrix_name) or _is_null_value(matrix_type):
                return GEOResult(
                    success=False,
                    error_message="No processed matrix information available in strategy config",
                )

            suppl_files = metadata.get("supplementary_file", [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            target_file = None
            for file_url in suppl_files:
                if matrix_name in file_url and matrix_type in file_url:
                    target_file = file_url
                    break

            if target_file:
                logger.debug(f"Found processed matrix file: {target_file}")
                matrix = self.service._download_and_parse_file(target_file, geo_id)

                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SUPPLEMENTARY,
                        processing_info={
                            "method": "processed_matrix_direct",
                            "file": f"{matrix_name}.{matrix_type}",
                            "data_type": self.service._determine_data_type_from_metadata(
                                metadata
                            ),
                        },
                        success=True,
                    )

            return GEOResult(
                success=False,
                error_message=f"Could not download processed matrix: {matrix_name}.{matrix_type}",
            )

        except Exception as e:
            logger.error(f"Error in processed matrix pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_raw_matrix_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try to directly download and use raw UMI/count matrix files based on LLM strategy config."""
        try:
            logger.debug(f"Attempting to use raw matrix for {geo_id}")

            stored_metadata = self.service.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get("strategy_config", {})

            matrix_name = strategy_config.get("raw_UMI_like_matrix_name", "")
            matrix_type = strategy_config.get("raw_UMI_like_matrix_filetype", "")

            if _is_null_value(matrix_name) or _is_null_value(matrix_type):
                return GEOResult(
                    success=False,
                    error_message="No raw matrix information available in strategy config",
                )

            suppl_files = metadata.get("supplementary_file", [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            target_file = None
            for file_url in suppl_files:
                if matrix_name in file_url and matrix_type in file_url:
                    target_file = file_url
                    break

            if target_file:
                logger.debug(f"Found raw matrix file: {target_file}")
                matrix = self.service._download_and_parse_file(target_file, geo_id)

                if matrix is not None and not matrix.empty:
                    return GEOResult(
                        data=matrix,
                        metadata=metadata,
                        source=GEODataSource.SUPPLEMENTARY,
                        processing_info={
                            "method": "raw_matrix_direct",
                            "file": f"{matrix_name}.{matrix_type}",
                            "data_type": self.service._determine_data_type_from_metadata(
                                metadata
                            ),
                        },
                        success=True,
                    )

            return GEOResult(
                success=False,
                error_message=f"Could not download raw matrix: {matrix_name}.{matrix_type}",
            )

        except Exception as e:
            logger.error(f"Error in raw matrix pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_h5_format_first(self, geo_id: str, metadata: Dict[str, Any]) -> GEOResult:
        """Try to prioritize H5/H5AD format files for efficient loading."""
        try:
            logger.debug(f"Attempting to use H5 format files for {geo_id}")

            stored_metadata = self.service.data_manager.metadata_store[geo_id]
            strategy_config = stored_metadata.get("strategy_config", {})

            processed_name = strategy_config.get("processed_matrix_name", "")
            processed_type = strategy_config.get("processed_matrix_filetype", "")
            raw_name = strategy_config.get("raw_UMI_like_matrix_name", "")
            raw_type = strategy_config.get("raw_UMI_like_matrix_filetype", "")

            h5_files = []
            if processed_type in ["h5", "h5ad"]:
                h5_files.append((processed_name, processed_type, "processed"))
            if raw_type in ["h5", "h5ad"]:
                h5_files.append((raw_name, raw_type, "raw"))

            if not h5_files:
                return GEOResult(
                    success=False,
                    error_message="No H5 format files found in strategy config",
                )

            suppl_files = metadata.get("supplementary_file", [])
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            for file_name, file_type, file_category in h5_files:
                target_file = None
                for file_url in suppl_files:
                    if file_name in file_url and file_type in file_url:
                        target_file = file_url
                        break

                if target_file:
                    logger.debug(f"Found H5 {file_category} file: {target_file}")
                    filename = target_file.split("/")[-1]
                    local_path = self.service.cache_dir / f"{geo_id}_h5_{filename}"

                    if not local_path.exists():
                        if not self.service.geo_downloader.download_file(
                            target_file, local_path
                        ):
                            continue

                    parse_result = self.service.geo_parser.parse_supplementary_file(local_path)
                    matrix = parse_result.data if isinstance(parse_result, ParseResult) else parse_result
                    if isinstance(parse_result, ParseResult) and parse_result.is_partial:
                        logger.warning(
                            f"Partial parse result for {geo_id}: {parse_result.truncation_reason} "
                            f"({parse_result.rows_read:,} rows read)"
                        )
                    if matrix is not None and not matrix.empty:
                        return GEOResult(
                            data=matrix,
                            metadata=metadata,
                            source=GEODataSource.SUPPLEMENTARY,
                            processing_info={
                                "method": f"h5_format_{file_category}",
                                "file": f"{file_name}.{file_type}",
                                "data_type": self.service._determine_data_type_from_metadata(
                                    metadata
                                ),
                            },
                            success=True,
                        )

            return GEOResult(
                success=False,
                error_message="Could not download or parse any H5 format files",
            )

        except Exception as e:
            logger.error(f"Error in H5 format pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_supplementary_first(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Try supplementary files as primary approach when no direct matrices available (with retry)."""
        try:
            logger.debug(f"Attempting supplementary files first for {geo_id}")

            # Pre-download SOFT file via HTTPS (bypasses GEOparse's unreliable FTP)
            pre_download_soft_file(geo_id, Path(self.service.cache_dir))

            result = self.service._retry_with_backoff(
                operation=lambda: GEOparse.get_GEO(
                    geo=geo_id, destdir=str(self.service.cache_dir)
                ),
                operation_name=f"Download {geo_id} for supplementary files",
                max_retries=5,
                is_ftp=True,
            )

            if not result.succeeded:
                return GEOResult(
                    success=False,
                    error_message=f"Failed to download {geo_id} for supplementary files after multiple retry attempts",
                )

            gse = result.value

            # Retrieve multimodal context from metadata store
            multimodal_info = None
            stored = self.service.data_manager._get_geo_metadata(geo_id)
            if stored:
                multimodal_info = stored.get("multimodal_info")

            data = self.service._process_supplementary_files(
                gse, geo_id, multimodal_info=multimodal_info
            )

            data_is_valid = False
            if data is not None:
                if isinstance(data, pd.DataFrame):
                    data_is_valid = not data.empty
                elif isinstance(data, anndata.AnnData):
                    data_is_valid = data.n_obs > 0 and data.n_vars > 0
                else:
                    data_is_valid = True

            if data_is_valid:
                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.SUPPLEMENTARY,
                    processing_info={
                        "method": "supplementary_first",
                        "data_type": self.service._determine_data_type_from_metadata(metadata),
                        "n_samples": len(gse.gsms) if hasattr(gse, "gsms") else 0,
                    },
                    success=True,
                )

            return GEOResult(
                success=False,
                error_message="No usable data found in supplementary files",
            )

        except Exception as e:
            logger.error(f"Error in supplementary first pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_supplementary_fallback(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Fallback method using supplementary files when primary approaches fail."""
        try:
            logger.debug(f"Trying supplementary fallback for {geo_id}")

            # Pre-download SOFT file via HTTPS (bypasses GEOparse's unreliable FTP)
            pre_download_soft_file(geo_id, Path(self.service.cache_dir))

            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.service.cache_dir))

            # Retrieve multimodal context from metadata store
            multimodal_info = None
            stored = self.service.data_manager._get_geo_metadata(geo_id)
            if stored:
                multimodal_info = stored.get("multimodal_info")

            data = self.service._process_supplementary_files(
                gse, geo_id, multimodal_info=multimodal_info
            )
            if _is_data_valid(data):
                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.SUPPLEMENTARY,
                    processing_info={
                        "method": "supplementary_fallback",
                        "data_type": self.service._determine_data_type_from_metadata(metadata),
                        "note": "Used as fallback after primary methods failed",
                    },
                    success=True,
                )

            return GEOResult(
                success=False,
                error_message="Supplementary fallback found no usable data",
            )

        except Exception as e:
            logger.error(f"Error in supplementary fallback pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_emergency_fallback(
        self, geo_id: str, metadata: Dict[str, Any]
    ) -> GEOResult:
        """Emergency fallback when all other methods fail."""
        try:
            logger.warning(
                f"Using emergency fallback for {geo_id} - all other methods failed"
            )

            # Pre-download SOFT file via HTTPS (bypasses GEOparse's unreliable FTP)
            pre_download_soft_file(geo_id, Path(self.service.cache_dir))

            gse = GEOparse.get_GEO(geo=geo_id, destdir=str(self.service.cache_dir))

            if hasattr(gse, "gsms") and gse.gsms:
                for gsm_id, gsm in list(gse.gsms.items())[:5]:
                    try:
                        if hasattr(gsm, "table") and gsm.table is not None:
                            matrix = gsm.table
                            if matrix.shape[0] > 0 and matrix.shape[1] > 0:
                                matrix.index = [
                                    f"{gsm_id}_{idx}" for idx in matrix.index
                                ]

                                logger.warning(
                                    f"Emergency fallback found data from sample {gsm_id}: {matrix.shape}"
                                )
                                return GEOResult(
                                    data=matrix,
                                    metadata=metadata,
                                    source=GEODataSource.GEOPARSE,
                                    processing_info={
                                        "method": "emergency_fallback_single_sample",
                                        "sample_id": gsm_id,
                                        "data_type": self.service._determine_data_type_from_metadata(
                                            metadata
                                        ),
                                        "note": "Emergency fallback - only partial data recovered",
                                    },
                                    success=True,
                                )
                    except Exception as e:
                        logger.debug(f"Could not get data from sample {gsm_id}: {e}")
                        continue

            return GEOResult(
                success=False,
                error_message="Emergency fallback could not recover any data",
            )

        except Exception as e:
            logger.error(f"Error in emergency fallback pipeline: {e}")
            return GEOResult(success=False, error_message=str(e))

    def _try_geoparse_download(
        self,
        geo_id: str,
        metadata: Dict[str, Any],
        use_intersecting_genes_only: bool = None,
    ) -> GEOResult:
        """Pipeline step: Try standard GEOparse download with proper single-cell/bulk handling (with retry)."""
        try:
            logger.debug(f"Trying GEOparse download for {geo_id}")

            # Pre-download SOFT file via HTTPS (bypasses GEOparse's unreliable FTP)
            pre_download_soft_file(geo_id, Path(self.service.cache_dir))

            result = self.service._retry_with_backoff(
                operation=lambda: GEOparse.get_GEO(
                    geo=geo_id, destdir=str(self.service.cache_dir)
                ),
                operation_name=f"Download {geo_id} data",
                max_retries=5,
                is_ftp=False,
            )

            if not result.succeeded:
                return GEOResult(
                    success=False,
                    error_message=f"Failed to download {geo_id} after multiple retry attempts",
                )

            gse = result.value
            data_type = self.service._determine_data_type_from_metadata(metadata)

            # Use instance variable if set, otherwise use parameter default
            concat_strategy = getattr(
                self.service, "_use_intersecting_genes_only", use_intersecting_genes_only
            )

            # Try sample matrices
            sample_info = self.service._get_sample_info(gse)
            if sample_info:
                sample_matrices = self.service._download_sample_matrices(sample_info, geo_id)

                # Detect sample types for type-aware validation
                sample_types = self.service._detect_sample_types(metadata)
                logger.info(
                    f"Sample type detection for {geo_id}: found {len(sample_types)} modalities "
                    f"across {sum(len(gsm_list) for gsm_list in sample_types.values())} samples"
                )
                if sample_types:
                    for modality, gsm_list in sample_types.items():
                        logger.info(f"  - {modality}: {len(gsm_list)} samples")

                validated_matrices = self.service._validate_matrices(
                    sample_matrices, sample_types
                )

                if validated_matrices:
                    if len(validated_matrices) > 1:
                        logger.info(
                            f"Download complete for {geo_id}: {len(validated_matrices)} samples"
                        )
                        logger.info(
                            "Processing pipeline: Store samples -> Concatenate -> Create final AnnData"
                        )
                        logger.info(
                            f"Step 1/3: Storing {len(validated_matrices)} samples as AnnData objects..."
                        )

                        stored_samples = self.service._store_samples_as_anndata(
                            validated_matrices, geo_id, metadata
                        )

                        if stored_samples:
                            logger.info(
                                f"Step 1/3: Successfully stored {len(stored_samples)} samples"
                            )
                            logger.info(
                                "Step 2/3: Concatenating samples (this may take 30-90s for large datasets)..."
                            )

                            concatenated_dataset = self.service._concatenate_stored_samples(
                                geo_id, stored_samples, concat_strategy
                            )

                            if concatenated_dataset is not None:
                                logger.info("Step 2/3: Concatenation complete")
                                logger.info(
                                    "Step 3/3: Validating final AnnData structure..."
                                )
                                logger.info(
                                    "Step 3/3: Validation complete - dataset ready!"
                                )

                            if concatenated_dataset is not None:
                                return GEOResult(
                                    data=concatenated_dataset,
                                    metadata=metadata,
                                    source=GEODataSource.GEOPARSE,
                                    processing_info={
                                        "method": "geoparse_samples_concatenated",
                                        "data_type": data_type,
                                        "n_samples": len(validated_matrices),
                                        "stored_sample_ids": stored_samples,
                                        "use_intersecting_genes_only": concat_strategy,
                                        "batch_info": {
                                            gsm_id: gsm_id
                                            for gsm_id in validated_matrices.keys()
                                        },
                                        "note": f"Samples concatenated with unified ConcatenationService ({data_type})",
                                    },
                                    success=True,
                                )
                    else:
                        stored_samples = self.service._store_samples_as_anndata(
                            validated_matrices, geo_id, metadata
                        )

                        if stored_samples:
                            modality_name = stored_samples[0]
                            single_sample = self.service.data_manager.get_modality(
                                modality_name
                            )

                            return GEOResult(
                                data=single_sample,
                                metadata=metadata,
                                source=GEODataSource.SAMPLE_MATRICES,
                                processing_info={
                                    "method": "geoparse_single_sample",
                                    "n_samples": 1,
                                    "data_type": data_type,
                                    "stored_sample_id": modality_name,
                                },
                                success=True,
                            )

            # Try supplementary files as fallback
            data = self.service._process_supplementary_files(gse, geo_id)
            if _is_data_valid(data):
                data_type = "unknown"
                strategy_config = None

                try:
                    from lobster.agents.data_expert.assistant import DataExpertAssistant

                    assistant = DataExpertAssistant()

                    modality_result = assistant.detect_modality(metadata, geo_id)
                    if modality_result and modality_result.modality == "bulk_rna":
                        data_type = "bulk_rna_seq"
                        logger.info(
                            f"Detected bulk RNA-seq for {geo_id} in supplementary path"
                        )
                    elif modality_result and modality_result.modality in [
                        "scrna_10x",
                        "scrna_smartseq",
                    ]:
                        data_type = "single_cell_rna_seq"
                        logger.info(
                            f"Detected single-cell RNA-seq for {geo_id} in supplementary path"
                        )
                    else:
                        n_samples = len(gse.gsms) if hasattr(gse, "gsms") else 0
                        n_obs = data.shape[0] if hasattr(data, "shape") else 0
                        if n_obs > 10000:
                            data_type = "single_cell_rna_seq"
                            logger.warning(
                                f"Cell count override for {geo_id}: {n_obs} cells indicates single-cell despite {n_samples} GSM samples"
                            )
                        elif n_samples < 500:
                            data_type = "bulk_rna_seq"
                        else:
                            data_type = "single_cell_rna_seq"
                        logger.warning(
                            f"Using sample count heuristic for {geo_id}: {n_samples} samples -> {data_type}"
                        )

                    strategy_config = assistant.extract_strategy_config(
                        metadata, geo_id
                    )
                    if strategy_config:
                        logger.info(
                            f"Persisting strategy_config to metadata_store for {geo_id} (supplementary path)"
                        )
                        self.service.data_manager._store_geo_metadata(
                            geo_id=geo_id,
                            metadata=metadata,
                            stored_by="geo_service_supplementary",
                            strategy_config=(
                                strategy_config.model_dump()
                                if hasattr(strategy_config, "model_dump")
                                else strategy_config
                            ),
                        )

                except Exception as e:
                    logger.warning(
                        f"Metadata detection failed for {geo_id} in supplementary path: {e}"
                    )
                    n_samples = len(gse.gsms) if hasattr(gse, "gsms") else 0
                    data_type = (
                        "bulk_rna_seq" if n_samples < 500 else "single_cell_rna_seq"
                    )

                return GEOResult(
                    data=data,
                    metadata=metadata,
                    source=GEODataSource.GEOPARSE,
                    processing_info={
                        "method": "geoparse_supplementary",
                        "n_samples": len(gse.gsms) if hasattr(gse, "gsms") else 0,
                        "data_type": data_type,
                    },
                    success=True,
                )

            return GEOResult(
                success=False, error_message="GEOparse could not find usable data"
            )

        except Exception as e:
            logger.warning(f"GEOparse download failed: {e}")
            return GEOResult(success=False, error_message=str(e))
