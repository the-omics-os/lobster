"""
GEO sample storage and concatenation.

Extracted from geo_service.py as part of Phase 4 GEO Service Decomposition.
Contains 5 methods that handle sample storage as AnnData, gene coverage
analysis, concatenation strategy selection, and clinical metadata injection.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class SampleConcatenator:
    """GEO sample storage and concatenation.

    Handles all concatenation-related operations including:
    - Storing individual samples as AnnData objects
    - Gene coverage analysis for join strategy selection
    - Sample concatenation via ConcatenationService
    - Single sample storage as modality
    - Clinical metadata injection from GEO characteristics
    """

    def __init__(self, service):
        """Initialize with reference to parent GEOService.

        Args:
            service: Parent GEOService instance providing shared state
        """
        self.service = service

    def _store_samples_as_anndata(
        self,
        validated_matrices: Dict[str, pd.DataFrame],
        gse_id: str,
        metadata: Dict[str, Any],
    ) -> List[str]:
        """
        Store each sample as an individual AnnData object in DataManagerV2.

        Args:
            validated_matrices: Dictionary of validated sample matrices
            gse_id: GEO series ID
            metadata: Metadata from GEO

        Returns:
            List[str]: List of modality names that were successfully stored
        """
        stored_samples = []

        try:
            logger.info(
                f"Storing {len(validated_matrices)} samples as individual AnnData objects"
            )

            for gsm_id, matrix in validated_matrices.items():
                try:
                    modality_name = f"geo_{gse_id.lower()}_sample_{gsm_id.lower()}"

                    sample_metadata = {}
                    if "samples" in metadata and gsm_id in metadata["samples"]:
                        sample_metadata = metadata["samples"][gsm_id]

                    enhanced_metadata = {
                        "dataset_id": gse_id,
                        "sample_id": gsm_id,
                        "dataset_type": "GEO_Sample",
                        "parent_dataset": gse_id,
                        "sample_metadata": sample_metadata,
                        "processing_date": pd.Timestamp.now().isoformat(),
                        "data_source": "individual_sample_matrix",
                        "is_preprocessed": False,
                        "needs_concatenation": True,
                    }

                    # Use metadata-based modality detection
                    try:
                        # Lazy import to avoid circular imports
                        from lobster.agents.data_expert.assistant import (
                            DataExpertAssistant,
                        )

                        assistant = DataExpertAssistant()

                        modality_result = assistant.detect_modality(metadata, gse_id)

                        if modality_result and modality_result.modality == "bulk_rna":
                            adapter_name = "transcriptomics_bulk"
                            logger.info(
                                f"{gse_id}: Detected bulk RNA-seq (confidence: {modality_result.confidence:.2f})"
                            )
                        elif modality_result and modality_result.modality in [
                            "scrna_10x",
                            "scrna_smartseq",
                        ]:
                            adapter_name = "transcriptomics_single_cell"
                            logger.info(
                                f"{gse_id}: Detected single-cell RNA-seq (confidence: {modality_result.confidence:.2f})"
                            )
                        else:
                            n_samples = len(metadata.get("samples", {}))
                            sample_n_obs = (
                                matrix.shape[0] if hasattr(matrix, "shape") else 0
                            )
                            if sample_n_obs > 10000:
                                adapter_name = "transcriptomics_single_cell"
                                logger.warning(
                                    f"{gse_id}: Cell count override - {sample_n_obs} cells indicates single-cell despite {n_samples} GSM samples"
                                )
                            elif n_samples < 500:
                                adapter_name = "transcriptomics_bulk"
                                logger.warning(
                                    f"{gse_id}: Using sample count heuristic - {n_samples} samples suggests bulk RNA-seq"
                                )
                            else:
                                adapter_name = "transcriptomics_single_cell"
                                logger.warning(
                                    f"{gse_id}: Using sample count heuristic - {n_samples} samples suggests single-cell"
                                )
                    except Exception as e:
                        logger.error(f"Modality detection failed for {gse_id}: {e}")
                        n_samples = len(metadata.get("samples", {}))
                        adapter_name = (
                            "transcriptomics_bulk"
                            if n_samples < 500
                            else "transcriptomics_single_cell"
                        )
                        logger.warning(
                            f"Falling back to sample count: {n_samples} samples -> {adapter_name}"
                        )

                    if adapter_name == "transcriptomics_bulk":
                        enhanced_metadata["transpose_info"] = {
                            "transpose_applied": True,
                            "transpose_reason": "GEO SOFT format specification (genes x samples)",
                            "format_specific": True,
                        }

                    adata = self.service.data_manager.load_modality(
                        name=modality_name,
                        source=matrix,
                        adapter=adapter_name,
                        validate=False,
                        transpose=(
                            True if adapter_name == "transcriptomics_bulk" else False
                        ),
                        **enhanced_metadata,
                    )

                    save_path = f"{gse_id.lower()}_{gsm_id.lower()}_raw.h5ad"
                    (self.service.data_manager.data_dir / gse_id.lower()).mkdir(
                        exist_ok=True
                    )
                    self.service.data_manager.save_modality(
                        name=modality_name, path=save_path
                    )

                    stored_samples.append(modality_name)
                    logger.info(
                        f"Stored sample {gsm_id} as modality '{modality_name}' ({adata.shape})"
                    )

                except Exception as e:
                    logger.error(f"Failed to store sample {gsm_id}: {e}")
                    continue

            logger.info(
                f"Successfully stored {len(stored_samples)} samples as individual AnnData objects"
            )
            return stored_samples

        except Exception as e:
            logger.error(f"Error storing samples as AnnData: {e}")
            return stored_samples

    def _analyze_gene_coverage_and_decide_join(
        self, sample_modalities: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze gene coverage variance across samples and decide optimal join strategy.

        Args:
            sample_modalities: List of modality names to analyze

        Returns:
            Tuple of (use_intersecting_genes_only, analysis_metadata)
        """
        try:
            gene_counts = []
            for modality in sample_modalities:
                try:
                    adata = self.service.data_manager.get_modality(modality)
                    gene_counts.append(adata.n_vars)
                except Exception as e:
                    logger.warning(f"Could not get gene count for {modality}: {e}")
                    continue

            if not gene_counts:
                logger.warning("No valid gene counts found, defaulting to inner join")
                return True, {
                    "decision": "inner",
                    "reasoning": "No valid samples found",
                }

            min_genes = int(np.min(gene_counts))
            max_genes = int(np.max(gene_counts))
            mean_genes = float(np.mean(gene_counts))
            std_genes = float(np.std(gene_counts))
            cv = std_genes / mean_genes if mean_genes > 0 else 0.0

            VARIANCE_THRESHOLD = 0.20
            RANGE_RATIO_THRESHOLD = 1.5

            range_ratio = max_genes / min_genes if min_genes > 0 else float("inf")
            use_inner_join = (
                cv <= VARIANCE_THRESHOLD and range_ratio <= RANGE_RATIO_THRESHOLD
            )

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "n_samples_analyzed": len(gene_counts),
                "min_genes": min_genes,
                "max_genes": max_genes,
                "mean_genes": mean_genes,
                "std_genes": std_genes,
                "coefficient_variation": cv,
                "range_ratio": range_ratio,
                "variance_threshold": VARIANCE_THRESHOLD,
                "range_ratio_threshold": RANGE_RATIO_THRESHOLD,
                "decision": "inner" if use_inner_join else "outer",
                "reasoning": (
                    f"Gene coverage CV={cv:.1%} <= {VARIANCE_THRESHOLD:.1%} AND range ratio={range_ratio:.2f}x <= {RANGE_RATIO_THRESHOLD:.2f}x: Consistent coverage"
                    if use_inner_join
                    else (
                        f"Gene coverage variability detected: CV={cv:.1%} (threshold: {VARIANCE_THRESHOLD:.1%}) OR range ratio={range_ratio:.2f}x (threshold: {RANGE_RATIO_THRESHOLD:.2f}x) - using union to preserve all genes"
                    )
                ),
            }

            logger.info("=" * 70)
            logger.info("CONCATENATION STRATEGY DECISION")
            logger.info("=" * 70)
            logger.info(
                f"Analyzing {len(sample_modalities)} samples for gene coverage..."
            )
            logger.info(
                f"   Gene count range: {min_genes:,} - {max_genes:,} ({range_ratio:.2f}x difference)"
            )
            logger.info(f"   Mean: {mean_genes:,.0f} +/- {std_genes:,.0f}")
            logger.info(f"   Coefficient of Variation: {cv:.1%}")
            logger.info("")
            logger.info("Decision Criteria:")
            logger.info(
                f"   CV threshold: {VARIANCE_THRESHOLD:.1%}, Range ratio threshold: {RANGE_RATIO_THRESHOLD:.2f}x"
            )
            logger.info("")

            if use_inner_join:
                logger.info("Selected: INNER JOIN (intersection of genes)")
                logger.info(f"  Reason: {metadata['reasoning']}")
                logger.info(
                    "  Effect: Only genes present in ALL samples will be retained"
                )
                logger.info("  Warning: Genes unique to some samples will be excluded")
            else:
                logger.info("VARIABILITY DETECTED")
                logger.info("Selected: OUTER JOIN (union of all genes)")
                logger.info(f"  Reason: {metadata['reasoning']}")
                logger.info(
                    "  Effect: ALL genes included, missing values filled with zeros"
                )
                logger.info("  Note: This preserves maximum biological information")

            logger.info("=" * 70)

            return use_inner_join, metadata

        except Exception as e:
            logger.error(f"Error analyzing gene coverage: {e}")
            logger.warning("Defaulting to outer join for safety")
            return False, {
                "decision": "outer",
                "reasoning": f"Error during analysis: {str(e)}",
                "error": str(e),
            }

    def _concatenate_stored_samples(
        self,
        geo_id: str,
        stored_samples: List[str],
        use_intersecting_genes_only: bool = None,
    ) -> Optional[pd.DataFrame]:
        """
        Concatenate stored AnnData samples using ConcatenationService with intelligent strategy selection.

        Args:
            geo_id: GEO series ID
            stored_samples: List of modality names that were stored
            use_intersecting_genes_only: Join strategy selection

        Returns:
            AnnData: Concatenated AnnData object or None if concatenation fails
        """
        try:
            # Lazy import to avoid circular imports
            from lobster.services.data_management.concatenation_service import (
                ConcatenationService,
            )

            concat_service = ConcatenationService(self.service.data_manager)

            logger.info(
                f"Using ConcatenationService to concatenate {len(stored_samples)} stored samples for {geo_id}"
            )

            analysis_metadata = {}
            auto_detected = False

            if use_intersecting_genes_only is None:
                logger.info(
                    "No explicit join strategy specified - performing intelligent auto-detection..."
                )
                use_intersecting_genes_only, analysis_metadata = (
                    self._analyze_gene_coverage_and_decide_join(stored_samples)
                )
                auto_detected = True
                logger.info(
                    f"Auto-detection complete: using {'INNER' if use_intersecting_genes_only else 'OUTER'} join"
                )
            else:
                join_type = "inner" if use_intersecting_genes_only else "outer"
                logger.info(
                    f"Using explicitly specified join strategy: {join_type.upper()} join"
                )
                analysis_metadata = {
                    "decision": join_type,
                    "reasoning": f"Explicitly specified by user: use_intersecting_genes_only={use_intersecting_genes_only}",
                    "timestamp": datetime.now().isoformat(),
                }

            concatenated_adata, statistics = concat_service.concatenate_from_modalities(
                modality_names=stored_samples,
                output_name=None,
                use_intersecting_genes_only=use_intersecting_genes_only,
                batch_key="batch",
            )

            provenance_info = {
                **analysis_metadata,
                **statistics,
                "samples_concatenated": len(stored_samples),
                "resulting_shape": (
                    concatenated_adata.n_obs,
                    concatenated_adata.n_vars,
                ),
                "auto_detected": auto_detected,
                "timestamp": datetime.now().isoformat(),
            }

            self.service.data_manager.log_tool_usage(
                tool_name="concatenate_geo_samples",
                parameters={
                    "geo_id": geo_id,
                    "n_samples": len(stored_samples),
                    "join_strategy": (
                        "inner" if use_intersecting_genes_only else "outer"
                    ),
                    "auto_detected": auto_detected,
                    **provenance_info,
                },
            )

            modality_name = f"geo_{geo_id.lower()}"
            concatenation_info = {
                "join_strategy": "inner" if use_intersecting_genes_only else "outer",
                "auto_detected": auto_detected,
                "analysis": analysis_metadata,
                "statistics": statistics,
                "quality_impact": (
                    "Only genes present in all samples retained"
                    if use_intersecting_genes_only
                    else "All genes included, missing values filled with zeros"
                ),
                "provenance_tracked": True,
                "timestamp": datetime.now().isoformat(),
            }

            existing_entry = self.service.data_manager._get_geo_metadata(modality_name)
            if existing_entry:
                self.service.data_manager._enrich_geo_metadata(
                    modality_name,
                    concatenation_decision=concatenation_info,
                )
            else:
                self.service.data_manager._store_geo_metadata(
                    geo_id=modality_name,
                    metadata={},
                    stored_by="_handle_multi_sample_concatenation",
                    concatenation_decision=concatenation_info,
                )

            logger.info(
                "Concatenation decision stored in metadata_store for supervisor access"
            )
            logger.info("Provenance tracked in tool_usage_history")
            logger.info(f"ConcatenationService completed: {statistics}")

            return concatenated_adata

        except Exception as e:
            logger.error(
                f"Error concatenating stored samples using ConcatenationService: {e}"
            )
            return None

    def _store_single_sample_as_modality(
        self, gsm_id: str, matrix: pd.DataFrame, gsm, gse_id: Optional[str] = None
    ) -> str:
        """Store single sample as modality in DataManagerV2."""
        try:
            modality_name = f"geo_sample_{gsm_id.lower()}"

            sample_metadata = {}
            if hasattr(gsm, "metadata"):
                for key, value in gsm.metadata.items():
                    if isinstance(value, list) and len(value) == 1:
                        sample_metadata[key] = value[0]
                    else:
                        sample_metadata[key] = value

            enhanced_metadata = {
                "sample_id": gsm_id,
                "dataset_type": "GEO_Sample",
                "sample_metadata": sample_metadata,
                "processing_date": pd.Timestamp.now().isoformat(),
                "data_source": "single_sample",
            }

            n_obs, n_vars = matrix.shape
            try:
                # Lazy import to avoid circular imports
                from lobster.agents.data_expert.assistant import DataExpertAssistant

                assistant = DataExpertAssistant()

                if gse_id:
                    metadata_for_detection = (
                        self.service.data_manager.metadata_store.get(gse_id, {})
                    )
                else:
                    metadata_for_detection = (
                        self.service.data_manager.metadata_store.get(gsm_id, {})
                    )

                modality_result = assistant.detect_modality(
                    metadata_for_detection, gsm_id
                )

                if modality_result and modality_result.modality == "bulk_rna":
                    adapter_name = "transcriptomics_bulk"
                    logger.info(
                        f"Detected bulk RNA-seq for {gsm_id} using metadata analysis"
                    )
                elif modality_result and modality_result.modality in [
                    "scrna_10x",
                    "scrna_smartseq",
                ]:
                    adapter_name = "transcriptomics_single_cell"
                    logger.info(
                        f"Detected single-cell RNA-seq for {gsm_id} using metadata analysis"
                    )
                else:
                    n_samples = len(metadata_for_detection.get("samples", {}))
                    if n_obs > 10000:
                        adapter_name = "transcriptomics_single_cell"
                        logger.warning(
                            f"Cell count override for {gsm_id}: {n_obs} cells indicates single-cell despite {n_samples} GSM samples"
                        )
                    elif n_samples < 500:
                        adapter_name = "transcriptomics_bulk"
                    else:
                        adapter_name = "transcriptomics_single_cell"
                    logger.warning(
                        f"Using heuristic for {gsm_id}: {n_obs} cells, {n_samples} samples -> {adapter_name}"
                    )
            except Exception as e:
                logger.warning(f"Metadata-based detection failed for {gsm_id}: {e}")
                n_samples = n_obs if n_obs < 1000 else n_vars
                adapter_name = (
                    "transcriptomics_bulk"
                    if n_samples < 500
                    else "transcriptomics_single_cell"
                )
                logger.warning(f"Using conservative fallback: {adapter_name}")

            if adapter_name == "transcriptomics_bulk":
                enhanced_metadata["transpose_info"] = {
                    "transpose_applied": True,
                    "transpose_reason": "GEO SOFT format specification (genes x samples)",
                    "format_specific": True,
                }

            adata = self.service.data_manager.load_modality(
                name=modality_name,
                source=matrix,
                adapter=adapter_name,
                validate=True,
                transpose=(True if adapter_name == "transcriptomics_bulk" else False),
                **enhanced_metadata,
            )

            save_path = f"{gsm_id.lower()}_sample.h5ad"
            self.service.data_manager.save_modality(modality_name, save_path)

            return f"""Successfully downloaded single-cell sample {gsm_id}!

Modality: '{modality_name}' ({adata.n_obs} cells x {adata.n_vars} genes)
Adapter: {adapter_name}
Saved to: {save_path}
Ready for single-cell analysis!"""

        except Exception as e:
            logger.error(f"Error storing sample {gsm_id}: {e}")
            return f"Error storing sample {gsm_id}: {str(e)}"

    def _inject_clinical_metadata(self, adata, geo_id: str) -> None:
        """
        Inject clinical metadata from GEO characteristics_ch1 into adata.obs.

        Parses "key: value" pairs from characteristics_ch1 for each sample and
        maps them to the corresponding observations in adata.obs.

        Args:
            adata: AnnData object to inject metadata into (mutated in-place)
            geo_id: GEO accession ID for metadata store lookup
        """
        try:
            stored = self.service.data_manager.metadata_store.get(geo_id, {})
            metadata = stored.get("metadata", {})
            samples = metadata.get("samples", {})

            if not samples:
                logger.debug(
                    f"No sample metadata found for {geo_id}, skipping clinical metadata injection"
                )
                return

            all_keys = set()
            parsed_samples = {}
            for gsm_id, sample_meta in samples.items():
                chars = sample_meta.get("characteristics_ch1", [])
                if not isinstance(chars, list):
                    continue
                parsed = {}
                for char_str in chars:
                    char_str = str(char_str)
                    if ": " in char_str:
                        key, value = char_str.split(": ", 1)
                        key = key.strip().lower().replace(" ", "_").replace("-", "_")
                        value = value.strip()
                        if value:
                            parsed[key] = value
                            all_keys.add(key)
                if parsed:
                    parsed_samples[gsm_id] = parsed

            if not parsed_samples:
                logger.debug(f"No parseable characteristics_ch1 found for {geo_id}")
                return

            existing_cols = set(adata.obs.columns)
            keys_to_inject = all_keys - existing_cols
            conflicting = all_keys & existing_cols
            if conflicting:
                logger.debug(
                    f"Skipping {len(conflicting)} conflicting columns: {conflicting}"
                )

            if not keys_to_inject:
                logger.debug(
                    f"All clinical metadata keys conflict with existing columns for {geo_id}"
                )
                return

            has_sample_id_col = "sample_id" in adata.obs.columns
            obs_names_upper = {name.upper(): name for name in adata.obs_names}

            gsm_to_rows = {}

            for gsm_id in parsed_samples:
                gsm_upper = gsm_id.upper()
                if has_sample_id_col:
                    mask = adata.obs["sample_id"].astype(str).str.upper() == gsm_upper
                    matching_rows = adata.obs_names[mask].tolist()
                    if matching_rows:
                        gsm_to_rows[gsm_id] = matching_rows
                if gsm_id not in gsm_to_rows:
                    if gsm_upper in obs_names_upper:
                        gsm_to_rows[gsm_id] = [obs_names_upper[gsm_upper]]

            if not gsm_to_rows:
                for gsm_id, sample_meta in samples.items():
                    title = str(sample_meta.get("title", "")).strip()
                    if title and title.upper() in obs_names_upper:
                        gsm_to_rows[gsm_id] = [obs_names_upper[title.upper()]]

            if not gsm_to_rows and len(parsed_samples) == adata.n_obs:
                gsm_ids_ordered = list(parsed_samples.keys())
                obs_names_ordered = adata.obs_names.tolist()
                for gsm_id, obs_name in zip(gsm_ids_ordered, obs_names_ordered):
                    gsm_to_rows[gsm_id] = [obs_name]
                logger.info(
                    f"Using positional mapping for {geo_id}: "
                    f"{len(gsm_to_rows)} samples matched by order "
                    f"(sample count == obs count == {adata.n_obs})"
                )

            if not gsm_to_rows:
                logger.debug(
                    f"Could not map any GSM IDs to obs rows for {geo_id}. "
                    f"GSM IDs: {list(parsed_samples.keys())[:5]}, "
                    f"obs_names: {adata.obs_names[:5].tolist()}, "
                    f"obs sample_ids: {adata.obs['sample_id'].unique()[:5].tolist() if has_sample_id_col else 'N/A'}"
                )
                return

            for key in keys_to_inject:
                adata.obs[key] = pd.Series(
                    [None] * adata.n_obs, index=adata.obs_names, dtype=object
                )

            injected_count = 0
            for gsm_id, rows in gsm_to_rows.items():
                parsed = parsed_samples[gsm_id]
                for key in keys_to_inject:
                    if key in parsed:
                        adata.obs.loc[rows, key] = parsed[key]
                injected_count += 1

            for key in keys_to_inject:
                col = adata.obs[key]
                non_null = col.dropna()
                if len(non_null) > 0:
                    try:
                        converted = pd.to_numeric(non_null, errors="coerce")
                        if converted.notna().sum() / len(non_null) > 0.8:
                            adata.obs[key] = pd.to_numeric(
                                adata.obs[key], errors="coerce"
                            )
                    except (ValueError, TypeError):
                        pass

            logger.info(
                f"Injected clinical metadata for {geo_id}: "
                f"{len(keys_to_inject)} columns ({', '.join(sorted(keys_to_inject))}), "
                f"{injected_count}/{len(parsed_samples)} samples mapped"
            )

        except Exception as e:
            logger.warning(
                f"Failed to inject clinical metadata for {geo_id}: {e}. "
                f"Continuing without clinical metadata."
            )
