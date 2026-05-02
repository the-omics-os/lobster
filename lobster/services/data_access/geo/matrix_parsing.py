"""
GEO matrix validation, file classification, sample downloads.

Extracted from geo_service.py as part of Phase 4 GEO Service Decomposition.
Contains 18 methods that handle sample info collection, matrix download,
file classification, 10X trio validation, matrix validation, transpose logic,
and metadata formatting.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd

try:
    import GEOparse
except ImportError:
    GEOparse = None

from lobster.services.data_access.geo.parser import ParseResult
from lobster.services.data_access.geo.soft_download import pre_download_soft_file
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class MatrixParser:
    """GEO matrix validation, file classification, and sample downloads.

    Handles all matrix-related operations including:
    - Sample info collection and filtering
    - Individual sample matrix downloads (parallel)
    - Supplementary file classification with scoring
    - 10X trio validation and single-cell file handling
    - Matrix validation with biology-aware thresholds
    - Biological transpose detection
    - Metadata summary formatting
    """

    def __init__(self, service):
        """Initialize with reference to parent GEOService.

        Args:
            service: Parent GEOService instance providing shared state
        """
        self.service = service

    def _get_sample_info(self, gse) -> Dict[str, Dict[str, Any]]:
        """
        Get sample information for downloading individual matrices.

        For multi-modal datasets, filters samples to only include supported modalities (RNA).

        Args:
            gse: GEOparse GSE object

        Returns:
            dict: Sample information dictionary (filtered for multi-modal datasets)
        """
        sample_info = {}

        try:
            geo_id = (
                gse.metadata.get("geo_accession", [""])[0]
                if hasattr(gse, "metadata")
                else ""
            )
            multimodal_info = None
            if geo_id and geo_id in self.service.data_manager.metadata_store:
                stored_entry = self.service.data_manager._get_geo_metadata(geo_id)
                if stored_entry:
                    multimodal_info = stored_entry.get("multimodal_info")

            if hasattr(gse, "gsms"):
                for gsm_id, gsm in gse.gsms.items():
                    if multimodal_info and multimodal_info.get("is_multimodal"):
                        rna_sample_ids = multimodal_info.get("sample_types", {}).get(
                            "rna", []
                        )
                        if gsm_id not in rna_sample_ids:
                            logger.debug(
                                f"Skipping non-RNA sample {gsm_id} (multi-modal dataset)"
                            )
                            continue

                    sample_info[gsm_id] = {
                        "title": (
                            getattr(gsm, "metadata", {}).get("title", [""])[0]
                            if hasattr(gsm, "metadata")
                            else ""
                        ),
                        "platform": (
                            getattr(gsm, "metadata", {}).get("platform_id", [""])[0]
                            if hasattr(gsm, "metadata")
                            else ""
                        ),
                        "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}",
                        "download_url": f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm_id[:6]}nnn/{gsm_id}/suppl/",
                    }

            if multimodal_info and multimodal_info.get("is_multimodal"):
                logger.info(
                    f"Multi-modal filtering: collected {len(sample_info)} RNA samples "
                    f"(excluded {len(gse.gsms) - len(sample_info)} unsupported samples)"
                )
            else:
                logger.debug(f"Collected information for {len(sample_info)} samples")

            return sample_info

        except Exception as e:
            logger.error(f"Error getting sample info: {e}")
            return {}

    def _download_sample_matrices(
        self, sample_info: Dict[str, Dict[str, Any]], gse_id: str
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Download individual sample expression matrices.

        Args:
            sample_info: Dictionary of sample information
            gse_id: GEO series ID

        Returns:
            dict: Dictionary of sample matrices
        """
        sample_matrices = {}

        logger.info(f"Downloading matrices for {len(sample_info)} samples...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_sample = {}
            for gsm_id, info in sample_info.items():
                future = executor.submit(
                    self._download_single_sample, gsm_id, info, gse_id
                )
                future_to_sample[future] = gsm_id
                time.sleep(0.5)

            for future in as_completed(future_to_sample):
                gsm_id = future_to_sample[future]
                try:
                    matrix = future.result()
                    sample_matrices[gsm_id] = matrix
                    if matrix is not None:
                        logger.debug(
                            f"Successfully downloaded matrix for {gsm_id}: {matrix.shape}"
                        )
                    else:
                        logger.debug(f"No matrix data found for {gsm_id}")
                except Exception as e:
                    logger.error(f"Error downloading {gsm_id}: {e}")
                    sample_matrices[gsm_id] = None

        return sample_matrices

    def _download_single_sample(
        self, gsm_id: str, info: Dict[str, Any], gse_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download a single sample matrix with enhanced single-cell support.

        Args:
            gsm_id: GEO sample ID
            info: Sample information
            gse_id: GEO series ID

        Returns:
            DataFrame: Sample expression matrix or None
        """
        gsm = None
        try:
            # Pre-download SOFT file via HTTPS (bypasses GEOparse's unreliable FTP)
            pre_download_soft_file(gsm_id, Path(self.service.cache_dir))

            gsm = GEOparse.get_GEO(geo=gsm_id, destdir=str(self.service.cache_dir))

            if hasattr(gsm, "metadata"):
                suppl_files_mapped = self._extract_supplementary_files_from_metadata(
                    gsm.metadata, gsm_id
                )
                df = self._download_and_combine_single_cell_files(
                    suppl_files_mapped, gsm_id
                )
                return df

        except Exception as e:
            logger.debug(f"Single-cell download attempt failed for {gsm_id}: {e}")

            is_single_cell = False
            if gse_id in self.service.data_manager.metadata_store:
                metadata = self.service.data_manager.metadata_store[gse_id]
                data_type = metadata.get("data_type", "")
                is_single_cell = "single_cell" in data_type.lower()

            if not is_single_cell:
                if gsm is not None and hasattr(gsm, "table") and gsm.table is not None:
                    logger.info(
                        f"Using expression table for non-single-cell sample {gsm_id}"
                    )
                    matrix = gsm.table
                    return self.service._store_single_sample_as_modality(
                        gsm_id, matrix, gsm
                    )
            else:
                logger.error(
                    f"Single-cell sample {gsm_id} failed to download 10X/H5 files. "
                    f"Expression table (series matrix) not suitable for single-cell data. "
                    f"Consider using SUPPLEMENTARY_FIRST or H5_FIRST strategies."
                )
                return None

            logger.debug(f"No expression data found for {gsm_id}")
            return None

        except Exception as e:
            logger.error(f"Error downloading sample {gsm_id}: {e}")
            return None

    def _extract_supplementary_files_from_metadata(
        self, metadata: Dict[str, Any], gsm_id: str
    ) -> Dict[str, str]:
        """
        Extract and classify supplementary files using robust pattern matching and scoring.

        Args:
            metadata: Sample metadata dictionary
            gsm_id: GEO sample ID for logging

        Returns:
            Dict[str, str]: Dictionary mapping file types to URLs with highest confidence scores
        """
        try:
            file_type_patterns = self._initialize_file_type_patterns()
            file_urls = self._extract_all_supplementary_urls(metadata, gsm_id)

            if not file_urls:
                logger.debug(f"No supplementary files found for {gsm_id}")
                return {}

            classified_files = {}
            file_scores = {}

            for url in file_urls:
                filename = url.split("/")[-1]
                file_classification = self._classify_single_file(
                    filename, url, file_type_patterns
                )

                for file_type, score in file_classification.items():
                    if score > 0:
                        if (
                            file_type not in classified_files
                            or score > file_scores.get(file_type, 0)
                        ):
                            classified_files[file_type] = url
                            file_scores[file_type] = score
                            logger.debug(
                                f"Updated {file_type}: {filename} (score: {score:.2f})"
                            )

            logger.debug(f"Final classification for {gsm_id}:")
            for file_type, url in classified_files.items():
                filename = url.split("/")[-1]
                score = file_scores[file_type]
                logger.debug(f"  {file_type}: {filename} (confidence: {score:.2f})")

            if "matrix" in classified_files:
                self._validate_10x_trio_completeness(classified_files, gsm_id)

            return classified_files

        except Exception as e:
            logger.error(
                f"Error extracting supplementary files from metadata for {gsm_id}: {e}"
            )
            return {}

    def _initialize_file_type_patterns(
        self,
    ) -> Dict[str, Dict[str, Union[List[re.Pattern], float]]]:
        """Initialize comprehensive file type classification patterns with confidence scoring."""
        patterns = {
            "matrix": {
                "patterns": [
                    re.compile(r"matrix\.mtx(\.gz)?$", re.IGNORECASE),
                    re.compile(r"matrix\.txt(\.gz)?$", re.IGNORECASE),
                    re.compile(r"matrix\.csv(\.gz)?$", re.IGNORECASE),
                    re.compile(r"matrix\.tsv(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_matrix\.(mtx|txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*-matrix\.(mtx|txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*\.matrix\.(mtx|txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*matrix.*\.(mtx|txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*(count|expr|expression).*matrix.*", re.IGNORECASE),
                ],
                "base_score": 1.0,
                "boost_keywords": ["count", "expression", "sparse", "10x", "chromium"],
            },
            "barcodes": {
                "patterns": [
                    re.compile(r"barcodes\.tsv(\.gz)?$", re.IGNORECASE),
                    re.compile(r"barcodes\.txt(\.gz)?$", re.IGNORECASE),
                    re.compile(r"barcodes\.csv(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_barcode.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*-barcode.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*\.barcode.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*barcode.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(
                        r".*(cell|bc).*id.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE
                    ),
                ],
                "base_score": 1.0,
                "boost_keywords": ["cell", "10x", "chromium", "droplet"],
            },
            "features": {
                "patterns": [
                    re.compile(r"features\.tsv(\.gz)?$", re.IGNORECASE),
                    re.compile(r"genes\.tsv(\.gz)?$", re.IGNORECASE),
                    re.compile(r"features\.txt(\.gz)?$", re.IGNORECASE),
                    re.compile(r"genes\.txt(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_feature.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_gene.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*-feature.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*-gene.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*feature.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*gene.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*annotation.*\.(tsv|txt|csv)(\.gz)?$", re.IGNORECASE),
                ],
                "base_score": 1.0,
                "boost_keywords": ["gene", "ensembl", "symbol", "annotation", "10x"],
            },
            "h5_data": {
                "patterns": [
                    re.compile(r".*\.h5$", re.IGNORECASE),
                    re.compile(r".*\.h5ad$", re.IGNORECASE),
                    re.compile(r".*\.hdf5$", re.IGNORECASE),
                    re.compile(r".*_filtered.*\.h5$", re.IGNORECASE),
                    re.compile(r".*_raw.*\.h5$", re.IGNORECASE),
                    re.compile(r".*matrix.*\.h5$", re.IGNORECASE),
                ],
                "base_score": 1.0,
                "boost_keywords": ["filtered", "raw", "matrix", "10x", "chromium"],
            },
            "expression": {
                "patterns": [
                    re.compile(
                        r".*(expr|expression|count|fpkm|tpm|rpkm).*\.(txt|csv|tsv)(\.gz)?$",
                        re.IGNORECASE,
                    ),
                    re.compile(r".*_counts?\.(txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_expr\.(txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*_data\.(txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                    re.compile(r".*\.(txt|csv|tsv)(\.gz)?$", re.IGNORECASE),
                ],
                "base_score": 0.5,
                "boost_keywords": ["normalized", "filtered", "processed", "log"],
            },
            "archive": {
                "patterns": [
                    re.compile(r".*\.(tar(\.gz|\.bz2)?|tgz)$", re.IGNORECASE),
                    re.compile(r".*\.zip$", re.IGNORECASE),
                    re.compile(r".*\.rar$", re.IGNORECASE),
                ],
                "base_score": 0.8,
                "boost_keywords": ["raw", "supplementary", "all", "complete"],
            },
        }

        return patterns

    def _extract_all_supplementary_urls(
        self, metadata: Dict[str, Any], gsm_id: str
    ) -> List[str]:
        """Extract all supplementary file URLs from metadata using flexible key detection."""
        file_urls = []

        supplementary_key_patterns = [
            re.compile(r".*supplement.*file.*", re.IGNORECASE),
            re.compile(r".*suppl.*file.*", re.IGNORECASE),
            re.compile(r".*additional.*file.*", re.IGNORECASE),
            re.compile(r".*raw.*file.*", re.IGNORECASE),
            re.compile(r".*data.*file.*", re.IGNORECASE),
        ]

        matching_keys = []
        for key in metadata.keys():
            for pattern in supplementary_key_patterns:
                if pattern.match(key):
                    matching_keys.append(key)
                    break

        if not matching_keys:
            matching_keys = [
                key for key in metadata.keys() if "supplement" in key.lower()
            ]

        logger.debug(f"Found supplementary keys for {gsm_id}: {matching_keys}")

        for key in matching_keys:
            urls = metadata[key]
            if isinstance(urls, str):
                urls = [urls]
            elif not isinstance(urls, list):
                continue

            for url in urls:
                if url and isinstance(url, str) and ("http" in url or "ftp" in url):
                    file_urls.append(url)

        logger.info(f"Extracted {len(file_urls)} supplementary file URLs for {gsm_id}")
        return file_urls

    def _classify_single_file(
        self, filename: str, url: str, patterns: Dict
    ) -> Dict[str, float]:
        """Classify a single file using pattern matching and scoring."""
        scores = {}
        filename_lower = filename.lower()
        url_lower = url.lower()

        for file_type, type_config in patterns.items():
            type_patterns = type_config["patterns"]
            base_score = type_config["base_score"]
            boost_keywords = type_config.get("boost_keywords", [])

            max_pattern_score = 0.0

            for i, pattern in enumerate(type_patterns):
                if pattern.search(filename):
                    pattern_confidence = 1.0 - (i * 0.1)
                    max_pattern_score = max(max_pattern_score, pattern_confidence)

            if max_pattern_score > 0:
                total_score = base_score * max_pattern_score

                keyword_boost = 0.0
                for keyword in boost_keywords:
                    if keyword in filename_lower or keyword in url_lower:
                        keyword_boost += 0.1

                total_score += keyword_boost
                scores[file_type] = min(total_score, 2.0)

        return scores

    def _validate_10x_trio_completeness(
        self, classified_files: Dict[str, str], gsm_id: str
    ) -> None:
        """Validate and report on 10X Genomics file trio completeness."""
        required_10x_files = {"matrix", "barcodes", "features"}
        found_10x_files = set(classified_files.keys()) & required_10x_files
        missing_10x_files = required_10x_files - found_10x_files

        if len(found_10x_files) == 3:
            logger.debug(f"Complete 10X trio found for {gsm_id}")
        elif len(found_10x_files) >= 1:
            logger.debug(
                f"Incomplete 10X trio for {gsm_id}. Found: {list(found_10x_files)}, Missing: {list(missing_10x_files)}"
            )

            if "h5_data" in classified_files:
                logger.debug(f"H5 format available as alternative for {gsm_id}")
            elif "expression" in classified_files:
                logger.debug(
                    f"Generic expression file available as fallback for {gsm_id}"
                )
        else:
            logger.debug(f"No 10X format files detected for {gsm_id}")

    def _download_and_combine_single_cell_files(
        self, supplementary_files_info: Dict[str, str], gsm_id: str
    ) -> Optional[pd.DataFrame]:
        """Download and combine single-cell format files (matrix, barcodes, features)."""
        try:
            logger.debug(f"Downloading and combining single-cell files for {gsm_id}")

            if all(
                key in supplementary_files_info
                for key in ["matrix", "barcodes", "features"]
            ):
                logger.debug(f"Found complete 10X trio for {gsm_id}")
                return self.service._download_10x_trio(supplementary_files_info, gsm_id)

            elif "h5_data" in supplementary_files_info:
                logger.debug(f"Found H5 data for {gsm_id}")
                return self.service._download_h5_file(
                    supplementary_files_info["h5_data"], gsm_id
                )

            elif "expression" in supplementary_files_info:
                logger.debug(f"Found expression file for {gsm_id}")
                return self._download_single_expression_file(
                    supplementary_files_info["expression"], gsm_id
                )

            elif "matrix" in supplementary_files_info:
                logger.debug(f"Found matrix file only for {gsm_id}")
                return self._download_single_expression_file(
                    supplementary_files_info["matrix"], gsm_id
                )

            else:
                logger.debug(f"No suitable file combination found for {gsm_id}")
                return None

        except Exception as e:
            logger.error(f"Error downloading and combining files for {gsm_id}: {e}")
            return None

    def _detect_features_format(self, features_path: Path) -> str:
        """Detect 10X features file format by inspecting column count."""
        return self.service.tenx_loader.detect_features_format(features_path)

    def _load_10x_manual(
        self, temp_dir: Path, features_format: str, gse_id: str
    ) -> anndata.AnnData:
        """Manually load 10X MTX when features file is non-standard."""
        return self.service.tenx_loader.load_10x_manual(
            temp_dir, features_format, gse_id
        )

    def _validate_matrices(
        self,
        sample_matrices: Dict[str, Optional[pd.DataFrame]],
        sample_types: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Validate downloaded matrices and filter out invalid ones using multithreading."""
        validated = {}

        valid_matrices = {
            gsm_id: matrix
            for gsm_id, matrix in sample_matrices.items()
            if matrix is not None
        }

        if not valid_matrices:
            logger.warning("No matrices to validate")
            return validated

        gsm_to_type: Dict[str, str] = {}
        if sample_types:
            for modality, gsm_list in sample_types.items():
                for gsm_id in gsm_list:
                    gsm_to_type[gsm_id] = modality
            logger.info(
                f"Type-aware validation enabled: {len(gsm_to_type)} samples classified "
                f"across {len(sample_types)} modalities"
            )
        else:
            logger.debug(
                "No sample type information provided - defaulting all samples to 'rna' type"
            )

        logger.info(
            f"Validating {len(valid_matrices)} matrices using multithreading..."
        )

        with ThreadPoolExecutor(max_workers=min(8, len(valid_matrices))) as executor:
            future_to_sample = {
                executor.submit(
                    self._validate_single_matrix,
                    gsm_id,
                    matrix,
                    gsm_to_type.get(gsm_id, "rna"),
                ): gsm_id
                for gsm_id, matrix in valid_matrices.items()
            }

            for future in as_completed(future_to_sample):
                gsm_id = future_to_sample[future]
                try:
                    is_valid, validation_info = future.result()
                    if is_valid:
                        validated[gsm_id] = valid_matrices[gsm_id]
                        sample_type = gsm_to_type.get(gsm_id, "rna")
                        logger.info(
                            f"Validated {gsm_id} (type: {sample_type}): {validation_info}"
                        )
                    else:
                        sample_type = gsm_to_type.get(gsm_id, "rna")
                        logger.warning(
                            f"Skipping {gsm_id} (type: {sample_type}): {validation_info}"
                        )
                except Exception as e:
                    logger.error(f"Error validating {gsm_id}: {e}")

        logger.info(f"Validated {len(validated)}/{len(sample_matrices)} matrices")
        return validated

    def _validate_single_matrix(
        self, gsm_id: str, matrix: pd.DataFrame, sample_type: str = "rna"
    ) -> Tuple[bool, str]:
        """Validate a single matrix with biology-aware thresholds and type-aware duplicate checking."""
        try:
            n_obs, n_vars = matrix.shape

            if n_obs == 0 or n_vars == 0:
                return False, f"Empty matrix ({n_obs}x{n_vars})"

            if matrix.columns.duplicated().any():
                n_dup = matrix.columns.duplicated().sum()
                logger.warning(
                    f"{gsm_id}: Found {n_dup} duplicate gene IDs. "
                    f"These will be aggregated during loading."
                )

            if matrix.index.duplicated().any():
                n_dup = matrix.index.duplicated().sum()
                duplicate_rate = n_dup / len(matrix)

                if sample_type == "vdj":
                    logger.info(
                        f"{gsm_id}: VDJ data has {n_dup} repeated cell barcodes "
                        f"({duplicate_rate:.1%} of total) - expected for multi-chain data"
                    )
                else:
                    return (
                        False,
                        f"Duplicate cell/sample IDs ({n_dup} duplicates, {duplicate_rate:.1%}) "
                        f"- invalid for {sample_type} data",
                    )

            if n_vars >= 10000:
                if n_obs >= 2:
                    if not self._is_valid_expression_matrix(matrix):
                        return (
                            False,
                            "Invalid matrix format (non-numeric or all-zero data)",
                        )
                    return True, f"Valid matrix: {n_obs} obs x {n_vars} genes"
                else:
                    return (
                        False,
                        f"Only {n_obs} observation(s) - insufficient for analysis (need at least 2)",
                    )

            elif n_obs >= 10000:
                if n_vars >= 100:
                    if not self._is_valid_expression_matrix(matrix):
                        return (
                            False,
                            "Invalid matrix format (non-numeric or all-zero data)",
                        )
                    return True, f"Valid matrix: {n_obs} obs x {n_vars} vars"
                elif n_vars >= 4 and n_obs > 50000:
                    if not self._is_valid_expression_matrix(matrix):
                        return (
                            False,
                            "Invalid matrix format (non-numeric or all-zero data)",
                        )
                    return (
                        True,
                        f"Valid but unusual matrix: {n_obs} obs x {n_vars} vars (likely needs transpose - genes as obs)",
                    )
                else:
                    return (
                        False,
                        f"Only {n_vars} variables - likely transpose error or corrupted data (need at least 100 vars for >10K obs)",
                    )

            else:
                if n_obs >= 10 and n_vars >= 10:
                    if not self._is_valid_expression_matrix(matrix):
                        return (
                            False,
                            "Invalid matrix format (non-numeric or all-zero data)",
                        )
                    return (
                        True,
                        f"Small matrix: {n_obs} obs x {n_vars} vars (may be test/subset data)",
                    )
                else:
                    return (
                        False,
                        f"Matrix too small for analysis ({n_obs}x{n_vars}) - need at least 10x10",
                    )

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _is_valid_expression_matrix(self, matrix: pd.DataFrame) -> bool:
        """Optimized check if a matrix is a valid expression matrix."""
        try:
            if not isinstance(matrix, pd.DataFrame):
                return False

            numeric_dtypes = set(
                ["int16", "int32", "int64", "float16", "float32", "float64"]
            )
            has_numeric = any(str(dtype) in numeric_dtypes for dtype in matrix.dtypes)

            if not has_numeric:
                return False

            if matrix.size > 1_000_000:
                sample_size = min(100_000, int(matrix.size * 0.1))
                flat_sample = matrix.select_dtypes(include=[np.number]).values.flatten()
                if len(flat_sample) > sample_size:
                    indices = np.random.choice(
                        len(flat_sample), sample_size, replace=False
                    )
                    sample_data = flat_sample[indices]
                else:
                    sample_data = flat_sample

                if np.any(sample_data < 0):
                    logger.warning(
                        "Matrix contains negative values (detected in sample)"
                    )

                max_val = np.max(sample_data)
                if max_val > 1e6:
                    logger.info(
                        "Matrix contains very large values (possibly raw counts)"
                    )
            else:
                numeric_data = matrix.select_dtypes(include=[np.number])
                values = numeric_data.values

                if np.any(values < 0):
                    logger.warning("Matrix contains negative values")

                max_val = np.max(values)
                if max_val > 1e6:
                    logger.info(
                        "Matrix contains very large values (possibly raw counts)"
                    )

            return True

        except Exception as e:
            logger.error(f"Error validating matrix: {e}")
            return False

    def _determine_transpose_biologically(
        self,
        matrix: pd.DataFrame,
        gsm_id: str,
        geo_id: Optional[str] = None,
        data_type_hint: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Determine transpose using biological knowledge instead of naive shape comparison."""
        n_rows, n_cols = matrix.shape

        logger.debug(f"Transpose decision for {gsm_id}: shape={n_rows}x{n_cols}")

        if n_rows > 50000 and n_cols < 50000:
            reason = f"Large row count ({n_rows}) indicates genes as rows, transposing to samples/cells x genes"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return True, reason

        if n_cols > 50000 and n_rows < 50000:
            reason = f"Large column count ({n_cols}) indicates genes as columns, keeping as samples/cells x genes"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

        if n_rows > 10000 and n_cols > 10000:
            reason = f"Both dimensions large ({n_rows}x{n_cols}) -> likely cells x genes format (single-cell)"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

        if n_rows > 10000 and n_cols >= 100 and n_cols < 10000:
            reason = f"Many observations, moderate variables ({n_rows}x{n_cols}) -> likely cells x genes (gene panel or filtered data)"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

        if n_rows < 1000 and n_cols > 10000:
            reason = f"Few rows, many columns ({n_rows}x{n_cols}) -> likely samples x genes format (bulk RNA-seq)"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

        data_type = data_type_hint
        if not data_type and geo_id:
            try:
                stored_metadata = self.service.data_manager._get_geo_metadata(geo_id)
                if stored_metadata:
                    data_type = self.service._determine_data_type_from_metadata(
                        stored_metadata["metadata"]
                    )
                    logger.debug(f"Detected data type from metadata: {data_type}")
            except Exception as e:
                logger.warning(
                    f"Could not get data type for biological transpose guidance: {e}"
                )

        if data_type:
            if data_type == "bulk_rna_seq":
                if n_rows < n_cols:
                    reason = f"Bulk RNA-seq: {n_rows} samples x {n_cols} genes (likely correct orientation)"
                    logger.info(f"Transpose decision for {gsm_id}: {reason}")
                    return False, reason
                else:
                    reason = f"Bulk RNA-seq: {n_rows}x{n_cols} likely genes x samples, transposing to samples x genes"
                    logger.info(f"Transpose decision for {gsm_id}: {reason}")
                    return True, reason

            elif data_type == "single_cell_rna_seq":
                if n_rows >= n_cols:
                    reason = f"Single-cell: {n_rows} cells x {n_cols} genes (likely correct orientation)"
                    logger.info(f"Transpose decision for {gsm_id}: {reason}")
                    return False, reason
                else:
                    reason = f"Single-cell: {n_rows}x{n_cols} likely genes x cells, transposing to cells x genes"
                    logger.info(f"Transpose decision for {gsm_id}: {reason}")
                    return True, reason

        if n_cols > n_rows * 100:
            reason = f"Extreme imbalance ({n_rows}x{n_cols}, {n_cols / n_rows:.1f}x) suggests genes as rows, transposing"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return True, reason
        else:
            reason = f"Ambiguous shape ({n_rows}x{n_cols}), defaulting to no transpose (safer - assume samples/cells x genes)"
            logger.info(f"Transpose decision for {gsm_id}: {reason}")
            return False, reason

    def _download_single_expression_file(
        self, url: str, gsm_id: str, geo_id: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Download and parse a single expression file with enhanced support."""
        try:
            logger.info(f"Processing single expression file for {gsm_id}")

            filename = url.split("/")[-1]
            local_path = self.service.cache_dir / f"{gsm_id}_expr_{filename}"

            if not local_path.exists():
                logger.info(f"Downloading expression file: {url}")
                if not self.service.geo_downloader.download_file(url, local_path):
                    logger.error("Failed to download expression file")
                    return None
            else:
                logger.info(f"Using cached expression file: {local_path}")

            parse_result = self.service.geo_parser.parse_supplementary_file(local_path)
            matrix = (
                parse_result.data
                if isinstance(parse_result, ParseResult)
                else parse_result
            )
            if isinstance(parse_result, ParseResult) and parse_result.is_partial:
                logger.warning(
                    f"Partial parse result for {gsm_id}: {parse_result.truncation_reason} "
                    f"({parse_result.rows_read:,} rows read)"
                )
            if matrix is not None and not matrix.empty:
                should_transpose, reason = self._determine_transpose_biologically(
                    matrix=matrix, gsm_id=gsm_id, geo_id=geo_id
                )

                logger.info(f"Transpose decision for {gsm_id}: {reason}")

                if should_transpose:
                    matrix = matrix.T
                    logger.debug(f"Matrix transposed: {matrix.shape}")

                matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]

                logger.info(
                    f"Successfully parsed expression file for {gsm_id}: {matrix.shape}"
                )
                return matrix

            return None

        except Exception as e:
            logger.error(f"Error processing expression file for {gsm_id}: {e}")
            return None

    def _determine_data_type_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Determine likely data type from metadata."""
        from lobster.core.omics_registry import DataTypeDetector

        try:
            return DataTypeDetector().determine_data_type(metadata)
        except Exception as e:
            logger.warning(f"Error determining data type: {e}")
            return "single_cell_rna_seq"

    def _format_metadata_summary(
        self,
        geo_id: str,
        metadata: Dict[str, Any],
        validation_result: Dict[str, Any] = None,
    ) -> str:
        """Format comprehensive metadata summary for user review."""
        try:
            title = str(metadata.get("title", "N/A")).strip()
            summary = str(metadata.get("summary", "N/A")).strip()
            overall_design = str(metadata.get("overall_design", "N/A")).strip()

            samples = metadata.get("samples", {})
            if not isinstance(samples, dict):
                samples = {}
            sample_count = len(samples)

            platforms = metadata.get("platforms", {})
            if not isinstance(platforms, dict):
                platforms = {}
            platform_info = []
            for platform_id, platform_data in platforms.items():
                if isinstance(platform_data, dict):
                    title_info = platform_data.get("title", "N/A")
                    platform_info.append(f"{platform_id}: {title_info}")
                else:
                    platform_info.append(f"{platform_id}: N/A")

            contact_name = str(metadata.get("contact_name", "N/A")).strip()
            contact_institute = str(metadata.get("contact_institute", "N/A")).strip()
            pubmed_id = str(metadata.get("pubmed_id", "Not available")).strip()
            submission_date = str(metadata.get("submission_date", "N/A")).strip()
            last_update = str(metadata.get("last_update_date", "N/A")).strip()

            sample_preview = []
            for i, (sample_id, sample_data) in enumerate(samples.items()):
                if i < 3:
                    if isinstance(sample_data, dict):
                        chars = sample_data.get("characteristics_ch1", [])
                        if isinstance(chars, list) and chars:
                            sample_preview.append(
                                f"  - {sample_id}: {str(chars[0]).strip()}"
                            )
                        else:
                            title_info = sample_data.get("title", "No title")
                            sample_preview.append(
                                f"  - {sample_id}: {str(title_info).strip()}"
                            )
                    else:
                        sample_preview.append(f"  - {sample_id}: No title")

            if sample_count > 3:
                sample_preview.append(f"  ... and {sample_count - 3} more samples")

            validation_status = "UNKNOWN"
            alignment_pct_formatted = "UNKNOWN"
            predicted_type = "UNKNOWN"
            aligned_fields = "UNKNOWN"
            missing_fields = "UNKNOWN"

            if validation_result and isinstance(validation_result, dict):
                validation_status = str(
                    validation_result.get("validation_status", "UNKNOWN")
                ).strip()

                alignment_raw = validation_result.get("alignment_percentage", None)
                if alignment_raw is not None:
                    try:
                        alignment_float = float(alignment_raw)
                        alignment_pct_formatted = f"{alignment_float:.1f}"
                    except (ValueError, TypeError):
                        alignment_pct_formatted = str(alignment_raw)

                predicted_type_raw = validation_result.get(
                    "predicted_data_type", "unknown"
                )
                if predicted_type_raw:
                    predicted_type = str(predicted_type_raw).replace("_", " ").title()

                aligned_raw = validation_result.get("schema_aligned_fields", None)
                if aligned_raw is not None:
                    try:
                        aligned_fields = int(aligned_raw)
                    except (ValueError, TypeError):
                        aligned_fields = str(aligned_raw)

                missing_raw = validation_result.get("schema_missing_fields", None)
                if missing_raw is not None:
                    try:
                        missing_fields = int(missing_raw)
                    except (ValueError, TypeError):
                        missing_fields = str(missing_raw)

            summary_text = f"""GEO Dataset Metadata Summary: {geo_id}

Study Information:
- Title: {title}
- Summary: {summary}
- Design: {overall_design}
- Predicted Type: {predicted_type}

Research Details:
- Contact: {contact_name} ({contact_institute})
- PubMed ID: {pubmed_id}
- Submission: {submission_date}
- Last Update: {last_update}

Platform Information:
{chr(10).join(platform_info) if platform_info else "- No platform information available"}

Sample Information ({sample_count} samples):
{chr(10).join(sample_preview) if sample_preview else "- No sample information available"}

Schema Validation:
- Status: {validation_status}
- Schema Alignment: {alignment_pct_formatted}% of expected fields present
- Aligned Fields: {aligned_fields}
- Missing Fields: {missing_fields}

Next Steps:
1. Review this metadata to ensure it matches your research needs
2. Confirm the predicted data type is correct for your analysis
3. Proceed to download the full dataset if satisfied
4. Use: download_geo_dataset('{geo_id}') to download expression data

Note: This metadata has been cached and validated against our transcriptomics schema.
The actual expression data download will be much faster now that metadata is prepared."""

            return summary_text

        except Exception as e:
            logger.error(f"Error formatting metadata summary: {e}")
            logger.exception("Full traceback for metadata formatting error:")
            return f"Error formatting metadata summary for {geo_id}: {str(e)}"
