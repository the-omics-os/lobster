"""
MetaboLights Download Service implementing IDownloadService interface.

This module provides MetaboLights study downloading with HTTP protocol support,
integrating with the DownloadOrchestrator pattern for queue-based workflow.

Key features:
- HTTP download from MetaboLights REST API and FTP
- MAF (Metabolite Assignment File) parsing for processed intensity matrices
- mzML and vendor raw file download support
- Integration with MetabolomicsAdapter for AnnData loading
- Provenance tracking via AnalysisStep IR
- Support for multiple download strategies

MetaboLights API:
- Study metadata: https://www.ebi.ac.uk/metabolights/ws/studies/{MTBLS_ID}
- File list: https://www.ebi.ac.uk/metabolights/ws/studies/{MTBLS_ID}/files
- Download: https://www.ebi.ac.uk/metabolights/ws/studies/{MTBLS_ID}/download

Download Strategies:
- MAF_FIRST: Prioritize Metabolite Assignment Files (tab-separated, processed data)
- MZML_FIRST: Prioritize mzML raw spectral data files
- RAW_FIRST: Download vendor raw instrument files (.raw, .wiff, .d)
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import requests

from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.download_service import IDownloadService
from lobster.core.schemas.download_queue import DownloadQueueEntry
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# MetaboLights API base URL
METABOLIGHTS_API_BASE = "https://www.ebi.ac.uk/metabolights/ws/studies"

# MetaboLights FTP base for direct file downloads
METABOLIGHTS_FTP_BASE = "ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"

# MTBLS accession pattern
MTBLS_PATTERN = re.compile(r"^MTBLS\d+$", re.IGNORECASE)

# File extensions for each strategy
MAF_EXTENSIONS = (".tsv", ".txt")
MZML_EXTENSIONS = (".mzml", ".mzml.gz")
RAW_EXTENSIONS = (".raw", ".wiff", ".wiff2", ".d", ".mzxml")


class MetaboLightsDownloadService(IDownloadService):
    """
    MetaboLights download service implementing IDownloadService.

    Handles downloading metabolomics datasets from the EBI MetaboLights repository
    via HTTP/FTP, with MAF file parsing for processed intensity matrices and
    integration with the MetabolomicsAdapter for AnnData loading.

    Download strategies:
    - MAF_FIRST: Metabolite Assignment Files (primary, tab-separated, ready to use)
    - MZML_FIRST: mzML raw spectral data files
    - RAW_FIRST: Vendor raw instrument files (.raw, .wiff, .d)

    Attributes:
        data_manager: DataManagerV2 instance
        session: requests.Session for HTTP communication
    """

    def __init__(self, data_manager: DataManagerV2):
        """
        Initialize MetaboLights download service.

        Args:
            data_manager: DataManagerV2 instance for storage and provenance
        """
        super().__init__(data_manager)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Lobster-AI/1.0 (MetaboLights Download Service)",
            }
        )
        logger.info("MetaboLightsDownloadService initialized")

    @classmethod
    def supports_database(cls, database: str) -> bool:
        """
        Check if this service handles MetaboLights database.

        Args:
            database: Database identifier (case-insensitive)

        Returns:
            bool: True for 'metabolights', False otherwise
        """
        return database.lower() in ["metabolights"]

    def supported_databases(self) -> List[str]:
        """
        Get list of databases supported by this service.

        Returns:
            List[str]: ["metabolights"]
        """
        return ["metabolights"]

    @classmethod
    def get_supported_strategies(cls) -> List[str]:
        """
        Get list of download strategies supported by MetaboLights service.

        Strategies in order of preference:
        1. MAF_FIRST - Metabolite Assignment Files (processed intensity matrices)
        2. MZML_FIRST - Standardized mzML spectral files
        3. RAW_FIRST - Vendor raw instrument files (large)

        Returns:
            List[str]: Supported strategy names
        """
        return ["MAF_FIRST", "MZML_FIRST", "RAW_FIRST"]

    def validate_strategy_params(
        self, strategy_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate MetaboLights-specific strategy parameters.

        Args:
            strategy_params: Strategy parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        allowed_params = {
            "file_format": str,  # "maf", "mzml", "raw"
            "include_metadata": bool,  # Whether to include ISA-Tab metadata files
            "max_file_size_mb": (int, float),  # Maximum file size filter
            "assay_index": int,  # Which assay to download (0-based) when multiple exist
        }

        for param, value in strategy_params.items():
            if param not in allowed_params:
                logger.warning(f"Unknown strategy parameter '{param}', ignoring")
                continue

            expected_type = allowed_params[param]
            if not isinstance(value, expected_type):
                return (
                    False,
                    f"Parameter '{param}' must be {expected_type}, got {type(value)}",
                )

        # Validate assay_index is non-negative
        if "assay_index" in strategy_params:
            if strategy_params["assay_index"] < 0:
                return (
                    False,
                    "Parameter 'assay_index' must be non-negative",
                )

        return True, None

    def download_dataset(
        self,
        queue_entry: DownloadQueueEntry,
        strategy_override: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ad.AnnData, Dict[str, Any], AnalysisStep]:
        """
        Execute MetaboLights dataset download from queue entry.

        Workflow:
        1. Validate accession format (MTBLS*)
        2. Fetch study file list from MetaboLights API
        3. Select files based on strategy (MAF_FIRST default)
        4. Download selected files via HTTP
        5. Parse MAF into AnnData (samples x metabolites intensity matrix)
        6. Attach study metadata to adata.uns
        7. Store modality and return 3-tuple

        Args:
            queue_entry: Queue entry with MTBLS accession and metadata
            strategy_override: Optional strategy override

        Returns:
            Tuple of (adata, stats, ir)

        Raises:
            ValueError: If invalid parameters or accession format
            RuntimeError: If download or parsing fails
        """
        # Validate database type
        if not self.supports_database(queue_entry.database):
            raise ValueError(
                f"MetaboLightsDownloadService cannot handle "
                f"database '{queue_entry.database}'"
            )

        dataset_id = queue_entry.dataset_id.upper()

        # Validate MTBLS accession format
        if not MTBLS_PATTERN.match(dataset_id):
            raise ValueError(
                f"Invalid MetaboLights accession format: '{dataset_id}'. "
                f"Expected pattern: MTBLS followed by digits (e.g., MTBLS123)"
            )

        logger.info(f"Starting MetaboLights download for {dataset_id}")
        start_time = time.time()

        try:
            # Step 1: Determine strategy
            strategy_name = (
                strategy_override.get("strategy_name")
                if strategy_override
                else (
                    queue_entry.recommended_strategy.strategy_name
                    if queue_entry.recommended_strategy
                    else "MAF_FIRST"
                )
            )

            strategy_params = (
                strategy_override.get("strategy_params", {})
                if strategy_override
                else {}
            )

            logger.info(f"Using strategy: {strategy_name}")

            # Step 2: Fetch study file list from MetaboLights API
            study_files = self._fetch_study_files(dataset_id)
            if not study_files:
                raise RuntimeError(
                    f"No files found for MetaboLights study {dataset_id}"
                )

            logger.info(f"Found {len(study_files)} files for study {dataset_id}")

            # Step 3: Select files based on strategy
            selected_files = self._select_files_by_strategy(
                study_files, strategy_name, strategy_params
            )

            if not selected_files:
                raise RuntimeError(
                    f"No suitable files found for strategy '{strategy_name}' "
                    f"in study {dataset_id}"
                )

            logger.info(f"Selected {len(selected_files)} file(s) to download")

            # Step 4: Download files
            download_dir = (
                self.data_manager.workspace_dir / "downloads" / dataset_id.lower()
            )
            download_dir.mkdir(parents=True, exist_ok=True)

            local_files = []
            total_size = 0
            for file_info in selected_files:
                local_path = self._download_file(dataset_id, file_info, download_dir)
                if local_path:
                    local_files.append(local_path)
                    total_size += local_path.stat().st_size

            if not local_files:
                raise RuntimeError(f"All file downloads failed for study {dataset_id}")

            # Step 5: Parse into AnnData
            if strategy_name == "MAF_FIRST":
                adata = self._parse_maf_to_anndata(local_files, dataset_id)
            else:
                # For mzML and RAW strategies, create a minimal AnnData
                # pointing to the downloaded files for downstream processing
                adata = self._create_file_reference_anndata(
                    local_files, dataset_id, strategy_name
                )

            # Step 6: Attach study metadata
            adata.uns["metabolights_accession"] = dataset_id
            adata.uns["download_strategy"] = strategy_name
            adata.uns["source_database"] = "metabolights"
            if queue_entry.metadata:
                adata.uns["study_metadata"] = queue_entry.metadata

            # Step 7: Store modality
            modality_name = f"metabolights_{dataset_id.lower()}_metabolomics"
            self.data_manager.modalities[modality_name] = adata

            download_time = time.time() - start_time

            # Step 8: Build stats
            stats = {
                "dataset_id": dataset_id,
                "database": "metabolights",
                "modality_name": modality_name,
                "strategy_used": strategy_name,
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
                "files_downloaded": len(local_files),
                "download_time_seconds": round(download_time, 2),
                "file_size_bytes": total_size,
                "warnings": [],
            }

            # Step 9: Create IR
            ir = self._create_download_ir(dataset_id, strategy_name, strategy_params)

            logger.info(
                f"Successfully downloaded MetaboLights study {dataset_id}: "
                f"{adata.shape} in {download_time:.1f}s"
            )
            return adata, stats, ir

        except Exception as e:
            logger.error(f"Error downloading MetaboLights study {dataset_id}: {e}")
            raise RuntimeError(f"MetaboLights download failed: {str(e)}")

    # =========================================================================
    # API METHODS
    # =========================================================================

    def _fetch_study_files(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Fetch file list for a MetaboLights study via REST API.

        Uses the MetaboLights webservice to retrieve the list of files
        associated with a study, including MAF files, raw data, and metadata.

        Args:
            dataset_id: MTBLS accession (e.g., "MTBLS123")

        Returns:
            List of file info dicts with keys: filename, type, status, directory

        Raises:
            RuntimeError: If API request fails
        """
        url = f"{METABOLIGHTS_API_BASE}/{dataset_id}/files"
        logger.debug(f"Fetching file list from {url}")

        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            # MetaboLights API returns files in 'study' list or nested structure
            files = []
            if isinstance(data, dict):
                # Handle various API response formats
                file_list = (
                    data.get("study", [])
                    or data.get("files", [])
                    or data.get("data", [])
                )
                if isinstance(file_list, list):
                    files = file_list
                elif isinstance(file_list, dict):
                    # Some endpoints return {"files": [...]}
                    files = file_list.get("files", [])
            elif isinstance(data, list):
                files = data

            # Normalize file entries
            normalized = []
            for f in files:
                if isinstance(f, dict):
                    normalized.append(
                        {
                            "filename": f.get("file", f.get("filename", "")),
                            "type": f.get("type", f.get("fileType", "unknown")),
                            "status": f.get("status", "active"),
                            "directory": f.get("directory", f.get("relativePath", "")),
                        }
                    )
                elif isinstance(f, str):
                    normalized.append(
                        {
                            "filename": f,
                            "type": self._classify_file(f),
                            "status": "active",
                            "directory": "",
                        }
                    )

            return normalized

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise RuntimeError(f"MetaboLights study {dataset_id} not found (404)")
            raise RuntimeError(f"MetaboLights API error for {dataset_id}: {e}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Network error fetching MetaboLights study {dataset_id}: {e}"
            )

    def _fetch_study_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """
        Fetch study metadata from MetaboLights API.

        Args:
            dataset_id: MTBLS accession

        Returns:
            Dict with study metadata (title, description, organism, etc.)
        """
        url = f"{METABOLIGHTS_API_BASE}/{dataset_id}"
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Could not fetch metadata for {dataset_id}: {e}")
            return {}

    # =========================================================================
    # FILE SELECTION
    # =========================================================================

    def _classify_file(self, filename: str) -> str:
        """
        Classify a file by its extension into a category.

        Args:
            filename: Name of the file

        Returns:
            File category string: "maf", "mzml", "raw", "metadata", or "other"
        """
        lower = filename.lower()

        # MAF files (Metabolite Assignment Files)
        if lower.startswith("m_") and lower.endswith(MAF_EXTENSIONS):
            return "maf"

        # Sample metadata files
        if lower.startswith("s_") and lower.endswith(MAF_EXTENSIONS):
            return "sample_metadata"

        # Investigation files (ISA-Tab)
        if lower.startswith("i_") and lower.endswith(MAF_EXTENSIONS):
            return "investigation"

        # Assay files
        if lower.startswith("a_") and lower.endswith(MAF_EXTENSIONS):
            return "assay"

        # mzML files
        if lower.endswith(MZML_EXTENSIONS):
            return "mzml"

        # Vendor raw files
        if lower.endswith(RAW_EXTENSIONS):
            return "raw"

        return "other"

    def _select_files_by_strategy(
        self,
        study_files: List[Dict[str, Any]],
        strategy: str,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Select files to download based on strategy.

        Args:
            study_files: List of file info dicts from MetaboLights API
            strategy: Strategy name (MAF_FIRST, MZML_FIRST, RAW_FIRST)
            params: Strategy parameters

        Returns:
            List of file dicts to download
        """
        selected = []

        # Classify all files
        classified = {}
        for f in study_files:
            fname = f.get("filename", "")
            file_type = self._classify_file(fname)
            if file_type not in classified:
                classified[file_type] = []
            classified[file_type].append(f)

        assay_index = params.get("assay_index", None)

        if strategy == "MAF_FIRST":
            # Prioritize MAF files (processed intensity matrices)
            maf_files = classified.get("maf", [])
            if maf_files:
                if assay_index is not None and assay_index < len(maf_files):
                    selected.append(maf_files[assay_index])
                else:
                    # Download all MAF files (each assay has one)
                    selected.extend(maf_files)
            # Also grab sample metadata for obs annotations
            sample_meta = classified.get("sample_metadata", [])
            if sample_meta:
                selected.extend(sample_meta[:1])

        elif strategy == "MZML_FIRST":
            # Prioritize mzML spectral files
            mzml_files = classified.get("mzml", [])
            if mzml_files:
                selected.extend(mzml_files[:10])  # Limit to 10 files

        elif strategy == "RAW_FIRST":
            # Vendor raw instrument files
            raw_files = classified.get("raw", [])
            if raw_files:
                selected.extend(raw_files[:10])  # Limit to 10 files

        # Apply size filter if specified
        max_size_mb = params.get("max_file_size_mb")
        if max_size_mb:
            max_bytes = max_size_mb * 1024 * 1024
            selected = [
                f
                for f in selected
                if (f.get("size", 0) or 0) <= max_bytes or f.get("size") is None
            ]

        # Always include metadata files if requested
        if params.get("include_metadata", False):
            for meta_type in ["investigation", "assay", "sample_metadata"]:
                for f in classified.get(meta_type, []):
                    if f not in selected:
                        selected.append(f)

        return selected

    # =========================================================================
    # DOWNLOAD METHODS
    # =========================================================================

    def _download_file(
        self,
        dataset_id: str,
        file_info: Dict[str, Any],
        download_dir: Path,
        max_retries: int = 3,
    ) -> Optional[Path]:
        """
        Download a single file from MetaboLights via HTTP.

        Attempts the MetaboLights FTP mirror first, then falls back to
        the REST API download endpoint.

        Args:
            dataset_id: MTBLS accession
            file_info: File info dict with filename and directory
            download_dir: Local directory to save the file
            max_retries: Maximum retry attempts

        Returns:
            Path to downloaded file, or None if failed
        """
        filename = file_info.get("filename", "")
        if not filename:
            logger.warning("Empty filename in file info, skipping")
            return None

        output_path = download_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"File already exists, skipping download: {filename}")
            return output_path

        # Build download URL using FTP-over-HTTP proxy or direct API
        directory = file_info.get("directory", "")
        if directory:
            download_url = (
                f"https://www.ebi.ac.uk/metabolights/{dataset_id}/files/"
                f"{directory}/{filename}"
            )
        else:
            download_url = (
                f"https://www.ebi.ac.uk/metabolights/{dataset_id}/files/" f"{filename}"
            )

        # Fallback URL via the study download API
        fallback_url = (
            f"{METABOLIGHTS_API_BASE}/{dataset_id}/download?" f"file={filename}"
        )

        for attempt in range(max_retries):
            try:
                url = download_url if attempt == 0 else fallback_url
                logger.info(
                    f"Downloading {filename} (attempt {attempt + 1}/{max_retries})"
                )

                response = self.session.get(url, stream=True, timeout=300)
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                file_size = output_path.stat().st_size
                logger.info(f"Downloaded: {filename} ({file_size / 1024:.1f} KB)")
                return output_path

            except requests.exceptions.HTTPError as e:
                if (
                    e.response is not None
                    and e.response.status_code == 404
                    and attempt == 0
                ):
                    logger.warning(
                        f"Primary URL returned 404 for {filename}, "
                        f"trying fallback URL"
                    )
                    continue
                logger.warning(
                    f"HTTP error downloading {filename} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
            except (requests.exceptions.ConnectionError, TimeoutError) as e:
                logger.warning(
                    f"Connection error downloading {filename} "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )

            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logger.error(f"Failed to download {filename} after {max_retries} attempts")
        return None

    # =========================================================================
    # PARSING METHODS
    # =========================================================================

    def _parse_maf_to_anndata(
        self,
        local_files: List[Path],
        dataset_id: str,
    ) -> ad.AnnData:
        """
        Parse Metabolite Assignment File(s) into AnnData.

        MAF files are tab-separated with metabolite annotations in columns
        and sample intensity values. The file typically has the structure:
        - First columns: metabolite identifiers (name, HMDB, ChEBI, etc.)
        - Remaining columns: sample intensity values

        Args:
            local_files: List of downloaded file paths
            dataset_id: MTBLS accession for metadata

        Returns:
            AnnData object with samples as obs and metabolites as var

        Raises:
            RuntimeError: If MAF parsing fails
        """
        maf_files = [f for f in local_files if self._classify_file(f.name) == "maf"]
        sample_meta_files = [
            f for f in local_files if self._classify_file(f.name) == "sample_metadata"
        ]

        if not maf_files:
            raise RuntimeError(
                f"No MAF files found among downloaded files for {dataset_id}"
            )

        # Parse the primary MAF file
        primary_maf = maf_files[0]
        logger.info(f"Parsing MAF file: {primary_maf.name}")

        try:
            maf_df = pd.read_csv(primary_maf, sep="\t", low_memory=False)
        except Exception as e:
            raise RuntimeError(f"Failed to parse MAF file {primary_maf.name}: {e}")

        if maf_df.empty:
            raise RuntimeError(f"MAF file {primary_maf.name} is empty")

        # Identify annotation columns vs intensity columns
        # Standard MAF annotation columns (non-numeric, metabolite metadata)
        known_annotation_cols = {
            "database_identifier",
            "chemical_formula",
            "smiles",
            "inchi",
            "metabolite_identification",
            "mass_to_charge",
            "fragmentation",
            "modifications",
            "charge",
            "retention_time",
            "taxid",
            "species",
            "database",
            "database_version",
            "reliability",
            "uri",
            "search_engine",
            "best_search_engine_score[1]",
            "search_engine_score[1]_ms_run[1]",
            "smallmolecule_abundance_sub",
            "smallmolecule_abundance_stdev_sub",
            "smallmolecule_abundance_std_error_sub",
        }

        # Determine which columns are annotations vs sample intensities
        annotation_cols = []
        intensity_cols = []

        for col in maf_df.columns:
            col_lower = col.lower().strip()
            if col_lower in {c.lower() for c in known_annotation_cols}:
                annotation_cols.append(col)
            else:
                # Check if column contains mostly numeric data (intensity)
                numeric_ratio = (
                    pd.to_numeric(maf_df[col], errors="coerce").notna().mean()
                )
                if numeric_ratio > 0.5:
                    intensity_cols.append(col)
                else:
                    annotation_cols.append(col)

        if not intensity_cols:
            # Fallback: treat all non-annotation columns as potential samples
            logger.warning(
                "No clear intensity columns detected in MAF. "
                "Using all numeric columns as sample data."
            )
            for col in maf_df.columns:
                if col not in annotation_cols:
                    intensity_cols.append(col)

        if not intensity_cols:
            raise RuntimeError(
                f"Could not identify sample intensity columns in "
                f"MAF file {primary_maf.name}. Columns found: "
                f"{list(maf_df.columns[:10])}"
            )

        logger.info(
            f"MAF structure: {len(annotation_cols)} annotation columns, "
            f"{len(intensity_cols)} intensity columns (samples)"
        )

        # Build intensity matrix (samples x metabolites)
        # MAF is metabolites x samples, so we need to transpose
        intensity_df = maf_df[intensity_cols].apply(pd.to_numeric, errors="coerce")
        X = intensity_df.values.T  # Transpose: samples x metabolites

        # Preserve NaN values — they represent below-LOD measurements (MNAR).
        # Domain-specific imputation services handle NaN appropriately.
        # Converting NaN→0 here would silently corrupt downstream DE and imputation.
        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            total = X.size
            logger.info(
                f"MetaboLights matrix contains {nan_count}/{total} NaN values "
                f"({nan_count/total*100:.1f}%). Preserved for imputation."
            )

        # Build var (metabolite annotations)
        var_df = pd.DataFrame(index=range(len(maf_df)))
        for col in annotation_cols:
            var_df[col] = maf_df[col].values

        # Use metabolite_identification or first annotation col as var index
        if "metabolite_identification" in maf_df.columns:
            var_names = maf_df["metabolite_identification"].fillna(
                [f"metabolite_{i}" for i in range(len(maf_df))]
            )
        elif "database_identifier" in maf_df.columns:
            var_names = maf_df["database_identifier"].fillna(
                [f"metabolite_{i}" for i in range(len(maf_df))]
            )
        else:
            var_names = pd.Series([f"metabolite_{i}" for i in range(len(maf_df))])

        # Ensure unique var names
        var_names = var_names.astype(str)
        if var_names.duplicated().any():
            var_names = pd.Series(_make_unique(var_names.tolist()))
        var_df.index = var_names.values

        # Build obs (sample annotations)
        obs_df = pd.DataFrame(index=intensity_cols)
        obs_df.index.name = "sample_id"

        # Load sample metadata if available
        if sample_meta_files:
            try:
                sample_df = pd.read_csv(
                    sample_meta_files[0], sep="\t", low_memory=False
                )
                # Try to match sample names to metadata rows
                if "Sample Name" in sample_df.columns:
                    sample_df = sample_df.set_index("Sample Name")
                    # Merge with obs, keeping only matching samples
                    common_samples = obs_df.index.intersection(sample_df.index)
                    if len(common_samples) > 0:
                        obs_df = obs_df.join(sample_df.loc[common_samples], how="left")
                        logger.info(
                            f"Merged sample metadata for "
                            f"{len(common_samples)} samples"
                        )
            except Exception as e:
                logger.warning(f"Could not parse sample metadata file: {e}")

        # Create AnnData
        adata = ad.AnnData(
            X=X.astype(np.float32),
            obs=obs_df,
            var=var_df,
        )

        # Store raw intensities as a layer
        adata.layers["raw_intensity"] = X.astype(np.float32)

        logger.info(
            f"Created AnnData from MAF: {adata.n_obs} samples x "
            f"{adata.n_vars} metabolites"
        )

        return adata

    def _create_file_reference_anndata(
        self,
        local_files: List[Path],
        dataset_id: str,
        strategy: str,
    ) -> ad.AnnData:
        """
        Create a minimal AnnData referencing downloaded raw files.

        For mzML and RAW strategies, the downloaded files require further
        processing (peak detection, alignment). This method creates a
        placeholder AnnData that references the file paths.

        Args:
            local_files: List of downloaded file paths
            dataset_id: MTBLS accession
            strategy: Strategy name used

        Returns:
            AnnData with file references in uns
        """
        # Create one observation per file
        obs_df = pd.DataFrame(
            {
                "filename": [f.name for f in local_files],
                "file_path": [str(f) for f in local_files],
                "file_size_bytes": [f.stat().st_size for f in local_files],
                "file_type": [f.suffix.lstrip(".") for f in local_files],
            },
            index=[f.stem for f in local_files],
        )

        # Minimal X matrix
        X = np.zeros((len(local_files), 1), dtype=np.float32)

        var_df = pd.DataFrame(
            {"feature": ["placeholder"]},
            index=["unprocessed"],
        )

        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
        adata.uns["file_references"] = [str(f) for f in local_files]
        adata.uns["requires_processing"] = True
        adata.uns["processing_note"] = (
            f"Raw {strategy.replace('_FIRST', '')} files downloaded. "
            f"Requires peak detection and alignment before analysis."
        )

        logger.info(
            f"Created file-reference AnnData for {len(local_files)} "
            f"{strategy} files"
        )

        return adata

    # =========================================================================
    # PROVENANCE
    # =========================================================================

    def _create_download_ir(
        self, dataset_id: str, strategy: str, params: Dict[str, Any]
    ) -> AnalysisStep:
        """
        Create AnalysisStep IR for MetaboLights download provenance.

        Args:
            dataset_id: MTBLS accession
            strategy: Strategy name used
            params: Strategy parameters

        Returns:
            AnalysisStep for provenance tracking
        """
        return AnalysisStep(
            operation="metabolights.download.download_dataset",
            tool_name="MetaboLightsDownloadService",
            description=f"Download MetaboLights study {dataset_id}",
            library="lobster.services.data_access.metabolights_download_service",
            code_template="""# MetaboLights dataset download
import requests
import pandas as pd
import numpy as np
import anndata as ad

# Fetch study files
dataset_id = {{ dataset_id | tojson }}
api_url = f"https://www.ebi.ac.uk/metabolights/ws/studies/{dataset_id}/files"
response = requests.get(api_url)
files = response.json()

# Download MAF (Metabolite Assignment File)
# MAF files contain the processed intensity matrix
maf_url = f"https://www.ebi.ac.uk/metabolights/{dataset_id}/files/<maf_filename>"
maf_data = pd.read_csv(maf_url, sep="\\t")

# Parse MAF into AnnData (transpose: metabolites x samples -> samples x metabolites)
# Identify annotation vs intensity columns, build AnnData
# adata = ad.AnnData(X=intensity_matrix, obs=sample_metadata, var=metabolite_annotations)
""",
            imports=[
                "import requests",
                "import pandas as pd",
                "import numpy as np",
                "import anndata as ad",
            ],
            parameters={
                "dataset_id": dataset_id,
                "strategy": strategy,
                **params,
            },
            parameter_schema={
                "dataset_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="MetaboLights accession (MTBLS format)",
                ),
            },
            input_entities=[dataset_id],
            output_entities=[f"metabolights_{dataset_id.lower()}_metabolomics"],
        )


# =========================================================================
# MODULE-LEVEL HELPERS
# =========================================================================


def _make_unique(names: List[str]) -> List[str]:
    """
    Make a list of names unique by appending suffixes to duplicates.

    Args:
        names: List of potentially duplicate names

    Returns:
        List of unique names
    """
    seen = {}
    result = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result
