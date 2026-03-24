"""
GEO archive processing - TAR extraction, nested archives, 10X handling.

Extracted from geo_service.py as part of Phase 4 GEO Service Decomposition.
Contains 8 methods that handle supplementary file processing, TAR extraction,
nested archive handling, Kallisto/Salmon quantification files, and 10X data.
"""

import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import anndata
import pandas as pd

from lobster.services.data_access.geo.downloader import GEODownloadError
from lobster.services.data_access.geo.helpers import (
    _is_archive_url,
    _is_unsupported_modality_file,
    _score_expression_file,
)
from lobster.services.data_access.geo.parser import ParseResult
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ArchiveProcessor:
    """GEO archive processing - TAR extraction, nested archives, 10X handling.

    Handles all archive-related operations including:
    - Supplementary file processing (TAR, expression files, 10X trio)
    - TAR archive download and extraction with security checks
    - Nested TAR.GZ extraction for 10X samples
    - Kallisto/Salmon quantification file detection and loading
    - Single file download and parsing
    """

    def __init__(self, service):
        """Initialize with reference to parent GEOService.

        Args:
            service: Parent GEOService instance providing shared state
        """
        self.service = service

    def _process_supplementary_files(
        self, gse, gse_id: str, multimodal_info: Optional[Dict] = None
    ) -> Optional[Union[pd.DataFrame, anndata.AnnData]]:
        """
        Process supplementary files (TAR archives, etc.) to extract expression data.

        This method now supports:
        - Series-level 10x trio files (new: for multi-modal datasets like GSE150825)
        - Kallisto/Salmon quantification files (new in Phase 3)
        - TAR archives
        - Direct expression files

        Args:
            gse: GEOparse GSE object
            gse_id: GEO series ID
            multimodal_info: Optional multimodal context from metadata enrichment

        Returns:
            DataFrame or AnnData: Combined expression matrix from supplementary files or None
        """
        try:
            if not hasattr(gse, "metadata") or "supplementary_file" not in gse.metadata:
                logger.debug(f"No supplementary files found for {gse_id}")
                return None

            suppl_files = gse.metadata["supplementary_file"]
            if not isinstance(suppl_files, list):
                suppl_files = [suppl_files]

            # Convert NCBI's FTP URLs to HTTPS for reliable downloads
            suppl_files = [
                (
                    url.replace("ftp://", "https://", 1)
                    if url.startswith("ftp://")
                    else url
                )
                for url in suppl_files
            ]

            logger.debug(f"Found {len(suppl_files)} supplementary files for {gse_id}")

            # STEP 1: Check for Kallisto/Salmon quantification files FIRST
            has_quant, tool_type, quant_filenames, estimated_samples = (
                self._detect_kallisto_salmon_files(suppl_files)
            )

            if has_quant:
                logger.info(
                    f"{gse_id}: Detected {tool_type} quantification files "
                    f"({estimated_samples} estimated samples)"
                )

                # STEP 2: Check for pre-merged matrix files as alternative
                matrix_files = [
                    f
                    for f in suppl_files
                    if any(
                        ext in f.lower()
                        for ext in [
                            "_matrix.txt",
                            "_counts.txt",
                            "_expression.txt",
                            ".h5ad",
                            "_tpm.txt",
                            "_fpkm.txt",
                        ]
                    )
                ]

                if matrix_files:
                    logger.info(
                        f"{gse_id}: Found {len(matrix_files)} pre-merged matrix files "
                        f"alongside quantification. Using matrix files (faster loading)."
                    )
                else:
                    logger.info(
                        f"{gse_id}: No pre-merged matrix found. "
                        f"Loading {tool_type} quantification files..."
                    )

                    logger.info(
                        f"{gse_id}: Quantification file loading requires TAR extraction. "
                        f"Proceeding with TAR file processing."
                    )

            # STEP 2: Check for series-level 10x trio files
            series_10x_result = self.service._try_series_level_10x_trio(
                suppl_files, gse_id
            )
            if series_10x_result is not None:
                logger.info(
                    f"{gse_id}: Successfully loaded series-level 10x trio files"
                )
                return series_10x_result

            # Look for TAR files first (most common for expression data)
            tar_files = [f for f in suppl_files if _is_archive_url(f)]

            if tar_files:
                logger.debug(f"Processing TAR file: {tar_files[0]}")
                return self._process_tar_file(
                    tar_files[0], gse_id, multimodal_info=multimodal_info
                )

            # Look for other expression data files using scoring heuristic
            candidate_files = [
                f
                for f in suppl_files
                if any(
                    ext in f.lower()
                    for ext in [".txt.gz", ".csv.gz", ".tsv.gz", ".h5", ".h5ad"]
                )
            ]

            # Filter out unsupported modality files (e.g. ATAC in multiome datasets)
            if multimodal_info and multimodal_info.get("is_multimodal"):
                unsupported = multimodal_info.get("unsupported_types", [])
                if unsupported:
                    candidate_files = [
                        f
                        for f in candidate_files
                        if not _is_unsupported_modality_file(
                            f.split("/")[-1], unsupported
                        )
                    ]

            scored_files = [
                (f, _score_expression_file(f.split("/")[-1])) for f in candidate_files
            ]
            expression_files = [
                f for f, score in sorted(scored_files, key=lambda x: -x[1]) if score > 0
            ]

            if expression_files:
                logger.debug(
                    f"Processing expression file: {expression_files[0]} "
                    f"(selected from {len(candidate_files)} candidates via scoring heuristic)"
                )
                return self._download_and_parse_file(expression_files[0], gse_id)

            logger.warning(
                f"No suitable expression files found in supplementary files for {gse_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error processing supplementary files: {e}")
            return None

    def _process_tar_file(
        self, tar_url: str, gse_id: str, multimodal_info: Optional[Dict] = None
    ) -> Optional[Union[pd.DataFrame, anndata.AnnData]]:
        """
        Download and process a TAR file containing expression data.

        This method can return either DataFrame or AnnData depending on the data source:
        - Quantification files (Kallisto/Salmon): Returns AnnData directly
        - Other expression files: Returns DataFrame for adapter processing

        On exception, extraction directories are cleaned up to avoid leaving
        potentially gigabytes of data on disk.

        Args:
            tar_url: URL to TAR file
            gse_id: GEO series ID

        Returns:
            Union[DataFrame, AnnData]: Expression data or None if processing fails
        """
        # Define extraction directories BEFORE try so they're visible in except
        extract_dir = self.service.cache_dir / f"{gse_id}_extracted"
        nested_extract_dir = self.service.cache_dir / f"{gse_id}_nested_extracted"

        try:
            # Download TAR file
            tar_file_path = self.service.cache_dir / f"{gse_id}_RAW.tar"

            if not tar_file_path.exists():
                logger.debug(f"Downloading TAR file from: {tar_url}")

                # Convert FTP to HTTPS for reliability
                if tar_url.startswith("ftp://"):
                    tar_url = tar_url.replace("ftp://", "https://", 1)
                    logger.debug(f"Converted FTP to HTTPS: {tar_url}")

                if not self.service.geo_downloader.download_file(
                    tar_url, tar_file_path
                ):
                    raise GEODownloadError(f"Failed to download TAR file: {tar_url}")

                logger.debug(f"Downloaded TAR file: {tar_file_path}")
            else:
                logger.debug(f"Using cached TAR file: {tar_file_path}")

            # Extract TAR file
            if not extract_dir.exists():
                logger.info(f"Extracting TAR file to: {extract_dir}")
                extract_dir.mkdir(exist_ok=True)

                with tarfile.open(tar_file_path, "r") as tar:
                    # Security check for path traversal
                    def is_safe_member(member):
                        member_path = Path(member.name)
                        try:
                            target_path = (extract_dir / member_path).resolve()
                            common_path = Path(
                                os.path.commonpath([extract_dir.resolve(), target_path])
                            )
                            return common_path == extract_dir.resolve()
                        except (ValueError, RuntimeError):
                            return False

                    safe_members = [m for m in tar.getmembers() if is_safe_member(m)]
                    tar.extractall(path=extract_dir, members=safe_members)

                logger.debug(f"Extracted {len(safe_members)} files from TAR")

            # STEP 1: Check for Kallisto/Salmon quantification files in extracted directory
            try:
                logger.debug(f"Checking for quantification files in {extract_dir}")

                # Lazy import to avoid heavy deps at module level
                from lobster.services.analysis.bulk_rnaseq_service import (
                    BulkRNASeqService,
                )

                bulk_service = BulkRNASeqService()
                tool_type = bulk_service._detect_quantification_tool(extract_dir)

                logger.info(
                    f"{gse_id}: Detected {tool_type} quantification files in TAR archive"
                )

                adata_result = self._load_quantification_files(
                    quantification_dir=extract_dir,
                    tool_type=tool_type,
                    gse_id=gse_id,
                    data_type="bulk",
                )

                if adata_result is not None:
                    logger.info(
                        f"{gse_id}: Successfully loaded quantification files: "
                        f"{adata_result.n_obs} samples x {adata_result.n_vars} genes"
                    )
                    return adata_result
                else:
                    logger.warning(
                        f"{gse_id}: Quantification file loading returned None, "
                        f"falling back to standard processing"
                    )

            except ValueError as e:
                logger.debug(
                    f"{gse_id}: No quantification files detected in TAR: {e}. "
                    f"Continuing with standard TAR processing."
                )
            except Exception as e:
                logger.warning(
                    f"{gse_id}: Error during quantification file detection/loading: {e}. "
                    f"Falling back to standard TAR processing."
                )

            # STEP 2: Process nested archives and find expression data
            nested_extract_dir.mkdir(exist_ok=True)

            # Extract any nested TAR.GZ files
            nested_archives = list(extract_dir.glob("*.tar.gz"))
            if nested_archives:
                n_samples = len(nested_archives)
                logger.info(f"Found {n_samples} nested TAR.GZ files (10X samples)")

                # MEMORY ESTIMATION
                estimated_sparse_mb = n_samples * 50
                try:
                    import psutil

                    available_mb = psutil.virtual_memory().available / 1024 / 1024
                    if estimated_sparse_mb > available_mb * 0.5:
                        logger.warning(
                            f"Memory warning: {n_samples} samples may use ~{estimated_sparse_mb:.0f} MB RAM. "
                            f"Available: {available_mb:.0f} MB. If processing fails, try reducing sample count."
                        )
                except ImportError:
                    logger.debug("psutil not available for memory estimation")

                all_matrices = []
                for i, archive_path in enumerate(nested_archives, start=1):
                    try:
                        sample_id = archive_path.stem.split(".")[0]
                        sample_extract_dir = nested_extract_dir / sample_id
                        sample_extract_dir.mkdir(exist_ok=True)

                        logger.info(f"Processing sample {i}/{n_samples}: {sample_id}")

                        logger.debug(f"Extracting nested archive: {archive_path.name}")
                        with tarfile.open(archive_path, "r:gz") as nested_tar:
                            nested_tar.extractall(path=sample_extract_dir)

                        adata_sample = self.service.geo_parser.parse_10x_data(
                            sample_extract_dir, sample_id
                        )
                        if adata_sample is not None:
                            all_matrices.append(adata_sample)
                            logger.info(
                                f"Sample {i}/{n_samples} parsed: {adata_sample.n_obs} cells x {adata_sample.n_vars} genes"
                            )
                        else:
                            logger.warning(
                                f"Sample {i}/{n_samples} returned None (skipping)"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract/parse sample {i}/{n_samples} ({archive_path.name}): {e}"
                        )
                        continue

                if all_matrices:
                    logger.info(f"Concatenating {len(all_matrices)} AnnData objects...")
                    try:
                        import anndata

                        combined_adata = anndata.concat(
                            all_matrices,
                            axis=0,
                            join="outer",
                            label="sample_id",
                            keys=[
                                adata.obs["sample_id"].iloc[0] for adata in all_matrices
                            ],
                            index_unique=None,
                        )
                        logger.info(
                            f"Combined dataset: {combined_adata.n_obs} cells x {combined_adata.n_vars} genes "
                            f"(sparse: {(combined_adata.X.data.nbytes + combined_adata.X.indices.nbytes + combined_adata.X.indptr.nbytes) / 1024 / 1024:.1f} MB)"
                        )
                        return combined_adata
                    except Exception as e:
                        logger.error(f"Failed to concatenate AnnData objects: {e}")
                        logger.warning("Returning first sample only as fallback")
                        return all_matrices[0] if all_matrices else None

            # Fallback: look for regular expression files
            expression_files = []
            for file_path in extract_dir.rglob("*"):
                if file_path.is_file() and any(
                    ext in file_path.name.lower()
                    for ext in [".txt", ".csv", ".tsv", ".gz"]
                ):
                    if file_path.stat().st_size > 100000:
                        expression_files.append(file_path)

            # Filter out unsupported modality files (e.g. ATAC in multiome datasets)
            if multimodal_info and multimodal_info.get("is_multimodal"):
                unsupported = multimodal_info.get("unsupported_types", [])
                if unsupported:
                    before_count = len(expression_files)
                    expression_files = [
                        f
                        for f in expression_files
                        if not _is_unsupported_modality_file(f.name, unsupported)
                    ]
                    filtered = before_count - len(expression_files)
                    if filtered:
                        logger.info(
                            f"Filtered {filtered} unsupported modality files "
                            f"(unsupported: {unsupported})"
                        )

            if expression_files:
                logger.debug(
                    f"Found {len(expression_files)} potential expression files"
                )

                expression_files.sort(key=lambda x: x.stat().st_size, reverse=True)

                for file_path in expression_files[:3]:
                    try:
                        logger.debug(f"Attempting to parse: {file_path.name}")
                        parse_result = self.service.geo_parser.parse_expression_file(
                            file_path
                        )
                        matrix = (
                            parse_result.data
                            if isinstance(parse_result, ParseResult)
                            else parse_result
                        )
                        if (
                            isinstance(parse_result, ParseResult)
                            and parse_result.is_partial
                        ):
                            logger.warning(
                                f"Partial parse result for {gse_id}: {parse_result.truncation_reason} "
                                f"({parse_result.rows_read:,} rows read)"
                            )
                        if (
                            matrix is not None
                            and matrix.shape[0] > 0
                            and matrix.shape[1] > 0
                        ):
                            logger.debug(
                                f"Successfully parsed expression matrix: {matrix.shape}"
                            )
                            return matrix
                    except Exception as e:
                        logger.warning(f"Failed to parse {file_path.name}: {e}")
                        continue

            logger.warning(
                f"Could not parse any expression files from TAR for {gse_id}"
            )
            return None

        except Exception as e:
            logger.error(f"Error processing TAR file: {e}")
            for cleanup_dir in [extract_dir, nested_extract_dir]:
                if cleanup_dir.exists():
                    shutil.rmtree(cleanup_dir, ignore_errors=True)
                    logger.debug(
                        f"Cleaned up failed extraction directory: {cleanup_dir}"
                    )
            return None

    def _download_and_parse_file(
        self, file_url: str, gse_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download and parse a single expression file with progress tracking.

        Args:
            file_url: URL to expression file
            gse_id: GEO series ID

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            file_name = file_url.split("/")[-1]
            local_file = self.service.cache_dir / f"{gse_id}_{file_name}"

            if not local_file.exists():
                logger.info(f"Downloading file: {file_url}")
                if not self.service.geo_downloader.download_file(file_url, local_file):
                    logger.error(f"Failed to download file: {file_url}")
                    return None
                logger.debug(f"Successfully downloaded: {local_file}")
            else:
                logger.debug(f"Using cached file: {local_file}")

            parse_result = self.service.geo_parser.parse_expression_file(local_file)
            matrix = (
                parse_result.data
                if isinstance(parse_result, ParseResult)
                else parse_result
            )
            if isinstance(parse_result, ParseResult) and parse_result.is_partial:
                logger.warning(
                    f"Partial parse result for {gse_id}: {parse_result.truncation_reason} "
                    f"({parse_result.rows_read:,} rows read)"
                )
            return matrix

        except Exception as e:
            logger.error(f"Error downloading and parsing file: {e}")
            return None

    def _detect_kallisto_salmon_files(
        self,
        supplementary_files: List[str],
    ) -> Tuple[bool, str, List[str], int]:
        """
        Detect if dataset contains Kallisto/Salmon per-sample quantification files.

        Args:
            supplementary_files: List of supplementary file URLs/paths

        Returns:
            Tuple of (has_quant_files, tool_type, matched_filenames, estimated_samples)
        """
        kallisto_patterns = [
            "abundance.tsv",
            "abundance.h5",
            "abundance.txt",
        ]

        salmon_patterns = [
            "quant.sf",
            "quant.genes.sf",
        ]

        kallisto_files = []
        salmon_files = []
        abundance_files = []

        for file_path in supplementary_files:
            filename = os.path.basename(file_path).lower()

            if any(pattern in filename for pattern in kallisto_patterns):
                kallisto_files.append(file_path)
                if "abundance.tsv" in filename or "abundance.h5" in filename:
                    abundance_files.append(filename)

            if any(pattern in filename for pattern in salmon_patterns):
                salmon_files.append(file_path)
                if "quant.sf" in filename:
                    abundance_files.append(filename)

        has_quant = len(kallisto_files) > 0 or len(salmon_files) > 0

        if not has_quant:
            return False, "", [], 0

        if len(kallisto_files) > 0 and len(salmon_files) > 0:
            tool_type = "mixed"
            matched = kallisto_files + salmon_files
        elif len(kallisto_files) > 0:
            tool_type = "kallisto"
            matched = kallisto_files
        else:
            tool_type = "salmon"
            matched = salmon_files

        estimated_samples = len(abundance_files)

        return has_quant, tool_type, matched, estimated_samples

    def _load_quantification_files(
        self,
        quantification_dir: Path,
        tool_type: str,
        gse_id: str,
        data_type: str = "bulk",
    ) -> Optional[anndata.AnnData]:
        """
        Load Kallisto/Salmon quantification files into AnnData.

        Args:
            quantification_dir: Directory containing per-sample subdirectories
            tool_type: "kallisto" or "salmon"
            gse_id: GEO series ID
            data_type: Data type for TranscriptomicsAdapter

        Returns:
            AnnData: Processed AnnData object or None if loading fails
        """
        try:
            logger.info(
                f"Loading {tool_type} quantification files from {quantification_dir}"
            )

            # Lazy imports to avoid heavy deps at module level
            from lobster.core.adapters.transcriptomics_adapter import (
                TranscriptomicsAdapter,
            )
            from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

            bulk_service = BulkRNASeqService(data_manager=self.service.data_manager)

            try:
                df, metadata = bulk_service.load_from_quantification_files(
                    quantification_dir=quantification_dir,
                    tool=tool_type,
                )
                logger.info(
                    f"Successfully merged {metadata['n_samples']} {tool_type} samples "
                    f"x {metadata['n_genes']} genes"
                )
            except Exception as e:
                logger.error(f"Failed to merge {tool_type} files: {e}")
                raise

            adapter = TranscriptomicsAdapter(data_type=data_type)

            try:
                adata = adapter.from_quantification_dataframe(
                    df=df,
                    data_type="bulk_rnaseq",
                    metadata=metadata,
                )
                logger.info(
                    f"Created AnnData from quantification: "
                    f"{adata.n_obs} samples x {adata.n_vars} genes"
                )
            except Exception as e:
                logger.error(f"Failed to create AnnData from quantification: {e}")
                raise

            adata.uns["geo_metadata"] = {
                "geo_id": gse_id,
                "data_source": "quantification_files",
                "quantification_tool": tool_type,
                "n_files_merged": metadata["n_samples"],
            }

            logger.info(
                f"Successfully loaded {gse_id} from {tool_type} files: "
                f"{adata.n_obs} samples x {adata.n_vars} genes"
            )

            return adata

        except Exception as e:
            logger.error(f"Error loading quantification files: {e}")
            logger.exception("Full traceback for quantification loading error:")
            return None

    def _try_series_level_10x_trio(
        self, suppl_files: List[str], gse_id: str
    ) -> Optional[anndata.AnnData]:
        """
        Check for and process series-level 10x trio files.

        Args:
            suppl_files: List of supplementary file URLs
            gse_id: GEO series ID

        Returns:
            AnnData: Loaded 10x data or None if not found/failed
        """
        # Delegate to modular TenXGenomicsLoader
        return self.service.tenx_loader.try_series_level_10x_trio(suppl_files, gse_id)

    def _download_10x_trio(
        self, files_info: Dict[str, str], gsm_id: str
    ) -> Optional[pd.DataFrame]:
        """
        Download and combine 10X format trio (matrix, barcodes, features).

        Args:
            files_info: Dictionary with 'matrix', 'barcodes', 'features' URLs
            gsm_id: GEO sample ID

        Returns:
            DataFrame: Combined 10X expression matrix or None
        """
        try:
            logger.debug(f"Processing 10X trio for {gsm_id}")

            local_files = {}
            for file_type, url in files_info.items():
                if file_type in ["matrix", "barcodes", "features"]:
                    filename = url.split("/")[-1]
                    local_path = (
                        self.service.cache_dir / f"{gsm_id}_{file_type}_{filename}"
                    )

                    if not local_path.exists():
                        logger.debug(f"Downloading {file_type} file: {url}")
                        if self.service.geo_downloader.download_file(url, local_path):
                            local_files[file_type] = local_path
                        else:
                            logger.error(f"Failed to download {file_type} file")
                            return None
                    else:
                        logger.debug(f"Using cached {file_type} file: {local_path}")
                        local_files[file_type] = local_path

            if len(local_files) != 3:
                logger.error(f"Could not download all three 10X files for {gsm_id}")
                return None

            try:
                import scipy.io as sio
            except ImportError:
                logger.error(
                    "scipy is required for parsing 10X matrix files but not available"
                )
                return None

            matrix_file = local_files["matrix"]
            logger.info(f"Parsing matrix file: {matrix_file}")

            try:
                if matrix_file.name.endswith(".gz"):
                    import gzip

                    with gzip.open(matrix_file, "rt") as f:
                        matrix = sio.mmread(f)
                else:
                    matrix = sio.mmread(matrix_file)

            except (Exception,) as e:
                # Handle gzip and EOF errors
                error_name = type(e).__name__
                if error_name in ("BadGzipFile", "EOFError"):
                    logger.error(f"Gzip corruption detected for {gsm_id}: {e}")
                    logger.error(f"Removing corrupted cache: {matrix_file}")
                    if matrix_file.exists():
                        matrix_file.unlink()
                    return None
                elif isinstance(e, OSError):
                    logger.error(f"File I/O error processing {gsm_id}: {e}")
                    logger.error(f"Matrix file path: {matrix_file}")
                    return None
                raise

            if hasattr(matrix, "todense"):
                matrix_dense = matrix.todense()
            else:
                matrix_dense = matrix

            matrix_dense = matrix_dense.T
            logger.debug(f"Matrix shape after transpose: {matrix_dense.shape}")

            # Read barcodes
            barcodes_file = local_files["barcodes"]
            logger.debug(f"Reading barcodes from: {barcodes_file}")
            cell_ids = []

            try:
                import gzip

                if barcodes_file.name.endswith(".gz"):
                    with gzip.open(barcodes_file, "rt") as f:
                        cell_ids = [line.strip() for line in f]
                else:
                    with open(barcodes_file, "r") as f:
                        cell_ids = [line.strip() for line in f]
                logger.info(f"Read {len(cell_ids)} cell barcodes")
            except Exception as e:
                logger.warning(f"Error reading barcodes file: {e}")
                cell_ids = [f"{gsm_id}_cell_{i}" for i in range(matrix_dense.shape[0])]

            # Read features
            features_file = local_files["features"]
            logger.info(f"Reading features from: {features_file}")
            gene_ids = []
            gene_names = []

            try:
                import gzip

                if features_file.name.endswith(".gz"):
                    with gzip.open(features_file, "rt") as f:
                        lines = f.readlines()
                else:
                    with open(features_file, "r") as f:
                        lines = f.readlines()

                for line in lines:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        gene_ids.append(parts[0])
                        gene_names.append(parts[1])
                    elif len(parts) == 1:
                        gene_ids.append(parts[0])
                        gene_names.append(parts[0])

                logger.info(f"Read {len(gene_ids)} gene features")
            except Exception as e:
                logger.warning(f"Error reading features file: {e}")
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            if not cell_ids or len(cell_ids) != matrix_dense.shape[0]:
                cell_ids = [f"{gsm_id}_cell_{i}" for i in range(matrix_dense.shape[0])]
            else:
                cell_ids = [f"{gsm_id}_{cell_id}" for cell_id in cell_ids]

            if not gene_names or len(gene_names) != matrix_dense.shape[1]:
                gene_names = [f"Gene_{i}" for i in range(matrix_dense.shape[1])]

            df = pd.DataFrame(matrix_dense, index=cell_ids, columns=gene_names)

            if df.columns.duplicated().any():
                duplicates = df.columns[df.columns.duplicated()].unique()
                logger.warning(
                    f"{gsm_id}: Found {len(duplicates)} duplicate gene IDs. "
                    f"Aggregating by sum. Examples: {list(duplicates[:5])}"
                )
                original_shape = df.shape
                df = df.T.groupby(level=0).sum().T
                logger.info(
                    f"{gsm_id}: Aggregated duplicate genes. "
                    f"Shape: {original_shape} -> {df.shape}"
                )

            logger.info(f"Successfully created 10X DataFrame for {gsm_id}: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error processing 10X trio for {gsm_id}: {e}")
            return None

    def _download_h5_file(self, url: str, gsm_id: str) -> Optional[pd.DataFrame]:
        """
        Download and parse H5 format single-cell data.

        Args:
            url: URL to H5 file
            gsm_id: GEO sample ID

        Returns:
            DataFrame: Expression matrix or None
        """
        try:
            logger.info(f"Processing H5 file for {gsm_id}")

            filename = url.split("/")[-1]
            local_path = self.service.cache_dir / f"{gsm_id}_h5_{filename}"

            if not local_path.exists():
                logger.info(f"Downloading H5 file: {url}")
                if not self.service.geo_downloader.download_file(url, local_path):
                    logger.error("Failed to download H5 file")
                    return None
            else:
                logger.info(f"Using cached H5 file: {local_path}")

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
                matrix.index = [f"{gsm_id}_{idx}" for idx in matrix.index]
                logger.info(f"Successfully parsed H5 file for {gsm_id}: {matrix.shape}")
                return matrix

            return None

        except Exception as e:
            logger.error(f"Error processing H5 file for {gsm_id}: {e}")
            return None
