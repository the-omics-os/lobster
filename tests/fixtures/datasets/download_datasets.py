"""
Configuration-driven dataset downloader.

Reads datasets.yml and downloads datasets based on structure.
Supports hybrid URLs: auto-constructs GEO URLs or uses explicit URLs.

Usage:
    python download_datasets.py --all
    python download_datasets.py --omics-type single_cell
    python download_datasets.py --omics-type bulk_rnaseq --analysis-type differential_expression
    python download_datasets.py --dataset GSE132044
"""

import argparse
import gzip
import logging
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Config-driven dataset downloader with 3-level nesting support."""

    def __init__(self, config_path: Path):
        """Initialize downloader from YAML config."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.datasets_dir = config_path.parent

    def construct_geo_url(self, geo_id: str) -> str:
        """Auto-construct GEO FTP URL from GEO ID."""
        series = geo_id[:-3] + "nnn"
        return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series}/{geo_id}/suppl/"

    def get_dataset_path(
        self, omics_type: str, analysis_type: str, platform: str, dataset_id: str
    ) -> Path:
        """Get output path for dataset following 3-level structure."""
        return self.datasets_dir / omics_type / analysis_type / platform / dataset_id

    def download_dataset(
        self,
        dataset: Dict[str, Any],
        omics_type: str,
        analysis_type: str,
        platform: str,
    ) -> None:
        """Download single dataset based on config."""
        dataset_id = dataset["id"]
        description = dataset.get("description", "")
        output_dir = self.get_dataset_path(
            omics_type, analysis_type, platform, dataset_id
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_id}")
        logger.info(f"Description: {description}")
        logger.info(f"Path: {output_dir.relative_to(self.datasets_dir)}")
        logger.info(f"{'='*60}")

        # Handle metadata-only datasets (for rejection tests)
        if dataset.get("metadata_only"):
            logger.info("Metadata-only dataset (for rejection testing)")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "README.txt", "w") as f:
                f.write(f"Dataset: {dataset_id}\n")
                f.write(f"Description: {description}\n")
                f.write(f"Platform: {dataset.get('platform', 'N/A')}\n")
                f.write(f"Expected Error: {dataset.get('expected_error', 'N/A')}\n")
            logger.info("✓ Created metadata file\n")
            return

        # Handle generated datasets (ambiguous test cases)
        if dataset.get("generated"):
            self._generate_ambiguous_dataset(dataset, output_dir.parent)
            return

        # Download real datasets
        if "url" in dataset:
            url = dataset["url"]  # Explicit URL (hybrid approach)
            logger.info(f"Using explicit URL: {url}")
        elif "geo_id" in dataset:
            url = self.construct_geo_url(dataset["geo_id"])  # Auto-construct
            logger.info(f"Constructed URL from GEO ID: {url}")
        else:
            raise ValueError(f"Dataset {dataset_id} needs 'url' or 'geo_id'")

        # Download files
        output_dir.mkdir(parents=True, exist_ok=True)
        self._download_from_url(url, output_dir)

        # Extract archives
        for tar_file in output_dir.glob("*.tar"):
            self._extract_tar(tar_file, output_dir)
            tar_file.unlink()

        logger.info(f"✓ Completed {dataset_id}\n")

    def _download_from_url(self, url: str, output_dir: Path) -> None:
        """Download from URL using wget or curl."""
        try:
            cmd = f"wget -r -np -nd -P {output_dir} {url}"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.warning("wget failed, trying curl...")
            try:
                cmd = f"curl -L -o {output_dir}/download.tar {url}"
                subprocess.run(cmd, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Download failed: {e}")
                raise

    def _extract_tar(self, tar_path: Path, output_dir: Path) -> None:
        """Extract tar/tar.gz files."""
        logger.info(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=output_dir)

    def _generate_ambiguous_dataset(self, dataset: Dict, output_dir: Path) -> None:
        """Generate ambiguous test datasets."""
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_id = dataset["id"]
        shape = dataset.get("shape", [100, 100])

        logger.info(f"Generating {dataset_id} with shape {shape}")

        if "with_headers" in dataset_id or dataset_id == "ambiguous_with_headers":
            # With generic headers
            data = np.random.poisson(lam=40, size=tuple(shape))
            df = pd.DataFrame(
                data,
                columns=[f"Col{i}" for i in range(shape[1])],
                index=[f"Row{i}" for i in range(shape[0])],
            )
            filepath = output_dir / f"{dataset_id}.csv"
            df.to_csv(filepath)
        else:
            # No headers
            data = np.random.poisson(lam=50, size=tuple(shape))
            df = pd.DataFrame(data)
            filepath = output_dir / f"{dataset_id}.csv"
            df.to_csv(filepath, index=False, header=False)

        logger.info(f"✓ Generated {filepath.name}\n")

    def download_all(
        self, omics_type: Optional[str] = None, analysis_type: Optional[str] = None
    ) -> None:
        """Download all datasets or filter by type."""
        for omic_key, omic_data in self.config.items():
            if omics_type and omic_key != omics_type:
                continue

            logger.info(f"\n{'#'*60}")
            logger.info(f"# Omics Type: {omic_key}")
            logger.info(f"{'#'*60}")

            for analysis_key, analysis_data in omic_data.items():
                if analysis_type and analysis_key != analysis_type:
                    continue

                logger.info(f"\n## Analysis Type: {analysis_key}")

                for platform_key, datasets in analysis_data.items():
                    logger.info(f"\n### Platform: {platform_key}")

                    for dataset in datasets:
                        try:
                            self.download_dataset(
                                dataset, omic_key, analysis_key, platform_key
                            )
                        except Exception as e:
                            logger.error(f"Failed to download {dataset['id']}: {e}")
                            continue

    def download_single(self, dataset_id: str) -> bool:
        """Download a single dataset by ID."""
        for omic_key, omic_data in self.config.items():
            for analysis_key, analysis_data in omic_data.items():
                for platform_key, datasets in analysis_data.items():
                    for dataset in datasets:
                        if dataset["id"] == dataset_id:
                            self.download_dataset(
                                dataset, omic_key, analysis_key, platform_key
                            )
                            return True
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Configuration-driven dataset downloader"
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument(
        "--omics-type",
        help="Download specific omics type (single_cell, bulk_rnaseq, edge_cases)",
    )
    parser.add_argument(
        "--analysis-type",
        help="Download specific analysis type (requires --omics-type)",
    )
    parser.add_argument(
        "--dataset", help="Download specific dataset by ID (e.g., GSE132044)"
    )
    parser.add_argument(
        "--config", default="datasets.yml", help="Path to datasets.yml config file"
    )

    args = parser.parse_args()

    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    downloader = DatasetDownloader(config_path)

    if args.all:
        logger.info("Downloading ALL datasets...")
        downloader.download_all()
        logger.info("\n✅ All datasets downloaded successfully!")

    elif args.omics_type:
        logger.info(f"Downloading omics type: {args.omics_type}")
        if args.analysis_type:
            logger.info(f"  Analysis type filter: {args.analysis_type}")
        downloader.download_all(
            omics_type=args.omics_type, analysis_type=args.analysis_type
        )
        logger.info("\n✅ Download complete!")

    elif args.dataset:
        logger.info(f"Downloading dataset: {args.dataset}")
        if downloader.download_single(args.dataset):
            logger.info("\n✅ Dataset downloaded successfully!")
        else:
            logger.error(f"Dataset {args.dataset} not found in config")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
