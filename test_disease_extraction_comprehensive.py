#!/usr/bin/env python3
"""
Comprehensive disease extraction testing with accurate classification.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_disease_from_raw_fields(metadata: pd.DataFrame, study_context=None):
    """Extract disease from diverse SRA field names."""
    # Strategy 1: Existing disease column
    existing_disease_cols = ["disease", "disease_state", "condition", "diagnosis"]
    for col in existing_disease_cols:
        if col in metadata.columns and metadata[col].notna().sum() > 0:
            if col != "disease":
                metadata["disease"] = metadata[col]
            return "disease"

    # Strategy 2: Phenotype fields
    phenotype_cols = ["host_phenotype", "phenotype", "host_disease", "health_status"]
    for col in phenotype_cols:
        if col in metadata.columns:
            populated_count = metadata[col].notna().sum()
            if populated_count > 0:
                metadata["disease"] = metadata[col].fillna("unknown")
                return "disease"

    # Strategy 3: Boolean disease flags
    disease_flag_cols = [c for c in metadata.columns if c.endswith("_disease")]

    if disease_flag_cols:
        def extract_from_flags(row):
            active_diseases = []
            for flag_col in disease_flag_cols:
                flag_value = row.get(flag_col)
                if flag_value in ["Yes", "YES", "yes", "TRUE", "True", "true", True, 1, "1"]:
                    disease_name = flag_col.replace("_disease", "").replace("_", "")
                    disease_map = {
                        "crohns": "cd",
                        "crohn": "cd",
                        "inflammbowel": "ibd",
                        "inflambowel": "ibd",
                        "ulcerativecolitis": "uc",
                        "colitis": "uc",
                        "parkinson": "parkinsons",
                        "parkinsons": "parkinsons",
                    }
                    standardized = disease_map.get(disease_name, disease_name)
                    active_diseases.append(standardized)

            if active_diseases:
                return ";".join(active_diseases)

            all_false = all(
                row.get(flag_col) in ["No", "NO", "no", "FALSE", "False", "false", False, 0, "0"]
                for flag_col in disease_flag_cols
            )
            if all_false:
                return "healthy"

            return "unknown"

        metadata["disease"] = metadata.apply(extract_from_flags, axis=1)
        return "disease"

    return None


def load_metadata_file(file_path: Path) -> pd.DataFrame:
    """Load metadata from JSON file into DataFrame."""
    try:
        with open(file_path) as f:
            data = json.load(f)
            samples = data.get("data", {}).get("samples", [])
            if not samples:
                return pd.DataFrame()
            return pd.DataFrame(samples)
    except Exception as e:
        print(f"  Error loading {file_path.name}: {e}")
        return pd.DataFrame()


def test_file(file_path: Path) -> Dict[str, Any]:
    """Test disease extraction on a single file."""
    df = load_metadata_file(file_path)
    if df.empty:
        return None

    total_samples = len(df)

    # BEFORE extraction
    before_count = df["disease"].notna().sum() if "disease" in df.columns else 0

    # APPLY extraction
    try:
        disease_col = _extract_disease_from_raw_fields(df)
        extraction_successful = disease_col is not None
    except Exception as e:
        return {
            "file": file_path.name,
            "samples": total_samples,
            "extraction_successful": False,
            "error": str(e)
        }

    if not extraction_successful:
        return {
            "file": file_path.name,
            "samples": total_samples,
            "before_count": before_count,
            "after_count": 0,
            "disease_count": 0,
            "healthy_count": 0,
            "unknown_count": 0,
            "extraction_successful": False
        }

    # AFTER extraction - categorize results
    value_counts = df["disease"].value_counts().to_dict()

    # Count actual diseases (not "healthy" or "unknown")
    disease_samples = df[~df["disease"].isin(["healthy", "unknown"])]
    disease_count = len(disease_samples)

    healthy_count = value_counts.get("healthy", 0)
    unknown_count = value_counts.get("unknown", 0)
    after_count = before_count + disease_count + healthy_count  # Total with info

    return {
        "file": file_path.name,
        "samples": total_samples,
        "before_count": before_count,
        "after_count": after_count,
        "disease_count": disease_count,
        "healthy_count": healthy_count,
        "unknown_count": unknown_count,
        "extraction_successful": True,
        "unique_diseases": len(disease_samples["disease"].unique()) if disease_count > 0 else 0
    }


def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE DISEASE EXTRACTION TEST - REAL DATA")
    print("="*80)

    workspace_path = Path("/Users/tyo/GITHUB/omics-os/lobster/.lobster_workspace/metadata")
    sra_files = sorted(workspace_path.glob("sra_*_samples.json"))

    print(f"\nTesting 30 real SRA metadata files...")

    results = []
    for file_path in sra_files[:30]:
        result = test_file(file_path)
        if result:
            results.append(result)
            if result["disease_count"] > 0:
                print(f"✓ {result['file']}: {result['disease_count']}/{result['samples']} with disease")

    # Aggregate statistics
    total_samples = sum(r["samples"] for r in results)
    total_before = sum(r["before_count"] for r in results)
    total_disease = sum(r["disease_count"] for r in results)
    total_healthy = sum(r["healthy_count"] for r in results)
    total_unknown = sum(r["unknown_count"] for r in results)

    # Coverage calculations
    before_pct = (total_before / total_samples * 100) if total_samples > 0 else 0
    disease_pct = (total_disease / total_samples * 100) if total_samples > 0 else 0
    healthy_pct = (total_healthy / total_samples * 100) if total_samples > 0 else 0
    informative_pct = ((total_disease + total_healthy) / total_samples * 100) if total_samples > 0 else 0

    print("\n" + "="*80)
    print("MEASURED RESULTS - NO PROJECTIONS")
    print("="*80)

    print(f"\nFiles tested: {len(results)}")
    print(f"Total samples: {total_samples:,}")

    print(f"\n**BEFORE extraction:**")
    print(f"  Disease coverage: {total_before}/{total_samples:,} ({before_pct:.1f}%)")

    print(f"\n**AFTER extraction:**")
    print(f"  Samples with disease: {total_disease:,}/{total_samples:,} ({disease_pct:.1f}%)")
    print(f"  Healthy controls: {total_healthy:,}/{total_samples:,} ({healthy_pct:.1f}%)")
    print(f"  Unknown/missing: {total_unknown:,}/{total_samples:,}")
    print(f"  Total informative: {total_disease + total_healthy:,}/{total_samples:,} ({informative_pct:.1f}%)")

    print(f"\n**ACTUAL IMPROVEMENT:**")
    print(f"  Disease extraction: +{disease_pct - before_pct:.1f}%")
    print(f"  Total informative data: +{informative_pct - before_pct:.1f}%")

    # Top files with disease data
    print(f"\n" + "="*80)
    print("FILES WITH MOST DISEASE DATA (Top 10)")
    print("="*80)

    disease_files = sorted([r for r in results if r["disease_count"] > 0],
                          key=lambda x: x["disease_count"], reverse=True)[:10]

    for r in disease_files:
        disease_pct_file = (r["disease_count"] / r["samples"] * 100)
        print(f"{r['file']:<45} {r['disease_count']:>5}/{r['samples']:<6} ({disease_pct_file:>5.1f}%) - {r['unique_diseases']} diseases")

    # Success metrics
    files_with_disease = sum(1 for r in results if r["disease_count"] > 0)
    success_rate = (files_with_disease / len(results) * 100) if results else 0

    print(f"\n" + "="*80)
    print("EFFECTIVENESS ASSESSMENT")
    print("="*80)

    print(f"\nFiles with disease extracted: {files_with_disease}/{len(results)} ({success_rate:.1f}%)")

    if disease_pct >= 20:
        effectiveness = "EXCELLENT"
    elif disease_pct >= 10:
        effectiveness = "GOOD"
    elif disease_pct > 0:
        effectiveness = "MODERATE"
    else:
        effectiveness = "POOR"

    print(f"Overall effectiveness: {effectiveness}")
    print(f"\nBased on {len(results)} real files, {total_samples:,} real samples")
    print(f"Measured disease extraction: +{disease_pct - before_pct:.1f}%")


if __name__ == "__main__":
    main()
