#!/usr/bin/env python3
"""
CRITICAL MISSION: Measure ACTUAL Disease Extraction Improvements (NO PROJECTIONS)

This script tests the disease extraction system on REAL workspace metadata files
and reports MEASURED results only.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys

# Add lobster to path
sys.path.insert(0, str(Path(__file__).parent))

from lobster.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_disease_from_raw_fields(
    metadata: pd.DataFrame,
    study_context: Optional[Dict] = None
) -> Optional[str]:
    """
    Extract disease information from diverse, study-specific SRA field names.

    Copied from metadata_assistant.py for standalone testing.
    """
    # Strategy 1: Check for existing unified disease column
    existing_disease_cols = ["disease", "disease_state", "condition", "diagnosis"]
    for col in existing_disease_cols:
        if col in metadata.columns and metadata[col].notna().sum() > 0:
            logger.debug(f"Found existing disease column: {col}")
            # Rename to standard "disease" if different
            if col != "disease":
                metadata["disease"] = metadata[col]
                metadata["disease_original"] = metadata[col]
            return "disease"

    # Strategy 2: Extract from free-text phenotype fields
    phenotype_cols = ["host_phenotype", "phenotype", "host_disease", "health_status"]
    for col in phenotype_cols:
        if col in metadata.columns:
            # Count non-empty values
            populated_count = metadata[col].notna().sum()
            if populated_count > 0:
                # Create unified disease column from phenotype
                metadata["disease"] = metadata[col].fillna("unknown")
                metadata["disease_original"] = metadata[col].fillna("unknown")
                logger.debug(
                    f"Extracted disease from {col} "
                    f"({populated_count}/{len(metadata)} samples, "
                    f"{populated_count/len(metadata)*100:.1f}%)"
                )
                return "disease"

    # Strategy 3: Consolidate boolean disease flags
    # Find columns ending with "_disease" (crohns_disease, inflam_bowel_disease, etc.)
    disease_flag_cols = [c for c in metadata.columns if c.endswith("_disease")]

    if disease_flag_cols:
        logger.debug(f"Found {len(disease_flag_cols)} disease flag columns: {disease_flag_cols}")

        def extract_from_flags(row):
            """Extract disease from boolean flags."""
            active_diseases = []

            for flag_col in disease_flag_cols:
                # Check if flag is TRUE (handles Yes, TRUE, True, 1, "1")
                flag_value = row.get(flag_col)
                if flag_value in ["Yes", "YES", "yes", "TRUE", "True", "true", True, 1, "1"]:
                    # Convert flag name to disease term
                    disease_name = flag_col.replace("_disease", "").replace("_", "")

                    # Map common patterns to standard terms
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
                # If multiple diseases, join with semicolon
                return ";".join(active_diseases)

            # Check for negative controls (all flags FALSE)
            all_false = all(
                row.get(flag_col) in ["No", "NO", "no", "FALSE", "False", "false", False, 0, "0"]
                for flag_col in disease_flag_cols
            )
            if all_false:
                return "healthy"

            return "unknown"

        # Apply extraction
        metadata["disease"] = metadata.apply(extract_from_flags, axis=1)
        metadata["disease_original"] = metadata.apply(
            lambda row: ";".join([f"{col}={row[col]}" for col in disease_flag_cols if pd.notna(row.get(col))]),
            axis=1
        )

        # Count successful extractions
        extracted_count = (metadata["disease"] != "unknown").sum()
        logger.debug(
            f"Extracted disease from {len(disease_flag_cols)} boolean flags "
            f"({extracted_count}/{len(metadata)} samples, "
            f"{extracted_count/len(metadata)*100:.1f}%)"
        )

        return "disease"

    # Strategy 4: Use study context (publication-level disease focus)
    if study_context and "disease_focus" in study_context:
        # All samples in this study share the publication's disease focus
        metadata["disease"] = study_context["disease_focus"]
        metadata["disease_original"] = f"inferred from publication: {study_context['disease_focus']}"
        logger.debug(f"Assigned disease from publication context: {study_context['disease_focus']}")
        return "disease"

    logger.warning("No disease information found in metadata fields or study context")
    return None


def load_metadata_file(file_path: Path) -> pd.DataFrame:
    """Load metadata from JSON file into DataFrame."""
    with open(file_path) as f:
        data = json.load(f)
        samples = data.get("data", {}).get("samples", [])
        if not samples:
            return pd.DataFrame()
        return pd.DataFrame(samples)


def count_disease_coverage(df: pd.DataFrame) -> tuple[int, float]:
    """Count how many samples have disease information."""
    if "disease" not in df.columns:
        return 0, 0.0

    disease_count = df["disease"].notna().sum()
    percentage = (disease_count / len(df) * 100) if len(df) > 0 else 0.0
    return disease_count, percentage


def identify_disease_fields(df: pd.DataFrame) -> List[str]:
    """Identify which fields contain disease-related information."""
    disease_keywords = ['disease', 'phenotype', 'diagnosis', 'condition', 'health', 'status']
    disease_fields = []

    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in disease_keywords):
            # Check if field has any non-null values
            if df[col].notna().sum() > 0:
                disease_fields.append(col)

    return disease_fields


def test_file(file_path: Path) -> Dict[str, Any]:
    """Test disease extraction on a single metadata file."""
    print(f"\n{'='*80}")
    print(f"Testing: {file_path.name}")
    print(f"{'='*80}")

    # Load data
    df = load_metadata_file(file_path)
    if df.empty:
        print(f"  ⚠️  No samples found in file")
        return None

    total_samples = len(df)
    print(f"  Total samples: {total_samples}")

    # Identify disease fields BEFORE extraction
    disease_fields_before = identify_disease_fields(df)
    print(f"  Disease-related fields found: {disease_fields_before}")

    # Count BEFORE extraction
    before_count, before_pct = count_disease_coverage(df)
    print(f"  BEFORE extraction: {before_count}/{total_samples} ({before_pct:.1f}%)")

    # Store original disease column if exists
    original_disease_col = df["disease"].copy() if "disease" in df.columns else None

    # APPLY EXTRACTION (the actual function being tested)
    try:
        disease_col = _extract_disease_from_raw_fields(df, study_context=None)
        extraction_successful = disease_col is not None
    except Exception as e:
        print(f"  ❌ Extraction FAILED: {e}")
        return {
            "file": file_path.name,
            "samples": total_samples,
            "before_count": before_count,
            "before_pct": before_pct,
            "after_count": 0,
            "after_pct": 0.0,
            "improvement": 0.0,
            "extraction_successful": False,
            "error": str(e)
        }

    # Count AFTER extraction
    after_count, after_pct = count_disease_coverage(df)
    improvement = after_pct - before_pct

    print(f"  AFTER extraction: {after_count}/{total_samples} ({after_pct:.1f}%)")
    print(f"  IMPROVEMENT: +{improvement:.1f}%")
    print(f"  Extraction method: {disease_col if disease_col else 'NONE'}")

    # Determine extraction method
    extraction_method = "none"
    if extraction_successful and "disease" in df.columns:
        # Check which strategy was used
        if "disease_original" in df.columns:
            sample_original = df["disease_original"].iloc[0] if len(df) > 0 else ""
            if "inferred from publication" in str(sample_original):
                extraction_method = "study_context"
            elif "=" in str(sample_original):  # Boolean flags format
                extraction_method = "boolean_flags"
            else:
                extraction_method = "phenotype_field"
        else:
            extraction_method = "existing_column"

    print(f"  Extraction method: {extraction_method}")

    result = {
        "file": file_path.name,
        "samples": total_samples,
        "before_count": before_count,
        "before_pct": before_pct,
        "after_count": after_count,
        "after_pct": after_pct,
        "improvement": improvement,
        "extraction_successful": extraction_successful,
        "extraction_method": extraction_method,
        "disease_fields_before": disease_fields_before
    }

    # Sample validation (spot check 5 samples)
    if extraction_successful and after_count > 0:
        print(f"\n  Spot-checking extraction accuracy (5 samples):")
        samples_with_disease = df[df["disease"].notna()].head(5)

        for idx, (_, sample) in enumerate(samples_with_disease.iterrows(), 1):
            extracted_disease = sample.get("disease", "N/A")
            original_source = sample.get("disease_original", "N/A")

            # Find which raw field was used
            raw_fields = {}
            for field in disease_fields_before:
                if field in sample and pd.notna(sample[field]):
                    raw_fields[field] = sample[field]

            print(f"    Sample {idx}:")
            print(f"      Raw fields: {raw_fields}")
            print(f"      Extracted: {extracted_disease}")
            print(f"      Source: {original_source}")

    return result


def main():
    """Run disease extraction tests on REAL workspace metadata."""
    print("\n" + "="*80)
    print("ACTUAL DISEASE EXTRACTION TEST (REAL DATA)")
    print("="*80)

    workspace_path = Path("/Users/tyo/GITHUB/omics-os/lobster/.lobster_workspace/metadata")

    # Find all SRA sample files
    sra_files = sorted(workspace_path.glob("sra_*_samples.json"))

    print(f"\nFound {len(sra_files)} SRA sample files in workspace")
    print(f"Testing first 10 files with actual data...\n")

    # Test files
    results = []
    files_tested = 0

    for file_path in sra_files[:10]:  # Test first 10 files
        result = test_file(file_path)
        if result:
            results.append(result)
            files_tested += 1

    # Generate aggregate report
    print("\n" + "="*80)
    print("AGGREGATE RESULTS (MEASURED)")
    print("="*80)

    if not results:
        print("❌ No files successfully tested")
        return

    total_samples = sum(r["samples"] for r in results)
    total_before = sum(r["before_count"] for r in results)
    total_after = sum(r["after_count"] for r in results)

    before_pct_overall = (total_before / total_samples * 100) if total_samples > 0 else 0
    after_pct_overall = (total_after / total_samples * 100) if total_samples > 0 else 0
    improvement_overall = after_pct_overall - before_pct_overall

    print(f"\nFiles tested: {files_tested}")
    print(f"Total samples: {total_samples}")
    print(f"Disease coverage BEFORE: {total_before}/{total_samples} ({before_pct_overall:.1f}%)")
    print(f"Disease coverage AFTER: {total_after}/{total_samples} ({after_pct_overall:.1f}%)")
    print(f"ACTUAL IMPROVEMENT: +{improvement_overall:.1f}%")

    # Detailed results table
    print("\n" + "="*80)
    print("DETAILED RESULTS BY FILE")
    print("="*80)
    print(f"{'File':<40} {'Samples':<10} {'Before':<15} {'After':<15} {'Improvement':<12} {'Method':<20}")
    print("-"*110)

    for r in results:
        before_str = f"{r['before_count']}/{r['samples']} ({r['before_pct']:.1f}%)"
        after_str = f"{r['after_count']}/{r['samples']} ({r['after_pct']:.1f}%)"
        improvement_str = f"+{r['improvement']:.1f}%"

        print(f"{r['file']:<40} {r['samples']:<10} {before_str:<15} {after_str:<15} {improvement_str:<12} {r['extraction_method']:<20}")

    # Method breakdown
    print("\n" + "="*80)
    print("EXTRACTION METHOD BREAKDOWN")
    print("="*80)

    method_counts = {}
    for r in results:
        method = r["extraction_method"]
        method_counts[method] = method_counts.get(method, 0) + 1

    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count} files")

    # Success rate
    successful = sum(1 for r in results if r["extraction_successful"])
    success_rate = (successful / files_tested * 100) if files_tested > 0 else 0

    print(f"\nExtraction success rate: {successful}/{files_tested} files ({success_rate:.1f}%)")

    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if improvement_overall >= 50:
        effectiveness = "EXCELLENT"
    elif improvement_overall >= 20:
        effectiveness = "GOOD"
    elif improvement_overall > 0:
        effectiveness = "MODERATE"
    else:
        effectiveness = "POOR"

    print(f"\nDisease extraction system effectiveness: {effectiveness}")
    print(f"Based on {files_tested} real files, {total_samples} real samples")
    print(f"Actual measured improvement: +{improvement_overall:.1f}%")


if __name__ == "__main__":
    main()
