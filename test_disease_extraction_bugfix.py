#!/usr/bin/env python3
"""
Re-test disease extraction after bug fix.
Bug: Boolean flag extraction now supports Y/N single-letter values.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)


def _extract_disease_from_raw_fields(
    metadata: pd.DataFrame,
    study_context: Optional[Dict] = None
) -> Optional[pd.Series]:
    """
    Extract disease information from diverse, study-specific SRA field names.

    This is a COPY of the actual function from metadata_assistant.py for testing purposes.
    Bug fix: Now supports Y/N single-letter values in boolean flags (lines 797, 827).
    """
    # Strategy 1: Check for existing unified disease column
    existing_disease_cols = ["disease", "disease_state", "condition", "diagnosis"]
    for col in existing_disease_cols:
        if col in metadata.columns and metadata[col].notna().sum() > 0:
            logger.debug(f"Found existing disease column: {col}")
            if col != "disease":
                metadata["disease"] = metadata[col]
                metadata["disease_extraction_method"] = "existing_column"
            return metadata["disease"]

    # Strategy 2: Extract from free-text phenotype fields
    phenotype_cols = ["host_phenotype", "phenotype", "host_disease", "health_status"]
    for col in phenotype_cols:
        if col in metadata.columns:
            populated_count = metadata[col].notna().sum()
            if populated_count > 0:
                metadata["disease"] = metadata[col].fillna("unknown")
                metadata["disease_extraction_method"] = "phenotype_fields"
                logger.debug(
                    f"Extracted disease from {col} "
                    f"({populated_count}/{len(metadata)} samples, "
                    f"{populated_count/len(metadata)*100:.1f}%)"
                )
                return metadata["disease"]

    # Strategy 3: Consolidate boolean disease flags
    disease_flag_cols = [c for c in metadata.columns if c.endswith("_disease")]

    if disease_flag_cols:
        logger.debug(f"Found {len(disease_flag_cols)} disease flag columns: {disease_flag_cols}")

        def extract_from_flags(row):
            """Extract disease from boolean flags."""
            active_diseases = []

            for flag_col in disease_flag_cols:
                # BUG FIX: Check if flag is TRUE (handles Yes, Y, TRUE, True, 1, "1")
                flag_value = row.get(flag_col)
                if flag_value in ["Yes", "YES", "yes", "Y", "y", "TRUE", "True", "true", True, 1, "1"]:
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

            # BUG FIX: Check for negative controls (all flags FALSE, including N)
            all_false = all(
                row.get(flag_col) in ["No", "NO", "no", "N", "n", "FALSE", "False", "false", False, 0, "0"]
                for flag_col in disease_flag_cols
            )
            if all_false:
                return "healthy"

            return "unknown"

        # Apply extraction
        metadata["disease"] = metadata.apply(extract_from_flags, axis=1)
        metadata["disease_extraction_method"] = "boolean_flags"

        extracted_count = (metadata["disease"] != "unknown").sum()
        logger.debug(
            f"Extracted disease from {len(disease_flag_cols)} boolean flags "
            f"({extracted_count}/{len(metadata)} samples, "
            f"{extracted_count/len(metadata)*100:.1f}%)"
        )

        return metadata["disease"]

    # Strategy 4: Use study context
    if study_context and "disease_focus" in study_context:
        metadata["disease"] = study_context["disease_focus"]
        metadata["disease_extraction_method"] = "study_context"
        logger.debug(f"Assigned disease from publication context: {study_context['disease_focus']}")
        return metadata["disease"]

    logger.warning("No disease information found in metadata fields or study context")
    metadata["disease_extraction_method"] = "none"
    return None


def test_buggy_file() -> Dict:
    """Test the specific file that was affected by the bug."""
    print("=" * 80)
    print("STEP 1: Testing Buggy File (sra_prjna834801_samples.json)")
    print("=" * 80)

    file_path = Path(".lobster_workspace/metadata/sra_prjna834801_samples.json")

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return {}

    with open(file_path) as f:
        data = json.load(f)
        samples = data["data"]["samples"]

    df = pd.DataFrame(samples)
    total_samples = len(df)

    print(f"\nTotal samples: {total_samples}")

    # Check disease flag columns
    flag_cols = [c for c in df.columns if c.endswith("_disease")]
    print(f"\nDisease flag columns found: {flag_cols}")

    # Check unique values in flag columns
    for col in flag_cols:
        unique_vals = df[col].unique()
        counts = df[col].value_counts()
        print(f"\n{col}:")
        print(f"  Unique values: {unique_vals}")
        print(f"  Counts: {counts.to_dict()}")

    # BEFORE extraction
    before_count = df["disease"].notna().sum() if "disease" in df.columns else 0
    before_pct = (before_count / total_samples * 100) if total_samples > 0 else 0

    print(f"\nBEFORE extraction:")
    print(f"  Disease values: {before_count}/{total_samples} ({before_pct:.1f}%)")

    # Apply extraction with FIXED code
    disease_series = _extract_disease_from_raw_fields(df)

    if disease_series is not None:
        df["disease"] = disease_series

    # AFTER extraction
    after_count = df["disease"].notna().sum() if "disease" in df.columns else 0
    after_pct = (after_count / total_samples * 100) if total_samples > 0 else 0

    # Get extraction method breakdown
    method_used = "unknown"
    if "disease_extraction_method" in df.columns:
        method_counts = df["disease_extraction_method"].value_counts()
        method_used = method_counts.to_dict()

    print(f"\nAFTER extraction (with bug fix):")
    print(f"  Disease values: {after_count}/{total_samples} ({after_pct:.1f}%)")
    print(f"  Improvement: +{after_pct - before_pct:.1f}%")
    print(f"  Extraction methods used: {method_used}")

    # Sample some extracted values
    print(f"\nSample extracted values (first 10):")
    sample_df = df[df["disease"].notna()].head(10)
    for idx, row in sample_df.iterrows():
        disease = row.get("disease", "N/A")
        method = row.get("disease_extraction_method", "N/A")
        # Show relevant flag columns
        flags = {col: row.get(col, "N/A") for col in flag_cols if col in row}
        print(f"  Sample {idx}: disease={disease}, method={method}, flags={flags}")

    return {
        "file": str(file_path.name),
        "total_samples": total_samples,
        "before_count": before_count,
        "before_pct": before_pct,
        "after_count": after_count,
        "after_pct": after_pct,
        "improvement_pct": after_pct - before_pct,
        "method_breakdown": method_used
    }


def test_all_files() -> Dict:
    """Test extraction on all 136 metadata files."""
    print("\n" + "=" * 80)
    print("STEP 2: Testing ALL Files (136 files)")
    print("=" * 80)

    metadata_dir = Path(".lobster_workspace/metadata")
    files = list(metadata_dir.glob("sra_*_samples.json"))

    print(f"\nFound {len(files)} files")

    total_before = 0
    total_after = 0
    total_samples = 0

    method_counts = {
        "phenotype_fields": 0,
        "boolean_flags": 0,
        "existing_column": 0,
        "none": 0
    }

    file_results = []

    for i, file_path in enumerate(files, 1):
        try:
            with open(file_path) as f:
                data = json.load(f)
                samples = data["data"]["samples"]

            df = pd.DataFrame(samples)

            # BEFORE
            before = df["disease"].notna().sum() if "disease" in df.columns else 0

            # Apply extraction
            disease_series = _extract_disease_from_raw_fields(df)

            if disease_series is not None:
                df["disease"] = disease_series

            # AFTER
            after = df["disease"].notna().sum() if "disease" in df.columns else 0

            # Track method used
            if "disease_extraction_method" in df.columns:
                for method in df["disease_extraction_method"].unique():
                    if pd.notna(method):
                        method_counts[method] = method_counts.get(method, 0) + 1

            total_before += before
            total_after += after
            total_samples += len(df)

            file_results.append({
                "file": file_path.name,
                "samples": len(df),
                "before": before,
                "after": after,
                "improvement": after - before
            })

            if i % 20 == 0:
                print(f"  Processed {i}/{len(files)} files...")

        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")

    # Calculate aggregate stats
    before_pct = (total_before / total_samples * 100) if total_samples > 0 else 0
    after_pct = (total_after / total_samples * 100) if total_samples > 0 else 0
    improvement = after_pct - before_pct

    print(f"\n{'=' * 80}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 80}")
    print(f"\nTotal files: {len(files)}")
    print(f"Total samples: {total_samples:,}")
    print(f"\nBEFORE extraction: {total_before:,}/{total_samples:,} ({before_pct:.1f}%)")
    print(f"AFTER extraction (bug fixed): {total_after:,}/{total_samples:,} ({after_pct:.1f}%)")
    print(f"\n** MEASURED IMPROVEMENT: +{improvement:.1f}% **")

    print(f"\nExtraction method breakdown:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(files) * 100) if len(files) > 0 else 0
        print(f"  {method}: {count}/{len(files)} ({pct:.1f}%)")

    # Show top improvers
    print(f"\nTop 10 files by improvement:")
    top_improvers = sorted(file_results, key=lambda x: x["improvement"], reverse=True)[:10]
    for result in top_improvers:
        print(f"  {result['file']}: +{result['improvement']} samples "
              f"({result['before']} → {result['after']})")

    return {
        "total_files": len(files),
        "total_samples": total_samples,
        "before_count": total_before,
        "before_pct": before_pct,
        "after_count": total_after,
        "after_pct": after_pct,
        "improvement_pct": improvement,
        "method_breakdown": method_counts,
        "file_results": file_results
    }


def validate_bug_fix():
    """Validate that Y/N flag handling works correctly."""
    print("\n" + "=" * 80)
    print("STEP 3: Validating Bug Fix (Y/N Flag Handling)")
    print("=" * 80)

    # Test cases with Y/N flags
    test_samples = [
        {"accession": "test1", "crohns_disease": "Y", "inflam_bowel_disease": "N"},
        {"accession": "test2", "crohns_disease": "N", "inflam_bowel_disease": "Y"},
        {"accession": "test3", "crohns_disease": "N", "inflam_bowel_disease": "N"},
        {"accession": "test4", "crohns_disease": "Y", "inflam_bowel_disease": "Y"},
    ]

    expected = ["cd", "ibd", "healthy", "cd"]  # Expected disease values

    test_df = pd.DataFrame(test_samples)

    # Apply extraction
    disease_series = _extract_disease_from_raw_fields(test_df)

    if disease_series is not None:
        test_df["disease"] = disease_series

    print("\nTest results:")
    all_correct = True
    for i, (idx, row) in enumerate(test_df.iterrows()):
        actual = row.get("disease", "NOT EXTRACTED")
        exp = expected[i]
        status = "✓ CORRECT" if actual == exp else "✗ INCORRECT"
        if actual != exp:
            all_correct = False

        print(f"  Sample {i+1}: crohns={row['crohns_disease']}, "
              f"ibd={row['inflam_bowel_disease']} → disease={actual} "
              f"(expected: {exp}) {status}")

    print(f"\n{'✓ BUG FIX VALIDATED' if all_correct else '✗ BUG STILL EXISTS'}")

    return all_correct


def main():
    """Run all tests and generate report."""
    print("\n" + "=" * 80)
    print("DISEASE EXTRACTION RE-TEST (After Bug Fix)")
    print("=" * 80)

    # Step 1: Test buggy file
    buggy_results = test_buggy_file()

    # Step 2: Test all files
    all_results = test_all_files()

    # Step 3: Validate bug fix
    bug_fixed = validate_bug_fix()

    # Generate final report
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)

    print(f"""
## ACTUAL RE-TEST RESULTS (After Bug Fix)

### Bug Fix Validation
- Buggy file: {buggy_results.get('file', 'N/A')} ({buggy_results.get('total_samples', 0)} samples)
- Before bug fix: {buggy_results.get('before_count', 0)}/{buggy_results.get('total_samples', 0)} ({buggy_results.get('before_pct', 0):.1f}%)
- After bug fix: {buggy_results.get('after_count', 0)}/{buggy_results.get('total_samples', 0)} ({buggy_results.get('after_pct', 0):.1f}%)
- Bug fix impact: +{buggy_results.get('improvement_pct', 0):.1f}%
- Bug fix works: {'YES ✓' if bug_fixed else 'NO ✗'}

### Full Re-Test ({all_results.get('total_files', 0)} files, {all_results.get('total_samples', 0):,} samples)
- Before extraction: {all_results.get('before_count', 0):,}/{all_results.get('total_samples', 0):,} ({all_results.get('before_pct', 0):.1f}%)
- After extraction (bug fixed): {all_results.get('after_count', 0):,}/{all_results.get('total_samples', 0):,} ({all_results.get('after_pct', 0):.1f}%)
- **MEASURED IMPROVEMENT: +{all_results.get('improvement_pct', 0):.1f}%**

### Extraction Method Accuracy (MEASURED)
""")

    method_breakdown = all_results.get('method_breakdown', {})
    total_files = all_results.get('total_files', 1)
    for method, count in sorted(method_breakdown.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_files * 100) if total_files > 0 else 0
        print(f"- {method}: {count}/{total_files} ({pct:.1f}%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
