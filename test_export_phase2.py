#!/usr/bin/env python3
"""
Phase 2: Export Group A test entries to CSV and validate.
Uses the schema-driven export system from export_schemas.py
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

# Add lobster to path
sys.path.insert(0, "/Users/tyo/GITHUB/omics-os/lobster")

from lobster.core.schemas.export_schemas import get_ordered_export_columns, infer_data_type

# Test configuration
WORKSPACE_BASE = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11")
METADATA_DIR = WORKSPACE_BASE / "metadata"
EXPORT_DIR = METADATA_DIR / "exports"

# Selected test entries (SMALL, MEDIUM, LARGE)
TEST_ENTRIES = [
    {
        "label": "SMALL",
        "entry_id": "pub_queue_doi_10_3389_fmicb_2023_1154508",
        "dataset_id": "PRJNA937653",
        "sample_files": ["sra_PRJNA937653_samples"]
    },
    {
        "label": "MEDIUM",
        "entry_id": "pub_queue_doi_10_3390_nu13062032",
        "dataset_ids": ["PRJEB36385", "PRJEB32411"],
        "sample_files": ["sra_PRJEB36385_samples", "sra_PRJEB32411_samples"]
    },
    {
        "label": "LARGE",
        "entry_id": "pub_queue_doi_10_1038_s41586-022-05435-0",
        "dataset_id": "PRJNA811533",
        "sample_files": ["sra_PRJNA811533_samples"]
    },
]

def load_samples_from_workspace(sample_file_key):
    """
    Load samples from workspace metadata file.

    Returns:
        (samples, metadata_info) - List of sample dicts and file info
    """
    file_path = METADATA_DIR / f"{sample_file_key}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Workspace file not found: {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    # Extract samples (structure is data.samples)
    if isinstance(data, dict) and "data" in data:
        samples = data["data"].get("samples", [])
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError(f"Unexpected data structure in {file_path}")

    metadata_info = {
        "file": sample_file_key,
        "sample_count": len(samples),
        "content_type": data.get("content_type", "unknown"),
        "description": data.get("description", "N/A")
    }

    return samples, metadata_info

def add_publication_context(samples, entry_id, doi=None, pmid=None):
    """Add publication provenance fields to samples."""
    for sample in samples:
        sample["source_entry_id"] = entry_id
        if doi:
            sample["source_doi"] = doi
        if pmid:
            sample["source_pmid"] = pmid
    return samples

def export_to_csv(samples, output_path, data_type="sra_amplicon"):
    """
    Export samples to CSV using schema-driven column ordering.

    Args:
        samples: List of sample dicts
        output_path: Path to output CSV file
        data_type: Data type for schema lookup (default: sra_amplicon)

    Returns:
        (df, column_count, ordered_cols) - DataFrame, column count, ordered column list
    """
    # Get ordered columns using export schema
    ordered_cols = get_ordered_export_columns(samples, data_type=data_type)

    # Create DataFrame with ordered columns
    df = pd.DataFrame(samples)[ordered_cols]

    # Export to CSV
    df.to_csv(output_path, index=False)

    return df, len(ordered_cols), ordered_cols

def validate_export(df, ordered_cols, sample_file_metadata):
    """
    Validate exported CSV against schema requirements.

    Returns:
        validation_results dict
    """
    results = {
        "passed": True,
        "issues": [],
        "column_count": len(ordered_cols),
        "row_count": len(df),
        "expected_samples": sample_file_metadata["sample_count"]
    }

    # Check 1: Sample count matches
    if len(df) != sample_file_metadata["sample_count"]:
        results["passed"] = False
        results["issues"].append(
            f"Sample count mismatch: CSV has {len(df)}, workspace has {sample_file_metadata['sample_count']}"
        )

    # Check 2: Core identifiers are first 4 columns
    expected_first_4 = ["run_accession", "sample_accession", "biosample", "bioproject"]
    actual_first_4 = ordered_cols[:4]

    if actual_first_4 != expected_first_4:
        results["passed"] = False
        results["issues"].append(
            f"Column ordering incorrect. First 4 should be {expected_first_4}, got {actual_first_4}"
        )

    # Check 3: Harmonized fields present
    harmonized_fields = ["disease", "age", "sex", "sample_type", "tissue"]
    present_harmonized = [f for f in harmonized_fields if f in ordered_cols]

    if not present_harmonized:
        results["passed"] = False
        results["issues"].append("No harmonized fields found in export")

    results["harmonized_fields_present"] = present_harmonized

    # Check 4: Download URLs present
    download_url_fields = ["ena_fastq_http", "ncbi_url", "aws_url"]
    present_download_urls = [f for f in download_url_fields if f in ordered_cols]

    if not present_download_urls:
        results["passed"] = False
        results["issues"].append("No download URLs found in export")

    results["download_url_fields_present"] = present_download_urls

    # Check 5: Publication context present
    pub_context_fields = ["source_doi", "source_pmid", "source_entry_id"]
    present_pub_context = [f for f in pub_context_fields if f in ordered_cols]

    results["publication_context_fields_present"] = present_pub_context

    return results

# Create export directory
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("PHASE 2: GROUP A EXPORT TESTING")
print("=" * 100)
print()

all_results = []

for test_entry in TEST_ENTRIES:
    label = test_entry["label"]
    entry_id = test_entry["entry_id"]
    sample_files = test_entry.get("sample_files", [])

    print(f"\n{label}: {entry_id}")
    print("-" * 100)

    # Load samples from all sample files (some entries have multiple datasets)
    all_samples = []
    all_metadata_info = []

    for sample_file_key in sample_files:
        try:
            samples, metadata_info = load_samples_from_workspace(sample_file_key)
            print(f"  Loaded {len(samples)} samples from {sample_file_key}")
            all_samples.extend(samples)
            all_metadata_info.append(metadata_info)
        except Exception as e:
            print(f"  ERROR loading {sample_file_key}: {e}")
            continue

    if not all_samples:
        print(f"  SKIP: No samples loaded for {label}")
        continue

    # Add publication context
    all_samples = add_publication_context(all_samples, entry_id)

    # Infer data type
    data_type = infer_data_type(all_samples)
    print(f"  Inferred data type: {data_type}")

    # Export to CSV
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_filename = f"{entry_id}_{timestamp}.csv"
    output_path = EXPORT_DIR / output_filename

    try:
        df, col_count, ordered_cols = export_to_csv(all_samples, output_path, data_type=data_type)
        print(f"  Exported to: {output_path}")
        print(f"  Columns: {col_count}, Rows: {len(df)}")

        # Validate export
        combined_metadata = {
            "sample_count": sum(m["sample_count"] for m in all_metadata_info),
            "files": [m["file"] for m in all_metadata_info]
        }

        validation_results = validate_export(df, ordered_cols, combined_metadata)

        # Print validation summary
        print(f"\n  VALIDATION: {'PASS' if validation_results['passed'] else 'FAIL'}")
        if validation_results["issues"]:
            for issue in validation_results["issues"]:
                print(f"    - {issue}")

        print(f"  Harmonized fields: {', '.join(validation_results['harmonized_fields_present'])}")
        print(f"  Download URLs: {', '.join(validation_results['download_url_fields_present'])}")
        print(f"  Publication context: {', '.join(validation_results['publication_context_fields_present'])}")

        # Store results
        all_results.append({
            "label": label,
            "entry_id": entry_id,
            "output_file": str(output_path),
            "validation": validation_results
        })

    except Exception as e:
        print(f"  ERROR during export: {e}")
        import traceback
        traceback.print_exc()

# Final summary
print("\n\n" + "=" * 100)
print("GROUP A EXPORT SUMMARY")
print("=" * 100)

passed_count = sum(1 for r in all_results if r["validation"]["passed"])
total_count = len(all_results)

print(f"\nExports completed: {total_count}/{len(TEST_ENTRIES)}")
print(f"Validation passed: {passed_count}/{total_count}")

if passed_count == total_count:
    print("\n✓ ALL EXPORTS VALIDATED SUCCESSFULLY")
else:
    print(f"\n✗ {total_count - passed_count} EXPORT(S) FAILED VALIDATION")

print("\nExported files:")
for result in all_results:
    status = "✓ PASS" if result["validation"]["passed"] else "✗ FAIL"
    print(f"  {status} - {result['output_file']}")
