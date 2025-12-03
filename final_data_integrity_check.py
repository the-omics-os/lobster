#!/usr/bin/env python3
"""
Final data integrity check for Group A exports.
Spot-check sample rows to verify field values are intact.
"""
import sys
sys.path.insert(0, "/Users/tyo/GITHUB/omics-os/lobster")

import pandas as pd
import json
from pathlib import Path

EXPORT_DIR = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11/metadata/exports")
METADATA_DIR = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11/metadata")

# Test configuration
tests = [
    {
        "label": "SMALL",
        "csv": "pub_queue_doi_10_3389_fmicb_2023_1154508_2025-12-02.csv",
        "workspace_file": "sra_PRJNA937653_samples.json"
    }
]

print("=" * 100)
print("FINAL DATA INTEGRITY CHECK - SPOT-CHECK VALIDATION")
print("=" * 100)

for test_config in tests:
    label = test_config["label"]
    csv_file = test_config["csv"]
    workspace_file = test_config["workspace_file"]

    print(f"\n\n{label} Export:")
    print("-" * 100)

    # Load CSV
    csv_path = EXPORT_DIR / csv_file
    df = pd.read_csv(csv_path)

    # Load workspace metadata
    workspace_path = METADATA_DIR / workspace_file
    with open(workspace_path) as f:
        workspace_data = json.load(f)

    workspace_samples = workspace_data["data"]["samples"]

    print(f"  CSV samples: {len(df)}")
    print(f"  Workspace samples: {len(workspace_samples)}")

    # Spot-check 5 random samples
    import random
    random.seed(42)  # Reproducible
    sample_indexes = random.sample(range(len(workspace_samples)), min(5, len(workspace_samples)))

    print(f"\n  Spot-checking {len(sample_indexes)} samples:")

    fields_to_check = [
        "run_accession",
        "biosample",
        "organism_name",
        "library_strategy",
        "total_spots",
        "ena_fastq_http"
    ]

    all_match = True

    for i, idx in enumerate(sample_indexes, 1):
        workspace_sample = workspace_samples[idx]
        csv_sample = df[df["run_accession"] == workspace_sample["run_accession"]]

        if csv_sample.empty:
            print(f"\n  {i}. ✗ MISSING: run_accession {workspace_sample['run_accession']} not found in CSV")
            all_match = False
            continue

        print(f"\n  {i}. Sample {workspace_sample['run_accession']}:")

        for field in fields_to_check:
            workspace_val = workspace_sample.get(field)
            csv_val = csv_sample.iloc[0].get(field)

            # Handle type conversions
            if pd.isna(csv_val):
                csv_val = None
            elif isinstance(csv_val, (int, float)):
                csv_val = str(int(csv_val))
            else:
                csv_val = str(csv_val)

            if isinstance(workspace_val, (int, float)):
                workspace_val = str(int(workspace_val))
            elif workspace_val is not None:
                workspace_val = str(workspace_val)

            match = workspace_val == csv_val
            marker = "✓" if match else "✗"

            if not match:
                print(f"    {marker} {field}: workspace='{workspace_val}' vs csv='{csv_val}'")
                all_match = False
            else:
                # Only show first 60 chars for brevity
                display_val = str(workspace_val)[:60] if workspace_val else "None"
                print(f"    {marker} {field}: {display_val}")

    print(f"\n  Overall: {'✓ ALL FIELDS MATCH' if all_match else '✗ MISMATCHES DETECTED'}")

print("\n\n" + "=" * 100)
print("DATA INTEGRITY VALIDATION COMPLETE")
print("=" * 100)

if all_match:
    print("\n✓ NO DATA CORRUPTION DETECTED")
    print("✓ CSV exports faithfully represent workspace metadata")
else:
    print("\n✗ DATA INTEGRITY ISSUES FOUND")
