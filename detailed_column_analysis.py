#!/usr/bin/env python3
"""
Detailed column analysis for Group A exports.
Validates that columns follow schema priority ordering.
"""
import sys
sys.path.insert(0, "/Users/tyo/GITHUB/omics-os/lobster")

from lobster.core.schemas.export_schemas import (
    ExportSchemaRegistry,
    ExportPriority,
    get_ordered_export_columns
)
import pandas as pd
from pathlib import Path

EXPORT_DIR = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11/metadata/exports")

exports = [
    ("SMALL", "pub_queue_doi_10_3389_fmicb_2023_1154508_2025-12-02.csv", "sra_amplicon"),
    ("MEDIUM", "pub_queue_doi_10_3390_nu13062032_2025-12-02.csv", "sra_amplicon"),
    ("LARGE", "pub_queue_doi_10_1038_s41586-022-05435-0_2025-12-02.csv", "transcriptomics"),
]

print("=" * 100)
print("DETAILED COLUMN ANALYSIS")
print("=" * 100)

for label, filename, data_type in exports:
    print(f"\n\n{label} Export: {filename}")
    print("-" * 100)

    # Load CSV
    csv_path = EXPORT_DIR / filename
    df = pd.read_csv(csv_path, nrows=1)  # Just header
    actual_columns = list(df.columns)

    # Get schema for this data type
    schema = ExportSchemaRegistry.get_export_schema(data_type)
    if not schema:
        print(f"  ERROR: No schema found for {data_type}")
        continue

    print(f"  Data Type: {data_type}")
    print(f"  Total Columns: {len(actual_columns)}")
    print()

    # Analyze column groups
    priority_groups = schema["priority_groups"]

    column_index = 0
    for priority in sorted(priority_groups.keys(), key=lambda p: p.value):
        group_name = priority.name
        expected_fields = priority_groups[priority]

        print(f"\n  {priority.value}. {group_name}:")

        present_fields = []
        for field in expected_fields:
            if field in actual_columns:
                pos = actual_columns.index(field)
                present_fields.append((field, pos))

        if present_fields:
            for field, pos in present_fields:
                marker = "✓" if pos == column_index else f"⚠ pos {pos}"
                print(f"    {marker} {field}")
                if pos == column_index:
                    column_index += 1
        else:
            print(f"    (none present in data)")

    # Show first 10 actual columns
    print(f"\n  First 10 actual columns:")
    for i, col in enumerate(actual_columns[:10], 1):
        print(f"    {i}. {col}")

    # Show extra fields count
    all_schema_fields = []
    for fields in priority_groups.values():
        all_schema_fields.extend(fields)

    extra_fields = [c for c in actual_columns if c not in all_schema_fields]
    print(f"\n  Extra fields (not in schema): {len(extra_fields)}")
    if len(extra_fields) <= 10:
        print(f"    {', '.join(extra_fields)}")
    else:
        print(f"    Sample: {', '.join(extra_fields[:10])}...")

print("\n\n" + "=" * 100)
print("COLUMN ORDERING VALIDATION COMPLETE")
print("=" * 100)
