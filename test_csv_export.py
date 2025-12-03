#!/usr/bin/env python3
"""
CSV Export Testing Script for Harmonized Metadata
Tests the export schema and column structure
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any

# Schema definition based on sra_amplicon schema
# From lobster/core/schemas/transcriptomics_schema.py
HARMONIZED_PRIORITY_FIELDS = [
    'disease', 'disease_original', 'sample_type', 'age', 'sex', 'tissue'
]

CORE_SRA_FIELDS = [
    'run_accession', 'biosample', 'bioproject', 'study_title',
    'library_strategy', 'library_source', 'organism_name',
    'instrument_model', 'total_spots', 'total_size'
]

def load_sample_metadata(workspace_path: Path, sample_key: str) -> List[Dict]:
    """Load sample metadata from workspace"""
    metadata_file = workspace_path / 'metadata' / f'{sample_key}.json'

    with open(metadata_file, 'r') as f:
        data = json.load(f)
        # Handle workspace cache format
        if isinstance(data, dict) and 'data' in data:
            inner_data = data['data']
            if isinstance(inner_data, dict) and 'samples' in inner_data:
                return inner_data['samples']
    return []

def detect_library_strategy(samples: List[Dict]) -> str:
    """Detect most common library_strategy"""
    strategies = {}
    for sample in samples:
        strategy = sample.get('library_strategy', 'UNKNOWN')
        strategies[strategy] = strategies.get(strategy, 0) + 1

    if not strategies:
        return 'UNKNOWN'

    # Return most common
    return sorted(strategies.items(), key=lambda x: x[1], reverse=True)[0][0]

def infer_schema(library_strategy: str) -> str:
    """Infer schema from library_strategy"""
    if library_strategy == 'AMPLICON':
        return 'sra_amplicon'
    elif library_strategy in ['RNA-Seq', 'RNA-seq']:
        return 'transcriptomics'
    elif library_strategy == 'WGS':
        return 'sra_wgs'
    else:
        return 'generic'

def get_schema_columns(schema: str, samples: List[Dict]) -> List[str]:
    """Get column list based on detected schema"""

    # Always start with harmonized priority fields
    columns = HARMONIZED_PRIORITY_FIELDS.copy()

    # Add core SRA fields
    columns.extend(CORE_SRA_FIELDS)

    # Add additional fields from samples (dynamic discovery)
    all_keys = set()
    for sample in samples[:50]:  # Sample first 50
        all_keys.update(sample.keys())

    # Add remaining keys that aren't already in columns
    for key in sorted(all_keys):
        if key not in columns:
            columns.append(key)

    return columns

def export_to_csv(samples: List[Dict], output_path: Path, schema: str):
    """Export samples to CSV with schema-based column ordering"""
    if not samples:
        print(f"  ⚠️  No samples to export")
        return

    columns = get_schema_columns(schema, samples)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for sample in samples:
            # Convert None and empty strings to ''
            row = {k: (v if v not in [None, '', 'NA', 'N/A'] else '')
                   for k, v in sample.items()}
            writer.writerow(row)

    print(f"  ✅ Exported {len(samples)} samples to {output_path.name}")
    print(f"     Schema: {schema}, Columns: {len(columns)}")

    # Verify harmonized fields in first 7-12 column range
    harmonized_positions = {col: idx for idx, col in enumerate(columns)
                           if col in HARMONIZED_PRIORITY_FIELDS}
    print(f"     Harmonized field positions: {harmonized_positions}")

def main():
    workspace_path = Path('/Users/tyo/GITHUB/omics-os/lobster/results')
    output_dir = workspace_path / 'exports' / 'test_group_b'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test entries
    test_entries = [
        ('pub_queue_doi_10_1158_2326-6066_cir-19-1014', ['sra_PRJNA1268942_samples', 'sra_PRJNA1268786_samples', 'sra_PRJNA1269092_samples']),
        ('pub_queue_doi_10_1016_j_immuni_2024_12_012', ['sra_PRJNA1183750_samples', 'sra_PRJNA960626_samples', 'sra_PRJNA1183753_samples']),
        ('pub_queue_doi_10_1038_s41591-019-0405-7', ['sra_PRJEB27928_samples']),
        ('pub_queue_doi_10_1038_s41586-020-2983-4', ['sra_PRJNA665727_samples', 'sra_PRJNA665536_samples']),
        ('pub_queue_doi_10_3389_fphys_2022_854545', ['sra_PRJNA824020_samples']),
        ('pub_queue_doi_10_1016_j_clcc_2023_10_004', ['sra_PRJNA1277165_samples']),
        ('pub_queue_doi_10_1016_j_bbi_2020_03_026', ['sra_PRJNA591924_samples']),
        ('pub_queue_doi_10_1371_journal_pone_0319750', ['sra_PRJEB46474_samples']),
    ]

    print("="*80)
    print("CSV EXPORT TESTING - GROUP B")
    print("="*80)

    for entry_id, sample_keys in test_entries:
        print(f"\n{'='*80}")
        print(f"Entry: {entry_id}")
        print(f"{'='*80}")

        # Load all samples
        all_samples = []
        for sample_key in sample_keys:
            samples = load_sample_metadata(workspace_path, sample_key)
            print(f"  Loaded: {sample_key} ({len(samples)} samples)")
            all_samples.extend(samples)

        if not all_samples:
            print(f"  ❌ No samples found, skipping export")
            continue

        # Detect library strategy and infer schema
        library_strategy = detect_library_strategy(all_samples)
        schema = infer_schema(library_strategy)

        print(f"\n  Library Strategy: {library_strategy}")
        print(f"  Inferred Schema: {schema}")

        # Export to CSV
        output_file = output_dir / f"{entry_id}.csv"
        export_to_csv(all_samples, output_file, schema)

    print(f"\n{'='*80}")
    print(f"EXPORT COMPLETE - Files saved to: {output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
