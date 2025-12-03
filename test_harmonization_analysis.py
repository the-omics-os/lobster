#!/usr/bin/env python3
"""
Harmonization Quality Analysis Script for Group B Entries
Tests harmonized metadata field completeness and quality
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Test entries (Group B - entries with sample metadata)
TEST_ENTRIES = [
    {
        'line': 138,
        'entry_id': 'pub_queue_doi_10_1158_2326-6066_cir-19-1014',
        'sample_keys': ['sra_PRJNA1268942_samples', 'sra_PRJNA1268786_samples', 'sra_PRJNA1269092_samples']
    },
    {
        'line': 142,
        'entry_id': 'pub_queue_doi_10_1016_j_immuni_2024_12_012',
        'sample_keys': ['sra_PRJNA1183750_samples', 'sra_PRJNA960626_samples', 'sra_PRJNA1183753_samples']
    },
    {
        'line': 204,
        'entry_id': 'pub_queue_doi_10_1038_s41591-019-0405-7',
        'sample_keys': ['sra_PRJEB27928_samples']
    },
    {
        'line': 210,
        'entry_id': 'pub_queue_doi_10_1038_s41586-020-2983-4',
        'sample_keys': ['sra_PRJNA665727_samples', 'sra_PRJNA665536_samples']
    },
    {
        'line': 435,
        'entry_id': 'pub_queue_doi_10_3389_fphys_2022_854545',
        'sample_keys': ['sra_PRJNA824020_samples']
    },
    {
        'line': 438,
        'entry_id': 'pub_queue_doi_10_1016_j_clcc_2023_10_004',
        'sample_keys': ['sra_PRJNA1277165_samples']
    },
    {
        'line': 449,
        'entry_id': 'pub_queue_doi_10_1016_j_bbi_2020_03_026',
        'sample_keys': ['sra_PRJNA591924_samples']
    },
    {
        'line': 455,
        'entry_id': 'pub_queue_doi_10_1371_journal_pone_0319750',
        'sample_keys': ['sra_PRJEB46474_samples']
    }
]

# Harmonized fields to check
HARMONIZED_FIELDS = ['disease', 'age', 'sex', 'sample_type', 'tissue']

def load_sample_metadata(workspace_path: Path, sample_key: str) -> List[Dict]:
    """Load sample metadata from workspace"""
    metadata_file = workspace_path / 'metadata' / f'{sample_key}.json'
    if not metadata_file.exists():
        print(f"  ⚠️  File not found: {metadata_file}")
        return []

    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
            # Handle workspace cache format (with 'data' wrapper)
            if isinstance(data, dict) and 'data' in data:
                inner_data = data['data']
                if isinstance(inner_data, dict) and 'samples' in inner_data:
                    return inner_data['samples']
                elif isinstance(inner_data, list):
                    return inner_data
            # Handle direct list format
            elif isinstance(data, list):
                return data
            # Handle dict with 'samples' key
            elif isinstance(data, dict) and 'samples' in data:
                return data['samples']
            else:
                print(f"  ⚠️  Unexpected format in {sample_key}")
                print(f"      Keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
                return []
    except Exception as e:
        print(f"  ❌ Error loading {sample_key}: {e}")
        return []

def analyze_harmonization(samples: List[Dict]) -> Dict[str, float]:
    """Analyze harmonization completeness for a list of samples"""
    if not samples:
        return {field: 0.0 for field in HARMONIZED_FIELDS}

    field_counts = {field: 0 for field in HARMONIZED_FIELDS}
    total_samples = len(samples)

    for sample in samples:
        for field in HARMONIZED_FIELDS:
            value = sample.get(field)
            # Count as present if not None, not empty string, and not "NA"
            if value is not None and value != '' and value != 'NA' and value != 'N/A':
                field_counts[field] += 1

    # Calculate percentages
    return {field: (count / total_samples * 100) if total_samples > 0 else 0.0
            for field, count in field_counts.items()}

def analyze_field_values(samples: List[Dict], field: str) -> Dict[str, int]:
    """Analyze unique values for a specific field"""
    value_counts = defaultdict(int)
    for sample in samples:
        value = sample.get(field)
        if value is not None and value != '' and value != 'NA' and value != 'N/A':
            value_counts[str(value)] = value_counts.get(str(value), 0) + 1
    return dict(value_counts)

def quality_score(completeness: Dict[str, float]) -> str:
    """Calculate overall quality score"""
    avg = sum(completeness.values()) / len(completeness) if completeness else 0
    if avg >= 80:
        return "EXCELLENT"
    elif avg >= 60:
        return "GOOD"
    elif avg >= 40:
        return "FAIR"
    else:
        return "POOR"

def main():
    workspace_path = Path('/Users/tyo/GITHUB/omics-os/lobster/results')

    print("="*80)
    print("HARMONIZATION COMPLETENESS ANALYSIS - GROUP B")
    print("="*80)

    results = []
    all_samples = []

    for entry in TEST_ENTRIES:
        print(f"\n{'='*80}")
        print(f"Entry Line {entry['line']}: {entry['entry_id']}")
        print(f"{'='*80}")

        entry_samples = []
        for sample_key in entry['sample_keys']:
            print(f"\n  Loading: {sample_key}")
            samples = load_sample_metadata(workspace_path, sample_key)
            print(f"    Samples loaded: {len(samples)}")
            entry_samples.extend(samples)

        if not entry_samples:
            print(f"  ❌ No samples found for this entry")
            results.append({
                'entry_id': entry['entry_id'],
                'line': entry['line'],
                'total_samples': 0,
                'completeness': {field: 0.0 for field in HARMONIZED_FIELDS},
                'quality': 'NO DATA'
            })
            continue

        all_samples.extend(entry_samples)

        # Analyze completeness
        completeness = analyze_harmonization(entry_samples)
        quality = quality_score(completeness)

        print(f"\n  Total samples: {len(entry_samples)}")
        print(f"  Harmonization completeness:")
        for field, pct in completeness.items():
            print(f"    {field:15s}: {pct:6.2f}%")
        print(f"  Quality Score: {quality}")

        # Analyze unique values for key fields
        print(f"\n  Unique values:")
        for field in ['disease', 'sample_type', 'library_strategy']:
            values = analyze_field_values(entry_samples, field)
            if values:
                print(f"    {field}: {list(values.keys())[:5]}")  # Show first 5

        results.append({
            'entry_id': entry['entry_id'],
            'line': entry['line'],
            'total_samples': len(entry_samples),
            'completeness': completeness,
            'quality': quality
        })

    # Summary Table
    print(f"\n\n{'='*80}")
    print("HARMONIZATION COMPLETENESS MATRIX")
    print(f"{'='*80}")

    header = f"{'Line':5s} | {'Samples':7s} | {'Disease':8s} | {'Age':8s} | {'Sex':8s} | {'Sample Type':12s} | {'Tissue':8s} | {'Quality':10s}"
    print(header)
    print("-" * len(header))

    for r in results:
        c = r['completeness']
        print(f"{r['line']:5d} | {r['total_samples']:7d} | {c['disease']:7.1f}% | {c['age']:7.1f}% | {c['sex']:7.1f}% | {c['sample_type']:11.1f}% | {c['tissue']:7.1f}% | {r['quality']:10s}")

    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}")

    if all_samples:
        overall_completeness = analyze_harmonization(all_samples)
        print(f"Total samples across all entries: {len(all_samples)}")
        print(f"\nAverage completeness:")
        for field, pct in overall_completeness.items():
            print(f"  {field:15s}: {pct:6.2f}%")
        print(f"\nOverall Quality: {quality_score(overall_completeness)}")

        # Field value distribution
        print(f"\n{'='*80}")
        print("FIELD VALUE DISTRIBUTION")
        print(f"{'='*80}")

        for field in ['disease', 'sample_type', 'sex']:
            print(f"\n{field.upper()}:")
            values = analyze_field_values(all_samples, field)
            for value, count in sorted(values.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {value:30s}: {count:4d} samples ({count/len(all_samples)*100:5.1f}%)")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
