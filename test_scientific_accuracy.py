#!/usr/bin/env python3
"""
Scientific Accuracy Validation for Harmonized Metadata
Validates heuristic extraction quality
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any

def load_samples_from_json(workspace_path: Path, sample_key: str) -> List[Dict]:
    """Load samples from JSON metadata"""
    metadata_file = workspace_path / 'metadata' / f'{sample_key}.json'

    with open(metadata_file, 'r') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'data' in data:
            inner_data = data['data']
            if isinstance(inner_data, dict) and 'samples' in inner_data:
                return inner_data['samples']
    return []

def validate_age_extraction(samples: List[Dict]) -> Dict[str, Any]:
    """Validate age field extraction quality"""
    total = len(samples)
    with_age = [s for s in samples if s.get('age') not in [None, '', 'NA', 'N/A']]

    errors = []
    age_values = []

    for sample in with_age:
        age = sample.get('age')

        # Check if numeric
        try:
            age_num = float(age)
            age_values.append(age_num)

            # Check reasonable range
            if age_num < 0 or age_num > 120:
                errors.append({
                    'sample': sample.get('run_accession', 'unknown'),
                    'age': age,
                    'error': f'Age out of range: {age_num}'
                })
        except (ValueError, TypeError):
            errors.append({
                'sample': sample.get('run_accession', 'unknown'),
                'age': age,
                'error': f'Non-numeric age: {age}'
            })

    return {
        'total_samples': total,
        'samples_with_age': len(with_age),
        'completeness': len(with_age) / total * 100 if total > 0 else 0,
        'numeric_valid': len(age_values),
        'errors': errors,
        'error_rate': len(errors) / len(with_age) * 100 if with_age else 0,
        'age_stats': {
            'min': min(age_values) if age_values else None,
            'max': max(age_values) if age_values else None,
            'mean': sum(age_values) / len(age_values) if age_values else None
        }
    }

def validate_sex_extraction(samples: List[Dict]) -> Dict[str, Any]:
    """Validate sex field extraction quality"""
    total = len(samples)
    with_sex = [s for s in samples if s.get('sex') not in [None, '', 'NA', 'N/A']]

    valid_values = {'male', 'female', 'unknown', 'not collected', 'not applicable'}
    errors = []
    sex_distribution = {}

    for sample in with_sex:
        sex = sample.get('sex')
        sex_lower = str(sex).lower().strip()

        sex_distribution[sex_lower] = sex_distribution.get(sex_lower, 0) + 1

        if sex_lower not in valid_values:
            errors.append({
                'sample': sample.get('run_accession', 'unknown'),
                'sex': sex,
                'error': f'Invalid sex value: {sex}'
            })

    return {
        'total_samples': total,
        'samples_with_sex': len(with_sex),
        'completeness': len(with_sex) / total * 100 if total > 0 else 0,
        'valid_count': len(with_sex) - len(errors),
        'errors': errors,
        'error_rate': len(errors) / len(with_sex) * 100 if with_sex else 0,
        'distribution': sex_distribution
    }

def validate_tissue_extraction(samples: List[Dict]) -> Dict[str, Any]:
    """Validate tissue field extraction quality"""
    total = len(samples)
    with_tissue = [s for s in samples if s.get('tissue') not in [None, '', 'NA', 'N/A']]

    tissue_distribution = {}

    for sample in with_tissue:
        tissue = sample.get('tissue')
        tissue_distribution[tissue] = tissue_distribution.get(tissue, 0) + 1

    return {
        'total_samples': total,
        'samples_with_tissue': len(with_tissue),
        'completeness': len(with_tissue) / total * 100 if total > 0 else 0,
        'distribution': tissue_distribution
    }

def validate_sample_type_inference(samples: List[Dict]) -> Dict[str, Any]:
    """Check sample_type inference from isolation_source"""
    total = len(samples)
    with_isolation = [s for s in samples if s.get('isolation_source') or s.get('isolate')]

    # Expected sample_type based on isolation_source keywords
    expected_mappings = []

    for sample in with_isolation[:10]:  # Check first 10
        iso_source = sample.get('isolation_source') or sample.get('isolate') or ''
        sample_type = sample.get('sample_type')

        iso_lower = iso_source.lower()

        # Infer expected
        expected = None
        if any(kw in iso_lower for kw in ['fecal', 'stool', 'feces']):
            expected = 'fecal'
        elif any(kw in iso_lower for kw in ['tissue', 'biopsy']):
            expected = 'tissue'
        elif any(kw in iso_lower for kw in ['blood', 'serum', 'plasma']):
            expected = 'blood'

        if expected:
            expected_mappings.append({
                'sample': sample.get('run_accession', 'unknown'),
                'isolation_source': iso_source,
                'expected_type': expected,
                'actual_type': sample_type or 'NOT SET',
                'match': sample_type == expected
            })

    return {
        'total_samples': total,
        'samples_with_isolation': len(with_isolation),
        'sample_mappings': expected_mappings
    }

def main():
    workspace_path = Path('/Users/tyo/GITHUB/omics-os/lobster/results')

    # Test on entry 435 (PRJNA824020) - the good quality entry
    sample_key = 'sra_PRJNA824020_samples'

    print("="*80)
    print("SCIENTIFIC ACCURACY VALIDATION")
    print("="*80)
    print(f"\nEntry: {sample_key}")
    print("="*80)

    samples = load_samples_from_json(workspace_path, sample_key)
    print(f"\nTotal samples: {len(samples)}")

    # Age validation
    print(f"\n{'='*80}")
    print("AGE EXTRACTION VALIDATION")
    print(f"{'='*80}")
    age_results = validate_age_extraction(samples)

    print(f"Completeness: {age_results['completeness']:.1f}%")
    print(f"Numeric valid: {age_results['numeric_valid']}/{age_results['samples_with_age']}")
    print(f"Error rate: {age_results['error_rate']:.1f}%")

    if age_results['age_stats']['min']:
        print(f"\nAge statistics:")
        print(f"  Min: {age_results['age_stats']['min']:.0f}")
        print(f"  Max: {age_results['age_stats']['max']:.0f}")
        print(f"  Mean: {age_results['age_stats']['mean']:.1f}")

    if age_results['errors']:
        print(f"\nErrors found:")
        for error in age_results['errors'][:5]:
            print(f"  - {error['sample']}: {error['error']}")
    else:
        print(f"\n✅ No age extraction errors")

    # Sex validation
    print(f"\n{'='*80}")
    print("SEX EXTRACTION VALIDATION")
    print(f"{'='*80}")
    sex_results = validate_sex_extraction(samples)

    print(f"Completeness: {sex_results['completeness']:.1f}%")
    print(f"Valid values: {sex_results['valid_count']}/{sex_results['samples_with_sex']}")
    print(f"Error rate: {sex_results['error_rate']:.1f}%")

    print(f"\nSex distribution:")
    for sex, count in sorted(sex_results['distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {sex:20s}: {count:4d} samples ({count/sex_results['samples_with_sex']*100:5.1f}%)")

    if sex_results['errors']:
        print(f"\nErrors found:")
        for error in sex_results['errors'][:5]:
            print(f"  - {error['sample']}: {error['error']}")
    else:
        print(f"\n✅ No sex extraction errors")

    # Tissue validation
    print(f"\n{'='*80}")
    print("TISSUE EXTRACTION VALIDATION")
    print(f"{'='*80}")
    tissue_results = validate_tissue_extraction(samples)

    print(f"Completeness: {tissue_results['completeness']:.1f}%")

    print(f"\nTissue distribution:")
    for tissue, count in sorted(tissue_results['distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {tissue:20s}: {count:4d} samples ({count/tissue_results['samples_with_tissue']*100:5.1f}%)")

    # Sample type inference
    print(f"\n{'='*80}")
    print("SAMPLE TYPE INFERENCE VALIDATION")
    print(f"{'='*80}")
    type_results = validate_sample_type_inference(samples)

    print(f"Samples with isolation_source: {type_results['samples_with_isolation']}/{type_results['total_samples']}")
    print(f"\nSample inference check (first 10 with isolation_source):")

    for mapping in type_results['sample_mappings']:
        match_str = "✅" if mapping['match'] else "❌"
        print(f"\n{match_str} {mapping['sample']}")
        print(f"    Isolation: {mapping['isolation_source'][:60]}")
        print(f"    Expected: {mapping['expected_type']}")
        print(f"    Actual: {mapping['actual_type']}")

    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
