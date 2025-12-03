"""
Test disease extraction on REAL workspace metadata files.

Reports ONLY measured results from actual data.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


def _extract_disease_from_raw_fields(
    metadata: pd.DataFrame,
    study_context: Optional[Dict] = None
) -> Optional[str]:
    """
    Extract disease information from diverse, study-specific SRA field names.

    COPY OF ACTUAL CODE FROM metadata_assistant.py (lines 722-861)
    """
    # Strategy 1: Check for existing unified disease column
    existing_disease_cols = ["disease", "disease_state", "condition", "diagnosis"]
    for col in existing_disease_cols:
        if col in metadata.columns and metadata[col].notna().sum() > 0:
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
                return "disease"

    # Strategy 3: Consolidate boolean disease flags
    disease_flag_cols = [c for c in metadata.columns if c.endswith("_disease")]

    if disease_flag_cols:
        def extract_from_flags(row):
            """Extract disease from boolean flags."""
            active_diseases = []

            for flag_col in disease_flag_cols:
                # Check if flag is TRUE
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
                return ";".join(active_diseases)

            # Check for negative controls
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

        return "disease"

    # Strategy 4: Use study context
    if study_context and "disease_focus" in study_context:
        metadata["disease"] = study_context["disease_focus"]
        metadata["disease_original"] = f"inferred from publication: {study_context['disease_focus']}"
        return "disease"

    return None


def test_file(file_path: Path) -> Dict[str, Any]:
    """Test disease extraction on a single metadata file."""
    with open(file_path) as f:
        data = json.load(f)

    # Extract samples from nested structure
    if "data" in data and "samples" in data["data"]:
        samples = data["data"]["samples"]
    elif "samples" in data:
        samples = data["samples"]
    else:
        return {
            "file": file_path.name,
            "error": "No samples found in file structure"
        }

    if not samples:
        return {
            "file": file_path.name,
            "samples": 0,
            "error": "Empty samples list"
        }

    # Create DataFrame
    df = pd.DataFrame(samples)
    total_samples = len(df)

    # BEFORE extraction
    before_count = df["disease"].notna().sum() if "disease" in df.columns else 0
    before_pct = (before_count / total_samples * 100) if total_samples > 0 else 0

    # Identify what fields exist BEFORE extraction
    existing_fields = list(df.columns)

    # Apply ACTUAL extraction
    df_copy = df.copy()  # Work on copy to preserve original
    disease_col = _extract_disease_from_raw_fields(df_copy, study_context=None)

    # AFTER extraction
    after_count = df_copy["disease"].notna().sum() if disease_col and "disease" in df_copy.columns else 0
    after_pct = (after_count / total_samples * 100) if total_samples > 0 else 0

    # Improvement
    improvement = after_pct - before_pct

    # Identify extraction strategy used
    strategy = "none"
    if disease_col:
        if "disease" in existing_fields or "disease_state" in existing_fields:
            strategy = "existing_column"
        elif any(col in existing_fields for col in ["host_phenotype", "phenotype", "host_disease", "health_status"]):
            strategy = "phenotype_fields"
        elif any(col.endswith("_disease") for col in existing_fields):
            strategy = "boolean_flags"

    # Sample extracted values (first 3)
    sample_values = []
    if disease_col and "disease" in df_copy.columns:
        for idx in df_copy.head(3).index:
            row = df_copy.loc[idx]
            original = row.get("disease_original", "N/A")
            extracted = row.get("disease", "N/A")
            sample_values.append({
                "original": str(original),
                "extracted": str(extracted)
            })

    return {
        "file": file_path.name,
        "samples": total_samples,
        "before_count": before_count,
        "before_pct": before_pct,
        "after_count": after_count,
        "after_pct": after_pct,
        "improvement": improvement,
        "strategy": strategy,
        "sample_values": sample_values,
        "fields_found": existing_fields[:10]  # First 10 fields
    }


def main():
    """Test on REAL workspace metadata files."""
    workspace = Path(".lobster_workspace/metadata")

    # Find all SRA metadata files
    files = list(workspace.glob("sra_*_samples.json"))

    print(f"\n{'='*80}")
    print(f"DISEASE EXTRACTION TEST - REAL DATA ONLY")
    print(f"{'='*80}\n")
    print(f"Total files found: {len(files)}\n")

    if len(files) == 0:
        print("❌ No metadata files found in workspace")
        return

    # Test first 20 files
    test_files = files[:20]
    print(f"Testing first {len(test_files)} files...\n")

    results = []
    for file_path in test_files:
        result = test_file(file_path)
        results.append(result)
        print(f"✓ Tested: {result['file']}")

    print(f"\n{'='*80}")
    print("MEASURED RESULTS")
    print(f"{'='*80}\n")

    # Aggregate statistics
    total_files = len(results)
    total_samples = sum(r["samples"] for r in results if "samples" in r)
    total_before = sum(r["before_count"] for r in results if "before_count" in r)
    total_after = sum(r["after_count"] for r in results if "after_count" in r)

    before_pct = (total_before / total_samples * 100) if total_samples > 0 else 0
    after_pct = (total_after / total_samples * 100) if total_samples > 0 else 0
    improvement = after_pct - before_pct

    print(f"## Aggregate MEASURED Results")
    print(f"- Files tested: {total_files}")
    print(f"- Total samples: {total_samples}")
    print(f"- Disease BEFORE: {total_before}/{total_samples} ({before_pct:.1f}%)")
    print(f"- Disease AFTER: {total_after}/{total_samples} ({after_pct:.1f}%)")
    print(f"- **ACTUAL IMPROVEMENT: +{improvement:.1f}%**\n")

    # Strategy breakdown
    strategy_counts = {}
    for r in results:
        strategy = r.get("strategy", "none")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    print(f"## Extraction Methods OBSERVED")
    for strategy, count in strategy_counts.items():
        print(f"- {strategy}: {count} files")

    print(f"\n{'='*80}")
    print("DETAILED FILE RESULTS")
    print(f"{'='*80}\n")

    # Show detailed results
    for r in results[:10]:  # First 10 files
        if "error" in r:
            print(f"❌ {r['file']}: {r['error']}")
        else:
            print(f"\n### {r['file']}")
            print(f"- Samples: {r['samples']}")
            print(f"- Before: {r['before_count']}/{r['samples']} ({r['before_pct']:.1f}%)")
            print(f"- After: {r['after_count']}/{r['samples']} ({r['after_pct']:.1f}%)")
            print(f"- Improvement: +{r['improvement']:.1f}%")
            print(f"- Strategy: {r['strategy']}")

            # Show sample extractions
            if r['sample_values']:
                print(f"- Sample extractions:")
                for i, sample in enumerate(r['sample_values'], 1):
                    print(f"  {i}. Original: {sample['original'][:50]}... → Extracted: {sample['extracted']}")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
