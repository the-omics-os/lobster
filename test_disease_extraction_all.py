"""
Test disease extraction on ALL workspace metadata files.

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
    try:
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
        df_copy = df.copy()
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

        # Check for bug: boolean_flags with Y/N values
        bug_detected = False
        if strategy == "boolean_flags":
            flag_cols = [c for c in existing_fields if c.endswith("_disease")]
            for col in flag_cols:
                if "Y" in df[col].values or "N" in df[col].values:
                    bug_detected = True
                    break

        return {
            "file": file_path.name,
            "samples": total_samples,
            "before_count": before_count,
            "before_pct": before_pct,
            "after_count": after_count,
            "after_pct": after_pct,
            "improvement": improvement,
            "strategy": strategy,
            "bug_detected": bug_detected,
            "success": improvement > 0
        }
    except Exception as e:
        return {
            "file": file_path.name,
            "error": str(e)
        }


def main():
    """Test on ALL workspace metadata files."""
    workspace = Path(".lobster_workspace/metadata")

    # Find all SRA metadata files
    files = list(workspace.glob("sra_*_samples.json"))

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE DISEASE EXTRACTION TEST")
    print(f"{'='*80}\n")
    print(f"Testing ALL {len(files)} metadata files...\n")

    results = []
    for i, file_path in enumerate(files, 1):
        result = test_file(file_path)
        results.append(result)
        if i % 10 == 0:
            print(f"Progress: {i}/{len(files)}")

    print(f"\n{'='*80}")
    print("MEASURED RESULTS")
    print(f"{'='*80}\n")

    # Filter out errors
    valid_results = [r for r in results if "error" not in r]
    error_results = [r for r in results if "error" in r]

    # Aggregate statistics
    total_files = len(valid_results)
    total_samples = sum(r["samples"] for r in valid_results)
    total_before = sum(r["before_count"] for r in valid_results)
    total_after = sum(r["after_count"] for r in valid_results)

    before_pct = (total_before / total_samples * 100) if total_samples > 0 else 0
    after_pct = (total_after / total_samples * 100) if total_samples > 0 else 0
    improvement = after_pct - before_pct

    print(f"## Aggregate MEASURED Results")
    print(f"- Files tested: {total_files}")
    print(f"- Files with errors: {len(error_results)}")
    print(f"- Total samples: {total_samples}")
    print(f"- Disease BEFORE: {total_before}/{total_samples} ({before_pct:.1f}%)")
    print(f"- Disease AFTER: {total_after}/{total_samples} ({after_pct:.1f}%)")
    print(f"- **ACTUAL IMPROVEMENT: +{improvement:.1f}%**\n")

    # Strategy breakdown
    strategy_counts = {}
    for r in valid_results:
        strategy = r.get("strategy", "none")
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    print(f"## Extraction Methods OBSERVED")
    for strategy, count in strategy_counts.items():
        files_pct = (count / total_files * 100) if total_files > 0 else 0
        print(f"- {strategy}: {count} files ({files_pct:.1f}%)")

    # Bug detection
    bug_files = [r for r in valid_results if r.get("bug_detected", False)]
    if bug_files:
        print(f"\n## BUG DETECTED")
        print(f"- Files with Y/N boolean flags: {len(bug_files)}")
        print(f"- Current code DOES NOT handle single-letter Y/N values")
        print(f"- Missing from check: 'Y', 'y', 'N', 'n'")
        print(f"- Affected samples: {sum(r['samples'] for r in bug_files)}")

    # Success rate
    success_files = [r for r in valid_results if r.get("success", False)]
    success_rate = (len(success_files) / total_files * 100) if total_files > 0 else 0
    print(f"\n## Success Metrics")
    print(f"- Files with improvement: {len(success_files)}/{total_files} ({success_rate:.1f}%)")
    print(f"- Files with no extraction: {total_files - len(success_files)}")

    # Top performing files
    top_files = sorted(valid_results, key=lambda x: x.get("improvement", 0), reverse=True)[:5]
    print(f"\n## Top 5 Files by Improvement")
    for r in top_files:
        print(f"- {r['file']}: +{r['improvement']:.1f}% ({r['samples']} samples, strategy: {r['strategy']})")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
