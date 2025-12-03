"""
Spot-check accuracy of disease extraction by inspecting actual field values.
"""
import json
import pandas as pd
from pathlib import Path


def spot_check_file(file_path: Path, num_samples: int = 5):
    """Spot-check accuracy for a specific file."""
    with open(file_path) as f:
        data = json.load(f)

    samples = data["data"]["samples"] if "data" in data else data.get("samples", [])
    df = pd.DataFrame(samples)

    print(f"\n{'='*80}")
    print(f"SPOT-CHECK: {file_path.name}")
    print(f"{'='*80}\n")
    print(f"Total samples: {len(df)}")

    # Show which phenotype/disease fields exist
    phenotype_cols = ["host_phenotype", "phenotype", "host_disease", "health_status", "disease", "disease_state"]
    existing_phenotype = [c for c in phenotype_cols if c in df.columns]
    disease_flag_cols = [c for c in df.columns if c.endswith("_disease")]

    print(f"\nExisting phenotype/disease fields: {existing_phenotype}")
    print(f"Boolean disease flags: {disease_flag_cols}")

    # Show first N samples with their raw values
    print(f"\nFirst {num_samples} samples:\n")
    for idx in df.head(num_samples).index:
        row = df.loc[idx]
        print(f"  Sample {idx} (run: {row.get('run_accession', 'N/A')}):")

        # Show phenotype fields
        for col in existing_phenotype:
            val = row.get(col)
            if pd.notna(val):
                print(f"    {col}: {val}")

        # Show boolean flags
        for col in disease_flag_cols:
            val = row.get(col)
            if pd.notna(val):
                print(f"    {col}: {val}")

        print()


def main():
    """Spot-check top performing files."""
    workspace = Path(".lobster_workspace/metadata")

    # Files with 100% improvement
    top_files = [
        "sra_prjna784939_samples.json",  # phenotype_fields
        "sra_prjna766641_samples.json",  # phenotype_fields
        "sra_prjna591924_samples.json",  # phenotype_fields
        "sra_prjna834801_samples.json",  # boolean_flags (BUG)
        "sra_prjna1040974_samples.json", # existing_column
    ]

    for filename in top_files:
        file_path = workspace / filename
        if file_path.exists():
            spot_check_file(file_path, num_samples=5)


if __name__ == "__main__":
    main()
