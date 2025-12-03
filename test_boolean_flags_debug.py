"""Debug boolean flags extraction for PRJNA834801"""
import json
import pandas as pd
from pathlib import Path

file_path = Path(".lobster_workspace/metadata/sra_prjna834801_samples.json")

with open(file_path) as f:
    data = json.load(f)

samples = data["data"]["samples"]
df = pd.DataFrame(samples)

print(f"Total samples: {len(df)}")
print(f"\nColumns in dataset:")
for col in df.columns:
    print(f"  - {col}")

# Find disease flag columns
disease_flag_cols = [c for c in df.columns if c.endswith("_disease")]
print(f"\nDisease flag columns found: {disease_flag_cols}")

if disease_flag_cols:
    print(f"\nFirst 5 samples:")
    for idx in df.head(5).index:
        row = df.loc[idx]
        print(f"\n  Sample {idx}:")
        for col in disease_flag_cols:
            val = row.get(col)
            print(f"    {col} = {val} (type: {type(val).__name__})")

    # Check value distribution
    print(f"\nValue distribution for each flag:")
    for col in disease_flag_cols:
        counts = df[col].value_counts(dropna=False)
        print(f"\n  {col}:")
        for val, count in counts.items():
            print(f"    {val}: {count}")
