#!/usr/bin/env python3
"""
Select 3 representative HANDOFF_READY entries for export testing (Group A).
Criteria: diverse dataset sizes (small, medium, large).
"""
import json
from pathlib import Path

queue_path = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11/.lobster/queues/publication_queue.jsonl")

# Load all entries
all_entries = []
with open(queue_path) as f:
    for idx, line in enumerate(f, 1):
        if line.strip():
            entry = json.loads(line)
            entry["_line_number"] = idx
            all_entries.append(entry)

# Filter for HANDOFF_READY with all criteria
ready_entries = [
    e for e in all_entries
    if e.get("status") == "handoff_ready"
       and len(e.get("extracted_identifiers", {})) > 0
       and len(e.get("dataset_ids", [])) > 0
       and len(e.get("workspace_metadata_keys", [])) > 0
]

# Sort by number of datasets
ready_entries.sort(key=lambda x: len(x.get("dataset_ids", [])))

print(f"Total HANDOFF_READY entries: {len(ready_entries)}")
print("\nDataset size distribution:")
print(f"  Smallest: {len(ready_entries[0]['dataset_ids'])} datasets")
print(f"  Median: {len(ready_entries[len(ready_entries)//2]['dataset_ids'])} datasets")
print(f"  Largest: {len(ready_entries[-1]['dataset_ids'])} datasets")

# Select small, medium, large
small = ready_entries[0]
medium = ready_entries[len(ready_entries) // 2]
large = ready_entries[-1]

# Also pick an alternative medium if available
if len(ready_entries) > 5:
    medium_alt = ready_entries[len(ready_entries) // 3]
else:
    medium_alt = medium

print("\n" + "=" * 100)
print("SELECTED TEST ENTRIES (PHASE 2 EXPORT):")
print("=" * 100)

for label, entry in [("SMALL", small), ("MEDIUM", medium), ("LARGE", large)]:
    line_num = entry.get("_line_number")
    entry_id = entry.get("entry_id", "N/A")
    title = entry.get("title", "N/A")[:70]
    dataset_ids = entry.get("dataset_ids", [])
    workspace_keys = entry.get("workspace_metadata_keys", [])

    print(f"\n{label}:")
    print(f"  Line: {line_num}")
    print(f"  Entry ID: {entry_id}")
    print(f"  Title: {title}...")
    print(f"  Datasets: {len(dataset_ids)} ({dataset_ids[:3]})")
    print(f"  Workspace keys ({len(workspace_keys)} files):")
    for key in workspace_keys:
        print(f"    - {key}")

# Now check if the workspace metadata files exist
print("\n" + "=" * 100)
print("WORKSPACE FILE VALIDATION:")
print("=" * 100)

workspace_base = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11/metadata")

for label, entry in [("SMALL", small), ("MEDIUM", medium), ("LARGE", large)]:
    entry_id = entry.get("entry_id", "N/A")
    workspace_keys = entry.get("workspace_metadata_keys", [])

    print(f"\n{label} ({entry_id}):")

    # Find sample metadata files (those with "_samples" suffix)
    sample_files = [k for k in workspace_keys if k.endswith("_samples")]

    if sample_files:
        for sample_key in sample_files:
            file_path = workspace_base / f"{sample_key}.json"
            if file_path.exists():
                # Count samples in file
                with open(file_path) as f:
                    data = json.load(f)
                    samples = data if isinstance(data, list) else []
                    print(f"  ✓ {sample_key}.json - {len(samples)} samples")
            else:
                print(f"  ✗ {sample_key}.json - NOT FOUND")
    else:
        print(f"  ⚠ No sample metadata files found in workspace_keys")

print("\n" + "=" * 100)
print("READY TO PROCEED WITH PHASE 2 EXPORTS")
print("=" * 100)
