#!/usr/bin/env python3
"""
Examine HANDOFF_READY entries in detail to understand their state.
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

# Filter for HANDOFF_READY
handoff_ready = [e for e in all_entries if e.get("status") == "handoff_ready"]

print(f"Total HANDOFF_READY entries: {len(handoff_ready)}\n")
print("=" * 100)

# Examine first 5 in detail
for i, entry in enumerate(handoff_ready[:5], 1):
    line_num = entry.get("_line_number")
    entry_id = entry.get("entry_id", "N/A")
    title = entry.get("title", "N/A")[:70]

    print(f"\n{i}. [Line {line_num}] {entry_id}")
    print(f"   Title: {title}...")

    # Show all available fields
    print(f"\n   Available fields:")
    for key, value in entry.items():
        if key == "_line_number":
            continue
        if isinstance(value, (list, dict)):
            if value:
                print(f"   - {key}: {type(value).__name__} with {len(value)} items")
                if isinstance(value, list) and len(value) <= 3:
                    print(f"     Contents: {value}")
                elif isinstance(value, list):
                    print(f"     Sample: {value[:3]}")
            else:
                print(f"   - {key}: empty {type(value).__name__}")
        else:
            print(f"   - {key}: {value}")

    print("-" * 100)

# Summary of key fields across all HANDOFF_READY entries
print("\n\nSUMMARY ACROSS ALL HANDOFF_READY ENTRIES:")
print("=" * 100)

with_identifiers = [e for e in handoff_ready if len(e.get("extracted_identifiers", [])) > 0]
with_datasets = [e for e in handoff_ready if len(e.get("dataset_ids", [])) > 0]
with_workspace = [e for e in handoff_ready if len(e.get("workspace_metadata_keys", [])) > 0]

print(f"With extracted_identifiers: {len(with_identifiers)}")
print(f"With dataset_ids: {len(with_datasets)}")
print(f"With workspace_metadata_keys: {len(with_workspace)}")

# Check if entries have all three (the handoff criteria)
fully_ready = [
    e for e in handoff_ready
    if len(e.get("extracted_identifiers", [])) > 0
    and len(e.get("dataset_ids", [])) > 0
    and len(e.get("workspace_metadata_keys", [])) > 0
]

print(f"\nFully ready (all 3 criteria met): {len(fully_ready)}")

if fully_ready:
    print("\nFully ready entries for export testing:")
    for i, entry in enumerate(fully_ready[:10], 1):
        line_num = entry.get("_line_number")
        entry_id = entry.get("entry_id", "N/A")
        title = entry.get("title", "N/A")[:60]
        n_ids = len(entry.get("extracted_identifiers", []))
        n_datasets = len(entry.get("dataset_ids", []))
        n_workspace = len(entry.get("workspace_metadata_keys", []))

        print(f"\n{i}. [Line {line_num}] {entry_id}")
        print(f"   Title: {title}...")
        print(f"   IDs: {n_ids} | Datasets: {n_datasets} | Workspace files: {n_workspace}")
        print(f"   Workspace keys: {entry.get('workspace_metadata_keys', [])[:2]}")
