#!/usr/bin/env python3
"""
Script to inspect publication queue entries by line index (Group A test mission).

Test indexes (line numbers): 419, 415, 434, 433, 437, 448, 454, 452, 455
Note: These are 1-indexed (line numbers), but Python lists are 0-indexed.
"""
import json
from pathlib import Path

# Test indexes for Group A (1-indexed as per test mission)
TARGET_INDEXES = [419, 415, 434, 433, 437, 448, 454, 452, 455]

queue_path = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11/.lobster/queues/publication_queue.jsonl")

# Load all entries
all_entries = []
with open(queue_path) as f:
    for line in f:
        if line.strip():
            all_entries.append(json.loads(line))

print(f"Total entries in queue: {len(all_entries)}")
print(f"Requested indexes: {TARGET_INDEXES}")
print("=" * 80)

# Get entries by index (convert 1-indexed to 0-indexed)
selected_entries = []
for idx in TARGET_INDEXES:
    if 0 < idx <= len(all_entries):
        entry = all_entries[idx - 1]  # Convert to 0-indexed
        entry["_line_number"] = idx  # Add for tracking
        selected_entries.append(entry)
    else:
        print(f"WARNING: Index {idx} out of range (queue has {len(all_entries)} entries)")

# Sort by line number
selected_entries.sort(key=lambda x: x.get("_line_number", 0))

print(f"\nFound {len(selected_entries)} entries from Group A test indexes\n")
print("=" * 80)

for entry in selected_entries:
    line_num = entry.get("_line_number")
    entry_id = entry.get("entry_id", "N/A")
    status = entry.get("status", "UNKNOWN")
    doi = entry.get("doi", "N/A")
    pmid = entry.get("pmid", "N/A")
    title = entry.get("title", "N/A")[:80] + "..."

    # Count identifiers
    identifiers = entry.get("extracted_identifiers", [])
    dataset_ids = entry.get("dataset_ids", [])
    workspace_keys = entry.get("workspace_metadata_keys", [])
    total_samples = entry.get("total_samples", 0)

    print(f"\n[Line {line_num}] Entry ID: {entry_id}")
    print(f"Title: {title}")
    print(f"Status: {status}")
    print(f"DOI: {doi}")
    print(f"PMID: {pmid}")
    print(f"Extracted Identifiers: {len(identifiers)} items")
    print(f"Dataset IDs: {len(dataset_ids)} items")
    print(f"Workspace Metadata Keys: {len(workspace_keys)} files")
    print(f"Total Samples: {total_samples}")

    if dataset_ids:
        print(f"  Sample dataset IDs: {dataset_ids[:3]}")
    if workspace_keys:
        print(f"  Sample workspace keys: {workspace_keys[:2]}")

    print("-" * 80)

print(f"\n\nGROUP A SUMMARY:")
print(f"Total entries found: {len(selected_entries)}")
ready_for_export = [e for e in selected_entries if e.get("status") in ["HANDOFF_READY", "COMPLETED"]]
print(f"Ready for export (HANDOFF_READY/COMPLETED): {len(ready_for_export)}")
with_samples = [e for e in selected_entries if e.get("total_samples", 0) > 0]
print(f"With sample counts: {len(with_samples)}")

if with_samples:
    total_samples = sum(e.get("total_samples", 0) for e in with_samples)
    print(f"Total samples across all entries: {total_samples}")
    print(f"Average samples per entry: {total_samples / len(with_samples):.1f}")

# Identify representative entries for Phase 2 export testing
print("\n\nRECOMMENDED ENTRIES FOR PHASE 2 EXPORT (diverse sizes):")
with_samples_sorted = sorted(with_samples, key=lambda x: x.get("total_samples", 0))
if len(with_samples_sorted) >= 3:
    # Pick small, medium, large
    small = with_samples_sorted[0]
    medium = with_samples_sorted[len(with_samples_sorted) // 2]
    large = with_samples_sorted[-1]

    print(f"1. SMALL: Line {small['_line_number']}, Entry {small['entry_id']}, {small['total_samples']} samples")
    print(f"2. MEDIUM: Line {medium['_line_number']}, Entry {medium['entry_id']}, {medium['total_samples']} samples")
    print(f"3. LARGE: Line {large['_line_number']}, Entry {large['entry_id']}, {large['total_samples']} samples")
elif with_samples_sorted:
    print("Fewer than 3 entries with samples. Testing all:")
    for i, e in enumerate(with_samples_sorted, 1):
        print(f"{i}. Line {e['_line_number']}, Entry {e['entry_id']}, {e['total_samples']} samples")
else:
    print("NO ENTRIES WITH SAMPLES FOUND - cannot proceed with Phase 2 exports")
