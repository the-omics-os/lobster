#!/usr/bin/env python3
"""
Script to inspect publication queue entries for Group A test mission.
"""
import json
from pathlib import Path

# Test indexes for Group A
TARGET_IDS = [419, 415, 434, 433, 437, 448, 454, 452, 455]

queue_path = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11/.lobster/queues/publication_queue.jsonl")

entries = []
with open(queue_path) as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            if entry.get("entry_id") in TARGET_IDS:
                entries.append(entry)

# Sort by entry_id
entries.sort(key=lambda x: x.get("entry_id", 0))

print(f"Found {len(entries)} entries from Group A test indexes\n")
print("=" * 80)

for entry in entries:
    entry_id = entry.get("entry_id")
    status = entry.get("status", "UNKNOWN")
    doi = entry.get("doi", "N/A")
    pmid = entry.get("pmid", "N/A")

    # Count identifiers
    identifiers = entry.get("extracted_identifiers", [])
    dataset_ids = entry.get("dataset_ids", [])
    workspace_keys = entry.get("workspace_metadata_keys", [])
    total_samples = entry.get("total_samples", 0)

    print(f"\nEntry ID: {entry_id}")
    print(f"Status: {status}")
    print(f"DOI: {doi}")
    print(f"PMID: {pmid}")
    print(f"Extracted Identifiers: {len(identifiers)} items")
    print(f"Dataset IDs: {len(dataset_ids)} items")
    print(f"Workspace Metadata Keys: {len(workspace_keys)} files")
    print(f"Total Samples: {total_samples}")

    if dataset_ids:
        print(f"  Sample dataset IDs: {dataset_ids[:3]}...")
    if workspace_keys:
        print(f"  Sample workspace keys: {workspace_keys[:2]}...")

    print("-" * 80)

print(f"\n\nSUMMARY:")
print(f"Total entries found: {len(entries)}")
ready_for_export = [e for e in entries if e.get("status") in ["HANDOFF_READY", "COMPLETED"]]
print(f"Ready for export (HANDOFF_READY/COMPLETED): {len(ready_for_export)}")
with_samples = [e for e in entries if e.get("total_samples", 0) > 0]
print(f"With sample counts: {len(with_samples)}")

if with_samples:
    total_samples = sum(e.get("total_samples", 0) for e in with_samples)
    print(f"Total samples across all entries: {total_samples}")
    print(f"Average samples per entry: {total_samples / len(with_samples):.1f}")
