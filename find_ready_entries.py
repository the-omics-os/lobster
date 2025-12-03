#!/usr/bin/env python3
"""
Find entries in publication queue that are ready for export testing.
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

print(f"Total entries in queue: {len(all_entries)}\n")

# Filter for entries ready for export
ready_entries = [
    e for e in all_entries
    if e.get("status") in ["HANDOFF_READY", "COMPLETED"]
       and e.get("total_samples", 0) > 0
       and len(e.get("workspace_metadata_keys", [])) > 0
]

print(f"Entries ready for export (HANDOFF_READY/COMPLETED with samples): {len(ready_entries)}")

if ready_entries:
    # Sort by sample count
    ready_entries.sort(key=lambda x: x.get("total_samples", 0))

    print("\nFirst 10 ready entries (sorted by sample count):")
    print("=" * 100)

    for i, entry in enumerate(ready_entries[:10], 1):
        line_num = entry.get("_line_number")
        entry_id = entry.get("entry_id", "N/A")
        status = entry.get("status", "UNKNOWN")
        total_samples = entry.get("total_samples", 0)
        dataset_ids = entry.get("dataset_ids", [])
        workspace_keys = entry.get("workspace_metadata_keys", [])
        title = entry.get("title", "N/A")[:70]

        print(f"\n{i}. [Line {line_num}] {entry_id}")
        print(f"   Title: {title}...")
        print(f"   Status: {status} | Samples: {total_samples} | Datasets: {len(dataset_ids)} | Metadata files: {len(workspace_keys)}")

    # Identify diverse set for testing
    print("\n\n" + "=" * 100)
    print("RECOMMENDED TEST SET (diverse sample sizes):")
    print("=" * 100)

    small = ready_entries[0]
    medium = ready_entries[len(ready_entries) // 2]
    large = ready_entries[-1]

    for label, entry in [("SMALL", small), ("MEDIUM", medium), ("LARGE", large)]:
        line_num = entry.get("_line_number")
        entry_id = entry.get("entry_id", "N/A")
        total_samples = entry.get("total_samples", 0)
        print(f"\n{label}:")
        print(f"  Line: {line_num}")
        print(f"  Entry ID: {entry_id}")
        print(f"  Samples: {total_samples}")
        print(f"  Workspace keys: {entry.get('workspace_metadata_keys', [])[:2]}")

else:
    print("\nNO ENTRIES READY FOR EXPORT FOUND!")
    print("\nStatus breakdown:")
    status_counts = {}
    for e in all_entries:
        status = e.get("status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count} entries")

    print("\nEntries with samples (any status):")
    with_samples = [e for e in all_entries if e.get("total_samples", 0) > 0]
    print(f"  {len(with_samples)} entries have samples")

    if with_samples:
        print("\nFirst 5 entries with samples:")
        for i, e in enumerate(with_samples[:5], 1):
            print(f"  {i}. Line {e['_line_number']}: {e.get('entry_id')} ({e.get('total_samples')} samples, status: {e.get('status')})")
