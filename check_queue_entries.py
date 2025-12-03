#!/usr/bin/env python3
"""
Script to check all entry_ids in publication queue.
"""
import json
from pathlib import Path

queue_path = Path("/Users/tyo/GITHUB/omics-os/lobster/results/v11/.lobster/queues/publication_queue.jsonl")

entry_ids = []
with open(queue_path) as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            entry_ids.append(entry.get("entry_id"))

print(f"Total entries: {len(entry_ids)}")
print(f"Entry ID range: {min(entry_ids)} - {max(entry_ids)}")
print(f"\nFirst 20 entry_ids: {sorted(entry_ids)[:20]}")
print(f"\nLast 20 entry_ids: {sorted(entry_ids)[-20:]}")

# Check around the target range
target_range = range(410, 460)
in_range = [eid for eid in entry_ids if eid in target_range]
print(f"\nEntry IDs in range 410-460: {sorted(in_range)}")
