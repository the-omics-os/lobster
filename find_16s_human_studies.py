#!/usr/bin/env python3
"""
Script to identify 16S amplicon human gut microbiome studies in RIS file.

Usage: python find_16s_human_studies.py
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple

def parse_ris_file(ris_path: Path) -> List[Dict[str, str]]:
    """Parse RIS file into list of entry dictionaries."""
    entries = []
    current_entry = {}
    current_field = None

    with open(ris_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')

            # End of record
            if line.startswith('ER  -'):
                if current_entry:
                    entries.append(current_entry)
                    current_entry = {}
                current_field = None
                continue

            # Field tag (e.g., "TI  -", "AB  -")
            if re.match(r'^[A-Z0-9]{2}\s\s-', line):
                parts = line.split('  - ', 1)
                if len(parts) == 2:
                    tag, value = parts
                    tag = tag.strip()
                    current_field = tag
                    if tag in current_entry:
                        current_entry[tag] += ' ' + value
                    else:
                        current_entry[tag] = value
            # Continuation line
            elif current_field and line.strip():
                current_entry[current_field] += ' ' + line.strip()

    return entries

def score_entry_for_16s_human_gut(entry: Dict[str, str]) -> Tuple[int, Dict[str, bool]]:
    """
    Score an entry based on relevance to 16S human gut microbiome studies.

    Returns:
        (score, criteria_dict) where higher score = more relevant
    """
    score = 0
    criteria = {}

    # Combine searchable text
    text = ' '.join([
        entry.get('TI', ''),  # Title
        entry.get('AB', ''),  # Abstract
        entry.get('KW', ''),  # Keywords
    ]).lower()

    # Criterion 1: 16S amplicon sequencing (CRITICAL)
    amplicon_keywords = [
        r'\b16s\b', r'\b16s rrna\b', r'\bamplicon\b',
        r'\bv3-v4\b', r'\bv4 region\b', r'\bv3 region\b',
        r'\b16s ribosomal rna\b', r'\b16s rdna\b',
        r'\b16s rrna gene sequencing\b', r'\b16s amplicon sequencing\b'
    ]
    has_16s = any(re.search(pattern, text) for pattern in amplicon_keywords)
    criteria['has_16s'] = has_16s
    if has_16s:
        score += 10  # CRITICAL

    # Criterion 2: Human subjects
    human_keywords = [
        r'\bhuman\b', r'\bhomo sapiens\b', r'\bpatient\b',
        r'\bparticipant\b', r'\bsubject\b', r'\bpersons\b'
    ]
    has_human = any(re.search(pattern, text) for pattern in human_keywords)
    criteria['has_human'] = has_human
    if has_human:
        score += 5

    # Criterion 3: Gut/fecal/stool samples (HIGH VALUE)
    gut_keywords = [
        r'\bfecal\b', r'\bfaecal\b', r'\bstool\b', r'\bgut\b',
        r'\bintestinal\b', r'\bcolon\b', r'\brectal\b', r'\bmicrobiota\b',
        r'\bmicrobiome\b', r'\bfeces\b'
    ]
    has_gut = any(re.search(pattern, text) for pattern in gut_keywords)
    criteria['has_gut'] = has_gut
    if has_gut:
        score += 3

    # Criterion 4: CRC/cancer context (NICE TO HAVE)
    cancer_keywords = [
        r'\bcrc\b', r'\bcolorectal cancer\b', r'\bcolon cancer\b',
        r'\bcancer\b', r'\btumor\b', r'\bcarcinoma\b',
        r'\bibd\b', r'\bcrohn\b', r'\bulcerative colitis\b'
    ]
    has_cancer = any(re.search(pattern, text) for pattern in cancer_keywords)
    criteria['has_cancer'] = has_cancer
    if has_cancer:
        score += 2

    # Criterion 5: Exclude animal-only studies (PENALTY)
    animal_only_keywords = [
        r'\bmouse\b', r'\bmice\b', r'\brat\b', r'\bbovine\b',
        r'\bporcine\b', r'\bcanine\b', r'\bmurine\b'
    ]
    # Only penalize if animal terms appear but no human terms
    has_animal_only = (
        any(re.search(pattern, text) for pattern in animal_only_keywords)
        and not has_human
    )
    criteria['animal_only'] = has_animal_only
    if has_animal_only:
        score -= 5

    # Criterion 6: Exclude methodology/primer design papers (PENALTY)
    method_keywords = [
        r'\bprimer design\b', r'\bsoftware\b', r'\btool\b',
        r'\bpipeline development\b', r'\balgorithm\b', r'\bmethod comparison\b'
    ]
    is_methods_paper = any(re.search(pattern, text) for pattern in method_keywords)
    # Only penalize if it's ONLY about methods (no actual gut microbiome study)
    if is_methods_paper and not has_gut:
        criteria['methods_only'] = True
        score -= 3
    else:
        criteria['methods_only'] = False

    return score, criteria

def main():
    ris_path = Path('kevin_notes/databiomix/CRC_microbiome.ris')

    print(f"Parsing RIS file: {ris_path}")
    entries = parse_ris_file(ris_path)
    print(f"Total entries found: {len(entries)}\n")

    # Score all entries
    scored_entries = []
    for idx, entry in enumerate(entries):
        score, criteria = score_entry_for_16s_human_gut(entry)
        scored_entries.append({
            'index': idx,
            'score': score,
            'criteria': criteria,
            'title': entry.get('TI', 'No title')[:100],
            'entry': entry
        })

    # Sort by score (descending)
    scored_entries.sort(key=lambda x: x['score'], reverse=True)

    # Filter for high-confidence 16S human gut studies
    # Minimum score: 10 (has_16s=10) + 5 (has_human=5) = 15
    high_confidence = [e for e in scored_entries if e['score'] >= 15]

    print("=" * 80)
    print(f"HIGH CONFIDENCE 16S HUMAN GUT STUDIES (score >= 15)")
    print("=" * 80)
    print(f"Found {len(high_confidence)} entries\n")

    # Display top 20 results
    for i, result in enumerate(high_confidence[:20], 1):
        idx = result['index']
        score = result['score']
        title = result['title']
        criteria = result['criteria']

        print(f"{i}. Entry {idx} (Score: {score})")
        print(f"   Title: {title}")
        print(f"   Criteria: 16S={criteria['has_16s']}, Human={criteria['has_human']}, "
              f"Gut={criteria['has_gut']}, Cancer={criteria['has_cancer']}")
        print()

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_16s = sum(1 for e in scored_entries if e['criteria']['has_16s'])
    total_human = sum(1 for e in scored_entries if e['criteria']['has_human'])
    total_gut = sum(1 for e in scored_entries if e['criteria']['has_gut'])
    total_16s_human = sum(1 for e in scored_entries if e['criteria']['has_16s'] and e['criteria']['has_human'])
    total_16s_human_gut = sum(1 for e in scored_entries if
                               e['criteria']['has_16s'] and
                               e['criteria']['has_human'] and
                               e['criteria']['has_gut'])

    print(f"Entries with 16S: {total_16s}/{len(entries)} ({total_16s/len(entries)*100:.1f}%)")
    print(f"Entries with Human: {total_human}/{len(entries)} ({total_human/len(entries)*100:.1f}%)")
    print(f"Entries with Gut/Fecal: {total_gut}/{len(entries)} ({total_gut/len(entries)*100:.1f}%)")
    print(f"Entries with 16S + Human: {total_16s_human}/{len(entries)} ({total_16s_human/len(entries)*100:.1f}%)")
    print(f"Entries with 16S + Human + Gut: {total_16s_human_gut}/{len(entries)} ({total_16s_human_gut/len(entries)*100:.1f}%)")

    # Recommended test entries
    print("\n" + "=" * 80)
    print("RECOMMENDED TEST ENTRIES (Top 10)")
    print("=" * 80)
    recommended_indices = [e['index'] for e in high_confidence[:10]]
    print(f"Entry indices: {','.join(map(str, recommended_indices))}")
    print("\nCommand to test:")
    print(f"python tests/manual/test_publication_processing.py \\")
    print(f"    --ris-file kevin_notes/databiomix/CRC_microbiome.ris \\")
    print(f"    --entry {','.join(map(str, recommended_indices))} \\")
    print(f"    --parallel 6 \\")
    print(f"    --test-handoff \\")
    print(f"    --filter-criteria \"16S human fecal CRC\"")

if __name__ == '__main__':
    main()
