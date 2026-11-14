"""Write a FICTIONAL RNA-seq review dataset for the decision-components demo.

This is a synthetic example. Nothing here comes from a real organism, assay,
person or dataset. Each case writes the same kinds of file into
``<workspace>/data/``, where Lobster lists uploaded data:

* three count sources (two raw-count matrices on Build-A, one normalized
  summary with an unknown build), plus a manifest describing them;
* a sample table where some conditions are "unknown" and every display label
  is shared by two different samples;
* an auxiliary manifest that suggests a condition for one unknown sample;
* made-up QC numbers that match the supplied warnings;
* the analysis policy for this example, a README and a checksums file.

Cases:

* ``review8`` (default): 8 samples S01-S08, seed 20261029, file prefix
  ``syn_rnaseq_``. The same files and counts as the earlier demo.
* ``review72``: 72 samples S01-S72 in 2 batches (34 control, 34 treated,
  4 unknown, one of them with a suggestion), 8 QC warnings plus the 4
  unresolved conditions, seed 20261103, file prefix ``syn_rnaseq72_``.

Each seed is fixed, so the same command always writes the same bytes, and
``<prefix>_checksums.sha256`` lets anyone check that (``sha256sum -c``).

Usage:
    uv run python docs/dynamic-ui/demo/make_synthetic_rnaseq.py <workspace> \
        [--case review8|review72]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_GENES = 500
LOW_LIBRARY_FACTOR = 0.2  # a warned "low library size" sample gets 1/5 of the reads
BATCH_SHIFT = {"B1": 1.0, "B2": 1.15}
LOW_LIBRARY = "Low library size"
HIGH_MITO = "High mitochondrial fraction"
UNRESOLVED = "Condition label unresolved"  # the warning every unknown sample carries
PREFIX8, PREFIX72 = "syn_rnaseq", "syn_rnaseq72"

# The policy for this synthetic example. Its rules hold for this made-up data
# only; they are not general analysis advice.
POLICY_TEMPLATE = """\
FICTIONAL ANALYSIS POLICY - synthetic example for the decision-components demo
Snapshot: {snapshot}
These rules belong to this made-up dataset only. They are not general advice.

1. Count sources: only raw gene counts on reference Build-A can be used here.
   A source with another representation or another build is not usable in
   this example. That is a rule of the example, not a judgement on such data.
2. Unknown conditions: a sample whose condition is "unknown" has to be
   resolved by a person, or kept out of the comparison. Never guess it.
3. Warnings and exclusions: a supplied warning is information, not a
   decision. It does not remove a sample on its own, and every sample that
   is removed needs a stated reason.
4. Direction: a positive effect means higher expression in the treated
   samples than in the control samples.
5. Display labels: {label_rule}
   A shared label does not make them the same sample; do not pair them.
6. Groups: control and treated must each keep at least one sample. A
   deferred sample does not count toward its group.
"""


@dataclass
class Case:
    """Everything that differs between the two synthetic cases."""

    name: str
    prefix: str
    snapshot: str
    seed: int
    # (sample id, display label, condition, batch, supplied warning)
    samples: list[tuple[str, str, str, str, str]]
    # The hidden truth behind each "unknown" condition; it shapes the counts.
    hidden_condition: dict[str, str]
    high_mito: dict[str, float]  # fixed QC values for warned samples
    suggestions: dict[str, str]  # sample id -> condition the auxiliary file suggests
    label_rule: str
    summary: str
    low_library: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.low_library = {s[0] for s in self.samples if s[4] == LOW_LIBRARY}


def _sources(prefix: str) -> list[tuple[str, ...]]:
    """The three count sources. Which ones are usable follows from the policy."""
    return [
        # id, file, description, representation, reference build, provenance
        (
            "SRC-A",
            f"{prefix}_SRC-A_author_counts.csv",
            "Author-supplied matrix",
            "raw gene counts",
            "Build-A",
            "Synthetic author manifest A",
        ),
        (
            "SRC-B",
            f"{prefix}_SRC-B_repository_counts.csv",
            "Repository-generated matrix",
            "raw gene counts",
            "Build-A",
            "Synthetic processing manifest B",
        ),
        (
            "SRC-C",
            f"{prefix}_SRC-C_author_summary.csv",
            "Author summary table",
            "normalized expression (log2 CPM)",
            "unknown",
            "Processing details unavailable",
        ),
    ]


def _counts(case: Case, rng: np.random.Generator) -> np.ndarray:
    """Raw counts (genes x samples). Treated samples raise genes 1-25 and lower 26-50."""
    base = rng.gamma(shape=2.0, scale=60.0, size=N_GENES)
    effect = np.ones(N_GENES)
    effect[:25], effect[25:50] = 4.0, 0.25
    matrix = np.zeros((N_GENES, len(case.samples)), dtype=np.int64)
    for j, (sid, _label, condition, batch, _warning) in enumerate(case.samples):
        treated = case.hidden_condition.get(sid, condition) == "treated"
        library = LOW_LIBRARY_FACTOR if sid in case.low_library else 1.0
        mean = base * (effect if treated else 1.0) * library * BATCH_SHIFT[batch]
        # Negative binomial drawn as gamma-Poisson (dispersion 0.1).
        lam = rng.gamma(shape=10.0, scale=mean / 10.0)
        matrix[:, j] = rng.poisson(lam)
    return matrix


def _write_rows(path: Path, header, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_matrix(path: Path, case: Case, genes, values, fmt: str) -> None:
    # The index header says FICTIONAL; a comment line would break CSV loaders.
    _write_rows(
        path,
        ["FICTIONAL_gene_id", *[s[0] for s in case.samples]],
        ([gene, *[fmt.format(v) for v in row]] for gene, row in zip(genes, values)),
    )


def _readme(case: Case) -> str:
    p = case.prefix
    lines = [
        f"FICTIONAL DATA. Every file whose name starts with {p}_ is made up.",
        "",
        f"Case: {case.name} (snapshot {case.snapshot}, seed {case.seed})",
        "Written by docs/dynamic-ui/demo/make_synthetic_rnaseq.py as a synthetic",
        "example for the decision-components demo. No real organism, assay,",
        "person or dataset is represented, and the numbers measure nothing.",
        "",
        case.summary,
        "",
        "Files (the three SRC files are genes x samples tables):",
        f"  {p}_SRC-A_author_counts.csv      raw counts, Build-A",
        f"  {p}_SRC-B_repository_counts.csv  raw counts, Build-A",
        f"  {p}_SRC-C_author_summary.csv     log2 CPM, build unknown",
        f"  {p}_sources_manifest.csv         where each source came from",
        f"  {p}_sample_metadata.csv          one row per sample",
        f"  {p}_auxiliary_manifest.csv       suggested conditions, not facts",
        f"  {p}_qc_metrics.csv               made-up QC values",
        f"  {p}_analysis_policy.txt          the rules for this example",
        f"  {p}_checksums.sha256             SHA-256 of every other file",
    ]
    return "\n".join(lines) + "\n"


def _write_case(case: Case, workspace: Path) -> Path:
    """Write every file of one case into <workspace>/data/; return that folder."""
    # The order of random draws matters: it fixes the bytes for each seed.
    rng = np.random.default_rng(case.seed)
    out = workspace / "data"
    out.mkdir(parents=True, exist_ok=True)
    p = case.prefix
    genes = [f"SYNG{i:04d}" for i in range(1, N_GENES + 1)]
    sources = _sources(p)

    src_a = _counts(case, rng)
    # SRC-B: the same samples run through a (made-up) second pipeline, so the
    # counts differ a little.
    src_b = np.rint(src_a * rng.uniform(0.9, 1.1, size=src_a.shape)).astype(np.int64)
    # SRC-C: a normalized summary of SRC-A (log2 CPM), not raw counts.
    src_c = np.log2(src_a / src_a.sum(axis=0, keepdims=True) * 1e6 + 1.0)
    _write_matrix(out / sources[0][1], case, genes, src_a, "{:d}")
    _write_matrix(out / sources[1][1], case, genes, src_b, "{:d}")
    _write_matrix(out / sources[2][1], case, genes, src_c, "{:.4f}")

    _write_rows(
        out / f"{p}_sources_manifest.csv",
        [
            "source_id",
            "file",
            "description",
            "representation",
            "reference_build",
            "provenance",
            "note",
        ],
        [(*row, "FICTIONAL source") for row in sources],
    )
    _write_rows(
        out / f"{p}_sample_metadata.csv",
        [
            "sample_id",
            "display_label",
            "condition",
            "batch",
            "supplied_warning",
            "note",
        ],
        [(*row, "FICTIONAL sample") for row in case.samples],
    )
    _write_rows(
        out / f"{p}_auxiliary_manifest.csv",
        ["sample_id", "proposed_condition", "authority", "note"],
        [
            (
                sid,
                proposed,
                "suggestion_only",
                "FICTIONAL. A second manifest suggests this; a person must "
                "confirm it before it is used",
            )
            for sid, proposed in case.suggestions.items()
        ],
    )

    qc_rows = []
    for j, (sid, *_rest) in enumerate(case.samples):
        drawn = round(float(rng.uniform(0.03, 0.08)), 3)  # drawn for every sample
        qc_rows.append(
            (
                sid,
                int(src_a[:, j].sum()),
                f"{case.high_mito.get(sid, drawn):.3f}",
                int((src_a[:, j] > 0).sum()),
                "FICTIONAL synthetic value, not a measurement",
            )
        )
    _write_rows(
        out / f"{p}_qc_metrics.csv",
        ["sample_id", "library_size", "mito_fraction", "genes_detected", "note"],
        qc_rows,
    )

    (out / f"{p}_analysis_policy.txt").write_text(
        POLICY_TEMPLATE.format(snapshot=case.snapshot, label_rule=case.label_rule)
    )

    files = [s[1] for s in sources] + [
        f"{p}_{name}"
        for name in (
            "sources_manifest.csv",
            "sample_metadata.csv",
            "auxiliary_manifest.csv",
            "qc_metrics.csv",
            "analysis_policy.txt",
            "README_SYNTHETIC.txt",
        )
    ]
    (out / f"{p}_README_SYNTHETIC.txt").write_text(_readme(case))

    # Plain "<hash>  <name>" lines only, so any sha256sum -c accepts the file.
    # It is the one file without the word FICTIONAL; the README covers it.
    (out / f"{p}_checksums.sha256").write_text(
        "".join(
            f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}\n"
            for n in sorted(files)
        )
    )
    return out


def generate_review8(workspace: Path) -> Path:
    """The 8-sample case: S08's condition is unknown and has one suggestion."""
    samples = [
        ("S01", "Donor A", "control", "B1", ""),
        ("S02", "Donor B", "control", "B1", LOW_LIBRARY),
        ("S03", "Donor C", "control", "B2", ""),
        ("S04", "Donor D", "control", "B2", ""),
        ("S05", "Donor A", "treated", "B1", ""),
        ("S06", "Donor F", "treated", "B1", HIGH_MITO),
        ("S07", "Donor G", "treated", "B2", ""),
        ("S08", "Donor H", "unknown", "B2", UNRESOLVED),
    ]
    case = Case(
        name="review8",
        prefix=PREFIX8,
        snapshot="synthetic-rnaseq-review-v1",
        seed=20261029,
        samples=samples,
        hidden_condition={"S08": "treated"},
        high_mito={"S06": 0.19},
        suggestions={"S08": "treated"},
        label_rule='S01 and S05 both carry "Donor A".',
        summary=(
            "8 samples, S01-S08: 4 control, 3 treated, 1 unknown (S08).\n"
            "Warnings: S02 low library size, S06 high mitochondrial fraction,\n"
            "S08 condition unresolved."
        ),
    )
    return _write_case(case, workspace)


# review72 layout: S01-S36 are batch B1, S37-S72 batch B2. Odd numbers are
# control and even numbers treated, so samples 2k-1 and 2k share the display
# label "Donor k" while sitting in different groups. Four samples (two per
# batch, two from each group) have their condition replaced by "unknown".
R72_UNKNOWN = ("S15", "S30", "S49", "S68")
R72_SUGGESTED = "S68"  # the one unknown sample with an auxiliary suggestion
R72_WARNINGS = {
    # one low-library and one high-mito sample per batch-and-group cell
    "S04": LOW_LIBRARY,
    "S21": LOW_LIBRARY,
    "S43": LOW_LIBRARY,
    "S58": LOW_LIBRARY,
    "S09": HIGH_MITO,
    "S26": HIGH_MITO,
    "S39": HIGH_MITO,
    "S62": HIGH_MITO,
}
R72_HIGH_MITO = {"S09": 0.18, "S26": 0.21, "S39": 0.19, "S62": 0.24}


def generate_review72(workspace: Path) -> Path:
    """The 72-sample case: more rows than one screen and than nine number keys."""
    samples, hidden = [], {}
    for i in range(1, 73):
        sid = f"S{i:02d}"
        planned = "control" if i % 2 else "treated"
        condition = planned
        warning = R72_WARNINGS.get(sid, "")
        if sid in R72_UNKNOWN:
            condition, hidden[sid], warning = "unknown", planned, UNRESOLVED
        samples.append(
            (
                sid,
                f"Donor {(i + 1) // 2:02d}",
                condition,
                "B1" if i <= 36 else "B2",
                warning,
            )
        )
    case = Case(
        name="review72",
        prefix=PREFIX72,
        snapshot="synthetic-rnaseq-review72-v1",
        seed=20261103,
        samples=samples,
        hidden_condition=hidden,
        high_mito=R72_HIGH_MITO,
        suggestions={R72_SUGGESTED: hidden[R72_SUGGESTED]},
        label_rule=(
            "each label is used by two samples (S01 and S02\n"
            '   both carry "Donor 01").'
        ),
        summary=(
            "72 samples, S01-S72, batches B1 (S01-S36) and B2 (S37-S72):\n"
            f"34 control, 34 treated, 4 unknown ({', '.join(R72_UNKNOWN)}).\n"
            f"Only {R72_SUGGESTED} has a suggested condition.\n"
            "Low library size: "
            + ", ".join(s for s, w in R72_WARNINGS.items() if w == LOW_LIBRARY)
            + ".\nHigh mitochondrial fraction: "
            + ", ".join(s for s, w in R72_WARNINGS.items() if w == HIGH_MITO)
            + f".\nCondition label unresolved: {', '.join(R72_UNKNOWN)}.\n"
            "That is 8 QC warnings plus 4 unresolved conditions."
        ),
    )
    return _write_case(case, workspace)


CASES = {"review8": generate_review8, "review72": generate_review72}
PREFIXES = {"review8": PREFIX8, "review72": PREFIX72}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workspace", type=Path, help="Lobster workspace directory")
    parser.add_argument("--case", choices=sorted(CASES), default="review8")
    args = parser.parse_args()
    out = CASES[args.case](args.workspace)

    prefix = PREFIXES[args.case]
    with (out / f"{prefix}_sample_metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = Counter(r["condition"] for r in rows)
    warnings = Counter(r["supplied_warning"] for r in rows if r["supplied_warning"])
    unresolved = warnings.pop(UNRESOLVED, 0)
    print(f"FICTIONAL synthetic case {args.case}: wrote {prefix}_* files to {out}")
    print(
        f"  {len(rows)} samples: "
        + ", ".join(f"{groups[c]} {c}" for c in ("control", "treated", "unknown"))
        + f"; {sum(warnings.values())} QC warnings + {unresolved} unresolved"
        + (" condition" if unresolved == 1 else " conditions")
    )


if __name__ == "__main__":
    main()
