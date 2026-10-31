"""User values in IR code templates are written as Python literals.

Exported notebook code is rendered from Jinja templates. A value pasted between
hand-written quotes, into f-string text, or into a ``#`` comment can break the
generated code or inject code when it contains a quote, newline, brace or
backslash. Templates now pass such values through the ``repr`` filter.

- ``test_no_hand_quoted_placeholders`` / ``test_no_raw_placeholders_in_comments``
  are a ratchet over the source tree (a simple line scan, not a proof).
- The render tests build real IR with a hostile value and check that the code
  still parses, carries the value exactly, and imports nothing extra.
"""

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HOSTILE = "x'\"); import os  #\n{y}\\"

SAFE_FILTERS = ("repr", "tojson")
QUOTES = ("'", '"')

# Lines that match the scan but are safe, each with its reason.
ALLOWED_QUOTED = {
    # docstring example, not a template
    "lobster/services/data_access/proteomics_parsers/base_parser.py": "docstring",
    # value is only quoted after an isidentifier() check in the template
    "packages/lobster-transcriptomics/lobster/agents/transcriptomics/annotation_expert.py": "isidentifier guard",
}
# Numeric settings in comments; a number cannot break out of a comment.
NUMERIC_IN_COMMENTS = {
    "min_genes_per_cell",
    "max_genes_per_cell",
    "min_cells_per_gene",
    "max_mito_percent",
    "max_ribo_percent",
    "n_top_genes",
    "min_confidence",
    "n_factors",
    "extreme_log2fc_threshold",
    "modality_names|length",
    "OUTPUT_DIR",  # Python f-string example text, not a Jinja placeholder
}


def _source_files():
    roots = [REPO / "lobster", *sorted((REPO / "packages").glob("*/lobster"))]
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "/tests/" not in path.as_posix():
                yield path


def _placeholders(line):
    """(start, end, expression) for each ``{{ ... }}`` on a source line.

    Also finds the doubled ``{{{{ x }}}}`` form used in Python f-strings that
    build templates; start/end cover the whole brace run. Templates here write
    placeholders with inner spaces; ``{{name}}`` without them is a Python
    f-string brace escape, not Jinja.
    """
    for match in re.finditer(r"\{\{+\s+([^{}]*?)\s+\}\}+", line):
        yield match.start(), match.end(), match.group(1)


def _unsafe(expression):
    return not any(f in expression for f in SAFE_FILTERS)


def test_no_hand_quoted_placeholders():
    hits = []
    for path in _source_files():
        rel = path.relative_to(REPO).as_posix()
        if rel in ALLOWED_QUOTED:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for start, end, expression in _placeholders(line):
                before = line[start - 1] if start else ""
                after = line[end] if end < len(line) else ""
                # A closing quote followed by another quote ends a Python
                # triple-quoted literal, not a template string.
                after_next = line[end + 1] if end + 1 < len(line) else ""
                quoted = before in QUOTES or (
                    after in QUOTES and after_next not in QUOTES
                )
                if quoted and _unsafe(expression):
                    hits.append(f"{rel}:{number}: {line.strip()}")
    assert hits == []


def test_no_raw_placeholders_in_comments():
    hits = []
    for path in _source_files():
        rel = path.relative_to(REPO).as_posix()
        for number, line in enumerate(path.read_text().splitlines(), 1):
            comment = re.search(r'(^\s*|"""|^\s*")#', line)
            if not comment:
                continue
            for _, _, expression in _placeholders(line[comment.end() :]):
                if _unsafe(expression) and expression not in NUMERIC_IN_COMMENTS:
                    hits.append(f"{rel}:{number}: {line.strip()}")
    assert hits == []


# --- hostile values through real IR builders ---------------------------------


def _strings(tree):
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _imports(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def _subcluster(value):
    from lobster.services.analysis.clustering_service import ClusteringService

    service = object.__new__(ClusteringService)
    return service._create_subcluster_ir(
        cluster_key=value, clusters_to_refine=[value], resolution=0.5
    )


def test_clustering_ir_rejects_unknown_algorithm():
    """The template calls ``sc.tl.<algorithm>``, so only known names pass."""
    from lobster.services.analysis.clustering_service import ClusteringService

    service = object.__new__(ClusteringService)
    with pytest.raises(ValueError, match="Unknown clustering algorithm"):
        service._create_clustering_ir(resolution=1.0, algorithm=HOSTILE)
    assert (
        "sc.tl.louvain("
        in service._create_clustering_ir(resolution=1.0, algorithm="Louvain").render()
    )


def _bulk_de(value):
    from lobster.services.analysis.bulk_rnaseq_service import BulkRNASeqService

    service = object.__new__(BulkRNASeqService)
    return service._create_de_ir("deseq2", groupby=value, group1=value, group2="b")


def _plot(value):
    from lobster.core.plot_manager import PlotManager

    manager = object.__new__(PlotManager)
    return manager._create_add_plot_ir(value, value, {})


def _modality_info(value):
    from lobster.services.data_management.modality_management_service import (
        ModalityManagementService,
    )

    service = object.__new__(ModalityManagementService)
    return service._create_get_info_ir(value)


def _formula_de(value):
    from lobster.agents.transcriptomics.de_analysis_expert import _formula_de_ir

    return _formula_de_ir(value, [value, "a", "b"], None, 0.05, 0.0)


@pytest.mark.parametrize(
    "build", [_subcluster, _bulk_de, _plot, _modality_info, _formula_de]
)
def test_hostile_value_stays_a_string(build):
    benign = ast.parse(build("leiden").render())
    code = build(HOSTILE).render()

    tree = ast.parse(code)  # must not raise

    assert any(HOSTILE in s for s in _strings(tree)), code
    assert _imports(tree) == _imports(benign)
