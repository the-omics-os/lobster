"""Text given to a tool stays text in every notebook cell Lobster writes.

Each ``AnalysisStep.code_template`` in the source is rendered with a hostile
string in every text value it uses, once as a Jinja variable and once where an
f-string puts the value into the template itself. A site fails when the string
becomes code in the cell (the cell no longer parses, or ``__import__`` is a name
in it) or when Jinja evaluates it while Lobster renders the cell.

``KNOWN`` lists the sites whose value never comes from the model or from a file
name: each says where its value comes from. A site whose value the model or the
user's data supplies is fixed, not listed.
"""

import ast
from pathlib import Path

import pytest
from jinja2 import meta

from lobster.core.provenance.analysis_ir import _JINJA_ENV

ROOT = Path(__file__).resolve().parents[3]
PAYLOAD = "x'\");\n__import__('os').getcwd()#){{ 7 * 191 }}"
EVALUATED = "1337"
TEXT = {"str", "Optional[str]", "str | None", "Any", "Optional[Any]"}

# (file, function, value): where the value comes from
KNOWN = {
    (
        "lobster/cli_internal/mcp_record.py",
        "outcome_ir",
        "STOPPED_KEPT",
    ): "a module constant",
    (
        "lobster/cli_internal/mcp_record.py",
        "outcome_ir",
        "outcome",
    ): "checked against UNFINISHED_OUTCOMES first",
    (
        "lobster/core/provenance/analysis_ir.py",
        "create_minimal_ir",
        "code",
    ): "Lobster's own fallback code; no caller passes tool text",
    (
        "lobster/core/provenance/analysis_ir.py",
        "create_minimal_ir",
        "operation",
    ): "an operation name written in Lobster code",
    (
        "lobster/core/provenance/analysis_ir.py",
        "create_data_loading_ir",
        "input_param_name",
    ): "a parameter name written in Lobster code",
    (
        "lobster/core/provenance/analysis_ir.py",
        "create_adapter_loading_ir",
        "input_param_name",
    ): "a parameter name written in Lobster code",
    (
        "lobster/core/provenance/analysis_ir.py",
        "create_data_saving_ir",
        "output_prefix_param",
    ): "a parameter name written in Lobster code",
    (
        "lobster/services/analysis/clustering_service.py",
        "_create_neighbors_embed_ir",
        "embed_code",
    ): "fixed lines chosen in the function",
    (
        "lobster/services/data_access/geo_download_service.py",
        "_anchored_load_ir",
        "raw_name",
    ): "the name Lobster gave the raw file it saved",
    (
        "lobster/services/execution/custom_code_execution_service.py",
        "_create_ir",
        "code",
    ): "custom code is code by design; the exporter builds its cell",
    (
        "packages/lobster-metadata/lobster/services/metadata/disease_standardization_service.py",
        "_create_standardization_ir",
        "disease_mappings",
    ): "the service's fixed mapping table",
    (
        "packages/lobster-ml/lobster/services/quality/cross_validation_service.py",
        "_create_stratified_kfold_ir",
        "model_class",
    ): "the class name of a Python caller's model factory; no tool calls it",
    (
        "packages/lobster-proteomics/lobster/agents/proteomics/shared_tools.py",
        "import_affinity_data",
        "detected_platform",
    ): "a parser name from the fixed platform map or the detection",
    (
        "packages/lobster-transcriptomics/lobster/services/analysis/enhanced_singlecell_service.py",
        "_create_marker_genes_ir",
        "extreme_log2fc_threshold",
    ): "the module constant EXTREME_LOG2FC_THRESHOLD",
    (
        "packages/lobster-visualization/lobster/agents/visualization_expert.py",
        "_plot_ir",
        "service_method",
    ): "a method name each caller writes as a constant",
    (
        "packages/lobster-visualization/lobster/agents/visualization_expert.py",
        "_plot_ir",
        "kwargs_template",
    ): "built from the keyword names; each value is repr'd",
}
# The drug-discovery request cells pass the module's DEFAULT_HTTP_TIMEOUT.
_DRUG_DISCOVERY = "packages/lobster-drug-discovery/lobster/services/drug_discovery/"
KNOWN.update(
    {
        (_DRUG_DISCOVERY + name, function, "timeout"): "DEFAULT_HTTP_TIMEOUT"
        for name, functions in {
            "chembl_service.py": [
                "_create_ir_get_bioactivity",
                "_create_ir_get_target_compounds",
                "_create_ir_search_compounds",
            ],
            "opentargets_service.py": [
                "_create_ir_assess_tractability",
                "_create_ir_get_drug_indications",
                "_create_ir_get_safety_profile",
                "_create_ir_get_target_disease_evidence",
                "_create_ir_score_target",
            ],
            "pubchem_service.py": [
                "_create_ir_get_compound_properties",
                "_create_ir_get_compound_synonyms",
                "_create_ir_search_by_similarity",
            ],
        }.items()
        for function in functions
    }
)

# Templates built in ways this test cannot read, checked by hand: (file, function)
UNREAD = {
    ("lobster/cli_internal/mcp_record.py", "_host_text_ir"): "the text is repr'd",
    (
        "lobster/services/analysis/clustering_service.py",
        "_create_clustering_ir",
    ): "algorithm is checked against leiden/louvain first; the rest are numbers",
    (
        "lobster/services/analysis/clustering_service.py",
        "_create_embedding_clustering_ir",
    ): "algorithm is checked against leiden/louvain first; the rest are numbers",
    (
        "lobster/services/data_access/proteomics_parsers/base_parser.py",
        "_create_parse_ir",
    ): "no caller",
    (
        "lobster/services/peptidomics/peptide_library_service.py",
        "_service_call_ir",
    ): "each value is repr'd",
    ("lobster/tools/user_interaction.py", "_decision_ir"): "each value is repr'd",
    (
        "packages/lobster-metadata/lobster/agents/metadata_assistant/metadata_assistant.py",
        "_combine_analysis_steps",
    ): "joins templates read above",
    (
        "packages/lobster-metadata/lobster/services/metadata/metadata_filtering_service.py",
        "_create_ir",
    ): "repr of criteria built from fixed keyword lists",
    (
        "packages/lobster-metadata/lobster/services/metadata/microbiome_filtering_service.py",
        "_create_16s_ir",
    ): "a dict and a flag",
    (
        "packages/lobster-metadata/lobster/services/metadata/microbiome_filtering_service.py",
        "_create_amplicon_validation_ir",
    ): "lists, numbers and a flag",
    (
        "packages/lobster-metadata/lobster/services/metadata/microbiome_filtering_service.py",
        "_create_host_ir",
    ): "lists and numbers",
    (
        "packages/lobster-metadata/lobster/services/metadata/microbiome_filtering_service.py",
        "_create_sample_type_ir",
    ): "lists and a flag",
    (
        "packages/lobster-ml/lobster/agents/machine_learning/machine_learning_expert.py",
        "_ml_split_ir",
    ): "each text value is repr'd",
    (
        "packages/lobster-ml/lobster/services/ml/ml_preparation_service.py",
        "select_ml_features",
    ): "each text value is repr'd",
    (
        "packages/lobster-ml/lobster/services/ml/multi_omics_integration_service.py",
        "_create_integration_ir",
    ): "numbers, flags and a list",
    (
        "packages/lobster-transcriptomics/lobster/services/quality/quality_service.py",
        "_create_quality_ir",
    ): "numbers",
}


def _sources():
    for base in [ROOT / "lobster", *sorted((ROOT / "packages").glob("*/lobster"))]:
        for path in sorted(base.rglob("*.py")):
            if "tests" not in path.parts:
                yield path


class _Template:
    """The text of one code_template, with each f-string value as a named hole."""

    def __init__(self, tree, call_line, func):
        self.holes = {}
        self.scope = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                self._bind(node, 0)
        if func is not None:
            assigns = (
                n for n in ast.walk(func) if isinstance(n, (ast.Assign, ast.AugAssign))
            )
            for node in sorted(assigns, key=lambda n: n.lineno):
                self._bind(node, node.lineno)
        self.call_line = call_line

    def _bind(self, node, line):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value = ast.BinOp(left=ast.Name(id=name), op=node.op, right=node.value)
            self.scope.setdefault(name, []).append((line, value))
            return
        for target in getattr(node, "targets", []):
            if isinstance(target, ast.Name):
                self.scope.setdefault(target.id, []).append((line, node.value))

    def text(self, node, before):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            parts = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(value.value)
                    continue
                hole = f"\x00{len(self.holes)}\x00"
                expression = value.value
                name = expression.id if isinstance(expression, ast.Name) else None
                self.holes[hole] = (name, value.conversion == ord("r"))
                parts.append(hole)
            return "".join(parts)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self.text(node.left, before)
            right = self.text(node.right, before)
            return None if left is None or right is None else left + right
        if isinstance(node, ast.Name):
            earlier = [b for b in self.scope.get(node.id, []) if b[0] < before]
            if earlier:
                line, value = earlier[-1]
                return self.text(value, line)
        return None


def _templates(path):
    tree = ast.parse(path.read_text())
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        if not (isinstance(node, ast.keyword) and node.arg == "code_template"):
            continue
        func = node
        while func in parents and not isinstance(
            func, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            func = parents[func]
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func = None
        line = node.value.lineno
        template = _Template(tree, line, func)
        yield func, line, template, template.text(node.value, line + 1)


def _not_text(func):
    """Arguments annotated as anything but text (numbers, flags, lists, literals):
    a tool's schema refuses text for them."""
    if func is None:
        return set()
    return {
        arg.arg
        for arg in [*func.args.args, *func.args.kwonlyargs]
        if arg.annotation is not None and ast.unparse(arg.annotation) not in TEXT
    }


def _render(source, names, hostile):
    for neutral in (1, "x", [], {}):
        values = {name: neutral for name in names}
        values.update(hostile)
        try:
            return _JINJA_ENV.from_string(source).render(**values)
        except Exception:  # noqa: BLE001 - try the next neutral value
            continue
    return None


def _becomes_code(baseline, rendered):
    if EVALUATED in rendered:
        return True
    try:
        ast.parse(baseline)
    except SyntaxError:
        # A fragment of a cell: the payload stays text only inside a repr.
        return PAYLOAD in rendered
    try:
        tree = ast.parse(rendered)
    except SyntaxError:
        return True
    return any(isinstance(n, ast.Name) and n.id == "__import__" for n in ast.walk(tree))


def _fill(text, template, chosen, value):
    """Put each f-string value into the text as Python does: ``value`` for the
    chosen name, a plain word for the others."""
    for hole, (name, as_repr) in template.holes.items():
        filled = value if name == chosen else "x"
        text = text.replace(hole, repr(filled) if as_repr else filled)
    return text


def _hits():
    hits, unread = {}, []
    for path in _sources():
        rel = path.relative_to(ROOT).as_posix()
        for func, line, template, text in _templates(path):
            where = func.name if func else "<module>"
            if text is None:
                unread.append((rel, where))
                continue
            skip = _not_text(func)
            holes = {name for name, _ in template.holes.values() if name}
            baseline_text = _fill(text, template, None, "x")
            names = meta.find_undeclared_variables(_JINJA_ENV.parse(baseline_text))
            for name in sorted((names | holes) - skip):
                if name in holes:
                    source = _fill(text, template, name, PAYLOAD)
                    if PAYLOAD.split("{{")[0] not in source:
                        continue
                    hostile = {}
                else:
                    source, hostile = baseline_text, {name: PAYLOAD}
                baseline = _render(baseline_text, names, {})
                try:
                    found = meta.find_undeclared_variables(_JINJA_ENV.parse(source))
                    rendered = _render(source, found, hostile)
                except Exception:  # noqa: BLE001 - the value broke the template
                    rendered = "{{"
                if baseline is None:
                    hits[(rel, where, name)] = f"{rel}:{line} does not render"
                elif rendered is None or _becomes_code(baseline, rendered):
                    hits[(rel, where, name)] = f"{rel}:{line} {name}"
    return hits, unread


@pytest.fixture(scope="module")
def scan():
    return _hits()


def test_text_given_to_a_tool_never_becomes_code_in_its_cell(scan):
    hits, _ = scan
    new = sorted(text for key, text in hits.items() if key not in KNOWN)
    assert new == [], "\n".join(new)


def test_every_listed_site_is_still_there(scan):
    hits, _ = scan
    assert sorted(set(KNOWN) - set(hits)) == []


def test_a_template_it_cannot_read_has_been_checked_by_hand(scan):
    _, unread = scan
    assert sorted(set(unread)) == sorted(UNREAD)
