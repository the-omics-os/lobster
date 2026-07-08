"""Regression tests for the ``repr`` Jinja2 filter on AnalysisStep rendering.

Bug: templates across the codebase emit Python literals via ``{{ x | repr }}``
(and the no-space ``{{ x|repr }}`` form). jinja2 has no built-in ``repr`` filter,
so the previous bare ``Template(self.code_template)`` raised
"No filter named 'repr'" at parse time — breaking notebook export for EVERY
genomics/GWAS workflow and the scRNA sub-clustering export.

Three layers:
  (b1) contract: every real shipped ``code_template`` that uses ``| repr`` renders
       without raising the missing-filter error and leaves no unrendered ``{{ }}``
       (including clustering_service's no-space form). It does NOT ``ast.parse`` each
       harvested template — control-flow templates with generic dummy params can
       legitimately produce non-parseable fragments; executable-Python validity is
       proven by the exporter round-trip in (c) and the direct-render tests in
       test_analysis_ir.py.
  (b2) both spacing variants (``| repr`` and ``|repr``) render identically.
  (c)  exporter round-trip: an IR using ``| repr`` survives NotebookExporter into a
       valid .ipynb whose code cells all ``ast.parse``.
"""

import ast
import re
from pathlib import Path

import pytest

from lobster.core.provenance.analysis_ir import AnalysisStep


def _extract_repr_templates(source_path: Path):
    """Harvest triple-quoted ``code_template`` blocks that contain ``|repr``.

    Regex-based (not import-based) so the test does not need to load heavy
    genomics/scanpy modules to reach the template strings.
    """
    text = source_path.read_text()
    templates = []
    # match code_template="""...""" or code_template='''...'''
    for m in re.finditer(
        r'code_template\s*=\s*(?P<q>"""|\'\'\')(?P<body>.*?)(?P=q)', text, re.DOTALL
    ):
        body = m.group("body")
        if re.search(r"\|\s*repr", body):
            templates.append(body)
    return templates


def _repo_files():
    """Locate the source files carrying |repr templates, relative to repo root."""
    root = Path(__file__).resolve().parents[3]  # tests/unit/core/ -> repo root
    candidates = [
        root / "packages/lobster-genomics/lobster/agents/genomics/genomics_expert.py",
        root / "packages/lobster-genomics/lobster/services/analysis/gwas_service.py",
        root / "lobster/services/analysis/clustering_service.py",
    ]
    return [c for c in candidates if c.exists()]


def _dummy_params_for(tpl: str) -> dict:
    """Supply a literal-shaped dummy value for every variable a template references.

    Uses jinja2.meta to discover undeclared variables so no template is rendered with
    a missing param (which would emit invalid Python). Strings are given an apostrophe
    so ``repr`` must do real escaping work; every value is JSON/literal-shaped, matching
    the ``| repr`` param contract.
    """
    from jinja2 import Environment, meta

    # Parse on an env that knows ``repr``; a vanilla Environment raises
    # "No filter named 'repr'" at parse time (the very bug under test), which would
    # break variable discovery before render() is ever reached.
    parse_env = Environment()
    parse_env.filters["repr"] = repr
    ast_nodes = parse_env.parse(tpl)
    names = meta.find_undeclared_variables(ast_nodes)
    # A string satisfies both ``{{ x | repr }}`` (-> quoted literal) and bare
    # ``{{ x }}`` interpolation, and is truthy for ``{% if x %}`` branches.
    return {name: f"val_{name}'q" for name in names}


def _render_template_string(tpl: str) -> str:
    """Render a raw template string through AnalysisStep with auto-discovered params."""
    ir = AnalysisStep(
        operation="contract.render",
        tool_name="contract",
        description="render real repr template",
        library="test",
        code_template=tpl,
        imports=[],
        parameters=_dummy_params_for(tpl),
        parameter_schema={},
    )
    return ir.render()


class TestReprFilterContract:
    """(b) Every real shipped ``|repr`` template must render without the filter error."""

    def test_source_files_present(self):
        assert _repo_files(), "no source files with |repr templates found"

    def test_all_real_repr_templates_render(self):
        """Rendering must not raise 'No filter named repr' for any shipped template.

        We assert the render SUCCEEDS and leaves no unrendered placeholders. We do not
        ast.parse the result: with generic dummy values, control-flow templates
        (``{% if %}``) can legitimately produce non-parseable fragments — full-code
        validity with real params is covered by the exporter round-trip below and the
        existing exporter tests. The regression here is specifically the missing filter.
        """
        files = _repo_files()
        total = 0
        for f in files:
            for tpl in _extract_repr_templates(f):
                total += 1
                try:
                    code = _render_template_string(tpl)
                except Exception as e:  # noqa: BLE001 - surface which template broke
                    pytest.fail(f"render failed for a |repr template in {f.name}: {e}")
                assert "No filter named" not in code
                assert (
                    "{{" not in code
                ), f"unrendered placeholder in template from {f.name}"
        assert total >= 1, "harvested zero |repr templates — extraction likely broke"

    def test_clustering_service_nospace_repr(self):
        """The no-space ``{{ x|repr }}`` form (clustering_service) must work."""
        ir = AnalysisStep(
            operation="scanpy.subcluster",
            tool_name="subcluster",
            description="t",
            library="scanpy",
            code_template="cluster_key = {{ cluster_key|repr }}\nalgo = {{ algorithm|repr }}",
            imports=[],
            parameters={"cluster_key": "leiden", "algorithm": "leiden"},
            parameter_schema={},
        )
        code = ir.render()
        assert "cluster_key = 'leiden'" in code
        assert "algo = 'leiden'" in code
        ast.parse(code)

    def test_repr_both_spacings_identical(self):
        def _render(tpl):
            return AnalysisStep(
                operation="t",
                tool_name="t",
                description="t",
                library="t",
                code_template=tpl,
                imports=[],
                parameters={"file_path": "it's a path.vcf.gz"},
                parameter_schema={},
            ).render()

        spaced = _render("x = {{ file_path | repr }}")
        nospace = _render("x = {{ file_path|repr }}")
        assert spaced == nospace
        # repr kept the apostrophe safely (picks double-quotes around an apostrophe string)
        assert "it's a path.vcf.gz" in spaced
        ast.parse(spaced)


class TestReprFilterExporterRoundTrip:
    """(c) An IR using ``|repr`` survives export into a valid notebook."""

    def test_repr_ir_exports_to_valid_notebook(self, tmp_path):
        nbformat = pytest.importorskip("nbformat")
        from unittest.mock import Mock, patch

        from lobster.core.notebooks.exporter import NotebookExporter
        from lobster.core.provenance.provenance import ProvenanceTracker
        from lobster.core.runtime.data_manager import DataManagerV2

        tracker = ProvenanceTracker()
        tracker.create_activity(
            activity_type="load_vcf",
            agent="genomics_expert",
            description="Load a VCF",
            parameters={
                "file_path": "/data/HG002.vcf.gz",
                "region": None,
                "samples": None,
            },
            outputs=[{"id": "vcf_modality", "type": "loaded"}],
            ir=AnalysisStep(
                operation="cyvcf2.VCF.load",
                tool_name="load_vcf",
                description="Load a VCF",
                library="cyvcf2",
                code_template=(
                    "adata = load(\n"
                    "    source={{ file_path | repr }},\n"
                    "    region={{ region | repr }},\n"
                    "    samples={{ samples | repr }},\n"
                    ")"
                ),
                imports=["from cyvcf2 import VCF"],
                parameters={
                    "file_path": "/data/HG002.vcf.gz",
                    "region": None,
                    "samples": None,
                },
                parameter_schema={},
                input_entities=[],
                output_entities=["adata"],
                exportable=True,
            ),
        )

        dm = Mock(spec=DataManagerV2)
        dm.modalities = {}
        dm.workspace_path = tmp_path
        exporter = NotebookExporter(tracker, dm)
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = exporter.export(
                name="repr_roundtrip", description="repr round-trip"
            )
        assert result is not None and Path(result).exists()

        nb = nbformat.read(str(result), as_version=4)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells, "no code cells in exported notebook"

        # the load_vcf cell must contain repr'd literals and every code cell must parse
        joined = "\n".join(c.source for c in code_cells)
        assert "source='/data/HG002.vcf.gz'" in joined
        assert "region=None" in joined
        bad = []
        for c in code_cells:
            try:
                ast.parse(c.source)
            except SyntaxError as e:
                bad.append(e)
        assert not bad, f"invalid Python in exported cells: {bad}"
