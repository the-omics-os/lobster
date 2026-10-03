"""The modality-management IR templates must be replayable and self-contained.

Exported notebooks may include management steps as well as analysis steps. Each generated
cell must use objects available in the notebook rather than depend on a service constructor
that is not defined there. These tests exercise the templates directly.
"""

import ast

import pytest

from lobster.services.data_management.modality_management_service import (
    ModalityManagementService,
)


@pytest.fixture
def service():
    """The IR builders are pure; they do not touch the data manager."""
    return ModalityManagementService.__new__(ModalityManagementService)


def _all_irs(service):
    """One IR per modality-management operation, with representative arguments."""
    return {
        "list_modalities": service._create_list_ir("pbmc*"),
        "list_modalities_nofilter": service._create_list_ir(None),
        "get_modality_info": service._create_get_info_ir("pbmc3k_pca"),
        "remove_modality": service._create_remove_ir("pbmc3k_embedded"),
        "validate_compatibility": service._create_validate_ir(["a_mod", "b_mod"]),
        "load_modality": service._create_load_ir(
            "pbmc3k", "/data/pbmc3k.h5ad", "h5ad", "single_cell"
        ),
    }


def test_all_templates_render_to_valid_python(service):
    """A template that does not parse cannot be replayed."""
    for name, ir in _all_irs(service).items():
        code = ir.render()
        try:
            ast.parse(code)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{name} rendered invalid Python: {exc}\n\n{code}")


def test_no_template_references_a_live_runtime(service):
    """Templates must not reference runtime objects unavailable in notebooks.

    An exported notebook holds a single ``adata`` and the Papermill parameters. A
    template that instantiates a service against ``data_manager`` raises NameError
    wherever it runs, which is every time.
    """
    forbidden = ("data_manager", "ModalityManagementService(")
    for name, ir in _all_irs(service).items():
        code = ir.render()
        for token in forbidden:
            assert token not in code, (
                f"{name} template references '{token}', which does not exist in an "
                f"exported notebook:\n\n{code}"
            )


def test_state_mutating_operations_are_replayable_but_not_exportable(service):
    """Curation and reproducibility answer different questions.

    State-changing operations belong in a faithful replay, while a curated notebook may omit
    them. Replayability is therefore independent of exportability.
    """
    irs = _all_irs(service)
    for name in ("remove_modality", "load_modality"):
        ir = irs[name]
        assert ir.replayable is True, f"{name} must be replayable"
        assert ir.exportable is False, f"{name} is not curated output"


def test_read_only_operations_remain_replayable_by_default(service):
    """Read-only operations must also be represented during replay.

    Introspection steps do not mutate state, but they remain part of the recorded workflow.
    """
    irs = _all_irs(service)
    for name in ("list_modalities", "get_modality_info", "validate_compatibility"):
        assert irs[name].replayable is True, f"{name} must stay replayable"


def test_remove_modality_does_not_delete_the_working_object(service):
    """`del adata` would be wrong and destructive.

    ``remove_modality`` drops a *named registry entry*. The notebook's single
    ``adata`` is not that entry, so deleting it would destroy the object every
    later cell depends on.
    """
    code = service._create_remove_ir("pbmc3k_embedded").render()
    assert "del adata" not in code
    assert "pbmc3k_embedded" in code, "the removed modality must be named in the record"


def test_remove_modality_records_the_mutation_visibly(service):
    """A replay that silently skips a mutation diverges without saying so."""
    code = service._create_remove_ir("pbmc3k_embedded").render()
    assert "provenance:" in code
    assert "removed" in code
