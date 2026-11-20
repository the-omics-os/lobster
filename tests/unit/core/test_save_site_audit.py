"""Every save site checks its target before it stores the modality.

A save over a file the session loaded is refused (RecordedInputError). A tool
must not reach that refusal after it has already replaced a modality in memory:
before ``store_modality`` it either takes a result name no loaded file has
(``unloaded_name``) or checks a name it was given (``assert_save_target``).

The check is by variable: the name handed to ``save_modality`` must, in the same
function, come from ``unloaded_name()``/``_unloaded_result_name()`` (also
``x = a or unloaded_name(...)``) or be passed to ``assert_save_target()``, and
that must be the last such statement that dominates the save (same block or an
enclosing one, before it). A safe assignment in one ``if`` branch does not cover
a plain assignment above it. Text elsewhere in the function does not count.

New tools and core saves (packages/ and lobster/) are held to it here; a site
that is not proved is listed with its reason, and a new unlisted site fails.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SAFE_CALLS = {"unloaded_name", "_unloaded_result_name"}
DM = "lobster/core/runtime/data_manager.py"
TX = "packages/lobster-transcriptomics/lobster/agents/transcriptomics"
# (file relative to the checkout, function) -> why it is safe without the proof
ALLOWED = {
    (DM, "store_modality"): (
        "legacy auto_save: assert_save_target(name) under `if auto_save` before "
        "the store, and the save under the same condition after it"
    ),
    (DM, "save_workspace"): (
        "writes every modality into the named workspace folder, not into data/; "
        "save_modality still refuses a loaded file there"
    ),
    (DM, "save_processed_data"): (
        "the file name is generated (source, dataset, step, time stamp), and "
        "save_modality refuses a loaded file before the write; nothing is "
        "stored first"
    ),
    (DM, "auto_save_state"): (
        "may update a modality's own *_autosave.h5ad snapshot; any other "
        "loaded target is skipped with a warning, never raised"
    ),
    (DM, "run_notebook"): "writes a temporary input file for the notebook run",
    (
        "packages/lobster-ml/lobster/agents/machine_learning/machine_learning_expert.py",
        "create_ml_splits",
    ): (
        "assert_save_target() of the three split names (f-strings) before the "
        "input is annotated or any store_modality"
    ),
    (
        "packages/lobster-proteomics/lobster/agents/proteomics/shared_tools.py",
        "import_affinity_data",
    ): (
        "unloaded_name() on the derived name, assert_save_target() under "
        "`if save_result` for an explicit name"
    ),
    (f"{TX}/de_analysis_expert.py", "_two_group_de"): (
        "de_modality_name from _unloaded_result_name() in the branch that will "
        "run (two branches)"
    ),
    (f"{TX}/de_analysis_expert.py", "run_de_with_formula"): (
        "saves an existing modality to a new PATH built with unloaded_name()"
    ),
}


def _callname(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _is_safe_value(value):
    if isinstance(value, ast.Call):
        calls = [value]
    elif isinstance(value, ast.BoolOp):
        calls = [v for v in value.values if isinstance(v, ast.Call)]
    else:
        calls = []
    return any(_callname(c) in SAFE_CALLS for c in calls)


def _sites_in(source):
    """(function, line, proved) for every save_modality call in ``source``."""
    tree = ast.parse(source)
    parent = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}

    def segment(node):
        return ast.get_source_segment(source, node) or ""

    def block_of(stmt):  # (owning node, field) the statement sits in
        owner = parent[stmt]
        for field, value in ast.iter_fields(owner):
            if isinstance(value, list) and stmt in value:
                return owner, field

    def enclosing_blocks(node, function):
        blocks = set()
        while node is not function:
            if isinstance(node, ast.stmt):
                blocks.add(block_of(node))
            node = parent[node]
        return blocks

    def function_of(node):
        while node in parent:
            node = parent[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node

    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "save_modality"
            and call.args
        ):
            continue
        function = function_of(call)
        if function is None:  # module level: no tool saves there
            continue
        name = segment(call.args[0])
        blocks = enclosing_blocks(call, function)
        events = []  # (line, safe) of each dominating statement on the name
        for node in ast.walk(function):
            if getattr(node, "lineno", call.lineno) >= call.lineno:
                continue
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                if block_of(node) in blocks:
                    events.append((node.lineno, _is_safe_value(node.value)))
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and _callname(node.value) == "assert_save_target"
                and node.value.args
                and segment(node.value.args[0]) == name
                and block_of(node) in blocks
            ):
                events.append((node.lineno, True))
        proved = bool(events) and max(events)[1]
        yield function.name, call.lineno, proved


def _save_sites():
    """(file, function, line, proved) for every save_modality call."""
    paths = [
        *sorted((ROOT / "packages").glob("*/lobster/**/*.py")),
        *sorted((ROOT / "lobster").glob("**/*.py")),
    ]
    for path in paths:
        for function, line, proved in _sites_in(path.read_text()):
            yield path.relative_to(ROOT).as_posix(), function, line, proved


SITES = list(_save_sites())


def test_the_audit_finds_the_save_sites():
    assert len(SITES) >= 40


def test_every_allowed_site_still_exists_and_says_why():
    found = {(file, function) for file, function, _, _ in SITES}
    assert set(ALLOWED) <= found, set(ALLOWED) - found
    assert all(len(reason) > 20 for reason in ALLOWED.values())


DEBRIS_SHAPE = """
def tool(dm, m, remove):
    name = f"{m}_marked"
    if remove:
        name = dm.unloaded_name(f"{m}_removed")
    dm.save_modality(name, "x.h5ad")
"""
PROVED_SHAPES = {
    "unloaded": """
def tool(dm, m):
    name = dm.unloaded_name(f"{m}_marked")
    dm.save_modality(name, "x.h5ad")
""",
    "or_unloaded": """
def tool(dm, m, given):
    name = given or dm.unloaded_name(f"{m}_marked")
    dm.save_modality(name, "x.h5ad")
""",
    "asserted": """
def tool(dm, name, save):
    dm.assert_save_target(name)
    if save:
        dm.save_modality(name, "x.h5ad")
""",
    "both_reassigned": """
def tool(dm, m, remove):
    name = f"{m}_marked"
    if remove:
        name = f"{m}_removed"
    name = dm.unloaded_name(name)
    dm.save_modality(name, "x.h5ad")
""",
}


def test_a_plain_name_above_a_safe_branch_is_not_proved():
    """The mark_clusters_as_debris shape: the safe name only in one branch."""
    assert [proved for _, _, proved in _sites_in(DEBRIS_SHAPE)] == [False]


@pytest.mark.parametrize("shape", list(PROVED_SHAPES))
def test_the_safe_shapes_are_proved(shape):
    assert [proved for _, _, proved in _sites_in(PROVED_SHAPES[shape])] == [True]


@pytest.mark.parametrize(
    "file, function, line, proved",
    SITES,
    ids=[f"{file.split('/')[-1]}:{line}" for file, _, line, _ in SITES],
)
def test_a_save_site_checks_its_target_first(file, function, line, proved):
    if (file, function) in ALLOWED:
        return
    assert proved, (
        f"{file}:{line} ({function}) saves a name that does not come from "
        "unloaded_name() and is not checked by assert_save_target() on every "
        "path to the save; add one, or list it in ALLOWED with why"
    )
