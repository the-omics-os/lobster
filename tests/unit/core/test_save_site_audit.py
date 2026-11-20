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
ML = "packages/lobster-ml/lobster/agents/machine_learning/machine_learning_expert.py"
ML_SPLITS = (
    "assert_save_target() of the three split names (f-strings) before the "
    "input is annotated or any store_modality"
)
DE_BRANCH = (
    "de_modality_name from _unloaded_result_name() in the branch that will run "
    "(two branches)"
)
PROTEOMICS = "packages/lobster-proteomics/lobster/agents/proteomics/shared_tools.py"
PROTEOMICS_IMPORT = (
    "name = modality_name or unloaded_name(...); assert_save_target(name) under "
    "`if save_result` before the store, and the save under the same condition "
    "after it"
)
# (file relative to the checkout, function, saved-name expression) -> why it is
# safe without the proof. A new save in the same function is audited.
ALLOWED = {
    (DM, "store_modality", "name"): (
        "legacy auto_save: assert_save_target(name) under `if auto_save` before "
        "the store, and the save under the same condition after it"
    ),
    (DM, "save_workspace", "modality_name"): (
        "writes every modality into the named workspace folder, not into data/; "
        "save_modality still refuses a loaded file there"
    ),
    (DM, "save_processed_data", "modality_name"): (
        "the file name is generated (source, dataset, step, time stamp), and "
        "save_modality refuses a loaded file before the write; nothing is "
        "stored first"
    ),
    (DM, "auto_save_state", "modality_name"): (
        "may update a modality's own *_autosave.h5ad snapshot; any other "
        "loaded target is skipped with a warning, never raised"
    ),
    (DM, "run_notebook", "input_modality"): (
        "writes a temporary input file for the notebook run"
    ),
    (ML, "create_ml_splits", "train_modality"): ML_SPLITS,
    (ML, "create_ml_splits", "test_modality"): ML_SPLITS,
    (ML, "create_ml_splits", "val_modality"): ML_SPLITS,
    **{
        (PROTEOMICS, function, "name"): PROTEOMICS_IMPORT
        for function in (
            "import_affinity_data",
            "_import_generic_matrix",
            "import_proteomics_data",
            "import_ptm_sites",
        )
    },
    (f"{TX}/de_analysis_expert.py", "_two_group_de", "de_modality_name"): DE_BRANCH,
    (f"{TX}/de_analysis_expert.py", "run_de_with_formula", "pseudobulk_modality"): (
        "saves an existing modality to a new PATH built with unloaded_name()"
    ),
}


def _callname(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def _local_nodes(function):
    """The nodes of ``function``'s own scope (nested defs and lambdas left out)."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        ):
            stack.extend(ast.iter_child_nodes(node))


def _binds(node, name):
    """Whether ``node`` binds ``name``: an assignment of any kind, a for or with
    target, or :=."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
        targets = [node.target]
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        targets = [node.target]
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        targets = [i.optional_vars for i in node.items if i.optional_vars]
    else:
        return False
    return any(
        isinstance(n, ast.Name) and n.id == name for t in targets for n in ast.walk(t)
    )


def _sites_in(source):
    """(function, line, saved-name expression, proved) for every save_modality
    call in ``source``."""
    tree = ast.parse(source)
    parent = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}

    def segment(node):
        return ast.get_source_segment(source, node) or ""

    def block_of(stmt):  # (owning node, field) the statement sits in
        owner = parent.get(stmt)
        for field, value in ast.iter_fields(owner) if owner else ():
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

    def innermost_loop(node, function):
        while node is not function:
            node = parent[node]
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                return node

    def asserted(nodes, blocks, name, before):
        """Lines of assert_save_target(name) that dominate, before ``before``."""
        return [
            n.lineno
            for n in nodes
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and _callname(n.value) == "assert_save_target"
            and n.value.args
            and segment(n.value.args[0]) == name
            and n.lineno < before
            and block_of(n) in blocks
        ]

    def safe_value(value, nodes, blocks, line):
        """unloaded_name()/_unloaded_result_name(), or ``a or <that>`` where each
        other part is a name checked by assert_save_target() first."""
        if isinstance(value, ast.Call):
            return _callname(value) in SAFE_CALLS
        if not isinstance(value, ast.BoolOp):
            return False
        calls = [v for v in value.values if isinstance(v, ast.Call)]
        others = [v for v in value.values if not isinstance(v, ast.Call)]
        return (
            bool(calls)
            and all(_callname(c) in SAFE_CALLS for c in calls)
            and all(
                isinstance(o, ast.Name) and asserted(nodes, blocks, o.id, line)
                for o in others
            )
        )

    def safe_binding(node, name, nodes, function):
        return (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and safe_value(
                node.value, nodes, enclosing_blocks(node, function), node.lineno
            )
        )

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
        nodes = list(_local_nodes(function))
        blocks = enclosing_blocks(call, function)
        bindings = [n for n in nodes if _binds(n, name)]
        # the statements on the name that dominate the save (same block or an
        # enclosing one, before it): the last one must be safe
        events = [
            (n.lineno, safe_binding(n, name, nodes, function))
            for n in bindings
            if n.lineno < call.lineno and block_of(n) in blocks
        ]
        events += [(line, True) for line in asserted(nodes, blocks, name, call.lineno)]
        proved = bool(events) and max(events)[1]
        if proved:
            # and no other binding of the name after it on any path (a later
            # branch, a for or with target, :=, later in a loop around the save),
            # unless that binding is safe itself
            last = max(events)[0]
            loop = innermost_loop(call, function)
            in_loop = set(ast.walk(loop)) if loop is not None else set()
            proved = not any(
                not safe_binding(n, name, nodes, function)
                for n in bindings
                if last < n.lineno < call.lineno
                or (n in in_loop and n.lineno >= call.lineno)
            )
        yield function.name, call.lineno, name, proved


def _save_sites():
    """(file, function, line, proved) for every save_modality call."""
    paths = [
        *sorted((ROOT / "packages").glob("*/lobster/**/*.py")),
        *sorted((ROOT / "lobster").glob("**/*.py")),
    ]
    for path in paths:
        for site in _sites_in(path.read_text()):
            yield (path.relative_to(ROOT).as_posix(), *site)


def _unproved(sites):
    """(file, function, line, saved name) of each site neither proved nor allowed."""
    return [
        (file, function, line, name)
        for file, function, line, name, proved in sites
        if not proved and (file, function, name) not in ALLOWED
    ]


SITES = list(_save_sites())


def test_the_audit_finds_the_save_sites():
    assert len(SITES) >= 40


def test_every_allowed_site_still_exists_and_says_why():
    found = {(file, function, name) for file, function, _, name, _ in SITES}
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
    "or_unloaded_given_checked": """
def tool(dm, m, given):
    dm.assert_save_target(given)
    name = given or dm.unloaded_name(f"{m}_marked")
    dm.save_modality(name, "x.h5ad")
""",
    "safe_rebound_in_a_branch_after": """
def tool(dm, m, other):
    name = dm.unloaded_name(f"{m}_a")
    if other:
        name = dm.unloaded_name(f"{m}_b")
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


# Each rebinds the name after its last safe event, or takes it unchecked.
UNPROVED_SHAPES = {
    "debris": DEBRIS_SHAPE,
    "aug_assign": """
def tool(dm, m):
    name = dm.unloaded_name(f"{m}_x")
    name += "_v2"
    dm.save_modality(name, "x.h5ad")
""",
    "plain_in_a_branch_after": """
def tool(dm, m, other):
    name = dm.unloaded_name(f"{m}_x")
    if other:
        name = f"{m}_y"
    dm.save_modality(name, "x.h5ad")
""",
    "for_target": """
def tool(dm, m, names):
    name = dm.unloaded_name(f"{m}_x")
    for name in names:
        pass
    dm.save_modality(name, "x.h5ad")
""",
    "with_target": """
def tool(dm, m, ctx):
    name = dm.unloaded_name(f"{m}_x")
    with ctx() as name:
        dm.save_modality(name, "x.h5ad")
""",
    "walrus": """
def tool(dm, m):
    name = dm.unloaded_name(f"{m}_x")
    if (name := f"{m}_y"):
        pass
    dm.save_modality(name, "x.h5ad")
""",
    "unpack": """
def tool(dm, m, pair):
    name = dm.unloaded_name(f"{m}_x")
    name, other = pair
    dm.save_modality(name, "x.h5ad")
""",
    "annotated": """
def tool(dm, m):
    name = dm.unloaded_name(f"{m}_x")
    name: str = f"{m}_y"
    dm.save_modality(name, "x.h5ad")
""",
    "or_unloaded_given_unchecked": """
def tool(dm, m, given):
    name = given or dm.unloaded_name(f"{m}_marked")
    dm.save_modality(name, "x.h5ad")
""",
    "rebound_later_in_the_loop": """
def tool(dm, m, items):
    name = dm.unloaded_name(f"{m}_x")
    for item in items:
        dm.save_modality(name, "x.h5ad")
        name = item
""",
}


@pytest.mark.parametrize("shape", list(UNPROVED_SHAPES))
def test_a_name_rebound_or_unchecked_is_not_proved(shape):
    assert [s[-1] for s in _sites_in(UNPROVED_SHAPES[shape])] == [False]


def test_a_new_save_in_an_allowed_function_is_audited():
    """ALLOWED names the saved expression: another save in the same function is
    held to the proof."""
    file, function, name = next(iter(ALLOWED))
    source = f"""
def {function}(self, other):
    self.save_modality({name}, "a.h5ad")
    self.save_modality(other, "b.h5ad")
"""
    sites = [(file, *site) for site in _sites_in(source)]
    assert _unproved(sites) == [(file, function, 4, "other")]


def test_a_plain_name_above_a_safe_branch_is_not_proved():
    """The mark_clusters_as_debris shape: the safe name only in one branch."""
    assert [site[-1] for site in _sites_in(DEBRIS_SHAPE)] == [False]


@pytest.mark.parametrize("shape", list(PROVED_SHAPES))
def test_the_safe_shapes_are_proved(shape):
    assert [site[-1] for site in _sites_in(PROVED_SHAPES[shape])] == [True]


def test_every_save_site_checks_its_target_first():
    assert _unproved(SITES) == [], (
        "these save a name that does not come from unloaded_name() and is not "
        "checked by assert_save_target() on every path to the save; add one, or "
        "list the site in ALLOWED with why"
    )
