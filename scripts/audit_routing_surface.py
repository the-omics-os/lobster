#!/usr/bin/env python3
"""Audit which implemented capabilities the supervisor's routing can actually reach.

Why this exists
---------------
Lobster's graph is topologically a single node: routing is not a graph decision but
the supervisor LLM choosing among ~17 tool schemas. The only information it has about
a specialist is two strings from that agent's ``AgentRegistryConfig``:

  * ``description``               -> rendered into <Agent Directory>
  * ``handoff_tool_description``  -> rendered into <Task Routing Guide> and the
                                     handoff tool schema itself

Those two strings are that agent's ROUTING SURFACE. A capability that is implemented
but absent from its owner's surface is unreachable by routing: the supervisor has no
evidence any specialist can do it, so it falls back to ``execute_custom_code``.

This script classifies every ``@tool`` in the repo as COVERED, ORPHANED, or
MIS-ANCHORED (its vocabulary appears only in a *different* agent's surface, so the
task gets routed to a competing claimant). MIS-ANCHORED is the dangerous class: it is
why descriptions must be edited as a competing set rather than one at a time.

Design decisions that matter
----------------------------
* **Surfaces are derived from the live registry, never transcribed.** Hardcoding
  surface text produces an audit that silently goes stale the moment a description is
  edited. We read ``ComponentRegistry`` and reuse ``_supervisor_accessible_agents()``
  so "reachable by the supervisor" means exactly what it means in the prompt builder.
* **Tools are inventoried by AST**, so this needs no scientific stack, no network and
  no credentials, and runs in CI in seconds.
* **Attribution comes from ``factory_function`` module paths, not file stems.**
  ``de_analysis_expert.py`` exists in BOTH lobster-proteomics and
  lobster-transcriptomics; keying on the stem collapses them and misattributes every
  proteomics DE tool. Helper modules (e.g. ``shared_tools.py``) are attributed by
  which agent module imports them. Anything unattributable is reported loudly, never
  silently guessed.
* **The ancestor map is committed data with a comment per entry.** Naive keyword
  matching is known to be wrong: "make a UMAP" routes correctly even though "UMAP"
  appears in no surface, because "visualizations" subsumes it. The map is deliberately
  GENEROUS — over-crediting coverage understates the problem, which is the safe
  direction for a regression gate.

Usage
-----
    python scripts/audit_routing_surface.py                 # full report
    python scripts/audit_routing_surface.py --dump-surface   # exact surfaces, diffable
    python scripts/audit_routing_surface.py --check          # CI gate vs baseline
    python scripts/audit_routing_surface.py --json out.json  # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).resolve().parent / "routing_surface_baseline.json"

COVERED, ORPHANED, MIS_ANCHORED = "COVERED", "ORPHANED", "MIS-ANCHORED"

# ---------------------------------------------------------------------------
# Ancestor map: superordinate terms that legitimately subsume a capability.
#
# Read as: if any listed term appears in an agent's routing surface, that surface
# covers the tool-name keywords on the left. One entry per line with a rationale;
# this is auditable data, not a heuristic. Deliberately generous (see module
# docstring) — a false COVERED understates the problem, a false ORPHANED would
# create busywork and erode trust in the gate.
# ---------------------------------------------------------------------------
ANCESTORS: dict[str, tuple[str, ...]] = {
    # dimensionality reduction is routinely requested by name but described generically
    "pca": ("dimensionality reduction", "clustering", "preprocessing", "qc"),
    "umap": ("dimensionality reduction", "visualization", "clustering"),
    "tsne": ("dimensionality reduction", "visualization", "clustering"),
    # export/report verbs are near-universally implied by owning the analysis
    "export": ("export", "report", "results", "save", "write"),
    "report": ("report", "export", "results", "summary"),
    "summary": ("summary", "report", "overview", "results"),
    # status/availability probes are plumbing, not user-facing capabilities
    "check": ("status", "check", "validate", "availability", "qc"),
    "status": ("status", "check"),
    "validate": ("validate", "validation", "qc", "quality"),
    # differential expression vocabulary
    "de": ("differential expression", "de", "pseudobulk"),
    "formula": ("differential expression", "de", "design", "formula"),
    # survival analysis
    "survival": ("survival", "cox", "kaplan", "time-to-event", "prognosis"),
    "cox": ("survival", "cox", "proportional hazards", "prognosis"),
    "kaplan": ("survival", "kaplan", "time-to-event"),
    "meier": ("survival", "kaplan", "time-to-event"),
    "risk": ("risk", "survival", "prognosis", "stratification", "classification"),
    "threshold": ("threshold", "cutoff", "classification", "stratification"),
    # metadata harmonization
    "metadata": ("metadata", "sample metadata", "harmonization", "annotation"),
    "standardize": ("standardize", "harmonization", "ontology", "normalization"),
    "enrich": ("enrich", "annotation", "harmonization", "enrichment"),
    "map": ("mapping", "map", "id mapping", "harmonization", "integration"),
    "merge": ("merge", "join", "integration", "harmonization", "metadata"),
    "disease": ("disease", "ontology", "annotation", "clinical", "mondo"),
    # peptide / protein chemistry
    "peptide": ("peptide", "proteomics", "protein"),
    "properties": ("properties", "physicochemical", "cheminformatics", "descriptors"),
    "activity": ("activity", "bioactivity", "prediction", "screening"),
    "variants": ("variant", "variants", "mutagenesis", "sar", "sequence"),
    "embedding": ("embedding", "representation", "language model", "featurization"),
    "antibody": ("antibody", "affinity", "specificity", "validation", "proteomics"),
    "specificity": ("specificity", "validation", "antibody"),
}

# Tokens carrying no routing signal — dropped before matching. Generic verbs and
# nouns must be here: without them a tool matches any surface sharing a common word
# ("perform", "comprehensive"), which manufactures false MIS-ANCHORED findings and
# would send reviewers chasing phantom conflicts. Matching tolerates simple inflection, so
# short generic tokens are especially dangerous.
_STOPWORD_TEXT = (
    "a an and are as at be by for from get in into is it of on or the to with "
    "all any both each such via per its their this that these those not non "
    "data set sets run runs running create creates created make makes made "
    "perform performs performed performing apply applies applied compute computes "
    "calculate calculates calculated generate generates generated build builds "
    "return returns returning extract extracts prepare prepares provide provides "
    "result results value values list lists new use uses using "
    "tool tools step steps analysis analyze analyses analysis_summary "
    "comprehensive optional required standard general based"
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split(" "))


# Core modules whose tools are SHARED INFRASTRUCTURE, handed to many agents rather
# than owned by any one of them: workspace I/O, filesystem access, the code sandbox,
# generic plotting, ID lookups, todos, HITL. They are deliberately out of scope —
# they are not domain capabilities, so "is this agent's description advertising it?"
# is the wrong question. Excluded at scan time AND during transitive attribution,
# otherwise importing agents would each inherit ~43 phantom capabilities.
SHARED_INFRASTRUCTURE_PREFIXES = (
    "lobster.tools.",
    "lobster.services.",
    "lobster.agents.graph",
    "lobster.agents.supervisor",
)


def is_shared_infrastructure(module: str) -> bool:
    return module.startswith(SHARED_INFRASTRUCTURE_PREFIXES)


class AuditError(RuntimeError):
    """Raised when the audit cannot proceed honestly (e.g. unattributable tool)."""


@dataclass
class ToolRecord:
    """One implemented capability.

    A tool is counted ONCE even when several agents expose it: a helper module like
    ``machine_learning/shared_tools.py`` is imported by both ``feature_selection_expert``
    and ``survival_analysis_expert``, but that is one capability, not two. ``owners``
    therefore holds every agent exposing it, and ``roots`` every supervisor-accessible
    ancestor whose surface could legitimately advertise it.
    """

    name: str
    module: str
    owners: list[str]
    roots: list[str]
    doc_summary: str = ""
    status: str = ORPHANED
    claimed_by: list[str] = field(default_factory=list)
    # True when a CHILD agent's own surface does advertise this capability, but that
    # child is not supervisor-accessible so the text is never rendered into the
    # routing guide. Distinguishes two different fixes: lift existing vocabulary to
    # the parent agent vs. authoring new vocabulary.
    described_in_invisible_child: bool = False

    @property
    def owner_label(self) -> str:
        return ", ".join(sorted(self.owners))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "owners": sorted(self.owners),
            "roots": sorted(self.roots),
            "status": self.status,
            "claimed_by": sorted(self.claimed_by),
            "described_in_invisible_child": self.described_in_invisible_child,
        }


# ---------------------------------------------------------------------------
# 1. Registry: surfaces and the parent chain
# ---------------------------------------------------------------------------


def load_registry() -> tuple[dict, list, dict[str, str]]:
    """Return (all_configs, accessible_configs, child->root map) from the registry.

    Reuses ``_supervisor_accessible_agents`` so "supervisor-accessible" is defined in
    exactly one place. A child with several parents (``metadata_assistant`` is a child
    of both ``data_expert_agent`` and ``research_agent``) maps to the first accessible
    parent by sorted name for determinism; it is judged as covered if ANY of its
    parents' surfaces claim it, which is the generous-by-design direction.
    """
    from lobster.agents.supervisor import _supervisor_accessible_agents
    from lobster.config.agent_registry import get_worker_agents

    configs = get_worker_agents()
    accessible = _supervisor_accessible_agents(list(configs.keys()))
    accessible_names = {cfg.name for cfg in accessible}

    child_parents: dict[str, list[str]] = {}
    for name, cfg in configs.items():
        for child in cfg.child_agents or []:
            child_parents.setdefault(child, []).append(name)

    child_to_root: dict[str, str] = {}
    for name in configs:
        if name in accessible_names:
            child_to_root[name] = name
            continue
        parents = sorted(
            p for p in child_parents.get(name, []) if p in accessible_names
        )
        # Unreachable agents (no accessible parent) map to themselves; they surface in
        # the report as their own root so the condition is visible rather than hidden.
        child_to_root[name] = parents[0] if parents else name

    return configs, accessible, child_to_root


def surface_text(cfg) -> str:
    """The exact strings the supervisor sees for one agent."""
    return " ".join(filter(None, (cfg.description, cfg.handoff_tool_description)))


def all_parents(configs: dict, agent: str) -> list[str]:
    return sorted(n for n, c in configs.items() if agent in (c.child_agents or []))


# ---------------------------------------------------------------------------
# 2. Tool inventory (AST only)
# ---------------------------------------------------------------------------


def _is_tool_decorator(node: ast.AST) -> bool:
    """True for @tool and @tool(...) — the LangChain tool decorator."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id == "tool"
    if isinstance(target, ast.Attribute):
        return target.attr == "tool"
    return False


def _doc_summary(node: ast.AST) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.strip().split("\n", 1)[0]


def scan_tools(path: Path) -> list[tuple[str, str]]:
    """Return [(tool_name, doc_summary)] for @tool-decorated defs in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise AuditError(f"cannot parse {path}: {exc}") from exc

    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(_is_tool_decorator(d) for d in node.decorator_list):
            found.append((node.name, _doc_summary(node)))
    return found


def module_path_for(path: Path) -> str | None:
    """Convert a file path to its dotted ``lobster.*`` module path.

    PEP 420 namespace packages mean the import root is the directory containing
    ``lobster/``, which differs between core and each agent package. Note the repo
    checkout is itself named ``lobster``, so we anchor on the LAST ``lobster``
    component that begins an importable path — using the first would yield
    ``lobster.packages.lobster-transcriptomics.lobster.agents...``, which is not a
    module path (and contains a hyphen, so it never could be).
    """
    parts = path.with_suffix("").parts
    indices = [i for i, part in enumerate(parts) if part == "lobster"]
    if not indices:
        return None
    return ".".join(parts[indices[-1] :])


def _imported_modules(path: Path) -> set[str]:
    """Dotted module names imported by one file (absolute imports only)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def build_inventory(configs: dict, child_to_root: dict[str, str]) -> list[ToolRecord]:
    """Inventory every @tool and attribute it to an owning agent.

    Attribution order:
      1. the file IS an agent's factory module  -> that agent (authoritative)
      2. an agent module in the tree imports it -> the importing agent(s)
      3. otherwise                              -> AuditError (never guess)
    """
    module_to_agent: dict[str, str] = {}
    for name, cfg in configs.items():
        factory = cfg.factory_function or ""
        module = factory.rsplit(".", 1)[0] if "." in factory else factory
        if module:
            module_to_agent[module] = name

    search_roots = [REPO_ROOT / "lobster", *(REPO_ROOT / "packages").glob("*/lobster")]
    tool_files: dict[Path, list[tuple[str, str]]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(p in {"tests", "build", "__pycache__"} for p in path.parts):
                continue
            module = module_path_for(path)
            if module and is_shared_infrastructure(module):
                continue
            tools = scan_tools(path)
            if tools:
                tool_files[path] = tools

    # Rule 2, resolved transitively: an agent module imports a helper, which may
    # itself import another helper. Iterate to a fixed point so a two-hop chain
    # (agent -> shared_tools -> peptide_tools) still attributes correctly.
    imports_by_module = {
        module_path_for(p): _imported_modules(p)
        for p in tool_files
        if module_path_for(p) is not None
    }
    # Helper modules do not always contain tools themselves, so also index any
    # importable module under an agent tree.
    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            module = module_path_for(path)
            if module and module not in imports_by_module:
                imports_by_module[module] = _imported_modules(path)

    helper_importers: dict[str, set[str]] = {}
    for _ in range(4):  # depth cap; chains deeper than this are a design smell
        changed = False
        for module, imported in imports_by_module.items():
            agent = module_to_agent.get(module)
            owners = {agent} if agent else set(helper_importers.get(module, ()))
            if not owners:
                continue
            for target in imported:
                if is_shared_infrastructure(target):
                    continue  # never attribute shared tooling to a domain agent
                before = set(helper_importers.get(target, ()))
                if not before >= owners:
                    helper_importers.setdefault(target, set()).update(owners)
                    changed = True
        if not changed:
            break

    records: list[ToolRecord] = []
    unattributed: list[str] = []
    skipped_uninstalled: set[str] = set()

    for path, tools in tool_files.items():
        module = module_path_for(path)
        if module is None:
            continue

        owners: list[str] = []
        if module in module_to_agent:
            owners = [module_to_agent[module]]
        elif module in helper_importers:
            owners = sorted(helper_importers[module])
        else:
            # Uninstalled agent package: no entry point exists, so no routing surface
            # can claim these tools. Excluded, and surfaced in the report.
            package = _package_dir_for(path)
            if package and _is_uninstalled_package(module, module_to_agent):
                skipped_uninstalled.add(package)
                continue
            unattributed.append(module)
            continue

        roots = sorted({child_to_root.get(owner, owner) for owner in owners})
        for tool_name, doc in tools:
            records.append(
                ToolRecord(
                    name=tool_name,
                    module=module,
                    owners=sorted(owners),
                    roots=roots,
                    doc_summary=doc,
                )
            )

    if unattributed:
        raise AuditError(
            "unattributable @tool modules — the package is installed but no agent "
            "claims these tools. Register the agent, or have an agent module import "
            "the helper:\n  " + "\n  ".join(sorted(set(unattributed)))
        )

    if skipped_uninstalled:
        print(
            "NOTE: excluded tools from packages present on disk but NOT installed, "
            "so no routing surface can claim them:\n  "
            + "\n  ".join(sorted(skipped_uninstalled))
            + "\n  Install them for a complete audit "
            "(e.g. `uv pip install -e packages/<name>`).\n",
            file=sys.stderr,
        )

    return records


def _package_dir_for(path: Path) -> str | None:
    """The ``packages/<name>`` directory owning a file, if any."""
    parts = path.parts
    if "packages" in parts:
        index = parts.index("packages")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _is_uninstalled_package(module: str, module_to_agent: dict[str, str]) -> bool:
    """True when no registered agent module shares this module's domain package.

    ``lobster.agents.drug_discovery.*`` is uninstalled when no registered agent lives
    under ``lobster.agents.drug_discovery``.
    """
    prefix = module.rsplit(".", 1)[0]
    return not any(registered.startswith(prefix) for registered in module_to_agent)


# ---------------------------------------------------------------------------
# 3. Classification
# ---------------------------------------------------------------------------


def keywords_for(record: ToolRecord) -> set[str]:
    """Content tokens for a tool: its name plus its docstring summary."""
    raw = re.split(r"[^a-z0-9]+", f"{record.name} {record.doc_summary}".lower())
    return {t for t in raw if len(t) > 2 and t not in STOPWORDS}


def surface_claims(keywords: set[str], surface: str) -> bool:
    """True when a surface plausibly advertises these keywords.

    Two ways to match, both auditable:
      * lexical   — a keyword appears in the surface as a whole word
      * ancestral — a superordinate term from ANCESTORS appears in the surface

    Word-boundary matching matters: plain substring matching lets a short token hit
    inside an unrelated word ("de" in "model", "map" in "mapping" is fine but "pca"
    in "capcase" is not), which inflates coverage and manufactures false competitor
    claims. Multi-word ancestor terms are matched as phrases.
    """
    low = surface.lower()
    tokens = set(re.split(r"[^a-z0-9]+", low))

    def has_term(term: str) -> bool:
        if " " in term or "-" in term:
            return term in low
        if term in tokens:
            return True
        # Tolerate simple inflection: a surface saying "visualizations" or "clustering"
        # does advertise "visualization"/"cluster". Both sides require a reasonably
        # long stem, so this cannot reintroduce spurious short-token matches.
        if len(term) < 5:
            return False
        for token in tokens:
            if len(token) < 5:
                continue
            if token.startswith(term) or term.startswith(token):
                return True
        return False

    for keyword in keywords:
        if has_term(keyword):
            return True
        for anchor, ancestors in ANCESTORS.items():
            # The anchor identifies which ancestor family this keyword belongs to;
            # equality or a clean prefix/suffix relation, not any shared substring.
            related = (
                keyword == anchor
                or keyword.startswith(anchor)
                or keyword.endswith(anchor)
            )
            if related and any(has_term(term) for term in ancestors):
                return True
    return False


def classify(
    records: list[ToolRecord], configs: dict, child_to_root: dict[str, str]
) -> list[ToolRecord]:
    """Label every tool COVERED / ORPHANED / MIS-ANCHORED."""
    surfaces = {name: surface_text(cfg) for name, cfg in configs.items()}
    roots = sorted(set(child_to_root.values()))

    for record in records:
        keywords = keywords_for(record)

        # A capability is legitimately reachable through any of its roots, or through
        # any parent of any agent exposing it. Generous by design (see module
        # docstring): crediting coverage understates the problem, which is the safe
        # direction for a regression gate.
        own_surfaces = set(record.roots)
        for owner in record.owners:
            own_surfaces.update(all_parents(configs, owner))

        if any(surface_claims(keywords, surfaces.get(a, "")) for a in own_surfaces):
            record.status = COVERED
            continue

        # The capability may be well described by a child agent that the supervisor
        # cannot see (no handoff tool / supervisor_accessible=False). The vocabulary
        # exists but is never rendered into the routing guide — a lift, not a rewrite.
        record.described_in_invisible_child = any(
            owner not in own_surfaces
            and surface_claims(keywords, surfaces.get(owner, ""))
            for owner in record.owners
        )

        competitors = [
            root
            for root in roots
            if root not in own_surfaces
            and surface_claims(keywords, surfaces.get(root, ""))
        ]
        if competitors:
            record.status = MIS_ANCHORED
            record.claimed_by = competitors
        else:
            record.status = ORPHANED

    return records


# ---------------------------------------------------------------------------
# 4. Reporting
# ---------------------------------------------------------------------------


def per_agent_counts(records: list[ToolRecord]) -> dict[str, dict[str, int]]:
    """Counts keyed by supervisor-accessible root.

    A capability exposed under two roots is counted under each, so per-agent totals
    can sum above the deduplicated inventory. That is intentional: each root is
    independently responsible for advertising what it exposes.
    """
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        for root in record.roots:
            bucket = counts.setdefault(
                root, {"total": 0, "unreachable": 0, "mis_anchored": 0}
            )
            bucket["total"] += 1
            if record.status != COVERED:
                bucket["unreachable"] += 1
            if record.status == MIS_ANCHORED:
                bucket["mis_anchored"] += 1
    return counts


def print_report(records: list[ToolRecord], verbose: bool) -> None:
    unreachable = [r for r in records if r.status != COVERED]
    mis = [r for r in records if r.status == MIS_ANCHORED]

    print("=" * 78)
    print("ROUTING SURFACE AUDIT")
    print("=" * 78)
    print(f"@tool functions inventoried : {len(records)}")
    print(
        f"UNREACHABLE by routing      : {len(unreachable)}"
        f"  ({len(unreachable) / max(len(records), 1):.1%})"
    )
    print(f"  ...of which MIS-ANCHORED  : {len(mis)}  (claimed by another agent)")
    liftable = [r for r in unreachable if r.described_in_invisible_child]
    print(
        f"  ...describable by a child : {len(liftable)}  "
        "(vocabulary exists but the child is invisible to the supervisor)"
    )
    print()

    counts = per_agent_counts(records)
    for agent in sorted(counts, key=lambda a: -counts[a]["unreachable"]):
        c = counts[agent]
        if not c["unreachable"] and not verbose:
            continue
        print(f"### {agent}  --  {c['unreachable']}/{c['total']} unreachable")
        for record in sorted(unreachable, key=lambda r: (r.owner_label, r.name)):
            if agent not in record.roots:
                continue
            suffix = (
                f"  [MIS-ANCHORED -> {','.join(sorted(record.claimed_by))}]"
                if record.claimed_by
                else "  [ORPHANED]"
            )
            if record.described_in_invisible_child:
                suffix += "  (described by an invisible child — lift to parent)"
            print(f"    {record.owner_label:30} {record.name:36}{suffix}")
        print()


def dump_surface(configs: dict, child_to_root: dict[str, str]) -> None:
    """Print every agent's exact routing surface, stably ordered for diffing.

    This is the mechanism that makes routing updates safe: strengthening one description can
    steal traffic from another, so descriptions must be reviewed as a competing set
    with a before/after diff of this output.
    """
    print("# Routing surfaces (description + handoff_tool_description)")
    print("# Stable order for diffing. Regenerate with --dump-surface.")
    for name in sorted(configs):
        cfg = configs[name]
        root = child_to_root.get(name, name)
        marker = "ACCESSIBLE" if root == name else f"child-of:{root}"
        print(f"\n## {name}  [{marker}]")
        print(f"description             : {cfg.description or '(none)'}")
        print(f"handoff_tool_description: {cfg.handoff_tool_description or '(none)'}")


# ---------------------------------------------------------------------------
# 5. CI gate
# ---------------------------------------------------------------------------


def check_against_baseline(records: list[ToolRecord], baseline_path: Path) -> int:
    """Fail when unreachable capabilities increase, in total or for any agent.

    Per-agent deltas matter: a fix in one agent can mask a regression in another,
    which is exactly the failure mode the competing-set description work risks.
    """
    counts = per_agent_counts(records)
    total = sum(c["unreachable"] for c in counts.values())

    if not baseline_path.exists():
        print(f"No baseline at {baseline_path}. Write one with --write-baseline.")
        return 1

    baseline = json.loads(baseline_path.read_text())
    base_total = baseline.get("total_unreachable", 0)
    base_agents = baseline.get("per_agent", {})

    problems = []
    if total > base_total:
        problems.append(f"total unreachable rose {base_total} -> {total}")

    for agent, c in sorted(counts.items()):
        before = base_agents.get(agent, {}).get("unreachable", 0)
        if c["unreachable"] > before:
            problems.append(f"{agent}: unreachable rose {before} -> {c['unreachable']}")

    print(
        f"baseline total unreachable: {base_total}  (from {baseline.get('commit', '?')})"
    )
    print(f"current  total unreachable: {total}")

    if problems:
        print("\nFAILED — routing coverage regressed:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nIf this is intentional, update the baseline with --write-baseline and "
            "justify it in the PR."
        )
        return 1

    if total < base_total:
        print(f"\nIMPROVED by {base_total - total}. Refresh with --write-baseline.")
    print("\nSUCCESS")
    return 0


def write_baseline(records: list[ToolRecord], baseline_path: Path, commit: str) -> None:
    counts = per_agent_counts(records)
    payload = {
        "commit": commit,
        "total_tools": len(records),
        "total_unreachable": sum(c["unreachable"] for c in counts.values()),
        "total_mis_anchored": sum(c["mis_anchored"] for c in counts.values()),
        "per_agent": {k: counts[k] for k in sorted(counts)},
    }
    baseline_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote baseline to {baseline_path}")
    print(f"  total_unreachable = {payload['total_unreachable']}")


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-surface", action="store_true", help="print surfaces only"
    )
    parser.add_argument("--check", action="store_true", help="CI gate against baseline")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument(
        "--commit", default="unknown", help="commit for --write-baseline"
    )
    parser.add_argument("--json", dest="json_out", help="write full inventory to JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="show all agents")
    args = parser.parse_args()

    try:
        configs, _accessible, child_to_root = load_registry()
        if args.dump_surface:
            dump_surface(configs, child_to_root)
            return 0

        records = classify(
            build_inventory(configs, child_to_root), configs, child_to_root
        )
    except AuditError as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([r.to_dict() for r in records], indent=2) + "\n"
        )

    if args.write_baseline:
        write_baseline(records, BASELINE_PATH, args.commit)
        return 0

    if args.check:
        return check_against_baseline(records, BASELINE_PATH)

    print_report(records, args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
