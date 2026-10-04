"""Tests for scripts/audit_routing_surface.py.

The audit is the instrument that makes routing-description changes safe: two thirds of unreachable
capabilities are already claimed by a *competing* agent, so strengthening one
description at a time steals traffic from another. If the audit's own logic is wrong,
that safety net is worthless -- so these tests pin the properties it is relied on for:

  * surfaces come from the LIVE registry (an edited config changes the verdict),
  * attribution never collapses same-named modules in different packages,
  * shared core tooling is never counted as a domain capability,
  * a capability is counted once no matter how many agents expose it,
  * the CI gate trips on a regression and reports per-agent deltas.

Deterministic, no model calls, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "audit_routing_surface.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_routing_surface", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["audit_routing_surface"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def live(audit):
    """The real registry-derived audit, computed once."""
    configs, accessible, child_to_root = audit.load_registry()
    records = audit.classify(
        audit.build_inventory(configs, child_to_root), configs, child_to_root
    )
    return configs, accessible, child_to_root, records


def test_script_exists():
    assert SCRIPT.is_file()


class TestModulePathResolution:
    """The repo checkout is itself named 'lobster', which is a real trap."""

    def test_anchors_on_last_lobster_component(self, audit):
        path = REPO_ROOT / "packages/lobster-transcriptomics/lobster/agents/x.py"
        assert audit.module_path_for(path) == "lobster.agents.x"

    def test_core_path_resolves(self, audit):
        assert (
            audit.module_path_for(REPO_ROOT / "lobster/agents/supervisor.py")
            == "lobster.agents.supervisor"
        )

    def test_hyphenated_package_dir_never_leaks_into_module_path(self, audit):
        """A module path containing '-' could never be imported."""
        path = REPO_ROOT / "packages/lobster-ml/lobster/agents/machine_learning/y.py"
        assert "-" not in (audit.module_path_for(path) or "")

    def test_non_lobster_path_returns_none(self, audit):
        assert (
            audit.module_path_for(
                Path(
                    "/tmp/other/thing.py"  # nosec B108 # Read-only path classification fixture; no file is created.
                )
            )
            is None
        )


class TestAttribution:
    def test_same_named_modules_in_different_packages_do_not_collapse(self, live):
        """de_analysis_expert.py exists in BOTH proteomics and transcriptomics.

        Keying attribution on the file stem would merge them and credit every
        proteomics DE tool to transcriptomics.
        """
        _, _, _, records = live
        de_modules = {
            r.module for r in records if r.module.endswith("de_analysis_expert")
        }
        assert "lobster.agents.transcriptomics.de_analysis_expert" in de_modules
        assert "lobster.agents.proteomics.de_analysis_expert" in de_modules

        owners_by_module = {
            m: {o for r in records if r.module == m for o in r.owners}
            for m in de_modules
        }
        transcriptomics = owners_by_module[
            "lobster.agents.transcriptomics.de_analysis_expert"
        ]
        proteomics = owners_by_module["lobster.agents.proteomics.de_analysis_expert"]
        assert transcriptomics != proteomics
        assert not (transcriptomics & proteomics)

    def test_shared_core_tooling_is_excluded(self, audit, live):
        """Workspace/filesystem/sandbox tools are handed to many agents.

        They are infrastructure, not domain capabilities, so asking whether an
        agent's description advertises them is the wrong question. Counting them
        previously inflated the inventory by 43 phantom capabilities.
        """
        _, _, _, records = live
        assert records, "inventory unexpectedly empty"
        for record in records:
            assert not audit.is_shared_infrastructure(record.module), record.module

    def test_known_shared_tool_names_absent(self, live):
        _, _, _, records = live
        names = {r.name for r in records}
        for infra in (
            "delete_from_workspace",
            "list_available_modalities",
            "shell_execute",
            "execute_custom_code",
        ):
            assert infra not in names, f"{infra} counted as a domain capability"

    def test_capability_counted_once_across_sibling_owners(self, live):
        """ml/shared_tools.py is imported by two sibling agents.

        That is one capability exposed twice, not two capabilities.
        """
        _, _, _, records = live
        keys = [(r.module, r.name) for r in records]
        assert len(keys) == len(set(keys)), "duplicate (module, tool) records"

        shared = [
            r
            for r in records
            if r.module == "lobster.agents.machine_learning.shared_tools"
        ]
        assert shared, "expected tools from ml shared_tools.py"
        assert any(
            len(r.owners) > 1 for r in shared
        ), "multi-owner case not represented"


class TestSurfaceDerivedFromRegistry:
    """The point of : surfaces are read live, never transcribed."""

    def test_editing_a_config_changes_the_verdict(self, audit, live):
        configs, _, child_to_root = live[:3]
        records = audit.build_inventory(configs, child_to_root)

        target = next(
            (
                r
                for r in audit.classify(list(records), configs, child_to_root)
                if r.status != audit.COVERED
            ),
            None,
        )
        assert target is not None, "no unreachable capability to test with"

        root = min(target.roots)
        keywords = " ".join(sorted(audit.keywords_for(target)))
        patched = dict(configs)
        patched[root] = replace(
            configs[root],
            handoff_tool_description=(configs[root].handoff_tool_description or "")
            + " Also handles: "
            + keywords,
        )

        after = audit.classify(
            audit.build_inventory(patched, child_to_root), patched, child_to_root
        )
        now = next(
            r for r in after if r.name == target.name and r.module == target.module
        )
        assert now.status == audit.COVERED, (
            "adding the tool's own vocabulary to its owner's surface must make it "
            "reachable -- otherwise surfaces are not really being read"
        )

    def test_surface_text_uses_both_registry_fields(self, audit):
        class Cfg:
            description = "DESC_TOKEN"
            handoff_tool_description = "HANDOFF_TOKEN"

        text = audit.surface_text(Cfg())
        assert "DESC_TOKEN" in text and "HANDOFF_TOKEN" in text


class TestMatching:
    def test_whole_word_matching_avoids_false_claims(self, audit):
        """Substring matching manufactured false MIS-ANCHORED findings."""
        assert not audit.surface_claims({"pca"}, "capcase handling")
        assert audit.surface_claims({"pca"}, "runs PCA and clustering")

    def test_generic_verbs_do_not_create_claims(self, audit):
        """'perform' once made the protein-structure agent claim Kaplan-Meier."""
        keywords = audit.keywords_for(
            audit.ToolRecord(
                name="run_kaplan_meier",
                module="m",
                owners=["a"],
                roots=["a"],
                doc_summary="Perform Kaplan-Meier survival analysis.",
            )
        )
        assert "perform" not in keywords
        assert "analysis" not in keywords
        assert "kaplan" in keywords

    def test_ancestor_map_credits_superordinate_terms(self, audit):
        """'make a UMAP' routes fine when a surface says 'visualization'."""
        assert audit.surface_claims({"umap"}, "creates publication visualizations")

    def test_ancestor_map_entries_are_documented_data(self, audit):
        assert isinstance(audit.ANCESTORS, dict) and audit.ANCESTORS
        for anchor, ancestors in audit.ANCESTORS.items():
            assert isinstance(anchor, str) and anchor
            assert isinstance(ancestors, tuple) and ancestors


class TestInvisibleChildDetection:
    def test_flags_capabilities_described_only_by_an_unreachable_child(self, live):
        """survival_analysis_expert's handoff text names Cox and hazard ratios, but
        supervisor_accessible=False means it is never rendered. The vocabulary exists
        and needs lifting to the parent -- a different fix from authoring new text."""
        _, _, _, records = live
        liftable = [r for r in records if r.described_in_invisible_child]
        assert liftable, "expected at least one liftable capability"
        assert any("survival" in r.module or "survival" in r.name for r in liftable)


class TestBaselineGate:
    def test_passes_on_head(self, audit, live):
        _, _, _, records = live
        assert audit.check_against_baseline(records, audit.BASELINE_PATH) == 0

    def test_trips_when_total_rises(self, audit, live, tmp_path):
        _, _, _, records = live
        baseline = tmp_path / "b.json"
        counts = audit.per_agent_counts(records)
        total = sum(c["unreachable"] for c in counts.values())
        baseline.write_text(
            json.dumps(
                {
                    "commit": "test",
                    "total_unreachable": total - 1,
                    "per_agent": {k: dict(v) for k, v in counts.items()},
                }
            )
        )
        assert audit.check_against_baseline(records, baseline) == 1

    def test_trips_on_per_agent_regression_even_when_total_holds(
        self, audit, live, tmp_path
    ):
        """A fix in one agent must not mask a regression in another -- exactly the
        failure mode the competing-set description work risks."""
        _, _, _, records = live
        counts = audit.per_agent_counts(records)
        total = sum(c["unreachable"] for c in counts.values())
        agents = sorted(counts)
        shifted = {k: dict(v) for k, v in counts.items()}
        # Same total, redistributed: one agent better, another worse.
        shifted[agents[0]]["unreachable"] += 1
        shifted[agents[-1]]["unreachable"] = max(
            0, shifted[agents[-1]]["unreachable"] - 1
        )

        baseline = tmp_path / "b.json"
        baseline.write_text(
            json.dumps(
                {"commit": "t", "total_unreachable": total, "per_agent": shifted}
            )
        )
        assert audit.check_against_baseline(records, baseline) == 1

    def test_missing_baseline_fails_rather_than_passing_silently(
        self, audit, live, tmp_path
    ):
        _, _, _, records = live
        assert audit.check_against_baseline(records, tmp_path / "nope.json") == 1

    def test_committed_baseline_records_its_commit(self, audit):
        payload = json.loads(audit.BASELINE_PATH.read_text())
        assert payload.get("commit") not in (None, "", "unknown")
        assert payload["total_unreachable"] >= 0
        assert payload["per_agent"]


class TestReportedShape:
    def test_reproduces_expected_magnitude_on_head(self, live):
        """Guards against a silent collapse of the inventory (e.g. an over-broad
        exclusion) that would make the gate vacuously green."""
        _, _, _, records = live
        assert 150 <= len(records) <= 230, len(records)
        unreachable = [r for r in records if r.status != "COVERED"]
        assert 5 <= len(unreachable) <= 60, len(unreachable)

    def test_every_record_has_a_reachable_root(self, live):
        _, _, _, records = live
        for record in records:
            assert record.owners and record.roots

    def test_statuses_are_from_the_documented_set(self, audit, live):
        _, _, _, records = live
        allowed = {audit.COVERED, audit.ORPHANED, audit.MIS_ANCHORED}
        assert {r.status for r in records} <= allowed

    def test_mis_anchored_records_name_their_competitor(self, audit, live):
        _, _, _, records = live
        for record in records:
            if record.status == audit.MIS_ANCHORED:
                assert record.claimed_by
                assert not set(record.claimed_by) & set(record.roots)
