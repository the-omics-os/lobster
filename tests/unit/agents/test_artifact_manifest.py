"""Tests for the handoff artifact manifest.

The manifest should preserve default behavior when disabled, fail open, distinguish an
empty snapshot from an unavailable snapshot, and keep generated output bounded.

Deterministic; no model calls, no real DataManagerV2.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from lobster.agents.artifact_manifest import (
    MANIFEST_CLOSE,
    MANIFEST_ENV_VAR,
    MANIFEST_OPEN,
    MAX_MANIFEST_CHARS,
    ArtifactDiff,
    WorkspaceSnapshot,
    append_manifest,
    diff_snapshots,
    manifests_enabled,
    render_manifest,
    take_snapshot,
)


def fake_dm(modalities=None, files=None):
    """A DataManagerV2 stand-in exposing only the two methods we snapshot."""
    dm = MagicMock()
    dm.list_modality_records.return_value = [
        {"name": name, "n_obs": obs, "n_vars": var}
        for name, (obs, var) in (modalities or {}).items()
    ]
    dm.list_workspace_files.return_value = {
        category: [{"name": name, "path": f"{category}/{name}", "size": size}]
        for category, (name, size) in (files or {}).items()
    }
    return dm


@pytest.fixture(autouse=True)
def _manifest_off(monkeypatch):
    """Explicitly disable so no test depends on the default or leaks the flag."""
    monkeypatch.setenv(MANIFEST_ENV_VAR, "0")


class TestFlag:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv(MANIFEST_ENV_VAR, raising=False)
        assert manifests_enabled() is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on ", ""])
    def test_non_off_values_enable(self, monkeypatch, value):
        monkeypatch.setenv(MANIFEST_ENV_VAR, value)
        assert manifests_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " off "])
    def test_off_values_disable(self, monkeypatch, value):
        monkeypatch.setenv(MANIFEST_ENV_VAR, value)
        assert manifests_enabled() is False


class TestSnapshot:
    def test_captures_names_and_shapes(self):
        snap = take_snapshot(fake_dm(modalities={"pbmc": (100, 2000)}))
        assert not snap.failed
        assert snap.modalities["pbmc"].n_obs == 100
        assert snap.modalities["pbmc"].n_vars == 2000

    def test_none_data_manager_is_marked_failed(self):
        assert take_snapshot(None).failed is True

    def test_modality_read_failure_marks_failed(self):
        dm = MagicMock()
        dm.list_modality_records.side_effect = RuntimeError("backend down")
        dm.list_workspace_files.return_value = {}
        assert take_snapshot(dm).failed is True

    def test_file_read_failure_marks_failed(self):
        dm = MagicMock()
        dm.list_modality_records.return_value = []
        dm.list_workspace_files.side_effect = OSError("permission denied")
        assert take_snapshot(dm).failed is True

    @pytest.mark.parametrize(
        "error",
        [RuntimeError("x"), ValueError("y"), MemoryError(), AttributeError("z")],
    )
    def test_no_ordinary_exception_escapes(self, error):
        """Any ordinary failure degrades to a failed snapshot, never a raise."""
        dm = MagicMock()
        dm.list_modality_records.side_effect = error
        dm.list_workspace_files.return_value = {}
        assert take_snapshot(dm).failed is True

    def test_keyboard_interrupt_still_propagates(self):
        """Fail-open must not extend to swallowing interrupts.

        A user pressing Ctrl-C during a long delegation should stop the run, not be
        silently absorbed into a 'snapshot failed' note.
        """
        dm = MagicMock()
        dm.list_modality_records.side_effect = KeyboardInterrupt()
        dm.list_workspace_files.return_value = {}
        with pytest.raises(KeyboardInterrupt):
            take_snapshot(dm)

    def test_snapshot_does_not_materialize_data(self):
        """Only the shape-level APIs may be touched -- never get_modality()."""
        dm = fake_dm(modalities={"a": (1, 1)})
        take_snapshot(dm)
        dm.get_modality.assert_not_called()


class TestDiff:
    def test_detects_created_modality(self):
        diff = diff_snapshots(
            take_snapshot(fake_dm()),
            take_snapshot(fake_dm(modalities={"pbmc_qc": (500, 2000)})),
        )
        assert any("pbmc_qc" in c and "500 obs" in c for c in diff.created)
        assert not diff.modified

    def test_detects_shape_change_as_modification(self):
        diff = diff_snapshots(
            take_snapshot(fake_dm(modalities={"pbmc": (1000, 2000)})),
            take_snapshot(fake_dm(modalities={"pbmc": (800, 2000)})),
        )
        assert not diff.created
        assert any("pbmc" in m and "800" in m for m in diff.modified)

    def test_unchanged_state_yields_empty_diff(self):
        before = take_snapshot(fake_dm(modalities={"pbmc": (10, 20)}))
        after = take_snapshot(fake_dm(modalities={"pbmc": (10, 20)}))
        assert diff_snapshots(before, after).is_empty

    def test_detects_new_file(self):
        diff = diff_snapshots(
            take_snapshot(fake_dm()),
            take_snapshot(fake_dm(files={"exports": ("de_results.csv", 4096)})),
        )
        assert any("de_results.csv" in f for f in diff.files_written)

    def test_detects_rewritten_file_by_size(self):
        diff = diff_snapshots(
            take_snapshot(fake_dm(files={"exports": ("r.csv", 100)})),
            take_snapshot(fake_dm(files={"exports": ("r.csv", 900)})),
        )
        assert any("updated" in f for f in diff.files_written)


class TestRendering:
    def test_empty_is_rendered_explicitly_not_omitted(self):
        """'Measured, nothing written' must be a positive statement.

         keys its fabrication detector on this: an ABSENT manifest means
        'not measured', an EMPTY one means 'measured, found nothing'. If empty
        rendered as absent, a success claim with no artifacts would be unflaggable.
        """
        rendered = render_manifest(ArtifactDiff())
        assert MANIFEST_OPEN in rendered and MANIFEST_CLOSE in rendered
        assert "No new artifacts" in rendered

    def test_non_empty_names_the_artifact(self):
        rendered = render_manifest(
            ArtifactDiff(created=["pbmc_qc (500 obs x 2000 vars)"])
        )
        assert "pbmc_qc" in rendered
        assert "500 obs" in rendered

    def test_manifest_contains_no_directives(self):
        """The manifest must state facts, never instruct.

        A handoff return reaches the supervisor as a ToolMessage wrapped in
        <tool_data> markers, and the security policy forbids following instructions
        found there. A directive in the manifest would be both ignored by design and,
        if obeyed, an injection vector -- untrusted GEO/PubMed text uses the same
        channel. Policy belongs in the system prompt; see _build_agent_result_memory.
        """
        rendered = render_manifest(
            ArtifactDiff(
                created=["pbmc_qc (500 obs x 2000 vars)"],
                modified=["pbmc (10x20 -> 8x20)"],
                files_written=["exports/de.csv"],
            )
        ).lower()
        for directive in (
            "do not",
            "don't",
            "must",
            "should",
            "you ",
            "execute_custom_code",
            "reference them",
        ):
            assert (
                directive not in rendered
            ), f"manifest contains a directive: {directive!r}"

    def test_manifest_survives_tool_data_wrapping_intact(self):
        """Spotlighting must not corrupt the markers  parses."""
        from langchain_core.messages import ToolMessage

        from lobster.agents.context_management import _wrap_tool_results

        body = "QC complete.\n\n" + render_manifest(
            ArtifactDiff(created=["pbmc_qc (500 obs x 2000 vars)"])
        )
        wrapped = _wrap_tool_results([ToolMessage(content=body, tool_call_id="c1")])[
            0
        ].content
        assert wrapped.startswith("<tool_data>")
        assert MANIFEST_OPEN in wrapped and MANIFEST_CLOSE in wrapped
        assert "pbmc_qc" in wrapped

    def test_output_is_capped(self):
        diff = ArtifactDiff(
            created=[f"modality_{i} (100 obs x 200 vars)" for i in range(500)]
        )
        rendered = render_manifest(diff)
        assert len(rendered) <= MAX_MANIFEST_CHARS
        assert rendered.endswith(MANIFEST_CLOSE), "must stay parseable when truncated"

    def test_truncation_is_disclosed_not_silent(self):
        """Silent truncation reads as 'that is everything'."""
        diff = ArtifactDiff(created=[f"m{i} (1 obs x 1 vars)" for i in range(80)])
        assert "more" in render_manifest(diff) or "truncated" in render_manifest(diff)

    def test_reports_no_data_values(self):
        rendered = render_manifest(
            ArtifactDiff(
                created=["counts (3 obs x 2 vars)"], files_written=["exports/x.csv"]
            )
        )
        assert "counts" in rendered
        # Shapes and names only; nothing resembling a matrix payload.
        assert "[" not in rendered and "array" not in rendered


class TestAppend:
    def test_appends_when_snapshots_are_healthy(self):
        out = append_manifest(
            "Done.",
            take_snapshot(fake_dm()),
            take_snapshot(fake_dm(modalities={"pbmc_qc": (500, 2000)})),
        )
        assert out.startswith("Done.")
        assert "pbmc_qc" in out

    def test_failed_before_snapshot_suppresses_manifest(self):
        """A partial diff could imply 'wrote nothing' when we simply could not read."""
        out = append_manifest(
            "Done.", WorkspaceSnapshot.empty_failed(), take_snapshot(fake_dm())
        )
        assert out == "Done."

    def test_failed_after_snapshot_suppresses_manifest(self):
        out = append_manifest(
            "Done.", take_snapshot(fake_dm()), WorkspaceSnapshot.empty_failed()
        )
        assert out == "Done."

    def test_render_failure_returns_prose_unchanged(self, monkeypatch):
        monkeypatch.setattr(
            "lobster.agents.artifact_manifest.diff_snapshots",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        out = append_manifest(
            "Done.", take_snapshot(fake_dm()), take_snapshot(fake_dm())
        )
        assert out == "Done."


class TestInvokeAndStoreIntegration:
    """The regression lock: behaviour on the delegation hot path."""

    @staticmethod
    def _agent(text="Analysis complete."):
        agent = MagicMock()

        async def ainvoke(_payload, config=None):
            message = MagicMock()
            message.content = text
            return {"messages": [message]}

        agent.ainvoke = ainvoke
        return agent

    def _run(self, **kwargs):
        from lobster.agents.graph import _invoke_and_store

        return asyncio.run(_invoke_and_store(**kwargs))

    def test_flag_off_is_byte_identical(self):
        """The default path must not change at all."""
        dm = fake_dm(modalities={"pbmc": (10, 20)})
        with_dm = self._run(
            agent=self._agent(),
            agent_name="transcriptomics_expert",
            task_description="t",
            store=None,
            data_manager=dm,
        )
        assert with_dm == "Analysis complete."
        assert MANIFEST_OPEN not in with_dm
        dm.list_modality_records.assert_not_called()

    def test_flag_off_matches_no_data_manager_path(self):
        baseline = self._run(
            agent=self._agent(),
            agent_name="a",
            task_description="t",
            store=None,
            data_manager=None,
        )
        assert baseline == "Analysis complete."

    def test_flag_on_appends_manifest_for_written_modality(self, monkeypatch):
        monkeypatch.setenv(MANIFEST_ENV_VAR, "1")

        dm = MagicMock()
        dm.list_workspace_files.return_value = {}
        # Empty before the handoff, one modality after.
        dm.list_modality_records.side_effect = [
            [],
            [{"name": "pbmc_qc", "n_obs": 500, "n_vars": 2000}],
        ]

        out = self._run(
            agent=self._agent(),
            agent_name="transcriptomics_expert",
            task_description="t",
            store=None,
            data_manager=dm,
        )
        assert "Analysis complete." in out
        assert "pbmc_qc" in out and "500 obs" in out

    def test_flag_on_with_no_writes_says_so_explicitly(self, monkeypatch):
        monkeypatch.setenv(MANIFEST_ENV_VAR, "1")
        out = self._run(
            agent=self._agent("I could not find the data."),
            agent_name="a",
            task_description="t",
            store=None,
            data_manager=fake_dm(),
        )
        assert "No new artifacts" in out

    def test_manifest_failure_does_not_break_delegation(self, monkeypatch):
        monkeypatch.setenv(MANIFEST_ENV_VAR, "1")
        dm = MagicMock()
        dm.list_modality_records.side_effect = RuntimeError("backend exploded")
        dm.list_workspace_files.return_value = {}

        out = self._run(
            agent=self._agent(),
            agent_name="a",
            task_description="t",
            store=None,
            data_manager=dm,
        )
        assert out == "Analysis complete."

    def test_store_key_still_appended_with_manifest_on(self, monkeypatch):
        monkeypatch.setenv(MANIFEST_ENV_VAR, "1")
        store = MagicMock()
        out = self._run(
            agent=self._agent(),
            agent_name="a",
            task_description="t",
            store=store,
            data_manager=fake_dm(),
        )
        assert "[store_key=" in out
        assert MANIFEST_OPEN in out

    def test_no_response_path_unaffected(self, monkeypatch):
        monkeypatch.setenv(MANIFEST_ENV_VAR, "1")
        agent = MagicMock()

        async def ainvoke(_payload, config=None):
            return {"messages": []}

        agent.ainvoke = ainvoke
        out = self._run(
            agent=agent,
            agent_name="a",
            task_description="t",
            store=None,
            data_manager=fake_dm(),
        )
        assert out == "Agent a returned no response."


class TestSupervisorPolicyHalf:
    """The manifest is inert without the prompt policy that interprets it.

    Splitting facts (tool output) from policy (system prompt) is what makes this work
    under spotlighting. If either half is missing the feature silently does nothing, so
    both are asserted.
    """

    @staticmethod
    def _prompt():
        from lobster.agents.supervisor import create_supervisor_prompt

        dm = MagicMock()
        dm.list_modalities.return_value = []
        dm.workspace_path = "/tmp/does-not-exist"  # nosec B108 # Nonexistent fixture path; no temporary file is created.
        return create_supervisor_prompt(
            dm, active_agents=["transcriptomics_expert", "research_agent"]
        )

    def test_prompt_explains_how_to_read_a_manifest(self):
        prompt = self._prompt()
        assert "ARTIFACT MANIFESTS" in prompt
        assert MANIFEST_OPEN in prompt

    def test_prompt_forbids_recomputing_listed_artifacts(self):
        """The anti-recompute policy must live here, where it has authority."""
        prompt = self._prompt()
        section = prompt[prompt.find("ARTIFACT MANIFESTS") :]
        assert "ALREADY EXISTS" in section
        assert "execute_custom_code" in section

    def test_prompt_treats_empty_manifest_plus_success_prose_as_unverified(self):
        """This is the I9 fabrication case, stated as policy."""
        section = self._prompt()
        section = section[section.find("ARTIFACT MANIFESTS") :]
        assert "UNVERIFIED" in section

    def test_prompt_distinguishes_absent_from_empty(self):
        section = self._prompt()
        section = section[section.find("ARTIFACT MANIFESTS") :]
        assert "absent" in section.lower()

    def test_prompt_stays_within_budget(self):
        """The supervisor prompt stays under 15K chars."""
        assert len(self._prompt()) < 15000


class TestBothFactoriesWired:
    """Parent→child handoffs go through the lazy factory.

    Wiring only the supervisor→specialist factory would silently miss every
    parent→child delegation, which is where the transcriptomics DE and annotation
    work actually happens.
    """

    def test_agent_tool_accepts_data_manager(self):
        import inspect

        from lobster.agents.graph import _create_agent_tool

        assert "data_manager" in inspect.signature(_create_agent_tool).parameters

    def test_lazy_delegation_tool_accepts_data_manager(self):
        import inspect

        from lobster.agents.graph import _create_lazy_delegation_tool

        assert (
            "data_manager" in inspect.signature(_create_lazy_delegation_tool).parameters
        )

    def test_lazy_tool_appends_manifest(self, monkeypatch):
        monkeypatch.setenv(MANIFEST_ENV_VAR, "1")
        from lobster.agents.graph import _create_lazy_delegation_tool

        agent = MagicMock()

        async def ainvoke(_payload, config=None):
            message = MagicMock()
            message.content = "Child done."
            return {"messages": [message]}

        agent.ainvoke = ainvoke

        dm = MagicMock()
        dm.list_workspace_files.return_value = {}
        dm.list_modality_records.side_effect = [
            [],
            [{"name": "de_results", "n_obs": 12, "n_vars": 3}],
        ]

        tool_fn = _create_lazy_delegation_tool(
            "de_analysis_expert",
            {"de_analysis_expert": agent},
            "Run DE.",
            store=None,
            data_manager=dm,
        )
        out = asyncio.run(tool_fn.coroutine(task_description="Run DE"))
        assert "Child done." in out
        assert "de_results" in out


class TestContentsLevelKeys:
    """`.obsm`/`.uns` key reporting detects in-place changes that preserve shape."""

    def test_in_place_pca_is_no_longer_invisible(self):
        """An in-place change is visible even when the shape is unchanged."""
        from lobster.agents.artifact_manifest import (
            WorkspaceSnapshot,
            _ModalityState,
            diff_snapshots,
        )

        before = WorkspaceSnapshot(modalities={"pbmc3k": _ModalityState(2700, 32738)})
        after = WorkspaceSnapshot(
            modalities={"pbmc3k": _ModalityState(2700, 32738, ("X_pca",), ("pca",))}
        )
        diff = diff_snapshots(before, after)
        assert not diff.is_empty, "in-place PCA must not be invisible"
        assert "X_pca" in diff.modified[0]
        assert "pca" in diff.modified[0]

    def test_created_modality_lists_its_keys(self):
        from lobster.agents.artifact_manifest import (
            WorkspaceSnapshot,
            _ModalityState,
            diff_snapshots,
        )

        after = WorkspaceSnapshot(
            modalities={
                "pbmc3k_pca": _ModalityState(2700, 20, ("X_pca",), ("pca", "neighbors"))
            }
        )
        diff = diff_snapshots(WorkspaceSnapshot(), after)
        rendered = diff.created[0]
        assert "obsm: X_pca" in rendered
        assert "uns:" in rendered and "pca" in rendered

    def test_modified_reports_only_the_newly_added_keys(self):
        """The supervisor needs what THIS handoff added, not a full re-listing."""
        from lobster.agents.artifact_manifest import (
            WorkspaceSnapshot,
            _ModalityState,
            diff_snapshots,
        )

        before = WorkspaceSnapshot(
            modalities={"m": _ModalityState(10, 5, ("X_pca",), ("pca",))}
        )
        after = WorkspaceSnapshot(
            modalities={
                "m": _ModalityState(10, 5, ("X_pca", "X_umap"), ("pca", "neighbors"))
            }
        )
        text = diff_snapshots(before, after).modified[0]
        assert "X_umap" in text and "neighbors" in text
        assert "X_pca" not in text, "already-present keys are not news"

    def test_shape_change_and_key_change_both_reported(self):
        from lobster.agents.artifact_manifest import (
            WorkspaceSnapshot,
            _ModalityState,
            diff_snapshots,
        )

        before = WorkspaceSnapshot(modalities={"m": _ModalityState(100, 50)})
        after = WorkspaceSnapshot(
            modalities={"m": _ModalityState(80, 50, ("X_pca",), ())}
        )
        text = diff_snapshots(before, after).modified[0]
        assert "100x50 -> 80x50" in text
        assert "X_pca" in text

    def test_identical_state_still_reports_nothing(self):
        """Keys must not manufacture phantom changes."""
        from lobster.agents.artifact_manifest import (
            WorkspaceSnapshot,
            _ModalityState,
            diff_snapshots,
        )

        state = _ModalityState(10, 5, ("X_pca",), ("pca",))
        diff = diff_snapshots(
            WorkspaceSnapshot(modalities={"m": state}),
            WorkspaceSnapshot(modalities={"m": state}),
        )
        assert diff.is_empty

    def test_key_removal_still_says_something(self):
        """A bare name with no explanation would be useless."""
        from lobster.agents.artifact_manifest import (
            WorkspaceSnapshot,
            _ModalityState,
            diff_snapshots,
        )

        before = WorkspaceSnapshot(
            modalities={"m": _ModalityState(10, 5, ("X_pca",), ())}
        )
        after = WorkspaceSnapshot(modalities={"m": _ModalityState(10, 5, (), ())})
        text = diff_snapshots(before, after).modified[0]
        assert "contents changed" in text

    def test_keys_are_capped(self):
        from lobster.agents.artifact_manifest import (
            MAX_KEYS_LISTED,
            WorkspaceSnapshot,
            _ModalityState,
            diff_snapshots,
        )

        many = tuple(f"k{i}" for i in range(MAX_KEYS_LISTED + 5))
        after = WorkspaceSnapshot(modalities={"m": _ModalityState(1, 1, many, ())})
        text = diff_snapshots(WorkspaceSnapshot(), after).created[0]
        assert "more)" in text, "silent truncation reads as completeness"

    def test_no_data_values_leak_into_the_manifest(self):
        """Key NAMES only -- the module's standing rule."""
        from lobster.agents.artifact_manifest import (
            WorkspaceSnapshot,
            _ModalityState,
            diff_snapshots,
            render_manifest,
        )

        after = WorkspaceSnapshot(
            modalities={"m": _ModalityState(2, 2, ("X_pca",), ("pca",))}
        )
        rendered = render_manifest(diff_snapshots(WorkspaceSnapshot(), after))
        assert "0.045" not in rendered and "variance_ratio" not in rendered


class TestKeyReadingIsCheap:
    """The snapshot runs twice per delegation; it must not load data to list keys."""

    def test_cold_modalities_are_not_fetched(self):
        """Loading every cold modality off disk twice per handoff would be worse than
        the problem the manifest reports on."""
        from lobster.agents.artifact_manifest import _read_keys

        class _DM:
            @property
            def modalities(self):
                raise AssertionError("cold modality must not be fetched")

        assert _read_keys(_DM(), "cold_one", {"data_status": "cold"}) == ((), ())

    def test_hot_modality_keys_are_read(self):
        from lobster.agents.artifact_manifest import _read_keys

        class _AData:
            obsm = {"X_pca": object()}
            uns = {"pca": {"variance_ratio": [0.1]}}

        class _DM:
            modalities = {"m": _AData()}

        # Descend one level to name the nested field, rather than only its container.
        assert _read_keys(_DM(), "m", {"data_status": "hot"}) == (
            ("X_pca",),
            ("pca[variance_ratio]",),
        )

    def test_uns_descends_exactly_one_level(self):
        """Deeper nesting is NOT followed: .uns can be deep or self-referential, and this
        runs twice per delegation."""
        from lobster.agents.artifact_manifest import _flatten_uns

        keys = _flatten_uns({"a": {"b": {"c": {"d": 1}}}})
        assert keys == ["a[b]"], keys

    def test_self_referential_uns_does_not_hang(self):
        from lobster.agents.artifact_manifest import _flatten_uns

        loop: dict = {"x": {}}
        loop["x"]["self"] = loop
        assert _flatten_uns(loop) == ["x[self]"]

    def test_key_read_failure_is_not_fatal(self):
        """Keys are a bonus; a failure must not suppress the whole manifest."""
        from lobster.agents.artifact_manifest import _read_keys

        class _AData:
            @property
            def obsm(self):
                raise RuntimeError("backend exploded")

        class _DM:
            modalities = {"m": _AData()}

        assert _read_keys(_DM(), "m", {"data_status": "hot"}) == ((), ())

    def test_missing_modality_returns_empty(self):
        from lobster.agents.artifact_manifest import _read_keys

        class _DM:
            modalities = {}

        assert _read_keys(_DM(), "gone", {"data_status": "hot"}) == ((), ())
