"""
Contract tests for plugin registration via pyproject.toml entry-point declarations.

These tests validate that pyproject.toml is correctly configured with the required
entry-point declarations for both lobster.queue_preparers and lobster.download_services
groups. They use the real installed package via importlib.metadata — NO mocking.

IMPORTANT: After any pyproject.toml entry-point change, the package must be reinstalled
for tests to reflect the new state:

    uv pip install -e .

These tests FAIL RED until Plan 02 adds the entry-point declarations to pyproject.toml
and the package is reinstalled. They serve as the contract that Plan 02 must satisfy.

Requirements covered: PLUG-03, PLUG-04
"""

from importlib.metadata import entry_points


EXPECTED_DATABASES = {"geo", "sra", "pride", "massive", "metabolights"}


class TestPluginRegistrationContract:
    """
    Contract tests that validate pyproject.toml entry-point declarations.

    All tests use the real importlib.metadata discovery — no mocks.
    Tests FAIL RED until pyproject.toml declares the entry points and
    `uv pip install -e .` is run to regenerate dist-info.
    """

    def test_all_5_queue_preparer_databases_discoverable(self):
        """
        Assert all 5 queue preparer databases are discoverable via entry points.

        Validates the [project.entry-points."lobster.queue_preparers"] section
        in pyproject.toml declares geo, sra, pride, massive, and metabolights.

        FAILS RED: No lobster.queue_preparers entry points declared yet.
        """
        discovered = {ep.name for ep in entry_points(group="lobster.queue_preparers")}
        assert EXPECTED_DATABASES <= discovered, (
            f"Missing queue preparer entry points: {EXPECTED_DATABASES - discovered}. "
            "Add [project.entry-points.\"lobster.queue_preparers\"] to pyproject.toml "
            "then run `uv pip install -e .` to register them."
        )

    def test_all_5_download_service_databases_discoverable(self):
        """
        Assert all 5 download service databases are discoverable via entry points.

        Validates the [project.entry-points."lobster.download_services"] section
        in pyproject.toml declares geo, sra, pride, massive, and metabolights.

        FAILS RED: No lobster.download_services entry points declared yet.
        """
        discovered = {ep.name for ep in entry_points(group="lobster.download_services")}
        assert EXPECTED_DATABASES <= discovered, (
            f"Missing download service entry points: {EXPECTED_DATABASES - discovered}. "
            "Add [project.entry-points.\"lobster.download_services\"] to pyproject.toml "
            "then run `uv pip install -e .` to register them."
        )

    def test_queue_preparer_entry_points_load(self):
        """
        Assert each discovered queue preparer entry point loads a callable class.

        Verifies each entry-point declaration points to a real, importable class.
        This catches typos in pyproject.toml module paths.

        FAILS RED: No queue preparer entry points declared yet (discovered set is empty,
        and the subset assertion in the discovery test will fail first).
        """
        discovered = list(entry_points(group="lobster.queue_preparers"))
        # This assertion will fail first if no entry points are declared
        assert len(discovered) >= len(EXPECTED_DATABASES), (
            f"Expected at least {len(EXPECTED_DATABASES)} queue preparer entry points, "
            f"got {len(discovered)}. Run `uv pip install -e .` after updating pyproject.toml."
        )
        for ep in discovered:
            loaded = ep.load()
            assert callable(loaded), (
                f"Queue preparer entry point '{ep.name}' loaded a non-callable: {loaded!r}. "
                "Entry points must point to classes, not instances or modules."
            )

    def test_download_service_entry_points_load(self):
        """
        Assert each discovered download service entry point loads a callable class.

        Verifies each entry-point declaration points to a real, importable class.
        This catches typos in pyproject.toml module paths.

        FAILS RED: No download service entry points declared yet (discovered set is empty,
        and the subset assertion in the discovery test will fail first).
        """
        discovered = list(entry_points(group="lobster.download_services"))
        # This assertion will fail first if no entry points are declared
        assert len(discovered) >= len(EXPECTED_DATABASES), (
            f"Expected at least {len(EXPECTED_DATABASES)} download service entry points, "
            f"got {len(discovered)}. Run `uv pip install -e .` after updating pyproject.toml."
        )
        for ep in discovered:
            loaded = ep.load()
            assert callable(loaded), (
                f"Download service entry point '{ep.name}' loaded a non-callable: {loaded!r}. "
                "Entry points must point to classes, not instances or modules."
            )
