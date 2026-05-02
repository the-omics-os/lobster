"""
Tests for DataLocalityProvider protocol and LocalDataLocalityProvider.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from lobster.core.interfaces.locality import (
    DataLocalityProvider,
    LocalDataLocalityProvider,
    _safe_filename,
)


class TestSafeFilename:
    def test_simple_name(self):
        assert _safe_filename("rna") == "rna"

    def test_strips_directory(self):
        assert _safe_filename("../../etc/passwd") == "passwd"

    def test_strips_parent_traversal(self):
        assert _safe_filename("../rna") == "rna"

    def test_dotdot_replacement(self):
        assert _safe_filename("..") == "_"

    def test_empty_string(self):
        assert _safe_filename("") == "_unnamed"

    def test_normal_modality_name(self):
        assert _safe_filename("geo_GSE131928_combined_quality_assessed") == "geo_GSE131928_combined_quality_assessed"


class TestLocalDataLocalityProvider:
    @pytest.fixture
    def mock_dm(self, tmp_path):
        dm = Mock()
        dm.data_dir = tmp_path / "data"
        dm.data_dir.mkdir()
        dm.list_modalities.return_value = ["in_memory_mod"]
        return dm

    @pytest.fixture
    def provider(self, mock_dm):
        return LocalDataLocalityProvider(mock_dm)

    def test_ensure_local_finds_disk_file(self, provider, mock_dm, tmp_path):
        h5ad = mock_dm.data_dir / "rna.h5ad"
        h5ad.write_text("fake h5ad")
        result = provider.ensure_local("rna", tmp_path / "target")
        assert result == h5ad

    def test_ensure_local_raises_for_missing(self, provider, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found on local disk"):
            provider.ensure_local("nonexistent", tmp_path / "target")

    def test_list_available_combines_memory_and_disk(self, provider, mock_dm):
        (mock_dm.data_dir / "disk_mod.h5ad").write_text("fake")
        available = provider.list_available()
        assert "in_memory_mod" in available
        assert "disk_mod" in available

    def test_list_available_deduplicates(self, provider, mock_dm):
        (mock_dm.data_dir / "in_memory_mod.h5ad").write_text("fake")
        available = provider.list_available()
        assert available.count("in_memory_mod") == 1

    def test_is_local_true_when_exists(self, provider, mock_dm):
        (mock_dm.data_dir / "rna.h5ad").write_text("fake")
        assert provider.is_local("rna") is True

    def test_is_local_false_when_missing(self, provider):
        assert provider.is_local("nonexistent") is False

    def test_protocol_compliance(self, provider):
        assert isinstance(provider, DataLocalityProvider)


class TestDataLocalityProviderProtocol:
    def test_custom_provider_satisfies_protocol(self):
        class MyProvider:
            def ensure_local(self, modality_name: str, target_dir: Path) -> Path:
                return target_dir / f"{modality_name}.h5ad"

            def list_available(self) -> list[str]:
                return ["mod1"]

            def is_local(self, modality_name: str) -> bool:
                return True

        assert isinstance(MyProvider(), DataLocalityProvider)
