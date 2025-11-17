"""
Unit tests for H5AD serialization utilities.

Tests all sanitization functions and validation utilities to ensure
data can be safely saved to H5AD format.
"""

import collections
from pathlib import Path

import numpy as np
import pytest

from lobster.core.utils.h5ad_utils import (
    sanitize_dict,
    sanitize_key,
    sanitize_value,
    validate_for_h5ad,
)


class TestSanitizeKey:
    """Tests for sanitize_key function."""

    def test_sanitize_key_with_slash(self):
        """Test that slashes in keys are replaced."""
        assert sanitize_key("path/to/key") == "path__to__key"
        assert sanitize_key("a/b/c") == "a__b__c"

    def test_sanitize_key_custom_replacement(self):
        """Test custom slash replacement."""
        assert sanitize_key("path/to/key", slash_replacement="--") == "path--to--key"

    def test_sanitize_key_no_slash(self):
        """Test that keys without slashes are unchanged."""
        assert sanitize_key("normal_key") == "normal_key"
        assert sanitize_key("key-with-dash") == "key-with-dash"

    def test_sanitize_key_non_string(self):
        """Test that non-string keys are converted to strings."""
        assert sanitize_key(123) == "123"
        assert sanitize_key(123.45) == "123.45"


class TestSanitizeValue:
    """Tests for sanitize_value function."""

    def test_sanitize_none(self):
        """Test that None is converted to empty string."""
        assert sanitize_value(None) == ""

    def test_sanitize_bool(self):
        """Test that booleans are converted to strings."""
        assert sanitize_value(True) == "True"
        assert sanitize_value(False) == "False"

    def test_sanitize_path_object(self):
        """Test that Path objects are converted to strings."""
        path = Path("/tmp/data.csv")
        result = sanitize_value(path)
        assert isinstance(result, str)
        assert result == str(path)

    def test_sanitize_tuple(self):
        """Test that tuples are converted to lists."""
        result = sanitize_value((1, 2, 3))
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_sanitize_nested_tuple(self):
        """Test that nested tuples are recursively converted."""
        result = sanitize_value({"shape": (100, 200), "nested": (1, (2, 3))})
        assert result == {"shape": [100, 200], "nested": [1, [2, 3]]}

    def test_sanitize_ordered_dict(self):
        """Test that OrderedDict is converted to regular dict."""
        od = collections.OrderedDict([("a", 1), ("b", 2)])
        result = sanitize_value(od)
        assert isinstance(result, dict)
        assert not isinstance(result, collections.OrderedDict)
        assert result == {"a": 1, "b": 2}

    def test_sanitize_numpy_scalar(self):
        """Test that numpy scalars are converted to Python types."""
        assert sanitize_value(np.int64(42)) == 42
        assert isinstance(sanitize_value(np.int64(42)), int)

        assert sanitize_value(np.float32(3.14)) == pytest.approx(3.14)
        assert isinstance(sanitize_value(np.float32(3.14)), float)

    def test_sanitize_numpy_array_numeric(self):
        """Test that numeric numpy arrays are preserved."""
        arr = np.array([1, 2, 3, 4, 5])
        result = sanitize_value(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_sanitize_numpy_array_object(self):
        """Test that object arrays are converted to string arrays."""
        arr = np.array([Path("/tmp/a"), Path("/tmp/b"), "text"])
        result = sanitize_value(arr)
        assert isinstance(result, np.ndarray)
        assert result.dtype.kind in ('U', 'O')  # Unicode or object
        expected = np.array([str(Path("/tmp/a")), str(Path("/tmp/b")), "text"])
        np.testing.assert_array_equal(result, expected)

    def test_sanitize_dict_with_nested_structures(self):
        """Test complex nested dictionaries."""
        data = {
            "path": Path("/tmp/data.csv"),
            "shape": (100, 200),
            "metadata": {
                "version": np.int64(1),
                "active": True,
                "files": [Path("/a"), Path("/b")],
            },
        }
        result = sanitize_value(data)

        assert result["path"] == str(Path("/tmp/data.csv"))
        assert result["shape"] == [100, 200]
        assert result["metadata"]["version"] == 1
        assert result["metadata"]["active"] == "True"
        assert result["metadata"]["files"] == [str(Path("/a")), str(Path("/b"))]

    def test_sanitize_list_with_mixed_types(self):
        """Test lists with mixed types."""
        data = [1, "text", Path("/tmp"), (1, 2), None]
        result = sanitize_value(data)

        assert result[0] == 1
        assert result[1] == "text"
        assert result[2] == str(Path("/tmp"))
        assert result[3] == [1, 2]
        assert result[4] == ""

    def test_sanitize_dict_with_slash_in_keys(self):
        """Test that dict keys with slashes are sanitized."""
        data = {"path/to/data": "value", "normal_key": "value2"}
        result = sanitize_value(data)

        assert "path__to__data" in result
        assert "path/to/data" not in result
        assert result["normal_key"] == "value2"

    def test_sanitize_primitive_types_unchanged(self):
        """Test that primitive types pass through."""
        assert sanitize_value(42) == 42
        assert sanitize_value(3.14) == 3.14
        assert sanitize_value("string") == "string"
        assert sanitize_value([1, 2, 3]) == [1, 2, 3]
        assert sanitize_value({"a": 1}) == {"a": 1}

    def test_sanitize_custom_object_to_string(self):
        """Test that non-serializable objects are converted to strings."""
        class CustomObject:
            def __str__(self):
                return "CustomObject()"

        obj = CustomObject()
        result = sanitize_value(obj)
        assert isinstance(result, str)
        assert result == "CustomObject()"

    def test_sanitize_strict_mode_raises(self):
        """Test that strict mode raises ValueError for non-serializable objects."""
        class CustomObject:
            pass

        obj = CustomObject()
        with pytest.raises(ValueError, match="Cannot sanitize non-serializable"):
            sanitize_value(obj, strict=True)


class TestSanitizeDict:
    """Tests for sanitize_dict convenience function."""

    def test_sanitize_dict_basic(self):
        """Test basic dictionary sanitization."""
        data = {
            "file/path": Path("/tmp/data.csv"),
            "count": (1, 2, 3),
        }
        result = sanitize_dict(data)

        assert "file__path" in result
        assert result["file__path"] == str(Path("/tmp/data.csv"))
        assert result["count"] == [1, 2, 3]

    def test_sanitize_dict_nested(self):
        """Test nested dictionary sanitization."""
        data = {
            "metadata": {
                "path/to/file": Path("/tmp/test"),
                "shape": (10, 20),
            }
        }
        result = sanitize_dict(data)

        assert "path__to__file" in result["metadata"]
        assert result["metadata"]["path__to__file"] == str(Path("/tmp/test"))
        assert result["metadata"]["shape"] == [10, 20]


class TestValidateForH5ad:
    """Tests for validate_for_h5ad validation function."""

    def test_validate_clean_data(self):
        """Test that clean data passes validation."""
        data = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        issues = validate_for_h5ad(data)
        assert len(issues) == 0

    def test_validate_detects_path_objects(self):
        """Test that Path objects are detected."""
        data = {"file": Path("/tmp/data.csv")}
        issues = validate_for_h5ad(data)

        assert len(issues) > 0
        assert any("Path object" in issue for issue in issues)

    def test_validate_detects_tuples(self):
        """Test that tuples are detected."""
        data = {"shape": (100, 200)}
        issues = validate_for_h5ad(data)

        assert len(issues) > 0
        assert any("Tuple" in issue for issue in issues)

    def test_validate_detects_slash_in_keys(self):
        """Test that keys with slashes are detected."""
        data = {"path/to/data": "value"}
        issues = validate_for_h5ad(data)

        assert len(issues) > 0
        assert any("Key with '/'" in issue for issue in issues)

    def test_validate_nested_structures(self):
        """Test validation of nested structures."""
        data = {
            "metadata": {
                "file": Path("/tmp/test"),
                "shape": (10, 20),
                "items": [1, 2, Path("/tmp/item")],
            }
        }
        issues = validate_for_h5ad(data)

        # Should detect Path objects and tuple
        assert len(issues) >= 3
        assert any("Path object" in issue for issue in issues)
        assert any("Tuple" in issue for issue in issues)

    def test_validate_custom_objects(self):
        """Test that custom non-serializable objects are detected."""
        class CustomObject:
            pass

        data = {"custom": CustomObject()}
        issues = validate_for_h5ad(data)

        assert len(issues) > 0
        assert any("Non-serializable" in issue for issue in issues)

    def test_validate_provides_paths(self):
        """Test that validation issues include path information."""
        data = {
            "level1": {
                "level2": {
                    "problematic": Path("/tmp/test")
                }
            }
        }
        issues = validate_for_h5ad(data)

        assert len(issues) > 0
        assert any("root.level1.level2.problematic" in issue for issue in issues)


class TestRealWorldScenarios:
    """Tests for real-world GEO metadata scenarios."""

    def test_geo_metadata_sanitization(self):
        """Test sanitization of typical GEO metadata."""
        geo_metadata = {
            "dataset_id": "GSE194070",
            "dataset_type": "GEO",
            "processing_date": "2024-11-15T03:59:36",
            "cache_path": Path("/tmp/geo_cache"),
            "file_paths": [
                Path("/tmp/file1.csv"),
                Path("/tmp/file2.csv"),
            ],
            "shape": (1000, 2000),  # tuple from adata.shape
            "metadata": collections.OrderedDict([
                ("sample/type", "single_cell"),
                ("version", np.int64(1)),
            ]),
        }

        result = sanitize_value(geo_metadata)

        # Verify all problematic types are fixed
        assert isinstance(result["cache_path"], str)
        assert all(isinstance(p, str) for p in result["file_paths"])
        assert isinstance(result["shape"], list)
        assert isinstance(result["metadata"], dict)
        assert "sample__type" in result["metadata"]
        assert isinstance(result["metadata"]["version"], int)

    def test_quantification_metadata_sanitization(self):
        """Test sanitization of Kallisto/Salmon metadata."""
        quant_metadata = {
            "quantification_tool": "kallisto",
            "version": "0.46.1",
            "index_path": Path("/data/index"),
            "files": [Path("/data/sample1"), Path("/data/sample2")],
            "parameters": collections.OrderedDict([
                ("bootstrap", np.int64(100)),
                ("threads", np.int64(8)),
            ]),
        }

        result = sanitize_value(quant_metadata)

        assert isinstance(result["index_path"], str)
        assert all(isinstance(f, str) for f in result["files"])
        assert isinstance(result["parameters"], dict)
        assert isinstance(result["parameters"]["bootstrap"], int)

    def test_transpose_info_sanitization(self):
        """Test sanitization of transpose_info dict."""
        transpose_info = {
            "transpose_applied": True,
            "transpose_reason": "Quantification format",
            "original_shape": (60000, 4),  # tuple!
            "final_shape": (4, 60000),     # tuple!
            "data_type": "bulk_rnaseq",
            "format_specific": True,
        }

        result = sanitize_value(transpose_info)

        assert result["transpose_applied"] == "True"  # bool → str
        assert isinstance(result["original_shape"], list)
        assert isinstance(result["final_shape"], list)
        assert result["original_shape"] == [60000, 4]
        assert result["final_shape"] == [4, 60000]
