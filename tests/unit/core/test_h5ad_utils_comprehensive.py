"""
Comprehensive unit tests for H5AD serialization utilities - Bug Fix #4.

This test suite validates the fix for the critical numpy scalar bug and
ensures ALL edge cases are properly handled for H5AD serialization.

Bug Fixed: Numpy scalars (np.int64, np.float64, etc.) were returned immediately
after conversion to Python primitives, bypassing stringification, causing
"Can't implicitly convert non-string objects to strings" errors.

Test Coverage:
- Category 1: Numpy scalar types (15+ tests)
- Category 2: Pandas types (5+ tests)
- Category 3: Datetime types (5+ tests)
- Category 4: Nested structures (10+ tests)
- Category 5: GEO metadata structures (5+ tests)
- Category 6: Edge cases (10+ tests)
- Category 7: Regression tests (5+ tests)
"""

import collections
import datetime
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from lobster.core.utils.h5ad_utils import sanitize_value, sanitize_dict

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TestNumpyScalarTypes:
    """Category 1: Numpy scalar type sanitization (15+ tests)."""

    def test_numpy_int64_sanitization(self):
        """Test np.int64 → string conversion."""
        result = sanitize_value(np.int64(42))
        assert result == "42"
        assert isinstance(result, str)

    def test_numpy_int32_sanitization(self):
        """Test np.int32 → string conversion."""
        result = sanitize_value(np.int32(42))
        assert result == "42"
        assert isinstance(result, str)

    def test_numpy_int16_sanitization(self):
        """Test np.int16 → string conversion."""
        result = sanitize_value(np.int16(42))
        assert result == "42"
        assert isinstance(result, str)

    def test_numpy_int8_sanitization(self):
        """Test np.int8 → string conversion."""
        result = sanitize_value(np.int8(42))
        assert result == "42"
        assert isinstance(result, str)

    def test_numpy_uint64_sanitization(self):
        """Test np.uint64 → string conversion."""
        result = sanitize_value(np.uint64(42))
        assert result == "42"
        assert isinstance(result, str)

    def test_numpy_float64_sanitization(self):
        """Test np.float64 → string conversion."""
        result = sanitize_value(np.float64(3.14159))
        assert result == "3.14159"
        assert isinstance(result, str)

    def test_numpy_float32_sanitization(self):
        """Test np.float32 → string conversion."""
        result = sanitize_value(np.float32(3.14))
        # float32 has less precision
        assert isinstance(result, str)
        assert result.startswith("3.1")

    def test_numpy_float16_sanitization(self):
        """Test np.float16 → string conversion."""
        result = sanitize_value(np.float16(3.14))
        assert isinstance(result, str)

    def test_numpy_bool_sanitization(self):
        """Test np.bool_ → string conversion."""
        result_true = sanitize_value(np.bool_(True))
        result_false = sanitize_value(np.bool_(False))
        assert result_true == "True"
        assert result_false == "False"
        assert isinstance(result_true, str)
        assert isinstance(result_false, str)

    def test_numpy_str_sanitization(self):
        """Test np.str_ → string conversion."""
        result = sanitize_value(np.str_("test"))
        assert result == "test"
        assert isinstance(result, str)

    def test_numpy_bytes_sanitization(self):
        """Test np.bytes_ → string conversion."""
        result = sanitize_value(np.bytes_(b"test"))
        assert isinstance(result, str)
        assert "test" in result

    def test_numpy_scalar_negative_values(self):
        """Test negative numpy scalars."""
        result = sanitize_value(np.int64(-42))
        assert result == "-42"
        assert isinstance(result, str)

    def test_numpy_scalar_zero(self):
        """Test numpy scalar zero."""
        result = sanitize_value(np.int64(0))
        assert result == "0"
        assert isinstance(result, str)

    def test_numpy_scalar_large_values(self):
        """Test large numpy scalars."""
        result = sanitize_value(np.int64(9223372036854775807))  # max int64
        assert isinstance(result, str)
        assert result == "9223372036854775807"

    def test_numpy_scalar_in_nested_dict(self):
        """Test numpy scalars in nested dictionaries (critical bug scenario)."""
        data = {
            'metadata': {
                'sample_count': np.int64(13),
                'contact_zip': np.int64(12345),
                'score': np.float64(3.14159),
                'is_valid': np.bool_(True),
            }
        }
        result = sanitize_value(data)

        # All numpy scalars should be strings
        assert isinstance(result['metadata']['sample_count'], str)
        assert result['metadata']['sample_count'] == "13"
        assert isinstance(result['metadata']['contact_zip'], str)
        assert result['metadata']['contact_zip'] == "12345"
        assert isinstance(result['metadata']['score'], str)
        assert result['metadata']['score'] == "3.14159"
        assert isinstance(result['metadata']['is_valid'], str)
        assert result['metadata']['is_valid'] == "True"


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
class TestPandasTypes:
    """Category 2: Pandas type sanitization (5+ tests)."""

    def test_pandas_na_sanitization(self):
        """Test pd.NA → empty string conversion."""
        result = sanitize_value(pd.NA)
        assert result == ""

    def test_pandas_nat_sanitization(self):
        """Test pd.NaT → empty string conversion."""
        result = sanitize_value(pd.NaT)
        assert result == ""

    def test_pandas_timestamp_sanitization(self):
        """Test pd.Timestamp → ISO format string."""
        ts = pd.Timestamp("2024-01-15 14:30:00")
        result = sanitize_value(ts)
        assert isinstance(result, str)
        assert "2024-01-15" in result
        assert "14:30:00" in result

    def test_pandas_timedelta_sanitization(self):
        """Test pd.Timedelta → string conversion."""
        td = pd.Timedelta(days=5, hours=3, minutes=30)
        result = sanitize_value(td)
        assert isinstance(result, str)
        assert "days" in result or "5" in result

    def test_pandas_period_sanitization(self):
        """Test pd.Period → string conversion."""
        period = pd.Period("2024-01", freq="M")
        result = sanitize_value(period)
        assert isinstance(result, str)
        assert "2024" in result


class TestDatetimeTypes:
    """Category 3: Datetime type sanitization (5+ tests)."""

    def test_datetime_datetime_sanitization(self):
        """Test datetime.datetime → ISO format string."""
        dt = datetime.datetime(2024, 1, 15, 14, 30, 0)
        result = sanitize_value(dt)
        assert isinstance(result, str)
        assert result == "2024-01-15T14:30:00"

    def test_datetime_date_sanitization(self):
        """Test datetime.date → ISO format string."""
        d = datetime.date(2024, 1, 15)
        result = sanitize_value(d)
        assert isinstance(result, str)
        assert result == "2024-01-15"

    def test_datetime_time_sanitization(self):
        """Test datetime.time → ISO format string."""
        t = datetime.time(14, 30, 0)
        result = sanitize_value(t)
        assert isinstance(result, str)
        assert result == "14:30:00"

    def test_datetime_timedelta_sanitization(self):
        """Test datetime.timedelta → string conversion."""
        td = datetime.timedelta(days=5, hours=3, minutes=30)
        result = sanitize_value(td)
        assert isinstance(result, str)
        assert "5" in result  # 5 days

    def test_datetime_with_microseconds(self):
        """Test datetime with microseconds."""
        dt = datetime.datetime(2024, 1, 15, 14, 30, 0, 123456)
        result = sanitize_value(dt)
        assert isinstance(result, str)
        assert "2024-01-15" in result


class TestNumericEdgeCases:
    """Category 4: Numeric type edge cases."""

    def test_decimal_sanitization(self):
        """Test Decimal → string conversion."""
        result = sanitize_value(Decimal("3.14159265358979323846"))
        assert isinstance(result, str)
        assert "3.14159" in result

    def test_fraction_sanitization(self):
        """Test Fraction → string conversion."""
        result = sanitize_value(Fraction(1, 3))
        assert isinstance(result, str)
        assert "1/3" in result

    def test_complex_number_sanitization(self):
        """Test complex number → string conversion."""
        result = sanitize_value(complex(3, 4))
        assert isinstance(result, str)
        assert "3" in result and "4" in result

    def test_python_int_sanitization(self):
        """Test Python int → string conversion (Bug Fix #3)."""
        result = sanitize_value(42)
        assert result == "42"
        assert isinstance(result, str)

    def test_python_float_sanitization(self):
        """Test Python float → string conversion (Bug Fix #3)."""
        result = sanitize_value(3.14159)
        assert result == "3.14159"
        assert isinstance(result, str)


class TestCollectionTypes:
    """Category 5: Collection type sanitization."""

    def test_set_sanitization(self):
        """Test set → numpy string array conversion."""
        result = sanitize_value({1, 2, 3, "a", "b"})
        assert isinstance(result, np.ndarray)
        assert result.dtype.kind in ('U', 'S', 'O')  # String types
        # Sets are unordered, but we sort them for consistency
        assert len(result) == 5

    def test_frozenset_sanitization(self):
        """Test frozenset → numpy string array conversion."""
        result = sanitize_value(frozenset([1, 2, 3]))
        assert isinstance(result, np.ndarray)
        assert result.dtype.kind in ('U', 'S', 'O')
        assert len(result) == 3

    def test_empty_set_sanitization(self):
        """Test empty set conversion."""
        result = sanitize_value(set())
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_bytes_sanitization(self):
        """Test bytes → string conversion."""
        result = sanitize_value(b"test bytes")
        assert isinstance(result, str)
        assert "test" in result

    def test_bytearray_sanitization(self):
        """Test bytearray → string conversion."""
        result = sanitize_value(bytearray(b"test"))
        assert isinstance(result, str)


class TestNestedStructures:
    """Category 6: Nested structure sanitization (10+ tests)."""

    def test_deeply_nested_dict_4_levels(self):
        """Test 4-level nested dictionary."""
        data = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': np.int64(42)
                    }
                }
            }
        }
        result = sanitize_value(data)
        assert result['level1']['level2']['level3']['level4'] == "42"

    def test_list_of_numpy_scalars(self):
        """Test list containing numpy scalars."""
        data = [np.int64(1), np.int64(2), np.int64(3)]
        result = sanitize_value(data)
        assert isinstance(result, np.ndarray)
        # All elements should be strings
        assert all(isinstance(x, (str, np.str_)) for x in result)

    def test_list_of_dicts_with_numpy_scalars(self):
        """Test list of dicts containing numpy scalars."""
        data = [
            {'count': np.int64(1), 'value': np.float64(1.1)},
            {'count': np.int64(2), 'value': np.float64(2.2)},
        ]
        result = sanitize_value(data)
        assert isinstance(result, np.ndarray)
        # Lists of dicts should be stringified
        assert all(isinstance(x, (str, np.str_)) for x in result)

    def test_tuple_of_numpy_scalars(self):
        """Test tuple containing numpy scalars."""
        data = (np.int64(1), np.int64(2), np.int64(3))
        result = sanitize_value(data)
        assert isinstance(result, np.ndarray)
        assert result.dtype.kind in ('U', 'S', 'O')

    def test_dict_with_mixed_numpy_python_types(self):
        """Test dict with both numpy and Python types."""
        data = {
            'numpy_int': np.int64(42),
            'python_int': 42,
            'numpy_float': np.float64(3.14),
            'python_float': 3.14,
            'numpy_bool': np.bool_(True),
            'python_bool': True,
        }
        result = sanitize_value(data)

        # All should be strings
        assert all(isinstance(v, str) for v in result.values())

    def test_ordered_dict_with_numpy_scalars(self):
        """Test OrderedDict containing numpy scalars."""
        data = collections.OrderedDict([
            ('first', np.int64(1)),
            ('second', np.int64(2)),
        ])
        result = sanitize_value(data)

        assert isinstance(result, dict)
        assert not isinstance(result, collections.OrderedDict)
        assert result['first'] == "1"
        assert result['second'] == "2"

    def test_empty_structures(self):
        """Test empty collections."""
        assert sanitize_value({}) == {}
        assert sanitize_value([]) == []
        empty_arr = sanitize_value(np.array([]))
        assert isinstance(empty_arr, np.ndarray)
        assert len(empty_arr) == 0

    def test_none_in_nested_structures(self):
        """Test None values in nested structures."""
        data = {
            'value': None,
            'list': [1, None, 3],
            'nested': {'inner': None}
        }
        result = sanitize_value(data)

        assert result['value'] == ""
        assert "" in result['list']
        assert result['nested']['inner'] == ""

    def test_path_objects_in_lists(self):
        """Test Path objects in lists."""
        data = [Path("/tmp/a"), Path("/tmp/b"), Path("/tmp/c")]
        result = sanitize_value(data)

        assert isinstance(result, np.ndarray)
        assert all(isinstance(x, (str, np.str_)) for x in result)

    def test_mixed_type_list(self):
        """Test list with many different types."""
        data = [
            42,                      # Python int
            np.int64(42),           # Numpy int
            3.14,                   # Python float
            np.float64(3.14),       # Numpy float
            True,                   # Python bool
            np.bool_(True),         # Numpy bool
            "string",               # String
            None,                   # None
            Path("/tmp/test"),      # Path
        ]
        result = sanitize_value(data)

        assert isinstance(result, np.ndarray)
        assert result.dtype.kind in ('U', 'S', 'O')


class TestGEOMetadataStructures:
    """Category 7: Real GEO metadata structures (5+ tests)."""

    def test_gse267814_metadata_structure(self):
        """Test exact GSE267814 metadata structure that was failing."""
        metadata = {
            'contact_zip/postal_code': np.int64(12345),  # Numpy scalar with '/' in key
            'sample_count': np.int64(13),
            'is_processed': np.bool_(True),
            'submission_date': None,
            'platforms': ['GPL20795', 'GPL24676'],
            'nested_data': {
                'value': np.int64(42),
                'list': [np.int64(1), np.int64(2)],
            }
        }
        result = sanitize_value(metadata)

        # Keys with '/' should be replaced
        assert 'contact_zip__postal_code' in result
        assert 'contact_zip/postal_code' not in result

        # All numpy scalars should be strings
        assert isinstance(result['contact_zip__postal_code'], str)
        assert result['contact_zip__postal_code'] == "12345"
        assert isinstance(result['sample_count'], str)
        assert result['sample_count'] == "13"
        assert isinstance(result['is_processed'], str)
        assert result['is_processed'] == "True"

        # None should be empty string
        assert result['submission_date'] == ""

        # Lists should be numpy arrays
        assert isinstance(result['platforms'], np.ndarray)

        # Nested structures should work
        assert isinstance(result['nested_data']['value'], str)
        assert result['nested_data']['value'] == "42"

    def test_geo_provenance_structure(self):
        """Test GEO provenance structure in .uns."""
        provenance = {
            'activities': [
                {
                    'parameters': {
                        'sample_metadata': {
                            'contact_zip/postal_code': np.int64(12345),
                            'sample_count': np.int64(13),
                            'is_processed': np.bool_(True),
                            'platforms': ['GPL20795'],
                        }
                    }
                }
            ]
        }
        result = sanitize_value(provenance)

        # Lists of dicts should be stringified
        assert isinstance(result['activities'], np.ndarray)

    def test_quantification_metadata(self):
        """Test Kallisto/Salmon quantification metadata."""
        metadata = {
            'tool': 'kallisto',
            'version': '0.46.1',
            'bootstrap': np.int64(100),
            'threads': np.int64(8),
            'index_size': np.int64(123456789),
        }
        result = sanitize_value(metadata)

        assert all(isinstance(v, str) for k, v in result.items() if k != 'tool' and k != 'version')

    def test_transpose_info_structure(self):
        """Test transpose_info metadata structure."""
        transpose_info = {
            'transpose_applied': np.bool_(True),
            'original_shape': (60000, 4),
            'final_shape': (4, 60000),
            'reason': 'Quantification format',
        }
        result = sanitize_value(transpose_info)

        assert result['transpose_applied'] == "True"
        assert isinstance(result['original_shape'], np.ndarray)
        assert isinstance(result['final_shape'], np.ndarray)

    def test_complex_geo_metadata_full(self):
        """Test complete complex GEO metadata structure."""
        metadata = {
            'dataset_id': 'GSE267814',
            'dataset_type': 'GEO',
            'processing_date': datetime.datetime(2024, 11, 17, 3, 59, 36),
            'cache_path': Path('/tmp/geo_cache/GSE267814'),
            'file_paths': [
                Path('/tmp/file1.csv'),
                Path('/tmp/file2.csv'),
            ],
            'shape': (1000, 2000),
            'metadata': collections.OrderedDict([
                ('sample/type', 'single_cell'),
                ('version', np.int64(1)),
                ('scores', [np.float64(1.1), np.float64(2.2), np.float64(3.3)]),
            ]),
            'sample_counts': np.array([np.int64(100), np.int64(200), np.int64(300)]),
        }
        result = sanitize_value(metadata)

        # All Path objects should be strings
        assert isinstance(result['cache_path'], str)
        assert all(isinstance(x, (str, np.str_)) for x in result['file_paths'])

        # Datetime should be ISO string
        assert isinstance(result['processing_date'], str)
        assert "2024-11-17" in result['processing_date']

        # Tuples should be numpy arrays
        assert isinstance(result['shape'], np.ndarray)

        # OrderedDict should be regular dict
        assert isinstance(result['metadata'], dict)
        assert not isinstance(result['metadata'], collections.OrderedDict)


class TestRegressionPrevious:
    """Category 8: Regression tests for previous Bug Fix #3."""

    def test_bug_fix_3_python_ints_in_dicts(self):
        """Ensure Bug Fix #3 still works: Python int → string."""
        data = {'count': 42, 'nested': {'value': 100}}
        result = sanitize_value(data)

        assert result['count'] == "42"
        assert result['nested']['value'] == "100"

    def test_bug_fix_3_python_floats_in_dicts(self):
        """Ensure Bug Fix #3 still works: Python float → string."""
        data = {'score': 3.14, 'nested': {'value': 2.71}}
        result = sanitize_value(data)

        assert result['score'] == "3.14"
        assert result['nested']['value'] == "2.71"

    def test_bug_fix_3_lists_to_numpy_arrays(self):
        """Ensure Bug Fix #3 still works: Lists → numpy arrays."""
        data = {'items': [1, 2, 3]}
        result = sanitize_value(data)

        assert isinstance(result['items'], np.ndarray)

    def test_bug_fix_3_lists_of_dicts(self):
        """Ensure Bug Fix #3 still works: Lists of dicts → stringified."""
        data = {'activities': [{'id': 1, 'value': 'a'}, {'id': 2, 'value': 'b'}]}
        result = sanitize_value(data)

        assert isinstance(result['activities'], np.ndarray)
        # Elements should be stringified
        assert all(isinstance(x, (str, np.str_)) for x in result['activities'])

    def test_bug_fix_3_keys_with_slashes(self):
        """Ensure Bug Fix #3 still works: Keys with '/' replaced."""
        data = {'path/to/data': 42, 'normal_key': 100}
        result = sanitize_value(data)

        assert 'path__to__data' in result
        assert 'path/to/data' not in result
        assert result['path__to__data'] == "42"


class TestEdgeCasesAndBoundaries:
    """Category 9: Edge cases and boundary conditions."""

    def test_very_large_nested_structure(self):
        """Test very deeply nested structure (10 levels)."""
        data = {'l0': {'l1': {'l2': {'l3': {'l4': {'l5': {'l6': {'l7': {'l8': {'l9': np.int64(42)}}}}}}}}}}
        result = sanitize_value(data)

        # Should handle deep nesting
        value = result
        for i in range(10):
            value = value[f'l{i}']
        assert value == "42"

    def test_unicode_strings_preserved(self):
        """Test Unicode strings are preserved."""
        data = {'text': '你好世界', 'emoji': '🔬🧬'}
        result = sanitize_value(data)

        assert result['text'] == '你好世界'
        assert result['emoji'] == '🔬🧬'

    def test_special_float_values(self):
        """Test special float values (inf, -inf, nan)."""
        data = {
            'inf': float('inf'),
            'neg_inf': float('-inf'),
            'nan': float('nan'),
            'np_inf': np.float64('inf'),
            'np_nan': np.float64('nan'),
        }
        result = sanitize_value(data)

        # All should be strings
        assert all(isinstance(v, str) for v in result.values())
        assert result['inf'] == "inf"
        assert result['neg_inf'] == "-inf"
        # NaN values are caught by pd.isna() and converted to empty string
        # This is correct behavior for HDF5 compatibility
        assert result['nan'] == ""  # pd.isna() catches float('nan')
        assert result['np_nan'] == ""  # pd.isna() catches np.float64('nan')

    def test_very_long_strings(self):
        """Test very long strings are preserved."""
        long_string = "A" * 10000
        result = sanitize_value(long_string)
        assert result == long_string
        assert len(result) == 10000

    def test_empty_string_preserved(self):
        """Test empty string is preserved."""
        result = sanitize_value("")
        assert result == ""

    def test_whitespace_preserved(self):
        """Test whitespace is preserved."""
        data = {'spaces': '   ', 'tabs': '\t\t', 'newlines': '\n\n'}
        result = sanitize_value(data)

        assert result['spaces'] == '   '
        assert result['tabs'] == '\t\t'
        assert result['newlines'] == '\n\n'

    def test_numeric_string_not_converted(self):
        """Test numeric strings stay as strings."""
        data = {'str_num': '42', 'str_float': '3.14'}
        result = sanitize_value(data)

        assert result['str_num'] == '42'
        assert result['str_float'] == '3.14'


class TestValidationIntegration:
    """Category 10: Integration with H5AD backend."""

    def test_sanitize_dict_with_comprehensive_data(self):
        """Test sanitize_dict with all problematic types."""
        data = {
            'numpy_int': np.int64(42),
            'numpy_float': np.float64(3.14),
            'numpy_bool': np.bool_(True),
            'python_int': 42,
            'python_float': 3.14,
            'python_bool': True,
            'none_value': None,
            'path': Path('/tmp/test'),
            'tuple': (1, 2, 3),
            'set': {1, 2, 3},
            'list': [1, 2, 3],
            'key/with/slash': np.int64(100),
        }
        result = sanitize_dict(data)

        # All values should be safe for H5AD
        assert isinstance(result['numpy_int'], str)
        assert isinstance(result['numpy_float'], str)
        assert isinstance(result['numpy_bool'], str)
        assert isinstance(result['python_int'], str)
        assert isinstance(result['python_float'], str)
        assert isinstance(result['python_bool'], str)
        assert result['none_value'] == ""
        assert isinstance(result['path'], str)
        assert isinstance(result['tuple'], np.ndarray)
        assert isinstance(result['set'], np.ndarray)
        assert isinstance(result['list'], np.ndarray)

        # Key with slash should be sanitized
        assert 'key__with__slash' in result
        assert 'key/with/slash' not in result


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
