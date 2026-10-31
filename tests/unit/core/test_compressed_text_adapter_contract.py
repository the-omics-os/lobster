"""Real text-loader contracts; transport, conversion and axes are not mocked."""

import csv
import gzip
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lobster.core.adapters import base, transcriptomics_adapter
from lobster.core.adapters.base import BaseAdapter
from lobster.core.adapters.transcriptomics_adapter import TranscriptomicsAdapter
from lobster.services.data_access.geo import parser


@pytest.fixture(autouse=True)
def owned_source_origins():
    root = Path(__file__).resolve().parents[3]
    expected = {
        base: "lobster/core/adapters/base.py",
        transcriptomics_adapter: "lobster/core/adapters/transcriptomics_adapter.py",
        parser: "lobster/services/data_access/geo/parser.py",
    }
    for module, relative in expected.items():
        assert Path(module.__file__).resolve() == root / relative


def _table(genes_as_rows):
    genes = [f"ENSG{i:011d}" for i in range(1, 25)]
    cells = [f"cell_{i}" for i in range(1, 7)]
    values = np.arange(1, 145, dtype=np.int64).reshape(24, 6)
    table = pd.DataFrame(values, index=genes, columns=cells)
    return table if genes_as_rows else table.T


def _write_table(path, table, separator, compressed):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=separator, lineterminator="\n")
    writer.writerow(["", *table.columns])
    for name, row in table.iterrows():
        writer.writerow([name, *row.tolist()])
    payload = stream.getvalue().encode("utf-8")
    path.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_cells_by_genes(adata, source, genes_as_rows):
    expected = source.T if genes_as_rows else source
    assert adata.obs_names.tolist() == expected.index.tolist()
    assert adata.var_names.tolist() == expected.columns.tolist()
    assert adata.shape == expected.shape
    assert adata.X.dtype == np.float32
    np.testing.assert_array_equal(adata.X, expected.to_numpy(dtype=np.float32))


@pytest.mark.parametrize("genes_as_rows", [True, False], ids=["gene_rows", "cell_rows"])
def test_real_uncompressed_single_cell_loader_control(tmp_path, genes_as_rows):
    source = _table(genes_as_rows)
    path = tmp_path / "matrix.tsv"
    before = _write_table(path, source, "\t", compressed=False)
    adata = TranscriptomicsAdapter(data_type="single_cell").from_source(
        path, transpose=genes_as_rows
    )
    _assert_cells_by_genes(adata, source, genes_as_rows)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("separator,suffix", [("\t", ".tsv"), (",", ".csv")])
def test_real_uncompressed_base_loader_control(tmp_path, separator, suffix):
    source = _table(False)
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, separator, compressed=False)
    adata = BaseAdapter().from_source(path)
    _assert_cells_by_genes(adata, source, genes_as_rows=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "separator,suffix,format_name",
    [("\t", ".txt.gz", "txt"), ("\t", ".tsv.gz", "tsv"), (",", ".csv.gz", "csv")],
)
def test_real_compressed_base_loader_keeps_default_axes(
    tmp_path, separator, suffix, format_name
):
    source = _table(False)
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, separator, compressed=True)
    adapter = BaseAdapter()
    assert adapter.detect_format(path) == format_name
    adata = adapter.from_source(path)
    _assert_cells_by_genes(adata, source, genes_as_rows=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("genes_as_rows", [True, False], ids=["gene_rows", "cell_rows"])
@pytest.mark.parametrize(
    "separator,suffix,format_name",
    [
        ("\t", ".txt.gz", "txt"),
        ("\t", ".TXT.GZ", "txt"),
        ("\t", ".tsv.gz", "tsv"),
        ("\t", ".TSV.GZ", "tsv"),
        (",", ".csv.gz", "csv"),
        (",", ".CSV.GZ", "csv"),
    ],
)
def test_real_compressed_single_cell_loader_resolves_axes_once(
    tmp_path, genes_as_rows, separator, suffix, format_name
):
    source = _table(genes_as_rows)
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, separator, compressed=True)
    adapter = TranscriptomicsAdapter(data_type="single_cell")
    assert adapter.detect_format(path) == format_name
    adata = adapter.from_source(path)
    _assert_cells_by_genes(adata, source, genes_as_rows)
    assert adata.uns["transpose_info"]["transpose_applied"] is genes_as_rows
    assert adata.uns["transpose_info"]["orientation_basis"] == "axis_labels"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("separator,suffix", [("\t", ".tsv"), (",", ".csv")])
def test_real_uncompressed_direct_reader_control(tmp_path, separator, suffix):
    source = _table(False)
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, separator, compressed=False)
    actual = BaseAdapter()._load_csv_data(path)
    pd.testing.assert_frame_equal(actual, source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize(
    "explicit_separator", [False, True], ids=["sniff", "explicit_sep"]
)
@pytest.mark.parametrize(
    "separator,suffix",
    [
        ("\t", ".txt.gz"),
        ("\t", ".TXT.GZ"),
        ("\t", ".tsv.gz"),
        ("\t", ".TSV.GZ"),
        (",", ".csv.gz"),
        (",", ".CSV.GZ"),
    ],
)
def test_real_compressed_direct_reader_reaches_transport_without_dispatch(
    tmp_path, separator, suffix, explicit_separator
):
    """Exercise sniff/read even while detect_format would refuse the suffix."""
    source = _table(False)
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, separator, compressed=True)
    kwargs = {"sep": separator} if explicit_separator else {}
    actual = BaseAdapter()._load_csv_data(path, **kwargs)
    pd.testing.assert_frame_equal(actual, source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("suffix", [".txt.gz", ".TXT.GZ"])
@pytest.mark.parametrize(
    "value,preservation,limit",
    [
        (16777215, "exact_integral_within_2pow24", 16777216),
        (16777216, "exact_integral_within_2pow24", 16777216),
        (0.125, "float32_precision", None),
        (0.1, "float32_precision", None),
        (1e-60, "float32_precision", None),
        # max's decimal CSV spelling is above the exact float32 bound; use its predecessor.
        (
            float(np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0))),
            "float32_precision",
            None,
        ),
    ],
    ids=[
        "limit_minus_one",
        "limit",
        "fraction",
        "precision",
        "underflow",
        "range_edge",
    ],
)
def test_real_compressed_single_cell_numeric_policy_admitted(
    tmp_path, suffix, value, preservation, limit
):
    source = _table(True).astype(np.float64)
    source.iloc[0, 0] = value
    if value == float(
        np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0))
    ):
        # The matrix is fractional, not an all-integral matrix above the limit.
        source.iloc[1, 0] = 1.5
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, "\t", compressed=True)
    adata = TranscriptomicsAdapter(data_type="single_cell").from_source(path)
    _assert_cells_by_genes(adata, source, genes_as_rows=True)
    assert adata.uns["numeric_storage_policy"] == {
        "dtype": "float32",
        "preservation": preservation,
        "limit": limit,
        "raw_count_provenance": "not_verified",
    }
    assert np.isfinite(adata.X).all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("suffix", [".txt.gz", ".TXT.GZ"])
@pytest.mark.parametrize(
    "value,closed_message",
    [
        (
            16777217,
            "Single-cell text matrix refused: integral values exceed the exact float32 range (2^24).",
        ),
        (
            16777218,
            "Single-cell text matrix refused: integral values exceed the exact float32 range (2^24).",
        ),
        (
            1e40,
            "Single-cell text matrix refused: values cannot be stored as finite float32.",
        ),
        (
            float(np.finfo(np.float32).max),
            "Single-cell text matrix refused: values cannot be stored as finite float32.",
        ),
        (
            -1,
            "Single-cell text matrix refused: content, completeness or orientation could not be established.",
        ),
        (
            np.nan,
            "Single-cell text matrix refused: content, completeness or orientation could not be established.",
        ),
        (
            np.inf,
            "Single-cell text matrix refused: content, completeness or orientation could not be established.",
        ),
        (
            -np.inf,
            "Single-cell text matrix refused: content, completeness or orientation could not be established.",
        ),
        (
            "not_numeric",
            "Single-cell text matrix refused: content, completeness or orientation could not be established.",
        ),
    ],
    ids=[
        "limit_plus_one",
        "representable_even_above_limit",
        "fractional_overflow",
        "csv_above_float32_bound",
        "negative",
        "missing",
        "infinite",
        "negative_infinite",
        "nonnumeric",
    ],
)
def test_real_compressed_single_cell_numeric_policy_refused(
    tmp_path, suffix, value, closed_message
):
    source = _table(True).astype(object if isinstance(value, str) else np.float64)
    source.iloc[0, 0] = value
    if value in (1e40, float(np.finfo(np.float32).max)):
        source.iloc[1, 0] = 1.5
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, "\t", compressed=True)
    # Reach the real loader/constructor. No validator, dtype cast or reader mock.
    with pytest.raises(parser.GEOFormatError) as refused:
        TranscriptomicsAdapter(data_type="single_cell").from_source(path)
    assert str(refused.value) == closed_message
    assert type(refused.value).__name__ == "SingleCellTextMatrixRefusal"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize(
    "separator,suffix",
    [
        ("\t", ".txt.gz"),
        ("\t", ".TXT.GZ"),
        ("\t", ".tsv.gz"),
        ("\t", ".TSV.GZ"),
        (",", ".csv.gz"),
        (",", ".CSV.GZ"),
    ],
)
def test_real_strict_reader_gzip_transport_preserves_required_contract(
    tmp_path, separator, suffix
):
    """Existing strict entry reaches the approved compression-only seam."""
    source = _table(True)
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, separator, compressed=True)
    result = parser.GEOParser().parse_required_expression_file(path)
    assert result.is_complete
    assert result.is_partial is False
    assert result.rows_read == len(source)
    assert result.original_tokens_valid is True
    assert result.source_format == "tabular"
    assert result.orientation == "unoriented"
    assert result.data.index.tolist() == source.index.tolist()
    assert result.data.columns.tolist() == source.columns.tolist()
    np.testing.assert_array_equal(result.data.to_numpy(), source.to_numpy())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("suffix", [".txt.gz", ".TXT.GZ"])
@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_genes",
        "duplicate_cells",
        "empty_gene",
        "empty_cell",
        "late_gene_conflict",
        "late_cell_conflict",
        "weak_genes",
        "numeric_genes",
        "unrecognized_cells",
    ],
)
def test_real_compressed_single_cell_full_axis_content_refusal(
    tmp_path, suffix, mutation
):
    source = _table(True)
    genes, cells = source.index.tolist(), source.columns.tolist()
    if mutation == "duplicate_genes":
        genes[-1] = genes[0]
    elif mutation == "duplicate_cells":
        cells[-1] = cells[0]
    elif mutation == "empty_gene":
        genes[-1] = ""
    elif mutation == "empty_cell":
        cells[-1] = ""
    elif mutation == "late_gene_conflict":
        genes[-1] = "sample_24"
    elif mutation == "late_cell_conflict":
        cells[-1] = "ENSG00000000099"
    elif mutation == "weak_genes":
        genes = [f"Custom{i}" for i in range(len(genes))]
    elif mutation == "numeric_genes":
        genes = [str(i) for i in range(len(genes))]
    elif mutation == "unrecognized_cells":
        cells = [f"arbitrary_{i}" for i in range(len(cells))]
    source.index, source.columns = genes, cells
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, "\t", compressed=True)
    with pytest.raises(parser.GEOFormatError) as refused:
        TranscriptomicsAdapter(data_type="single_cell").from_source(path)
    assert (
        str(refused.value)
        == "Single-cell text matrix refused: content, completeness or orientation could not be established."
    )
    assert type(refused.value).__name__ == "SingleCellTextMatrixRefusal"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("suffix", [".txt.gz", ".TXT.GZ"])
@pytest.mark.parametrize(
    "options",
    [
        {"transpose": False},
        {"first_column_names": False},
        {"gene_symbols_col": "cell_1"},
        {"sep": ","},
    ],
    ids=[
        "conflicting_transpose",
        "axis_coercion",
        "gene_column_coercion",
        "conflicting_separator",
    ],
)
def test_real_compressed_single_cell_conflicting_conversion_refused(
    tmp_path, suffix, options
):
    source = _table(True)
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, "\t", compressed=True)
    with pytest.raises(parser.GEOFormatError) as refused:
        TranscriptomicsAdapter(data_type="single_cell").from_source(path, **options)
    assert (
        str(refused.value)
        == "Single-cell text matrix refused: content, completeness or orientation could not be established."
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("suffix", [".txt.gz", ".TXT.GZ"])
@pytest.mark.parametrize("damage", ["truncated_trailer", "bad_crc"])
def test_real_compressed_single_cell_gzip_integrity_refused(tmp_path, suffix, damage):
    source = _table(True)
    path = tmp_path / f"matrix{suffix}"
    _write_table(path, source, "\t", compressed=True)
    payload = bytearray(path.read_bytes())
    if damage == "truncated_trailer":
        payload = payload[:-8]
    else:
        payload[-8] ^= 1
    path.write_bytes(payload)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(parser.GEOFormatError) as refused:
        TranscriptomicsAdapter(data_type="single_cell").from_source(path)
    assert (
        str(refused.value)
        == "Single-cell text matrix refused: content, completeness or orientation could not be established."
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == [path.name]


@pytest.mark.parametrize("suffix", [".txt.gz", ".TXT.GZ"])
def test_real_single_cell_text_declared_counts_fraction_refused(tmp_path, suffix):
    source = _table(True).astype(np.float64)
    source.iloc[0, 0] = 1.5
    path = tmp_path / f"matrix{suffix}"
    before = _write_table(path, source, "\t", compressed=True)
    loader = parser.GEOParser()
    baseline = loader.parse_required_expression_file(path)
    assert baseline.is_complete and baseline.original_tokens_valid is False
    method = getattr(loader, "parse_single_cell_text_matrix", None)
    assert callable(
        method
    ), "New single-cell text qualification contract is not implemented"
    with pytest.raises(parser.GEOFormatError) as refused:
        method(path, declared_counts=True)
    assert (
        str(refused.value)
        == "Single-cell text matrix refused: content, completeness or orientation could not be established."
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
