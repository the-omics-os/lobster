"""Read a bulk count table and a sample sheet the way the import tools do.

``import_bulk_counts`` and ``merge_sample_metadata`` call these, and so do the
notebook cells their steps record: a replay reads the same file into the same
object the session started from.
"""

import re
from pathlib import Path
from typing import Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd

# Same stable gene-ID pattern as lobster/services/data_access/geo/parser.py.
_GENE_ID = re.compile(r"(?:ENS[A-Z]*G\d+(?:\.\d+)?|[NX][MR]_\d+(?:\.\d+)?)$")
_FEATURECOUNTS_ANNOTATION = {"Chr", "Start", "End", "Strand", "Length"}


def bulk_genes_axis(df: pd.DataFrame):
    """Return ("rows"|"columns", reason) for the axis of a count table that holds
    genes, or (None, None) when neither gene IDs nor the shape settle it.

    Gene IDs on one axis win. Shape alone decides only when the table cannot be
    a targeted panel with more samples than genes: the larger axis is a whole
    transcriptome (>= 10,000), or it has >= 1,000 labels and the smaller has too
    few to be genes (< 20). A panel of under 20 genes over 1,000+ samples is out
    of scope: shape takes it the wrong way round; genes_axis settles it.
    """

    def id_share(labels):
        return np.mean([bool(_GENE_ID.match(str(label))) for label in labels])

    rows, cols = id_share(df.index), id_share(df.columns)
    if rows >= 0.9 and cols < 0.1:
        return "rows", "Ensembl/RefSeq gene IDs"
    if cols >= 0.9 and rows < 0.1:
        return "columns", "Ensembl/RefSeq gene IDs"
    larger, smaller = max(df.shape), min(df.shape)
    if larger >= 2 * smaller and (
        larger >= 10_000 or (larger >= 1_000 and smaller < 20)
    ):
        return ("rows" if df.shape[0] > df.shape[1] else "columns"), "shape"
    return None, None


def _counts_adata(table: pd.DataFrame) -> anndata.AnnData:
    """Samples x genes, the raw counts also in layers['counts']."""
    adata = anndata.AnnData(
        X=table.values.astype(np.float32),
        obs=pd.DataFrame(index=table.index),
        var=pd.DataFrame(index=table.columns),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


def _read_featurecounts(path: Path, source: str) -> Optional[anndata.AnnData]:
    """A featureCounts table (``#`` header or a Geneid column), else None."""
    try:
        with open(path) as f:
            first_lines = [f.readline() for _ in range(3)]
        is_fc = any(line.startswith("#") or "Geneid" in line for line in first_lines)
        if not (is_fc or source == "featurecounts"):
            return None
        df = pd.read_csv(path, sep="\t", comment="#")
        if "Geneid" in df.columns:
            df = df.set_index("Geneid")
        df = df.drop(columns=[c for c in _FEATURECOUNTS_ANNOTATION if c in df.columns])
        return _counts_adata(df.T)  # genes on rows in the file
    except Exception:  # nosec B110 # not featureCounts: read as a generic table
        return None


def read_bulk_count_table(
    path: Union[str, Path],
    source: str = "auto",
    gene_id_column: str = "auto",
    genes_axis: str = "auto",
    sep: Optional[str] = None,
) -> Tuple[
    Optional[anndata.AnnData],
    str,
    Optional[str],
    Optional[str],
    Optional[pd.DataFrame],
]:
    """Read one count file into samples x genes, raw counts also in layers['counts'].

    featureCounts output is recognised (``source='featurecounts'``, or ``'auto'``
    on a .txt/.tsv file); anything else is a CSV/TSV table with either layout
    (``genes_axis`` 'rows' or 'columns', or 'auto' to detect it).

    Returns ``(adata, source, genes_axis_used, axis_reason, table)``: ``source`` is
    'featurecounts', 'csv' or 'tsv'; ``adata`` is None when the genes axis cannot
    be told (``genes_axis_used`` None, ``table`` the table read).
    """
    path = Path(path)
    if source == "featurecounts" or (
        source == "auto" and path.suffix in (".txt", ".tsv")
    ):
        adata = _read_featurecounts(path, source)
        if adata is not None:
            return adata, "featurecounts", None, None, None

    sep = sep or ("\t" if path.suffix in (".tsv", ".txt") else ",")
    df = pd.read_csv(path, sep=sep, index_col=0)
    if gene_id_column != "auto" and gene_id_column in df.columns:
        df = df.set_index(gene_id_column)
    detected = "csv" if sep == "," else "tsv"

    axis_used, axis_reason = genes_axis, "given"
    if genes_axis == "auto":
        axis_used, axis_reason = bulk_genes_axis(df)
    if axis_used is None:
        return None, detected, None, None, df
    # AnnData wants samples x genes
    return (
        _counts_adata(df.T if axis_used == "rows" else df),
        detected,
        axis_used,
        axis_reason,
        df,
    )


def read_sample_sheet(path: Union[str, Path]) -> pd.DataFrame:
    """A sample sheet: Excel by its extension, .tsv tab-separated, else CSV."""
    path = Path(path)
    if path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def add_sample_columns(
    adata: anndata.AnnData, sheet: pd.DataFrame, sample_id_column: str
) -> Tuple[set, set, set]:
    """Add the sheet's columns that ``adata.obs`` lacks, matched on
    ``sample_id_column``; returns (columns added, samples matched, unmatched)."""
    sheet = sheet.set_index(sample_id_column)
    before = set(adata.obs.columns)
    matched = set(adata.obs_names) & set(sheet.index)
    unmatched = set(adata.obs_names) - set(sheet.index)
    for col in sheet.columns:
        if col not in adata.obs.columns:
            adata.obs[col] = sheet[col].reindex(adata.obs_names)
    return set(adata.obs.columns) - before, matched, unmatched
