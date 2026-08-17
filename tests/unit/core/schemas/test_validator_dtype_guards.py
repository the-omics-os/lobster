"""Assignment 4: schema validators must run under the pandas-3 string dtype.

These validators gated string-only checks on ``dtype == "object"``, which is
False for the pandas-3 ``str`` dtype (and pandas-2 ``StringDtype``), so the
checks silently no-op'd on exactly the data they exist to police. The fix routes
on ``is_string_dtype``. Each test asserts the check FIRES under both string-
inference regimes — under ``infer_string_on`` the column is ``str`` dtype, which
is the case the old guard skipped, so a passing assertion there is the
discriminating one.
"""
import anndata as ad
import numpy as np
import pandas as pd

from lobster.core.schemas.genomics import _validate_vcf_fields
from lobster.core.schemas.metagenomics import _validate_taxonomy
from lobster.core.schemas.transcriptomics import _validate_gene_symbols


def _adata(var: pd.DataFrame) -> ad.AnnData:
    n_var = len(var)
    X = np.ones((2, n_var), dtype=float)
    var.index = [f"F{i}" for i in range(n_var)]
    return ad.AnnData(X=X, var=var)


def test_gene_symbol_format_check_fires(pandas_infer_string):
    # two symbols do not start with a letter -> "unusual format" warning
    adata = _adata(pd.DataFrame({"gene_symbol": ["1BAD", "GENE", "2ALSO"]}))
    result = _validate_gene_symbols(adata)
    assert any("unusual format" in w for w in result.warnings), result.warnings


def test_vcf_ref_alt_base_check_fires(pandas_infer_string):
    adata = _adata(
        pd.DataFrame(
            {
                "CHROM": ["1", "2", "3"],
                "POS": [100, 200, 300],
                "REF": ["A", "Z", "C"],   # Z is non-standard
                "ALT": ["G", "Q", "T"],   # Q is non-standard
            }
        )
    )
    result = _validate_vcf_fields(adata)
    assert any("non-standard REF" in w for w in result.warnings), result.warnings
    assert any("non-standard ALT" in w for w in result.warnings), result.warnings


def test_taxonomy_unassigned_check_fires(pandas_infer_string):
    # 2/3 = 66% unassigned at Genus (> 30% threshold)
    adata = _adata(pd.DataFrame({"Genus": ["Unassigned", "unknown", "Escherichia"]}))
    result = _validate_taxonomy(adata)
    assert any("Unassigned" in w for w in result.warnings), result.warnings


# --- object-with-NaN / mixed regressions (Codex findings 3-5): is_string_dtype
# alone is False for these, so an is_string_dtype-only swap would skip them.


def test_gene_symbol_check_fires_on_object_with_missing():
    adata = _adata(pd.DataFrame({"gene_symbol": pd.Series(["ACTB", None, "1BAD"], dtype=object)}))
    result = _validate_gene_symbols(adata)
    assert any("unusual format" in w for w in result.warnings), result.warnings


def test_vcf_check_fires_on_object_ref_with_nan():
    adata = _adata(
        pd.DataFrame(
            {
                "CHROM": ["1", "2", "3"],
                "POS": [1, 2, 3],
                "REF": pd.Series(["A", np.nan, "X"], dtype=object),
                "ALT": pd.Series(["G", "T", "C"], dtype=object),
            }
        )
    )
    result = _validate_vcf_fields(adata)
    assert any("non-standard REF" in w for w in result.warnings), result.warnings


def test_taxonomy_check_fires_on_mixed_object():
    adata = _adata(pd.DataFrame({"Kingdom": pd.Series(["Unknown", 5], dtype=object)}))
    result = _validate_taxonomy(adata)
    assert any("Unassigned" in w for w in result.warnings), result.warnings
