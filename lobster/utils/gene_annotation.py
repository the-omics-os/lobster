"""
Unified gene annotation for QC metrics (mitochondrial and ribosomal genes).

Provides both a runtime callable and an equivalent string constant for
notebook export via the AnalysisStep helper_code mechanism. This ensures
runtime and exported notebooks use identical annotation logic.

The 5-pattern cascade matches the QualityService detection methods:
  1. Human HGNC (MT-, RPS/RPL)
  2. Mouse MGI (mt-, Rps/Rpl)
  3. Alt delimiters / generic lowercase (MT., rps/rpl)
  4. Ensembl IDs / compact regex (ENSG00000198*, RP[SL]\\d+)
  5. Generic fallback (mito/mitochondr, ribosom)
"""

import re
import warnings

import numpy as np


def annotate_qc_genes(adata):
    """
    Annotate mitochondrial and ribosomal genes in adata.var using a 5-pattern cascade.

    Sets adata.var['mt'] (bool) and adata.var['ribo'] (bool).
    Idempotent: skips columns that already exist.

    Args:
        adata: AnnData object with gene names in var_names.
    """
    if "mt" not in adata.var.columns:
        mt_mask = _detect_mt(adata.var_names.tolist())
        adata.var["mt"] = mt_mask
        n = int(np.sum(mt_mask))
        if n == 0:
            warnings.warn(
                "No mitochondrial genes detected by any pattern. "
                "MT QC metrics will be 0%."
            )
        else:
            print(f"Mitochondrial genes annotated: {n}")

    if "ribo" not in adata.var.columns:
        ribo_mask = _detect_ribo(adata.var_names.tolist())
        adata.var["ribo"] = ribo_mask
        n = int(np.sum(ribo_mask))
        if n == 0:
            warnings.warn(
                "No ribosomal genes detected by any pattern. "
                "Ribosomal QC metrics will be 0%."
            )
        else:
            print(f"Ribosomal genes annotated: {n}")


def _detect_mt(gene_names):
    """Detect mitochondrial genes with 5-pattern cascade."""
    arr = np.array(gene_names, dtype=str)

    # Pattern 1: Human HGNC (MT-)
    mask = np.array([g.startswith("MT-") for g in arr])
    if mask.any():
        return mask

    # Pattern 2: Mouse MGI (mt-)
    mask = np.array([g.startswith("mt-") for g in arr])
    if mask.any():
        return mask

    # Pattern 3: Alternative delimiter (MT.)
    mask = np.array([g.startswith("MT.") for g in arr])
    if mask.any():
        return mask

    # Pattern 4: Ensembl IDs (known MT genome ranges)
    mask = np.array(
        [g.startswith("ENSG00000198") or g.startswith("ENSG00000210") for g in arr]
    )
    if mask.any():
        return mask

    # Pattern 5: Generic fallback (contains "mito" or "mitochondr")
    pat = re.compile(r"mito|mitochondr", re.IGNORECASE)
    mask = np.array([bool(pat.search(g)) for g in arr])
    if mask.any():
        return mask

    return np.zeros(len(arr), dtype=bool)


def _detect_ribo(gene_names):
    """Detect ribosomal genes with 5-pattern cascade."""
    arr = np.array(gene_names, dtype=str)

    # Pattern 1: Human HGNC (RPS/RPL)
    mask = np.array([g.startswith("RPS") or g.startswith("RPL") for g in arr])
    if mask.any():
        return mask

    # Pattern 2: Mouse MGI (Rps/Rpl)
    mask = np.array([g.startswith("Rps") or g.startswith("Rpl") for g in arr])
    if mask.any():
        return mask

    # Pattern 3: Generic lowercase (rps/rpl)
    mask = np.array(
        [g.lower().startswith("rps") or g.lower().startswith("rpl") for g in arr]
    )
    if mask.any():
        return mask

    # Pattern 4: Compact regex (RP[SL]\d+)
    pat = re.compile(r"^RP[SL]\d+")
    mask = np.array([bool(pat.match(g)) for g in arr])
    if mask.any():
        return mask

    # Pattern 5: Generic fallback (contains "ribosom")
    mask = np.array(["ribosom" in g.lower() for g in arr])
    if mask.any():
        return mask

    return np.zeros(len(arr), dtype=bool)


# ---------------------------------------------------------------------------
# String constant for notebook helper_code injection
# ---------------------------------------------------------------------------
# This is the source code of annotate_qc_genes (+ private helpers) as a
# string, suitable for inclusion in an AnalysisStep.helper_code list.
# It uses only `re`, `warnings`, and `numpy` — no lobster imports.
# ---------------------------------------------------------------------------

ANNOTATE_QC_GENES_HELPER = '''\
import re as _re
import warnings as _warnings
import numpy as _np


def annotate_qc_genes(adata):
    """Annotate mitochondrial (var['mt']) and ribosomal (var['ribo']) genes."""
    if "mt" not in adata.var.columns:
        mt_mask = _detect_mt(adata.var_names.tolist())
        adata.var["mt"] = mt_mask
        n = int(_np.sum(mt_mask))
        if n == 0:
            _warnings.warn("No mitochondrial genes detected. MT QC metrics will be 0%.")
        else:
            print(f"Mitochondrial genes annotated: {n}")

    if "ribo" not in adata.var.columns:
        ribo_mask = _detect_ribo(adata.var_names.tolist())
        adata.var["ribo"] = ribo_mask
        n = int(_np.sum(ribo_mask))
        if n == 0:
            _warnings.warn("No ribosomal genes detected. Ribosomal QC metrics will be 0%.")
        else:
            print(f"Ribosomal genes annotated: {n}")


def _detect_mt(gene_names):
    arr = _np.array(gene_names, dtype=str)
    mask = _np.array([g.startswith("MT-") for g in arr])
    if mask.any():
        return mask
    mask = _np.array([g.startswith("mt-") for g in arr])
    if mask.any():
        return mask
    mask = _np.array([g.startswith("MT.") for g in arr])
    if mask.any():
        return mask
    mask = _np.array([g.startswith("ENSG00000198") or g.startswith("ENSG00000210") for g in arr])
    if mask.any():
        return mask
    pat = _re.compile(r"mito|mitochondr", _re.IGNORECASE)
    mask = _np.array([bool(pat.search(g)) for g in arr])
    if mask.any():
        return mask
    return _np.zeros(len(arr), dtype=bool)


def _detect_ribo(gene_names):
    arr = _np.array(gene_names, dtype=str)
    mask = _np.array([g.startswith("RPS") or g.startswith("RPL") for g in arr])
    if mask.any():
        return mask
    mask = _np.array([g.startswith("Rps") or g.startswith("Rpl") for g in arr])
    if mask.any():
        return mask
    mask = _np.array([g.lower().startswith("rps") or g.lower().startswith("rpl") for g in arr])
    if mask.any():
        return mask
    pat = _re.compile(r"^RP[SL]\\d+")
    mask = _np.array([bool(pat.match(g)) for g in arr])
    if mask.any():
        return mask
    mask = _np.array(["ribosom" in g.lower() for g in arr])
    if mask.any():
        return mask
    return _np.zeros(len(arr), dtype=bool)
'''
