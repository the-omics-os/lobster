"""
Loom file GEO metadata enrichment.

Standalone functions for merging GEO SOFT sample metadata into AnnData objects
loaded from Loom files. Usable from both the local CLI pipeline and cloud API.
"""

import re
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Pattern to extract GSE accession from filenames like "GSE162183_SomeDescription.loom"
_GSE_PATTERN = re.compile(r"(GSE\d+)", re.IGNORECASE)


def extract_geo_accession(file_path: str) -> Optional[str]:
    """Extract GSE accession from a filename if present.

    Args:
        file_path: Path or filename (e.g., "GSE162183_Skin.loom")

    Returns:
        GSE accession string or None
    """
    name = Path(file_path).name
    match = _GSE_PATTERN.search(name)
    return match.group(1).upper() if match else None


def fetch_soft_sample_metadata(
    geo_id: str, cache_dir: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Fetch sample-level metadata from GEO SOFT file.

    Parses characteristics_ch1 "key: value" pairs and sample titles.

    Args:
        geo_id: GEO accession (e.g., "GSE162183")
        cache_dir: Directory for caching SOFT files (uses auto-cleaned tempdir if None)

    Returns:
        Tuple of (parsed_samples, gsm_titles):
        - parsed_samples: {"GSM1234": {"tissue": "skin", "disease_state": "psoriasis"}}
        - gsm_titles: {"GSM1234": "skin control1 10x scrna-seq"}
    """
    try:
        import GEOparse
    except ImportError:
        logger.warning(
            "GEOparse not installed — cannot fetch SOFT metadata for Loom enrichment"
        )
        return {}, {}

    # Use TemporaryDirectory with auto-cleanup when no cache_dir provided
    _tmpdir_ctx = None
    if cache_dir:
        dest = cache_dir
    else:
        _tmpdir_ctx = tempfile.TemporaryDirectory(prefix="lobster_soft_")
        dest = _tmpdir_ctx.name

    try:
        # Try HTTPS pre-download first (more reliable than FTP)
        try:
            from lobster.services.data_access.geo.soft_download import (
                pre_download_soft_file,
            )

            pre_download_soft_file(geo_id, Path(dest))
        except Exception:
            pass  # GEOparse will try its own FTP fallback

        gse = GEOparse.get_GEO(geo=geo_id, destdir=str(dest), silent=True)
    except Exception as e:
        logger.warning(f"Failed to fetch SOFT metadata for {geo_id}: {e}")
        if _tmpdir_ctx:
            _tmpdir_ctx.cleanup()
        return {}, {}

    parsed_samples = {}
    gsm_titles = {}
    for gsm_name, gsm in gse.gsms.items():
        # Extract title
        titles = gsm.metadata.get("title", [])
        gsm_titles[gsm_name] = titles[0].lower() if titles else ""

        # Parse characteristics
        chars = gsm.metadata.get("characteristics_ch1", [])
        if not isinstance(chars, list):
            continue
        parsed = {}
        for char_str in chars:
            char_str = str(char_str)
            if ": " in char_str:
                key, value = char_str.split(": ", 1)
                key = key.strip().lower().replace(" ", "_").replace("-", "_")
                value = value.strip()
                if value:
                    parsed[key] = value
        if parsed:
            parsed_samples[gsm_name] = parsed

    if _tmpdir_ctx:
        _tmpdir_ctx.cleanup()

    if parsed_samples:
        logger.info(
            f"Fetched SOFT metadata for {geo_id}: "
            f"{len(parsed_samples)} samples, "
            f"fields: {sorted(set().union(*(p.keys() for p in parsed_samples.values())))}"
        )

    return parsed_samples, gsm_titles


# Common abbreviation mappings in scRNA-seq datasets
_ABBREVIATIONS = {
    "ctrl": "control",
    "ctr": "control",
    "ctl": "control",
    "psor": "psoriasis",
    "pso": "psoriasis",
    "norm": "normal",
    "nml": "normal",
    "tum": "tumor",
    "tumr": "tumor",
    "hlth": "healthy",
    "hlthy": "healthy",
    "dis": "disease",
    "pt": "patient",
    "samp": "sample",
    "exp": "experiment",
    "stim": "stimulated",
    "unstim": "unstimulated",
    "treat": "treated",
    "untreat": "untreated",
}


def _fuzzy_prefix_match(a: str, b: str) -> bool:
    """Check if two category identifiers likely refer to the same thing.

    Handles common abbreviations in scRNA-seq: ctrl~control, psor~psoriasis, etc.
    """
    if not a or not b:
        return False
    a, b = a.lower().strip(), b.lower().strip()

    if a == b:
        return True
    if a.startswith(b) or b.startswith(a):
        return True
    if len(a) >= 3 and len(b) >= 3 and a[:3] == b[:3]:
        return True

    a_expanded = _ABBREVIATIONS.get(a, a)
    b_expanded = _ABBREVIATIONS.get(b, b)
    if a_expanded == b_expanded:
        return True
    if a_expanded.startswith(b) or b_expanded.startswith(a):
        return True
    if b_expanded.startswith(a) or a_expanded.startswith(b):
        return True

    return False


def _try_barcode_prefix_mapping(
    adata,
    sample_metadata: Dict[str, Dict[str, str]],
    geo_id: str,
    gsm_titles: Optional[Dict[str, str]] = None,
) -> Dict[str, list]:
    """Try to map SOFT samples to cells via barcode prefixes.

    Many single-cell Loom files encode sample identity in the barcode prefix,
    e.g., "Ctrl1_10x_ACGTACGT" or "Patient1_ACGTACGT". This strategy:

    1. Extracts unique prefixes from obs_names (vectorized pandas split)
    2. Tries to match each prefix to a SOFT sample title using fuzzy substring matching
    3. Falls back to matching via existing obs columns (e.g., 'Patient')

    Args:
        adata: AnnData object
        sample_metadata: Pre-fetched SOFT metadata
        geo_id: GEO accession
        gsm_titles: Pre-fetched sample titles (avoids double GEOparse fetch)

    Returns:
        Dict mapping GSM IDs to lists of matching obs_names, or empty dict.
    """
    if gsm_titles is None:
        gsm_titles = {gsm_id: "" for gsm_id in sample_metadata}

    # Step 1: Vectorized barcode prefix extraction
    obs_index = adata.obs_names.to_series().astype(str)
    # Split on last underscore, keep prefix if suffix looks like a barcode (>=8 chars)
    split_result = obs_index.str.rsplit("_", n=1)
    prefixes = split_result.apply(
        lambda parts: (
            parts[0]
            if len(parts) == 2 and len(parts[1]) >= 8
            else parts[0] if len(parts) == 1 else "_".join(parts)
        )
    )

    prefix_groups = prefixes.groupby(prefixes).groups
    if not prefix_groups or len(prefix_groups) > 100:
        return {}

    # Build prefix→cells mapping
    prefix_to_cells = {
        prefix: obs_index.loc[idx].tolist() for prefix, idx in prefix_groups.items()
    }

    # Step 2: Build prefix → GSM mapping via title substring matching
    prefix_lower = {p.lower(): p for p in prefix_to_cells}
    gsm_to_rows = {}

    for gsm_id, title in gsm_titles.items():
        if gsm_id not in sample_metadata:
            continue
        for p_lower, p_orig in prefix_lower.items():
            p_parts = p_orig.split("_")
            p_ident = p_parts[0].lower()
            p_num = re.search(r"(\d+)$", p_ident)
            if not p_num:
                continue
            num = p_num.group(1)
            t_words = title.replace("-", " ").split()
            for word in t_words:
                w_num = re.search(r"(\d+)$", word)
                if w_num and w_num.group(1) == num:
                    p_alpha = re.sub(r"\d+$", "", p_ident)
                    w_alpha = re.sub(r"\d+$", "", word)
                    if _fuzzy_prefix_match(p_alpha, w_alpha):
                        gsm_to_rows[gsm_id] = prefix_to_cells[p_orig]
                        break
            if gsm_id in gsm_to_rows:
                break

    if gsm_to_rows:
        total_mapped = sum(len(rows) for rows in gsm_to_rows.values())
        logger.info(
            f"Using barcode prefix mapping for {geo_id}: "
            f"{total_mapped}/{adata.n_obs} cells mapped to "
            f"{len(gsm_to_rows)}/{len(sample_metadata)} samples "
            f"via title matching"
        )
        return gsm_to_rows

    # Step 3: Fallback — match via obs column with matching cardinality
    n_samples = len(sample_metadata)

    for col in adata.obs.columns:
        unique_vals = adata.obs[col].dropna().unique()
        if len(unique_vals) != n_samples:
            continue

        val_nums = {}
        for v in unique_vals:
            m = re.search(r"(\d+)$", str(v))
            if m:
                val_nums[int(m.group(1))] = v

        gsm_nums = {}
        for gsm_id in sample_metadata:
            title = gsm_titles.get(gsm_id, "")
            for word in title.replace("-", " ").split():
                m = re.search(r"(\d+)$", word)
                if m:
                    gsm_nums[int(m.group(1))] = gsm_id
                    break

        if val_nums and gsm_nums:
            for num, gsm_id in gsm_nums.items():
                for vnum, vname in val_nums.items():
                    if vnum != num:
                        continue
                    gsm_title = gsm_titles.get(gsm_id, "")
                    v_alpha = re.sub(r"\d+$", "", str(vname)).lower()
                    for word in gsm_title.replace("-", " ").split():
                        w_alpha = re.sub(r"\d+$", "", word)
                        if _fuzzy_prefix_match(v_alpha, w_alpha):
                            # Vectorized mask instead of per-value astype
                            col_str = adata.obs[col].astype(str)
                            mask = col_str == str(vname)
                            gsm_to_rows[gsm_id] = adata.obs_names[mask].tolist()
                            break
                    if gsm_id in gsm_to_rows:
                        break

        if gsm_to_rows and len(gsm_to_rows) == n_samples:
            total_mapped = sum(len(rows) for rows in gsm_to_rows.values())
            logger.info(
                f"Using obs column '{col}' mapping for {geo_id}: "
                f"{total_mapped}/{adata.n_obs} cells mapped to "
                f"{len(gsm_to_rows)}/{n_samples} samples"
            )
            return gsm_to_rows
        gsm_to_rows = {}

    return {}


def enrich_loom_adata_with_geo_metadata(
    adata,
    geo_id: str,
    sample_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    gsm_titles: Optional[Dict[str, str]] = None,
    cache_dir: Optional[str] = None,
) -> bool:
    """Enrich a Loom-loaded AnnData with GEO SOFT sample metadata.

    Attempts to map GSM sample metadata to cells using multiple strategies:
    1. Match by sample_id column in obs
    2. Match via barcode prefix to sample title (vectorized)
    3. Positional mapping (if sample count == obs count — bulk case)
    4. Barcode suffix mapping for single-cell (barcode-N maps to Nth sample)

    Args:
        adata: AnnData object to enrich (mutated in-place)
        geo_id: GEO accession for SOFT fetch
        sample_metadata: Pre-fetched metadata (if None, fetches from GEO)
        gsm_titles: Pre-fetched sample titles (avoids double fetch)
        cache_dir: Cache directory for SOFT files

    Returns:
        True if any metadata was injected, False otherwise
    """
    if sample_metadata is None:
        sample_metadata, gsm_titles = fetch_soft_sample_metadata(geo_id, cache_dir)

    if not sample_metadata:
        logger.warning(
            f"No sample metadata available for {geo_id}. "
            f"Condition-based analyses (DE, etc.) will require manual metadata."
        )
        return False

    if gsm_titles is None:
        gsm_titles = {gsm_id: "" for gsm_id in sample_metadata}

    # Collect all metadata keys across samples
    all_keys = set()
    for parsed in sample_metadata.values():
        all_keys.update(parsed.keys())

    # Skip keys that already exist in obs
    existing_cols = set(adata.obs.columns)
    keys_to_inject = all_keys - existing_cols
    if not keys_to_inject:
        logger.info(f"All SOFT metadata keys already present in obs for {geo_id}")
        return True

    # Strategy 1: Match by sample_id column
    gsm_to_rows = {}
    has_sample_id_col = "sample_id" in adata.obs.columns
    obs_names_upper = {str(name).upper(): name for name in adata.obs_names}

    for gsm_id in sample_metadata:
        gsm_upper = gsm_id.upper()
        if has_sample_id_col:
            mask = adata.obs["sample_id"].astype(str).str.upper() == gsm_upper
            matching_rows = adata.obs_names[mask].tolist()
            if matching_rows:
                gsm_to_rows[gsm_id] = matching_rows
        if gsm_id not in gsm_to_rows:
            if gsm_upper in obs_names_upper:
                gsm_to_rows[gsm_id] = [obs_names_upper[gsm_upper]]

    # Strategy 2: Barcode prefix mapping (vectorized, no double fetch)
    if not gsm_to_rows and adata.n_obs > len(sample_metadata):
        gsm_to_rows = _try_barcode_prefix_mapping(
            adata, sample_metadata, geo_id, gsm_titles=gsm_titles
        )

    # Strategy 3: Positional mapping (bulk: N samples == N obs)
    if not gsm_to_rows and len(sample_metadata) == adata.n_obs:
        gsm_ids_ordered = list(sample_metadata.keys())
        obs_names_ordered = adata.obs_names.tolist()
        for gsm_id, obs_name in zip(gsm_ids_ordered, obs_names_ordered):
            gsm_to_rows[gsm_id] = [obs_name]
        logger.info(
            f"Using positional mapping for {geo_id}: "
            f"{len(gsm_to_rows)} samples matched by order"
        )

    # Strategy 4: Barcode suffix mapping (vectorized)
    if not gsm_to_rows and adata.n_obs > len(sample_metadata):
        gsm_ids_ordered = list(sample_metadata.keys())
        obs_str = adata.obs_names.to_series().astype(str)
        suffix_match = obs_str.str.extract(r"-(\d+)$")
        if suffix_match[0].notna().any():
            suffix_match[0] = (
                pd.to_numeric(suffix_match[0], errors="coerce") - 1
            )  # 0-based
            suffix_match = suffix_match.dropna()
            max_suffix = int(suffix_match[0].max()) if len(suffix_match) > 0 else -1
            if max_suffix >= 0 and max_suffix < len(gsm_ids_ordered):
                for idx, gsm_id in enumerate(gsm_ids_ordered):
                    cell_mask = suffix_match[0] == idx
                    cells = suffix_match.index[cell_mask].tolist()
                    if cells:
                        gsm_to_rows[gsm_id] = cells
                if gsm_to_rows:
                    total_mapped = sum(len(rows) for rows in gsm_to_rows.values())
                    logger.info(
                        f"Using barcode suffix mapping for {geo_id}: "
                        f"{total_mapped}/{adata.n_obs} cells mapped to "
                        f"{len(gsm_to_rows)} samples"
                    )

    if not gsm_to_rows:
        logger.warning(
            f"Could not map GSM sample IDs to observations for {geo_id}. "
            f"GSM IDs: {list(sample_metadata.keys())[:5]}, "
            f"obs_names sample: {adata.obs_names[:5].tolist()}. "
            f"Manual metadata injection required for condition-based analyses."
        )
        return False

    # Report mapping coverage
    total_mapped = sum(len(rows) for rows in gsm_to_rows.values())
    coverage = total_mapped / adata.n_obs if adata.n_obs > 0 else 0
    if coverage < 1.0:
        logger.warning(
            f"Partial metadata mapping for {geo_id}: "
            f"{total_mapped}/{adata.n_obs} cells ({coverage:.1%}), "
            f"{len(gsm_to_rows)}/{len(sample_metadata)} samples. "
            f"Unmapped cells will have None values."
        )

    # Inject metadata columns
    for key in keys_to_inject:
        adata.obs[key] = pd.Series(
            [None] * adata.n_obs, index=adata.obs_names, dtype=object
        )

    injected_count = 0
    for gsm_id, rows in gsm_to_rows.items():
        parsed = sample_metadata[gsm_id]
        for key in keys_to_inject:
            if key in parsed:
                adata.obs.loc[rows, key] = parsed[key]
        injected_count += 1

    # Try numeric conversion where appropriate
    for key in keys_to_inject:
        col = adata.obs[key]
        non_null = col.dropna()
        if len(non_null) > 0:
            try:
                converted = pd.to_numeric(non_null, errors="coerce")
                if converted.notna().sum() / len(non_null) > 0.8:
                    adata.obs[key] = pd.to_numeric(adata.obs[key], errors="coerce")
            except (ValueError, TypeError):
                pass

    logger.info(
        f"Enriched Loom AnnData for {geo_id}: "
        f"{len(keys_to_inject)} columns ({', '.join(sorted(keys_to_inject))}), "
        f"{injected_count}/{len(sample_metadata)} samples mapped, "
        f"{coverage:.0%} cell coverage"
    )
    return True
