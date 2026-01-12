"""
Configuration and helper functions for metadata assistant agent.

This module contains:
- Log suppression utilities for Rich progress UI
- Metadata pattern detection (namespace separation)
- List-to-dict conversion utilities
- 4-phase disease extraction hierarchy:
  - Phase 1: Column re-scan for disease fields
  - Phase 2: LLM abstract extraction
  - Phase 3: LLM methods extraction
  - Phase 4: Manual mappings (hardcoded fallbacks)
"""

import json
import logging
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from lobster.utils.logger import get_logger

# Graceful fallback: Import premium DiseaseOntologyService if available
try:
    from lobster.services.metadata.disease_ontology_service import (
        DiseaseOntologyService,
    )
    HAS_ONTOLOGY_SERVICE = True
except ImportError:
    HAS_ONTOLOGY_SERVICE = False

logger = get_logger(__name__)

__all__ = [
    "suppress_logs",
    "detect_metadata_pattern",
    "convert_list_to_dict",
    "phase1_column_rescan",
    "extract_disease_with_llm",
    "phase2_llm_abstract_extraction",
    "phase3_llm_methods_extraction",
    "phase4_manual_mappings",
]


# =========================================================================
# Helper: Log Suppression for Progress UI
# =========================================================================


@contextmanager
def suppress_logs(min_level: int = logging.CRITICAL + 1):
    """
    Temporarily suppress logs during Rich progress display.

    This prevents log messages from interleaving with and disrupting the
    Rich progress bars. By default, ALL logs are suppressed (CRITICAL+1).

    Args:
        min_level: Minimum log level to show (default: CRITICAL+1 = suppress all).
                   Use logging.ERROR to show errors, logging.WARNING for warnings, etc.
    """
    original_levels = {}

    # Suppress root loggers (child loggers inherit via propagation)
    loggers_to_suppress = ["lobster", "urllib3", "httpx", "httpcore", "filelock"]
    for name in loggers_to_suppress:
        log = logging.getLogger(name)
        original_levels[name] = log.level
        log.setLevel(min_level)

    # Also capture all existing lobster.* child loggers for safety
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("lobster.") or logger_name.startswith("lobster_custom_"):
            log = logging.getLogger(logger_name)
            if logger_name not in original_levels:
                original_levels[logger_name] = log.level
                log.setLevel(min_level)

    try:
        yield
    finally:
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


# =========================================================================
# Helper: Metadata Pattern Detection (Option D - Namespace Separation)
# =========================================================================


def detect_metadata_pattern(data: dict) -> str:
    """Detect metadata storage pattern.

    Two patterns exist in metadata_store:
    - GEO pattern: {"metadata": {"samples": {...}}} - dict-of-dicts for single-dataset lookups
    - Aggregated pattern: {"samples": [...]} - list for batch processing/CSV export

    Key naming conventions:
    - geo_*, sra_*, metadata_* → Should use GEO pattern
    - aggregated_*, pub_queue_* → Should use Aggregated pattern

    Returns:
        "geo": GEO pattern {"metadata": {"samples": {...}}}
        "aggregated": Aggregated pattern {"samples": [...]}
        "unknown": Unrecognized pattern
    """
    if "metadata" in data and isinstance(data.get("metadata", {}).get("samples"), dict):
        return "geo"
    elif "samples" in data and isinstance(data["samples"], list):
        return "aggregated"
    return "unknown"


def convert_list_to_dict(samples_list: list, key_field: str = "run_accession") -> dict:
    """Convert list of samples to dict keyed by specified field.

    Used when reading aggregated pattern data with tools that expect dict-of-dicts.

    Args:
        samples_list: List of sample dicts
        key_field: Field to use as key (default: run_accession)

    Returns:
        Dict mapping sample IDs to sample data
    """
    if not samples_list:
        return {}

    # Fallback if key_field not present in first sample
    if samples_list and key_field not in samples_list[0]:
        key_field = next(iter(samples_list[0].keys()), "index")

    return {
        s.get(key_field, f"sample_{i}"): s
        for i, s in enumerate(samples_list)
    }


# =========================================================================
# Disease Enrichment Helpers (Phase 1-4)
# =========================================================================

DISEASE_EXTRACTION_PROMPT_TEMPLATE = """You are a biomedical information extraction expert analyzing a microbiome study.

PUBLICATION TITLE: {publication_title}
SAMPLE COUNT: {sample_count} samples

{text_content}

TASK: Identify the PRIMARY disease or condition studied in these samples.

CRITICAL RULES:
1. Focus on the actual condition of the SAMPLES, not what's being compared or referenced
2. If multiple conditions exist (e.g., CRC patients vs healthy controls), identify the DISEASE condition (CRC)
3. If study compares multiple diseases EQUALLY (e.g., UC vs CD comparison study), return "unknown" (mixed study)
4. If samples are from healthy controls ONLY, return "healthy"
5. Look for explicit statements: "patients with X", "X cohort", "diagnosed with X"

STANDARD TERMS (use EXACTLY as shown):
- "crc" → colorectal cancer, colon cancer, rectal cancer, CRC, colorectal carcinoma
- "uc" → ulcerative colitis, UC, colitis ulcerosa
- "cd" → Crohn's disease, Crohn's, CD, Crohn disease
- "healthy" → healthy controls, healthy volunteers, non-diseased, control subjects
- "unknown" → cannot determine, mixed conditions, insufficient information

OUTPUT FORMAT (JSON only, no markdown):
{{
  "disease": "one of: crc, uc, cd, healthy, unknown",
  "confidence": 0.0-1.0 (float),
  "evidence": "Brief quote or description supporting classification (max 200 chars)",
  "reasoning": "Explain why you chose this term (max 300 chars)"
}}

CONFIDENCE SCORING GUIDE:
- 1.0: Explicit statement ("50 CRC patients recruited")
- 0.9: Clear context ("colorectal cancer cohort study")
- 0.8: Strong inference ("tumor samples from colon cancer patients")
- 0.7: Weak inference ("IBD patients" without UC/CD specification)
- <0.7: Ambiguous or insufficient information

EXAMPLES:
Input: "We recruited 50 patients with colorectal cancer and 30 healthy controls"
Output: {{"disease": "crc", "confidence": 0.95, "evidence": "50 patients with colorectal cancer", "reasoning": "Explicit CRC patient recruitment"}}

Input: "Fecal samples from UC patients undergoing remission therapy"
Output: {{"disease": "uc", "confidence": 0.92, "evidence": "UC patients undergoing therapy", "reasoning": "Clear UC patient cohort"}}

Input: "Comparison of gut microbiome in UC, CD, and healthy controls"
Output: {{"disease": "unknown", "confidence": 0.0, "evidence": "Mixed study: UC, CD, healthy", "reasoning": "Three equal groups, no primary disease"}}

Input: "16S sequencing of fecal microbiota"
Output: {{"disease": "unknown", "confidence": 0.0, "evidence": "No disease information provided", "reasoning": "Methods only, no cohort description"}}

Return ONLY valid JSON, no additional text or formatting.
"""


def phase1_column_rescan(samples: List[Dict]) -> tuple:
    """
    Re-scan ALL columns in sample metadata for disease keywords.

    Premium version: Uses DiseaseOntologyService.match_disease() API (Phase 2 compatible)
    Public version: Uses hardcoded disease_keywords dict (fallback)

    Returns:
        Tuple of (enriched_count, detailed_log)
    """
    enriched_count = 0
    log = []

    if HAS_ONTOLOGY_SERVICE:
        # Premium version: Use centralized ontology service
        # (Phase 1: keywords, Phase 2: embeddings)
        ontology = DiseaseOntologyService.get_instance()

        for sample in samples:
            if sample.get('disease'):
                continue  # Already has disease, skip

            # Check ALL columns (not just standard ones)
            for col_name, col_value in sample.items():
                if not col_value:
                    continue

                col_str = str(col_value)

                # Use match_disease() API (Phase 2 compatible)
                matches = ontology.match_disease(col_str, k=1, min_confidence=0.7)
                if matches:
                    best_match = matches[0]
                    sample['disease'] = best_match.disease_id
                    sample['disease_original'] = col_str
                    sample['disease_source'] = f'column_remapped:{col_name}'
                    sample['disease_confidence'] = best_match.confidence
                    sample['disease_match_type'] = best_match.match_type
                    sample['enrichment_timestamp'] = datetime.now().isoformat()

                    enriched_count += 1
                    log.append(f"  - Sample {sample.get('run_accession', 'unknown')}: "
                             f"Found '{best_match.disease_id}' ({best_match.name}) "
                             f"in column '{col_name}' (value: {col_value})")
                    break  # Found disease, move to next sample
    else:
        # Public version (lobster-local): Use hardcoded fallback
        disease_keywords = {
            'crc': ['colorectal', 'colon_cancer', 'colon cancer', 'rectal_cancer', 'crc'],
            'uc': ['ulcerative', 'uc_', 'colitis_ulcerosa', 'ulcerative_colitis'],
            'cd': ['crohn', 'cd_', 'crohns', 'crohns_disease'],
            'healthy': ['healthy', 'control', 'non_ibd', 'non-ibd', 'non_diseased']
        }

        for sample in samples:
            if sample.get('disease'):
                continue  # Already has disease, skip

            # Check ALL columns (not just standard ones)
            for col_name, col_value in sample.items():
                if not col_value:
                    continue

                col_str = str(col_value).lower()

                # Try each disease mapping
                for disease, keywords in disease_keywords.items():
                    for keyword in keywords:
                        if keyword in col_str:
                            sample['disease'] = disease
                            sample['disease_original'] = str(col_value)
                            sample['disease_source'] = f'column_remapped:{col_name}'
                            sample['disease_confidence'] = 1.0
                            sample['enrichment_timestamp'] = datetime.now().isoformat()

                            enriched_count += 1
                            log.append(f"  - Sample {sample.get('run_accession', 'unknown')}: "
                                     f"Found '{disease}' in column '{col_name}' (value: {col_value})")
                            break

                    if sample.get('disease'):
                        break  # Found disease, move to next sample

                if sample.get('disease'):
                    break  # Found disease, move to next sample

    return enriched_count, log


def extract_disease_with_llm(
    llm: Any,
    abstract_text: Optional[str],
    methods_text: Optional[str],
    publication_title: str,
    sample_count: int
) -> Dict[str, Any]:
    """
    Extract disease from publication text using LLM with structured output.

    Args:
        llm: LLM instance (from self.llm or create_llm())
        abstract_text: Publication abstract (if available)
        methods_text: Methods section (if available)
        publication_title: Paper title for context
        sample_count: Number of samples in this publication

    Returns:
        Dict with:
        {
            "disease": "crc",
            "confidence": 0.92,
            "evidence": "Abstract states: recruited 50 CRC patients",
            "source": "abstract" or "methods"
        }
    """
    import re

    # Combine available text
    text_content = ""
    source = "none"

    if abstract_text:
        text_content = f"ABSTRACT:\n{abstract_text[:2000]}\n\n"
        source = "abstract"

    if methods_text:
        text_content += f"METHODS:\n{methods_text[:2000]}"
        if source == "none":
            source = "methods"
        else:
            source = "abstract+methods"

    if not text_content:
        return {"disease": "unknown", "confidence": 0.0, "evidence": "No text available", "source": "none"}

    # Create extraction prompt using template
    prompt = DISEASE_EXTRACTION_PROMPT_TEMPLATE.format(
        publication_title=publication_title,
        sample_count=sample_count,
        text_content=text_content
    )

    # Call LLM
    try:
        response = llm.invoke([
            {"role": "system", "content": "You are a biomedical information extraction expert. Return valid JSON only."},
            {"role": "user", "content": prompt}
        ])

        # Parse JSON response
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try direct JSON parse
            json_str = response_text.strip()

        result = json.loads(json_str)

        # Validate structure
        if 'disease' not in result or 'confidence' not in result:
            logger.warning(f"LLM returned invalid structure: {result}")
            return {"disease": "unknown", "confidence": 0.0, "evidence": "Invalid LLM response", "source": source}

        # Validate disease is in standard terms
        valid_diseases = ['crc', 'uc', 'cd', 'healthy', 'unknown']
        if result['disease'] not in valid_diseases:
            logger.warning(f"LLM returned invalid disease: {result['disease']}")
            return {"disease": "unknown", "confidence": 0.0, "evidence": "Invalid disease term", "source": source}

        result['source'] = source
        return result

    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return {"disease": "unknown", "confidence": 0.0, "evidence": f"Extraction error: {str(e)}", "source": source}


def phase2_llm_abstract_extraction(
    samples: List[Dict],
    data_manager: Any,
    confidence_threshold: float,
    llm: Any
) -> tuple:
    """
    Extract disease from publication abstracts using LLM.
    Groups samples by publication to avoid redundant LLM calls.

    Returns:
        Tuple of (enriched_count, detailed_log)
    """
    enriched_count = 0
    log = []

    # Group samples by publication
    pub_groups = defaultdict(list)
    for sample in samples:
        if not sample.get('disease'):
            pub_id = sample.get('publication_entry_id')
            if pub_id:
                pub_groups[pub_id].append(sample)

    log.append(f"  - Grouped samples by publication: {len(pub_groups)} publications to process")

    # Process each publication
    for pub_id, pub_samples in pub_groups.items():
        # Load cached abstract
        metadata_path = data_manager.workspace_path / "metadata" / f"{pub_id}_metadata.json"

        if not metadata_path.exists():
            log.append(f"  - {pub_id}: No cached metadata (skipping {len(pub_samples)} samples)")
            continue

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
                abstract_text = metadata.get("content", "")
        except (json.JSONDecodeError, IOError) as e:
            log.append(f"  - {pub_id}: Error reading metadata file ({e})")
            continue

        if not abstract_text:
            log.append(f"  - {pub_id}: Empty abstract (skipping {len(pub_samples)} samples)")
            continue

        # Extract disease using LLM
        disease_info = extract_disease_with_llm(
            llm=llm,
            abstract_text=abstract_text,
            methods_text=None,
            publication_title=pub_samples[0].get('publication_title', 'Unknown'),
            sample_count=len(pub_samples)
        )

        # Check confidence
        if disease_info['confidence'] < confidence_threshold:
            log.append(f"  - {pub_id}: Low confidence {disease_info['confidence']:.2f} "
                     f"(threshold: {confidence_threshold}, skipping {len(pub_samples)} samples)")
            continue

        # Apply to all samples from this publication
        for sample in pub_samples:
            sample['disease'] = disease_info['disease']
            sample['disease_original'] = disease_info.get('evidence', '')[:100]
            sample['disease_source'] = 'abstract_llm'
            sample['disease_confidence'] = disease_info['confidence']
            sample['disease_evidence'] = disease_info.get('evidence', '')[:200]
            sample['enrichment_timestamp'] = datetime.now().isoformat()
            enriched_count += 1

        log.append(f"  - {pub_id}: Extracted '{disease_info['disease']}' "
                 f"(confidence: {disease_info['confidence']:.2f}, "
                 f"enriched {len(pub_samples)} samples)")

    return enriched_count, log


def phase3_llm_methods_extraction(
    samples: List[Dict],
    data_manager: Any,
    confidence_threshold: float,
    llm: Any
) -> tuple:
    """
    Extract disease from publication methods sections using LLM.
    Only processes samples still missing disease after Phase 2.

    Returns:
        Tuple of (enriched_count, detailed_log)
    """
    enriched_count = 0
    log = []

    # Group remaining samples by publication
    pub_groups = defaultdict(list)
    for sample in samples:
        if not sample.get('disease'):
            pub_id = sample.get('publication_entry_id')
            if pub_id:
                pub_groups[pub_id].append(sample)

    if not pub_groups:
        return 0, ["  - No samples need methods extraction"]

    log.append(f"  - Processing methods for {len(pub_groups)} publications")

    # Process each publication
    for pub_id, pub_samples in pub_groups.items():
        # Load cached methods
        methods_path = data_manager.workspace_path / "metadata" / f"{pub_id}_methods.json"

        if not methods_path.exists():
            log.append(f"  - {pub_id}: No cached methods (skipping)")
            continue

        try:
            with open(methods_path) as f:
                methods_data = json.load(f)
                methods_text = methods_data.get("methods_text", "")
        except (json.JSONDecodeError, IOError) as e:
            log.append(f"  - {pub_id}: Error reading methods file ({e})")
            continue

        if not methods_text:
            log.append(f"  - {pub_id}: Empty methods section (skipping)")
            continue

        # Extract disease from methods
        disease_info = extract_disease_with_llm(
            llm=llm,
            abstract_text=None,
            methods_text=methods_text,
            publication_title=pub_samples[0].get('publication_title', 'Unknown'),
            sample_count=len(pub_samples)
        )

        if disease_info['confidence'] < confidence_threshold:
            log.append(f"  - {pub_id}: Low confidence {disease_info['confidence']:.2f} (skipping)")
            continue

        # Apply to samples
        for sample in pub_samples:
            sample['disease'] = disease_info['disease']
            sample['disease_original'] = disease_info.get('evidence', '')[:100]
            sample['disease_source'] = 'methods_llm'
            sample['disease_confidence'] = disease_info['confidence']
            sample['disease_evidence'] = disease_info.get('evidence', '')[:200]
            sample['enrichment_timestamp'] = datetime.now().isoformat()
            enriched_count += 1

        log.append(f"  - {pub_id}: Extracted '{disease_info['disease']}' from methods "
                 f"(confidence: {disease_info['confidence']:.2f}, "
                 f"enriched {len(pub_samples)} samples)")

    return enriched_count, log


def phase4_manual_mappings(
    samples: List[Dict],
    manual_mappings: Dict[str, str]
) -> tuple:
    """
    Apply user-provided disease mappings by publication ID.

    Args:
        samples: List of sample dicts
        manual_mappings: Dict mapping publication_entry_id to disease
            Example: {"pub_queue_doi_10_1234": "crc", "pub_queue_pmid_5678": "uc"}

    Returns:
        Tuple of (enriched_count, detailed_log)
    """
    if not manual_mappings:
        return 0, ["  - No manual mappings provided"]

    enriched_count = 0
    log = []

    for sample in samples:
        if sample.get('disease'):
            continue  # Already has disease

        pub_id = sample.get('publication_entry_id')
        if pub_id in manual_mappings:
            disease = manual_mappings[pub_id]

            # Validate disease is in standard terms
            if disease not in ['crc', 'uc', 'cd', 'healthy', 'unknown']:
                log.append(f"  - Warning: Invalid disease '{disease}' for {pub_id}, skipping")
                continue

            sample['disease'] = disease
            sample['disease_original'] = 'manual_mapping'
            sample['disease_source'] = 'manual_override'
            sample['disease_confidence'] = 1.0
            sample['enrichment_timestamp'] = datetime.now().isoformat()
            enriched_count += 1

    if enriched_count > 0:
        log.append(f"  - Applied manual mappings: {enriched_count} samples enriched")

    return enriched_count, log
