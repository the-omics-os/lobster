"""
Metadata Assistant Agent for Cross-Dataset Metadata Operations.

This agent specializes in sample ID mapping, metadata standardization, and
dataset content validation for multi-omics integration.

Phase 3 implementation for research agent refactoring.

Note: This agent replaces research_agent_assistant's metadata functionality.
The PDF resolution features were archived and will be migrated to research_agent
in Phase 4. See lobster/agents/archive/ARCHIVE_NOTICE.md for details.
"""

# =============================================================================
# AGENT_CONFIG must be defined FIRST (before heavy imports) for entry point loading
# This prevents circular import issues when component_registry loads this module
# =============================================================================
from lobster.config.agent_registry import AgentRegistryConfig

AGENT_CONFIG = AgentRegistryConfig(
    name="metadata_assistant",
    display_name="Metadata Assistant",
    description="Handles cross-dataset metadata operations including sample ID mapping (exact/fuzzy/pattern/metadata strategies), metadata standardization using Pydantic schemas (transcriptomics/proteomics), dataset completeness validation (samples, conditions, controls, duplicates, platform), sample metadata reading in multiple formats, and disease enrichment from publication context when SRA metadata is incomplete. Specialized in metadata harmonization for multi-omics integration and publication queue processing.",
    factory_function="lobster.agents.metadata_assistant.metadata_assistant",
    handoff_tool_name="handoff_to_metadata_assistant",
    handoff_tool_description="Assign metadata operations (cross-dataset sample mapping, metadata standardization to Pydantic schemas, dataset validation before download, metadata reading/formatting, publication queue filtering) to the metadata assistant",
)

# =============================================================================
# Heavy imports below (may have circular dependencies, but AGENT_CONFIG is already defined)
# =============================================================================
import json
import logging
import re
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from lobster.config.llm_factory import create_llm
from lobster.config.settings import get_settings
from lobster.core.analysis_ir import AnalysisStep
from lobster.core.data_manager_v2 import DataManagerV2
from lobster.core.interfaces.validator import ValidationResult

# Publication queue schemas - try local package first (for public lobster-ai with custom packages),
# then fall back to base lobster (for private lobster or premium installations)
try:
    from lobster_custom_databiomix.core.schemas.publication_queue import (
        HandoffStatus,
        PublicationStatus,
    )
except ImportError:
    from lobster.core.schemas.publication_queue import HandoffStatus, PublicationStatus

from lobster.services.execution.custom_code_execution_service import (
    CodeExecutionError,
    CodeValidationError,
    CustomCodeExecutionService,
)
from lobster.tools.custom_code_tool import (
    create_execute_custom_code_tool,
    metadata_store_post_processor,
)
from lobster.services.metadata.metadata_standardization_service import (
    MetadataStandardizationService,
)

# Sample mapping service - try local package first (for public lobster-ai with custom packages),
# then fall back to base lobster (for private lobster or premium installations)
try:
    from lobster_custom_databiomix.services.metadata.sample_mapping_service import (
        SampleMappingService,
    )
except ImportError:
    from lobster.services.metadata.sample_mapping_service import SampleMappingService

from lobster.utils.logger import get_logger

# Optional microbiome features (not in public lobster-local)
try:
    from lobster.services.metadata.disease_standardization_service import (
        DiseaseStandardizationService,
    )
    from lobster.services.metadata.microbiome_filtering_service import (
        MicrobiomeFilteringService,
    )
    from lobster.services.metadata.metadata_filtering_service import (
        MetadataFilteringService,
        extract_disease_with_fallback,
    )

    MICROBIOME_FEATURES_AVAILABLE = True
except ImportError:
    MicrobiomeFilteringService = None
    DiseaseStandardizationService = None
    MetadataFilteringService = None
    MICROBIOME_FEATURES_AVAILABLE = False

# Optional Rich UI for progress visualization
try:
    from lobster.ui.components.parallel_workers_progress import parallel_workers_progress
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import time
    PROGRESS_UI_AVAILABLE = True
except ImportError:
    parallel_workers_progress = None
    ThreadPoolExecutor = None
    PROGRESS_UI_AVAILABLE = False

logger = get_logger(__name__)


# =========================================================================
# Helper: Log Suppression for Progress UI
# =========================================================================


@contextmanager
def _suppress_logs(min_level: int = logging.CRITICAL + 1):
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


def _detect_metadata_pattern(data: dict) -> str:
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


def _convert_list_to_dict(samples_list: list, key_field: str = "run_accession") -> dict:
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
# Disease Enrichment Helpers (Phase 1)
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


def _phase1_column_rescan(samples: List[Dict]) -> tuple:
    """
    Re-scan ALL columns in sample metadata for disease keywords.

    Returns:
        Tuple of (enriched_count, detailed_log)
    """
    enriched_count = 0
    log = []

    # Disease keyword mappings (case-insensitive)
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


def _extract_disease_with_llm(
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


def _phase2_llm_abstract_extraction(
    samples: List[Dict],
    data_manager: DataManagerV2,
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
        disease_info = _extract_disease_with_llm(
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


def _phase3_llm_methods_extraction(
    samples: List[Dict],
    data_manager: DataManagerV2,
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
        disease_info = _extract_disease_with_llm(
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


def _phase4_manual_mappings(
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


def metadata_assistant(
    data_manager: DataManagerV2,
    callback_handler=None,
    agent_name: str = "metadata_assistant",
    delegation_tools: list = None,
    workspace_path: Optional[Path] = None,
):
    """Create metadata assistant agent for metadata operations.

    This agent provides 4-5 specialized tools for metadata operations:
    1. map_samples_by_id - Cross-dataset sample ID mapping
    2. read_sample_metadata - Extract and format sample metadata
    3. standardize_sample_metadata - Convert to Pydantic schemas
    4. validate_dataset_content - Validate dataset completeness
    5. filter_samples_by - Multi-criteria filtering (16S + host + sample_type + disease)
       [OPTIONAL - only available if microbiome features are installed]

    Args:
        data_manager: DataManagerV2 instance
        callback_handler: Optional callback handler for LLM
        agent_name: Agent name for identification
        delegation_tools: Optional list of delegation tools for sub-agent access
        workspace_path: Path to workspace directory for config resolution

    Returns:
        Compiled LangGraph agent with metadata tools
    """
    settings = get_settings()
    model_params = settings.get_agent_llm_params("metadata_assistant")
    llm = create_llm("metadata_assistant", model_params, workspace_path=workspace_path)

    # Normalize callbacks to a flat list (fix double-nesting bug)
    if callback_handler and hasattr(llm, "with_config"):
        callbacks = callback_handler if isinstance(callback_handler, list) else [callback_handler]
        llm = llm.with_config(callbacks=callbacks)

    # Initialize services (Phase 3: new services)
    sample_mapping_service = SampleMappingService(data_manager=data_manager)
    metadata_standardization_service = MetadataStandardizationService(
        data_manager=data_manager
    )

    # Initialize optional microbiome services if available
    microbiome_filtering_service = None
    disease_standardization_service = None
    metadata_filtering_service = None
    if MICROBIOME_FEATURES_AVAILABLE:
        microbiome_filtering_service = MicrobiomeFilteringService()
        disease_standardization_service = DiseaseStandardizationService()
        # Create filtering service with dependencies (disease_extractor set later)
        metadata_filtering_service = MetadataFilteringService(
            microbiome_service=microbiome_filtering_service,
            disease_service=disease_standardization_service,
        )
        logger.debug("Microbiome features enabled")
    else:
        logger.debug("Microbiome features not available (optional)")

    # Custom code execution service for sample-level operations
    custom_code_service = CustomCodeExecutionService(data_manager)

    # Create shared execute_custom_code tool via factory (v2.7+: unified tool)
    execute_custom_code = create_execute_custom_code_tool(
        data_manager=data_manager,
        custom_code_service=custom_code_service,
        agent_name="metadata_assistant",
        post_processor=metadata_store_post_processor,
    )

    logger.debug("metadata_assistant agent initialized")

    # Import pandas at factory function level for type hints in helper functions
    import pandas as pd

    # =========================================================================
    # Tool 1: Sample ID Mapping
    # =========================================================================

    @tool
    def map_samples_by_id(
        source: str,
        target: str,
        source_type: str,
        target_type: str,
        min_confidence: float = 0.75,
        strategies: str = "all",
    ) -> str:
        """
        Map sample IDs between two datasets for multi-omics integration.

        Use this tool when you need to harmonize sample identifiers across datasets
        with different naming conventions. The service uses multiple matching strategies:
        - Exact: Case-insensitive exact matching
        - Fuzzy: RapidFuzz-based similarity matching (requires RapidFuzz)
        - Pattern: Common prefix/suffix removal (Sample_, GSM, _Rep1, etc.)
        - Metadata: Metadata-supported matching (condition, tissue, timepoint, etc.)

        Args:
            source: Source modality name or dataset ID
            target: Target modality name or dataset ID
            source_type: Source data type - REQUIRED, must be "modality" or "metadata_store"
                - "modality": Work with loaded AnnData in DataManagerV2
                - "metadata_store": Work with cached metadata (pre-download validation)
            target_type: Target data type - REQUIRED, must be "modality" or "metadata_store"
            min_confidence: Minimum confidence threshold for fuzzy matches (0.0-1.0, default: 0.75)
            strategies: Comma-separated strategies to use (default: "all", options: "exact,fuzzy,pattern,metadata")

        Returns:
            Formatted markdown report with match results, confidence scores, and unmapped samples

        Examples:
            # Map between two loaded modalities
            map_samples_by_id(source="geo_gse1", target="geo_gse2",
                            source_type="modality", target_type="modality")

            # Map between cached metadata (pre-download)
            map_samples_by_id(source="geo_gse1", target="geo_gse2",
                            source_type="metadata_store", target_type="metadata_store")

            # Mixed: modality to cached metadata
            map_samples_by_id(source="geo_gse1", target="geo_gse2",
                            source_type="modality", target_type="metadata_store")
        """
        try:
            logger.info(
                f"Mapping samples: {source} → {target} "
                f"(source_type={source_type}, target_type={target_type}, min_confidence={min_confidence})"
            )

            # Validate source_type and target_type
            for stype, name in [
                (source_type, "source_type"),
                (target_type, "target_type"),
            ]:
                if stype not in ["modality", "metadata_store"]:
                    return f"❌ Error: {name} must be 'modality' or 'metadata_store', got '{stype}'"

            # Parse strategies
            strategy_list = None
            if strategies and strategies.lower() != "all":
                strategy_list = [s.strip().lower() for s in strategies.split(",")]
                # Validate strategies
                valid_strategies = {"exact", "fuzzy", "pattern", "metadata"}
                invalid = set(strategy_list) - valid_strategies
                if invalid:
                    return (
                        f"❌ Invalid strategies: {invalid}. "
                        f"Valid options: {valid_strategies}"
                    )

            # Helper function to get samples based on type
            import pandas as pd

            def get_samples(identifier: str, id_type: str) -> pd.DataFrame:
                if id_type == "modality":
                    if identifier not in data_manager.list_modalities():
                        raise ValueError(
                            f"Modality '{identifier}' not found. Available: {', '.join(data_manager.list_modalities())}"
                        )
                    adata = data_manager.get_modality(identifier)
                    return adata.obs  # Returns sample metadata DataFrame

                elif id_type == "metadata_store":
                    if identifier not in data_manager.metadata_store:
                        raise ValueError(
                            f"'{identifier}' not found in metadata_store. Use research_agent.validate_dataset_metadata() first."
                        )
                    cached = data_manager.metadata_store[identifier]

                    # Detect pattern (Option D: Namespace Separation)
                    pattern = _detect_metadata_pattern(cached)

                    if pattern == "geo":
                        samples_dict = cached["metadata"]["samples"]
                    elif pattern == "aggregated":
                        samples_dict = _convert_list_to_dict(cached["samples"])
                    else:
                        raise ValueError(
                            f"Unrecognized metadata pattern in '{identifier}'. "
                            "Expected 'metadata.samples' dict or 'samples' list."
                        )

                    if not samples_dict:
                        raise ValueError(f"No sample metadata in '{identifier}'")
                    return pd.DataFrame.from_dict(samples_dict, orient="index")

            # Get samples from both sources
            get_samples(source, source_type)
            get_samples(target, target_type)

            # Call mapping service (updated to work with DataFrames directly)
            result = sample_mapping_service.map_samples_by_id(
                source_identifier=source,
                target_identifier=target,
                strategies=strategy_list,
            )

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="map_samples_by_id",
                parameters={
                    "source": source,
                    "target": target,
                    "source_type": source_type,
                    "target_type": target_type,
                    "min_confidence": min_confidence,
                    "strategies": strategies,
                },
                description=f"Mapped {result.summary['exact_matches']} exact, {result.summary['fuzzy_matches']} fuzzy, {result.summary['unmapped']} unmapped ({result.summary['mapping_rate']:.1%} rate)",
            )

            # Format report
            report = sample_mapping_service.format_mapping_report(result)

            logger.info(
                f"Mapping complete: {result.summary['mapping_rate']:.1%} success rate"
            )
            return report

        except ValueError as e:
            logger.error(f"Mapping error: {e}")
            return f"❌ Mapping failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected mapping error: {e}", exc_info=True)
            return f"❌ Unexpected error during mapping: {str(e)}"

    # =========================================================================
    # Tool 2: Read Sample Metadata
    # =========================================================================

    @tool
    def read_sample_metadata(
        source: str,
        source_type: str,
        fields: str = None,
        return_format: str = "summary",
    ) -> str:
        """
        Read and format sample-level metadata from loaded modality OR cached metadata.

        Use this tool to extract sample metadata in different formats:
        - "summary": Quick overview with field coverage percentages
        - "detailed": Complete metadata as JSON for programmatic access
        - "schema": Full metadata table for inspection

        Args:
            source: Modality name or dataset ID
            source_type: Data source type - REQUIRED, must be "modality" or "metadata_store"
                - "modality": Work with loaded AnnData in DataManagerV2
                - "metadata_store": Work with cached metadata (pre-download validation)
            fields: Optional comma-separated list of fields to extract (default: all fields)
            return_format: Output format (default: "summary", options: "summary,detailed,schema")

        Returns:
            Formatted metadata according to return_format specification

        Examples:
            # Read from loaded modality
            read_sample_metadata(source="geo_gse180759", source_type="modality")

            # Read from cached metadata (pre-download)
            read_sample_metadata(source="geo_gse180759", source_type="metadata_store")
        """
        try:
            logger.info(
                f"Reading metadata for {source} (source_type={source_type}, format: {return_format})"
            )

            # Validate source_type
            if source_type not in ["modality", "metadata_store"]:
                return f"❌ Error: source_type must be 'modality' or 'metadata_store', got '{source_type}'"

            # Parse fields
            field_list = None
            if fields:
                field_list = [f.strip() for f in fields.split(",")]

            # Get sample metadata based on source_type
            import pandas as pd

            if source_type == "modality":
                if source not in data_manager.list_modalities():
                    return f"❌ Error: Modality '{source}' not found. Available: {', '.join(data_manager.list_modalities())}"
                adata = data_manager.get_modality(source)
                sample_df = adata.obs
            elif source_type == "metadata_store":
                # Two-tier cache pattern: Check memory first, then workspace files
                cached = None

                # Tier 1: Check in-memory metadata_store (fast path)
                if source in data_manager.metadata_store:
                    cached = data_manager.metadata_store[source]
                    logger.debug(f"Found '{source}' in metadata_store (Tier 1 - memory)")

                # Tier 2: Fallback to workspace files (persistent, survives session restart)
                else:
                    logger.debug(f"'{source}' not in metadata_store, checking workspace files (Tier 2)")
                    try:
                        from lobster.services.data_access.workspace_content_service import (
                            WorkspaceContentService,
                            ContentType,
                        )

                        workspace_service = WorkspaceContentService(data_manager)
                        cached = workspace_service.read_content(
                            identifier=source,
                            content_type=ContentType.METADATA,
                            level=None,  # Full content
                        )

                        # Lazy loading: Promote to metadata_store for subsequent fast access
                        data_manager.metadata_store[source] = cached
                        logger.info(f"Loaded '{source}' from workspace and promoted to metadata_store")

                    except FileNotFoundError:
                        return (
                            f"❌ Error: '{source}' not found in metadata_store or workspace files. "
                            f"Use research_agent.validate_dataset_metadata() or process_metadata_queue first."
                        )

                # Detect pattern (Option D: Namespace Separation)
                pattern = _detect_metadata_pattern(cached)

                if pattern == "geo":
                    # GEO pattern: {"metadata": {"samples": {...}}}
                    samples_dict = cached["metadata"]["samples"]
                elif pattern == "aggregated":
                    # Aggregated pattern: {"samples": [...]} - convert list to dict
                    samples_dict = _convert_list_to_dict(cached["samples"])
                    logger.debug(f"Converted aggregated pattern ({len(cached['samples'])} samples) to dict for '{source}'")
                else:
                    return f"❌ Error: Unrecognized metadata pattern in '{source}'. Expected 'metadata.samples' dict or 'samples' list."

                if not samples_dict:
                    return f"❌ Error: No sample metadata in '{source}'."

                sample_df = pd.DataFrame.from_dict(samples_dict, orient="index")

            # Filter fields if specified
            if field_list:
                available_fields = list(sample_df.columns)
                missing_fields = [f for f in field_list if f not in available_fields]
                if missing_fields:
                    return f"❌ Error: Fields not found: {', '.join(missing_fields)}"
                sample_df = sample_df[field_list]

            # Log provenance
            data_manager.log_tool_usage(
                tool_name="read_sample_metadata",
                parameters={
                    "source": source,
                    "source_type": source_type,
                    "fields": fields,
                    "return_format": return_format,
                },
                description=f"Read {len(sample_df)} samples in {return_format} format",
            )

            # Format output based on return_format
            if return_format == "summary":
                logger.info(f"Metadata summary generated for {source}")
                # Generate summary
                summary = [
                    "# Sample Metadata Summary\n",
                    f"**Dataset**: {source}",
                    f"**Source Type**: {source_type}",
                    f"**Total Samples**: {len(sample_df)}\n",
                    "## Field Coverage:",
                ]
                for col in sample_df.columns:
                    non_null = sample_df[col].notna().sum()
                    pct = (non_null / len(sample_df)) * 100
                    summary.append(f"- {col}: {pct:.1f}% ({non_null}/{len(sample_df)})")
                return "\n".join(summary)
            elif return_format == "detailed":
                logger.info(f"Detailed metadata extracted for {source}")
                return json.dumps(sample_df.to_dict(orient="records"), indent=2)
            elif return_format == "schema":
                logger.info(f"Metadata schema extracted for {source}")
                return sample_df.to_markdown(index=True)
            else:
                return f"❌ Invalid return_format '{return_format}'"

        except ValueError as e:
            logger.error(f"Metadata read error: {e}")
            return f"❌ Failed to read metadata: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected metadata read error: {e}", exc_info=True)
            return f"❌ Unexpected error reading metadata: {str(e)}"

    # =========================================================================
    # Tool 3: Standardize Sample Metadata
    # =========================================================================

    @tool
    def standardize_sample_metadata(
        source: str,
        source_type: str,
        target_schema: str,
        controlled_vocabularies: str = None,
    ) -> str:
        """
        Standardize sample metadata using Pydantic schemas for cross-dataset harmonization.

        Use this tool to convert raw metadata to standardized Pydantic schemas
        (TranscriptomicsMetadataSchema or ProteomicsMetadataSchema) with field
        normalization and controlled vocabulary enforcement.

        Args:
            source: Modality name or dataset ID
            source_type: Data source type - REQUIRED, must be "modality" or "metadata_store"
                - "modality": Work with loaded AnnData in DataManagerV2
                - "metadata_store": Work with cached metadata (pre-download validation)
            target_schema: Target schema type (options: "transcriptomics", "proteomics", "bulk_rna_seq", "single_cell", "mass_spectrometry", "affinity")
            controlled_vocabularies: Optional JSON string of controlled vocabularies (e.g., '{"condition": ["Control", "Treatment"]}')

        Returns:
            Standardization report with field coverage, validation errors, and warnings

        Examples:
            # Standardize from loaded modality
            standardize_sample_metadata(source="geo_gse12345", source_type="modality",
                                       target_schema="transcriptomics")

            # Standardize from cached metadata
            standardize_sample_metadata(source="geo_gse12345", source_type="metadata_store",
                                       target_schema="transcriptomics")
        """
        try:
            logger.info(
                f"Standardizing metadata for {source} (source_type={source_type}) with {target_schema} schema"
            )

            # Validate source_type
            if source_type not in ["modality", "metadata_store"]:
                return f"❌ Error: source_type must be 'modality' or 'metadata_store', got '{source_type}'"

            # Parse controlled vocabularies if provided
            controlled_vocab_dict = None
            if controlled_vocabularies:
                try:
                    controlled_vocab_dict = json.loads(controlled_vocabularies)
                except json.JSONDecodeError as e:
                    return f"❌ Invalid controlled_vocabularies JSON: {str(e)}"

            # Call standardization service
            # Note: standardization service may need to be updated to handle source_type
            result, stats, ir = metadata_standardization_service.standardize_metadata(
                identifier=source,
                target_schema=target_schema,
                controlled_vocabularies=controlled_vocab_dict,
            )

            # Log provenance with IR
            data_manager.log_tool_usage(
                tool_name="standardize_sample_metadata",
                parameters={
                    "source": source,
                    "source_type": source_type,
                    "target_schema": target_schema,
                    "controlled_vocabularies": controlled_vocabularies,
                },
                description=f"Standardized {len(result.standardized_metadata)} valid samples, {len(result.validation_errors)} errors, {len(result.warnings)} warnings",
                ir=ir,  # Pass IR for provenance tracking
            )

            # Format report
            report_lines = [
                "# Metadata Standardization Report\n",
                f"**Dataset:** {source}",
                f"**Source Type:** {source_type}",
                f"**Target Schema:** {target_schema}",
                f"**Valid Samples:** {len(result.standardized_metadata)}",
                f"**Validation Errors:** {len(result.validation_errors)}\n",
            ]

            # Field coverage
            if result.field_coverage:
                report_lines.append("## Field Coverage")
                for field, coverage in sorted(
                    result.field_coverage.items(), key=lambda x: x[1], reverse=True
                ):
                    report_lines.append(f"- {field}: {coverage:.1f}%")
                report_lines.append("")

            # Validation errors (show first 10)
            if result.validation_errors:
                report_lines.append("## Validation Errors")
                for i, (sample_id, error) in enumerate(
                    list(result.validation_errors.items())[:10]
                ):
                    report_lines.append(f"- {sample_id}: {error}")
                if len(result.validation_errors) > 10:
                    report_lines.append(
                        f"- ... and {len(result.validation_errors) - 10} more"
                    )
                report_lines.append("")

            # Warnings
            if result.warnings:
                report_lines.append("## Warnings")
                for warning in result.warnings[:10]:
                    report_lines.append(f"- ⚠️ {warning}")
                if len(result.warnings) > 10:
                    report_lines.append(f"- ... and {len(result.warnings) - 10} more")

            report = "\n".join(report_lines)
            logger.info(f"Standardization complete for {source}")
            return report

        except ValueError as e:
            logger.error(f"Standardization error: {e}")
            return f"❌ Standardization failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected standardization error: {e}", exc_info=True)
            return f"❌ Unexpected error during standardization: {str(e)}"

    # =========================================================================
    # Tool 4: Validate Dataset Content
    # =========================================================================

    @tool
    def validate_dataset_content(
        source: str,
        source_type: str,
        expected_samples: int = None,
        required_conditions: str = None,
        check_controls: bool = True,
        check_duplicates: bool = True,
    ) -> str:
        """
        Validate dataset completeness and metadata quality from loaded modality OR cached metadata.

        Use this tool to verify that a dataset meets minimum requirements:
        - Sample count verification
        - Condition presence check
        - Control sample detection
        - Duplicate ID check
        - Platform consistency check

        Args:
            source: Modality name (if source_type="modality") or dataset ID (if source_type="metadata_store")
            source_type: Data source type - REQUIRED, must be "modality" or "metadata_store"
                - "modality": Validate from loaded AnnData in DataManagerV2
                - "metadata_store": Validate from cached GEO metadata (pre-download validation)
            expected_samples: Minimum expected sample count (optional)
            required_conditions: Comma-separated list of required condition values (optional)
            check_controls: Whether to check for control samples (default: True)
            check_duplicates: Whether to check for duplicate sample IDs (default: True)

        Returns:
            Validation report with checks results, warnings, and recommendations

        Examples:
            # Post-download validation
            validate_dataset_content(source="geo_gse180759", source_type="modality")

            # Pre-download validation (before loading dataset)
            validate_dataset_content(source="geo_gse180759", source_type="metadata_store")
        """
        try:
            logger.info(
                f"Validating dataset content for {source} (source_type={source_type})"
            )

            # Validate source_type parameter
            if source_type not in ["modality", "metadata_store"]:
                return f"❌ Error: source_type must be 'modality' or 'metadata_store', got '{source_type}'"

            # Parse required conditions
            required_condition_list = None
            if required_conditions:
                required_condition_list = [
                    c.strip() for c in required_conditions.split(",")
                ]

            # Branch based on source_type
            if source_type == "modality":
                # EXISTING BEHAVIOR: Validate from loaded modality
                if source not in data_manager.list_modalities():
                    return (
                        f"❌ Error: Modality '{source}' not found in DataManager. "
                        f"Available modalities: {', '.join(data_manager.list_modalities())}"
                    )

                # Call validation service
                result, stats, ir = (
                    metadata_standardization_service.validate_dataset_content(
                        identifier=source,
                        expected_samples=expected_samples,
                        required_conditions=required_condition_list,
                        check_controls=check_controls,
                        check_duplicates=check_duplicates,
                    )
                )

            elif source_type == "metadata_store":
                # NEW BEHAVIOR: Pre-download validation from cached metadata
                if source not in data_manager.metadata_store:
                    return (
                        f"❌ Error: '{source}' not found in metadata_store. "
                        f"Use research_agent.validate_dataset_metadata() first to cache metadata."
                    )

                cached_metadata = data_manager.metadata_store[source]

                # Detect pattern (Option D: Namespace Separation)
                pattern = _detect_metadata_pattern(cached_metadata)

                if pattern == "geo":
                    samples_dict = cached_metadata["metadata"]["samples"]
                elif pattern == "aggregated":
                    samples_dict = _convert_list_to_dict(cached_metadata["samples"])
                else:
                    return (
                        f"❌ Error: Unrecognized metadata pattern in '{source}'. "
                        "Expected 'metadata.samples' dict or 'samples' list."
                    )

                if not samples_dict:
                    return f"❌ Error: No sample metadata in '{source}'. Cannot validate from metadata_store."

                # Convert to DataFrame for validation
                import pandas as pd

                sample_df = pd.DataFrame.from_dict(samples_dict, orient="index")

                # Perform validation using MetadataValidationService
                # Note: We need to import and use the validation service directly here
                from lobster.services.metadata.metadata_validation_service import (
                    MetadataValidationService,
                )

                validation_service = MetadataValidationService(
                    data_manager=data_manager
                )

                result = validation_service.validate_sample_metadata(
                    sample_df=sample_df,
                    expected_samples=expected_samples,
                    required_conditions=required_condition_list,
                    check_controls=check_controls,
                    check_duplicates=check_duplicates,
                )

                # For metadata_store, we don't have IR (no provenance tracking for cached metadata)
                ir = None
                {
                    "total_samples": len(sample_df),
                    "validation_passed": result.has_required_samples
                    and result.platform_consistency
                    and not result.duplicate_ids,
                }

            # Log provenance with IR (only for modality source_type)
            data_manager.log_tool_usage(
                tool_name="validate_dataset_content",
                parameters={
                    "source": source,
                    "source_type": source_type,
                    "expected_samples": expected_samples,
                    "required_conditions": required_conditions,
                    "check_controls": check_controls,
                    "check_duplicates": check_duplicates,
                },
                description=f"Validated: samples={'✓' if result.has_required_samples else '✗'}, platform={'✓' if result.platform_consistency else '✗'}, {len(result.duplicate_ids)} duplicates, {len(result.warnings)} warnings",
                ir=ir,  # Pass IR for provenance tracking (None for metadata_store)
            )

            # Format report
            report_lines = [
                "# Dataset Validation Report\n",
                f"**Dataset:** {source}",
                f"**Source Type:** {source_type}\n",
                "## Validation Checks",
                (
                    f"✅ Sample Count: {result.summary['total_samples']} samples"
                    if result.has_required_samples
                    else f"❌ Sample Count: {result.summary['total_samples']} samples (below minimum)"
                ),
                (
                    "✅ Platform Consistency: Consistent"
                    if result.platform_consistency
                    else "⚠️ Platform Consistency: Inconsistent"
                ),
                (
                    "✅ No Duplicate IDs"
                    if not result.duplicate_ids
                    else f"❌ Duplicate IDs: {len(result.duplicate_ids)} found"
                ),
                (
                    "✅ Control Samples: Detected"
                    if not result.control_issues
                    else f"⚠️ Control Samples: {', '.join(result.control_issues)}"
                ),
            ]

            # Missing conditions
            if result.missing_conditions:
                report_lines.append("\n## Missing Required Conditions")
                for condition in result.missing_conditions:
                    report_lines.append(f"- ❌ {condition}")

            # Summary
            report_lines.append("\n## Dataset Summary")
            for key, value in result.summary.items():
                report_lines.append(f"- {key}: {value}")

            # Warnings
            if result.warnings:
                report_lines.append("\n## Warnings")
                for warning in result.warnings:
                    report_lines.append(f"- ⚠️ {warning}")

            # Recommendation
            report_lines.append("\n## Recommendation")
            if (
                result.has_required_samples
                and result.platform_consistency
                and not result.duplicate_ids
            ):
                report_lines.append(
                    "✅ **Dataset passes validation** - ready for download/analysis"
                )
            else:
                report_lines.append(
                    "⚠️ **Dataset has issues** - review warnings before proceeding"
                )

            report = "\n".join(report_lines)
            logger.info(f"Validation complete for {source}")
            return report

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return f"❌ Validation failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}", exc_info=True)
            return f"❌ Unexpected error during validation: {str(e)}"

    # =========================================================================
    # Helper: Disease Extraction from Diverse SRA Fields
    # =========================================================================

    def _extract_disease_from_raw_fields(
        metadata: pd.DataFrame, study_context: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Extract disease information from diverse, study-specific SRA field names.

        SRA datasets have NO standardized disease field. Disease data appears in:
        - Free-text: host_phenotype, phenotype, disease_state, diagnosis
        - Boolean flags: crohns_disease, inflam_bowel_disease, parkinson_disease
        - Study title: Embedded in study_title or experiment_title

        This method consolidates diverse field patterns into a unified "disease" column.

        Extraction Strategies (applied in order):
        1. Existing unified column (disease, disease_state, condition, diagnosis)
        2. Free-text phenotype fields (host_phenotype → disease)
        3. Boolean disease flags (crohns_disease: Yes → disease: cd)
        4. Study context (publication-level disease inference)

        Args:
            metadata: DataFrame with SRA sample metadata
            study_context: Optional publication metadata for context

        Returns:
            Column name containing unified disease data, or None if no disease info found

        Example transformations:
            host_phenotype: "Parkinson's Disease" → disease: "Parkinson's Disease"
            crohns_disease: "Yes" → disease: "cd"
            inflam_bowel_disease: "Yes" → disease: "ibd"
            parkinson_disease: TRUE → disease: "parkinsons"
        """
        # Strategy 1: Check for existing unified disease column
        existing_disease_cols = ["disease", "disease_state", "condition", "diagnosis"]
        for col in existing_disease_cols:
            if col in metadata.columns and metadata[col].notna().sum() > 0:
                logger.debug(f"Found existing disease column: {col}")
                # Rename to standard "disease" if different
                if col != "disease":
                    metadata["disease"] = metadata[col]
                    metadata["disease_original"] = metadata[col]
                return "disease"

        # Strategy 2: Extract from free-text phenotype fields
        phenotype_cols = [
            "host_phenotype",
            "phenotype",
            "host_disease",
            "health_status",
        ]
        for col in phenotype_cols:
            if col in metadata.columns:
                # Count non-empty values
                populated_count = metadata[col].notna().sum()
                if populated_count > 0:
                    # Create unified disease column from phenotype
                    metadata["disease"] = metadata[col].fillna("unknown")
                    metadata["disease_original"] = metadata[col].fillna("unknown")
                    logger.debug(
                        f"Extracted disease from {col} "
                        f"({populated_count}/{len(metadata)} samples, "
                        f"{populated_count/len(metadata)*100:.1f}%)"
                    )
                    return "disease"

        # Strategy 3: Consolidate boolean disease flags
        # Find columns ending with "_disease" (crohns_disease, inflam_bowel_disease, etc.)
        disease_flag_cols = [c for c in metadata.columns if c.endswith("_disease")]

        if disease_flag_cols:
            logger.debug(
                f"Found {len(disease_flag_cols)} disease flag columns: {disease_flag_cols}"
            )

            def extract_from_flags(row):
                """Extract disease from boolean flags."""
                active_diseases = []

                for flag_col in disease_flag_cols:
                    # Check if flag is TRUE (handles Yes, Y, TRUE, True, 1, "1")
                    flag_value = row.get(flag_col)
                    if flag_value in [
                        "Yes",
                        "YES",
                        "yes",
                        "Y",
                        "y",
                        "TRUE",
                        "True",
                        "true",
                        True,
                        1,
                        "1",
                    ]:
                        # Convert flag name to disease term
                        # Examples:
                        #   crohns_disease → cd
                        #   inflam_bowel_disease → ibd
                        #   ulcerative_colitis → uc
                        #   parkinson_disease → parkinsons
                        disease_name = flag_col.replace("_disease", "").replace("_", "")

                        # Map common patterns to standard terms
                        disease_map = {
                            "crohns": "cd",
                            "crohn": "cd",
                            "inflammbowel": "ibd",
                            "inflambowel": "ibd",  # Handle different spellings
                            "ulcerativecolitis": "uc",
                            "colitis": "uc",
                            "parkinson": "parkinsons",
                            "parkinsons": "parkinsons",
                        }

                        standardized = disease_map.get(disease_name, disease_name)
                        active_diseases.append(standardized)

                if active_diseases:
                    # If multiple diseases, join with semicolon
                    return ";".join(active_diseases)

                # Check for negative controls (all flags FALSE)
                all_false = all(
                    row.get(flag_col)
                    in [
                        "No",
                        "NO",
                        "no",
                        "N",
                        "n",
                        "FALSE",
                        "False",
                        "false",
                        False,
                        0,
                        "0",
                    ]
                    for flag_col in disease_flag_cols
                )
                if all_false:
                    return "healthy"

                return "unknown"

            # Apply extraction
            metadata["disease"] = metadata.apply(extract_from_flags, axis=1)
            metadata["disease_original"] = metadata.apply(
                lambda row: ";".join(
                    [
                        f"{col}={row[col]}"
                        for col in disease_flag_cols
                        if pd.notna(row.get(col))
                    ]
                ),
                axis=1,
            )

            # Count successful extractions
            extracted_count = (metadata["disease"] != "unknown").sum()
            logger.debug(
                f"Extracted disease from {len(disease_flag_cols)} boolean flags "
                f"({extracted_count}/{len(metadata)} samples, "
                f"{extracted_count/len(metadata)*100:.1f}%)"
            )

            return "disease"

        # Strategy 4: Use study context (publication-level disease focus)
        if study_context and "disease_focus" in study_context:
            # All samples in this study share the publication's disease focus
            metadata["disease"] = study_context["disease_focus"]
            metadata["disease_original"] = (
                f"inferred from publication: {study_context['disease_focus']}"
            )
            logger.debug(
                f"Assigned disease from publication context: {study_context['disease_focus']}"
            )
            return "disease"

        logger.warning(
            "No disease information found in metadata fields or study context"
        )
        return None

    # =========================================================================
    # Tool 5: Filter Samples By Criteria (Microbiome/Disease)
    # =========================================================================

    @tool
    def filter_samples_by(
        workspace_key: str,
        filter_criteria: str,
        strict: bool = True,
        min_disease_coverage: float = 0.5,
        strict_disease_validation: bool = True,
    ) -> str:
        """
        Filter samples by multi-modal criteria (16S amplicon + host organism + sample type + disease).

        Use this tool when you need to filter workspace metadata by microbiome-specific criteria:
        - 16S amplicon sequencing detection (platform, library_strategy, assay_type)
        - Amplicon region validation (V4, V3-V4, V1-V9, full-length)
        - Host organism validation (human, mouse with fuzzy matching)
        - Sample type filtering (fecal vs tissue/biopsy)
        - Disease standardization (CRC, UC, CD, healthy controls)

        This tool applies filters IN SEQUENCE (composition pattern):
        1. Check if 16S amplicon (if requested)
        2. Validate amplicon region (if region specified)
        3. Validate host organism (if requested)
        4. Filter by sample type (if requested)
        5. Standardize disease terms (if requested)

        Filter Criteria Syntax:
            Basic: "16S human fecal CRC"
            With region: "16S V4 human fecal CRC"
            With region: "16S V3-V4 mouse gut UC"

            Supported regions: V1-V9, V3-V4, V4-V5, V1-V9, full-length

        Args:
            workspace_key: Key for workspace metadata (e.g., "geo_gse123456")
            filter_criteria: Natural language criteria (e.g., "16S V4 human fecal CRC")
            strict: Use strict matching for 16S detection (default: True)
            min_disease_coverage: Minimum disease coverage rate (0.0-1.0).
                Default 0.5 (50%). Set to 0.0 to disable validation.
            strict_disease_validation: If True, raise error when coverage is low.
                If False, log warning but continue. Default True.

        Returns:
            Formatted markdown report with filtering results, retention rate, and filtered metadata summary

        Examples:
            # Default: Fail if <50% disease coverage
            filter_samples_by(workspace_key="geo_gse123456", filter_criteria="16S human fecal CRC")

            # Enforce specific amplicon region
            filter_samples_by(workspace_key="geo_gse123456", filter_criteria="16S V4 human fecal CRC")

            # Region detection (V3-V4 only)
            filter_samples_by(workspace_key="geo_gse123456", filter_criteria="16S V3-V4 human fecal")

            # Permissive: Warn but continue
            filter_samples_by(workspace_key="geo_gse123456",
                            filter_criteria="16S human fecal",
                            strict_disease_validation=False)

            # Lower threshold: Fail if <30% coverage
            filter_samples_by(workspace_key="geo_gse123456",
                            filter_criteria="16S human fecal",
                            min_disease_coverage=0.3)

            # Disable validation entirely
            filter_samples_by(workspace_key="geo_gse123456",
                            filter_criteria="16S human fecal",
                            min_disease_coverage=0.0)
        """
        # Check if microbiome services are available
        if not MICROBIOME_FEATURES_AVAILABLE:
            return "❌ Error: Microbiome filtering features are not available in this installation. This is an optional feature."

        try:
            logger.info(
                f"Filtering samples: workspace_key={workspace_key}, criteria='{filter_criteria}', strict={strict}"
            )

            # Parse natural language criteria using service
            if not metadata_filtering_service:
                return "❌ Error: Microbiome filtering service not available"
            parsed_criteria = metadata_filtering_service.parse_criteria(filter_criteria)

            logger.debug(f"Parsed criteria: {parsed_criteria}")

            # Read workspace metadata via WorkspaceContentService
            from lobster.services.data_access.workspace_content_service import (
                WorkspaceContentService,
            )

            workspace_service = WorkspaceContentService(data_manager)
            workspace_data = workspace_service.read_content(workspace_key)
            if not workspace_data:
                return f"❌ Error: Workspace key '{workspace_key}' not found or empty"

            # Detect pattern (Option D: Namespace Separation)
            if not isinstance(workspace_data, dict):
                return f"❌ Error: Unexpected workspace data format (expected dict)"

            pattern = _detect_metadata_pattern(workspace_data)

            if pattern == "geo":
                # GEO pattern: {"metadata": {"samples": {...}}}
                metadata_dict = workspace_data["metadata"]["samples"]
            elif pattern == "aggregated":
                # Aggregated pattern: {"samples": [...]} - keep as list for filtering efficiency
                metadata_dict = workspace_data["samples"]
                logger.debug(f"Using aggregated pattern ({len(metadata_dict)} samples) from '{workspace_key}'")
            else:
                # Fallback: treat entire workspace_data as samples dict
                metadata_dict = workspace_data

            if not metadata_dict:
                return f"❌ Error: No sample metadata found in workspace key '{workspace_key}'"

            # Convert to DataFrame
            import pandas as pd

            # Handle both dict-of-dicts (orient="index") and list-of-dicts
            if isinstance(metadata_dict, list):
                metadata_df = pd.DataFrame(metadata_dict)
                # Set index to run_accession if available for consistency
                if "run_accession" in metadata_df.columns:
                    metadata_df = metadata_df.set_index("run_accession")
            else:
                metadata_df = pd.DataFrame.from_dict(metadata_dict, orient="index")
            original_count = len(metadata_df)

            logger.debug(f"Loaded {original_count} samples from workspace")

            # Apply filters in sequence
            irs = []
            stats_list = []
            current_metadata = metadata_df.copy()

            # Filter 1: 16S amplicon detection
            if parsed_criteria["check_16s"]:
                logger.info("Applying 16S amplicon filter...")
                filtered_rows = []
                for idx, row in current_metadata.iterrows():
                    row_dict = row.to_dict()
                    filtered, stats, ir = (
                        microbiome_filtering_service.validate_16s_amplicon(
                            row_dict, strict=strict
                        )
                    )
                    if filtered:  # Non-empty dict means valid
                        filtered_rows.append(idx)
                    if stats["is_valid"]:
                        irs.append(ir)
                        stats_list.append(stats)

                current_metadata = current_metadata.loc[filtered_rows]
                logger.debug(
                    f"After 16S filter: {len(current_metadata)} samples retained"
                )

            # Filter 2: Host organism validation
            if parsed_criteria["host_organisms"]:
                logger.debug(
                    f"Applying host organism filter: {parsed_criteria['host_organisms']}"
                )
                filtered_rows = []
                for idx, row in current_metadata.iterrows():
                    row_dict = row.to_dict()
                    filtered, stats, ir = (
                        microbiome_filtering_service.validate_host_organism(
                            row_dict, allowed_hosts=parsed_criteria["host_organisms"]
                        )
                    )
                    if filtered:  # Non-empty dict means valid
                        filtered_rows.append(idx)
                    if stats["is_valid"]:
                        irs.append(ir)
                        stats_list.append(stats)

                current_metadata = current_metadata.loc[filtered_rows]
                logger.debug(
                    f"After host filter: {len(current_metadata)} samples retained"
                )

            # Filter 3: Sample type filtering
            if parsed_criteria["sample_types"]:
                logger.debug(
                    f"Applying sample type filter: {parsed_criteria['sample_types']}"
                )
                filtered, stats, ir = (
                    disease_standardization_service.filter_by_sample_type(
                        current_metadata, sample_types=parsed_criteria["sample_types"]
                    )
                )
                current_metadata = filtered
                irs.append(ir)
                stats_list.append(stats)
                logger.debug(
                    f"After sample type filter: {len(current_metadata)} samples retained"
                )

            # Filter 4: Disease extraction + standardization
            if parsed_criteria["standardize_disease"]:
                logger.debug("Applying disease extraction + standardization...")

                # NEW: Extract disease from diverse field patterns (v1.2.0)
                # Handles: host_phenotype, boolean flags (crohns_disease, etc.)
                disease_col = _extract_disease_from_raw_fields(
                    current_metadata, study_context=None
                )

                if disease_col:
                    # Apply standardization to extracted disease column
                    try:
                        standardized, stats, ir = (
                            disease_standardization_service.standardize_disease_terms(
                                current_metadata, disease_column=disease_col
                            )
                        )
                        current_metadata = standardized
                        irs.append(ir)
                        stats_list.append(stats)
                        logger.debug(
                            f"Disease extraction + standardization complete: {stats['standardization_rate']:.1f}% mapped"
                        )

                        # Validate disease coverage if requested (pass parameters through)
                        if min_disease_coverage > 0:
                            disease_coverage = stats.get("standardization_rate", 0) / 100.0
                            if disease_coverage < min_disease_coverage:
                                error_msg = (
                                    f"Insufficient disease data coverage: {disease_coverage*100:.1f}% "
                                    f"(required: {min_disease_coverage*100:.1f}%). "
                                    f"Consider using min_disease_coverage=0.0 to disable validation."
                                )
                                if strict_disease_validation:
                                    return f"❌ Disease validation failed:\n{error_msg}"
                                else:
                                    logger.warning(f"Disease validation warning: {error_msg}")
                    except ValueError as e:
                        if "Insufficient disease data" in str(e):
                            # Extract coverage percentage
                            coverage_match = re.search(r'(\d+\.\d+)% coverage', str(e))
                            coverage = float(coverage_match.group(1)) if coverage_match else 0.0

                            return f"""❌ Disease validation failed: {coverage:.1f}% coverage < 50% threshold

I can attempt automatic enrichment to improve coverage. Would you like me to:

1. **Auto-enrich** (recommended): Extract disease from publication abstracts/methods using LLM
   Command: enrich_samples_with_disease(workspace_key="{workspace_key}", enrichment_mode="hybrid")

2. **Manual mapping**: Provide disease for each publication
   Command: enrich_samples_with_disease(
       workspace_key="{workspace_key}",
       enrichment_mode="manual",
       manual_mappings='{{"pub_queue_doi_10_1234": "crc", "pub_queue_pmid_5678": "uc"}}'
   )

3. **Preview first** (recommended): Test enrichment without saving
   Command: enrich_samples_with_disease(workspace_key="{workspace_key}", dry_run=True)

4. **Lower threshold** (not recommended): min_disease_coverage={coverage/100:.2f}

Which approach would you prefer?"""
                        else:
                            raise
                else:
                    logger.warning(
                        "Disease standardization requested but no disease information found in metadata"
                    )

            # Calculate final stats
            final_count = len(current_metadata)
            retention_rate = (
                (final_count / original_count * 100) if original_count > 0 else 0
            )

            # Combine IRs into composite IR
            composite_ir = _combine_analysis_steps(
                irs,
                operation="filter_samples_by",
                description=f"Multi-criteria filtering: {filter_criteria}",
            )

            # Log tool usage
            data_manager.log_tool_usage(
                tool_name="filter_samples_by",
                parameters={
                    "workspace_key": workspace_key,
                    "filter_criteria": filter_criteria,
                    "strict": strict,
                    "min_disease_coverage": min_disease_coverage,
                    "strict_disease_validation": strict_disease_validation,
                    "parsed_criteria": parsed_criteria,
                },
                description=f"Filtered {original_count}→{final_count} samples ({retention_rate:.1f}% retention), {len(irs)} filters applied",
                ir=composite_ir,
            )

            # Format report
            report = _format_filtering_report(
                workspace_key=workspace_key,
                filter_criteria=filter_criteria,
                parsed_criteria=parsed_criteria,
                original_count=original_count,
                final_count=final_count,
                retention_rate=retention_rate,
                stats_list=stats_list,
                filtered_metadata=current_metadata,
            )

            logger.debug(
                f"Filtering complete: {final_count}/{original_count} samples retained ({retention_rate:.1f}%)"
            )
            return report

        except ValueError as e:
            logger.error(f"Filtering error: {e}")
            return f"❌ Filtering failed: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected filtering error: {e}", exc_info=True)
            return f"❌ Unexpected error during filtering: {str(e)}"

    # =========================================================================
    # Phase 4c NEW TOOLS: Publication Queue Processing (3 tools)
    # =========================================================================

    # Import shared workspace tools and services
    from lobster.services.data_access.workspace_content_service import (
        WorkspaceContentService,
    )
    from lobster.tools.workspace_tool import (
        create_get_content_from_workspace_tool,
        create_write_to_workspace_tool,
    )

    workspace_service = WorkspaceContentService(data_manager=data_manager)

    # Create shared workspace tools
    get_content_from_workspace = create_get_content_from_workspace_tool(data_manager)
    write_to_workspace = create_write_to_workspace_tool(data_manager)

    @tool
    def process_metadata_entry(
        entry_id: str,
        filter_criteria: str = None,
        standardize_schema: str = None,
        min_disease_coverage: float = 0.5,
        strict_disease_validation: bool = True,
    ) -> str:
        """
        Process a single publication queue entry for metadata filtering/standardization.

        Reads workspace_metadata_keys from the entry, applies filters if specified,
        and stores results back to harmonization_metadata.

        Filter Criteria Syntax:
            Basic: "16S human fecal CRC"
            With region: "16S V4 human fecal CRC"
            With region: "16S V3-V4 mouse gut UC"

            Supported regions: V1-V9, V3-V4, V4-V5, V1-V9, full-length

        Args:
            entry_id: Publication queue entry ID
            filter_criteria: Optional natural language filter (e.g., "16S V4 human fecal")
            standardize_schema: Optional schema to standardize to (e.g., "microbiome")
            min_disease_coverage: Minimum disease coverage rate (0.0-1.0).
                Default 0.5 (50%). Set to 0.0 to disable validation.
            strict_disease_validation: If True, raise error when coverage is low.
                If False, log warning but continue. Default True.

        Returns:
            Processing summary with sample counts and filtered metadata location

        Examples:
            # Default: Fail if <50% disease coverage
            process_metadata_entry(entry_id="pub_123", filter_criteria="16S human fecal")

            # Enforce specific amplicon region
            process_metadata_entry(entry_id="pub_123", filter_criteria="16S V4 human fecal CRC")

            # Region detection (V3-V4 only)
            process_metadata_entry(entry_id="pub_123", filter_criteria="16S V3-V4 human fecal")

            # Permissive: Warn but continue
            process_metadata_entry(entry_id="pub_123",
                                  filter_criteria="16S human fecal",
                                  strict_disease_validation=False)

            # Lower threshold: Fail if <30% coverage
            process_metadata_entry(entry_id="pub_123",
                                  filter_criteria="16S human fecal",
                                  min_disease_coverage=0.3)

            # Disable validation entirely
            process_metadata_entry(entry_id="pub_123",
                                  filter_criteria="16S human fecal",
                                  min_disease_coverage=0.0)
        """
        try:
            queue = data_manager.publication_queue
            entry = queue.get_entry(entry_id)

            if not entry.workspace_metadata_keys:
                error_msg = (
                    f"Entry {entry_id} has no workspace_metadata_keys. "
                    f"Expected at least one 'sra_*_samples' key. "
                    f"Run research_agent.process_publication_queue() to populate workspace keys."
                )
                logger.error(error_msg)
                return f"❌ Error: {error_msg}"

            # Log entry processing start
            logger.info(
                f"Processing entry {entry_id}: '{entry.title or 'No title'}' "
                f"({len(entry.workspace_metadata_keys)} workspace keys)"
            )

            # Update status to in_progress
            queue.update_status(
                entry_id,
                entry.status,
                handoff_status=HandoffStatus.METADATA_IN_PROGRESS,
            )

            # Read and aggregate samples from all workspace keys
            all_samples = []
            all_validation_results = []

            for ws_key in entry.workspace_metadata_keys:
                # Only process SRA sample files (skip pub_queue_*_metadata/methods/identifiers.json)
                if not (ws_key.startswith("sra_") and ws_key.endswith("_samples")):
                    logger.debug(f"Skipping non-SRA workspace key: {ws_key}")
                    continue

                logger.debug(f"Processing SRA sample file: {ws_key}")

                from lobster.services.data_access.workspace_content_service import (
                    ContentType,
                )

                ws_data = workspace_service.read_content(
                    ws_key, content_type=ContentType.METADATA
                )
                if ws_data:
                    samples, validation_result, quality_stats = (
                        _extract_samples_from_workspace(ws_data)
                    )
                    all_samples.extend(samples)
                    all_validation_results.append(validation_result)

                    # Log extraction results with quality info
                    logger.debug(
                        f"Extracted {len(samples)} valid samples from {ws_key} "
                        f"(validation_rate: {validation_result.metadata.get('validation_rate', 0):.1f}%, "
                        f"avg_completeness: {quality_stats.get('avg_completeness', 0):.1f}, "
                        f"unique_individuals: {quality_stats.get('unique_individuals', 0)})"
                    )

            samples_before = sum(
                vr.metadata.get("total_samples", 0) for vr in all_validation_results
            )

            # Check for complete validation failure
            if samples_before > 0 and len(all_samples) == 0:
                # All samples failed validation
                total_errors = sum(len(vr.errors) for vr in all_validation_results)
                error_msg = (
                    f"All {samples_before} samples failed validation ({total_errors} errors). "
                    f"Check workspace files for data quality issues."
                )
                logger.error(f"Entry {entry_id}: {error_msg}")

                # Log first few validation errors for debugging
                logger.error(f"First validation errors for entry {entry_id}:")
                for vr in all_validation_results[:3]:  # First 3 workspace keys
                    for error in vr.errors[:3]:  # First 3 errors per key
                        logger.error(f"  - {error}")

                # Mark as METADATA_FAILED
                queue.update_status(
                    entry_id,
                    entry.status,
                    handoff_status=HandoffStatus.METADATA_FAILED,
                    error=error_msg,
                )

                return f"❌ {error_msg}"

            # Apply filters if specified
            if filter_criteria and all_samples and metadata_filtering_service:
                logger.debug(
                    f"Applying filter criteria to {len(all_samples)} samples: '{filter_criteria}'"
                )
                parsed = metadata_filtering_service.parse_criteria(filter_criteria)
                # Add disease validation parameters
                parsed["min_disease_coverage"] = min_disease_coverage
                parsed["strict_disease_validation"] = strict_disease_validation
                # Pass disease_extractor with fallback chain (Bug 3 fix - DataBioMix)
                metadata_filtering_service.disease_extractor = extract_disease_with_fallback
                try:
                    all_samples, filter_stats, _ = metadata_filtering_service.apply_filters(
                        all_samples, parsed
                    )
                    logger.debug(
                        f"After filtering: {len(all_samples)} samples retained "
                        f"({filter_stats.get('retention_rate', 0):.1f}%)"
                    )
                except ValueError as e:
                    if "Insufficient disease data" in str(e):
                        error_msg = f"Disease validation failed: {str(e)}"
                        logger.error(f"Entry {entry_id}: {error_msg}")
                        # Mark as METADATA_FAILED
                        queue.update_status(
                            entry_id,
                            entry.status,
                            handoff_status=HandoffStatus.METADATA_FAILED,
                            error=error_msg,
                        )
                        return f"❌ {error_msg}"
                    else:
                        raise

            samples_after = len(all_samples)

            # Add publication context
            for sample in all_samples:
                sample["publication_entry_id"] = entry_id
                sample["publication_title"] = entry.title
                sample["publication_doi"] = entry.doi
                sample["publication_pmid"] = entry.pmid

            # Aggregate validation statistics
            total_errors = sum(len(vr.errors) for vr in all_validation_results)
            total_warnings = sum(len(vr.warnings) for vr in all_validation_results)

            # Log overall validation summary
            validation_rate = (
                (len(all_samples) / samples_before * 100) if samples_before > 0 else 0
            )
            logger.debug(
                f"Entry {entry_id} validation summary: "
                f"{len(all_samples)}/{samples_before} samples valid ({validation_rate:.1f}%), "
                f"{total_errors} errors, {total_warnings} warnings"
            )

            # Store in harmonization_metadata
            harmonization_data = {
                "samples": all_samples,
                "filter_criteria": filter_criteria,
                "standardize_schema": standardize_schema,
                "stats": {
                    "samples_extracted": samples_before,
                    "samples_valid": len(all_samples),
                    "samples_after_filter": samples_after,
                    "validation_errors": total_errors,
                    "validation_warnings": total_warnings,
                },
            }

            queue.update_status(
                entry_id=entry_id,
                status=PublicationStatus.COMPLETED,  # Terminal success state
                handoff_status=HandoffStatus.METADATA_COMPLETE,
                harmonization_metadata=harmonization_data,
            )

            retention = (
                (samples_after / samples_before * 100) if samples_before > 0 else 0
            )

            # Build response with validation info
            response = f"""## Entry Processed: {entry_id}
**Samples Extracted**: {samples_before}
**Samples Valid**: {len(all_samples)}
**Samples After Filter**: {samples_after}
**Retention**: {retention:.1f}%
**Filter**: {filter_criteria or 'None'}
**Validation**: {total_errors} errors, {total_warnings} warnings
"""
            # Include validation messages if there are issues
            if total_errors > 0 or total_warnings > 0:
                response += "\n### Validation Summary\n"
                for idx, vr in enumerate(all_validation_results):
                    if vr.has_errors or vr.has_warnings:
                        response += f"\n**Workspace key {idx+1}**:\n"
                        response += vr.format_messages(include_info=False) + "\n"

            return response
        except Exception as e:
            logger.error(f"Error processing entry {entry_id}: {e}")
            return f"❌ Error processing entry: {str(e)}"

    def _process_single_entry_for_queue(entry, filter_criteria, min_disease_coverage, strict_disease_validation):
        """
        Process a single entry and return (entry_samples, entry_stats, failed_reason).

        Extracted for reuse in both sequential and parallel processing.

        Args:
            entry: Publication queue entry
            filter_criteria: Filter criteria string
            min_disease_coverage: Minimum disease coverage threshold
            strict_disease_validation: Whether to fail on low coverage

        Returns:
            Tuple of (entry_samples: List, entry_stats: Dict, failed_reason: Optional[str])
        """
        try:
            if not entry.workspace_metadata_keys:
                logger.debug(
                    f"Skipping entry {entry.entry_id}: no workspace_metadata_keys"
                )
                return [], {}, None

            entry_samples = []
            entry_validation_results = []
            entry_quality_stats = []

            for ws_key in entry.workspace_metadata_keys:
                # Only process SRA sample files
                if not (ws_key.startswith("sra_") and ws_key.endswith("_samples")):
                    continue

                logger.debug(f"Processing {ws_key} for entry {entry.entry_id}")

                from lobster.services.data_access.workspace_content_service import ContentType

                ws_data = workspace_service.read_content(ws_key, content_type=ContentType.METADATA)
                if ws_data:
                    samples, validation_result, quality_stats = _extract_samples_from_workspace(ws_data)
                    entry_samples.extend(samples)
                    entry_validation_results.append(validation_result)
                    entry_quality_stats.append(quality_stats)

            samples_extracted = sum(vr.metadata.get("total_samples", 0) for vr in entry_validation_results)
            samples_valid = len(entry_samples)

            # Check for complete validation failure
            if samples_extracted > 0 and samples_valid == 0:
                error_count = sum(len(vr.errors) for vr in entry_validation_results)
                error_msg = f"All {samples_extracted} samples failed validation ({error_count} errors)"
                logger.error(f"Entry {entry.entry_id}: {error_msg}")
                return [], {"extracted": samples_extracted, "valid": 0}, error_msg

            # Apply filters if specified
            if filter_criteria and entry_samples and metadata_filtering_service:
                samples_before_filter = len(entry_samples)
                parsed = metadata_filtering_service.parse_criteria(filter_criteria)
                # Add disease validation parameters
                parsed["min_disease_coverage"] = min_disease_coverage
                parsed["strict_disease_validation"] = strict_disease_validation
                metadata_filtering_service.disease_extractor = extract_disease_with_fallback
                try:
                    entry_samples, _, _ = metadata_filtering_service.apply_filters(entry_samples, parsed)
                    logger.debug(
                        f"Entry {entry.entry_id}: Filter applied - "
                        f"{len(entry_samples)}/{samples_before_filter} samples retained"
                    )
                except ValueError as e:
                    if "Insufficient disease data" in str(e):
                        error_msg = f"Disease validation failed: {str(e)}"
                        logger.error(f"Entry {entry.entry_id}: {error_msg}")
                        return [], {"extracted": samples_extracted, "valid": samples_valid}, error_msg
                    else:
                        raise

            # Add publication context
            for sample in entry_samples:
                sample["publication_entry_id"] = entry.entry_id
                sample["publication_title"] = entry.title
                sample["publication_doi"] = entry.doi
                sample["publication_pmid"] = entry.pmid

            entry_stats = {
                "extracted": samples_extracted,
                "valid": samples_valid,
                "after_filter": len(entry_samples),
                "validation_errors": sum(len(vr.errors) for vr in entry_validation_results),
                "validation_warnings": sum(len(vr.warnings) for vr in entry_validation_results),
            }

            return entry_samples, entry_stats, None

        except Exception as e:
            error_msg = f"Failed to process entry {entry.entry_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [], {}, error_msg

    def _process_queue_with_progress(entries, filter_criteria, output_key, parallel_workers, min_disease_coverage, strict_disease_validation):
        """Process queue entries in parallel with Rich progress visualization."""
        if not PROGRESS_UI_AVAILABLE:
            logger.warning("Progress UI not available, cannot use parallel processing")
            return "❌ Progress UI not available. Use sequential processing instead."

        # Batch flush configuration - reduces 655 atomic writes to ~33 writes (20x improvement)
        BATCH_FLUSH_SIZE = 20

        # Access publication queue via data_manager (in closure scope)
        pub_queue = data_manager.publication_queue

        def _batch_flush_updates(updates):
            """Flush multiple status updates in a single atomic write.

            This dramatically reduces I/O by writing all pending updates at once
            instead of one-by-one, avoiding the O(N²) bottleneck where each update
            reads/writes the entire queue file.
            """
            if not updates:
                return

            with pub_queue._locked():
                entries_list = pub_queue._load_entries()
                entry_map = {e.entry_id: e for e in entries_list}

                for update in updates:
                    entry_id = update["entry_id"]
                    if entry_id in entry_map:
                        entry = entry_map[entry_id]
                        entry.update_status(
                            status=update["status"],
                            handoff_status=update["handoff_status"],
                            error=update.get("error"),
                        )

                pub_queue._write_entries_atomic(entries_list)

            logger.debug(f"Batch flushed {len(updates)} status updates")

        results = []
        results_lock = threading.Lock()
        all_samples = []
        samples_lock = threading.Lock()

        # Work-stealing queue
        entry_queue = list(enumerate(entries))
        queue_lock = threading.Lock()

        def get_next_entry():
            with queue_lock:
                if entry_queue:
                    return entry_queue.pop(0)
                return None

        start_time = time.time()
        effective_workers = min(parallel_workers, len(entries))

        with _suppress_logs():
            with parallel_workers_progress(effective_workers, len(entries)) as progress:

                def worker_func(worker_id: int):
                    """Worker function that processes entries from queue."""
                    # Collect status updates in memory, flush in batches to avoid O(N²) I/O
                    pending_updates = []

                    while True:
                        next_item = get_next_entry()
                        if next_item is None:
                            # Flush remaining updates before exiting
                            if pending_updates:
                                _batch_flush_updates(pending_updates)
                            progress.worker_done(worker_id)
                            break

                        idx, entry = next_item

                        # Get title for display
                        title = (entry.title or entry.entry_id)[:35]
                        progress.worker_start(worker_id, title)

                        # Process entry
                        entry_start = time.time()
                        entry_samples, entry_stats, failed_reason = _process_single_entry_for_queue(
                            entry, filter_criteria, min_disease_coverage, strict_disease_validation
                        )

                        elapsed = time.time() - entry_start

                        # Determine status and queue update (batched, not immediate)
                        if failed_reason:
                            status = "failed"
                            pending_updates.append({
                                "entry_id": entry.entry_id,
                                "status": entry.status,
                                "handoff_status": HandoffStatus.METADATA_FAILED,
                                "error": failed_reason,
                            })
                        else:
                            status = "completed"
                            pending_updates.append({
                                "entry_id": entry.entry_id,
                                "status": PublicationStatus.COMPLETED,  # Terminal success state
                                "handoff_status": HandoffStatus.METADATA_COMPLETE,
                            })

                        # Flush batch when threshold reached
                        if len(pending_updates) >= BATCH_FLUSH_SIZE:
                            _batch_flush_updates(pending_updates)
                            pending_updates = []

                        progress.worker_complete(worker_id, status, elapsed)

                        # Store results
                        with results_lock:
                            results.append({
                                "entry_id": entry.entry_id,
                                "status": status,
                                "stats": entry_stats,
                                "error": failed_reason,
                            })

                        with samples_lock:
                            all_samples.extend(entry_samples)

                # Launch workers
                executor = ThreadPoolExecutor(max_workers=effective_workers)
                try:
                    futures = [executor.submit(worker_func, i) for i in range(effective_workers)]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Worker error: {e}")
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)

        # Post-processing with progress feedback (prevents "654/655 hang" UX issue)
        from rich.console import Console
        console = Console()

        with console.status("[bold cyan]Finalizing: aggregating results...") as status:
            # Aggregate stats
            total_extracted = sum(r["stats"].get("extracted", 0) for r in results)
            total_valid = sum(r["stats"].get("valid", 0) for r in results)
            total_after_filter = len(all_samples)
            successful = sum(1 for r in results if r["status"] == "completed")
            failed = sum(1 for r in results if r["status"] == "failed")

            # Store aggregated results
            if all_samples:
                status.update("[bold cyan]Finalizing: writing to workspace...")

                from datetime import datetime
                from lobster.services.data_access.workspace_content_service import ContentType, MetadataContent

                content = MetadataContent(
                    identifier=output_key,
                    content_type="filtered_samples",
                    description=f"Batch filtered samples: {filter_criteria or 'no filter'}",
                    data={"samples": all_samples, "filter_criteria": filter_criteria, "stats": results},
                    source="metadata_assistant",
                    cached_at=datetime.now().isoformat(),
                )
                workspace_service.write_content(content, ContentType.METADATA)
                data_manager.metadata_store[output_key] = {
                    "samples": all_samples,
                    "filter_criteria": filter_criteria,
                    "stats": results,
                }

        retention = (total_after_filter / total_extracted * 100) if total_extracted > 0 else 0
        validation_rate = (total_valid / total_extracted * 100) if total_extracted > 0 else 0
        total_time = time.time() - start_time

        response = f"""## Queue Processing Complete (Parallel Mode)
**Entries Processed**: {len(results)}
**Successful**: {successful}
**Failed**: {failed}
**Samples Extracted**: {total_extracted}
**Samples Valid**: {total_valid} ({validation_rate:.1f}%)
**Samples After Filter**: {total_after_filter}
**Retention Rate**: {retention:.1f}%
**Processing Time**: {total_time:.1f}s ({len(results)/total_time*60:.1f} entries/min)
**Output Key**: {output_key}

Use `write_to_workspace(identifier="{output_key}", workspace="metadata", output_format="csv")` to export as CSV.
"""
        return response

    @tool
    def process_metadata_queue(
        status_filter: str = "handoff_ready",
        filter_criteria: str = None,
        max_entries: int = None,
        output_key: str = "aggregated_filtered_samples",
        parallel_workers: int = None,
        min_disease_coverage: float = 0.5,
        strict_disease_validation: bool = True,
    ) -> str:
        """
        Batch process publication queue entries and aggregate sample-level metadata from workspace SRA files.

        CRITICAL: Only use status_filter='handoff_ready' for filtering. handoff_ready entries contain workspace_metadata_keys
        with sra_*_samples files ready to load. status='metadata_enriched' entries have NO workspace files (research_agent found
        no SRA data) and will extract 0 samples. status='completed' entries are already processed by this tool. The tool silently
        skips entries without workspace_metadata_keys, leading to unexpected 0-sample results if wrong status used.

        FILTERING RULE: Only use filter_criteria when user EXPLICITLY asks to filter (e.g., "filter for 16S human fecal").
        If user just asks to "process entries" or "aggregate samples", use filter_criteria=None to include ALL samples.

        Filter Criteria Syntax:
            Basic: "16S human fecal CRC"
            With region: "16S V4 human fecal CRC"
            With region: "16S V3-V4 mouse gut UC"

            Supported regions: V1-V9, V3-V4, V4-V5, V1-V9, full-length

        Args:
            status_filter: Queue status (default: 'handoff_ready' - ONLY valid choice for filtering). handoff_ready = has workspace SRA files.
            filter_criteria: Sample-level filter (e.g., "16S V4 human fecal CRC"). "16S"=AMPLICON, "shotgun"=WGS/WXS/METAGENOMIC (excludes WGA),
                             "16S shotgun"=BOTH (OR logic), "V4"=amplicon region, "human"=Homo sapiens, "fecal"=isolation_source match.
                             None=no filter (default behavior).
            max_entries: Max entries to process (None=all)
            output_key: Workspace key for aggregated CSV output (default: "aggregated_filtered_samples")
            parallel_workers: Workers for parallel processing (None=sequential, >1=parallel with Rich progress if available)
            min_disease_coverage: Minimum disease coverage rate (0.0-1.0).
                Default 0.5 (50%). Set to 0.0 to disable validation.
            strict_disease_validation: If True, raise error when coverage is low.
                If False, log warning but continue. Default True.

        Returns:
            Processing summary with sample counts, validation metrics, and workspace output location for CSV export

        Examples:
            # User: "Process handoff_ready entries and aggregate all samples"
            process_metadata_queue(status_filter="handoff_ready")

            # User: "Filter for 16S human fecal samples" (default: fail if <50% disease coverage)
            process_metadata_queue(status_filter="handoff_ready", filter_criteria="16S human fecal")

            # User: "Filter for specific amplicon region"
            process_metadata_queue(status_filter="handoff_ready", filter_criteria="16S V4 human fecal CRC")

            # User: "Filter V3-V4 region only"
            process_metadata_queue(status_filter="handoff_ready", filter_criteria="16S V3-V4 human fecal")

            # User: "Filter with permissive disease validation"
            process_metadata_queue(status_filter="handoff_ready",
                                 filter_criteria="16S human fecal",
                                 strict_disease_validation=False)

            # User: "Filter with lower disease threshold"
            process_metadata_queue(status_filter="handoff_ready",
                                 filter_criteria="16S human fecal",
                                 min_disease_coverage=0.3)

            # User: "Disable disease validation entirely"
            process_metadata_queue(status_filter="handoff_ready",
                                 filter_criteria="16S human fecal",
                                 min_disease_coverage=0.0)
        """
        try:
            from lobster.services.data_access.workspace_content_service import (
                ContentType,
                MetadataContent,
            )

            queue = data_manager.publication_queue
            entries = queue.list_entries(
                status=PublicationStatus(status_filter.lower())
            )

            if max_entries:
                entries = entries[:max_entries]

            if not entries:
                logger.info(f"No entries found with status '{status_filter}'")
                return f"No entries found with status '{status_filter}'"

            # Route to parallel processing if requested and available
            if parallel_workers and parallel_workers > 1 and PROGRESS_UI_AVAILABLE:
                logger.info(
                    f"Using parallel processing: {len(entries)} entries, "
                    f"{parallel_workers} workers, Rich UI enabled"
                )
                return _process_queue_with_progress(
                    entries, filter_criteria, output_key, parallel_workers,
                    min_disease_coverage, strict_disease_validation
                )
            elif parallel_workers and parallel_workers > 1 and not PROGRESS_UI_AVAILABLE:
                logger.warning(
                    f"Parallel processing requested but Rich UI not available. "
                    f"Falling back to sequential processing."
                )

            # Log batch processing start (sequential mode)
            logger.info(
                f"Starting batch processing: {len(entries)} entries "
                f"(filter: '{filter_criteria or 'none'}', mode: sequential)"
            )

            all_samples = []
            stats = {
                "processed": 0,
                "with_samples": 0,
                "failed": 0,
                "total_extracted": 0,
                "total_valid": 0,
                "total_after_filter": 0,
                "validation_errors": 0,
                "validation_warnings": 0,
                "flag_counts": {},  # Aggregate quality flags across all samples
                "samples_needing_manual_review": [],  # Sample IDs needing body_site review
            }
            failed_entries = []

            for entry in entries:
                try:
                    if not entry.workspace_metadata_keys:
                        logger.debug(
                            f"Skipping entry {entry.entry_id}: no workspace_metadata_keys"
                        )
                        continue

                    logger.debug(
                        f"Processing entry {entry.entry_id}: '{entry.title or 'No title'}'"
                    )

                    entry_samples = []
                    entry_validation_results = []
                    entry_quality_stats = []

                    for ws_key in entry.workspace_metadata_keys:
                        # Only process SRA sample files (skip pub_queue_*_metadata/methods/identifiers.json)
                        if not (
                            ws_key.startswith("sra_") and ws_key.endswith("_samples")
                        ):
                            logger.debug(f"Skipping non-SRA workspace key: {ws_key}")
                            continue

                        logger.debug(
                            f"Processing SRA sample file: {ws_key} for entry {entry.entry_id}"
                        )

                        from lobster.services.data_access.workspace_content_service import (
                            ContentType,
                        )

                        ws_data = workspace_service.read_content(
                            ws_key, content_type=ContentType.METADATA
                        )
                        if ws_data:
                            samples, validation_result, quality_stats = (
                                _extract_samples_from_workspace(ws_data)
                            )
                            entry_samples.extend(samples)
                            entry_validation_results.append(validation_result)
                            entry_quality_stats.append(quality_stats)

                    # Update stats with validation info
                    samples_extracted = sum(
                        vr.metadata.get("total_samples", 0)
                        for vr in entry_validation_results
                    )
                    samples_valid = len(entry_samples)

                    # Aggregate quality stats for this entry
                    entry_unique_individuals = set()
                    entry_flag_counts = {}
                    for qs in entry_quality_stats:
                        # Extract individual_id strings (not the full sample dicts)
                        for s in entry_samples:
                            ind_id = s.get("_individual_id")
                            if ind_id:
                                entry_unique_individuals.add(ind_id)
                        for flag, count in qs.get("flag_counts", {}).items():
                            entry_flag_counts[flag] = (
                                entry_flag_counts.get(flag, 0) + count
                            )

                    # Check for complete validation failure
                    if samples_extracted > 0 and samples_valid == 0:
                        # All samples in this entry failed validation
                        error_count = sum(
                            len(vr.errors) for vr in entry_validation_results
                        )
                        error_msg = f"All {samples_extracted} samples failed validation ({error_count} errors)"
                        logger.error(f"Entry {entry.entry_id}: {error_msg}")

                        # Mark entry as failed
                        queue.update_status(
                            entry.entry_id,
                            entry.status,
                            handoff_status=HandoffStatus.METADATA_FAILED,
                            error=error_msg,
                        )
                        failed_entries.append((entry.entry_id, error_msg))
                        stats["failed"] += 1
                        stats["processed"] += 1
                        continue

                    stats["total_extracted"] += samples_extracted
                    stats["total_valid"] += samples_valid
                    stats["validation_errors"] += sum(
                        len(vr.errors) for vr in entry_validation_results
                    )
                    stats["validation_warnings"] += sum(
                        len(vr.warnings) for vr in entry_validation_results
                    )

                    logger.debug(
                        f"Entry {entry.entry_id}: {samples_valid}/{samples_extracted} samples valid"
                    )

                    if filter_criteria and entry_samples and metadata_filtering_service:
                        samples_before_filter = len(entry_samples)
                        parsed = metadata_filtering_service.parse_criteria(filter_criteria)
                        # Add disease validation parameters
                        parsed["min_disease_coverage"] = min_disease_coverage
                        parsed["strict_disease_validation"] = strict_disease_validation
                        metadata_filtering_service.disease_extractor = extract_disease_with_fallback
                        try:
                            entry_samples, _, _ = metadata_filtering_service.apply_filters(
                                entry_samples, parsed
                            )
                            logger.debug(
                                f"Entry {entry.entry_id}: Filter applied - "
                                f"{len(entry_samples)}/{samples_before_filter} samples retained"
                            )
                        except ValueError as e:
                            if "Insufficient disease data" in str(e):
                                # Extract coverage percentage from error message
                                coverage_match = re.search(r'(\d+\.\d+)% coverage', str(e))
                                coverage = float(coverage_match.group(1)) if coverage_match else 0.0

                                # Count cached publications (for cost/time estimate)
                                pub_groups = defaultdict(set)
                                for sample in entry_samples:
                                    pub_id = sample.get('publication_entry_id')
                                    if pub_id:
                                        pub_groups[pub_id].add(pub_id)

                                num_pubs = len(pub_groups)
                                estimated_cost = num_pubs * 0.003
                                estimated_time = num_pubs * 2

                                # Return to supervisor with suggestion (DON'T auto-enrich)
                                return f"""❌ Disease validation failed: {coverage:.1f}% coverage < 50% threshold

I can attempt automatic enrichment to extract disease from publication abstracts/methods.

**Estimated improvement**: +30-60% coverage (based on {num_pubs} cached publications)
**Cost**: ~${estimated_cost:.2f} (LLM calls)
**Time**: ~{estimated_time}s

Should I proceed with automatic enrichment?"""
                            else:
                                raise

                    stats["total_after_filter"] += len(entry_samples)

                    # Aggregate quality flags into batch stats
                    for flag, count in entry_flag_counts.items():
                        stats["flag_counts"][flag] = (
                            stats["flag_counts"].get(flag, 0) + count
                        )

                    # Track samples needing manual body_site review
                    for qs in entry_quality_stats:
                        flagged_ids = qs.get("flagged_sample_ids", {})
                        if "missing_body_site" in flagged_ids:
                            stats["samples_needing_manual_review"].extend(
                                flagged_ids["missing_body_site"]
                            )

                    # Add publication context
                    for sample in entry_samples:
                        sample["publication_entry_id"] = entry.entry_id
                        sample["publication_title"] = entry.title
                        sample["publication_doi"] = entry.doi
                        sample["publication_pmid"] = entry.pmid

                    all_samples.extend(entry_samples)
                    stats["processed"] += 1
                    if entry_samples:
                        stats["with_samples"] += 1

                    # Update entry status to terminal success state
                    queue.update_status(
                        entry.entry_id,
                        PublicationStatus.COMPLETED,  # Terminal success state
                        handoff_status=HandoffStatus.METADATA_COMPLETE,
                    )

                except Exception as e:
                    # Graceful degradation: log error but continue with other entries
                    error_msg = f"Failed to process entry {entry.entry_id}: {str(e)}"
                    logger.error(error_msg, exc_info=True)

                    # Mark entry as failed
                    queue.update_status(
                        entry.entry_id,
                        entry.status,
                        handoff_status=HandoffStatus.METADATA_FAILED,
                        error=error_msg,
                    )
                    failed_entries.append((entry.entry_id, error_msg))
                    stats["failed"] += 1
                    stats["processed"] += 1

            # Compute study-level statistics for batch effect awareness
            study_stats = defaultdict(int)
            study_to_publications = defaultdict(set)

            for sample in all_samples:
                study_id = sample.get("study_accession", "unknown")
                pub_id = sample.get("publication_entry_id", "unknown")

                study_stats[study_id] += 1
                study_to_publications[study_id].add(pub_id)

            # Identify potential batch effect risks
            total_samples = len(all_samples)
            warnings = []

            # Skip warnings if no samples or no valid study IDs
            if total_samples == 0 or not study_stats:
                warnings = []
                study_stats = {}
                multi_pub_studies = {}
            else:
                # Warning 1: Studies spanning multiple publications
                multi_pub_studies = {
                    study_id: len(pubs)
                    for study_id, pubs in study_to_publications.items()
                    if len(pubs) > 1
                }

                if multi_pub_studies:
                    study_list = ", ".join([f"{s} ({n} pubs)" for s, n in list(multi_pub_studies.items())[:3]])
                    warnings.append(
                        f"⚠️ {len(multi_pub_studies)} studies appear in multiple publications: {study_list}. "
                        f"This may indicate overlapping datasets or batch effects."
                    )

                # Warning 2: Dominant study (>50% of samples)
                dominant_studies = {
                    study_id: count
                    for study_id, count in study_stats.items()
                    if count > total_samples * 0.5
                }

                if dominant_studies:
                    for study_id, count in dominant_studies.items():
                        pct = (count / total_samples) * 100
                        warnings.append(
                            f"⚠️ Study {study_id} dominates dataset: {count}/{total_samples} samples ({pct:.1f}%). "
                            f"Consider batch effect correction using study_accession field."
                        )

                # Warning 3: Highly imbalanced study sizes (coefficient of variation > 1.5)
                if len(study_stats) > 1:
                    counts = list(study_stats.values())
                    mean_count = sum(counts) / len(counts)
                    variance = sum((x - mean_count) ** 2 for x in counts) / len(counts)
                    std_dev = variance ** 0.5
                    cv = std_dev / mean_count if mean_count > 0 else 0

                    if cv > 1.5:
                        warnings.append(
                            f"⚠️ Highly imbalanced study sizes (CV={cv:.2f}). "
                            f"Range: {min(counts)}-{max(counts)} samples per study. "
                            f"Consider stratified analysis or batch correction."
                        )

                # Log all warnings
                for warning in warnings:
                    logger.warning(warning)

            # Store aggregated results to workspace
            if all_samples:
                from datetime import datetime

                content = MetadataContent(
                    identifier=output_key,
                    content_type="filtered_samples",
                    description=f"Batch filtered samples: {filter_criteria or 'no filter'}",
                    data={
                        "samples": all_samples,
                        "filter_criteria": filter_criteria,
                        "stats": stats,
                    },
                    source="metadata_assistant",
                    cached_at=datetime.now().isoformat(),
                )
                workspace_service.write_content(content, ContentType.METADATA)

                # Also store in metadata_store for write_to_workspace access
                data_manager.metadata_store[output_key] = {
                    "samples": all_samples,
                    "filter_criteria": filter_criteria,
                    "stats": stats,
                }

            retention = (
                (stats["total_after_filter"] / stats["total_extracted"] * 100)
                if stats["total_extracted"] > 0
                else 0
            )
            validation_rate = (
                (stats["total_valid"] / stats["total_extracted"] * 100)
                if stats["total_extracted"] > 0
                else 0
            )

            # Log comprehensive batch summary (changed to DEBUG to reduce console pollution - summary is in tool response)
            logger.debug("=" * 60)
            logger.debug(f"Batch processing complete for {len(entries)} entries")
            logger.debug(f"  Successful: {stats['processed'] - stats['failed']}")
            logger.debug(f"  Failed: {stats['failed']}")
            logger.debug(f"  Total samples extracted: {stats['total_extracted']}")
            logger.debug(
                f"  Total samples valid: {stats['total_valid']} ({validation_rate:.1f}%)"
            )
            logger.debug(
                f"  Total samples after filter: {stats['total_after_filter']} ({retention:.1f}%)"
            )
            logger.debug(f"  Validation errors: {stats['validation_errors']}")
            logger.debug(f"  Validation warnings: {stats['validation_warnings']}")
            if failed_entries:
                logger.warning(f"Failed entries ({len(failed_entries)}):")
                for entry_id, error in failed_entries:
                    logger.warning(f"  - {entry_id}: {error}")
            logger.debug("=" * 60)

            # Build response - check for 0 samples scenario first
            if stats['total_extracted'] == 0:
                response = f"""⚠️ **0 Samples Extracted** (status='{status_filter}', {len(entries)} entries, {stats['with_samples']} with metadata)

Likely cause: Wrong status_filter. Use `status_filter='handoff_ready'` (default) for entries ready for processing. 'completed' entries have no actionable metadata.

Fix: `process_metadata_queue(status_filter="handoff_ready", filter_criteria="{filter_criteria or ''}", output_key="{output_key}")`
"""
                return response

            # Check if samples were extracted but ALL filtered out
            if stats['total_after_filter'] == 0 and stats['total_extracted'] > 0 and filter_criteria:
                response = f"""⚠️ **All {stats['total_extracted']} Samples Filtered Out** (filter: '{filter_criteria}')

{stats['total_valid']}/{stats['total_extracted']} valid, 0 after filter. Note: sample_type filters (fecal/gut) are NOT YET IMPLEMENTED - only 16S/shotgun and host filters work.

Fix: Run without filter first to inspect data: `process_metadata_queue(status_filter="{status_filter}", filter_criteria=None, output_key="unfiltered")`
"""
                return response

            # Normal response for successful processing
            response = f"""## Queue Processing Complete
**Entries Processed**: {stats['processed']}
**Successful**: {stats['processed'] - stats['failed']}
**Failed**: {stats['failed']}
**Entries With Samples**: {stats['with_samples']}
**Samples Extracted**: {stats['total_extracted']}
**Samples Valid**: {stats['total_valid']} ({validation_rate:.1f}%)
**Samples After Filter**: {stats['total_after_filter']}
**Retention Rate**: {retention:.1f}%
**Validation**: {stats['validation_errors']} errors, {stats['validation_warnings']} warnings
**Output Key**: {output_key}
"""

            # Add study-level statistics section
            if study_stats:
                response += "\n## Study-Level Statistics\n"
                response += f"**Unique Studies**: {len(study_stats)}\n"
                if study_stats:
                    study_counts = list(study_stats.values())
                    avg_samples = sum(study_counts) / len(study_counts)
                    response += f"**Samples Per Study**: {min(study_counts)}-{max(study_counts)} (avg: {avg_samples:.1f})\n"
                response += f"**Studies in Multiple Publications**: {len(multi_pub_studies)}\n"
                response += "\n## Batch Effect Warnings\n"

                # Append warnings if any
                if warnings:
                    for warning in warnings:
                        response += f"\n{warning}\n"
                else:
                    response += "\n✅ No major batch effect risks detected.\n"

                response += """
## Recommendations
- Use **study_accession** field for batch effect correction in downstream analysis
- Consider PERMANOVA or ComBat-seq for batch adjustment
- Stratify analyses by study if imbalanced
"""

            # Add failed entries section if any
            if failed_entries:
                response += "\n### Failed Entries\n"
                for entry_id, error in failed_entries:
                    response += f"- {entry_id}: {error}\n"

            # Add manual inspection summary for samples missing body_site info
            samples_needing_review = stats.get("samples_needing_manual_review", [])
            if samples_needing_review:
                response += f"\n### ⚠️ Manual Inspection Needed ({len(samples_needing_review)} samples)\n"
                response += "The following samples are missing body_site/tissue type metadata and may require manual verification:\n"
                # Show up to 10 sample IDs, then summarize
                if len(samples_needing_review) <= 10:
                    for sample_id in samples_needing_review:
                        response += f"- {sample_id}\n"
                else:
                    for sample_id in samples_needing_review[:10]:
                        response += f"- {sample_id}\n"
                    response += f"- ... and {len(samples_needing_review) - 10} more\n"
                response += "\n**Recommended action**: Check the original publication methods section or SRA metadata for sample type information (fecal, tissue, oral, etc.).\n"

            # Add quality flag summary if present
            flag_counts = stats.get("flag_counts", {})
            if flag_counts:
                response += "\n### Quality Flag Summary\n"
                for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
                    response += f"- {flag}: {count} samples\n"

            response += f'\nUse `write_to_workspace(identifier="{output_key}", workspace="metadata", output_format="csv")` to export as CSV.\n'

            return response
        except Exception as e:
            logger.error(f"Error processing queue: {e}")
            return f"❌ Error processing queue: {str(e)}"

    @tool
    def update_metadata_status(
        entry_id: str,
        handoff_status: str = None,
        error_message: str = None,
    ) -> str:
        """
        Manually update metadata processing status for a queue entry.

        Use this to mark entries as complete, reset failed entries, or add error notes.

        Args:
            entry_id: Publication queue entry ID
            handoff_status: New handoff status (not_ready, ready_for_metadata, metadata_in_progress, metadata_complete)
            error_message: Optional error message to log

        Returns:
            Confirmation of status update
        """
        try:
            queue = data_manager.publication_queue
            entry = queue.get_entry(entry_id)

            update_kwargs = {"entry_id": entry_id, "status": entry.status}

            if handoff_status:
                update_kwargs["handoff_status"] = HandoffStatus(handoff_status)
            if error_message:
                update_kwargs["error"] = error_message

            queue.update_status(**update_kwargs)

            return (
                f"✓ Updated {entry_id}: handoff_status={handoff_status or 'unchanged'}"
            )
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return f"❌ Error updating status: {str(e)}"

    @tool
    def enrich_samples_with_disease(
        workspace_key: str,
        enrichment_mode: str = "hybrid",
        manual_mappings: Optional[str] = None,
        confidence_threshold: float = 0.8,
        auto_retry_validation: bool = True,
        dry_run: bool = False
    ) -> str:
        """
        Enrich samples with missing disease annotation using cached publication context.

        Use this tool when disease validation fails (<50% coverage) to automatically
        extract disease information from publication abstracts/methods, or to apply
        user-provided disease mappings.

        Args:
            workspace_key: Metadata workspace key containing samples to enrich.
                Typically "aggregated_filtered_samples" from process_metadata_queue.
                Can also be specific dataset keys like "sra_PRJNA123456_samples".

            enrichment_mode: Strategy for enrichment (default: "hybrid")
                - "column_scan": Only re-scan dataset metadata columns for missed fields
                - "llm_auto": Only use LLM extraction from cached publications
                - "manual": Only apply provided manual_mappings
                - "hybrid": Try all phases sequentially (recommended)

            manual_mappings: JSON string mapping publication IDs to diseases.
                Example: '{"pub_queue_doi_10_1234": "crc", "pub_queue_pmid_5678": "uc"}'
                Used in "manual" or "hybrid" mode as fallback.

            confidence_threshold: Minimum LLM confidence (0.0-1.0) to accept extraction.
                Default: 0.8 (high confidence required).
                Lower values accept more extractions but with higher error risk.

            auto_retry_validation: If True, automatically re-run disease validation
                after enrichment to check if 50% threshold is now met.
                If False, only enrich and report statistics.

            dry_run: If True, preview enrichment results without saving changes.
                Useful for testing enrichment logic before applying.

        Returns:
            Enrichment report with:
            - Phase-by-phase results (samples enriched per phase)
            - Coverage improvement (before → after)
            - Confidence distribution
            - Validation result (if auto_retry enabled)
            - Publication-level breakdown
            - Samples still missing disease (if any)

        Example usage:
            # Automatic enrichment (recommended)
            enrich_samples_with_disease(
                workspace_key="aggregated_filtered_samples",
                enrichment_mode="hybrid"
            )

            # Manual mappings only
            enrich_samples_with_disease(
                workspace_key="aggregated_filtered_samples",
                enrichment_mode="manual",
                manual_mappings='{"pub_queue_doi_10_1234": "crc", "pub_queue_pmid_5678": "uc"}'
            )

            # Preview without saving
            enrich_samples_with_disease(
                workspace_key="aggregated_filtered_samples",
                enrichment_mode="hybrid",
                dry_run=True
            )
        """
        # Step 1: Load samples from workspace
        if workspace_key not in data_manager.metadata_store:
            return f"❌ Workspace key '{workspace_key}' not found in metadata_store"

        workspace_data = data_manager.metadata_store[workspace_key]
        samples = workspace_data.get('samples', [])

        if not samples:
            return f"❌ No samples found in workspace key '{workspace_key}'"

        # Step 2: Calculate initial coverage
        initial_count = sum(1 for s in samples if s.get('disease'))
        initial_coverage = (initial_count / len(samples)) * 100 if samples else 0

        # Step 3: Parse manual mappings
        mappings_dict = {}
        if manual_mappings:
            try:
                mappings_dict = json.loads(manual_mappings)
            except json.JSONDecodeError as e:
                return f"❌ Invalid manual_mappings JSON: {e}"

        # Step 4: Run enrichment phases
        report = [
            f"## Disease Enrichment Report",
            f"**Workspace Key**: {workspace_key}",
            f"**Total Samples**: {len(samples)}",
            f"**Initial Coverage**: {initial_coverage:.1f}% ({initial_count}/{len(samples)})",
            f"**Mode**: {enrichment_mode}",
            f"**Dry Run**: {dry_run}",
            f"",
            f"### Enrichment Phases",
            f""
        ]

        total_enriched = 0

        # Make a copy of samples for dry_run to avoid modifying original
        working_samples = [s.copy() for s in samples] if dry_run else samples

        # Phase 1: Column re-scan (always run unless manual-only mode)
        if enrichment_mode in ["column_scan", "hybrid"]:
            report.append("#### Phase 1: Column Re-scan")
            count, log = _phase1_column_rescan(working_samples)
            total_enriched += count
            report.extend(log)
            report.append(f"**Phase 1 Result**: +{count} samples")
            report.append("")

        # Phase 2: LLM abstract extraction (unless manual-only mode)
        if enrichment_mode in ["llm_auto", "hybrid"]:
            report.append("#### Phase 2: LLM Abstract Extraction")
            count, log = _phase2_llm_abstract_extraction(
                working_samples,
                data_manager,
                confidence_threshold,
                llm
            )
            total_enriched += count
            report.extend(log)
            report.append(f"**Phase 2 Result**: +{count} samples")
            report.append("")

            # Phase 3: LLM methods extraction (only if hybrid and still low coverage)
            current_coverage = ((initial_count + total_enriched) / len(samples)) * 100
            if current_coverage < 50.0 and enrichment_mode == "hybrid":
                report.append("#### Phase 3: LLM Methods Extraction (triggered by low coverage)")
                count, log = _phase3_llm_methods_extraction(
                    working_samples,
                    data_manager,
                    confidence_threshold,
                    llm
                )
                total_enriched += count
                report.extend(log)
                report.append(f"**Phase 3 Result**: +{count} samples")
                report.append("")

        # Phase 4: Manual mappings
        if enrichment_mode in ["manual", "hybrid"] and mappings_dict:
            report.append("#### Phase 4: Manual Mappings")
            count, log = _phase4_manual_mappings(
                working_samples,
                mappings_dict
            )
            total_enriched += count
            report.extend(log)
            report.append(f"**Phase 4 Result**: +{count} samples")
            report.append("")

        # Step 5: Calculate final coverage
        final_count = sum(1 for s in working_samples if s.get('disease'))
        final_coverage = (final_count / len(samples)) * 100 if samples else 0
        improvement = final_coverage - initial_coverage

        report.append(f"### Summary")
        report.append(f"**Initial Coverage**: {initial_coverage:.1f}% ({initial_count} samples)")
        report.append(f"**Final Coverage**: {final_coverage:.1f}% ({final_count} samples)")
        report.append(f"**Improvement**: +{improvement:.1f}% (+{total_enriched} samples)")
        report.append(f"**Validation Threshold**: 50.0%")
        report.append("")

        # Step 6: Save enriched samples (if not dry_run)
        if not dry_run and total_enriched > 0:
            # Update metadata_store with enriched samples
            data_manager.metadata_store[workspace_key] = workspace_data
            report.append("✅ **Changes saved** to metadata_store")

            # Also save to workspace file
            try:
                from lobster.services.data_access.workspace_content_service import (
                    ContentType,
                    MetadataContent,
                    WorkspaceContentService
                )

                workspace_service = WorkspaceContentService(data_manager)
                content = MetadataContent(
                    identifier=workspace_key,
                    content_type="enriched_samples",
                    description=f"Disease-enriched samples: +{total_enriched} samples",
                    data=workspace_data,
                    source="metadata_assistant",
                    cached_at=datetime.now().isoformat(),
                )
                workspace_service.write_content(content, ContentType.METADATA)
                report.append("✅ **Changes saved** to workspace/metadata/")
            except Exception as e:
                logger.warning(f"Failed to save to workspace file: {e}")
                report.append(f"⚠️ **Warning**: Saved to metadata_store but workspace save failed: {e}")

            report.append("")
        elif dry_run:
            report.append("ℹ️ **Dry run**: Changes NOT saved (preview only)")
            report.append("")

        # Step 7: Auto-retry validation (if enabled and not dry_run)
        if auto_retry_validation and not dry_run and total_enriched > 0:
            report.append("### Re-validation")

            if final_coverage >= 50.0:
                report.append(f"✅ **Validation PASSED**: {final_coverage:.1f}% coverage ≥ 50% threshold")
                report.append("You can now proceed with filtering using this enriched dataset.")
            else:
                missing_count = len(samples) - final_count
                report.append(f"❌ **Validation FAILED**: {final_coverage:.1f}% coverage < 50% threshold")
                report.append(f"**Samples still missing disease**: {missing_count}")
                report.append("")
                report.append("**Next Steps**:")
                report.append("  1. Provide manual mappings for remaining publications")
                report.append(f"  2. Lower threshold: min_disease_coverage={final_coverage/100:.2f}")
                report.append("  3. Skip disease filtering (omit disease terms from filter_criteria)")

        return "\n".join(report)

    # Helper functions for queue processing
    def _extract_samples_from_workspace(
        ws_data,
    ) -> tuple[list, "ValidationResult", dict]:
        """
        Extract sample records from workspace data with validation and quality flagging.

        Uses SRASampleSchema for validation and compute_sample_completeness for
        soft flagging. All valid samples are returned with quality flags attached
        (soft filtering - user decides what to exclude).

        Handles various data structures produced by WorkspaceContentService:
        - Nested: {"data": {"samples": [...]}}
        - Direct: {"samples": [...]}
        - Dict-based: {"samples": {"id1": {...}, "id2": {...}}}

        Quality Flagging:
        - Each sample gets _quality_score (0-100) and _quality_flags fields
        - Flags indicate potential concerns (missing individual_id, controls, etc.)
        - Samples are NOT filtered out - user reviews flags and decides

        Args:
            ws_data: Workspace data dictionary (from WorkspaceContentService)

        Returns:
            (valid_samples, validation_result, quality_stats): Tuple containing:
            - valid_samples: List of valid samples with _quality_score and _quality_flags
            - validation_result: ValidationResult with batch statistics
            - quality_stats: Dict with completeness stats and flag counts

        Examples:
            >>> ws_data = {"data": {"samples": [{"run_accession": "SRR001", ...}]}}
            >>> samples, result, stats = _extract_samples_from_workspace(ws_data)
            >>> len(samples)  # All valid samples (not filtered)
            1
            >>> samples[0]["_quality_score"]  # Completeness score
            80.0
            >>> stats["unique_individuals"]
            45
        """
        from lobster.core.schemas.sra import (
            SRASampleSchema,
            compute_sample_completeness,
            validate_sra_sample,
            validate_sra_samples_batch,
        )

        # Extract raw samples from workspace structure
        raw_samples = []
        if isinstance(ws_data, dict):
            if "samples" in ws_data:
                samples = ws_data["samples"]
                if isinstance(samples, list):
                    raw_samples = samples
                elif isinstance(samples, dict):
                    # Convert dict to list of samples
                    raw_samples = [{"sample_id": k, **v} for k, v in samples.items()]
            if "data" in ws_data and isinstance(ws_data["data"], dict):
                # Recursive extraction for nested structure
                nested_samples, nested_result, _ = _extract_samples_from_workspace(
                    ws_data["data"]
                )
                raw_samples.extend(nested_samples)

        # If no samples extracted, return empty results
        if not raw_samples:
            empty_result = ValidationResult()
            empty_result.metadata["total_samples"] = 0
            empty_result.metadata["valid_samples"] = 0
            empty_result.metadata["validation_rate"] = 0.0
            empty_stats = {
                "total_samples": 0,
                "avg_completeness": 0.0,
                "unique_individuals": 0,
                "completeness_distribution": {"high": [], "medium": [], "low": []},
                "flag_counts": {},
            }
            return [], empty_result, empty_stats

        # Validate extracted samples using unified schema system
        validation_result = validate_sra_samples_batch(raw_samples)

        # Process valid samples with quality flagging (soft filtering)
        valid_samples = []
        quality_stats = {
            "total_samples": 0,
            "avg_completeness": 0.0,
            "unique_individuals": 0,
            "completeness_distribution": {"high": [], "medium": [], "low": []},
            "flag_counts": {},
            "flagged_sample_ids": {},
            "individuals": set(),
            "validation_errors": [],  # Track errors for aggregated summary
        }
        scores = []

        for sample in raw_samples:
            sample_result = validate_sra_sample(sample)
            if sample_result.is_valid:  # No critical errors
                # Validate and compute completeness
                try:
                    validated = SRASampleSchema.from_dict(sample)
                    score, flags = compute_sample_completeness(validated)

                    # Attach quality info to sample (soft flagging)
                    sample["_quality_score"] = score
                    sample["_quality_flags"] = [f.value for f in flags]

                    # Also attach heuristically extracted fields for downstream use
                    if validated.individual_id:
                        sample["_individual_id"] = validated.individual_id
                    if validated.timepoint:
                        sample["_timepoint"] = validated.timepoint
                    if validated.timepoint_numeric is not None:
                        sample["_timepoint_numeric"] = validated.timepoint_numeric

                    valid_samples.append(sample)
                    scores.append(score)

                    # Track statistics
                    if validated.individual_id:
                        quality_stats["individuals"].add(validated.individual_id)

                    # Categorize by completeness
                    run_acc = validated.run_accession
                    if score >= 80:
                        quality_stats["completeness_distribution"]["high"].append(
                            run_acc
                        )
                    elif score >= 50:
                        quality_stats["completeness_distribution"]["medium"].append(
                            run_acc
                        )
                    else:
                        quality_stats["completeness_distribution"]["low"].append(
                            run_acc
                        )

                    # Track flag counts
                    for flag in flags:
                        flag_name = flag.value
                        quality_stats["flag_counts"][flag_name] = (
                            quality_stats["flag_counts"].get(flag_name, 0) + 1
                        )
                        if flag_name not in quality_stats["flagged_sample_ids"]:
                            quality_stats["flagged_sample_ids"][flag_name] = []
                        quality_stats["flagged_sample_ids"][flag_name].append(run_acc)

                except Exception as e:
                    logger.warning(f"Error computing completeness for sample: {e}")
                    sample["_quality_score"] = 0.0
                    sample["_quality_flags"] = ["validation_error"]
                    valid_samples.append(sample)
                    scores.append(0.0)

                # Log validation warnings at DEBUG level (avoids console spam)
                # These warnings are already tracked in quality_stats["flag_counts"]
                for warning in sample_result.warnings:
                    logger.debug(warning)
            else:
                # Track validation errors by type (aggregated, not per-sample)
                # These are data quality issues, not system errors
                for error in sample_result.errors:
                    quality_stats["validation_errors"].append(error)
                    logger.debug(f"Sample validation failed: {error}")

        # Log aggregated validation error summary (once per batch, not per sample)
        if quality_stats["validation_errors"]:
            # Group errors by type for concise summary
            error_types: Dict[str, int] = {}
            for error in quality_stats["validation_errors"]:
                # Extract error type from message (e.g., "Field 'organism_name':" -> "organism_name")
                if "Field '" in error:
                    field = error.split("Field '")[1].split("'")[0]
                    error_types[f"missing_{field}"] = error_types.get(f"missing_{field}", 0) + 1
                elif "No download URLs" in error:
                    error_types["no_download_url"] = error_types.get("no_download_url", 0) + 1
                else:
                    error_types["other"] = error_types.get("other", 0) + 1

            # Log summary at WARNING level (data quality issue, not system error)
            invalid_count = len(raw_samples) - len(valid_samples)
            error_summary = ", ".join(f"{k}: {v}" for k, v in sorted(error_types.items()))
            logger.warning(
                f"Validation summary: {invalid_count} samples excluded "
                f"({error_summary})"
            )

        # Finalize quality stats
        quality_stats["total_samples"] = len(valid_samples)
        quality_stats["avg_completeness"] = sum(scores) / len(scores) if scores else 0.0
        quality_stats["unique_individuals"] = len(quality_stats["individuals"])
        # Convert set to count (can't serialize set to JSON)
        del quality_stats["individuals"]

        logger.debug(
            f"Sample extraction complete: {len(valid_samples)}/{len(raw_samples)} valid "
            f"({validation_result.metadata.get('validation_rate', 0):.1f}%)"
        )
        logger.debug(
            f"Quality summary: avg_completeness={quality_stats['avg_completeness']:.1f}, "
            f"unique_individuals={quality_stats['unique_individuals']}, "
            f"high/medium/low={len(quality_stats['completeness_distribution']['high'])}/"
            f"{len(quality_stats['completeness_distribution']['medium'])}/"
            f"{len(quality_stats['completeness_distribution']['low'])}"
        )
        if quality_stats["flag_counts"]:
            logger.debug(f"Flag counts: {quality_stats['flag_counts']}")

        return valid_samples, validation_result, quality_stats

    # =========================================================================
    # Helper Functions for filter_samples_by
    # =========================================================================
    # NOTE: _parse_filter_criteria and _apply_metadata_filters have been moved to
    # lobster/services/metadata/metadata_filtering_service.py (MetadataFilteringService)
    # =========================================================================

    def _combine_analysis_steps(
        irs: List[AnalysisStep], operation: str, description: str
    ) -> AnalysisStep:
        """
        Combine multiple AnalysisSteps into a composite IR.

        Args:
            irs: List of individual AnalysisSteps
            operation: Composite operation name
            description: Composite description

        Returns:
            Composite AnalysisStep
        """
        if not irs:
            # Return minimal IR if no steps
            return AnalysisStep(
                operation=operation,
                tool_name="filter_samples_by",
                description=description,
                library="lobster.agents.metadata_assistant",
                code_template="# No filtering operations applied",
                imports=[],
                parameters={},
                parameter_schema={},
                input_entities=[],
                output_entities=[],
            )

        # Combine code templates
        combined_code = "\n\n".join(
            [
                f"# Step {i+1}: {ir.description}\n{ir.code_template}"
                for i, ir in enumerate(irs)
            ]
        )

        # Combine imports (deduplicate)
        all_imports = []
        for ir in irs:
            all_imports.extend(ir.imports)
        combined_imports = list(set(all_imports))

        # Combine parameters
        combined_params = {}
        for i, ir in enumerate(irs):
            combined_params[f"step_{i+1}"] = ir.parameters

        return AnalysisStep(
            operation=operation,
            tool_name="filter_samples_by",
            description=description,
            library="lobster.agents.metadata_assistant",
            code_template=combined_code,
            imports=combined_imports,
            parameters=combined_params,
            parameter_schema={},
            input_entities=[{"type": "workspace_metadata", "name": "input_metadata"}],
            output_entities=[{"type": "filtered_metadata", "name": "output_metadata"}],
        )

    def _format_filtering_report(
        workspace_key: str,
        filter_criteria: str,
        parsed_criteria: Dict[str, Any],
        original_count: int,
        final_count: int,
        retention_rate: float,
        stats_list: List[Dict[str, Any]],
        filtered_metadata: "pd.DataFrame",
    ) -> str:
        """
        Format filtering report as markdown.

        Args:
            workspace_key: Workspace key
            filter_criteria: Original criteria string
            parsed_criteria: Parsed criteria dict
            original_count: Original sample count
            final_count: Final sample count
            retention_rate: Retention percentage
            stats_list: List of per-filter statistics
            filtered_metadata: Final filtered DataFrame

        Returns:
            Markdown-formatted report
        """
        status_icon = (
            "✅" if retention_rate > 50 else ("⚠️" if retention_rate > 10 else "❌")
        )

        report_lines = [
            f"{status_icon} Sample Filtering Complete\n",
            f"**Workspace Key**: {workspace_key}",
            f"**Filter Criteria**: {filter_criteria}",
            f"**Original Samples**: {original_count}",
            f"**Filtered Samples**: {final_count}",
            f"**Retention Rate**: {retention_rate:.1f}%\n",
            "## Filters Applied\n",
        ]

        # List applied filters
        if parsed_criteria["check_16s"]:
            report_lines.append("- ✓ 16S amplicon detection")
        if parsed_criteria["host_organisms"]:
            report_lines.append(
                f"- ✓ Host organism: {', '.join(parsed_criteria['host_organisms'])}"
            )
        if parsed_criteria["sample_types"]:
            report_lines.append(
                f"- ✓ Sample type: {', '.join(parsed_criteria['sample_types'])}"
            )
        if parsed_criteria["standardize_disease"]:
            report_lines.append("- ✓ Disease standardization")

        # Add per-filter statistics
        if stats_list:
            report_lines.append("\n## Filter Statistics\n")
            for i, stats in enumerate(stats_list, 1):
                report_lines.append(f"**Filter {i}**:")
                for key, value in stats.items():
                    if key not in ["is_valid"]:  # Skip boolean flags
                        report_lines.append(f"  - {key}: {value}")

        # Add sample preview
        report_lines.append("\n## Filtered Metadata Preview\n")
        if not filtered_metadata.empty:
            # Show first 5 samples
            preview = filtered_metadata.head(5).to_markdown(index=True)
            report_lines.append(preview)
            if len(filtered_metadata) > 5:
                report_lines.append(
                    f"\n*... and {len(filtered_metadata) - 5} more samples*"
                )
        else:
            report_lines.append("*No samples passed filtering criteria*")

        # Add recommendation
        report_lines.append("\n## Recommendation\n")
        if retention_rate > 50:
            report_lines.append(
                f"✅ **Good retention rate** ({retention_rate:.1f}%) - Filtered dataset is suitable for analysis"
            )
        elif retention_rate > 10:
            report_lines.append(
                f"⚠️ **Moderate retention rate** ({retention_rate:.1f}%) - Consider relaxing filter criteria or verify input data quality"
            )
        else:
            report_lines.append(
                f"❌ **Low retention rate** ({retention_rate:.1f}%) - Most samples filtered out. Review criteria or input data"
            )

        return "\n".join(report_lines)

    # =========================================================================
    # Tool 10: Custom Code Execution for Sample-Level Operations
    # =========================================================================

    # execute_custom_code is created via factory at line ~273 (v2.7+: unified tool)
    # See create_execute_custom_code_tool() in lobster/tools/custom_code_tool.py
    # Post-processor: metadata_store_post_processor handles sample persistence

    # =========================================================================
    # Tool Registry
    # =========================================================================

    tools = [
        map_samples_by_id,
        read_sample_metadata,
        standardize_sample_metadata,
        validate_dataset_content,
        # Publication queue processing tools
        process_metadata_entry,
        process_metadata_queue,
        update_metadata_status,
        enrich_samples_with_disease,
        # Shared workspace tools
        get_content_from_workspace,
        write_to_workspace,
        # Custom code execution for sample-level operations
        execute_custom_code,
    ]

    if MICROBIOME_FEATURES_AVAILABLE:
        tools.append(filter_samples_by)

    # =========================================================================
    # System Prompt
    # =========================================================================

    system_prompt = """Identity and Role
You are the Metadata Assistant – an internal sample metadata and harmonization copilot. You never interact with end users or the supervisor. You only respond to instructions from:
	-	the research agent, and
	-	the data expert.

<your environment>
You are a langgraph agent in a supervisor-multi-agent architecture within the open-core python package called 'lobster-ai' (referred as lobster) developed by the company Omics-OS (www.omics-os.com) founded by Kevin Yar.
</your environment>

Hierarchy: supervisor > research agent == data expert >> metadata assistant.

Your responsibilities:
	-	Read and summarize sample metadata from cached tables or loaded modalities.
	-	Filter samples according to explicit criteria (assay, host, sample type, disease, etc.).
	-	Standardize metadata into requested schemas (transcriptomics, proteomics, microbiome).
	-	Map samples across datasets based on IDs or metadata.
	-	Validate dataset content and report quality metrics and limitations.
	-	Enrich samples with missing disease annotation using enrich_samples_with_disease tool:
		- 4-phase hierarchy: column re-scan → LLM abstract → LLM methods → manual mappings
		- Triggered when disease validation fails (<50% coverage threshold)
		- Full provenance tracking (disease_source, disease_confidence, disease_evidence)

You are not responsible for:
	-	Discovering or searching for datasets or publications.
	-	Downloading files or loading data into modalities.
	-	Running omics analyses (QC, alignment, normalization, clustering, DE).
	-	Changing or relaxing the user's filters or criteria.

Operating Principles
	1.	Strict source_type and target_type

	-	Every tool call you make must explicitly specify source_type and, where applicable, target_type.
	-	Allowed values are “metadata_store” and “modality”.
	-	“metadata_store” refers to cached metadata tables and artifacts (for example keys such as metadata_GSE12345_samples or metadata_GSE12345_samples_filtered_16S_human_fecal).
	-	“modality” refers to already loaded data modalities provided by the data expert.
	-	If an instruction does not clearly indicate which source_type and target_type you should use, you must treat this as a missing prerequisite and fail fast with an explanation.

	2.	Trust cache first

	-	Prefer operating on cached metadata in metadata_store or workspace keys provided by the research agent or data expert.
	-	Only operate on modalities when explicitly instructed to use source_type=“modality”.
	-	Never attempt to discover new datasets, publications, or files.

	3.	Follow instructions exactly

	-	Parse all filter criteria provided by the research agent or data expert into structured constraints:
	-	assay or technology (16S, shotgun, RNA-seq)
	-	amplicon region (V4, V3-V4, full-length) [v0.5.0+]
	-	host organism (Human, Mouse)
	-	sample type (fecal_stool, gut_luminal_content, gut_mucosal_biopsy, oral, skin) [v0.5.0+]
	-	disease or condition (crc, uc, cd, healthy)
	-	Do not broaden, relax, or reinterpret the requested criteria.
	-	If filters would eliminate nearly all samples, report clearly and suggest alternatives, but never change criteria yourself.

	4.	Structured, data-rich outputs

	-	All responses must use a consistent, compact sectioned format so the research agent and data expert can parse results reliably:
	-	Status: short code or phrase (for example success, partial, failed).
	-	Summary: 2–4 sentences describing what you did and the main outcome.
	-	Metrics: explicit numbers and percentages (for example mapping rate, field coverage, sample retention, confidence).
	-	Key Findings: a small set of bullet-like lines or short paragraphs highlighting the most important technical observations.
	-	Recommendation: one of “proceed”, “proceed with caveats”, or “stop”, plus a brief rationale.
	-	Returned Artifacts: list of workspace or metadata_store keys, schema names, or other identifiers that downstream agents should use next.
	-	Use concise language; avoid verbose narrative and speculation.

	5.	Never overstep

	-	Do not:
	-	search for datasets or publications,
	-	download or load any files,
	-	run omics analyses (QC, normalization, clustering, DE).
	-	If instructions require data that is missing (for example a workspace key that does not exist or a modality that is not loaded), fail fast:
	-	Clearly state which key, modality, or parameter is missing.
	-	Explain what the research agent or data expert must cache or load next to allow you to proceed.

	6.	Parameter Type Conventions

	CRITICAL: When calling tools with optional parameters:
	-	To skip an optional parameter, OMIT it entirely from the tool call.
	-	DO NOT pass string values like 'null', 'None', 'undefined', or empty strings for omitted parameters.
	-	Integer parameters (max_entries, limit, offset) must be actual integers or omitted completely.
	-	String parameters must be actual non-empty strings or omitted completely.

	Examples:
	-	WRONG: process_metadata_queue(max_entries='null')
	-	WRONG: process_metadata_queue(filter_criteria='null')
	-	WRONG: process_metadata_queue(output_key='None')
	-	CORRECT: process_metadata_queue()
	-	CORRECT: process_metadata_queue(max_entries=10)
	-	CORRECT: process_metadata_queue(max_entries=0)
	-	CORRECT: process_metadata_queue(filter_criteria="16S V4 human fecal_stool CRC")
	-	CORRECT: process_metadata_queue(status_filter="handoff_ready", output_key="filtered_samples")

	7.	Efficient Workspace Navigation

	CRITICAL: Avoid context overflow when discovering metadata keys:
	-	NEVER call get_content_from_workspace(workspace="metadata") without filters (returns 1000+ items)
	-	ALWAYS use the pattern parameter to narrow scope: get_content_from_workspace(workspace="metadata", pattern="aggregated_*")
	-	Parse output_key from tool responses (e.g., "**Output Key**: my_samples") and use directly
	-	For targeted discovery, use execute_custom_code to list metadata_store keys

	Examples:
	-	WRONG: get_content_from_workspace(workspace="metadata") # Returns 1294 items!
	-	CORRECT: get_content_from_workspace(workspace="metadata", pattern="aggregated_*") # Returns ~50 items
	-	CORRECT: get_content_from_workspace(workspace="metadata", pattern="sra_prjna*") # Returns ~10 items
	-	CORRECT: execute_custom_code(python_code="result = {{'keys': [k for k in metadata_store.keys() if 'aggregated' in k]}}") # Targeted discovery

Tool Selection Priority
For publication queue processing requests, follow this decision tree:

	1.	Publication Queue Processing (ALWAYS START HERE)

	Use process_metadata_queue when request involves:
	- "process publication queue" or "process handoff_ready entries"
	- "aggregate samples from publications"
	- "filter 16S" or "filter shotgun" or "filter microbiome samples"
	- "create export table" or "export to CSV"

	This tool:
	- Reads workspace_metadata_keys (sra_*_samples files) from ALL entries
	- Aggregates SAMPLE-LEVEL metadata (run_accession, biosample, organism, etc.)
	- Applies filter_criteria at SAMPLE level using MicrobiomeFilteringService
	- Validates with SRASampleSchema + quality scoring
	- Outputs 5,000-10,000+ sample rows ready for CSV export

	Example:
	```
	process_metadata_queue(
	    status_filter="handoff_ready",
	    filter_criteria="16S V4 human fecal_stool CRC",
	    output_key="aggregated_samples",
	    parallel_workers=4  # For >50 entries
	)
	```

	2.	Disease Enrichment (WHEN VALIDATION FAILS)

	Use enrich_samples_with_disease when:
	- Disease validation fails (<50% coverage threshold)
	- Supervisor confirms enrichment should proceed
	- Need to extract disease from publication abstracts/methods

	This tool (4-phase hierarchy):
	- Phase 1: Column re-scan (checks ALL columns for missed disease fields)
	- Phase 2: LLM abstract extraction (extracts from cached publication abstracts)
	- Phase 3: LLM methods extraction (if still <50% after Phase 2)
	- Phase 4: Manual mappings (user-provided JSON as fallback)

	Example:
	```
	enrich_samples_with_disease(
	    workspace_key="aggregated_filtered_samples",
	    enrichment_mode="hybrid",
	    confidence_threshold=0.8
	)
	```

	Decision Flow:
	```
	process_metadata_queue → Disease validation fails (<50%)
	         ↓
	  Ask supervisor for permission
	         ↓
	  enrich_samples_with_disease (hybrid mode)
	         ↓
	  Re-validate → ≥50%? → Continue filtering
	  Re-validate → <50%? → Suggest manual mappings
	```

	3.	Custom Code Execution (LAST RESORT)

	Use execute_custom_code ONLY when:
	- Standard tools insufficient (complex study-specific logic)
	- Non-disease enrichment (age, sex, tissue from publication)
	- Custom calculations not covered by standard tools

	NOT for:
	- Disease enrichment (use enrich_samples_with_disease instead)
	- CSV export preparation (write_to_workspace handles automatically)

Behavioral Rules for Modern Features (v0.5.1+)

	1.	Disease Validation Thresholds
	- Minimum coverage: 50% of samples must have disease annotation
	- Confidence threshold: 0.8 for LLM-extracted diseases (configurable)
	- When validation fails: Ask supervisor for enrichment permission
	- After enrichment: Auto-retry validation if coverage improves

	2.	Amplicon Region Syntax
	Filter criteria supports explicit amplicon regions:
	- "16S V4 human fecal_stool" → enforces V4 region only
	- "16S V3-V4 mouse gut_mucosal_biopsy" → enforces V3-V4
	- Valid regions: V1-V9, V3-V4, V4-V5, V1-V9, full-length
	- Prevents mixing regions (systematic bias in diversity estimates)

	3.	Sample Type Categories (v0.5.0+)
	Modern categories (biologically distinct):
	- fecal_stool (distal colon, passed stool)
	- gut_luminal_content (intestinal lumen, not passed)
	- gut_mucosal_biopsy (tissue-associated microbiome)
	- gut_lavage (bowel prep artifacts)
	- oral, skin (unchanged)

	Legacy aliases work with deprecation warnings:
	- "fecal" → "fecal_stool" (warning logged)
	- "luminal" → "gut_luminal_content"
	- "biopsy" → "gut_mucosal_biopsy"
	- "gut" → ValueError (too ambiguous)

	4.	Parallel Processing
	For process_metadata_queue with >50 entries:
	- Use parallel_workers=4 for optimal performance
	- Batch flush reduces I/O by 20x
	- Rich progress UI shows real-time status

	5.	Quality Flags Interpretation
	Quality flags are SOFT filters (don't auto-exclude):
	- MISSING_HEALTH_STATUS: Expected (70-85% of SRA samples)
	- NON_HUMAN_HOST: Should recommend exclusion
	- CONTROL_SAMPLE: Analyze separately from experimental samples
	- User decides final inclusion based on flags

Export Best Practice
**CORRECT Pattern**: Direct export after aggregation
```
process_metadata_queue(output_key='aggregated_samples')
         ↓
write_to_workspace(identifier='aggregated_samples', output_format='csv', export_mode='rich')
```
**Result**: 3 harmonized files in exports/ (rich CSV, strict CSV, audit TSV)

**Anti-Pattern**: Using execute_custom_code for export preparation (NOT NEEDED - write_to_workspace handles harmonization)

Quality Improvement Workflow (v0.5.1+)

When disease coverage is low after aggregation:
	1.	Assessment: Check coverage in process_metadata_queue report
	2.	Enrichment: If <50%, validation fails → use enrich_samples_with_disease
	3.	Re-validation: Tool auto-retries validation after enrichment
	4.	Decision: ≥50% → proceed; <50% → manual mappings or lower threshold

For non-disease fields (age, sex, tissue):
	- Use execute_custom_code with publication context extraction
	- Document source: field_source="inferred_from_methods"
	- Only extract explicit statements (no inference)

Execution Pattern
	1.	Confirm prerequisites

	-	For every incoming instruction from the research agent or data expert:
	-	Check that all referenced workspace or metadata_store keys exist.
	-	Check that any referenced modalities exist when source_type=“modality” is requested.
	-	Check that required parameters are present:
	-	source_type,
	-	target_type (when applicable),
	-	the filter criteria or target schema names,
	-	identifiers and keys for the datasets involved.
	-	If any prerequisite is missing:
	-	Respond with:
	-	Status: failed.
	-	Summary: explicitly state which key, modality, or parameter is missing.
	-	Metrics: only if applicable; otherwise minimal.
	-	Key Findings: list specific missing prerequisites.
	-	Recommendation: stop, and describe what the research agent or data expert must do to fix the issue.
	-	Returned Artifacts: existing keys if they are relevant, otherwise empty.

	2.	Execute requested tools

	-	For complex pipelines:
	-	Chain operations (for example filter_samples_by → standardize_sample_metadata → validate_dataset_content) in the requested order.
	-	Pass along the output keys from one step as inputs to the next step.
	-	For multi-step filtering:
	-	Run filter_samples_by in stages for each group of criteria, referencing the previous stage’s key as the new source.
	-	Track which filters are responsible for the largest reductions in sample count.

	3.	Persist outputs

	-	Whenever a tool produces new metadata or derived subsets:
	-	Persist the result in metadata_store or the appropriate workspace using clear, descriptive names.
	-	Follow and respect the naming conventions used by the research agent, such as:
	-	metadata_GSE12345_samples for full sample metadata.
	-	metadata_GSE12345_samples_filtered_16S_V4_human_fecal_stool_CRC for filtered subsets.
	-	standardized_GSE12345_transcriptomics for standardized metadata in a transcriptomics schema.
	-	In every response:
	-	In the Returned Artifacts section, list all new keys or schema names along with short descriptions of each artifact.

	4.	Close with explicit recommendations

	-	Every response must end with:
	-	A Recommendation value:
	-	proceed: the data is suitable for the intended next analysis or integration.
	-	proceed with caveats: the data is usable but with important limitations you describe clearly.
	-	stop: major problems make the requested next step unsafe, misleading, or impossible.
	-	Next-step guidance, such as:
	-	ready for standardization,
	-	ready for sample-level integration,
	-	cohort-level integration recommended due to mapping/coverage issues,
	-	needs additional age or sex metadata,
	-	research agent should refine dataset selection,
	-	data expert should download or reload data after specific conditions are met.

Quality Bars and Shared Thresholds
You must align your thresholds and semantics with those used by the research agent so the system behaves consistently.

Field coverage
	-	Report coverage per field (for example sample_id, condition, tissue, age, sex, batch).
	-	Flag any required field with coverage <80% as a significant limitation.
	-	Describe how missing fields affect analysis (for example missing batch or age fields may limit correction for confounders).
	-	Your Recommendation must reflect the impact of coverage gaps.

Filtering
	-	Always report:
	-	Original number of samples and retained number of samples.
	-	Retention percentage.
	-	Point out which filters caused the largest drops.
	-	If retention is very low (for example <30% of original samples), consider recommending:
	-	alternative filter strategies, or
	-	alternative datasets, depending on the instruction. 
	-	You still must not change any criteria yourself; instead, explain the consequences and required changes back to the research agent or data expert.

Validation semantics
	-	For validate_dataset_content and related quality checks:
	-	Mark each check (sample counts, condition coverage, duplicates, controls) as PASS or FAIL.
	-	Assign severity (minor, moderate, major) that corresponds to the practical impact:
	-	issues analogous to “CRITICAL” at the dataset level should push you toward a stop recommendation,
	-	moderate issues toward proceed with caveats,
	-	minor issues toward proceed.
	-	Make it clear why you recommend proceed, proceed with caveats, or stop.

Interaction with the Research Agent and Data Expert
	-	Research agent:
	-	Will primarily send you instructions referencing metadata_store keys and workspace names it has created (for example metadata_GSE12345_samples, metadata_GSE67890_samples_filtered_case_control, standardized_GSE12345_transcriptomics).
	-	Uses your Metrics, Key Findings, Recommendation, and Returned Artifacts to:
	-	decide whether sample-level or cohort-level integration is appropriate,
	-	advise the supervisor on whether datasets are ready for download and analysis by the data expert,
	-	determine whether additional metadata processing is required.
	-	Be precise and quantitative in your Metrics and Key Findings to support these decisions.
	-	Data expert:
	-	May request validations or transformations on modalities or newly loaded datasets.
	-	Will often use source_type=“modality” and target_type set to either “modality” or “metadata_store”, depending on whether results should be persisted back to metadata_store.
	-	Your structured outputs help the data expert decide whether to proceed with integration or specific analyses.

Style
	-	No user-facing dialog:
	-	Never speak directly to the end user or the supervisor.
	-	Never ask clarifying questions; instead, fail fast when prerequisites are missing and explain what is needed.
	-	Respond only to the research agent and data expert.
	-	Stay concise and data-focused:
	-	Use short sentences.
	-	Emphasize metrics, coverage, mapping rates, and concrete observations.
	-	Avoid speculation; base statements only on the data you have seen.
	-	Always respect and preserve filter criteria received from upstream agents; you may warn about their consequences, but you never alter them.

todays date: {current_date}
"""

    formatted_prompt = system_prompt.format(current_date=datetime.today().isoformat())

    # Import AgentState for state_schema
    from lobster.agents.state import AgentState

    # Add delegation tools if provided
    if delegation_tools:
        tools = tools + delegation_tools

    # Create LangGraph agent
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=formatted_prompt,
        name=agent_name,
        state_schema=AgentState,
    )
