"""
Disease Standardization Service

Normalizes disease/condition terminology and filters samples by type (fecal vs tissue).
Designed for IBD microbiome studies (CRC, UC, CD, Healthy controls).
"""

from typing import Dict, List, Tuple, Any, Optional, Set
import pandas as pd
import re
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec


class DiseaseStandardizationService:
    """
    Service for standardizing disease terminology and filtering by sample type.

    Key capabilities:
    - Maps messy disease labels to standard categories (CRC, UC, CD, Healthy)
    - 5-level fuzzy matching (exact → contains → reverse → token → unmapped)
    - Sample type filtering (fecal, biopsy, tissue, etc.)
    - Provenance tracking for all transformations
    """

    # Standard disease mappings (case-insensitive)
    DISEASE_MAPPINGS = {
        # Colorectal Cancer (CRC)
        "crc": ["crc", "colorectal cancer", "colon cancer", "rectal cancer",
                "colorectal carcinoma", "colon carcinoma", "rectal carcinoma",
                "adenocarcinoma", "tumor", "tumour", "cancer"],

        # Ulcerative Colitis (UC)
        "uc": ["uc", "ulcerative colitis", "colitis ulcerosa", "ulcerative_colitis"],

        # Crohn's Disease (CD)
        "cd": ["cd", "crohn's disease", "crohns disease", "crohn disease",
               "crohn's", "crohns", "crohn"],

        # Healthy Controls
        "healthy": ["healthy", "control", "normal", "non-ibd", "non ibd",
                   "non-diseased", "non diseased", "healthy control",
                   "normal control", "ctrl"]
    }

    # Sample type mappings
    SAMPLE_TYPE_MAPPINGS = {
        "fecal": ["fecal", "feces", "stool", "faecal", "faeces", "fecal sample"],
        "gut": ["gut", "intestinal", "colon", "colonic", "rectal", "ileal",
                "biopsy", "tissue", "mucosal", "mucosa"]
    }

    def __init__(self):
        """Initialize the disease standardization service."""
        # Build reverse lookup for faster matching
        self._reverse_disease_map = {}
        for standard_term, variants in self.DISEASE_MAPPINGS.items():
            for variant in variants:
                self._reverse_disease_map[variant.lower()] = standard_term

        self._reverse_sample_map = {}
        for sample_type, variants in self.SAMPLE_TYPE_MAPPINGS.items():
            for variant in variants:
                self._reverse_sample_map[variant.lower()] = sample_type

    def standardize_disease_terms(
        self,
        metadata: pd.DataFrame,
        disease_column: str = "disease"
    ) -> Tuple[pd.DataFrame, Dict[str, Any], AnalysisStep]:
        """
        Standardize disease terminology in metadata.

        Args:
            metadata: DataFrame with disease information
            disease_column: Column name containing disease labels

        Returns:
            Tuple of (standardized_metadata, statistics, provenance_ir)
        """
        if disease_column not in metadata.columns:
            raise ValueError(f"Disease column '{disease_column}' not found in metadata")

        result = metadata.copy()
        original_col = f"{disease_column}_original"
        result[original_col] = result[disease_column]

        # Track standardization statistics
        mapping_stats = {
            "exact_matches": 0,
            "contains_matches": 0,
            "reverse_contains_matches": 0,
            "token_matches": 0,
            "unmapped": 0
        }
        unique_mappings = {}

        # Standardize each term
        standardized_values = []
        for value in result[disease_column]:
            std_value, match_type = self._fuzzy_match(str(value), self.DISEASE_MAPPINGS)
            standardized_values.append(std_value)

            # Track statistics (unmapped doesn't have _matches suffix)
            key = match_type if match_type == "unmapped" else f"{match_type}_matches"
            mapping_stats[key] += 1
            if str(value).lower() != std_value.lower():
                unique_mappings[str(value)] = std_value

        result[disease_column] = standardized_values

        # Build statistics
        total = len(result)
        standardization_rate = (
            (total - mapping_stats["unmapped"]) / total * 100
            if total > 0 else 0.0
        )

        stats = {
            "total_samples": total,
            "standardization_rate": standardization_rate,
            "mapping_stats": mapping_stats,
            "unique_mappings": unique_mappings,
            "disease_distribution": result[disease_column].value_counts().to_dict(),
            "original_column": original_col
        }

        # Create provenance IR
        ir = self._create_standardization_ir(
            disease_column=disease_column,
            original_column=original_col,
            mapping_stats=mapping_stats,
            unique_mappings=unique_mappings
        )

        return result, stats, ir

    def filter_by_sample_type(
        self,
        metadata: pd.DataFrame,
        sample_types: List[str],
        sample_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any], AnalysisStep]:
        """
        Filter samples by sample type (fecal vs gut tissue).

        Args:
            metadata: DataFrame with sample information
            sample_types: List of sample types to keep (e.g., ["fecal"])
            sample_columns: Columns to check for sample type info
                           (if None, checks common columns)

        Returns:
            Tuple of (filtered_metadata, statistics, provenance_ir)
        """
        # Auto-detect sample columns if not specified
        if sample_columns is None:
            common_cols = ["sample_type", "tissue", "body_site", "sample_source",
                          "biosample_type", "material"]
            sample_columns = [col for col in common_cols if col in metadata.columns]

            if not sample_columns:
                raise ValueError(
                    "No sample type columns found. Specify sample_columns explicitly."
                )

        result = metadata.copy()
        result["sample_type_filter_match"] = False
        result["matched_column"] = ""
        result["matched_value"] = ""

        # Normalize requested sample types
        normalized_types = [t.lower() for t in sample_types]

        # Check each sample against all columns
        for idx, row in result.iterrows():
            for col in sample_columns:
                if pd.isna(row[col]):
                    continue

                # Try to match this value to requested sample types
                matched_type, _ = self._fuzzy_match(
                    str(row[col]),
                    {t: self.SAMPLE_TYPE_MAPPINGS.get(t, [t]) for t in normalized_types}
                )

                if matched_type in normalized_types:
                    result.at[idx, "sample_type_filter_match"] = True
                    result.at[idx, "matched_column"] = col
                    result.at[idx, "matched_value"] = str(row[col])
                    break

        # Filter to matching samples
        filtered = result[result["sample_type_filter_match"]].copy()
        filtered = filtered.drop(columns=["sample_type_filter_match"])

        # Build statistics
        stats = {
            "original_samples": len(metadata),
            "filtered_samples": len(filtered),
            "retention_rate": len(filtered) / len(metadata) * 100 if len(metadata) > 0 else 0,
            "sample_types_requested": sample_types,
            "columns_checked": sample_columns,
            "matches_by_column": {
                col: filtered[filtered["matched_column"] == col].shape[0]
                for col in sample_columns
            }
        }

        # Create provenance IR
        ir = self._create_filter_ir(
            sample_types=sample_types,
            sample_columns=sample_columns,
            original_samples=len(metadata),
            filtered_samples=len(filtered)
        )

        return filtered, stats, ir

    def _fuzzy_match(
        self,
        value: str,
        mappings: Dict[str, List[str]]
    ) -> Tuple[str, str]:
        """
        5-level fuzzy matching strategy.

        Args:
            value: Input string to match
            mappings: Dictionary of {standard_term: [variants]}

        Returns:
            Tuple of (matched_standard_term, match_type)

        Match types:
            - exact: Direct match in variant list
            - contains: Value contains a variant term
            - reverse_contains: Variant term contains value
            - token: Individual tokens match
            - unmapped: No match found (returns original value)
        """
        normalized_value = self._normalize_term(value)

        # Level 1: Exact match
        for standard, variants in mappings.items():
            if normalized_value in [self._normalize_term(v) for v in variants]:
                return standard, "exact"

        # Level 2: Value contains variant
        for standard, variants in mappings.items():
            for variant in variants:
                normalized_variant = self._normalize_term(variant)
                if normalized_variant in normalized_value:
                    return standard, "contains"

        # Level 3: Variant contains value (reverse)
        for standard, variants in mappings.items():
            for variant in variants:
                normalized_variant = self._normalize_term(variant)
                if normalized_value in normalized_variant:
                    return standard, "reverse_contains"

        # Level 4: Token-based matching
        value_tokens = set(self._tokenize(normalized_value))
        best_match = None
        best_overlap = 0

        for standard, variants in mappings.items():
            for variant in variants:
                variant_tokens = set(self._tokenize(self._normalize_term(variant)))
                overlap = len(value_tokens & variant_tokens)

                if overlap > 0 and overlap > best_overlap:
                    best_match = standard
                    best_overlap = overlap

        if best_match:
            return best_match, "token"

        # Level 5: No match found
        return value, "unmapped"

    def _normalize_term(self, term: str) -> str:
        """
        Normalize a term for matching.

        - Lowercase
        - Remove punctuation
        - Collapse whitespace
        """
        term = str(term).lower()
        term = re.sub(r"[^\w\s]", " ", term)  # Replace punctuation with space
        term = re.sub(r"\s+", " ", term).strip()  # Collapse whitespace
        return term

    def _tokenize(self, term: str) -> List[str]:
        """Split term into tokens (words)."""
        return [t for t in term.split() if len(t) > 1]  # Ignore single-char tokens

    def _create_standardization_ir(
        self,
        disease_column: str,
        original_column: str,
        mapping_stats: Dict[str, int],
        unique_mappings: Dict[str, str]
    ) -> AnalysisStep:
        """Create provenance IR for disease standardization."""
        return AnalysisStep(
            operation="disease_standardization",
            tool_name="standardize_disease_terms",
            description=f"Standardize disease terminology in '{disease_column}' column using 5-level fuzzy matching",
            library="custom",
            imports=[],
            code_template="""
# Disease standardization (manual implementation)
metadata['{{ original_column }}'] = metadata['{{ disease_column }}']

# Map disease terms using standardized categories
disease_mappings = {{ disease_mappings }}
for idx, value in metadata['{{ disease_column }}'].items():
    # Apply fuzzy matching logic (5 levels)
    standardized = fuzzy_match(value, disease_mappings)
    metadata.at[idx, '{{ disease_column }}'] = standardized
""",
            parameters={
                "disease_column": disease_column,
                "original_column": original_column,
                "disease_mappings": self.DISEASE_MAPPINGS,
                "mapping_stats": mapping_stats,
                "unique_mappings": unique_mappings
            },
            parameter_schema={
                "disease_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="disease",
                    required=True,
                    description="Column containing disease labels"
                ),
                "original_column": ParameterSpec(
                    param_type="str",
                    papermill_injectable=False,
                    default_value="disease_original",
                    required=False,
                    description="Backup column with original values"
                )
            },
            input_entities=["metadata"],
            output_entities=["standardized_metadata"],
            execution_context={
                "standardization_rate": (
                    100 - (mapping_stats["unmapped"] / sum(mapping_stats.values()) * 100)
                    if sum(mapping_stats.values()) > 0 else 0.0
                ),
                "total_mappings": len(unique_mappings)
            }
        )

    def _create_filter_ir(
        self,
        sample_types: List[str],
        sample_columns: List[str],
        original_samples: int,
        filtered_samples: int
    ) -> AnalysisStep:
        """Create provenance IR for sample type filtering."""
        return AnalysisStep(
            operation="sample_type_filter",
            tool_name="filter_by_sample_type",
            description=f"Filter samples by type: {', '.join(sample_types)}",
            library="custom",
            imports=[],
            code_template="""
# Sample type filtering
sample_columns = {{ sample_columns }}
sample_types = {{ sample_types }}

# Check each column for matching sample types
metadata['keep'] = False
for col in sample_columns:
    if col in metadata.columns:
        metadata['keep'] |= metadata[col].str.lower().str.contains('|'.join(sample_types), na=False)

filtered_metadata = metadata[metadata['keep']].drop(columns=['keep'])
""",
            parameters={
                "sample_types": sample_types,
                "sample_columns": sample_columns,
                "retention_rate": (filtered_samples / original_samples * 100) if original_samples > 0 else 0
            },
            parameter_schema={
                "sample_types": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=["fecal"],
                    required=True,
                    description="Sample types to retain"
                ),
                "sample_columns": ParameterSpec(
                    param_type="List[str]",
                    papermill_injectable=True,
                    default_value=[],
                    required=False,
                    description="Columns to check for sample type information"
                )
            },
            input_entities=["metadata"],
            output_entities=["filtered_metadata"],
            execution_context={
                "original_samples": original_samples,
                "filtered_samples": filtered_samples,
                "retention_rate": (filtered_samples / original_samples * 100) if original_samples > 0 else 0
            }
        )
