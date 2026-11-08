"""
Core exceptions for Lobster bioinformatics platform.

This module provides a comprehensive exception hierarchy for handling
errors throughout the data loading and analysis pipeline.
"""

from typing import Any, Dict, Literal, Optional

RefusalCategory = Literal[
    "context", "auth", "policy", "budget", "provider", "accounting"
]


class HdfReadRefusal(Exception):
    """Terminal refusal of the built-in pandas-HDF alternative, not HDF admission.

    Propagating callers must preserve this type without logging raw chains or
    source paths. Earlier unrelated successful loads may remain; this does not
    promise whole-operation rollback or confinement of ordinary HDF readers.
    """

    def __init__(self):
        self.reason = "pandas_hdf_not_admitted"
        self.retryable = False
        super().__init__("Pandas HDF loading is not supported.")


class InferenceRefused(Exception):
    """A terminal inference refusal that must not become a scientific fallback.

    Only a fixed category and retryability are exposed. Provider exception text,
    request payloads and credential values must never be passed as a message.
    Callers may retry only under the same authorized inference identity.
    """

    def __init__(self, category: RefusalCategory, *, retryable: bool = False):
        if category not in {
            "context",
            "auth",
            "policy",
            "budget",
            "provider",
            "accounting",
        }:
            raise ValueError("Unknown inference refusal category")
        self.category = category
        self.retryable = retryable
        super().__init__(f"Inference refused ({category}).")


def reraise_inference_refusal(error: BaseException) -> None:
    """Preserve a refusal hidden by a wrapper or executor exception group.

    Scientific recovery handlers must call this before logging an exception or
    returning a fallback. Inspect only typed exception links, never their text.
    """
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, InferenceRefused):
            raise current from None
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)


class LobsterCoreError(Exception):
    """Base exception for all Lobster core errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        return self.message


class UnsupportedFormatError(LobsterCoreError):
    """
    Raised when a file format cannot be loaded by an adapter.

    This exception provides structured information about:
    - The file path that failed to load
    - The error that occurred
    - Suggestions for resolution

    Example:
        try:
            adata = adapter.load_from_file(path)
        except UnsupportedFormatError as e:
            print(f"Cannot load file: {e.message}")
            for suggestion in e.details.get('suggestions', []):
                print(f"  - {suggestion}")
    """

    pass


class UnsupportedPlatformError(LobsterCoreError):
    """
    Raised when a GEO dataset uses an unsupported platform (e.g., microarray).

    This exception is raised BEFORE downloading files to save time and bandwidth.
    Provides early validation based on platform metadata (GPL IDs).

    Attributes:
        message: Human-readable error message
        details: Contains:
            - geo_id: Dataset identifier
            - unsupported_platforms: List of (GPL_ID, platform_title) tuples
            - platform_type: "microarray", "methylation_array", or both joined
              with "+" when a series mixes them (also given as "modality")
            - explanation: Why this platform is not supported
            - detected_platforms: Formatted string of detected platforms
            - suggestions: List of alternative approaches

    Example:
        try:
            geo_service.download_dataset(geo_id="GSE48452")
        except UnsupportedPlatformError as e:
            print(f"Platform: {e.details['platform_type']}")
            print(f"Detected: {e.details['detected_platforms']}")
            for suggestion in e.details['suggestions']:
                print(f"  → {suggestion}")
    """

    pass


class FeatureNotImplementedError(LobsterCoreError):
    """
    Raised when a requested feature exists but is not yet integrated.

    Distinct from NotImplementedError because:
    - The functionality may already exist in the codebase
    - A clear workaround or timeline is provided
    - The feature is on the roadmap

    Attributes:
        message: Human-readable error message
        details: Contains:
            - explanation: Why this feature is not yet available
            - current_workaround: How to work around the limitation
            - suggestions: List of alternative approaches
            - estimated_implementation: Development effort required
            - github_issue: Link to tracking issue

    Example:
        try:
            geo_service.download_dataset(geo_id="GSE130036")
        except FeatureNotImplementedError as e:
            print(f"Feature: {e.message}")
            print(f"Workaround: {e.details['current_workaround']}")
            print(f"Timeline: {e.details.get('estimated_implementation')}")
    """

    pass


class DataTypeAmbiguityError(LobsterCoreError):
    """
    Raised when data type (single-cell vs bulk) cannot be determined automatically.

    This exception signals that the agent should request user clarification.
    It provides structured information to help the user make an informed decision.

    Attributes:
        message: Human-readable error message
        details: Contains:
            - geo_id: Dataset identifier
            - shape: Data dimensions (rows, cols)
            - signals: Detection signals and their interpretations
            - confidence: Detection confidence score (0.0-1.0)
            - suggestions: List of valid data_type values to try
            - recommendation: Guidance for user decision

    Example:
        try:
            geo_service.download_dataset(geo_id="GSE123456")
        except DataTypeAmbiguityError as e:
            # Agent formats question for user
            print(f"Ambiguous data type for {e.details['geo_id']}")
            print(f"Shape: {e.details['shape']}")
            print(f"Confidence: {e.details['confidence']:.0%}")
            print("\nDetection signals:")
            for signal, value in e.details['signals'].items():
                print(f"  - {signal}: {value}")
            print("\nPlease specify:")
            for suggestion in e.details['suggestions']:
                print(f"  - {suggestion}")
    """

    pass


class DataOrientationError(LobsterCoreError):
    """
    Raised when data orientation validation fails after loading.

    This exception indicates that the final AnnData object has an unexpected
    shape that suggests incorrect transpose logic or corrupted data.

    Attributes:
        message: Human-readable error message
        details: Contains:
            - data_type: Expected data type ('bulk' or 'single_cell')
            - actual_shape: (n_obs, n_vars) tuple
            - expected_constraints: What was expected
            - transpose_info: Information about transpose decisions made
            - suggestions: How to fix the issue

    Example:
        try:
            adata = adapter.from_source(df, data_type='bulk')
        except DataOrientationError as e:
            print(f"Orientation error: {e.message}")
            print(f"Actual shape: {e.details['actual_shape']}")
            print(f"Expected: {e.details['expected_constraints']}")
    """

    pass
