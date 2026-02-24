"""
ChEMBL API service for compound search, bioactivity retrieval, and
target-compound association queries.

Wraps the ChEMBL REST API (https://www.ebi.ac.uk/chembl/api/data) with
synchronous httpx calls. All methods return 3-tuples
(None, Dict, AnalysisStep) since this is an API-only service that does
not modify AnnData objects.

All methods handle HTTP errors gracefully and return error information
in the stats dict rather than raising exceptions to the caller.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import httpx

from lobster.agents.drug_discovery.config import (
    CHEMBL_API_BASE,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_SEARCH_LIMIT,
)
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class ChEMBLService:
    """
    Stateless service for querying the ChEMBL REST API.

    Provides compound search, bioactivity retrieval, and target-compound
    association lookups. Each method returns (None, stats_dict, AnalysisStep)
    following the Lobster AI service pattern.
    """

    def __init__(self) -> None:
        """Initialize the ChEMBL service (stateless)."""
        logger.debug("Initializing stateless ChEMBLService")

    # =========================================================================
    # IR builders
    # =========================================================================

    def _create_ir_search_compounds(
        self, query: str, limit: int
    ) -> AnalysisStep:
        """Create IR for compound search."""
        return AnalysisStep(
            operation="chembl.molecule.search",
            tool_name="search_compounds",
            description=f"Search ChEMBL for compounds matching '{query}'",
            library="httpx",
            code_template="""# Search ChEMBL for compounds
import httpx

response = httpx.get(
    "{{ base_url }}/molecule/search",
    params={"q": {{ query | tojson }}, "format": "json", "limit": {{ limit }}},
    timeout={{ timeout }},
)
response.raise_for_status()
data = response.json()
molecules = data.get("molecules", [])
print(f"Found {len(molecules)} compounds matching '{{ query }}'")""",
            imports=["import httpx"],
            parameters={
                "query": query,
                "limit": limit,
                "base_url": CHEMBL_API_BASE,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "query": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Compound name, synonym, or SMILES to search",
                ),
                "limit": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=DEFAULT_SEARCH_LIMIT,
                    required=False,
                    validation_rule="limit > 0",
                    description="Maximum number of results to return",
                ),
            },
            input_entities=[],
            output_entities=["compound_results"],
        )

    def _create_ir_get_bioactivity(
        self,
        chembl_id: str,
        target_chembl_id: Optional[str],
    ) -> AnalysisStep:
        """Create IR for bioactivity retrieval."""
        return AnalysisStep(
            operation="chembl.activity.get",
            tool_name="get_bioactivity",
            description=(
                f"Retrieve bioactivity data for {chembl_id}"
                + (f" against target {target_chembl_id}" if target_chembl_id else "")
            ),
            library="httpx",
            code_template="""# Get bioactivity data from ChEMBL
import httpx

params = {
    "molecule_chembl_id": {{ chembl_id | tojson }},
    "format": "json",
    "limit": 100,
}
{% if target_chembl_id %}"params["target_chembl_id"] = {{ target_chembl_id | tojson }}{% endif %}
response = httpx.get("{{ base_url }}/activity", params=params, timeout={{ timeout }})
response.raise_for_status()
data = response.json()
activities = data.get("activities", [])
print(f"Found {len(activities)} bioactivity records for {{ chembl_id }}")""",
            imports=["import httpx"],
            parameters={
                "chembl_id": chembl_id,
                "target_chembl_id": target_chembl_id,
                "base_url": CHEMBL_API_BASE,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "chembl_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="ChEMBL molecule ID (e.g., CHEMBL25)",
                ),
                "target_chembl_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="Optional target ChEMBL ID to filter activities",
                ),
            },
            input_entities=[],
            output_entities=["bioactivity_results"],
        )

    def _create_ir_get_target_compounds(
        self,
        target_chembl_id: str,
        activity_type: str,
        limit: int,
    ) -> AnalysisStep:
        """Create IR for target compound lookup."""
        return AnalysisStep(
            operation="chembl.activity.by_target",
            tool_name="get_target_compounds",
            description=(
                f"Find compounds tested against target {target_chembl_id} "
                f"with {activity_type} data"
            ),
            library="httpx",
            code_template="""# Find compounds tested against a ChEMBL target
import httpx

params = {
    "target_chembl_id": {{ target_chembl_id | tojson }},
    "standard_type": {{ activity_type | tojson }},
    "format": "json",
    "limit": {{ limit }},
}
response = httpx.get("{{ base_url }}/activity", params=params, timeout={{ timeout }})
response.raise_for_status()
data = response.json()
activities = data.get("activities", [])
print(f"Found {len(activities)} {{{ activity_type | tojson }}} records for target {{ target_chembl_id }}")""",
            imports=["import httpx"],
            parameters={
                "target_chembl_id": target_chembl_id,
                "activity_type": activity_type,
                "limit": limit,
                "base_url": CHEMBL_API_BASE,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "target_chembl_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="ChEMBL target ID (e.g., CHEMBL202)",
                ),
                "activity_type": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="IC50",
                    required=False,
                    description="Activity type filter (IC50, Ki, EC50, etc.)",
                ),
                "limit": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=50,
                    required=False,
                    validation_rule="limit > 0",
                    description="Maximum number of activity records to return",
                ),
            },
            input_entities=[],
            output_entities=["target_compound_results"],
        )

    # =========================================================================
    # Internal HTTP helper
    # =========================================================================

    # Shared headers for ChEMBL API requests
    _HEADERS = {
        "Accept": "application/json",
        "User-Agent": "lobster-ai/1.0 (https://github.com/the-omics-os/lobster)",
    }

    def _get_json(
        self,
        url: str,
        params: Dict[str, Any],
        max_retries: int = 2,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute a GET request with retry logic for timeouts and 5xx errors.

        Retries up to *max_retries* times with exponential backoff plus jitter.
        Client errors (4xx) are never retried.  The httpx client is reused
        across retries to benefit from connection pooling.

        Returns:
            Tuple of (json_data, error_message). Exactly one will be None.
        """
        import random
        import time

        last_error: Optional[str] = None
        with httpx.Client(
            timeout=DEFAULT_HTTP_TIMEOUT, headers=self._HEADERS
        ) as client:
            for attempt in range(max_retries + 1):
                try:
                    response = client.get(url, params=params)
                    response.raise_for_status()
                    # Validate content-type before parsing
                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type:
                        msg = (
                            f"ChEMBL API returned non-JSON response "
                            f"({content_type}) for {url}"
                        )
                        if attempt < max_retries:
                            last_error = msg
                            logger.warning(
                                "%s (attempt %d/%d)",
                                msg,
                                attempt + 1,
                                max_retries + 1,
                            )
                        else:
                            logger.error(msg)
                            return None, msg
                    else:
                        try:
                            return response.json(), None
                        except (ValueError, json.JSONDecodeError) as exc:
                            msg = (
                                f"ChEMBL API returned invalid JSON for {url}: {exc}"
                            )
                            logger.error(msg)
                            return None, msg
                except httpx.TimeoutException:
                    last_error = (
                        f"ChEMBL API timeout after {DEFAULT_HTTP_TIMEOUT}s for {url}"
                    )
                    logger.warning(
                        "%s (attempt %d/%d)",
                        last_error,
                        attempt + 1,
                        max_retries + 1,
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code >= 500 and attempt < max_retries:
                        last_error = (
                            f"ChEMBL API HTTP {exc.response.status_code} for {url}"
                        )
                        logger.warning(
                            "%s (attempt %d/%d)",
                            last_error,
                            attempt + 1,
                            max_retries + 1,
                        )
                    else:
                        # 4xx errors or final attempt — don't retry
                        msg = (
                            f"ChEMBL API HTTP {exc.response.status_code} for {url}: "
                            f"{exc.response.text[:200]}"
                        )
                        logger.error(msg)
                        return None, msg
                except httpx.RequestError as exc:
                    msg = f"ChEMBL API request error for {url}: {exc}"
                    logger.error(msg)
                    return None, msg
                if attempt < max_retries:
                    time.sleep(2 ** (attempt + 1) + random.uniform(0, 1))

        logger.error(last_error)
        return None, last_error

    # =========================================================================
    # Public methods
    # =========================================================================

    def search_compounds(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Search ChEMBL for molecules by name, synonym, or SMILES fragment.

        Calls GET {CHEMBL_API_BASE}/molecule/search with the query string
        and returns parsed compound records including ChEMBL IDs, names,
        molecular properties, and structure data.

        Args:
            query: Compound name, synonym, or SMILES fragment to search.
            limit: Maximum number of results (default 20).

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
            On error, stats_dict contains an ``error`` key with details.
        """
        logger.info(f"Searching ChEMBL compounds: query='{query}', limit={limit}")

        ir = self._create_ir_search_compounds(query, limit)

        url = f"{CHEMBL_API_BASE}/molecule/search.json"
        params: Dict[str, Any] = {"q": query, "limit": limit, "format": "json"}

        data, error = self._get_json(url, params)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "query": query,
                "n_results": 0,
                "analysis_type": "chembl_compound_search",
            }
            return None, stats, ir

        molecules_raw: List[Dict[str, Any]] = data.get("molecules", [])

        # Extract a clean summary for each molecule
        compounds: List[Dict[str, Any]] = []
        for mol in molecules_raw:
            props = mol.get("molecule_properties") or {}
            compound = {
                "chembl_id": mol.get("molecule_chembl_id", ""),
                "pref_name": mol.get("pref_name", ""),
                "molecule_type": mol.get("molecule_type", ""),
                "max_phase": mol.get("max_phase", 0),
                "molecular_weight": props.get("full_mwt"),
                "alogp": props.get("alogp"),
                "hba": props.get("hba"),
                "hbd": props.get("hbd"),
                "psa": props.get("psa"),
                "ro5_violations": props.get("num_ro5_violations"),
                "canonical_smiles": (
                    mol.get("molecule_structures", {}) or {}
                ).get("canonical_smiles", ""),
            }
            compounds.append(compound)

        stats = {
            "query": query,
            "n_results": len(compounds),
            "compounds": compounds,
            "has_more": data.get("page_meta", {}).get("next") is not None,
            "analysis_type": "chembl_compound_search",
        }

        logger.info(
            f"ChEMBL search complete: {len(compounds)} compounds found for '{query}'"
        )
        return None, stats, ir

    def get_bioactivity(
        self,
        chembl_id: str,
        target_chembl_id: Optional[str] = None,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Retrieve bioactivity data (IC50, Ki, EC50, etc.) for a compound.

        Calls GET {CHEMBL_API_BASE}/activity filtered by molecule ChEMBL ID
        and optionally by target ChEMBL ID.

        Args:
            chembl_id: ChEMBL molecule identifier (e.g., "CHEMBL25").
            target_chembl_id: Optional target ChEMBL ID to filter results.

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
            stats_dict[``activities``] contains a list of activity records.
        """
        logger.info(
            f"Getting bioactivity for {chembl_id}"
            + (f" vs target {target_chembl_id}" if target_chembl_id else "")
        )

        ir = self._create_ir_get_bioactivity(chembl_id, target_chembl_id)

        url = f"{CHEMBL_API_BASE}/activity.json"
        params: Dict[str, Any] = {
            "molecule_chembl_id": chembl_id,
            "limit": 100,
            "format": "json",
        }
        if target_chembl_id:
            params["target_chembl_id"] = target_chembl_id

        data, error = self._get_json(url, params)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "chembl_id": chembl_id,
                "target_chembl_id": target_chembl_id,
                "n_activities": 0,
                "analysis_type": "chembl_bioactivity",
            }
            return None, stats, ir

        activities_raw: List[Dict[str, Any]] = data.get("activities", [])

        # Parse and summarize activity records
        activities: List[Dict[str, Any]] = []
        activity_type_counts: Dict[str, int] = {}
        for act in activities_raw:
            std_type = act.get("standard_type", "")
            activity_type_counts[std_type] = activity_type_counts.get(std_type, 0) + 1

            record = {
                "activity_id": act.get("activity_id"),
                "molecule_chembl_id": act.get("molecule_chembl_id", ""),
                "target_chembl_id": act.get("target_chembl_id", ""),
                "target_pref_name": act.get("target_pref_name", ""),
                "target_organism": act.get("target_organism", ""),
                "standard_type": std_type,
                "standard_value": act.get("standard_value"),
                "standard_units": act.get("standard_units", ""),
                "standard_relation": act.get("standard_relation", ""),
                "pchembl_value": act.get("pchembl_value"),
                "assay_type": act.get("assay_type", ""),
                "assay_description": act.get("assay_description", ""),
            }
            activities.append(record)

        stats = {
            "chembl_id": chembl_id,
            "target_chembl_id": target_chembl_id,
            "n_activities": len(activities),
            "activity_type_counts": activity_type_counts,
            "activities": activities,
            "analysis_type": "chembl_bioactivity",
        }

        logger.info(
            f"Bioactivity retrieval complete: {len(activities)} records for {chembl_id}"
        )
        return None, stats, ir

    def get_target_compounds(
        self,
        target_chembl_id: str,
        activity_type: str = "IC50",
        limit: int = 50,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Find compounds tested against a specific ChEMBL target.

        Calls GET {CHEMBL_API_BASE}/activity filtered by target ChEMBL ID
        and activity measurement type (e.g., IC50, Ki, EC50).

        Args:
            target_chembl_id: ChEMBL target identifier (e.g., "CHEMBL202").
            activity_type: Standard activity measurement type (default "IC50").
            limit: Maximum number of activity records (default 50).

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
            stats_dict[``compounds``] contains unique compound summaries
            with their best activity values.
        """
        logger.info(
            f"Searching compounds for target {target_chembl_id}: "
            f"type={activity_type}, limit={limit}"
        )

        ir = self._create_ir_get_target_compounds(
            target_chembl_id, activity_type, limit
        )

        url = f"{CHEMBL_API_BASE}/activity.json"
        params: Dict[str, Any] = {
            "target_chembl_id": target_chembl_id,
            "standard_type": activity_type,
            "limit": limit,
            "format": "json",
        }

        data, error = self._get_json(url, params)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "target_chembl_id": target_chembl_id,
                "activity_type": activity_type,
                "n_activities": 0,
                "n_unique_compounds": 0,
                "analysis_type": "chembl_target_compounds",
            }
            return None, stats, ir

        activities_raw: List[Dict[str, Any]] = data.get("activities", [])

        # Aggregate by compound, keeping the best (lowest) value for each
        compound_map: Dict[str, Dict[str, Any]] = {}
        for act in activities_raw:
            mol_id = act.get("molecule_chembl_id", "")
            if not mol_id:
                continue

            std_value = act.get("standard_value")
            try:
                numeric_value = float(std_value) if std_value is not None else None
            except (ValueError, TypeError):
                numeric_value = None

            pchembl = act.get("pchembl_value")
            try:
                pchembl_float = float(pchembl) if pchembl is not None else None
            except (ValueError, TypeError):
                pchembl_float = None

            if mol_id not in compound_map:
                compound_map[mol_id] = {
                    "molecule_chembl_id": mol_id,
                    "molecule_pref_name": act.get("molecule_pref_name", ""),
                    "best_value": numeric_value,
                    "best_units": act.get("standard_units", ""),
                    "best_pchembl": pchembl_float,
                    "n_measurements": 1,
                    "canonical_smiles": act.get("canonical_smiles", ""),
                }
            else:
                compound_map[mol_id]["n_measurements"] += 1
                # Keep the most potent (lowest) value
                existing = compound_map[mol_id]["best_value"]
                if numeric_value is not None and (
                    existing is None or numeric_value < existing
                ):
                    compound_map[mol_id]["best_value"] = numeric_value
                    compound_map[mol_id]["best_units"] = act.get(
                        "standard_units", ""
                    )
                # Keep the highest pChEMBL (more potent)
                existing_p = compound_map[mol_id]["best_pchembl"]
                if pchembl_float is not None and (
                    existing_p is None or pchembl_float > existing_p
                ):
                    compound_map[mol_id]["best_pchembl"] = pchembl_float

        # Sort compounds by potency (lowest value first, None last)
        compounds = sorted(
            compound_map.values(),
            key=lambda c: (
                c["best_value"] if c["best_value"] is not None else float("inf")
            ),
        )

        stats = {
            "target_chembl_id": target_chembl_id,
            "activity_type": activity_type,
            "n_activities": len(activities_raw),
            "n_unique_compounds": len(compounds),
            "compounds": compounds,
            "has_more": data.get("page_meta", {}).get("next") is not None,
            "analysis_type": "chembl_target_compounds",
        }

        logger.info(
            f"Target compound search complete: {len(compounds)} unique compounds "
            f"from {len(activities_raw)} {activity_type} records for {target_chembl_id}"
        )
        return None, stats, ir
