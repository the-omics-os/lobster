"""
PubChem PUG REST API service for compound property retrieval, similarity
search, and synonym lookup.

Wraps the PubChem Power User Gateway (https://pubchem.ncbi.nlm.nih.gov/rest/pug)
with synchronous httpx calls. All methods return 3-tuples
(None, Dict, AnalysisStep) since this is an API-only service that does
not modify AnnData objects.

The similarity search uses PubChem's asynchronous ListKey pattern:
a POST initiates the search and returns a ListKey, which is then polled
via GET until results are ready.

All methods handle HTTP errors gracefully and return error information
in the stats dict rather than raising exceptions to the caller.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from lobster.agents.drug_discovery.config import (
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_SEARCH_LIMIT,
    LIPINSKI_RULES,
    PUBCHEM_API_BASE,
)
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)

# Standard molecular properties requested from PubChem
_PROPERTY_LIST = (
    "MolecularFormula,MolecularWeight,XLogP,TPSA,"
    "HBondDonorCount,HBondAcceptorCount,RotatableBondCount,"
    "HeavyAtomCount"
)

# Maximum number of polling attempts for async similarity search
_MAX_POLL_ATTEMPTS = 15

# Seconds between polling attempts
_POLL_INTERVAL = 2.0


class PubChemService:
    """
    Stateless service for querying the PubChem PUG REST API.

    Provides compound property retrieval, structure similarity search,
    and synonym lookup. Each method returns (None, stats_dict, AnalysisStep)
    following the Lobster AI service pattern.
    """

    def __init__(self) -> None:
        """Initialize the PubChem service (stateless)."""
        logger.debug("Initializing stateless PubChemService")

    # =========================================================================
    # IR builders
    # =========================================================================

    def _create_ir_get_compound_properties(
        self, identifier: str, id_type: str
    ) -> AnalysisStep:
        """Create IR for compound property retrieval."""
        return AnalysisStep(
            operation="pubchem.compound.properties",
            tool_name="get_compound_properties",
            description=(
                f"Get molecular properties for compound "
                f"{id_type}={identifier} from PubChem"
            ),
            library="httpx",
            code_template="""# Get compound properties from PubChem
import httpx

properties = (
    "MolecularFormula,MolecularWeight,XLogP,TPSA,"
    "HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount"
)
url = f"{{ base_url }}/compound/{{ id_type }}/{{ identifier }}/property/{properties}/JSON"
response = httpx.get(url, timeout={{ timeout }})
response.raise_for_status()
data = response.json()
props = data["PropertyTable"]["Properties"][0]
print(f"MW={props['MolecularWeight']}, XLogP={props.get('XLogP', 'N/A')}, TPSA={props.get('TPSA', 'N/A')}")""",
            imports=["import httpx"],
            parameters={
                "identifier": identifier,
                "id_type": id_type,
                "base_url": PUBCHEM_API_BASE,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "identifier": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Compound identifier (name, CID, SMILES, or InChIKey)",
                ),
                "id_type": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="name",
                    required=False,
                    validation_rule="id_type in ['name', 'cid', 'smiles', 'inchikey']",
                    description="Identifier type: name, cid, smiles, or inchikey",
                ),
            },
            input_entities=[],
            output_entities=["compound_properties"],
        )

    def _create_ir_search_by_similarity(
        self, smiles: str, threshold: int, limit: int
    ) -> AnalysisStep:
        """Create IR for similarity search."""
        return AnalysisStep(
            operation="pubchem.compound.similarity_search",
            tool_name="search_by_similarity",
            description=(
                f"Search PubChem for compounds similar to SMILES={smiles} "
                f"(threshold={threshold}%)"
            ),
            library="httpx",
            code_template="""# PubChem similarity search (async ListKey pattern)
import httpx, time
from urllib.parse import quote

# Step 1: Initiate async search (URL-encode SMILES for safety)
smiles_encoded = quote({{ smiles | tojson }}, safe='')
url = f"{{ base_url }}/compound/similarity/smiles/{smiles_encoded}/JSON"
params = {"Threshold": {{ threshold }}, "MaxRecords": {{ limit }}}
response = httpx.post(url, params=params, timeout={{ timeout }})
response.raise_for_status()
list_key = response.json()["Waiting"]["ListKey"]

# Step 2: Poll for results
properties = (
    "MolecularFormula,MolecularWeight,XLogP,TPSA,"
    "HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount"
)
poll_url = f"{{ base_url }}/compound/listkey/{list_key}/property/{properties}/JSON"
for attempt in range(15):
    time.sleep(2)
    poll = httpx.get(poll_url, timeout={{ timeout }})
    if poll.status_code == 200:
        data = poll.json()
        compounds = data["PropertyTable"]["Properties"]
        print(f"Found {len(compounds)} similar compounds")
        break
""",
            imports=["import httpx", "import time"],
            parameters={
                "smiles": smiles,
                "threshold": threshold,
                "limit": limit,
                "base_url": PUBCHEM_API_BASE,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "smiles": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="SMILES string of the query compound",
                ),
                "threshold": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=85,
                    required=False,
                    validation_rule="0 < threshold <= 100",
                    description="Tanimoto similarity threshold (percent, 0-100)",
                ),
                "limit": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=DEFAULT_SEARCH_LIMIT,
                    required=False,
                    validation_rule="limit > 0",
                    description="Maximum number of similar compounds to return",
                ),
            },
            input_entities=[],
            output_entities=["similarity_results"],
        )

    def _create_ir_get_compound_synonyms(
        self, identifier: str, id_type: str
    ) -> AnalysisStep:
        """Create IR for synonym lookup."""
        return AnalysisStep(
            operation="pubchem.compound.synonyms",
            tool_name="get_compound_synonyms",
            description=(
                f"Get synonyms for compound {id_type}={identifier} from PubChem"
            ),
            library="httpx",
            code_template="""# Get compound synonyms from PubChem
import httpx

url = f"{{ base_url }}/compound/{{ id_type }}/{{ identifier }}/synonyms/JSON"
response = httpx.get(url, timeout={{ timeout }})
response.raise_for_status()
data = response.json()
synonyms = data["InformationList"]["Information"][0]["Synonym"]
print(f"Found {len(synonyms)} synonyms for '{{ identifier }}'")""",
            imports=["import httpx"],
            parameters={
                "identifier": identifier,
                "id_type": id_type,
                "base_url": PUBCHEM_API_BASE,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "identifier": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Compound identifier (name, CID, SMILES, or InChIKey)",
                ),
                "id_type": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="name",
                    required=False,
                    validation_rule="id_type in ['name', 'cid', 'smiles', 'inchikey']",
                    description="Identifier type: name, cid, smiles, or inchikey",
                ),
            },
            input_entities=[],
            output_entities=["compound_synonyms"],
        )

    # =========================================================================
    # Internal HTTP helpers
    # =========================================================================

    def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute a GET request and return parsed JSON or an error string.

        Returns:
            Tuple of (json_data, error_message). Exactly one will be None.
        """
        try:
            with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                try:
                    return response.json(), None
                except (ValueError, json.JSONDecodeError) as exc:
                    msg = (
                        f"PubChem API returned invalid JSON for {url}: {exc}"
                    )
                    logger.error(msg)
                    return None, msg
        except httpx.TimeoutException:
            msg = f"PubChem API timeout after {DEFAULT_HTTP_TIMEOUT}s for {url}"
            logger.error(msg)
            return None, msg
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            # PubChem returns 404 for "not found" which is a valid response
            if status == 404:
                msg = f"Compound not found in PubChem (HTTP 404)"
                logger.warning(msg)
                return None, msg
            msg = (
                f"PubChem API HTTP {status} for {url}: "
                f"{exc.response.text[:200]}"
            )
            logger.error(msg)
            return None, msg
        except httpx.RequestError as exc:
            msg = f"PubChem API request error for {url}: {exc}"
            logger.error(msg)
            return None, msg

    def _post_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute a POST request (used for async search initiation).

        Returns:
            Tuple of (json_data, error_message). Exactly one will be None.
        """
        try:
            with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                response = client.post(url, params=params)
                response.raise_for_status()
                try:
                    return response.json(), None
                except (ValueError, json.JSONDecodeError) as exc:
                    msg = (
                        f"PubChem API returned invalid JSON for POST {url}: {exc}"
                    )
                    logger.error(msg)
                    return None, msg
        except httpx.TimeoutException:
            msg = f"PubChem API timeout after {DEFAULT_HTTP_TIMEOUT}s for POST {url}"
            logger.error(msg)
            return None, msg
        except httpx.HTTPStatusError as exc:
            msg = (
                f"PubChem API HTTP {exc.response.status_code} for POST {url}: "
                f"{exc.response.text[:200]}"
            )
            logger.error(msg)
            return None, msg
        except httpx.RequestError as exc:
            msg = f"PubChem API request error for POST {url}: {exc}"
            logger.error(msg)
            return None, msg

    # =========================================================================
    # Internal Lipinski evaluation
    # =========================================================================

    @staticmethod
    def _evaluate_lipinski(props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate Lipinski Rule of Five compliance from property dict.

        Returns a dict with per-rule pass/fail and overall compliance.
        """
        mw = props.get("MolecularWeight")
        logp = props.get("XLogP")
        hbd = props.get("HBondDonorCount")
        hba = props.get("HBondAcceptorCount")

        rules = {}
        violations = 0

        # MW <= 500
        if mw is not None:
            try:
                mw_val = float(mw)
                passed = mw_val <= LIPINSKI_RULES["mw_max"]
                rules["mw_le_500"] = passed
                if not passed:
                    violations += 1
            except (ValueError, TypeError):
                rules["mw_le_500"] = None
        else:
            rules["mw_le_500"] = None

        # LogP <= 5
        if logp is not None:
            try:
                logp_val = float(logp)
                passed = logp_val <= LIPINSKI_RULES["logp_max"]
                rules["logp_le_5"] = passed
                if not passed:
                    violations += 1
            except (ValueError, TypeError):
                rules["logp_le_5"] = None
        else:
            rules["logp_le_5"] = None

        # HBD <= 5
        if hbd is not None:
            try:
                hbd_val = int(hbd)
                passed = hbd_val <= LIPINSKI_RULES["hbd_max"]
                rules["hbd_le_5"] = passed
                if not passed:
                    violations += 1
            except (ValueError, TypeError):
                rules["hbd_le_5"] = None
        else:
            rules["hbd_le_5"] = None

        # HBA <= 10
        if hba is not None:
            try:
                hba_val = int(hba)
                passed = hba_val <= LIPINSKI_RULES["hba_max"]
                rules["hba_le_10"] = passed
                if not passed:
                    violations += 1
            except (ValueError, TypeError):
                rules["hba_le_10"] = None
        else:
            rules["hba_le_10"] = None

        return {
            "rules": rules,
            "n_violations": violations,
            "compliant": violations <= 1,  # Lipinski allows 1 violation
        }

    # =========================================================================
    # Public methods
    # =========================================================================

    def get_compound_properties(
        self,
        identifier: str,
        id_type: str = "name",
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Get molecular properties for a compound from PubChem.

        Retrieves MolecularFormula, MolecularWeight, XLogP, TPSA,
        HBondDonorCount, HBondAcceptorCount, RotatableBondCount, and
        HeavyAtomCount. Also evaluates Lipinski Rule of Five compliance.

        Args:
            identifier: Compound name, CID, SMILES, or InChIKey.
            id_type: Identifier type -- one of "name", "cid", "smiles",
                     or "inchikey" (default "name").

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
        """
        logger.info(
            f"Getting PubChem properties: {id_type}='{identifier}'"
        )

        ir = self._create_ir_get_compound_properties(identifier, id_type)

        # URL-encode identifier when it's a SMILES string — characters like
        # # (triple bond), @ (chirality), / (stereochemistry) are URL-unsafe.
        from urllib.parse import quote

        safe_id = quote(identifier, safe="") if id_type == "smiles" else identifier
        url = (
            f"{PUBCHEM_API_BASE}/compound/{id_type}/{safe_id}"
            f"/property/{_PROPERTY_LIST}/JSON"
        )

        data, error = self._get_json(url)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "identifier": identifier,
                "id_type": id_type,
                "analysis_type": "pubchem_compound_properties",
            }
            return None, stats, ir

        # PubChem returns a PropertyTable with a Properties list
        properties_table = data.get("PropertyTable", {})
        properties_list = properties_table.get("Properties", [])

        if not properties_list:
            stats = {
                "error": f"No properties returned for {id_type}='{identifier}'",
                "identifier": identifier,
                "id_type": id_type,
                "analysis_type": "pubchem_compound_properties",
            }
            return None, stats, ir

        props = properties_list[0]
        cid = props.get("CID")

        # Evaluate Lipinski
        lipinski = self._evaluate_lipinski(props)

        stats = {
            "identifier": identifier,
            "id_type": id_type,
            "cid": cid,
            "molecular_formula": props.get("MolecularFormula", ""),
            "molecular_weight": props.get("MolecularWeight"),
            "xlogp": props.get("XLogP"),
            "tpsa": props.get("TPSA"),
            "hbond_donor_count": props.get("HBondDonorCount"),
            "hbond_acceptor_count": props.get("HBondAcceptorCount"),
            "rotatable_bond_count": props.get("RotatableBondCount"),
            "heavy_atom_count": props.get("HeavyAtomCount"),
            "lipinski": lipinski,
            "analysis_type": "pubchem_compound_properties",
        }

        logger.info(
            f"PubChem properties for '{identifier}': "
            f"MW={props.get('MolecularWeight')}, "
            f"XLogP={props.get('XLogP')}, "
            f"Lipinski={'pass' if lipinski['compliant'] else 'fail'}"
        )
        return None, stats, ir

    def search_by_similarity(
        self,
        smiles: str,
        threshold: int = 85,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Search PubChem for structurally similar compounds via Tanimoto similarity.

        Uses PubChem's asynchronous search pattern:
        1. POST to initiate the similarity search, receiving a ListKey.
        2. Poll GET with the ListKey until results are available.

        Results include molecular properties and Lipinski evaluation for
        each hit.

        Args:
            smiles: SMILES string of the query compound.
            threshold: Tanimoto similarity threshold in percent (default 85).
            limit: Maximum number of results (default 20).

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
        """
        logger.info(
            f"PubChem similarity search: SMILES='{smiles[:50]}...', "
            f"threshold={threshold}%, limit={limit}"
        )

        ir = self._create_ir_search_by_similarity(smiles, threshold, limit)

        # Step 1: Initiate async similarity search
        # URL-encode the SMILES — characters like # (triple bond), @ (chirality),
        # / (stereochemistry) are URL-unsafe and must be percent-encoded.
        from urllib.parse import quote

        search_url = (
            f"{PUBCHEM_API_BASE}/compound/similarity/smiles/"
            f"{quote(smiles, safe='')}/JSON"
        )
        search_params = {"Threshold": threshold, "MaxRecords": limit}

        init_data, error = self._post_json(search_url, params=search_params)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "smiles": smiles,
                "threshold": threshold,
                "n_results": 0,
                "analysis_type": "pubchem_similarity_search",
            }
            return None, stats, ir

        # Extract ListKey from the Waiting response
        list_key = None
        if init_data:
            waiting = init_data.get("Waiting")
            if waiting:
                list_key = waiting.get("ListKey")

        if not list_key:
            # Sometimes PubChem returns results directly for small searches
            properties_table = init_data.get("PropertyTable", {}) if init_data else {}
            if properties_table.get("Properties"):
                compounds = self._parse_property_list(
                    properties_table["Properties"]
                )
                stats = {
                    "smiles": smiles,
                    "threshold": threshold,
                    "n_results": len(compounds),
                    "compounds": compounds,
                    "analysis_type": "pubchem_similarity_search",
                }
                return None, stats, ir

            stats = {
                "error": "Failed to obtain ListKey from PubChem similarity search",
                "smiles": smiles,
                "threshold": threshold,
                "n_results": 0,
                "analysis_type": "pubchem_similarity_search",
            }
            return None, stats, ir

        logger.debug(f"PubChem similarity search initiated, ListKey={list_key}")

        # Step 2: Poll for results using the ListKey
        poll_url = (
            f"{PUBCHEM_API_BASE}/compound/listkey/{list_key}"
            f"/property/{_PROPERTY_LIST}/JSON"
        )

        compounds: List[Dict[str, Any]] = []
        poll_error: Optional[str] = None

        for attempt in range(1, _MAX_POLL_ATTEMPTS + 1):
            time.sleep(_POLL_INTERVAL)
            logger.debug(
                f"Polling PubChem similarity results (attempt {attempt}/{_MAX_POLL_ATTEMPTS})"
            )

            try:
                with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                    poll_response = client.get(poll_url)

                if poll_response.status_code == 200:
                    try:
                        poll_data = poll_response.json()
                    except (ValueError, json.JSONDecodeError) as exc:
                        poll_error = (
                            f"PubChem poll returned invalid JSON: {exc}"
                        )
                        logger.warning(poll_error)
                        continue
                    properties_table = poll_data.get("PropertyTable", {})
                    properties_list = properties_table.get("Properties", [])
                    compounds = self._parse_property_list(properties_list)
                    logger.info(
                        f"Similarity search complete: {len(compounds)} results"
                    )
                    break

                elif poll_response.status_code == 202:
                    # Still processing
                    logger.debug("PubChem search still processing...")
                    continue

                else:
                    # Unexpected status
                    poll_error = (
                        f"PubChem polling returned HTTP {poll_response.status_code}: "
                        f"{poll_response.text[:200]}"
                    )
                    logger.warning(poll_error)
                    # Continue polling in case it's transient
                    continue

            except httpx.TimeoutException:
                logger.warning(
                    f"PubChem poll timeout on attempt {attempt}"
                )
                continue
            except httpx.RequestError as exc:
                poll_error = f"PubChem poll request error: {exc}"
                logger.warning(poll_error)
                continue
        else:
            # Exhausted all polling attempts
            if not compounds:
                poll_error = (
                    f"PubChem similarity search did not complete after "
                    f"{_MAX_POLL_ATTEMPTS} polling attempts "
                    f"({_MAX_POLL_ATTEMPTS * _POLL_INTERVAL}s)"
                )
                logger.error(poll_error)

        if poll_error and not compounds:
            stats = {
                "error": poll_error,
                "smiles": smiles,
                "threshold": threshold,
                "n_results": 0,
                "analysis_type": "pubchem_similarity_search",
            }
            return None, stats, ir

        stats = {
            "smiles": smiles,
            "threshold": threshold,
            "n_results": len(compounds),
            "compounds": compounds,
            "analysis_type": "pubchem_similarity_search",
        }

        logger.info(
            f"PubChem similarity search: {len(compounds)} compounds "
            f"above {threshold}% Tanimoto for query SMILES"
        )
        return None, stats, ir

    def get_compound_synonyms(
        self,
        identifier: str,
        id_type: str = "name",
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Get synonyms (trade names, IUPAC names, registry numbers) for a compound.

        Args:
            identifier: Compound name, CID, SMILES, or InChIKey.
            id_type: Identifier type -- one of "name", "cid", "smiles",
                     or "inchikey" (default "name").

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
        """
        logger.info(
            f"Getting PubChem synonyms: {id_type}='{identifier}'"
        )

        ir = self._create_ir_get_compound_synonyms(identifier, id_type)

        # URL-encode identifier when it's a SMILES string
        from urllib.parse import quote

        safe_id = quote(identifier, safe="") if id_type == "smiles" else identifier
        url = (
            f"{PUBCHEM_API_BASE}/compound/{id_type}/{safe_id}/synonyms/JSON"
        )

        data, error = self._get_json(url)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "identifier": identifier,
                "id_type": id_type,
                "n_synonyms": 0,
                "analysis_type": "pubchem_compound_synonyms",
            }
            return None, stats, ir

        # Parse synonym response
        info_list = data.get("InformationList", {})
        info_entries = info_list.get("Information", [])

        if not info_entries:
            stats = {
                "error": f"No synonym data returned for {id_type}='{identifier}'",
                "identifier": identifier,
                "id_type": id_type,
                "n_synonyms": 0,
                "analysis_type": "pubchem_compound_synonyms",
            }
            return None, stats, ir

        entry = info_entries[0]
        cid = entry.get("CID")
        synonyms: List[str] = entry.get("Synonym", [])

        # Categorize synonyms heuristically
        iupac_names: List[str] = []
        trade_names: List[str] = []
        registry_numbers: List[str] = []
        other_names: List[str] = []

        for syn in synonyms:
            syn_stripped = syn.strip()
            if not syn_stripped:
                continue
            # CAS-like pattern: digits-digits-digit
            if (
                len(syn_stripped) <= 15
                and "-" in syn_stripped
                and syn_stripped.replace("-", "").isdigit()
            ):
                registry_numbers.append(syn_stripped)
            elif any(
                c in syn_stripped
                for c in ["(", ")", "[", "]", ","]
            ) and len(syn_stripped) > 30:
                iupac_names.append(syn_stripped)
            elif syn_stripped[0].isupper() and " " not in syn_stripped:
                trade_names.append(syn_stripped)
            else:
                other_names.append(syn_stripped)

        stats = {
            "identifier": identifier,
            "id_type": id_type,
            "cid": cid,
            "n_synonyms": len(synonyms),
            "synonyms": synonyms[:100],  # Cap at 100 for sanity
            "iupac_names": iupac_names[:5],
            "trade_names": trade_names[:20],
            "registry_numbers": registry_numbers[:10],
            "other_names": other_names[:20],
            "analysis_type": "pubchem_compound_synonyms",
        }

        logger.info(
            f"PubChem synonyms for '{identifier}': {len(synonyms)} total, "
            f"{len(trade_names)} trade names, {len(registry_numbers)} registry numbers"
        )
        return None, stats, ir

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _parse_property_list(
        self, properties_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Parse a PubChem PropertyTable Properties list into clean compound dicts.

        Evaluates Lipinski compliance for each compound.
        """
        compounds: List[Dict[str, Any]] = []
        for props in properties_list:
            lipinski = self._evaluate_lipinski(props)
            compounds.append({
                "cid": props.get("CID"),
                "molecular_formula": props.get("MolecularFormula", ""),
                "molecular_weight": props.get("MolecularWeight"),
                "xlogp": props.get("XLogP"),
                "tpsa": props.get("TPSA"),
                "hbond_donor_count": props.get("HBondDonorCount"),
                "hbond_acceptor_count": props.get("HBondAcceptorCount"),
                "rotatable_bond_count": props.get("RotatableBondCount"),
                "heavy_atom_count": props.get("HeavyAtomCount"),
                "lipinski": lipinski,
            })
        return compounds
