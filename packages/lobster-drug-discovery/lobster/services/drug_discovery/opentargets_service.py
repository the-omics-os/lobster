"""
Open Targets Platform API service for target-disease evidence, druggability
scoring, drug indication lookup, safety profiling, and tractability assessment.

Wraps the Open Targets GraphQL API (https://api.platform.opentargets.org/api/v4/graphql)
with synchronous httpx calls. All methods return 3-tuples
(None, Dict, AnalysisStep) since this is an API-only service that does
not modify AnnData objects.

All methods handle HTTP errors gracefully and return error information
in the stats dict rather than raising exceptions to the caller.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import httpx

from lobster.agents.drug_discovery.config import (
    DEFAULT_HTTP_TIMEOUT,
    OPENTARGETS_GRAPHQL,
    TARGET_EVIDENCE_WEIGHTS,
)
from lobster.core.analysis_ir import AnalysisStep, ParameterSpec
from lobster.utils.logger import get_logger

logger = get_logger(__name__)


class OpenTargetsService:
    """
    Stateless service for querying the Open Targets Platform GraphQL API.

    Provides target-disease evidence, composite druggability scoring,
    drug indication lookup, safety profiling, and tractability assessment.
    Each method returns (None, stats_dict, AnalysisStep) following the
    Lobster AI service pattern.
    """

    def __init__(self) -> None:
        """Initialize the Open Targets service (stateless)."""
        logger.debug("Initializing stateless OpenTargetsService")

    # =========================================================================
    # IR builders
    # =========================================================================

    def _create_ir_get_target_disease_evidence(
        self,
        ensembl_id: str,
        disease_id: Optional[str],
        limit: int,
    ) -> AnalysisStep:
        """Create IR for target-disease evidence retrieval."""
        return AnalysisStep(
            operation="opentargets.target.disease_evidence",
            tool_name="get_target_disease_evidence",
            description=(
                f"Get target-disease associations for {ensembl_id}"
                + (f" filtered to disease {disease_id}" if disease_id else "")
            ),
            library="httpx",
            code_template="""# Query Open Targets for target-disease evidence
import httpx

query = \"\"\"
query TargetDiseaseEvidence($ensemblId: String!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    associatedDiseases(page: {size: $size, index: 0}) {
      count
      rows {
        disease { id name }
        score
        datatypeScores { id score }
      }
    }
  }
}
\"\"\"
variables = {"ensemblId": {{ ensembl_id | tojson }}, "size": {{ limit }}}
response = httpx.post(
    "{{ graphql_url }}",
    json={"query": query, "variables": variables},
    timeout={{ timeout }},
)
response.raise_for_status()
data = response.json()["data"]["target"]
print(f"Target: {data['approvedSymbol']} ({data['approvedName']})")""",
            imports=["import httpx"],
            parameters={
                "ensembl_id": ensembl_id,
                "disease_id": disease_id,
                "limit": limit,
                "graphql_url": OPENTARGETS_GRAPHQL,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "ensembl_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Ensembl gene ID (e.g., ENSG00000157764 for BRAF)",
                ),
                "disease_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value=None,
                    required=False,
                    description="EFO disease ID to filter associations (e.g., EFO_0000311)",
                ),
                "limit": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=25,
                    required=False,
                    validation_rule="limit > 0",
                    description="Maximum number of disease associations to return",
                ),
            },
            input_entities=[],
            output_entities=["target_disease_evidence"],
        )

    def _create_ir_score_target(self, ensembl_id: str) -> AnalysisStep:
        """Create IR for composite druggability scoring."""
        return AnalysisStep(
            operation="opentargets.target.druggability_score",
            tool_name="score_target",
            description=f"Compute composite druggability score for target {ensembl_id}",
            library="httpx",
            code_template="""# Score target druggability via Open Targets
import httpx

query = \"\"\"
query TargetScore($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    tractability { label modality value }
    safetyLiabilities { event }
    associatedDiseases(page: {size: 5, index: 0}) {
      count
      rows { score datatypeScores { id score } }
    }
  }
}
\"\"\"
response = httpx.post(
    "{{ graphql_url }}",
    json={"query": query, "variables": {"ensemblId": {{ ensembl_id | tojson }}}},
    timeout={{ timeout }},
)
response.raise_for_status()
data = response.json()["data"]["target"]
print(f"Druggability score computed for {data['approvedSymbol']}")""",
            imports=["import httpx"],
            parameters={
                "ensembl_id": ensembl_id,
                "graphql_url": OPENTARGETS_GRAPHQL,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "ensembl_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Ensembl gene ID for the target to score",
                ),
            },
            input_entities=[],
            output_entities=["druggability_score"],
        )

    def _create_ir_get_drug_indications(
        self, chembl_id: str, limit: int
    ) -> AnalysisStep:
        """Create IR for drug indication lookup."""
        return AnalysisStep(
            operation="opentargets.drug.indications",
            tool_name="get_drug_indications",
            description=f"Get known drug indications for {chembl_id}",
            library="httpx",
            code_template="""# Get drug indications from Open Targets
import httpx

query = \"\"\"
query DrugIndications($chemblId: String!) {
  drug(chemblId: $chemblId) {
    id name
    maximumClinicalTrialPhase
    indications {
      count
      rows { disease { id name } maxPhaseForIndication references { source ids } }
    }
  }
}
\"\"\"
variables = {"chemblId": {{ chembl_id | tojson }}}
response = httpx.post(
    "{{ graphql_url }}",
    json={"query": query, "variables": variables},
    timeout={{ timeout }},
)
response.raise_for_status()
data = response.json()["data"]["drug"]
print(f"Drug: {data['name']}, max phase: {data['maximumClinicalTrialPhase']}")""",
            imports=["import httpx"],
            parameters={
                "chembl_id": chembl_id,
                "limit": limit,
                "graphql_url": OPENTARGETS_GRAPHQL,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "chembl_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="ChEMBL drug ID (e.g., CHEMBL25 for aspirin)",
                ),
                "limit": ParameterSpec(
                    param_type="int",
                    papermill_injectable=True,
                    default_value=25,
                    required=False,
                    validation_rule="limit > 0",
                    description="Maximum number of indications to return",
                ),
            },
            input_entities=[],
            output_entities=["drug_indications"],
        )

    def _create_ir_get_safety_profile(self, target_id: str) -> AnalysisStep:
        """Create IR for safety profile lookup."""
        return AnalysisStep(
            operation="opentargets.target.safety",
            tool_name="get_safety_profile",
            description=f"Get known adverse events and safety data for target {target_id}",
            library="httpx",
            code_template="""# Get target safety profile from Open Targets
import httpx

query = \"\"\"
query TargetSafety($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id approvedSymbol
    safetyLiabilities {
      event
      datasource
      literature
      url
      biosamples { tissueLabel tissueId }
      effects { dosing direction }
      studies { name description type }
    }
  }
}
\"\"\"
response = httpx.post(
    "{{ graphql_url }}",
    json={"query": query, "variables": {"ensemblId": {{ target_id | tojson }}}},
    timeout={{ timeout }},
)
response.raise_for_status()
data = response.json()["data"]["target"]
print(f"Safety profile for {data['approvedSymbol']}: {len(data.get('safetyLiabilities', []))} events")""",
            imports=["import httpx"],
            parameters={
                "target_id": target_id,
                "graphql_url": OPENTARGETS_GRAPHQL,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "target_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Ensembl gene ID for the target",
                ),
            },
            input_entities=[],
            output_entities=["safety_profile"],
        )

    def _create_ir_assess_tractability(self, target_id: str) -> AnalysisStep:
        """Create IR for tractability assessment."""
        return AnalysisStep(
            operation="opentargets.target.tractability",
            tool_name="assess_tractability",
            description=f"Assess antibody/small molecule tractability for target {target_id}",
            library="httpx",
            code_template="""# Assess target tractability from Open Targets
import httpx

query = \"\"\"
query TargetTractability($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id approvedSymbol
    tractability { label modality value }
  }
}
\"\"\"
response = httpx.post(
    "{{ graphql_url }}",
    json={"query": query, "variables": {"ensemblId": {{ target_id | tojson }}}},
    timeout={{ timeout }},
)
response.raise_for_status()
data = response.json()["data"]["target"]
print(f"Tractability for {data['approvedSymbol']}: {len(data.get('tractability', []))} assessments")""",
            imports=["import httpx"],
            parameters={
                "target_id": target_id,
                "graphql_url": OPENTARGETS_GRAPHQL,
                "timeout": DEFAULT_HTTP_TIMEOUT,
            },
            parameter_schema={
                "target_id": ParameterSpec(
                    param_type="str",
                    papermill_injectable=True,
                    default_value="",
                    required=True,
                    description="Ensembl gene ID for the target",
                ),
            },
            input_entities=[],
            output_entities=["tractability_assessment"],
        )

    # =========================================================================
    # Internal HTTP helper
    # =========================================================================

    def _post_graphql(
        self,
        query: str,
        variables: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Execute a GraphQL POST request and return parsed data or error.

        Returns:
            Tuple of (response_data_dict, error_message). Exactly one is None.
        """
        try:
            with httpx.Client(timeout=DEFAULT_HTTP_TIMEOUT) as client:
                response = client.post(
                    OPENTARGETS_GRAPHQL,
                    json={"query": query, "variables": variables},
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                try:
                    body = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    msg = (
                        f"Open Targets API returned invalid JSON: {exc}"
                    )
                    logger.error(msg)
                    return None, msg

            # Check for GraphQL-level errors
            if "errors" in body:
                error_msgs = [
                    e.get("message", "Unknown error") for e in body["errors"]
                ]
                msg = f"Open Targets GraphQL errors: {'; '.join(error_msgs)}"
                logger.error(msg)
                return None, msg

            return body.get("data", {}), None

        except httpx.TimeoutException:
            msg = (
                f"Open Targets API timeout after {DEFAULT_HTTP_TIMEOUT}s"
            )
            logger.error(msg)
            return None, msg
        except httpx.HTTPStatusError as exc:
            msg = (
                f"Open Targets API HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            )
            logger.error(msg)
            return None, msg
        except httpx.RequestError as exc:
            msg = f"Open Targets API request error: {exc}"
            logger.error(msg)
            return None, msg

    # =========================================================================
    # Public methods
    # =========================================================================

    def get_target_disease_evidence(
        self,
        ensembl_id: str,
        disease_id: Optional[str] = None,
        limit: int = 25,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Get target-disease associations from Open Targets.

        Queries for a target's associated diseases with overall and
        per-datatype scores. Optionally filters to a specific disease.

        Args:
            ensembl_id: Ensembl gene ID (e.g., "ENSG00000157764" for BRAF).
            disease_id: Optional EFO disease ID to filter (e.g., "EFO_0000311").
            limit: Maximum number of disease associations (default 25).

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
        """
        logger.info(
            f"Querying target-disease evidence for {ensembl_id}"
            + (f", disease={disease_id}" if disease_id else "")
        )

        ir = self._create_ir_get_target_disease_evidence(
            ensembl_id, disease_id, limit
        )

        query = """
        query TargetDiseaseEvidence($ensemblId: String!, $size: Int!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            associatedDiseases(page: {size: $size, index: 0}) {
              count
              rows {
                disease {
                  id
                  name
                }
                score
                datatypeScores {
                  id
                  score
                }
              }
            }
          }
        }
        """
        variables: Dict[str, Any] = {"ensemblId": ensembl_id, "size": limit}

        data, error = self._post_graphql(query, variables)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "ensembl_id": ensembl_id,
                "disease_id": disease_id,
                "n_associations": 0,
                "analysis_type": "opentargets_target_disease_evidence",
            }
            return None, stats, ir

        target_data = data.get("target")
        if target_data is None:
            stats = {
                "error": f"Target {ensembl_id} not found in Open Targets",
                "ensembl_id": ensembl_id,
                "disease_id": disease_id,
                "n_associations": 0,
                "analysis_type": "opentargets_target_disease_evidence",
            }
            return None, stats, ir

        assoc_data = target_data.get("associatedDiseases", {})
        rows = assoc_data.get("rows", [])

        # Filter to specific disease if requested
        if disease_id:
            rows = [
                r for r in rows
                if r.get("disease", {}).get("id", "") == disease_id
            ]

        # Parse associations
        associations: List[Dict[str, Any]] = []
        for row in rows:
            disease_info = row.get("disease", {})
            datatype_scores = {
                ds.get("id", ""): ds.get("score", 0.0)
                for ds in row.get("datatypeScores", [])
            }
            associations.append({
                "disease_id": disease_info.get("id", ""),
                "disease_name": disease_info.get("name", ""),
                "overall_score": row.get("score", 0.0),
                "datatype_scores": datatype_scores,
            })

        stats = {
            "ensembl_id": ensembl_id,
            "approved_symbol": target_data.get("approvedSymbol", ""),
            "approved_name": target_data.get("approvedName", ""),
            "disease_id": disease_id,
            "total_associated_diseases": assoc_data.get("count", 0),
            "n_associations": len(associations),
            "associations": associations,
            "analysis_type": "opentargets_target_disease_evidence",
        }

        logger.info(
            f"Target-disease evidence complete for {ensembl_id}: "
            f"{len(associations)} associations"
        )
        return None, stats, ir

    def score_target(
        self,
        ensembl_id: str,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Compute a composite druggability score for a target.

        Combines tractability data, safety liabilities, and genetic
        association evidence into a weighted composite score (0-1).

        Score components (from TARGET_EVIDENCE_WEIGHTS in config):
        - genetic_association (0.30): top disease association score
        - known_drug (0.25): tractability with known drug modalities
        - expression_specificity (0.20): placeholder (requires expression data)
        - pathogenicity (0.15): absence of broad safety liabilities
        - literature (0.10): number of associated diseases as evidence proxy

        Args:
            ensembl_id: Ensembl gene ID for the target.

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
            stats_dict contains component scores and composite druggability_score.
        """
        logger.info(f"Computing druggability score for target {ensembl_id}")

        ir = self._create_ir_score_target(ensembl_id)

        query = """
        query TargetScore($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            tractability {
              label
              modality
              value
            }
            safetyLiabilities {
              event
            }
            associatedDiseases(page: {size: 5, index: 0}) {
              count
              rows {
                score
                datatypeScores {
                  id
                  score
                }
              }
            }
          }
        }
        """
        variables = {"ensemblId": ensembl_id}

        data, error = self._post_graphql(query, variables)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "ensembl_id": ensembl_id,
                "druggability_score": 0.0,
                "analysis_type": "opentargets_druggability_score",
            }
            return None, stats, ir

        target_data = data.get("target")
        if target_data is None:
            stats = {
                "error": f"Target {ensembl_id} not found in Open Targets",
                "ensembl_id": ensembl_id,
                "druggability_score": 0.0,
                "analysis_type": "opentargets_druggability_score",
            }
            return None, stats, ir

        # ---- Compute component scores ----

        # 1. Genetic association: best overall disease association score
        assoc_rows = (
            target_data.get("associatedDiseases", {}).get("rows", [])
        )
        genetic_score = 0.0
        if assoc_rows:
            genetic_score = max(r.get("score", 0.0) for r in assoc_rows)

        # 2. Known drug: whether tractability entries include drug modalities
        tractability_entries = target_data.get("tractability") or []
        drug_modalities = {"Small molecule", "Antibody", "PROTAC", "Other modalities"}
        has_drug_modality = any(
            t.get("modality", "") in drug_modalities and t.get("value", False)
            for t in tractability_entries
        )
        known_drug_score = 1.0 if has_drug_modality else 0.0

        # 3. Expression specificity: placeholder (0.5 default)
        expression_score = 0.5

        # 4. Pathogenicity / safety: inverse of safety liability count
        safety_liabilities = target_data.get("safetyLiabilities") or []
        n_safety_issues = len(safety_liabilities)
        # Sigmoid decay: many safety issues -> lower score
        if n_safety_issues == 0:
            pathogenicity_score = 1.0
        elif n_safety_issues <= 2:
            pathogenicity_score = 0.7
        elif n_safety_issues <= 5:
            pathogenicity_score = 0.4
        else:
            pathogenicity_score = 0.1

        # 5. Literature: disease count as evidence density proxy
        total_diseases = (
            target_data.get("associatedDiseases", {}).get("count", 0)
        )
        # Normalize: 50+ diseases = max score
        literature_score = min(total_diseases / 50.0, 1.0)

        # ---- Weighted composite ----
        component_scores = {
            "genetic_association": genetic_score,
            "known_drug": known_drug_score,
            "expression_specificity": expression_score,
            "pathogenicity": pathogenicity_score,
            "literature": literature_score,
        }

        composite = sum(
            component_scores[k] * TARGET_EVIDENCE_WEIGHTS.get(k, 0.0)
            for k in component_scores
        )
        # Clamp to [0, 1]
        composite = max(0.0, min(1.0, composite))

        # Tractability summary
        tractability_summary = {}
        for t in tractability_entries:
            modality = t.get("modality", "unknown")
            if modality not in tractability_summary:
                tractability_summary[modality] = []
            tractability_summary[modality].append({
                "label": t.get("label", ""),
                "value": t.get("value", False),
            })

        stats = {
            "ensembl_id": ensembl_id,
            "approved_symbol": target_data.get("approvedSymbol", ""),
            "approved_name": target_data.get("approvedName", ""),
            "druggability_score": round(composite, 4),
            "component_scores": component_scores,
            "weights": TARGET_EVIDENCE_WEIGHTS,
            "tractability_summary": tractability_summary,
            "n_safety_liabilities": n_safety_issues,
            "total_associated_diseases": total_diseases,
            "analysis_type": "opentargets_druggability_score",
        }

        logger.info(
            f"Druggability score for {ensembl_id} "
            f"({target_data.get('approvedSymbol', '')}): {composite:.3f}"
        )
        return None, stats, ir

    def get_drug_indications(
        self,
        chembl_id: str,
        limit: int = 25,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Get known indications and max clinical trial phase for a drug.

        Queries the Open Targets drug entity for approved and
        investigational indications with supporting references.

        Args:
            chembl_id: ChEMBL drug ID (e.g., "CHEMBL25" for aspirin).
            limit: Maximum number of indications (default 25).

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
        """
        logger.info(f"Querying drug indications for {chembl_id}")

        ir = self._create_ir_get_drug_indications(chembl_id, limit)

        query = """
        query DrugIndications($chemblId: String!) {
          drug(chemblId: $chemblId) {
            id
            name
            maximumClinicalTrialPhase
            drugType
            mechanismsOfAction {
              rows {
                mechanismOfAction
                targets {
                  id
                  approvedSymbol
                }
              }
            }
            indications {
              count
              rows {
                disease {
                  id
                  name
                }
                maxPhaseForIndication
                references {
                  source
                  ids
                }
              }
            }
          }
        }
        """
        variables: Dict[str, Any] = {"chemblId": chembl_id}

        data, error = self._post_graphql(query, variables)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "chembl_id": chembl_id,
                "n_indications": 0,
                "analysis_type": "opentargets_drug_indications",
            }
            return None, stats, ir

        drug_data = data.get("drug")
        if drug_data is None:
            stats = {
                "error": f"Drug {chembl_id} not found in Open Targets",
                "chembl_id": chembl_id,
                "n_indications": 0,
                "analysis_type": "opentargets_drug_indications",
            }
            return None, stats, ir

        # Parse indications
        indications_data = drug_data.get("indications", {})
        indication_rows = indications_data.get("rows", [])
        indications: List[Dict[str, Any]] = []
        phase_distribution: Dict[int, int] = {}

        for row in indication_rows:
            disease = row.get("disease", {})
            phase = row.get("maxPhaseForIndication", 0)
            phase_int = int(phase) if phase is not None else 0
            phase_distribution[phase_int] = phase_distribution.get(phase_int, 0) + 1

            refs = row.get("references", [])
            reference_list = []
            for ref in refs:
                reference_list.append({
                    "source": ref.get("source", ""),
                    "ids": ref.get("ids", []),
                })

            indications.append({
                "disease_id": disease.get("id", ""),
                "disease_name": disease.get("name", ""),
                "max_phase": phase_int,
                "references": reference_list,
            })

        # Parse mechanisms of action
        moa_rows = (
            drug_data.get("mechanismsOfAction", {}).get("rows", [])
        )
        mechanisms: List[Dict[str, Any]] = []
        for moa in moa_rows:
            targets = [
                {
                    "id": t.get("id", ""),
                    "symbol": t.get("approvedSymbol", ""),
                }
                for t in moa.get("targets", [])
            ]
            mechanisms.append({
                "mechanism": moa.get("mechanismOfAction", ""),
                "targets": targets,
            })

        stats = {
            "chembl_id": chembl_id,
            "drug_name": drug_data.get("name", ""),
            "drug_type": drug_data.get("drugType", ""),
            "max_clinical_trial_phase": drug_data.get(
                "maximumClinicalTrialPhase", 0
            ),
            "total_indications": indications_data.get("count", 0),
            "n_indications": len(indications),
            "indications": indications,
            "phase_distribution": phase_distribution,
            "mechanisms_of_action": mechanisms,
            "analysis_type": "opentargets_drug_indications",
        }

        logger.info(
            f"Drug indications complete for {chembl_id}: "
            f"{len(indications)} indications, max phase "
            f"{drug_data.get('maximumClinicalTrialPhase', 'N/A')}"
        )
        return None, stats, ir

    def get_safety_profile(
        self,
        target_id: str,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Get known adverse events and safety liabilities for a target.

        Queries Open Targets for experimental and observed toxicity signals
        including tissue-specific effects and study references.

        Args:
            target_id: Ensembl gene ID (e.g., "ENSG00000157764").

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
        """
        logger.info(f"Querying safety profile for target {target_id}")

        ir = self._create_ir_get_safety_profile(target_id)

        query = """
        query TargetSafety($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            safetyLiabilities {
              event
              datasource
              literature
              url
              biosamples {
                tissueLabel
                tissueId
              }
              effects { dosing direction }
              studies {
                name
                description
                type
              }
            }
          }
        }
        """
        variables = {"ensemblId": target_id}

        data, error = self._post_graphql(query, variables)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "target_id": target_id,
                "n_safety_events": 0,
                "analysis_type": "opentargets_safety_profile",
            }
            return None, stats, ir

        target_data = data.get("target")
        if target_data is None:
            stats = {
                "error": f"Target {target_id} not found in Open Targets",
                "target_id": target_id,
                "n_safety_events": 0,
                "analysis_type": "opentargets_safety_profile",
            }
            return None, stats, ir

        liabilities_raw = target_data.get("safetyLiabilities") or []

        # Parse safety events
        events: List[Dict[str, Any]] = []
        event_categories: Dict[str, int] = {}
        affected_tissues: Dict[str, int] = {}

        for liability in liabilities_raw:
            event_name = liability.get("event", "Unknown")
            event_categories[event_name] = event_categories.get(event_name, 0) + 1

            # Collect tissue info
            biosamples = liability.get("biosamples") or []
            tissues = []
            for bs in biosamples:
                tissue_label = bs.get("tissueLabel", "")
                if tissue_label:
                    tissues.append(tissue_label)
                    affected_tissues[tissue_label] = (
                        affected_tissues.get(tissue_label, 0) + 1
                    )

            # Parse studies
            studies_raw = liability.get("studies") or []
            studies = [
                {
                    "name": s.get("name", ""),
                    "description": s.get("description", ""),
                    "type": s.get("type", ""),
                }
                for s in studies_raw
            ]

            # Parse effects (dosing + direction)
            effects_raw = liability.get("effects") or []
            effects = [
                {
                    "dosing": e.get("dosing", ""),
                    "direction": e.get("direction", ""),
                }
                for e in effects_raw
            ]

            events.append({
                "event": event_name,
                "datasource": liability.get("datasource", ""),
                "effects": effects,
                "tissues": tissues,
                "biosamples_from_source": liability.get("biosamplesFromSource", ""),
                "literature": liability.get("literature", ""),
                "url": liability.get("url", ""),
                "studies": studies,
            })

        # Risk assessment
        n_events = len(events)
        if n_events == 0:
            risk_level = "low"
        elif n_events <= 3:
            risk_level = "moderate"
        else:
            risk_level = "high"

        stats = {
            "target_id": target_id,
            "approved_symbol": target_data.get("approvedSymbol", ""),
            "approved_name": target_data.get("approvedName", ""),
            "n_safety_events": n_events,
            "risk_level": risk_level,
            "event_categories": event_categories,
            "affected_tissues": affected_tissues,
            "events": events,
            "analysis_type": "opentargets_safety_profile",
        }

        logger.info(
            f"Safety profile for {target_id}: {n_events} events, "
            f"risk level={risk_level}"
        )
        return None, stats, ir

    def assess_tractability(
        self,
        target_id: str,
    ) -> Tuple[None, Dict[str, Any], AnalysisStep]:
        """
        Assess antibody and small molecule tractability for a target.

        Queries Open Targets for tractability assessments across modalities
        (small molecule, antibody, PROTAC, other clinical modalities) and
        returns a structured summary with per-modality verdicts.

        Args:
            target_id: Ensembl gene ID (e.g., "ENSG00000157764").

        Returns:
            Tuple of (None, stats_dict, AnalysisStep).
        """
        logger.info(f"Assessing tractability for target {target_id}")

        ir = self._create_ir_assess_tractability(target_id)

        query = """
        query TargetTractability($ensemblId: String!) {
          target(ensemblId: $ensemblId) {
            id
            approvedSymbol
            approvedName
            tractability {
              label
              modality
              value
            }
          }
        }
        """
        variables = {"ensemblId": target_id}

        data, error = self._post_graphql(query, variables)

        if error is not None:
            stats: Dict[str, Any] = {
                "error": error,
                "target_id": target_id,
                "n_assessments": 0,
                "analysis_type": "opentargets_tractability",
            }
            return None, stats, ir

        target_data = data.get("target")
        if target_data is None:
            stats = {
                "error": f"Target {target_id} not found in Open Targets",
                "target_id": target_id,
                "n_assessments": 0,
                "analysis_type": "opentargets_tractability",
            }
            return None, stats, ir

        tractability_raw = target_data.get("tractability") or []

        # Group assessments by modality
        modality_map: Dict[str, List[Dict[str, Any]]] = {}
        for entry in tractability_raw:
            modality = entry.get("modality", "unknown")
            if modality not in modality_map:
                modality_map[modality] = []
            modality_map[modality].append({
                "label": entry.get("label", ""),
                "value": entry.get("value", False),
            })

        # Compute per-modality verdicts
        modality_verdicts: Dict[str, Dict[str, Any]] = {}
        for modality, assessments in modality_map.items():
            n_positive = sum(1 for a in assessments if a["value"])
            n_total = len(assessments)
            modality_verdicts[modality] = {
                "assessments": assessments,
                "n_positive": n_positive,
                "n_total": n_total,
                "tractable": n_positive > 0,
            }

        # Overall tractability: at least one modality is tractable
        is_tractable = any(v["tractable"] for v in modality_verdicts.values())
        tractable_modalities = [
            m for m, v in modality_verdicts.items() if v["tractable"]
        ]

        stats = {
            "target_id": target_id,
            "approved_symbol": target_data.get("approvedSymbol", ""),
            "approved_name": target_data.get("approvedName", ""),
            "n_assessments": len(tractability_raw),
            "is_tractable": is_tractable,
            "tractable_modalities": tractable_modalities,
            "modality_verdicts": modality_verdicts,
            "analysis_type": "opentargets_tractability",
        }

        logger.info(
            f"Tractability for {target_id}: "
            f"{'tractable' if is_tractable else 'not tractable'} "
            f"({', '.join(tractable_modalities) if tractable_modalities else 'none'})"
        )
        return None, stats, ir
