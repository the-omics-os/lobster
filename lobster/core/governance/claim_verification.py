"""Detect unsupported supervisor claims against recorded artifacts.

The supervisor may assert that an analysis succeeded even when no work was recorded.
This detector compares the response with the artifact manifest, which is derived from
``DataManagerV2`` state rather than prose.

Scope
-----
This detects the empty-manifest case — success claimed, nothing written. It does not
validate artifact contents or detect a wrong result. It records a flag but cannot prevent
a claim from reaching the user; enforcement would require a graph node or edge to gate on.

Detection is deliberately conservative. False positives create noisy records, so heuristics
abstain when uncertain. ``VERDICT_UNDETERMINED`` distinguishes an inconclusive check from
one that found no unsupported claim.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lobster.core.governance.routing_telemetry import SCHEMA_VERSION

#: Event kind written into the shared ``routing_events.jsonl`` stream. Both event types
#: describe the same session, so a reader need not join across files.
EVENT_CLAIM_CHECK = "claim_check"

VERDICT_SUPPORTED = "supported"
VERDICT_UNSUPPORTED = "unsupported"
VERDICT_NO_CLAIM = "no_claim"
VERDICT_UNDETERMINED = "undetermined"

#: Turn types. A turn that correctly writes nothing must never be flagged.
TURN_ANALYTICAL = "analytical"
TURN_CONVERSATIONAL = "conversational"

#: Cap on stored prose excerpts. Records are evidence, not transcripts.
MAX_EXCERPT_CHARS = 300

# ---------------------------------------------------------------------------
# Claim detection
#
# Deliberately narrow. These match prose that asserts a COMPLETED analysis, not prose
# that merely mentions one. Broad patterns ("done", "here are the results") were
# considered and rejected: they fire on legitimate summaries of work a specialist really
# did, and every such false positive makes the check less trustworthy.
# ---------------------------------------------------------------------------
_COMPLETION_CLAIM_PATTERNS = (
    r"\bsuccessfully\s+(?:calculated|computed|completed|ran|performed|generated|created|filtered|normalized|clustered|analyzed|analysed|identified)\b",
    r"\bi\s+(?:have\s+)?(?:calculated|computed|completed|ran|performed|generated|created|filtered|normalized|clustered|analyzed|analysed|identified)\b",
    r"\b(?:analysis|computation|clustering|normalization|filtering|qc|quality control)\s+(?:is\s+)?(?:now\s+)?complete[d]?\b",
    r"\bhas\s+been\s+(?:calculated|computed|completed|generated|created|filtered|normalized|clustered)\b",
)

#: Phrases that mark an *unfulfilled* claim. Their presence means the supervisor is
#: reporting a failure or a plan, so an empty manifest is consistent and expected.
_NEGATION_PATTERNS = (
    r"\b(?:could\s+not|couldn't|cannot|can't|unable\s+to|failed\s+to|was\s+not\s+able)\b",
    r"\bno\s+(?:data|modalities|modality|dataset)\b",
    r"\b(?:i\s+will|i'll|next\s+step|would\s+need|please\s+(?:upload|provide|load))\b",
    r"\b(?:error|exception|traceback|not\s+installed|missing\s+dependency)\b",
)

#: Signals that a turn is conversational rather than analytical. Concept explanations,
#: greetings and capability questions correctly produce no artifacts (Cognitive Protocol
#: category A), so they are excluded before any claim is evaluated.
_CONVERSATIONAL_REQUEST_PATTERNS = (
    r"^\s*(?:hi|hello|hey|thanks|thank\s+you)\b",
    r"\bwhat\s+(?:is|are|does|do)\b",
    r"\bwhy\s+(?:is|are|does|do)\b",
    r"\bhow\s+(?:do|does|would|should)\s+(?:i|you|we)\b",
    r"\b(?:explain|describe|tell\s+me\s+about|define)\b",
    r"\bwhich\s+agents?\b",
    r"\bwhat\s+can\s+you\s+do\b",
    r"\blist\s+(?:the\s+)?(?:available\s+)?(?:modalities|agents|files|tools)\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def classify_turn(user_request: str) -> str:
    """Classify a turn as analytical or conversational.

    Only analytical turns are expected to produce artifacts. Getting this wrong in the
    conversational direction is the main false-positive risk, so the conversational test
    is applied generously: when a request looks like a question or an explanation
    request, we treat it as conversational and never flag it.
    """
    text = (user_request or "").strip()
    if not text:
        # No request recorded: we cannot establish an expectation, so abstain by
        # treating it as conversational rather than risk a false flag.
        return TURN_CONVERSATIONAL
    if _matches_any(text, _CONVERSATIONAL_REQUEST_PATTERNS):
        return TURN_CONVERSATIONAL
    return TURN_ANALYTICAL


def find_completion_claim(response_text: str) -> str | None:
    """Return the matched claim phrase, or None when the prose asserts no completion.

    Returns None when a negation/failure marker is present anywhere in the response: a
    supervisor saying "I could not complete the analysis because the dependency is
    missing" is being honest, and an empty manifest is the correct outcome. Checking
    negation first is what keeps honest failure reports out of the flag log.
    """
    text = (response_text or "").strip()
    if not text:
        return None
    if _matches_any(text, _NEGATION_PATTERNS):
        return None
    return _matches_any(text, _COMPLETION_CLAIM_PATTERNS)


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

_MANIFEST_BLOCK = re.compile(
    r"<artifact_manifest>(.*?)</artifact_manifest>", re.DOTALL | re.IGNORECASE
)
_EMPTY_MANIFEST_MARKER = "no new artifacts"


@dataclass(frozen=True)
class ManifestState:
    """What the manifests in a turn say about artifacts.

    ``present=False`` means no manifest was found — the feature flag was off, or no handoff
    occurred. That is **not** the same as an empty manifest, and conflating them is the
    error that would make this detector unsound: absent means "not measured", empty means
    "measured, nothing written".
    """

    present: bool
    has_artifacts: bool
    excerpts: tuple[str, ...] = ()


def read_manifests(handoff_returns: list[str]) -> ManifestState:
    """Parse artifact manifests out of the handoff returns seen during a turn.

    A turn may include several handoffs; artifacts from any of them count, because the
    question is whether the claimed work exists at all, not which agent produced it.
    """
    excerpts: list[str] = []
    found = False
    has_artifacts = False

    for text in handoff_returns or []:
        for block in _MANIFEST_BLOCK.findall(text or ""):
            found = True
            body = block.strip()
            excerpts.append(body[:MAX_EXCERPT_CHARS])
            if _EMPTY_MANIFEST_MARKER not in body.lower():
                has_artifacts = True

    return ManifestState(
        present=found, has_artifacts=has_artifacts, excerpts=tuple(excerpts)
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass
class ClaimCheck:
    """One turn's verdict, and the evidence behind it."""

    verdict: str
    turn_type: str
    claim_excerpt: str | None = None
    manifest_present: bool = False
    manifest_has_artifacts: bool = False
    manifest_excerpts: tuple[str, ...] = ()
    request_excerpt: str | None = None
    tool_sequence: list[str] = field(default_factory=list)
    reason: str = ""

    @property
    def is_flagged(self) -> bool:
        return self.verdict == VERDICT_UNSUPPORTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": EVENT_CLAIM_CHECK,
            "v": SCHEMA_VERSION,
            "verdict": self.verdict,
            "turn_type": self.turn_type,
            "claim_excerpt": self.claim_excerpt,
            "manifest_present": self.manifest_present,
            "manifest_has_artifacts": self.manifest_has_artifacts,
            "manifest_excerpts": list(self.manifest_excerpts),
            "request_excerpt": self.request_excerpt,
            "tool_sequence": list(self.tool_sequence),
            "reason": self.reason,
        }


def verify_claim(
    user_request: str,
    response_text: str,
    handoff_returns: list[str] | None = None,
    tool_sequence: list[str] | None = None,
) -> ClaimCheck:
    """Check a supervisor response against the artifacts actually produced.

    Decision order is chosen so every ambiguous case abstains:

    1. conversational turn      -> ``no_claim``      (correctly writes nothing)
    2. no completion claim      -> ``no_claim``      (nothing asserted to verify)
    3. no manifest at all       -> ``undetermined``  (not measured; never "clean")
    4. manifest lists artifacts -> ``supported``
    5. manifest explicitly empty-> ``unsupported``   (the only flagged outcome)
    """
    handoff_returns = handoff_returns or []
    tool_sequence = tool_sequence or []
    request_excerpt = (user_request or "")[:MAX_EXCERPT_CHARS] or None

    turn_type = classify_turn(user_request)
    manifest = read_manifests(handoff_returns)

    base = {
        "turn_type": turn_type,
        "manifest_present": manifest.present,
        "manifest_has_artifacts": manifest.has_artifacts,
        "manifest_excerpts": manifest.excerpts,
        "request_excerpt": request_excerpt,
        "tool_sequence": list(tool_sequence),
    }

    if turn_type == TURN_CONVERSATIONAL:
        return ClaimCheck(
            verdict=VERDICT_NO_CLAIM,
            reason="conversational turn; artifacts are not expected",
            **base,
        )

    claim = find_completion_claim(response_text)
    if claim is None:
        return ClaimCheck(
            verdict=VERDICT_NO_CLAIM,
            reason="no completion claim asserted (or an explicit failure was reported)",
            **base,
        )

    if not manifest.present:
        # The honest answer. Reporting "clean" here would mean the detector goes quiet
        # exactly when its input is missing, which is how a monitoring system ends up
        # certifying a system it never observed.
        return ClaimCheck(
            verdict=VERDICT_UNDETERMINED,
            claim_excerpt=claim,
            reason=(
                "completion claimed but no artifact manifest was available "
                "(enable LOBSTER_HANDOFF_MANIFEST); cannot verify"
            ),
            **base,
        )

    if manifest.has_artifacts:
        return ClaimCheck(
            verdict=VERDICT_SUPPORTED,
            claim_excerpt=claim,
            reason="completion claimed and artifacts were written",
            **base,
        )

    return ClaimCheck(
        verdict=VERDICT_UNSUPPORTED,
        claim_excerpt=claim,
        reason=(
            "completion claimed but the manifest reports no artifacts were written; "
            "the claim is unverified"
        ),
        **base,
    )


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class ClaimVerificationRecorder:
    """Accumulates claim checks and persists them into the shared event stream.

    Writes ``claim_check`` records into the same ``routing_events.jsonl`` as routing
    telemetry: one stream per session, so a reader never has to join two files to ask
    "what happened, and was what the supervisor said about it true?"

    Observational and fail-open, like the AQUADIF monitor and routing telemetry. A
    verification bug must never break a session.
    """

    def __init__(self, session_dir: str | Path | None = None) -> None:
        self.checks: list[ClaimCheck] = []
        self._session_dir = Path(session_dir) if session_dir else None

    def record(
        self,
        user_request: str,
        response_text: str,
        handoff_returns: list[str] | None = None,
        tool_sequence: list[str] | None = None,
    ) -> ClaimCheck | None:
        """Verify one turn and store the result. Never raises."""
        try:
            check = verify_claim(
                user_request=user_request,
                response_text=response_text,
                handoff_returns=handoff_returns,
                tool_sequence=tool_sequence,
            )
            self.checks.append(check)
            return check
        except Exception:  # noqa: BLE001 - detection must never break a session
            return None

    @property
    def flagged(self) -> list[ClaimCheck]:
        return [c for c in self.checks if c.is_flagged]

    def base_rate(self) -> dict[str, Any]:
        """How often claims went unsupported, over turns where that was decidable.

        The denominator is deliberately *verifiable* turns only — those with a claim and
        a manifest. Including conversational turns would dilute the rate toward zero and
        make the number useless for judging whether a reviewer agent is worth its cost.
        """
        verdicts = [c.verdict for c in self.checks]
        decidable = [
            v for v in verdicts if v in (VERDICT_SUPPORTED, VERDICT_UNSUPPORTED)
        ]
        counts = {
            VERDICT_SUPPORTED: verdicts.count(VERDICT_SUPPORTED),
            VERDICT_UNSUPPORTED: verdicts.count(VERDICT_UNSUPPORTED),
            VERDICT_UNDETERMINED: verdicts.count(VERDICT_UNDETERMINED),
            VERDICT_NO_CLAIM: verdicts.count(VERDICT_NO_CLAIM),
        }
        return {
            "turns_total": len(verdicts),
            "turns_verifiable": len(decidable),
            "counts": counts,
            "unsupported_rate": (
                round(counts[VERDICT_UNSUPPORTED] / len(decidable), 4)
                if decidable
                # None, not 0.0: with nothing verifiable there is no rate to report,
                # and 0.0 would read as "no fabrication found".
                else None
            ),
        }

    def flush(self) -> Path | None:
        """Append records to ``routing_events.jsonl``. Never raises."""
        if self._session_dir is None or not self.checks:
            return None
        try:
            self._session_dir.mkdir(parents=True, exist_ok=True)
            path = self._session_dir / "routing_events.jsonl"
            with open(path, "a", encoding="utf-8") as handle:
                handle.writelines(
                    json.dumps(check.to_dict()) + "\n" for check in self.checks
                )
                handle.flush()
                os.fsync(handle.fileno())
            return path
        except Exception:  # noqa: BLE001 - fail open
            return None


def summarize_claim_checks(paths: list[Path]) -> dict[str, Any]:
    """Aggregate claim checks across sessions into a production base rate.

    This is what a future reviewer agent's cost/benefit case rests on: how often this
    actually fires in real use, rather than an assumption about how often it might.
    """
    counts = {
        VERDICT_SUPPORTED: 0,
        VERDICT_UNSUPPORTED: 0,
        VERDICT_UNDETERMINED: 0,
        VERDICT_NO_CLAIM: 0,
    }
    sessions = 0
    flagged_examples: list[dict[str, Any]] = []

    for path in paths:
        try:
            saw_any = False
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("kind") != EVENT_CLAIM_CHECK:
                    continue
                saw_any = True
                verdict = record.get("verdict")
                if verdict in counts:
                    counts[verdict] += 1
                if verdict == VERDICT_UNSUPPORTED and len(flagged_examples) < 20:
                    flagged_examples.append(
                        {
                            "claim_excerpt": record.get("claim_excerpt"),
                            "request_excerpt": record.get("request_excerpt"),
                            "tool_sequence": record.get("tool_sequence"),
                        }
                    )
            if saw_any:
                sessions += 1
        except (
            Exception
        ):  # nosec B112 # A malformed telemetry file must not abort independent file aggregation. noqa: BLE001 - one bad file must not stop the sweep
            continue

    decidable = counts[VERDICT_SUPPORTED] + counts[VERDICT_UNSUPPORTED]
    result: dict[str, Any] = {
        "sessions_with_checks": sessions,
        "counts": counts,
        "turns_verifiable": decidable,
        "flagged_examples": flagged_examples,
    }
    if decidable:
        result["unsupported_rate"] = round(counts[VERDICT_UNSUPPORTED] / decidable, 4)
    else:
        result["unsupported_rate"] = None
        result["note"] = (
            "no verifiable turns (a claim plus a manifest); the rate is undefined "
            "rather than zero. If undetermined dominates, LOBSTER_HANDOFF_MANIFEST "
            "was probably off."
        )
    return result
