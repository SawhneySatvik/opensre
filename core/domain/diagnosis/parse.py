"""Pure diagnosis parsing orchestration (no LLM or I/O)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.domain.diagnosis.result import (
    InvestigationResult,
    build_investigation_result,
    extract_last_assistant_text,
)


def build_diagnosis_extraction_prompt(last_text: str, evidence: dict[str, Any]) -> str:
    """Build the structured-extraction prompt for the diagnose stage."""
    evidence_keys = ", ".join(evidence.keys()) if evidence else "none"
    return f"""Extract the structured diagnosis from this investigation conclusion.

Investigation conclusion:
{last_text}

Evidence keys collected: {evidence_keys}
"""


def resolve_diagnosis_from_messages(
    messages: list[dict[str, Any]],
    *,
    alert_name: str = "",
    structured_parse: Callable[[str], InvestigationResult],
    legacy_parse: Callable[[str], InvestigationResult],
) -> InvestigationResult:
    """Resolve a diagnosis from agent messages using structured then legacy parsers."""
    last_text = extract_last_assistant_text(messages)
    if not last_text:
        return InvestigationResult.unknown(alert_name)

    try:
        return structured_parse(last_text)
    except Exception:
        return legacy_parse(last_text)


def investigation_result_from_schema(
    schema: dict[str, Any],
    *,
    alert_source: str = "",
) -> InvestigationResult:
    """Build an InvestigationResult from a structured diagnosis schema payload."""
    return build_investigation_result(
        root_cause=str(schema["root_cause"]),
        root_cause_category=str(schema["root_cause_category"]),
        causal_chain=list(schema.get("causal_chain") or []),
        validated_claims=list(schema.get("validated_claims") or []),
        non_validated_claims=list(schema.get("non_validated_claims") or []),
        remediation_steps=list(schema.get("remediation_steps") or []),
        validity_score=float(schema.get("validity_score") or 0.0),
        alert_source=alert_source,
    )
