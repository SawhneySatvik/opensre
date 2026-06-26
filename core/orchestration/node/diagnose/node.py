"""Diagnose node — structured investigation result parsed from the agent's final LLM response."""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel

from core.domain.alerts import resolve_alert_source
from core.domain.diagnosis import (
    InvestigationResult,
    build_diagnosis_extraction_prompt,
    build_diagnosis_schema,
    investigation_result_from_schema,
    resolve_diagnosis_from_messages,
    result_to_state,
    taxonomy_categories_for_alert_source,
)
from core.domain.state import InvestigationState

logger = logging.getLogger(__name__)


def parse_diagnosis(
    messages: list[dict[str, Any]],
    evidence: dict[str, Any],
    alert_name: str = "",
    alert_source: str = "",
) -> InvestigationResult:
    """Parse the agent's final response into a structured InvestigationResult."""
    return resolve_diagnosis_from_messages(
        messages,
        alert_name=alert_name,
        structured_parse=lambda last_text: _parse_via_structured_output(
            last_text, evidence, alert_source=alert_source
        ),
        legacy_parse=lambda last_text: _parse_via_legacy(
            last_text, evidence, alert_name, alert_source=alert_source
        ),
    )


def diagnose(state: InvestigationState) -> dict[str, Any]:
    """Parse investigation output into structured RCA fields."""
    if str(state.get("root_cause") or "").strip():
        return {}

    from platform.analytics.cli import capture_diagnosis_category_mismatch
    from platform.observability import get_progress_tracker

    tracker = get_progress_tracker()
    tracker.start("diagnose_root_cause", "Parsing investigation conclusion")

    messages = _list_of_dicts(state.get("agent_messages"))
    raw_evidence = state.get("evidence")
    evidence = cast(dict[str, Any], raw_evidence) if isinstance(raw_evidence, dict) else {}
    result = parse_diagnosis(
        messages,
        evidence,
        str(state.get("alert_name") or ""),
        alert_source=resolve_alert_source(cast(dict[str, Any], state)),
    )
    result.evidence = evidence
    result.evidence_entries = _list_of_dicts(state.get("evidence_entries"))
    result.agent_messages = messages

    if result.category_text_mismatch:
        capture_diagnosis_category_mismatch(
            root_cause_category=result.root_cause_category,
            mismatch_reason=result.category_text_mismatch_reason,
        )
        logger.warning(
            "Root cause category may not match explanation: %s",
            result.category_text_mismatch_reason,
        )

    tracker.complete(
        "diagnose_root_cause",
        fields_updated=["root_cause", "validated_claims", "remediation_steps"],
        message=f"validity:{result.validity_score:.0%} category:{result.root_cause_category}",
    )
    return result_to_state(result)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _parse_via_structured_output(
    last_text: str,
    evidence: dict[str, Any],
    *,
    alert_source: str = "",
) -> InvestigationResult:
    from services import get_llm_for_reasoning

    prompt = build_diagnosis_extraction_prompt(last_text, evidence)

    llm = get_llm_for_reasoning()
    schema_model = build_diagnosis_schema(taxonomy_categories_for_alert_source(alert_source))
    raw_schema = (
        llm.with_structured_output(schema_model)
        .with_config(run_name="LLM – Parse diagnosis")
        .invoke(prompt)
    )
    schema_instance = (
        raw_schema if isinstance(raw_schema, BaseModel) else schema_model.model_validate(raw_schema)
    )
    schema = cast(dict[str, Any], schema_instance.model_dump())

    return investigation_result_from_schema(schema, alert_source=alert_source)


def _parse_via_legacy(
    last_text: str,
    _evidence: dict[str, Any],
    alert_name: str,
    *,
    alert_source: str = "",
) -> InvestigationResult:
    from services import parse_root_cause

    try:
        rr = parse_root_cause(last_text)
        return investigation_result_from_schema(
            {
                "root_cause": rr.root_cause,
                "root_cause_category": rr.root_cause_category,
                "causal_chain": rr.causal_chain,
                "validated_claims": rr.validated_claims,
                "non_validated_claims": rr.non_validated_claims,
                "remediation_steps": rr.remediation_steps,
                "validity_score": 0.5,
            },
            alert_source=alert_source,
        )
    except Exception as err:
        logger.warning("Legacy parse_root_cause also failed: %s", err)
        return InvestigationResult.unknown(alert_name)
