"""Alert parsing, normalization, source routing, and tool-planning rules."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from core.domain.types.planning import PlannedInvestigationAction

CANONICAL_ALERT_SOURCES = frozenset({"opensre", "opensre_dataset"})

RAW_ALERT_DETAIL_FIELDS = (
    "kube_namespace",
    "cloudwatch_log_group",
    "error_message",
    "log_query",
    "eks_cluster",
    "pod_name",
    "deployment",
)


class AlertDetails(BaseModel):
    is_noise: bool = Field(default=False)
    alert_name: str = Field(default="unknown")
    pipeline_name: str = Field(default="unknown")
    severity: str = Field(default="unknown")
    alert_source: str | None = Field(default=None)
    environment: str | None = Field(default=None)
    summary: str | None = Field(default=None)
    kube_namespace: str | None = Field(default=None)
    cloudwatch_log_group: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    log_query: str | None = Field(default=None)
    eks_cluster: str | None = Field(default=None)
    pod_name: str | None = Field(default=None)
    deployment: str | None = Field(default=None)


def format_raw_alert(raw_alert: Any) -> str:
    if isinstance(raw_alert, str):
        return raw_alert
    if isinstance(raw_alert, dict):
        if raw_alert.get("text") and not needs_full_json_prompt(raw_alert):
            return str(raw_alert["text"])
        return json.dumps(raw_alert, indent=2, sort_keys=True)
    return json.dumps(raw_alert, indent=2, sort_keys=True)


def needs_full_json_prompt(raw_alert: dict[str, Any]) -> bool:
    src = str(raw_alert.get("alert_source", "")).lower()
    if src in CANONICAL_ALERT_SOURCES:
        return True
    if (
        raw_alert.get("commonLabels")
        or raw_alert.get("commonAnnotations")
        or raw_alert.get("alerts")
    ):
        return True
    for key in (
        "opensre_telemetry_relative",
        "opensre_dataset_root",
    ):
        if raw_alert.get(key):
            return True
        ann = raw_alert.get("commonAnnotations")
        if isinstance(ann, dict) and ann.get(key):
            return True
    meta = raw_alert.get("_meta")
    return bool(isinstance(meta, dict) and "opensre" in str(meta.get("purpose", "")).lower())


def fallback_details(state: Mapping[str, Any], raw_alert: Any) -> AlertDetails:
    alert_name = state.get("alert_name", "unknown")
    pipeline_name = state.get("pipeline_name", "unknown")
    severity = state.get("severity", "unknown")

    if isinstance(raw_alert, dict):
        labels = dict_value(raw_alert, "commonLabels") or dict_value(raw_alert, "labels")
        annotations = dict_value(raw_alert, "commonAnnotations") or dict_value(
            raw_alert, "annotations"
        )
        canonical = dict_value(raw_alert, "canonical_alert")

        alert_name = first_value(
            raw_alert.get("alert_name"),
            canonical.get("alert_name"),
            labels.get("alertname"),
            labels.get("alert_name"),
            alert_name,
        )
        pipeline_name = first_value(
            raw_alert.get("pipeline_name"),
            canonical.get("pipeline_name"),
            labels.get("pipeline_name"),
            labels.get("pipeline"),
            labels.get("service"),
            annotations.get("pipeline_name"),
            pipeline_name,
        )
        severity = first_value(
            raw_alert.get("severity"),
            canonical.get("severity"),
            labels.get("severity"),
            severity,
        )

    return AlertDetails(
        is_noise=False,
        alert_name=alert_name or "unknown",
        pipeline_name=pipeline_name or "unknown",
        severity=severity or "unknown",
    )


def dict_value(source: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key)
    return value if isinstance(value, dict) else {}


def first_value(*values: Any) -> Any:
    return next((value for value in values if value), None)


def make_problem_md(details: AlertDetails) -> str:
    parts = [
        f"# {details.alert_name}",
        f"Pipeline: {details.pipeline_name} | Severity: {details.severity}",
    ]
    if details.kube_namespace:
        parts.append(f"Namespace: {details.kube_namespace}")
    if details.error_message:
        parts.append(f"\nError: {details.error_message}")
    return "\n".join(parts)


def enrich_raw_alert(raw_alert: Any, details: AlertDetails) -> Any:
    if not isinstance(raw_alert, dict):
        raw_alert = {}
    enriched = dict(raw_alert)
    prior_source = str(raw_alert.get("alert_source", "")).lower()

    for field_name in RAW_ALERT_DETAIL_FIELDS:
        value = getattr(details, field_name)
        if value:
            enriched[field_name] = value

    if details.alert_source and prior_source not in CANONICAL_ALERT_SOURCES:
        enriched["alert_source"] = details.alert_source
    return enriched


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_tags(value: Any) -> dict[str, str]:
    """Parse Datadog-like tags into a dictionary.

    Supports:
    - comma-separated strings: "env:prod,service:payments"
    - list[str]: ["env:prod", "service:payments"]
    - dict[str, Any]: {"env": "prod"}
    """
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if _to_text(k) and _to_text(v)}

    items: Iterable[str]
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        items = [str(part).strip() for part in value if _to_text(part)]
    else:
        return {}

    parsed: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            continue
        key, raw_value = item.split(":", 1)
        key_text = _to_text(key)
        value_text = _to_text(raw_value)
        if key_text and value_text:
            parsed[key_text] = value_text
    return parsed


def _coerce_pid(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        pid = int(value)
        return pid if pid >= 0 else None
    text = _to_text(value)
    if text is None:
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    return pid if pid >= 0 else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def normalize_alert_payload(raw_alert: dict[str, Any]) -> dict[str, Any]:
    """Normalize an alert payload to canonical OpenSRE alert format.

    The returned payload preserves original fields and also adds:
    - ``commonLabels`` / ``commonAnnotations`` as dictionaries
    - top-level ``process_name`` / ``cmdline`` / ``pid`` when discovered
    - ``canonical_alert`` containing the normalized, vendor-agnostic shape
    """
    normalized = dict(raw_alert)

    raw_common_labels = normalized.get("commonLabels")
    labels = (
        _as_mapping(raw_common_labels)
        if raw_common_labels is not None
        else _as_mapping(normalized.get("labels"))
    )

    tags = _parse_tags(normalized.get("tags"))
    if tags:
        labels = {**tags, **labels}

    raw_common_annotations = normalized.get("commonAnnotations")
    annotations = (
        _as_mapping(raw_common_annotations)
        if raw_common_annotations is not None
        else _as_mapping(normalized.get("annotations"))
    )

    normalized["commonLabels"] = labels
    normalized["commonAnnotations"] = annotations

    process_name = _to_text(
        _first_present(
            normalized.get("process_name"),
            normalized.get("processName"),
            normalized.get("process.name"),
            normalized.get("procname"),
            labels.get("process_name"),
            labels.get("process"),
            annotations.get("process_name"),
        )
    )
    cmdline = _to_text(
        _first_present(
            normalized.get("cmdline"),
            normalized.get("command"),
            normalized.get("command_line"),
            normalized.get("process.cmdline"),
            normalized.get("process_command_line"),
            labels.get("cmdline"),
            annotations.get("cmdline"),
        )
    )
    pid = _coerce_pid(
        _first_present(
            normalized.get("pid"),
            normalized.get("process_id"),
            normalized.get("process.pid"),
            labels.get("pid"),
            annotations.get("pid"),
        )
    )

    if process_name and not _to_text(normalized.get("process_name")):
        normalized["process_name"] = process_name
    if cmdline and not _to_text(normalized.get("cmdline")):
        normalized["cmdline"] = cmdline
    if pid is not None and _coerce_pid(normalized.get("pid")) is None:
        normalized["pid"] = pid

    canonical_alert = {
        "schema": "opensre.alert.v1",
        "alert_name": _to_text(
            _first_present(
                normalized.get("alert_name"),
                normalized.get("title"),
                labels.get("alertname"),
                labels.get("alert_name"),
            )
        ),
        "pipeline_name": _to_text(
            _first_present(
                normalized.get("pipeline_name"),
                labels.get("pipeline_name"),
                labels.get("pipeline"),
                labels.get("service"),
            )
        ),
        "severity": _to_text(
            _first_present(
                normalized.get("severity"),
                labels.get("severity"),
                labels.get("priority"),
            )
        ),
        "alert_source": _to_text(normalized.get("alert_source")),
        "labels": dict(labels),
        "annotations": dict(annotations),
        "process": {
            "name": process_name,
            "cmdline": cmdline,
            "pid": pid,
        },
    }
    normalized["canonical_alert"] = canonical_alert
    return normalized


# Maps alert_source values to integration source keys (tool `.source` field).
# Used for broad prioritization/relevance, not automatic pre-seeding.
ALERT_SOURCE_TO_TOOL_SOURCES: dict[str, tuple[str, ...]] = {
    "grafana": ("grafana",),
    "datadog": ("datadog",),
    "cloudwatch": ("cloudwatch", "ec2", "rds", "cloudtrail"),
    "eks": ("eks", "ec2", "cloudtrail"),
    "alertmanager": ("eks", "cloudwatch", "grafana", "cloudtrail"),
    "sentry": ("sentry",),
    "honeycomb": ("honeycomb",),
    "coralogix": ("coralogix",),
    "airflow": ("airflow", "tracer_web"),
    "hermes": ("hermes",),
    "kafka": ("kafka",),
    "postgresql": ("postgresql",),
    "mysql": ("mysql",),
    "mariadb": ("mariadb",),
    "mongodb": ("mongodb", "mongodb_atlas"),
    "redis": ("redis",),
    "snowflake": ("snowflake",),
    "clickhouse": ("clickhouse",),
    "dagster": ("dagster",),
    "rabbitmq": ("rabbitmq",),
    "supabase": ("supabase",),
    "opensearch": ("opensearch",),
    "openobserve": ("openobserve",),
    "betterstack": ("betterstack",),
    "azure": ("azure", "azure_sql"),
    "github": ("github",),
    "gitlab": ("gitlab",),
    "bitbucket": ("bitbucket",),
    "argocd": ("eks",),
    "splunk": ("splunk",),
    "signoz": ("signoz",),
    "jenkins": ("jenkins",),
    "tempo": ("tempo",),
    "temporal": ("temporal",),
}

# Auto-called before the LLM loop starts. Keep this narrower than
# ALERT_SOURCE_TO_TOOL_SOURCES for expensive or context-dependent tools.
ALERT_SOURCE_TO_SEED_TOOL_SOURCES: dict[str, tuple[str, ...]] = {
    "grafana": ("grafana",),
    "datadog": ("datadog",),
    "cloudwatch": ("cloudwatch",),
    "eks": ("eks",),
    "alertmanager": ("grafana", "cloudwatch"),
    "sentry": ("sentry",),
    "honeycomb": ("honeycomb",),
    "coralogix": ("coralogix",),
    "airflow": ("airflow",),
    "hermes": ("hermes",),
    "kafka": ("kafka",),
    "postgresql": ("postgresql",),
    "mysql": ("mysql",),
    "mariadb": ("mariadb",),
    "mongodb": ("mongodb", "mongodb_atlas"),
    "redis": ("redis",),
    "snowflake": ("snowflake",),
    "clickhouse": ("clickhouse",),
    "dagster": ("dagster",),
    "rabbitmq": ("rabbitmq",),
    "supabase": ("supabase",),
    "opensearch": ("opensearch",),
    "openobserve": ("openobserve",),
    "betterstack": ("betterstack",),
    "azure": ("azure", "azure_sql"),
    "splunk": ("splunk",),
    "signoz": ("signoz",),
    "jenkins": ("jenkins",),
    "tempo": ("tempo",),
    "temporal": ("temporal",),
}

# Generic fallback sources: useful, but never primary when incident-specific
# integrations match.
SECONDARY_TOOL_SOURCES = frozenset({"knowledge", "openclaw", "google_docs"})

DB_KEYWORDS: tuple[str, ...] = ("database", "db connection", "connection pool")

SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "datadog": ("datadog", "datadoghq", "dd monitor"),
    "sentry": ("sentry", "exception", "stack trace", "stacktrace", "error tracking"),
    "vercel": ("vercel", "deploy", "deployment", "build failed"),
    "github": ("github", "commit", "pull request", "merge"),
    "gitlab": ("gitlab", "merge request"),
    "grafana": ("grafana", "loki", "mimir", "prometheus"),
    "honeycomb": ("honeycomb", "span", "trace latency"),
    "coralogix": ("coralogix",),
    "splunk": ("splunk",),
    "cloudwatch": ("cloudwatch", "lambda", "log group"),
    "eks": ("eks", "kubernetes", "k8s", "kubectl", "pod"),
    "ec2": ("ec2", "instance"),
    "rds": ("rds", "aurora", *DB_KEYWORDS),
    "postgresql": ("postgres", "postgresql", "psql", *DB_KEYWORDS),
    "mysql": ("mysql", *DB_KEYWORDS),
    "mariadb": ("mariadb", *DB_KEYWORDS),
    "mongodb": ("mongodb", "mongo", *DB_KEYWORDS),
    "redis": ("redis", "cache"),
    "snowflake": ("snowflake",),
    "clickhouse": ("clickhouse",),
    "dagster": ("dagster",),
    "airflow": ("airflow", "dag"),
    "kafka": ("kafka",),
    "rabbitmq": ("rabbitmq", "amqp"),
    "supabase": ("supabase",),
    "opensearch": ("opensearch", "elasticsearch"),
    "openobserve": ("openobserve",),
    "betterstack": ("betterstack", "better stack"),
    "azure": ("azure",),
    "signoz": ("signoz",),
    "jenkins": ("jenkins",),
    "tempo": ("tempo",),
    "temporal": ("temporal", "temporal workflow", "task queue"),
}


def primary_sources_for_alert(state: dict[str, Any]) -> tuple[str, ...]:
    """Return source keys that directly match the parsed alert source."""
    return ALERT_SOURCE_TO_TOOL_SOURCES.get(resolve_alert_source(state), ())


def declared_context_sources(state: dict[str, Any]) -> set[str]:
    """Return explicit context source annotations from the raw alert, if any."""
    raw = state.get("raw_alert")
    if not isinstance(raw, dict):
        return set()
    for block_key in ("commonAnnotations", "annotations", "commonLabels", "labels"):
        block = raw.get(block_key)
        if isinstance(block, dict):
            value = block.get("context_sources")
            if isinstance(value, str) and value.strip():
                return {item.strip().lower() for item in value.split(",") if item.strip()}
    return set()


def collect_alert_text(state: dict[str, Any]) -> str:
    """Collect searchable alert text for deterministic source/tool matching."""
    parts: list[str] = [
        str(state.get("alert_name") or ""),
        str(state.get("pipeline_name") or ""),
        str(state.get("message") or ""),
    ]
    raw = state.get("raw_alert")
    if isinstance(raw, dict):
        for key in ("alert_name", "title", "message", "text", "error_message", "kube_namespace"):
            value = raw.get(key)
            if isinstance(value, str):
                parts.append(value)
        for block_key in ("commonAnnotations", "annotations", "commonLabels", "labels"):
            block = raw.get(block_key)
            if isinstance(block, dict):
                parts.extend(str(v) for v in block.values() if isinstance(v, (str, int, float)))
    elif isinstance(raw, str):
        parts.append(raw)

    problem_md = state.get("problem_md")
    if isinstance(problem_md, str):
        parts.append(problem_md)

    return " ".join(part for part in parts if part).lower()


def relevant_sources_for_alert(
    state: dict[str, Any],
    candidate_sources: Iterable[str],
) -> list[str]:
    """Select candidate sources relevant to the alert content."""
    candidates = sorted(
        source for source in candidate_sources if source not in SECONDARY_TOOL_SOURCES
    )
    if not candidates:
        return []

    declared = declared_context_sources(state)
    if declared:
        from_declared = [source for source in candidates if source in declared]
        if from_declared:
            return from_declared

    text = collect_alert_text(state)
    if not text:
        return []

    matched: list[str] = []
    for source in candidates:
        keywords = {source, *SOURCE_ALIASES.get(source, ())}
        if any(keyword in text for keyword in keywords):
            matched.append(source)
    return matched


def resolve_alert_source(state: dict[str, Any]) -> str:
    source = str(state.get("alert_source") or "").lower().strip()
    if source:
        return source
    raw = state.get("raw_alert")
    if isinstance(raw, dict):
        source = str(raw.get("alert_source") or "").lower().strip()
        if source:
            return source
        labels = raw.get("commonLabels") or raw.get("labels") or {}
        if isinstance(labels, dict) and (
            labels.get("grafana_folder") or labels.get("datasource_uid")
        ):
            return "grafana"
        ext_url = raw.get("externalURL", "")
        if isinstance(ext_url, str) and "grafana" in ext_url.lower():
            return "grafana"
    return ""


FALLBACK_TOOL_NAMES: tuple[str, ...] = ("get_sre_guidance",)


def score_tools(
    state: dict[str, Any],
    tools: Sequence[Any],
) -> list[PlannedInvestigationAction]:
    primary_sources = set(primary_sources_for_alert(state))
    candidate_sources = {str(tool.source) for tool in tools}
    relevant_sources = set(relevant_sources_for_alert(state, candidate_sources))
    alert_text = collect_alert_text(state)
    existing_evidence = state.get("evidence")
    evidence_keys = set(existing_evidence) if isinstance(existing_evidence, dict) else set()

    scored = [
        score_tool(
            tool,
            alert_text=alert_text,
            primary_sources=primary_sources,
            relevant_sources=relevant_sources,
            evidence_keys=evidence_keys,
        )
        for tool in tools
    ]
    if scored and max(action.score for action in scored) <= 0:
        scored = [score_fallback_tool(action) for action in scored]

    return sorted(
        scored, key=lambda item: (-item.score, item.source in SECONDARY_TOOL_SOURCES, item.name)
    )


def score_tool(
    tool: Any,
    *,
    alert_text: str,
    primary_sources: set[str],
    relevant_sources: set[str],
    evidence_keys: set[str],
) -> PlannedInvestigationAction:
    source = str(tool.source)
    score = 0
    reasons: list[str] = []

    if source in primary_sources:
        score += 100
        reasons.append(f"source '{source}' matches alert source")
    if source in relevant_sources:
        score += 70
        reasons.append(f"source '{source}' matches alert context")
    if source in SECONDARY_TOOL_SOURCES:
        score -= 10
        reasons.append("secondary source, used after integration-specific tools")

    metadata_text = " ".join(
        [
            tool.description,
            " ".join(tool.use_cases),
            " ".join(tool.examples),
            " ".join(tool.tags),
            str(tool.evidence_type or ""),
        ]
    ).lower()
    metadata_matches = metadata_matches_for_alert(alert_text, metadata_text)
    if metadata_matches:
        score += min(len(metadata_matches), 5) * 4
        reasons.append(f"metadata matched alert terms: {', '.join(metadata_matches[:5])}")

    if tool.name in evidence_keys:
        score -= 25
        reasons.append("tool already has evidence in state")

    if not reasons:
        reasons.append("no source or metadata match")

    return PlannedInvestigationAction(
        name=tool.name,
        source=source,
        score=score,
        reasons=tuple(reasons),
    )


def metadata_matches_for_alert(alert_text: str, metadata_text: str) -> list[str]:
    if not alert_text or not metadata_text:
        return []
    terms = {
        term.strip(".,:;()[]{}").lower()
        for term in alert_text.split()
        if len(term.strip(".,:;()[]{}")) >= 4
    }
    return sorted(term for term in terms if term in metadata_text)


def score_fallback_tool(
    action: PlannedInvestigationAction,
) -> PlannedInvestigationAction:
    if action.name not in FALLBACK_TOOL_NAMES:
        return action
    return PlannedInvestigationAction(
        name=action.name,
        source=action.source,
        score=10,
        reasons=(*action.reasons, "included as deterministic fallback"),
    )
