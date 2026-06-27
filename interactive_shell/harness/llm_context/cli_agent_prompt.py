"""CLI-agent prompt construction for the interactive OpenSRE shell.

Builds the full conversational-assistant prompt from grounding sources, prior
investigation state, environment blocks, synthetic-run observations, and recent
conversation history. The turn runtime (``harness/agent.py``) calls
``build_cli_agent_prompt`` and stays out of the business of assembling prompt
text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from interactive_shell.harness.llm_context.assistant_system_prompt import (
    _build_environment_block,
    _build_observation_block,
    _build_system_prompt,
)
from interactive_shell.harness.llm_context.conversation_history import (
    format_recent_conversation,
)
from interactive_shell.harness.llm_context.grounding.investigation_flow_reference import (
    build_investigation_flow_reference_text,
)
from interactive_shell.harness.llm_context.session import (
    SUGGESTED_PROMPT_AFTER_FAILED_SYNTHETIC_TEST,
)
from interactive_shell.harness.turn_context import TurnContext
from interactive_shell.runtime import ReplSession

_logger = logging.getLogger(__name__)

_MAX_SYNTHETIC_OBSERVATION_PROMPT_CHARS = 120_000


def _summarize_evidence(evidence: Any) -> list[str]:
    """Render a short evidence preview for the prior-investigation grounding block.

    ``AgentState.evidence`` is a ``dict[str, Any]`` keyed by evidence id, but
    we accept list/other shapes defensively so an unexpected value doesn't
    silently drop all grounding context.
    """
    if isinstance(evidence, dict):
        sample_keys = list(evidence)[:3]
        sample = {key: evidence[key] for key in sample_keys}
        return [
            f"Evidence items: {len(evidence)}",
            "Evidence keys: " + ", ".join(map(str, sample_keys)),
            "Sample evidence:\n" + json.dumps(sample, indent=2, default=str)[:1500],
        ]
    if isinstance(evidence, list):
        return [
            f"Evidence items: {len(evidence)}",
            "Sample evidence:\n" + json.dumps(evidence[:3], indent=2, default=str)[:1500],
        ]
    return [
        f"Evidence type: {type(evidence).__name__}",
        f"Evidence summary:\n{str(evidence)[:1500]}",
    ]


def _summarize_last_state(state: dict[str, Any]) -> str:
    """Produce a compact text summary of the previous investigation for grounding."""
    parts: list[str] = []
    alert_name = state.get("alert_name")
    if alert_name:
        parts.append(f"Alert: {alert_name}")
    root_cause = state.get("root_cause")
    if root_cause:
        parts.append(f"Root cause: {root_cause}")
    problem_md = state.get("problem_md") or ""
    if problem_md:
        parts.append(f"Problem summary:\n{problem_md[:2000]}")
    slack_message = state.get("slack_message") or ""
    if slack_message:
        parts.append(f"Report:\n{slack_message[:2000]}")
    evidence = state.get("evidence")
    if evidence:
        try:
            parts.extend(_summarize_evidence(evidence))
        except (TypeError, ValueError) as exc:
            # Serialization can fail on exotic evidence values; tell the LLM
            # the context was withheld rather than silently dropping it.
            _logger.warning("could not serialize evidence for grounding: %s", exc)
            parts.append("(evidence present but could not be serialized for grounding)")
    return "\n\n".join(parts) or "(no prior investigation details available)"


def _user_message_requests_synthetic_failure_explanation(message: str) -> bool:
    """True when the user is likely asking about a failed synthetic benchmark."""
    m = message.strip().lower()
    if not m:
        return False
    suggested = SUGGESTED_PROMPT_AFTER_FAILED_SYNTHETIC_TEST.lower().rstrip("?")
    if m.rstrip("?") == suggested:
        return True
    if "why" in m and "fail" in m:
        return True
    return "what went wrong" in m


def _load_synthetic_observation_text(
    path_str: str, *, max_chars: int = _MAX_SYNTHETIC_OBSERVATION_PROMPT_CHARS
) -> str:
    try:
        raw = Path(path_str).read_text(encoding="utf-8")
    except OSError:
        return ""
    if len(raw) > max_chars:
        return (
            raw[:max_chars]
            + f"\n… [truncated for prompt size; observation is {len(raw)} characters total]"
        )
    return raw


def _build_integration_guard(ctx: TurnContext) -> str:
    """Render the no-integrations guidance block (pure over the snapshot)."""
    if not (ctx.configured_integrations_known and not ctx.configured_integrations):
        return ""

    return (
        "No integrations are configured in this session. You may still help the user "
        "configure one: when they ask to set up, connect, or add an integration, emit a "
        "run_interactive action for `/integrations setup <service>` (or `/mcp connect "
        "<server>`). Do NOT emit run_cli_command or slash actions to show/verify/remove "
        "integrations that are not configured; for those, answer with guidance only.\n\n"
    )


def _build_synthetic_failure_block(ctx: TurnContext) -> str:
    obs_path = ctx.last_synthetic_observation_path
    if not obs_path:
        return ""

    if not _user_message_requests_synthetic_failure_explanation(ctx.text):
        return ""

    obs_text = _load_synthetic_observation_text(obs_path)
    if not obs_text:
        return ""

    return (
        "The user is asking about a failed `opensre tests synthetic` run "
        "in this checkout. The JSON below is the saved observation "
        f"(scores, gates, stderr summary). Path: {obs_path}\n"
        "Use it to explain validation failures. Do not say nothing ran or "
        "that you lack context — the run completed and this file was written.\n\n"
        f"--- observation_json ---\n{obs_text}\n\n"
    )


def build_cli_agent_prompt(
    *,
    message: str,
    session: ReplSession,
    tool_observation: str | None,
    tool_observation_on_screen: bool,
    turn_ctx: TurnContext,
) -> str:
    """Read grounding sources / files / snapshot once and render the prompt string.

    All session and file reads happen here; the result is a single immutable
    prompt string ready to send to the reasoning LLM.
    """
    session.grounding.log_cache_diagnostics("cli_agent_grounding")

    system = _build_system_prompt(
        session.grounding.cli.build_text(),
        format_recent_conversation(list(turn_ctx.conversation_messages)),
        agents_md=session.grounding.agents_md.build_text(),
        investigation_flow=build_investigation_flow_reference_text(),
        prior_investigation=(
            _summarize_last_state(turn_ctx.last_state) if turn_ctx.last_state is not None else ""
        ),
        environment=_build_environment_block(session),
    )

    integration_guard = _build_integration_guard(turn_ctx)
    observation_block = _build_observation_block(
        tool_observation, on_screen=tool_observation_on_screen
    )
    synthetic_block = _build_synthetic_failure_block(turn_ctx)

    return (
        f"{system}\n"
        f"{integration_guard}"
        f"{observation_block}"
        f"{synthetic_block}"
        f"--- User message ---\n{message}"
    )
