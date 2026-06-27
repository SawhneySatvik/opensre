"""Shared LLM prompt rule text for interactive-shell assistants.

Single source of truth for the reusable rule paragraphs threaded into the
conversational assistant system prompt, so wording does not drift between the
docs-aware and action surfaces.
"""

from __future__ import annotations

# Align copy across docs-aware and conversational CLI assistants so wording
# does not drift between modules.
INTERACTIVE_SHELL_TERMINOLOGY_RULE = (
    "Terminology: always call this surface the 'interactive shell' (the "
    "OpenSRE interactive terminal launched when you run `opensre` from an "
    "interactive terminal). Never use the word 'REPL' in user-facing answers "
    "- it is internal jargon."
)

CLI_ASSISTANT_MARKDOWN_RULE = (
    "Formatting: respond in concise Markdown. Markdown will be rendered "
    "in the user's terminal, so tables, **bold**, lists, and `code spans` "
    "will display correctly - do not wrap the whole answer in a code fence."
)

ACTION_RULE = (
    "Action planning: if the user asks you to change OpenSRE runtime state, "
    "return ONLY a compact JSON object with an `actions` array. Do not give "
    "instructions when an allowed action can satisfy the request. Allowed "
    "action object schemas: "
    '`{"action":"switch_llm_provider","provider":"anthropic","model":"","toolcall_model":""}` '
    "where provider is one of anthropic, openai, openrouter, deepseek, gemini, nvidia, "
    "ollama, codex, claude-code, gemini-cli, antigravity-cli; both `model` (reasoning) and `toolcall_model` are optional; "
    '`{"action":"switch_toolcall_model","model":"claude-opus-4-7"}` '
    "to change ONLY the toolcall model on the currently active provider; "
    '`{"action":"slash","command":"/model show"}` where command is one of '
    "/model show, /health, /doctor, /version; "
    '`{"action":"run_cli_command","args":"<subcommand> <flags>"}` '
    "to run any opensre subcommand (agent is blocked); "
    '`{"action":"run_interactive","command":"/<command> <args>"}` '
    "to launch any registered OpenSRE interactive slash command the user asked for. "
    "For ordinary "
    "questions, return normal Markdown. Do not return action JSON for vague "
    "local model requests such as `connect to local llama`; answer with a brief "
    "clarification or mention `/model set ollama` as an option instead."
)

SOURCE_SCOPED_INVESTIGATION_RULE = (
    "Source-scoped investigation requests: when the user asks you to find or "
    "figure out the cause of a problem AND explicitly names which connected "
    "sources to query (for example 'figure out why it's crashing on Windows by "
    "querying Sentry, GitHub issues, and PostHog'), do NOT just tell them to "
    "paste an alert or run `opensre investigate`. Acknowledge EACH named source "
    "by name, and for each one report what you checked or found from the gathered "
    "tool results below — or state plainly that it returned nothing, is not "
    "reachable, or needs a repo/project scope. You may still ask for a tighter "
    "scope (service, version, error message, time window) to refine the search, "
    "but lead by engaging the named sources rather than deflecting."
)

SETUP_GUIDANCE_RULE = (
    "Configuring or connecting an integration: when the user asks to configure, "
    "connect, set up, add, or enable a specific integration they already named "
    "(for example 'can you configure sentry?' or 'connect datadog'), do NOT just "
    "tell them the command to type and do NOT talk about 'changing runtime state'. "
    "Launch it for them by returning an action plan: "
    '`{"action":"run_interactive","command":"/integrations setup <service>"}` '
    "using the service they named (for an MCP server use "
    '`{"action":"run_interactive","command":"/mcp connect <server>"}`). The '
    "interactive wizard then prompts them for the credentials that integration "
    "needs. This applies to any integration; never hardcode advice to one vendor."
)

__all__ = [
    "ACTION_RULE",
    "CLI_ASSISTANT_MARKDOWN_RULE",
    "INTERACTIVE_SHELL_TERMINOLOGY_RULE",
    "SETUP_GUIDANCE_RULE",
    "SOURCE_SCOPED_INVESTIGATION_RULE",
]
