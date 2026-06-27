# Session package rules

## Human summary

`interactive_shell/session/` is the single, canonical home for everything about a
REPL session: the in-memory domain object, the runtime context bundle assembled
for the controller, and all session persistence. Before this package existed,
session state lived in `runtime/core/session.py` + `runtime/core/context.py`
while persistence lived in `harness/state/sessions/`. Those two homes were
merged here.

Layers (mirrors the layered "harness/session" pattern):

- `state.py` — `ReplSession`, the in-memory session domain object. It delegates
  every write to an injected `SessionStorage`; it never touches the filesystem
  directly.
- `context.py` — `ReplRuntimeContext` plus `create_repl_runtime_context` /
  `prepare_repl_session`: the validated runtime bundle handed to the controller.
- `types.py` — the protocol contracts: `SessionPersistenceSource` (what storage
  needs from a session), `SessionStorage` (per-session writes), `SessionRepo`
  (cross-session queries), and the shared `CHAT_KINDS` constant.
- `paths.py` — the single owner of the on-disk layout (`~/.opensre/sessions/`),
  path resolution, and display-name derivation.
- `storage/` — `SessionStorage` backends: `jsonl.py` (`JsonlSessionStorage`,
  production) and `memory.py` (`InMemorySessionStorage`, tests).
- `repo.py` — `JsonlSessionRepo`, the cross-session read/query implementation.

## Architectural intent (locked)

- Keep the storage/repo split. Per-session writes (open/append/flush/reopen)
  belong on `SessionStorage`. Cross-session reads (`load_recent`,
  `load_session`, `count_prefix_matches`, investigation history/lookup) belong
  on `SessionRepo`. Do not merge them back into one monolithic `SessionStore`.
- `ReplSession` takes its storage by dependency injection
  (`ReplSession(storage=...)`, defaulting to `JsonlSessionStorage`). Code that
  persists session data must go through `session.storage` / a repo instance,
  not a module-level singleton baked into call sites.
- `paths.py` is the only module that resolves `OPENSRE_HOME_DIR / "sessions"`.
  Both the JSONL storage and repo go through it, so tests patch exactly one seam
  (`interactive_shell.session.paths.sessions_dir`).
- This package must not import from `interactive_shell.runtime.__init__`. The
  one allowed runtime dependency is `runtime.core.state` (used by `context.py`).
  `runtime/__init__.py` lazily re-exports this package's surface to avoid the
  resulting cycle — keep the dependency direction one-way (`runtime → session`).

## Test seam policy

- Patch `interactive_shell.session.paths.sessions_dir` (or set
  `config.constants.OPENSRE_HOME_DIR` to a temp dir) for filesystem isolation.
- Prefer `InMemorySessionStorage` for storage-behavior tests that do not need to
  assert on-disk JSONL format.
- Import canonical names from `interactive_shell.session`. Do not reintroduce a
  `SessionStore` symbol in source; it only survives in a few tests as a thin
  alias for an instance/facade and should not grow new call sites.
