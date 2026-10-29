# Development workflow

The shared agent instructions are [AGENTS.md](AGENTS.md). Human contributors and
agents use the same code boundaries, validation rules, and release procedures.

## 1. Orient and define the change

- Inspect `git status --short` and the relevant diff; preserve work already present.
- Read the current [handoff](doc/README.md), [architecture](doc/ARCHITECTURE.md), and
  source/tests for the behavior being changed. Use historical plans only as context.
- State the observable result and affected boundaries. For a prediction bug, retain
  the original category, description, source ancestry, language, and version.
- Continue within authorized scope. An ordinary fix does not require a separate
  spec document or approval round; honor explicit user constraints on execution.

## 2. Implement within the existing design

| Change | Ownership and checks to consider |
|---|---|
| HTTP input/output | `api/schemas.py`, `api/app.py`; type/bounds/unknown-field rejection and canonical response contract |
| Scheduling | `api/service.py`; queue limits, deadlines, readiness, thread ownership, and shutdown |
| Query interpretation | `query.py`, `matcher.py`; placeholder, hierarchy, ambiguity, audience, and spelling regressions |
| Retrieval/ranking | `retrieval.py`, `storage.py`, `encoder.py`, `reranker.py`; scoped candidates, deterministic ranks, and unchanged acceptance rules |
| Decisions/evaluation | `decision.py`, `datasets.py`, `calibration.py`; draft isolation, reviewed evidence, and holdout reuse |
| Release integrity | `artifacts.py`, `bundles.py`; manifest/model checks, staging, locks, activation, and rollback |
| Mutable persistence | `runtime.py`, `migrations.py`; bound parameters, transactions, compatibility, and backup/restore |

Paths in the table are relative to `src/category_matcher/`. Review
[SECURITY.md](doc/SECURITY.md) for any changed trust boundary. Avoid unrelated
refactors, new abstraction layers without a use case, and unrequested dependencies.

## 3. Verify proportionately

Run from the repository root when execution is authorized:

```bash
uv run ruff check path/to/changed_file.py
uv run ruff format --check path/to/changed_file.py
uv run pytest tests/test_relevant_behavior.py
```

Use the actual affected paths. Broaden checks when shared behavior or unresolved
failures justify it. `uv sync --frozen` restores the locked environment when needed;
it may require downloads on a new installation. Never silently update the lockfile
while checking an unrelated change.

- Documentation-only changes need local-link/anchor checks, valid examples, and an
  updated file reference; they do not need inference or artifact rebuilding.
- Tests should verify outcomes and failure boundaries, not mirror implementation.
  New helpers do not justify exhaustive tests for reversible trivial edits.
- If the user defers tests, record that explicitly and stop at permitted static
  inspection. Do not claim an unexecuted check passed.
- Record baseline failures separately. Do not format a fingerprinted matching file
  just to remove an unrelated lint finding; that can invalidate the active bundle.

## 4. Decide release impact

Use the [change/rebuild matrix](doc/OPERATIONS.md#change-rebuild-and-activate).
API-only changes normally need a restart; matching/data/model identity changes need
building and activation; documentation changes need neither. Runtime schema changes
append migrations and retain previous statements/checksums.

Prepare the concrete change and evidence before any required external approval.
Within the task's authorization, use staged builds and the documented API-stop /
activate / restart sequence. Preserve existing state and avoid starting duplicate
servers. Deployment, publishing, Git writes, and destructive data removal require
appropriate authorization; none is implied by reading this document.

## 5. Maintain the handoff

Update [current documentation](doc/README.md), [decisions](doc/DECISIONS.md),
[limitations/roadmap](doc/LIMITATIONS_AND_ROADMAP.md), and
[the file map](doc/FILE_REFERENCE.md) when the change affects them. Keep generated
reports out of Git. Preserve qualification evidence and pending reviews.

Report what changed, checks actually run or skipped, any baseline failures, and
required operational action. Keep the report short; do not create a new findings
file for every task.

## Agent entry points

| File | Role |
|---|---|
| `AGENTS.md` | Codex reads this file directly; it is also the shared source for the other agents |
| `CLAUDE.md` | Imports the shared instructions for Claude Code |
| `GEMINI.md` | Imports the shared instructions for Gemini CLI |

Start the agent in the RAG repository. If the tool does not auto-load a recognized
instruction file, explicitly ask it to read `AGENTS.md` first. Confirm that the
shared instructions are present in the tool's context after setup or an upgrade.
These files guide behavior; they do not sandbox execution or enforce CI controls.
Codex discovery follows the [official OpenAI documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
The import syntax follows the official
[Claude Code](https://code.claude.com/docs/en/memory) and
[Gemini CLI](https://geminicli.com/docs/cli/gemini-md/) documentation.
