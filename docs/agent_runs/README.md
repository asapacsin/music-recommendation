# Agent runs (Plan / Execute / Review)

Each **run** is a folder:

`docs/agent_runs/<run_id>/`

Recommended naming: `YYYYMMDD_short_topic` (example: `20260210_retrieval_matrix`).

## Files in a run

| File | Phase | Purpose |
|------|--------|---------|
| `PLAN.md` | Plan | Goal, commands, expected outputs — from [../templates/agent_plan.template.md](../templates/agent_plan.template.md) |
| `RUNLOG.md` | Execute | Append-only log of commands and results — from [../templates/agent_runlog.template.md](../templates/agent_runlog.template.md) |
| `REVIEW.md` | Review | Pass/fail checklist — from [../templates/agent_review.template.md](../templates/agent_review.template.md) |

## Workflow overview

See [../multi_agent_workflow.md](../multi_agent_workflow.md).

## Git and traceability

- **Default:** commit `PLAN.md`, `RUNLOG.md`, and `REVIEW.md` for thesis auditability (commands and verdicts stay in history).
- **Optional:** if logs become huge or contain machine-specific noise, add a pattern to `.gitignore` (e.g. per-team policy). Prefer redacting secrets over dropping the whole RUNLOG.

Large binary or generated **data** outputs stay under `data/` as elsewhere in the repo; PLAN should reference those paths explicitly.

## Starting a new run

1. Create `docs/agent_runs/<run_id>/`.
2. Copy the three templates into `PLAN.md`, `RUNLOG.md`, `REVIEW.md` (replace `<run_id>` in headers).
3. Fill PLAN first; Execute; then Review.

This folder may contain only `README.md` until you add run subfolders; that is expected.
