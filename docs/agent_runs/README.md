Agent Runs (Plan / Execute / Review)

Each run is a self-contained folder:

docs/agent_runs/<run_id>/

Recommended naming convention:

YYYYMMDD_short_topic
# e.g. 20260210_retrieval_matrix
📁 Required Files per Run+
File	Phase	Purpose
PLAN.md	Plan	Defines objective, approach, commands, and expected outputs
RUNLOG.md	Execute	Append-only execution log of commands, outputs, and observations
REVIEW.md	Review	Final evaluation checklist (pass/fail + notes)

All templates are sourced from:

../templates/agent_plan.template.md
../templates/agent_runlog.template.md
../templates/agent_review.template.md
🔄 Workflow Overview

Refer to the full system workflow:

../multi_agent_workflow.md

Standard flow:

Plan → define scope, steps, success criteria
Execute → run commands and append results to RUNLOG.md
Review → validate outcomes against PLAN.md checklist
🧾 Git & Traceability Policy
Default policy: commit all three files:
PLAN.md
RUNLOG.md
REVIEW.md

This ensures full auditability of:

decisions
execution history
final evaluation
If logs become excessively large or noisy:
prefer redaction of sensitive data
only as a last resort, consider .gitignore rules (team-specific)
All large outputs (datasets, artifacts, models) must be stored under:
data/

and referenced explicitly in PLAN.md.

🚀 Starting a New Run

Create directory:

docs/agent_runs/<run_id>/
Copy templates into:
PLAN.md
RUNLOG.md
REVIEW.md
Ensure headers inside each file are updated to <run_id>.
Write PLAN.md first before any execution
Proceed:
Execute → append to RUNLOG.md
Review → finalize REVIEW.md