---
name: "replication"
description: "Plan a replication check for a research claim: gated steps, explicit environment choice, nothing runs without approval. Use when the user asks to replicate, reproduce, or verify a paper's results by running code."
---

# Replication

## Purpose

Turn "replicate this paper" into a concrete, gated plan — and execute it only after the user approves each gate.

## Workflow

1. **Scope.** Identify the exact claim to replicate and the paper's artifacts (code repo, data, environment files). Use the `paper-audit` skill first when a public repo exists — don't plan a replication against un-audited code.
2. **Plan.** Write the replication plan: steps, expected outputs per step, compute needed (CPU/GPU, rough time), and the decision rule at each gate (e.g. "loss within 5% of reported value → proceed").
3. **Gate 1 — environment.** Present the environment options (local VM, user-provided machine, cloud) and the estimated cost/time. Do nothing until the user picks one.
4. **Gate 2 — dry run.** Show the exact commands that will run. Execute only after explicit approval.
5. **Execute.** Run step by step, logging outputs. If a step fails or diverges from the decision rule, stop and report — never improvise past a gate.
6. **Report.** What reproduced, what didn't, and the delta from the paper's numbers.

## Output Contract

- The plan is a standalone document saved to `~/workspace/research/replications/<slug>/PLAN.md` before anything runs.
- Execution log at `~/workspace/research/replications/<slug>/RUNLOG.md`.
- Final report: reproduced / partially reproduced / not reproduced, with numbers.

## Operating Rules

1. Nothing runs before Gate 1. No exceptions, no "quick test first".
2. State compute cost and time estimates before asking for the environment choice — approval without that information isn't informed.
3. Prefer the paper's own pinned environment (Dockerfile, requirements, lockfile). If it doesn't pin, say so in the plan — environment drift is the most common replication killer.
4. A failed replication is a result, not a failure. Report it cleanly.
