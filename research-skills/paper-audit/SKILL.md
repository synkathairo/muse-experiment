---
name: "paper-audit"
description: "Audit a paper's claims against its public codebase: do the code and the paper agree? Use when the user asks to check, audit, or verify a paper against its implementation."
---

# Paper Audit

## Purpose

Compare what a paper claims with what its public code actually does, and report mismatches with severity.

## Workflow

1. **Identify.** Get the paper (via the `paper-research` skill) and its public code repo (paper page, arXiv listing, or author links). If no public code exists, stop: report "no code to audit" rather than guessing.
2. **Extract claims.** From the paper, list the concrete, checkable claims: hyperparameters, dataset splits, architectures, evaluation protocols, reported numbers. Vague claims ("state of the art") are not auditable — note them as such and move on.
3. **Read the code.** Inspect the repo: README, training scripts, config files, data loaders, eval code. Focus on the claims from step 2.
4. **Grade mismatches** by severity:
   - **Critical:** reported numbers unreproducible from the code as given (different hyperparams, different eval, missing steps).
   - **Major:** code exists but defaults differ from the paper, or key details are only in code comments.
   - **Minor:** naming/doc drift that doesn't affect results.
5. **Report.** A table of claim → code reality → severity, plus a verdict: *reproducible from code*, *reproducible with fixes*, or *not reproducible from public code*.

## Output Contract

- Every mismatch cites the paper location (section/page) and the code location (file, line or function).
- No speculation about author intent. Report what the code does, not why.
- The verdict line comes last and is the only summary allowed to be terse.

## Operating Rules

1. Fetch the repo at a pinned commit when possible; note the commit in the report.
2. Do not execute the code unless the user explicitly asks for a replication run — auditing is reading, not running.
3. If the repo is huge, audit the paths relevant to the checkable claims first and say what was left unread.
