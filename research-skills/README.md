# research-skills

Five research workflow skills for an AI assistant: search, read, and question academic papers; produce literature reviews; audit papers against their code; watch a topic over time; plan gated replications.

They live here as a versioned project. The working copies the assistant actually loads sit in `~/workspace/skills/` on its machine — this directory is the published source of truth.

## Attribution

These skills were **adapted from design patterns in [Feynman](https://github.com/companion-inc/feynman)** — the open source AI research agent by Companion, Inc. (© 2026 Companion, Inc., [MIT License](https://github.com/companion-inc/feynman/blob/main/LICENSE)).

They are original rewrites in this workspace's skill format, not verbatim copies. The ideas carried over: Feynman's alpha-research loop (paper search, PDF Q&A, code reading, annotations), the verifier pass and provenance tracking from its literature-review workflow, the paper-vs-codebase audit, the research watch baseline, and the gated replication plan. Feynman's skills target its own runtime and alphaXiv login; these versions run on open, no-auth APIs instead (arXiv, OpenAlex, Semantic Scholar, IACR ePrint RSS) so they work without installing anything.

Nothing here was copied from Feynman's source tree — but the debt is real, and this note is to pay it.

## Skills

- `paper-research/` — find papers, fetch full text, ask questions grounded in it, inspect paper code repos, keep annotations.
- `literature-review/` — consensus / disagreements / open questions, with a mandatory citation-verifier pass and a provenance log.
- `paper-audit/` — do a paper's claims match what its public code actually does? Severity-graded report.
- `research-watch/` — baseline a topic, then scheduled checks that report only what's new.
- `replication/` — gated replication plans: no execution before the environment is chosen, no code runs before the dry-run is approved.
