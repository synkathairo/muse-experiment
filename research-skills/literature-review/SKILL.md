---
name: "literature-review"
description: "Produce a literature review on a research topic: paper search, consensus synthesis, open questions. Use when the user asks for a lit review, paper survey, state of the art, or academic landscape summary."
---

# Literature Review

## Purpose

Turn a research topic into a structured, citation-grounded literature review: what the field agrees on, where it disagrees, and what's open.

## Workflow

1. **Scope.** Confirm the topic and any bounds (time window, venues, crypto vs general ML). If unbounded, default to the last 5 years and say so.
2. **Gather.** Use the `paper-research` skill to collect 10–25 candidate papers. Deduplicate by DOI/arXiv id. Note citation counts and venues as a rough influence signal, not a verdict.
3. **Draft.** Write the review with these sections: *Consensus* (claims repeated across sources), *Disagreements* (direct contradictions, named), *Open questions*, *Key papers* (annotated list).
4. **Review pass.** Self-critique: flag any claim supported by only one paper, any missing sub-topic the gather step should have caught, any citation that doesn't actually support its sentence.
5. **Verifier pass.** Check every citation: does the ID resolve, does the cited claim appear in the paper's abstract or fetched text, are all links live. Fix or drop failures.
6. **Deliver.** Write the review as markdown. Include a provenance footer listing every paper consulted with its ID and where it was found.

## Output Contract

- Headings: Consensus, Disagreements, Open questions, Key papers, Provenance.
- Every factual claim carries an inline citation to a paper ID (arXiv/DOI/ePrint). No uncited factual claims.
- A companion `*_provenance.md` alongside the review records the search queries, the papers screened out and why, and the verifier's pass/fail log.
- Save final output to `~/workspace/your_files/` (or `~/workspace/research/`) and attach it.

## Operating Rules

1. One paper = one vote is wrong; weight by independent replication, not citation count alone.
2. Never let the review lean on a single paper for a section — flag thin coverage instead of padding it.
3. If the evidence can't support a requested section, say so rather than filling it with hedged filler.
4. The verifier pass is mandatory, not optional. A review that ships without it is a draft.
