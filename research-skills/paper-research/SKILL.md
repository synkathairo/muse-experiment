---
name: "paper-research"
description: "Search, read, and ask questions about academic papers via open APIs (arXiv, OpenAlex, Semantic Scholar, IACR ePrint RSS). Use when the user asks about a paper, wants research on a topic, needs to inspect a paper's code repo, or wants to save/find annotations on papers."
---

# Paper Research

Research papers using open, no-auth APIs. No install, no login, no API keys.

## Purpose

Find papers on a topic, fetch a paper's full text, answer questions grounded in that text, inspect the paper's public code, and keep persistent per-paper annotations.

## Tooling

All endpoints are public. Prefer them over web search for paper work; fall back to `browser.search` only when the APIs miss.

- **arXiv search:** `GET https://export.arxiv.org/api/query` with `search_query`, `start`, `max_results`, `sortBy`, `sortOrder`. Atom XML. Example: `?search_query=all:sparse+autoencoders&max_results=20&sortBy=submitted&sortOrder=descending`
- **Paper detail (OpenAlex):** `GET https://api.openalex.org/works/W...` or search `GET https://api.openalex.org/works?search=<q>&per-page=25`. Gives citations, referenced works, OA URL, topics.
- **Semantic Scholar:** `GET https://api.semanticscholar.org/graph/v1/paper/search?query=<q>&fields=title,abstract,url,year,citationCount,openAccessPdf,externalIds&limit=20`. Paper detail: `/graph/v1/paper/arXiv:<id>` or `/paper/DOI:<doi>`.
- **IACR ePrint (crypto):** `GET https://eprint.iacr.org/rss` — recent ~100 papers, RSS XML. Paper page: `https://eprint.iacr.org/<year>/<number>`. Full text at the same URL with `.pdf`.
- **Full text:** download the PDF with the browser tools and read it (converted to markdown by the reader). For arXiv source when available: `https://export.arxiv.org/e-print/<id>`.

## Auth

None. These are public APIs. Respect rate limits: arXiv asks for ≤3s between requests in bulk pulls; Semantic Scholar unauthenticated allows ~100 requests / 5 min.

## Operating Rules

1. Identify the paper by its canonical ID (arXiv id, DOI, OpenAlex id, or ePrint `YYYY/NNN`) before fetching full text.
2. Answer paper questions only from fetched text. If the PDF is unreadable, say so and summarize the abstract instead — never reconstruct "likely" methods from memory.
3. Annotations: save to `~/workspace/research/annotations.md` as `## <id> — <title>` with date-stamped notes. Read it first when asked "what do I have on X".
4. When a paper links a public code repo, inspect the repo (file listing, README, key source files) rather than paraphrasing the paper's claims about the code.
5. Prefer OA full text; note when a paper is paywalled and work from its preprint/abs only.
