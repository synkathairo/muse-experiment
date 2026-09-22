---
name: "research-watch"
description: "Track a research topic over time: establish a baseline, then check for new developments on a schedule. Use when the user wants to follow a topic, get updates on new papers, or watch a field."
---

# Research Watch

## Purpose

Maintain an ongoing watch on a research topic: a written baseline of what's known, plus scheduled checks that report only what's new since the last check.

## Workflow

1. **Baseline.** Use the `paper-research` skill to survey the topic and write `~/workspace/research/watch/<slug>/BASELINE.md`: the current state of the field, key papers, and the open questions. Record the baseline date and the newest paper seen.
2. **Schedule.** Create a cron job (with the user's approval of the cadence) that re-runs the search and diffs against the baseline plus prior check-ins.
3. **Check.** On each run: fetch recent papers (arXiv sorted by submission date, ePrint RSS for crypto, OpenAlex `from-publication-date` filter). Anything older than the last check date is noise — ignore it.
4. **Report.** Only genuinely new papers or developments go to the user. Each check-in appends to `~/workspace/research/watch/<slug>/LOG.md` and stays silent in chat unless there's something worth interrupting for.

## Output Contract

- `BASELINE.md` and `LOG.md` per watched topic under `~/workspace/research/watch/<slug>/`.
- Check-in reports are short: new papers with one-line summaries and why they matter relative to the baseline. No re-summarizing the field each time.
- Silence is a feature: "nothing new" stays in the log, not in the user's chat.

## Operating Rules

1. A watch needs a topic and a cadence from the user. Weekly is the default; suggest daily only for fast-moving preprints the user actively follows.
2. The first check after baselining must not re-report baseline papers — diff strictly on date.
3. If a watched topic goes quiet for a month, offer to widen the query or retire the watch.
