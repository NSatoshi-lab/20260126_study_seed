---
name: literature-refs
description: 文献探索ログとrefs.mdの更新に使う。Use when updating literature search logs, citations, or evidence notes in refs.md.
---

# Literature Refs

## Goal

Maintain `refs.md` as the authoritative log of literature searches and evidence notes.

## Workflow

1. Log each search under `refs/search/` (one search, one JSON file).
2. Summarize the search in `refs.md` with date, DB, query, and a link to the log file.
3. Record zero-hit checks before claiming "no evidence" in the manuscript.
4. Add new references with consistent R numbering and brief relevance notes.
5. Keep PDFs in OneDrive (`linked_file` via Zotero); do not add PDFs under `refs/`.
6. For non-OA or PDF-unavailable items, keep DOI+URL+search date+DB+query in `refs.md`.
7. Keep manuscript citations aligned with the R numbers in `paper.md` / `paper_en.md`.

## Notes

- Preserve existing sections and numbering in `refs.md`.
- Link to the OneDrive path when needed; do not commit PDFs into the repo.

## After completion

Reflect on whether AGENTS.md or this skill needs refinement; propose changes only.
