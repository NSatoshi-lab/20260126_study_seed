---
name: paper-writing
description: paper.md/paper_en.mdの執筆・更新とmd→docx生成の運用に使う。Use when editing manuscript Markdown and generating docx outputs.
---

# Paper Writing

## Goal

Maintain manuscript Markdown (`paper.md`, `paper_en.md`) as the authoritative source and generate submission-ready docx via Pandoc.

## Workflow

1. Confirm the target file(s) (`paper.md`, `paper_en.md`) and scope.
2. Review the relevant evidence runs and align with `docs/rules/statistical_reporting_policy.md`.
3. Apply `docs/rules/markdown_generation_rules.md` for formatting.
4. Write results with effect sizes and CIs; avoid causal language and p-value-only claims.
5. If new references are needed, run `$zotero-cite` to search/import and log first.
6. Update `refs.md` via `$literature-refs` when required and insert citekeys into Markdown.
7. Generate docx with `src/scripts/build_paper_docx.ps1` and place submission-ready files in `deliverables/`.

## Notes

- Keep headings unique and preserve existing structure unless the user requests a restructure.
- Markdown is authoritative; avoid editing docx directly.
- Keep Japanese and English manuscripts aligned for shared results when requested.

## After completion

Reflect on whether AGENTS.md or this skill needs refinement; propose changes only.
