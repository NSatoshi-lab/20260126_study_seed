---
name: zotero-cite
description: Zoteroローカルブリッジを使った文献検索・OA PDF取得・OneDrive保存（linked_file）・Zotero登録・citekey解決・refs/searchログ作成・refs.md要約更新の定型処理。paper.md/paper_en.mdへの引用追加や文献検索/登録が必要なときに使用する。
---

# Zotero Cite

## Goal

Automate OA-only literature retrieval and citation insertion via Zotero local bridge while recording refs/search logs and refs.md summaries.

## Preconditions

- Keep Zotero running with codex-zotero-bridge installed and "他のアプリケーションからの通信を許可" enabled.
- Keep Better BibTeX auto-export (Keep updated) writing `OneDrive/ZoteroLibrary/20260126_study_seed.bib`.
- Keep attachments as `linked_file` under `OneDrive/ZoteroLibrary/`; do not use Zotero File Storage.
- Set env vars via `.env` (do not commit): `ZOTERO_BRIDGE_TOKEN`, `ZOTERO_COLLECTION_NAME`, `ZOTERO_ONE_DRIVE_ROOT` (or OneDrive env vars).
- Recommended (to keep this study separate from the base study): set `ZOTERO_PDF_DIR_OA` to a project-scoped folder, e.g. `OneDrive/ZoteroLibrary/pdf/projects/seed/oa`.
- Set `ZOTERO_BBT_BIB_PATH` to the BibTeX export for this study, e.g. `OneDrive/ZoteroLibrary/20260126_study_seed.bib`.

## Workflow

1. Verify bridge readiness with `pwsh -File src/scripts/test_zotero_bridge.ps1`.
2. Run OA search/import with `python -m src.scripts.zotero_oa_pipeline --query "..." --max-results N`.
3. Confirm a log is created under `refs/search/` (tag `YYYYMMDD_HHMMSS_<slug>.json`) and `refs.md` summary updated before editing manuscripts.
4. Insert `[@citekey]` into `paper.md` / `paper_en.md` (use `--markdown` and `--placeholder` if automating).
5. If citekeys are missing, wait for BBT sync and re-run with `--dry-run` to resolve.
6. If non-OA or PDF-unavailable items appear, keep DOI+URL+search date+DB+query in `refs.md` (pipeline handles this; verify).

## Outputs

- `refs/search/<tag>.json` search log.
- `refs.md` summary entry with a log link.
- OA PDFs under `OneDrive/ZoteroLibrary/pdf/projects/seed/oa/<year>/` (recommended).
- Zotero items with `linked_file` attachments and citekeys.

## Notes

- Use `$literature-refs` if manual adjustments to `refs.md` are required.
- Use `src/scripts/check_citekeys_in_markdown.py` to validate citekeys when needed.
- Do not commit PDFs into the repository.

## After completion

Reflect on whether AGENTS.md or this skill needs refinement; propose changes only.
