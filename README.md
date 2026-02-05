# 青森調査: Repo Index (Codex Agent Navigation)

このリポジトリは、「青森県における浴室暖房の実態調査」を進めるための作業用リポジトリです。基盤repo（入浴統計）を**参照専用（read-only）**として扱い、成果物・新規解析・文献ログは本repoに集約します。

## Start Here (Most Important Paths)

- Manuscript Markdown (authoritative): `paper.md` / `paper_en.md`
- Final docx deliverables: `deliverables/`
- Evidence runs: `outputs/runs/`
- Literature log: `refs.md`, `refs/search/`
- Code entry points: `src/scripts/`
- Writing rules: `docs/rules/markdown_generation_rules.md`, `docs/rules/statistical_reporting_policy.md`
- Agent skills: `.codex/skills/`

## Base Repo (Read-Only)

Base repo: `..\入浴統計`（参照専用）

| Read from base | Purpose |
| --- | --- |
| `..\入浴統計\deliverables\` | 完成版成果物（完成稿） |
| `..\入浴統計\outputs\runs\` | 完成版成果物に直結する根拠run |
| `..\入浴統計\data\processed\` | 完成版成果物に直結するデータ |
| `..\入浴統計\refs.md` / `..\入浴統計\refs\search\` | 既存研究の文献探索ログ |

## Write Here (This Repo Only)

| Write here | Purpose |
| --- | --- |
| `paper.md` / `paper_en.md` | 原稿（mdが正） |
| `deliverables/` | 投稿用docxなど最終成果物 |
| `outputs/runs/` | 解析・感度分析のrun成果物 |
| `refs.md` / `refs/search/` | 文献探索ログ |
| `src/` | 青森調査の解析コード |

## Workflow (md → docx)

- 編集は `paper.md` / `paper_en.md` を正とする。
- 引用は `refs.md` と `refs/search/` に先に記録し、本文へ反映する。
- docx生成は `src/scripts/build_paper_docx.ps1` を使い、生成物は `outputs/runs/<tag>/` に出す。
- 投稿用の最終成果物は `deliverables/` に置く。

## Rules (Must Follow)

- Markdown rules: `docs/rules/markdown_generation_rules.md`
- Statistical reporting policy: `docs/rules/statistical_reporting_policy.md`
- 文献探索ログは `refs/search/` に保存し、`refs.md` へ要約してから本文へ反映する。
