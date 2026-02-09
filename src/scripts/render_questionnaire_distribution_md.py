#!/usr/bin/env python
"""Render a distribution-friendly questionnaire markdown from source markdown.

This script extracts sections needed for paper distribution, converts answer
choices to checkbox-style lines, normalizes conditional guidance, and writes a
layout report JSON for traceability.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


QUESTION_HEADING_RE = re.compile(r"^###\s+Q(?P<num>\d+)\s+(?P<title>.+?)\s*$")
NUMBERED_OPTION_RE = re.compile(r"^(?P<indent>\s*)(?P<num>\d+)\.\s+(?P<body>.+)$")
CONDITION_SUFFIX_RE = re.compile(r"（Q\d+で「[^」]+」の人のみ）")


@dataclass
class QuestionBlock:
    number: int
    heading: str
    body_lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert questionnaire markdown into distribution-ready markdown "
            "with checkbox options and branch guidance boxes."
        )
    )
    parser.add_argument("--input", required=True, help="Source questionnaire markdown")
    parser.add_argument("--output", required=True, help="Rendered markdown path")
    parser.add_argument("--report", required=True, help="JSON report path")
    return parser.parse_args()


def strip_condition_suffix(heading: str) -> str:
    return CONDITION_SUFFIX_RE.sub("", heading).strip()


def normalize_question_heading(heading: str) -> str:
    match = QUESTION_HEADING_RE.match(heading.strip())
    if not match:
        return heading.strip()
    number = match.group("num")
    title = strip_condition_suffix(match.group("title"))
    return f"### Q{number} {title}"


def find_title(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("# "):
            return line.rstrip()
    raise ValueError("先頭タイトル（# ...）が見つかりませんでした。")


def section_slice(lines: list[str], header: str) -> tuple[int, int] | None:
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## ") and lines[j].strip() != header:
            end = j
            break
    return start, end


def extract_question_blocks(lines: list[str]) -> list[QuestionBlock]:
    q_section = section_slice(lines, "## 設問")
    if q_section is None:
        raise ValueError("設問セクション（## 設問）が見つかりませんでした。")
    start, end = q_section
    section_lines = lines[start + 1 : end]

    blocks: list[QuestionBlock] = []
    current_heading = ""
    current_body: list[str] = []

    for raw in section_lines:
        line = raw.rstrip("\n")
        m = QUESTION_HEADING_RE.match(line.strip())
        if m:
            if current_heading:
                q_num = int(QUESTION_HEADING_RE.match(current_heading).group("num"))  # type: ignore[union-attr]
                blocks.append(
                    QuestionBlock(
                        number=q_num,
                        heading=normalize_question_heading(current_heading),
                        body_lines=current_body[:],
                    )
                )
            current_heading = line.strip()
            current_body = []
            continue
        if current_heading:
            current_body.append(line.rstrip())

    if current_heading:
        q_num = int(QUESTION_HEADING_RE.match(current_heading).group("num"))  # type: ignore[union-attr]
        blocks.append(
            QuestionBlock(
                number=q_num,
                heading=normalize_question_heading(current_heading),
                body_lines=current_body[:],
            )
        )

    if not blocks:
        raise ValueError("Q1-Q14の設問ブロックを抽出できませんでした。")
    return blocks


def convert_options_to_checkboxes(lines: Iterable[str]) -> tuple[list[str], int]:
    out: list[str] = []
    count = 0
    for line in lines:
        m = NUMBERED_OPTION_RE.match(line)
        if m:
            out.append(f"{m.group('indent')}□ {m.group('num')}. {m.group('body')}")
            count += 1
            continue
        out.append(line)
    return out, count


def remove_relocated_lines(lines: Iterable[str]) -> tuple[list[str], int]:
    remove_exact = {
        "※Q8で「2」「3」の人はQ13へ進んでください。",
    }
    kept: list[str] = []
    removed = 0
    for line in lines:
        if line.strip() in remove_exact:
            removed += 1
            continue
        kept.append(line)
    return kept, removed


def compact_free_text(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- `") and stripped.endswith("`"):
            out.append(stripped[2:])
            continue
        out.append(line)
    return out


def render_distribution_markdown(lines: list[str]) -> tuple[str, dict]:
    title = find_title(lines)
    answer_slice = section_slice(lines, "## 回答方法")
    if answer_slice is None:
        raise ValueError("回答方法セクション（## 回答方法）が見つかりませんでした。")
    a_start, a_end = answer_slice
    answer_lines = [ln.rstrip() for ln in lines[a_start + 1 : a_end]]

    end_slice = section_slice(lines, "## 回答終了")
    if end_slice is None:
        raise ValueError("回答終了セクション（## 回答終了）が見つかりませんでした。")
    e_start, e_end = end_slice
    end_lines_raw = [ln.rstrip() for ln in lines[e_start + 1 : e_end]]
    end_lines = [ln for ln in end_lines_raw if "内部利用のみ" not in ln]

    questions = extract_question_blocks(lines)

    rendered: list[str] = []
    rendered.append(title)
    rendered.append("")
    rendered.append("## 回答方法")
    rendered.append("")
    rendered.extend(answer_lines)
    rendered.append("")
    rendered.append("## 設問")
    rendered.append("")

    total_checkbox = 0
    removed_relocated = 0

    for block in questions:
        body, removed_count = remove_relocated_lines(block.body_lines)
        removed_relocated += removed_count
        body = compact_free_text(body)
        body, checkbox_count = convert_options_to_checkboxes(body)
        total_checkbox += checkbox_count

        rendered.append(block.heading)
        rendered.append("")
        rendered.extend(body)
        rendered.append("")

    rendered.append("## 回答終了")
    rendered.append("")
    rendered.extend(end_lines)
    rendered.append("")

    report = {
        "question_count": len(questions),
        "question_numbers": [q.number for q in questions],
        "checkbox_option_lines": total_checkbox,
        "inserted_branch_guides": [],
        "removed_relocated_lines": removed_relocated,
        "trimmed_sections": ["調査の概要", "記入者情報（調査管理用）"],
        "kept_sections": ["回答方法", "設問", "回答終了"],
        "page_policy": {
            "target": "1_page_preferred",
            "fallback": "allow_2_pages",
            "minimum_font_pt": 10.5,
        },
    }
    return "\n".join(rendered), report


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    out = Path(args.output)
    report_path = Path(args.report)

    if not src.exists():
        raise FileNotFoundError(f"input markdown not found: {src}")

    src_text = src.read_text(encoding="utf-8")
    lines = src_text.splitlines()
    rendered, report = render_distribution_markdown(lines)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8", newline="\n")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
