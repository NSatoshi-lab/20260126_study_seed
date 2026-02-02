from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExtractResult:
    text: str
    method: str
    score: int


def _extract_pdfminer(path: Path) -> str:
    from pdfminer.high_level import extract_text

    return extract_text(str(path))


def _extract_pymupdf(path: Path) -> str:
    import fitz

    doc = fitz.open(str(path))
    try:
        return "\n".join(doc.load_page(i).get_text("text") for i in range(doc.page_count))
    finally:
        doc.close()


def _score_text(text: str) -> int:
    if not text:
        return 0
    nonspace = sum(1 for ch in text if not ch.isspace())
    cjk = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff")
    return nonspace + cjk * 5


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _contains_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _count_keyword(text: str, keyword: str) -> int:
    if not keyword:
        return 0
    if _contains_ascii(keyword):
        return text.lower().count(keyword.lower())
    return text.count(keyword)


def _extract_auto(path: Path) -> ExtractResult:
    candidates: list[ExtractResult] = []
    errors: list[str] = []
    try:
        text = _extract_pdfminer(path)
        candidates.append(ExtractResult(text=text, method="pdfminer", score=_score_text(text)))
    except Exception as exc:  # pragma: no cover
        errors.append(f"pdfminer: {exc}")
    try:
        text = _extract_pymupdf(path)
        candidates.append(ExtractResult(text=text, method="pymupdf", score=_score_text(text)))
    except Exception as exc:  # pragma: no cover
        errors.append(f"pymupdf: {exc}")
    if not candidates:
        raise RuntimeError("; ".join(errors) or "no extractors available")
    return sorted(candidates, key=lambda x: x.score, reverse=True)[0]


def _extract(path: Path, method: str) -> ExtractResult:
    if method == "pdfminer":
        text = _extract_pdfminer(path)
        return ExtractResult(text=text, method=method, score=_score_text(text))
    if method == "pymupdf":
        text = _extract_pymupdf(path)
        return ExtractResult(text=text, method=method, score=_score_text(text))
    return _extract_auto(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def _iter_pdfs(pdfs: list[str]) -> Iterable[Path]:
    for item in pdfs:
        path = Path(item)
        if path.is_dir():
            yield from sorted(path.glob("*.pdf"))
        else:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="PDFからテキスト抽出を行います。")
    parser.add_argument("--pdf", action="append", default=[], help="PDFパス（複数指定可）")
    parser.add_argument("--pdf-list", type=str, default="", help="PDFパス一覧（1行1件）")
    parser.add_argument("--out-dir", type=str, default="", help="出力ディレクトリ（未指定なら出力しない）")
    parser.add_argument("--out-file", type=str, default="", help="単一PDFの出力ファイル")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "pdfminer", "pymupdf"],
        help="抽出エンジン",
    )
    parser.add_argument("--keywords", action="append", default=[], help="キーワード（複数指定可）")
    parser.add_argument("--normalize", action="store_true", help="空白除去した上でキーワードカウント")
    parser.add_argument("--preview", type=int, default=0, help="先頭N文字を表示（0で非表示）")
    args = parser.parse_args()

    pdfs: list[str] = list(args.pdf)
    if args.pdf_list:
        pdfs.extend(Path(args.pdf_list).read_text(encoding="utf-8").splitlines())
    pdf_paths = list(_iter_pdfs([p for p in pdfs if p.strip()]))
    if not pdf_paths:
        raise SystemExit("No PDFs provided.")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    out_file = Path(args.out_file).resolve() if args.out_file else None
    if out_file and len(pdf_paths) != 1:
        raise SystemExit("--out-file is only allowed for a single PDF.")

    for path in pdf_paths:
        if not path.exists():
            print(f"[skip] {path} (not found)")
            continue
        result = _extract(path, args.method)
        text = result.text
        if out_file:
            _write_text(out_file, text)
        elif out_dir:
            stem = path.stem + ".txt"
            _write_text(out_dir / stem, text)

        preview = text[: args.preview] if args.preview > 0 else ""
        print(f"[ok] {path} method={result.method} score={result.score} chars={len(text)}")
        if preview:
            print(preview)

        if args.keywords:
            if args.normalize:
                text_for_count = _normalize_text(text)
                keywords = [_normalize_text(k) for k in args.keywords]
            else:
                text_for_count = text
                keywords = args.keywords
            for keyword in keywords:
                count = _count_keyword(text_for_count, keyword)
                print(f"  - {keyword}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
