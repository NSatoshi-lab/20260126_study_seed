from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check citekeys in Markdown against .bib exports.")
    parser.add_argument(
        "--bib",
        action="append",
        default=[],
        help="Path or glob to .bib (repeatable). Default: refs/bib/*.bib",
    )
    parser.add_argument(
        "--markdown",
        action="append",
        default=[],
        help="Markdown file path (repeatable). Default: draft.md, umin_abstract.md (if present)",
    )
    return parser.parse_args()


def expand_paths(patterns: list[str], base: Path) -> list[Path]:
    expanded: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(str((base / pattern).resolve()), recursive=True)
        expanded.extend(Path(p) for p in matches)
    return sorted({p.resolve() for p in expanded})


def load_bib_keys(paths: list[Path]) -> set[str]:
    key_re = re.compile(r"^@\w+\s*\{\s*([^,]+)\s*,", re.MULTILINE)
    keys: set[str] = set()
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        keys.update(m.group(1).strip() for m in key_re.finditer(text))
    return keys


def extract_citekeys(md_text: str) -> set[str]:
    cite_re = re.compile(r"(?<!\w)@([A-Za-z0-9_:.+-]+)")
    return {m.group(1) for m in cite_re.finditer(md_text)}


def main() -> int:
    args = parse_args()
    root = repo_root()

    bib_patterns = args.bib or ["refs/bib/*.bib"]
    bib_paths = expand_paths(bib_patterns, root)
    if not bib_paths:
        print("No .bib found. Run sync first, e.g.: pwsh -File src/scripts/sync_zotero_bib.ps1", file=sys.stderr)
        return 2

    md_candidates = args.markdown or []
    if not md_candidates:
        for default in ["draft.md", "umin_abstract.md"]:
            if (root / default).exists():
                md_candidates.append(default)

    md_paths = [Path(p).resolve() if Path(p).is_absolute() else (root / p).resolve() for p in md_candidates]
    md_paths = [p for p in md_paths if p.exists()]
    if not md_paths:
        print("No Markdown files to check.", file=sys.stderr)
        return 0

    bib_keys = load_bib_keys(bib_paths)
    used_keys: set[str] = set()
    for md_path in md_paths:
        used_keys |= extract_citekeys(md_path.read_text(encoding="utf-8", errors="replace"))

    missing = sorted(k for k in used_keys if k not in bib_keys)
    if missing:
        print("Missing citekeys:", file=sys.stderr)
        for k in missing:
            print(f"- {k}", file=sys.stderr)
        return 1

    print(f"OK: checked {len(md_paths)} Markdown files against {len(bib_paths)} bib files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
