from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager


DEFAULT_SLEEP_S = 3.0


def _now_iso8601_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify_ascii(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "run"


def _build_tag(slug: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_slugify_ascii(slug)}"


def _http_get(url: str, *, timeout_s: int = 60, retries: int = 4, backoff_s: float = 8.0) -> str:
    headers = {
        "User-Agent": "aomori_survey/0.1 (+cinii-screening; contact: local)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.8,en;q=0.7",
        "Connection": "close",
    }
    request = urllib.request.Request(url, headers=headers, method="GET")
    last_error: Exception | None = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as res:
                raw = res.read()
            return raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:
                time.sleep(backoff_s * (2**i))
                continue
            raise
        except urllib.error.URLError as e:
            last_error = e
            time.sleep(backoff_s * (2**i))
            continue
    raise RuntimeError(f"Failed to fetch after retries: {url}\n{last_error}")


def _set_japanese_font() -> str | None:
    candidates = [
        "Yu Gothic",
        "YuGothic",
        "Meiryo",
        "MS Gothic",
        "MS PGothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None


def _latest_matching(path: Path, pattern: str) -> Path | None:
    hits = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


@dataclass(frozen=True)
class CiNiiRecord:
    crid: str
    url: str
    title: str | None
    journal: str | None
    authors: list[str]
    doi: str | None
    publisher: str | None
    firstpage: str | None
    lastpage: str | None


def _extract_citation_meta(html: str) -> dict[str, list[str]]:
    meta: dict[str, list[str]] = {}
    for m in re.finditer(r'<meta\s+name="(citation_[^"]+)"\s+content="([^"]*)"\s*/?>', html):
        meta.setdefault(m.group(1), []).append(m.group(2))
    return meta


def _pick_first(meta: dict[str, list[str]], key: str) -> str | None:
    vals = meta.get(key) or []
    for v in vals:
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _screen_keywords(text: str) -> dict[str, int]:
    keywords = [
        "普及",
        "設置",
        "導入",
        "普及率",
        "設置率",
        "市場",
        "販売",
        "シェア",
        "アンケート",
        "調査",
        "住宅設備",
        "換気",
        "乾燥",
        "暖房",
    ]
    out: dict[str, int] = {}
    for kw in keywords:
        out[kw] = text.count(kw)
    return out


def _fetch_record(url: str) -> CiNiiRecord:
    html = _http_get(url)
    meta = _extract_citation_meta(html)
    title = _pick_first(meta, "citation_title")
    journal = _pick_first(meta, "citation_journal_title")
    doi = _pick_first(meta, "citation_doi")
    publisher = _pick_first(meta, "citation_publisher")
    authors = meta.get("citation_author") or []
    firstpage = _pick_first(meta, "citation_firstpage")
    lastpage = _pick_first(meta, "citation_lastpage")

    m = re.search(r"/crid/([0-9]+)", url)
    crid = m.group(1) if m else url.rsplit("/", 1)[-1]

    return CiNiiRecord(
        crid=crid,
        url=url,
        title=title,
        journal=journal,
        authors=[a for a in authors if a.strip()],
        doi=doi.strip() if doi else None,
        publisher=publisher,
        firstpage=firstpage,
        lastpage=lastpage,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_md(path: Path, *, meta: dict[str, Any], records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# CiNiiヒット（周辺資料）の中身確認（スクリーニング）")
    lines.append("")
    lines.append(f"- 作成日時: {meta['created_at_local']}")
    lines.append(f"- 入力ログ: `{meta['source_log']}`")
    lines.append(f"- 記述ポリシー: `docs/rules/statistical_reporting_policy.md`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 目的")
    lines.append("")
    lines.append("- 青森県の浴室暖房普及率が低いことを直接扱う資料があるか、CiNiiでヒットした周辺資料の内容（題名・媒体）を確認する。")
    lines.append("")
    lines.append("## 2. 対象（上位ヒットのCRID）")
    lines.append("")
    lines.append("| CRID | タイトル | 媒体 | DOI | 備考（機械抽出のキーワード出現） |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in records:
        doi = r.get("doi") or ""
        title = (r.get("title") or "").replace("|", " ")
        journal = (r.get("journal") or "").replace("|", " ")
        note = ", ".join([f"{k}:{v}" for k, v in (r.get("keyword_counts") or {}).items() if v])
        lines.append(f"| {r['crid']} | {title} | {journal} | {doi} | {note} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="CiNii検索ログの上位ヒットCRIDを取得し、メタ情報を抽出してスクリーニングします。")
    parser.add_argument("--source-log", type=str, default="", help="入力ログ（CiNiiのrefs/search JSON）")
    parser.add_argument("--tag", type=str, default="", help="出力tag（未指定なら自動生成）")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S, help="リクエスト間スリープ秒（既定: 3.0）")
    args = parser.parse_args()

    _set_japanese_font()

    repo_root = Path(__file__).resolve().parents[2]
    source_log = Path(args.source_log).expanduser() if args.source_log else None
    if source_log is None:
        source_log = _latest_matching(repo_root / "refs" / "search", "*_aomori_bathroom_heater_literature_cinii.json")
    if source_log is None or not source_log.exists():
        raise FileNotFoundError("CiNiiの入力ログが見つかりません。--source-log で指定してください。")

    payload = _load_json(source_log)
    queries = payload.get("search", {}).get("filter", {}).get("queries", [])
    urls: list[str] = []
    for q in queries:
        for item in (q.get("top_results") or []):
            u = str(item.get("url") or "").strip()
            if re.search(r"/crid/\d+$", u):
                urls.append(u)

    # Preserve order; unique.
    seen: set[str] = set()
    crid_urls: list[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        crid_urls.append(u)

    tag = args.tag.strip() or _build_tag("cinii_top_hit_screening")
    out_dir = repo_root / "outputs" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for i, url in enumerate(crid_urls):
        rec = _fetch_record(url)
        text_for_kw = " ".join([rec.title or "", rec.journal or ""])
        records.append(
            {
                "crid": rec.crid,
                "url": rec.url,
                "title": rec.title,
                "journal": rec.journal,
                "authors": rec.authors,
                "doi": rec.doi,
                "publisher": rec.publisher,
                "firstpage": rec.firstpage,
                "lastpage": rec.lastpage,
                "keyword_counts": _screen_keywords(text_for_kw),
            }
        )
        if i != len(crid_urls) - 1:
            time.sleep(float(args.sleep))

    meta = {
        "created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_at": payload.get("search", {}).get("requested_at"),
        "source_log": str(source_log),
        "source_queries": [q.get("query") for q in queries],
        "generated_at": _now_iso8601_z(),
    }

    out_json = repo_root / "refs" / "search" / f"{tag}_cinii_top_hit_screening.json"
    _write_json(out_json, {"meta": meta, "records": records})

    out_md = out_dir / "cinii_top_hit_screening.md"
    _write_md(out_md, meta=meta, records=records)

    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
