from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SLEEP_S = 2.0


def _now_iso8601_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify_ascii(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "search"


def _build_tag(slug: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_slugify_ascii(slug)}"


def _http_get(url: str, *, timeout_s: int = 60, retries: int = 4, backoff_s: float = 8.0) -> str:
    headers = {
        "User-Agent": "aomori_survey/0.1 (+offline-literature-log; contact: local)",
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


@dataclass(frozen=True)
class SearchQuery:
    query: str
    url: str
    hit_count: int | None
    error: str | None
    top_results: list[dict[str, Any]]


def _jstage_url(q: str) -> str:
    return "https://www.jstage.jst.go.jp/result/global/-char/ja?globalSearchKey=" + urllib.parse.quote(q)


def _jstage_parse(html: str) -> tuple[int | None, list[dict[str, Any]]]:
    m = re.search(r"([0-9,]+)件中\s*([0-9,]+)-([0-9,]+)の結果を表示しています", html)
    total = int(m.group(1).replace(",", "")) if m else None
    items: list[dict[str, Any]] = []
    for m2 in re.finditer(
        r'<a href="(https://www\.jstage\.jst\.go\.jp/article/[^"]+/_article/-char/ja)"[^>]*title="([^"]+)"',
        html,
    ):
        url = m2.group(1)
        title = m2.group(2).strip()
        items.append({"title": title, "url": url})
        if len(items) >= 5:
            break
    if total is None and not items:
        # J-STAGE includes "not found" text as a hidden template even when results exist,
        # so avoid keyword-based detection and fallback to link presence.
        return 0, []
    return total, items


def _cinii_url(q: str) -> str:
    return "https://cir.nii.ac.jp/all?q=" + urllib.parse.quote(q)


def _cinii_parse(html: str) -> tuple[int | None, list[dict[str, Any]]]:
    m = re.search(r"検索結果</span>\s*([0-9,]+)\s*件", html)
    total = int(m.group(1).replace(",", "")) if m else None
    items: list[dict[str, Any]] = []
    for m2 in re.finditer(
        r'<dt class="item_mainTitle item_title">\s*<a[^>]+href="(/crid/[0-9]+)"[^>]*>(.*?)</a>',
        html,
        flags=re.MULTILINE | re.DOTALL,
    ):
        href = m2.group(1)
        title = re.sub(r"\s+", " ", re.sub(r"<.*?>", " ", m2.group(2))).strip()
        items.append({"title": title, "url": "https://cir.nii.ac.jp" + href})
        if len(items) >= 5:
            break
    return total, items


def _pubmed_url(q: str) -> str:
    return "https://pubmed.ncbi.nlm.nih.gov/?term=" + urllib.parse.quote(q)


def _pubmed_parse(html: str) -> tuple[int | None, list[dict[str, Any]]]:
    m = re.search(r'name="log_resultcount"\s+content="([0-9,]+)"', html)
    total = int(m.group(1).replace(",", "")) if m else None
    items: list[dict[str, Any]] = []
    for m2 in re.finditer(r'<a\s+class="docsum-title"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
        href = m2.group(1)
        title = re.sub(r"\s+", " ", re.sub(r"<.*?>", " ", m2.group(2))).strip()
        if not title:
            continue
        items.append({"title": title, "url": "https://pubmed.ncbi.nlm.nih.gov" + href})
        if len(items) >= 5:
            break
    return total, items


def _run_db(
    *,
    db: str,
    queries: list[str],
    sleep_s: float,
    build_url,
    parse_html,
) -> list[SearchQuery]:
    out: list[SearchQuery] = []
    for i, q in enumerate(queries):
        url = build_url(q)
        try:
            html = _http_get(url)
            hit_count, top = parse_html(html)
            if hit_count is None:
                # Occasional interstitial pages can omit the count; wait and retry once.
                time.sleep(max(float(sleep_s), 8.0))
                html_retry = _http_get(url)
                hit_count_retry, top_retry = parse_html(html_retry)
                if hit_count_retry is not None:
                    hit_count, top = hit_count_retry, top_retry
            err = None
            if hit_count is None:
                err = "parse_failed: hit_count not found"
            out.append(SearchQuery(query=q, url=url, hit_count=hit_count, error=err, top_results=top))
        except Exception as e:
            out.append(SearchQuery(query=q, url=url, hit_count=None, error=str(e), top_results=[]))
        if i != len(queries) - 1:
            time.sleep(sleep_s)
    return out


def _write_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="J-STAGE / CiNii / PubMedで青森県×浴室暖房普及に関する検索ログを作成します。")
    parser.add_argument("--tag", type=str, default="", help="ログタグ（未指定なら自動生成）")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_S, help="リクエスト間スリープ秒（既定: 2.0）")
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "jstage", "cinii", "pubmed"],
        help="実行するDB（既定: all）",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    tag = args.tag.strip() or _build_tag("aomori_bathroom_heater_literature")

    # Query design: Aomori-specific + broader equipment/adoption terms (Japanese + English).
    q_jp = [
        "青森 浴室 暖房 乾燥機 設置率",
        "青森 浴室 暖房 普及率",
        "青森 風呂 暖房 設置率",
        "浴室 暖房 乾燥機 設置率",
        "浴室 暖房 乾燥機",
        "住宅 土地統計調査 浴室 暖房 乾燥機",
    ]
    q_en = [
        "Aomori bathroom heating",
        "Aomori bathroom heater",
        "bathroom heater dryer Japan",
        "bathroom heating Japan",
    ]

    requested_at = _now_iso8601_z()

    logs: list[tuple[str, str, list[SearchQuery]]] = []
    if args.only in ("all", "jstage"):
        logs.append(
            ("jstage", "J-STAGE", _run_db(db="J-STAGE", queries=q_jp, sleep_s=float(args.sleep), build_url=_jstage_url, parse_html=_jstage_parse))
        )
    if args.only in ("all", "cinii"):
        logs.append(
            (
                "cinii",
                "CiNii Research",
                _run_db(db="CiNii Research", queries=q_jp, sleep_s=float(args.sleep), build_url=_cinii_url, parse_html=_cinii_parse),
            )
        )
    if args.only in ("all", "pubmed"):
        logs.append(
            ("pubmed", "PubMed", _run_db(db="PubMed", queries=q_jp + q_en, sleep_s=float(args.sleep), build_url=_pubmed_url, parse_html=_pubmed_parse))
        )

    for slug, db, results in logs:
        log_tag = f"{tag}_{slug}"
        payload = {
            "search": {
                "db": db,
                "query": "Aomori prefecture × bathroom heating/dryer adoption (installation rate) keyword search",
                "filter": {
                    "queries": [
                        {
                            "query": r.query,
                            "url": r.url,
                            "hit_count": r.hit_count,
                            "top_results": r.top_results,
                            "error": r.error,
                        }
                        for r in results
                    ]
                },
                "requested_at": requested_at,
                "search_date": datetime.now().strftime("%Y-%m-%d"),
                "log_tag": log_tag,
            },
            "generated_at": _now_iso8601_z(),
        }
        out_path = repo_root / "refs" / "search" / f"{log_tag}.json"
        _write_log(out_path, payload)
        print(f"[ok] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
