from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any


DEFAULT_BRIDGE_URL = "http://127.0.0.1:23119/codex/import"
DOTENV_FILENAME = ".env"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    url: str
    institution: str
    tags: list[str]


@dataclass(frozen=True)
class BridgeConfig:
    url: str
    token: str
    collection_name: str


def _now_tag(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{prefix}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv_if_present(*, repo_root: Path) -> None:
    dotenv_path = repo_root / DOTENV_FILENAME
    if not dotenv_path.exists():
        return

    for raw in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def _env_str(var: str) -> str:
    return os.environ.get(var, "").strip()


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout_s: int = 60,
) -> Any:
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    request.add_header("User-Agent", "aomori_survey-gray-import/0.1")
    if headers:
        for k, v in headers.items():
            request.add_header(k, v)

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = response.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {method} {url}\n{detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error for {method} {url}: {e}") from e

    if not payload:
        return None
    return json.loads(payload.decode("utf-8", errors="replace"))


def _bridge_base_url(import_url: str) -> str:
    if "/codex/import" in import_url:
        return import_url.split("/codex/import", 1)[0]
    parsed = urllib.parse.urlparse(import_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _bridge_ping(import_url: str) -> dict[str, Any]:
    base = _bridge_base_url(import_url).rstrip("/")
    url = f"{base}/codex/ping"
    data = _http_json("GET", url, headers={"Zotero-Allowed-Request": "1"}, timeout_s=10)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected ping response: {type(data)}")
    return data


def _resolve_bridge_config(args: argparse.Namespace) -> BridgeConfig:
    url = (args.bridge_url or _env_str("ZOTERO_BRIDGE_URL") or DEFAULT_BRIDGE_URL).strip()
    token = (args.bridge_token or _env_str("ZOTERO_BRIDGE_TOKEN")).strip()
    collection_name = (args.collection_name or _env_str("ZOTERO_COLLECTION_NAME") or "青森調査").strip()

    if not token:
        raise RuntimeError("ZOTERO_BRIDGE_TOKEN is required (.env).")
    if not collection_name:
        raise RuntimeError("ZOTERO_COLLECTION_NAME must not be empty.")

    return BridgeConfig(url=url, token=token, collection_name=collection_name)


def _sanitize_filename(name: str, *, max_len: int = 180) -> str:
    name = re.sub(r"[<>:\"/\\\\|?*]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_pdf(url: str, dest_path: Path) -> dict[str, Any]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url, method="GET")
    request.add_header("User-Agent", "aomori_survey-gray-import/0.1")

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            final_url = response.geturl()
            status = getattr(response, "status", None)
            headers = {k.lower(): v for k, v in dict(response.headers).items()}
            content_type = headers.get("content-type", "")
            content_length = headers.get("content-length", "")
            last_modified = headers.get("last-modified", "")

            hasher = hashlib.sha256()
            with dest_path.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    hasher.update(chunk)
                    f.write(chunk)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for GET {url}\n{detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error for GET {url}: {e}") from e

    head = dest_path.read_bytes()[:16].lstrip()
    if not head.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded content is not a PDF: {url} -> {dest_path}")

    return {
        "url": url,
        "final_url": final_url,
        "http_status": status,
        "content_type": content_type,
        "content_length": int(content_length) if str(content_length).isdigit() else None,
        "last_modified": last_modified,
        "sha256": hasher.hexdigest(),
        "path": str(dest_path),
    }


def _year_from_filename(basename: str) -> str | None:
    m = re.search(r"(19|20)\d{2}", basename)
    if not m:
        return None
    return m.group(0)


def _year_from_last_modified(last_modified: str) -> str | None:
    if not last_modified:
        return None
    try:
        dt = parsedate_to_datetime(last_modified)
    except Exception:
        return None
    return str(dt.year)


def _resolve_gray_root() -> Path:
    value = _env_str("ZOTERO_PDF_DIR_GRAY")
    if not value:
        raise RuntimeError("ZOTERO_PDF_DIR_GRAY is required (.env).")
    return Path(value).expanduser()


def _copy_to_gray_dir(src_path: Path, *, gray_root: Path, year: str) -> Path:
    dest_dir = gray_root / year
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src_path.name
    shutil.copy2(src_path, dest_path)
    return dest_path


def _bridge_import(cfg: BridgeConfig, item: dict[str, Any], *, file_path: Path, tags: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "collection": cfg.collection_name,
        "item": item,
        "tags": tags,
        "file": {
            "path": str(file_path),
            "title": file_path.stem,
            "contentType": "application/pdf",
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.token}",
        "Zotero-Allowed-Request": "1",
    }
    return _http_json("POST", cfg.url, headers=headers, data=data, timeout_s=60)


def _infer_title_from_pdf(path: Path) -> str | None:
    def looks_generic(value: str) -> bool:
        lowered = value.lower()
        if lowered in {"untitled", "unknown"}:
            return True
        if lowered.endswith(".pdf"):
            return True
        if "microsoft" in lowered:
            return True
        if "バージョン" in value or "version" in lowered:
            return True
        if value.startswith("（寄稿文）") or value.startswith("(寄稿文)"):
            return True
        return False

    pdfminer_text = ""
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        try:
            meta_title = (doc.metadata or {}).get("title") or ""
            meta_title = str(meta_title).strip()
            if meta_title and not looks_generic(meta_title):
                # Keep as fallback; prefer visible first-page text when available.
                pass
            if doc.page_count > 0:
                text = doc.load_page(0).get_text("text")
            else:
                text = ""
        finally:
            doc.close()
    except Exception:
        text = ""

    if not text:
        try:
            from pdfminer.high_level import extract_text  # type: ignore

            pdfminer_text = extract_text(str(path), maxpages=1)
        except Exception:
            pdfminer_text = ""

    text_for_lines = text or pdfminer_text or ""
    if not text_for_lines:
        return meta_title if "meta_title" in locals() and meta_title and not looks_generic(meta_title) else None

    lines = [ln.strip() for ln in text_for_lines.splitlines()]
    lines = [ln for ln in lines if ln]
    for ln in lines[:20]:
        if len(ln) < 6:
            continue
        if ln.startswith("http") or ln.startswith("www"):
            continue
        if re.fullmatch(r"\d+", ln):
            continue
        if looks_generic(ln):
            continue
        return ln
    if "meta_title" in locals() and meta_title and not looks_generic(meta_title):
        return meta_title
    return None


def _iso_date_from_last_modified(last_modified: str) -> str | None:
    if not last_modified:
        return None
    try:
        dt = parsedate_to_datetime(last_modified)
    except Exception:
        return None
    return dt.date().isoformat()


def _safe_relpath(path: Path, *, start: Path) -> str:
    try:
        return str(path.relative_to(start)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _append_section_to_refs_md(
    refs_md_path: Path,
    *,
    heading: str,
    run_date: str,
    log_rel_path: str,
    report_rel_path: str,
    sources: list[dict[str, Any]],
) -> None:
    text = refs_md_path.read_text(encoding="utf-8", errors="replace")

    if heading not in text:
        text = text.rstrip() + f"\n\n{heading}\n\n"

    lines = text.splitlines()
    heading_index = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            heading_index = i
            break
    if heading_index is None:
        raise RuntimeError("Failed to locate refs heading after insertion.")

    next_heading_index = None
    for i in range(heading_index + 1, len(lines)):
        if lines[i].startswith("## "):
            next_heading_index = i
            break
    insert_index = next_heading_index if next_heading_index is not None else len(lines)

    block_lines: list[str] = []
    block_lines.append(f"- 実施日: {run_date}")
    block_lines.append(f"- ログ: `{log_rel_path}`")
    block_lines.append("- 対象ソース:")
    for src in sources:
        title = src.get("title") or "(title pending)"
        url = src.get("url") or ""
        block_lines.append(f"  - {title} ({url})")
    block_lines.append(f"- 要約: `{report_rel_path}` を参照。")

    # Ensure blank lines around list block
    block_text = "\n".join(block_lines)

    before = lines[:insert_index]
    after = lines[insert_index:]

    if before and before[-1].strip() != "":
        before.append("")
    before.extend(block_text.splitlines())
    if after and after[0].strip() != "":
        before.append("")
    if not after:
        before.append("")

    updated = "\n".join(before + after)
    refs_md_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _build_report_md(
    *,
    run_tag: str,
    created_at_local: str,
    sources: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# 追加調査（指定URLの灰色文献）: 青森・浴室/脱衣室・寒冷地・暖房/断熱・溺死/溺水・ヒートショック")
    lines.append("")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- created_at_local: {created_at_local}")
    lines.append("")
    lines.append("## 対象ソース")
    lines.append("")
    lines.append("| source_id | 発行主体 | 推定日付 | URL | seed内パス | Zotero itemKey |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for src in sources:
        lines.append(
            "| {source_id} | {institution} | {date} | {url} | `{seed_path}` | {item_key} |".format(
                source_id=src.get("source_id", ""),
                institution=str(src.get("institution", "")),
                date=str(src.get("date", "")),
                url=str(src.get("url", "")),
                seed_path=str(src.get("seed_rel_path", "")),
                item_key=str(src.get("zotero_item_key") or ""),
            )
        )
    lines.append("")
    lines.append("## ソース別要約")
    lines.append("")
    for src in sources:
        title = src.get("title") or src.get("source_id") or "Untitled"
        lines.append(f"### {src.get('source_id')}: {title}")
        lines.append("")
        lines.append("- 浴室/脱衣室/浴槽:")
        lines.append("- 寒冷地（北海道/青森など）:")
        lines.append("- 暖房:")
        lines.append("- 断熱:")
        lines.append("- 溺死/溺水:")
        lines.append("- ヒートショック:")
        lines.append("")
    lines.append("## キーワード横断まとめ")
    lines.append("")
    lines.append("### 横断: 浴室/脱衣室/浴槽")
    lines.append("")
    lines.append("- （記入）")
    lines.append("")
    lines.append("### 横断: 寒冷地（北海道/青森など）")
    lines.append("")
    lines.append("- （記入）")
    lines.append("")
    lines.append("### 横断: 暖房")
    lines.append("")
    lines.append("- （記入）")
    lines.append("")
    lines.append("### 横断: 断熱")
    lines.append("")
    lines.append("- （記入）")
    lines.append("")
    lines.append("### 横断: 溺死/溺水")
    lines.append("")
    lines.append("- （記入）")
    lines.append("")
    lines.append("### 横断: ヒートショック")
    lines.append("")
    lines.append("- （記入）")
    lines.append("")
    lines.append("## 再現性メモ")
    lines.append("")
    lines.append("- 取得日: " + created_at_local.split(" ")[0])
    lines.append("- 取得元URLは `meta.json` と `refs/search` ログを参照。")
    lines.append("")
    return "\n".join(lines)


def _default_sources() -> list[SourceSpec]:
    return [
        SourceSpec(
            source_id="src01_onnetsu_forum_aomori",
            url="https://www.onnetsu-forum.jp/municipality/file/aomori.pdf",
            institution="温熱環境フォーラム",
            tags=["灰色文献", "青森", "入浴", "浴室", "脱衣室", "浴槽", "ヒートショック"],
        ),
        SourceSpec(
            source_id="src02_jsbc_9th_document",
            url="https://www.jsbc.or.jp/document/files/250213_9th_document.pdf",
            institution="Japan Sustainable Building Consortium (JSBC)",
            tags=["灰色文献", "住宅", "暖房", "断熱"],
        ),
        SourceSpec(
            source_id="src03_aomori_inochimamoru_pamphlet",
            url="https://www.pref.aomori.lg.jp/soshiki/kendo/kenju/files/inochimamoru_pamphlet202308.pdf",
            institution="青森県",
            tags=["灰色文献", "青森", "入浴", "浴室", "脱衣室", "ヒートショック", "溺死", "溺水"],
        ),
        SourceSpec(
            source_id="src04_aomori_dannetsu_book",
            url="https://www.pref.aomori.lg.jp/soshiki/kankyo/energy/files/2024_1010_1125.pdf",
            institution="青森県",
            tags=["灰色文献", "青森", "寒冷地", "暖房", "断熱"],
        ),
        SourceSpec(
            source_id="src05_onnetsu_forum_document7",
            url="https://www.onnetsu-forum.jp/file/document7.pdf",
            institution="温熱環境フォーラム",
            tags=["灰色文献", "入浴", "浴室", "脱衣室", "ヒートショック"],
        ),
        SourceSpec(
            source_id="src06_kenkocho_196_09",
            url="https://www.kenkocho.co.jp/html/publication/196/196_pdf/196_09.pdf",
            institution="株式会社健康長寿",
            tags=["灰色文献", "入浴", "浴室", "脱衣室", "ヒートショック", "溺死", "溺水"],
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="指定URLの灰色文献PDFを取得し、OneDrive（gray）へ保存してZoteroへ登録します。")
    parser.add_argument("--run-tag", type=str, default="", help="outputs/runs/<RUN_TAG> に保存するタグ")
    parser.add_argument("--repo-root", type=str, default="", help="リポジトリルート（未指定なら自動）")
    parser.add_argument("--bridge-url", type=str, default="", help="Zotero bridge import URL")
    parser.add_argument("--bridge-token", type=str, default="", help="Zotero bridge token")
    parser.add_argument("--collection-name", type=str, default="", help="Zotero collection name")
    parser.add_argument("--skip-zotero", action="store_true", help="Zotero登録をスキップ（DL/コピー/ログのみ）")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    _load_dotenv_if_present(repo_root=repo_root)

    run_tag = args.run_tag.strip() or _now_tag("aomori_gray_sources_heatshock")
    created_at_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    runs_dir = repo_root / "outputs" / "runs" / run_tag
    inputs_dir = runs_dir / "inputs" / "pdfs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    refs_md_path = repo_root / "refs.md"
    if not refs_md_path.exists():
        raise RuntimeError(f"refs.md not found under repo root: {repo_root}")

    gray_root = _resolve_gray_root()

    bridge_cfg = _resolve_bridge_config(args)
    zotero_ping: dict[str, Any] | None = None
    if not args.skip_zotero:
        zotero_ping = _bridge_ping(bridge_cfg.url)

    sources_out: list[dict[str, Any]] = []
    for spec in _default_sources():
        parsed = urllib.parse.urlparse(spec.url)
        base = Path(parsed.path).name or f"{spec.source_id}.pdf"
        base = _sanitize_filename(base)
        seed_pdf_path = inputs_dir / f"{spec.source_id}__{base}"

        dl = _download_pdf(spec.url, seed_pdf_path)
        last_modified = dl.get("last_modified") or ""

        year = _year_from_filename(base) or _year_from_last_modified(last_modified) or "nd"
        onedrive_pdf_path = _copy_to_gray_dir(seed_pdf_path, gray_root=gray_root, year=year)

        title = _infer_title_from_pdf(seed_pdf_path) or seed_pdf_path.stem
        title = _sanitize_filename(title, max_len=220)

        date_iso = _iso_date_from_last_modified(last_modified) or ""

        zotero_item_key = None
        zotero_attachment_key = None
        zotero_error = None
        if not args.skip_zotero:
            item_payload: dict[str, Any] = {
                "itemType": "report",
                "title": title,
                "institution": spec.institution,
                "url": spec.url,
            }
            if date_iso:
                item_payload["date"] = date_iso
            try:
                resp = _bridge_import(
                    bridge_cfg,
                    item_payload,
                    file_path=onedrive_pdf_path,
                    tags=spec.tags,
                )
                zotero_item_key = resp.get("itemKey")
                zotero_attachment_key = resp.get("attachmentKey")
            except Exception as exc:
                zotero_error = str(exc)

        sources_out.append(
            {
                "source_id": spec.source_id,
                "url": spec.url,
                "final_url": dl.get("final_url"),
                "institution": spec.institution,
                "tags": spec.tags,
                "date": date_iso,
                "last_modified": last_modified,
                "content_type": dl.get("content_type"),
                "content_length": dl.get("content_length"),
                "sha256": dl.get("sha256"),
                "seed_rel_path": _safe_relpath(seed_pdf_path, start=repo_root),
                "onedrive_path": str(onedrive_pdf_path),
                "title": title,
                "zotero_item_key": zotero_item_key,
                "zotero_attachment_key": zotero_attachment_key,
                "zotero_error": zotero_error,
                "year_bucket": year,
            }
        )

    meta = {
        "run_tag": run_tag,
        "created_at_local": created_at_local,
        "bridge_url": bridge_cfg.url,
        "collection_name": bridge_cfg.collection_name,
        "zotero_ping": zotero_ping,
        "gray_root": str(gray_root),
        "sources": sources_out,
    }
    (runs_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = runs_dir / "report.md"
    report_path.write_text(
        _build_report_md(run_tag=run_tag, created_at_local=created_at_local, sources=sources_out),
        encoding="utf-8",
    )
    report_rel_path = _safe_relpath(report_path, start=repo_root)

    log_rel_path = f"refs/search/{run_tag}_manual.json"
    log_path = repo_root / log_rel_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_payload = {
        "run_tag": run_tag,
        "created_at_local": created_at_local,
        "sources": sources_out,
    }
    log_path.write_text(json.dumps(log_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _append_section_to_refs_md(
        refs_md_path,
        heading="## 6. 追加調査（指定URLの灰色文献）",
        run_date=created_at_local.split(" ")[0],
        log_rel_path=log_rel_path,
        report_rel_path=report_rel_path,
        sources=sources_out,
    )

    print(f"[ok] run_tag={run_tag}")
    print(f"[ok] report={_safe_relpath(report_path, start=repo_root)}")
    print(f"[ok] meta={_safe_relpath(runs_dir / 'meta.json', start=repo_root)}")
    print(f"[ok] log={log_rel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
