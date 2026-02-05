from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OPENALEX_API_BASE = "https://api.openalex.org"
DEFAULT_BRIDGE_URL = "http://127.0.0.1:23119/codex/import"

DOTENV_FILENAME = ".env"
LOG_SECTION_HEADING = "## 8. 自動検索ログ（Zotero/Codex）"


def _now_iso8601_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slugify_ascii(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "search"


def _build_log_tag(query: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_slugify_ascii(query)}"


def _resolve_path(value: str, *, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path


def _format_query_for_md(query: str) -> str:
    cleaned = " ".join(query.replace("`", "'").split())
    return cleaned or "N/A"

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
    request.add_header("User-Agent", "shiin-stats-zotero-bridge-pipeline/0.2")
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


def _http_bytes(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout_s: int = 120,
) -> bytes:
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("User-Agent", "shiin-stats-zotero-bridge-pipeline/0.2")
    if headers:
        for k, v in headers.items():
            request.add_header(k, v)

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {method} {url}\n{detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error for {method} {url}: {e}") from e


def _sanitize_filename(name: str, *, max_len: int = 180) -> str:
    name = re.sub(r"[<>:\"/\\\\|?*]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def _one_drive_root() -> Path | None:
    for var in ("ZOTERO_ONE_DRIVE_ROOT", "OneDriveCommercial", "OneDriveConsumer", "OneDrive"):
        value = os.environ.get(var)
        if value:
            return Path(value)
    return None


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


def _env_path(var: str) -> Path | None:
    value = os.environ.get(var, "").strip()
    if value:
        return Path(value)
    return None


def _default_pdf_root_oa() -> Path:
    root = _one_drive_root()
    if not root:
        raise RuntimeError("OneDrive root not found. Set ZOTERO_ONE_DRIVE_ROOT or OneDrive env vars.")
    return root / "ZoteroLibrary" / "pdf" / "oa"


def _default_bbt_bib_path() -> Path:
    root = _one_drive_root()
    if not root:
        raise RuntimeError("OneDrive root not found. Set ZOTERO_ONE_DRIVE_ROOT or OneDrive env vars.")
    return root / "ZoteroLibrary" / "aomori_survey.bib"


def _bib_entry_key_by_doi(bib_text: str, doi: str) -> str | None:
    doi_norm = doi.strip().lower()
    entry_re = re.compile(r"^@\w+\s*\{\s*([^,]+)\s*,", re.MULTILINE)
    doi_re = re.compile(r"^\s*doi\s*=\s*[{\"]?([^}\",\n]+)", re.IGNORECASE | re.MULTILINE)

    for m in entry_re.finditer(bib_text):
        key = m.group(1).strip()
        start = m.start()
        end = bib_text.find("\n@", m.end())
        if end == -1:
            end = len(bib_text)
        block = bib_text[start:end]
        dm = doi_re.search(block)
        if dm and dm.group(1).strip().lower() == doi_norm:
            return key
    return None


@dataclass(frozen=True)
class BridgeConfig:
    url: str
    token: str
    collection_name: str


def _resolve_bridge_config(args: argparse.Namespace) -> BridgeConfig:
    url = (args.bridge_url or os.environ.get("ZOTERO_BRIDGE_URL", DEFAULT_BRIDGE_URL)).strip()
    token = (args.bridge_token or os.environ.get("ZOTERO_BRIDGE_TOKEN", "")).strip()
    if not token:
        raise RuntimeError("ZOTERO_BRIDGE_TOKEN is required (set it in env vars; do not commit it to git).")

    collection_name = (args.collection_name or os.environ.get("ZOTERO_COLLECTION_NAME", "青森調査")).strip()
    if not collection_name:
        raise RuntimeError("ZOTERO_COLLECTION_NAME must not be empty")

    return BridgeConfig(url=url, token=token, collection_name=collection_name)


def _openalex_search_oa(query: str, *, per_page: int = 5) -> list[dict[str, Any]]:
    params = {
        "search": query,
        "per-page": str(per_page),
        "filter": "open_access.is_oa:true",
    }
    url = f"{OPENALEX_API_BASE}/works?{urllib.parse.urlencode(params)}"
    data = _http_json("GET", url)
    return list(data.get("results", []))


def _openalex_best_pdf_url(work: dict[str, Any]) -> str | None:
    for key in ("best_oa_location", "primary_location"):
        loc = work.get(key) or {}
        pdf_url = loc.get("pdf_url")
        if pdf_url:
            return str(pdf_url)
    oa_url = (work.get("open_access") or {}).get("oa_url")
    if oa_url and str(oa_url).lower().endswith(".pdf"):
        return str(oa_url)
    return None


def _openalex_landing_url(work: dict[str, Any]) -> str | None:
    primary_location = work.get("primary_location") or {}
    url = primary_location.get("landing_page_url") or work.get("id")
    if url:
        return str(url).strip()
    return None


def _openalex_doi(work: dict[str, Any]) -> str | None:
    doi = work.get("doi")
    if not doi:
        return None
    doi = str(doi)
    if doi.lower().startswith("https://doi.org/"):
        return doi[len("https://doi.org/") :]
    return doi


def _openalex_title(work: dict[str, Any]) -> str:
    return str(work.get("title") or "").strip()


def _openalex_year(work: dict[str, Any]) -> str | None:
    year = work.get("publication_year")
    if year is None:
        return None
    return str(year)


def _openalex_first_author_lastname(work: dict[str, Any]) -> str | None:
    authorships = work.get("authorships") or []
    if not authorships:
        return None
    author = (authorships[0] or {}).get("author") or {}
    name = str(author.get("display_name") or "").strip()
    if not name:
        return None
    return name.split()[-1]


def _openalex_to_zotero_creators(work: dict[str, Any]) -> list[dict[str, str]]:
    creators: list[dict[str, str]] = []
    for authorship in work.get("authorships") or []:
        author = (authorship or {}).get("author") or {}
        name = str(author.get("display_name") or "").strip()
        if not name:
            continue
        parts = name.split()
        if len(parts) == 1:
            creators.append({"creatorType": "author", "firstName": "", "lastName": parts[0]})
        else:
            creators.append({"creatorType": "author", "firstName": " ".join(parts[:-1]), "lastName": parts[-1]})
    return creators


def _download_pdf(pdf_url: str, dest_path: Path) -> None:
    content = _http_bytes("GET", pdf_url, timeout_s=180)
    if not content.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded content is not a PDF: {pdf_url}")
    dest_path.write_bytes(content)


def _build_pdf_filename(work: dict[str, Any]) -> str:
    title = _openalex_title(work) or "untitled"
    year = _openalex_year(work) or "nd"
    author = _openalex_first_author_lastname(work) or "anon"
    short_title = re.sub(r"\W+", " ", title, flags=re.UNICODE).strip()
    short_title = " ".join(short_title.split()[:12]) or "untitled"
    base = f"{author}_{year}_{short_title}.pdf"
    return _sanitize_filename(base)


def _build_item_payload(work: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "itemType": "journalArticle",
        "title": _openalex_title(work),
    }
    if (year := _openalex_year(work)):
        payload["date"] = year
    if (doi := _openalex_doi(work)):
        payload["DOI"] = doi
    if (url := _openalex_landing_url(work)):
        payload["url"] = url
    host_venue = (work.get("host_venue") or {}).get("display_name")
    if host_venue:
        payload["publicationTitle"] = str(host_venue)
    creators = _openalex_to_zotero_creators(work)
    if creators:
        payload["creators"] = creators
    return payload


def _bridge_import(cfg: BridgeConfig, item: dict[str, Any], *, file_path: Path | None, tags: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "collection": cfg.collection_name,
        "item": item,
        "tags": tags,
    }
    if file_path:
        payload["file"] = {
            "path": str(file_path),
            "title": file_path.stem,
            "contentType": "application/pdf",
        }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.token}",
        "Zotero-Allowed-Request": "1",
    }
    return _http_json("POST", cfg.url, headers=headers, data=data, timeout_s=60)


def _is_list_item(line: str) -> bool:
    stripped = line.lstrip()
    if stripped.startswith("- ") or stripped.startswith("* "):
        return True
    return re.match(r"\d+\.\s", stripped) is not None


def _write_search_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    log_path.write_text(f"{data}\n", encoding="utf-8")


def _build_refs_entry(
    *,
    search_date: str,
    db_name: str,
    query: str,
    log_rel_path: str,
    non_oa_entries: list[dict[str, str]],
) -> str:
    query_md = _format_query_for_md(query)
    entry = f"- {search_date} ({db_name}) query: `{query_md}` log: `{log_rel_path}`"
    if non_oa_entries:
        items = []
        for entry_item in non_oa_entries:
            doi = entry_item.get("doi") or "(none)"
            url = entry_item.get("url") or "(none)"
            items.append(f"DOI: {doi} URL: {url}")
        entry += f"; non_oa: {' / '.join(items)}"
    return entry


def _collect_non_oa_entries(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for result in results:
        if result.get("pdf_path"):
            continue
        doi = result.get("doi") or ""
        url = result.get("landing_url") or result.get("pdf_url") or ""
        entries.append({"doi": doi, "url": url})
    return entries


def _insert_refs_entry(text: str, *, heading: str, entry: str) -> str:
    if heading not in text:
        return text.rstrip() + f"\n\n{heading}\n\n{entry}\n"

    lines = text.splitlines()
    heading_index = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            heading_index = i
            break
    if heading_index is None:
        return text.rstrip() + f"\n\n{heading}\n\n{entry}\n"

    next_heading_index = None
    for i in range(heading_index + 1, len(lines)):
        if lines[i].startswith("## "):
            next_heading_index = i
            break

    insert_index = next_heading_index if next_heading_index is not None else len(lines)
    entry_lines = [entry]

    prev_index = insert_index - 1
    while prev_index >= 0 and lines[prev_index].strip() == "":
        prev_index -= 1
    if prev_index >= 0 and not _is_list_item(lines[prev_index]):
        if insert_index == 0 or lines[insert_index - 1].strip() != "":
            entry_lines.insert(0, "")

    if insert_index == len(lines):
        entry_lines.append("")
    elif lines[insert_index].strip() != "":
        entry_lines.append("")

    updated_lines = lines[:insert_index] + entry_lines + lines[insert_index:]
    return "\n".join(updated_lines) + "\n"


def _append_refs_summary(
    refs_md_path: Path,
    *,
    log_rel_path: str,
    search_date: str,
    db_name: str,
    query: str,
    non_oa_entries: list[dict[str, str]],
) -> bool:
    if not refs_md_path.exists():
        raise RuntimeError(f"refs.md not found: {refs_md_path}")

    text = refs_md_path.read_text(encoding="utf-8", errors="replace")
    if log_rel_path in text:
        return False

    entry = _build_refs_entry(
        search_date=search_date,
        db_name=db_name,
        query=query,
        log_rel_path=log_rel_path,
        non_oa_entries=non_oa_entries,
    )
    updated = _insert_refs_entry(text, heading=LOG_SECTION_HEADING, entry=entry)
    refs_md_path.write_text(updated, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search OA works, download PDFs to OneDrive, register in Zotero via local bridge, "
            "and output citekeys."
        )
    )
    parser.add_argument("--query", required=True, help="Search query (keywords).")
    parser.add_argument("--max-results", type=int, default=3, help="Maximum OA works to import.")
    parser.add_argument("--dry-run", action="store_true", help="Do not download or write to Zotero.")
    parser.add_argument("--skip-without-pdf", action="store_true", help="Skip items without a downloadable PDF.")
    parser.add_argument("--wait-bib-seconds", type=int, default=180, help="Wait time for BBT export to include new DOIs.")
    parser.add_argument(
        "--bib-path",
        default="",
        help="BBT .bib path (defaults to OneDrive/ZoteroLibrary/aomori_survey.bib).",
    )
    parser.add_argument("--pdf-dir-oa", default="", help="OA PDF root dir (defaults to OneDrive/ZoteroLibrary/pdf/oa).")
    parser.add_argument("--bridge-url", default="", help="Local bridge URL (defaults to http://127.0.0.1:23119/codex/import).")
    parser.add_argument("--bridge-token", default="", help="Local bridge token (defaults to ZOTERO_BRIDGE_TOKEN).")
    parser.add_argument("--collection-name", default="", help="Zotero collection name (defaults to ZOTERO_COLLECTION_NAME).")
    parser.add_argument("--log-dir", default="refs/search", help="Search log directory (default: refs/search).")
    parser.add_argument("--log-tag", default="", help="Log tag (default: YYYYMMDD_HHMMSS_<slug>).")
    parser.add_argument("--refs-md", default="refs.md", help="refs.md path to update summary.")
    parser.add_argument("--skip-refs-md", action="store_true", help="Skip refs.md summary update.")
    parser.add_argument("--sync-bib-to-repo", action="store_true", help="Copy bib file to refs/bib/ (ignored by git).")
    parser.add_argument(
        "--repo-bib-name",
        default="aomori_survey.bib",
        help="Destination bib filename under refs/bib/ when syncing.",
    )
    parser.add_argument("--markdown", default="", help="Markdown file to update with citekeys (optional).")
    parser.add_argument(
        "--placeholder",
        default="{{CITE}}",
        help="Placeholder text to replace in Markdown (default: {{CITE}}).",
    )
    parser.add_argument(
        "--insert-mode",
        choices=["replace", "append"],
        default="replace",
        help="How to update Markdown when --markdown is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    _load_dotenv_if_present(repo_root=repo_root)
    bridge_cfg = _resolve_bridge_config(args)

    pdf_root = (
        Path(args.pdf_dir_oa).expanduser()
        if args.pdf_dir_oa
        else _env_path("ZOTERO_PDF_DIR_OA") or _default_pdf_root_oa()
    )
    bib_path = (
        Path(args.bib_path).expanduser()
        if args.bib_path
        else _env_path("ZOTERO_BBT_BIB_PATH") or _default_bbt_bib_path()
    )
    log_dir = _resolve_path(args.log_dir, repo_root=repo_root)
    log_tag = args.log_tag.strip() or _build_log_tag(args.query)
    log_path = log_dir / f"{log_tag}.json"
    refs_md_path = _resolve_path(args.refs_md, repo_root=repo_root)
    pdf_root.mkdir(parents=True, exist_ok=True)

    works = _openalex_search_oa(args.query, per_page=max(args.max_results, 1) * 3)
    selected: list[dict[str, Any]] = []
    seen_doi: set[str] = set()

    for work in works:
        title = _openalex_title(work)
        if not title:
            continue
        doi = _openalex_doi(work)
        if doi and doi in seen_doi:
            continue
        pdf_url = _openalex_best_pdf_url(work)
        if not pdf_url and args.skip_without_pdf:
            continue
        selected.append(work)
        if doi:
            seen_doi.add(doi)
        if len(selected) >= args.max_results:
            break

    results: list[dict[str, Any]] = []
    for work in selected:
        doi = _openalex_doi(work)
        pdf_url = _openalex_best_pdf_url(work)
        landing_url = _openalex_landing_url(work)
        openalex_id = str(work.get("id") or "").strip() or None
        year = _openalex_year(work) or "nd"
        pdf_dir = pdf_root / year
        filename = _build_pdf_filename(work)
        pdf_path = pdf_dir / filename if pdf_url else None

        tags = ["oa", "codex_import"]
        download_error: str | None = None

        if pdf_url and not args.dry_run:
            pdf_dir.mkdir(parents=True, exist_ok=True)
            if pdf_path and not pdf_path.exists():
                try:
                    _download_pdf(pdf_url, pdf_path)
                except Exception as e:
                    download_error = str(e)
                    if args.skip_without_pdf:
                        continue
                    pdf_path = None
                    tags.append("pdf_download_failed")
                    tags.append("no_pdf")
        elif not pdf_url:
            tags.append("no_pdf")

        if args.dry_run:
            results.append(
                {
                    "doi": doi,
                    "title": _openalex_title(work),
                    "pdf_url": pdf_url,
                    "pdf_path": str(pdf_path) if pdf_path else None,
                    "landing_url": landing_url,
                    "openalex_id": openalex_id,
                    "zotero_item_key": None,
                    "zotero_attachment_key": None,
                    "citekey": None,
                    "error": download_error,
                }
            )
            continue

        item_payload = _build_item_payload(work)
        try:
            response = _bridge_import(bridge_cfg, item_payload, file_path=pdf_path, tags=tags)
            results.append(
                {
                    "doi": doi,
                    "title": _openalex_title(work),
                    "pdf_url": pdf_url,
                    "pdf_path": str(pdf_path) if pdf_path else None,
                    "landing_url": landing_url,
                    "openalex_id": openalex_id,
                    "zotero_item_key": response.get("parentItemKey"),
                    "zotero_attachment_key": response.get("attachmentKey"),
                    "citekey": None,
                    "bridge_response": response,
                    "error": download_error,
                }
            )
        except Exception as e:
            results.append(
                {
                    "doi": doi,
                    "title": _openalex_title(work),
                    "pdf_url": pdf_url,
                    "pdf_path": str(pdf_path) if pdf_path else None,
                    "landing_url": landing_url,
                    "openalex_id": openalex_id,
                    "zotero_item_key": None,
                    "zotero_attachment_key": None,
                    "citekey": None,
                    "error": str(e),
                }
            )

    # Attempt to resolve citekeys from BBT export (requires Zotero + BBT auto-export)
    if results and args.wait_bib_seconds > 0:
        deadline = time.time() + args.wait_bib_seconds
        pending = {r["doi"] for r in results if r.get("doi")}
        while pending and time.time() < deadline:
            if bib_path.exists():
                bib_text = bib_path.read_text(encoding="utf-8", errors="replace")
                for r in results:
                    doi = r.get("doi")
                    if doi and r.get("citekey") is None:
                        key = _bib_entry_key_by_doi(bib_text, doi)
                        if key:
                            r["citekey"] = key
                            pending.discard(doi)
            if pending:
                time.sleep(3)

    search_date = datetime.now().date().isoformat()
    db_name = "OpenAlex"
    log_results: list[dict[str, Any]] = []
    for result in results:
        log_results.append(
            {
                "title": result.get("title"),
                "doi": result.get("doi"),
                "landing_url": result.get("landing_url"),
                "openalex_id": result.get("openalex_id"),
                "pdf_url": result.get("pdf_url"),
                "pdf_path": result.get("pdf_path"),
                "zotero_item_key": result.get("zotero_item_key"),
                "zotero_attachment_key": result.get("zotero_attachment_key"),
                "citekey": result.get("citekey"),
                "error": result.get("error"),
            }
        )

    log_payload = {
        "search": {
            "db": db_name,
            "query": args.query,
            "filter": "open_access.is_oa:true",
            "requested_at": _now_iso8601_z(),
            "search_date": search_date,
            "max_results": args.max_results,
            "dry_run": args.dry_run,
            "skip_without_pdf": args.skip_without_pdf,
            "log_tag": log_tag,
        },
        "pipeline": {
            "bridge_url": bridge_cfg.url,
            "collection_name": bridge_cfg.collection_name,
            "pdf_dir_oa": str(pdf_root),
            "bib_path": str(bib_path),
        },
        "results": log_results,
        "generated_at": _now_iso8601_z(),
    }
    _write_search_log(log_path, log_payload)

    log_rel_path = str(log_path)
    try:
        log_rel_path = log_path.relative_to(repo_root).as_posix()
    except ValueError:
        pass

    refs_md_updated = False
    if not args.skip_refs_md:
        non_oa_entries = _collect_non_oa_entries(log_results)
        refs_md_updated = _append_refs_summary(
            refs_md_path,
            log_rel_path=log_rel_path,
            search_date=search_date,
            db_name=db_name,
            query=args.query,
            non_oa_entries=non_oa_entries,
        )

    # Optional: sync bib to repo (local-only; bib files are gitignored)
    repo_bib_path: str | None = None
    if args.sync_bib_to_repo and bib_path.exists():
        dest = repo_root / "refs" / "bib" / args.repo_bib_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(bib_path.read_bytes())
        repo_bib_path = str(dest)

    # Optional: update Markdown with citekeys (Pandoc-style)
    if args.markdown:
        md_path = Path(args.markdown).expanduser()
        citekeys = [r["citekey"] for r in results if r.get("citekey")]
        if citekeys:
            citation = "[@{}]".format("; @".join(citekeys))
            text = md_path.read_text(encoding="utf-8", errors="replace")
            if args.insert_mode == "replace" and args.placeholder in text:
                text = text.replace(args.placeholder, citation)
            elif args.insert_mode == "append":
                if not text.endswith("\n"):
                    text += "\n"
                text += citation + "\n"
            md_path.write_text(text, encoding="utf-8")

    print(
        json.dumps(
            {
                "results": results,
                "bib_path": str(bib_path),
                "repo_bib_path": repo_bib_path,
                "search_log_path": log_rel_path,
                "refs_md_path": str(refs_md_path),
                "refs_md_updated": refs_md_updated,
                "generated_at": _now_iso8601_z(),
            },
            # NOTE: Windows consoles may use cp932 and fail on characters like EN DASH.
            # Keep stdout ASCII-safe while preserving UTF-8 logs on disk.
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
