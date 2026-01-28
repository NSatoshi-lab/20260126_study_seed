from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import font_manager

ESTAT_API_URL = "https://www.e-stat.go.jp/retrieve/api_file"
ESTAT_DOWNLOAD_URL = "https://www.e-stat.go.jp/stat-search/file-download"

DEFAULT_TOUKEI_CODE = "00650408"
DEFAULT_QUERY = '"第3-1表" 北海道 暖房'

REGIONS = ["北海道", "東北", "関東甲信", "北陸", "東海", "近畿", "中国", "四国", "九州", "沖縄"]

COL_PATTERNS: dict[str, list[list[str]]] = {
    "central_electric_boiler_pp": [
        ["使用している", "電気", "温水"],
        ["使用している", "熱源", "電気"],
    ],
    "central_gas_boiler_pp": [
        ["使用している", "ガス", "温水"],
        ["使用している", "熱源", "ガス"],
    ],
    "central_kerosene_boiler_pp": [
        ["使用している", "灯油", "温水"],
        ["使用している", "熱源", "灯油"],
    ],
    "central_duct_pp": [
        ["使用している", "ダクト"],
    ],
    "central_not_use_pp": [
        ["使用していない"],
    ],
    "central_unknown_pp": [
        ["不明"],
    ],
}


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, *, params: dict[str, Any], dest: Path, timeout_sec: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, params=params, timeout=timeout_sec, stream=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)


def _strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass(frozen=True)
class Entry:
    stat_inf_id: str
    file_kind: int
    table_id: str
    tstat_name: str
    survey_year: int | None
    era_label: str | None


def _era_to_year(label: str) -> int | None:
    m = re.search(r"(令和|平成)(\d+)年度", label)
    if not m:
        return None
    era, num = m.group(1), int(m.group(2))
    if era == "令和":
        return 2018 + num
    if era == "平成":
        return 1988 + num
    return None


def _parse_entries(items_html: str) -> list[Entry]:
    entries: list[Entry] = []
    chunks = items_html.split('<article class="stat-resource_list-item')
    for chunk in chunks[1:]:
        m = re.search(r"file-download\?statInfId=(\d+)&fileKind=(\d+)", chunk)
        if not m:
            continue
        stat_inf_id = m.group(1)
        file_kind = int(m.group(2))

        m_table = re.search(
            r'stat-resource_list-detail-item-text[\s\S]*?<span class="stat-separator">\s*([^<]+)\s*</span>',
            chunk,
        )
        table_id = _strip_tags(m_table.group(1)) if m_table else ""

        li_blocks = re.findall(r'<li class="stat-resource_list-detail-item">\s*(.*?)\s*</li>', chunk, flags=re.S)
        li_blocks = [_strip_tags(x) for x in li_blocks]
        tstat_name = li_blocks[1] if len(li_blocks) > 1 else ""

        survey_year = None
        if len(li_blocks) > 2:
            m_year = re.search(r"(\d{4})年度", li_blocks[2])
            if m_year:
                survey_year = int(m_year.group(1))

        era_label = None
        m_era = re.search(r"(令和|平成)\d+年度", tstat_name)
        if m_era:
            era_label = m_era.group(0)
            survey_year = survey_year or _era_to_year(era_label)

        entries.append(
            Entry(
                stat_inf_id=stat_inf_id,
                file_kind=file_kind,
                table_id=table_id,
                tstat_name=tstat_name,
                survey_year=survey_year,
                era_label=era_label,
            )
        )
    return entries


def _find_header_row(raw: pd.DataFrame) -> int:
    for r in range(raw.shape[0]):
        row = raw.iloc[r, :].astype(str)
        if row.str.contains("使用している").any() and row.str.contains("使用していない").any():
            return r
    raise ValueError("header row not found for central heating labels")


def _find_region_col(raw: pd.DataFrame) -> int:
    for c in range(raw.shape[1]):
        col = raw.iloc[:, c].astype(str)
        if col.isin(REGIONS).any():
            return c
    raise ValueError("region column not found")


def _extract_central_heating_region_rates(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, header=None)
    header_row = _find_header_row(raw)
    header = raw.iloc[header_row, :].astype(str)

    section_col = None
    target_label = "セントラル暖房システムの使用状況"
    for r in range(raw.shape[0]):
        row = raw.iloc[r, :].astype(str)
        for c, val in enumerate(row.tolist()):
            if str(val).strip() == target_label:
                section_col = int(c)
                break
        if section_col is not None:
            break
    if section_col is not None:
        search_cols = list(range(section_col, min(section_col + 12, raw.shape[1])))
    else:
        search_cols = list(range(raw.shape[1]))

    def find_col_idx(patterns: list[list[str]], *, required: bool = True) -> int | None:
        for keywords in patterns:
            for idx in search_cols:
                val = str(header.iloc[idx])
                if all(k in val for k in keywords):
                    return idx
        if required:
            raise ValueError(f"column not found for patterns={patterns}")
        return None

    col_idx: dict[str, int | None] = {}
    for key, patterns in COL_PATTERNS.items():
        required = key != "central_duct_pp"
        col_idx[key] = find_col_idx(patterns, required=required)

    region_col = _find_region_col(raw)
    region_rows = raw[raw.iloc[:, region_col].astype(str).isin(REGIONS)].copy()
    if region_rows.empty:
        raise ValueError("region rows not found")

    region_rows = region_rows.drop_duplicates(subset=[region_col])
    region_rows = region_rows.sort_values(by=region_col, key=lambda s: s.map({r: i for i, r in enumerate(REGIONS)}))

    out = pd.DataFrame({"region": region_rows.iloc[:, region_col].astype(str).tolist()})
    for key, idx in col_idx.items():
        if idx is None:
            out[key] = 0.0
        else:
            out[key] = pd.to_numeric(region_rows.iloc[:, idx], errors="coerce").fillna(0.0).astype(float).to_numpy()

    out["central_use_pp"] = out[
        [
            "central_electric_boiler_pp",
            "central_gas_boiler_pp",
            "central_kerosene_boiler_pp",
            "central_duct_pp",
        ]
    ].sum(axis=1)
    out["central_non_not_use_pp"] = out[
        [
            "central_electric_boiler_pp",
            "central_gas_boiler_pp",
            "central_kerosene_boiler_pp",
            "central_duct_pp",
            "central_unknown_pp",
        ]
    ].sum(axis=1)
    out = out.reset_index(drop=True)
    return out


def _df_to_md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns.tolist()]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                values.append("")
            else:
                values.append(str(v).replace("|", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _fetch_items_html(*, toukei: str, query: str) -> str:
    params = {
        "layout": "dataset",
        "page": "1",
        "toukei": toukei,
        "query": query,
        "metadata": "1",
        "data": "1",
    }
    r = requests.get(ESTAT_API_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return str(data.get("items", ""))


def _plot_timeseries(df: pd.DataFrame, *, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    x = df["year"].astype(int).to_numpy()
    y = df["central_non_not_use_pp"].astype(float).to_numpy()
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xlabel("調査年度")
    ax.set_ylabel("セントラル暖房システム使用率（%）")
    ax.set_title("北海道: セントラル暖房システム使用率（平成29-令和5年度）")
    ax.set_xticks(x)
    ax.set_xticklabels(df["year_label"].tolist(), rotation=0)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="e-Stat（家庭部門のCO2排出実態統計調査）の第3-1表から北海道のセントラル暖房使用率の年次推移を抽出します。"
    )
    parser.add_argument("--toukei", type=str, default=DEFAULT_TOUKEI_CODE, help="政府統計コード（家庭部門CO2統計）")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="e-Stat検索クエリ")
    parser.add_argument("--out-dir", type=str, default="", help="出力ディレクトリ（未指定なら outputs/runs/<tag>/）")
    parser.add_argument("--tag", type=str, default="", help="outputs/runs/<tag> の tag")
    args = parser.parse_args()

    _set_japanese_font()

    repo_root = Path(__file__).resolve().parents[2]
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = repo_root / "outputs" / "runs" / f"{stamp}_hokkaido_central_heating_timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    items_html = _fetch_items_html(toukei=args.toukei, query=args.query)
    entries = _parse_entries(items_html)
    entries = [e for e in entries if e.table_id == "3-1" and "全国" in e.tstat_name]
    entries = [e for e in entries if e.survey_year is not None]
    entries = sorted(entries, key=lambda e: e.survey_year)

    if not entries:
        raise ValueError("no entries found for table 3-1 (nationwide)")

    rows: list[dict[str, Any]] = []
    for e in entries:
        ext = "xlsx" if e.file_kind == 4 else "xls"
        estat_path = inputs_dir / f"estat_{e.stat_inf_id}_fileKind{e.file_kind}.{ext}"
        if not estat_path.exists():
            _download_file(
                ESTAT_DOWNLOAD_URL,
                params={"statInfId": e.stat_inf_id, "fileKind": str(e.file_kind)},
                dest=estat_path,
                timeout_sec=60,
            )

        region = _extract_central_heating_region_rates(estat_path)
        hk = region.loc[region["region"] == "北海道"].copy()
        if hk.empty:
            raise ValueError(f"北海道 row not found in statInfId={e.stat_inf_id}")
        hk_row = hk.iloc[0]

        rows.append(
            {
                "year": int(e.survey_year),
                "year_label": e.era_label or f"{e.survey_year}年度",
                "stat_inf_id": e.stat_inf_id,
                "file_kind": int(e.file_kind),
                "central_non_not_use_pp": float(hk_row["central_non_not_use_pp"]),
                "central_not_use_pp": float(hk_row["central_not_use_pp"]),
                "central_use_pp": float(hk_row["central_use_pp"]),
                "estat_xlsx": str(estat_path),
                "estat_xlsx_sha256": _sha256(estat_path),
            }
        )

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    df.to_csv(out_dir / "hokkaido_central_heating_timeseries.csv", index=False, encoding="utf-8")

    table = df[["year_label", "central_non_not_use_pp", "central_not_use_pp"]].copy()
    table = table.rename(
        columns={
            "year_label": "年度",
            "central_non_not_use_pp": "セントラル暖房使用率（%）",
            "central_not_use_pp": "セントラル暖房「使用していない」（%）",
        }
    )
    for col in ["セントラル暖房使用率（%）", "セントラル暖房「使用していない」（%）"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(2)
    (out_dir / "hokkaido_central_heating_timeseries_table.md").write_text(
        _df_to_md_table(table) + "\n",
        encoding="utf-8",
    )

    fig_path = out_dir / "figures" / "hokkaido_central_heating_timeseries.png"
    _plot_timeseries(df, out_png=fig_path)

    meta = {
        "created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "query": args.query,
        "toukei": args.toukei,
        "n_entries": int(df.shape[0]),
        "entries": df[["year", "year_label", "stat_inf_id", "file_kind"]].to_dict(orient="records"),
        "outputs": {
            "csv": str(out_dir / "hokkaido_central_heating_timeseries.csv"),
            "table_md": str(out_dir / "hokkaido_central_heating_timeseries_table.md"),
            "figure": str(fig_path),
        },
        "packages": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "requests": requests.__version__,
        },
    }
    (out_dir / "meta_hokkaido_central_heating_timeseries.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
