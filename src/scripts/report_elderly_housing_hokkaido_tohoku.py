from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_PREFS = [
    "北海道",
    "青森県",
    "秋田県",
    "岩手県",
    "山形県",
    "宮城県",
    "福島県",
]

BUILD_TYPE_4CAT_ORDER = ["一戸建", "長屋建", "共同住宅", "その他"]
BUILD_TYPE_2CAT_ORDER = ["一戸建・長屋建", "共同住宅・その他"]
BUILD_PERIOD_ORDER = ["1970年以前", "1971～1980年", "1981～1990年", "1991～2000年", "2001～2010年", "2011～2020年", "2021～2023年9月"]


def _slugify_ascii(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "run"


def _default_tag(slug: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_slugify_ascii(slug)}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _find_latest_estat_run(repo_root: Path) -> Path:
    runs_dir = repo_root / "outputs" / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError("outputs/runs not found")
    pattern = re.compile(r"^\d{8}_\d{6}_estat_elderly_housing$")
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not candidates:
        raise FileNotFoundError("No estat_elderly_housing runs found in outputs/runs")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _format_percent(series: pd.Series, decimals: int = 2) -> pd.Series:
    return series.astype(float).round(decimals)


def _format_int(series: pd.Series) -> pd.Series:
    return series.astype("Int64")


def _prepare_4cat_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df["area_name"].isin(TARGET_PREFS)].copy()
    df["area_order"] = df["area_name"].map({name: i for i, name in enumerate(TARGET_PREFS)})
    df["build_order"] = df["build_type_name"].map({name: i for i, name in enumerate(BUILD_TYPE_4CAT_ORDER)})
    df = df.sort_values(["area_order", "build_order"])

    share = df.pivot_table(index="area_name", columns="build_type_name", values="share_percent", aggfunc="first")
    share = share.reindex(columns=BUILD_TYPE_4CAT_ORDER)
    totals = df.groupby("area_name", as_index=False)["total_households"].first().set_index("area_name")
    share = share.merge(totals, left_index=True, right_index=True, how="left")
    share = share.reset_index()

    share = share.rename(
        columns={
            "area_name": "都道府県",
            "total_households": "総数",
            "一戸建": "一戸建（%）",
            "長屋建": "長屋建（%）",
            "共同住宅": "共同住宅（%）",
            "その他": "その他（%）",
        }
    )
    for col in ["一戸建（%）", "長屋建（%）", "共同住宅（%）", "その他（%）"]:
        share[col] = _format_percent(share[col])
    share["総数"] = _format_int(share["総数"])

    counts = df.pivot_table(index="area_name", columns="build_type_name", values="households", aggfunc="first")
    counts = counts.reindex(columns=BUILD_TYPE_4CAT_ORDER)
    counts = counts.merge(totals, left_index=True, right_index=True, how="left")
    counts = counts.reset_index()
    counts = counts.rename(
        columns={
            "area_name": "都道府県",
            "total_households": "総数",
            "一戸建": "一戸建（世帯数）",
            "長屋建": "長屋建（世帯数）",
            "共同住宅": "共同住宅（世帯数）",
            "その他": "その他（世帯数）",
        }
    )
    for col in ["一戸建（世帯数）", "長屋建（世帯数）", "共同住宅（世帯数）", "その他（世帯数）", "総数"]:
        counts[col] = _format_int(counts[col])

    share = share.sort_values("都道府県", key=lambda s: s.map({name: i for i, name in enumerate(TARGET_PREFS)}))
    counts = counts.sort_values("都道府県", key=lambda s: s.map({name: i for i, name in enumerate(TARGET_PREFS)}))
    return share, counts


def _prepare_2cat_summary_from_49(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["area_name"].isin(TARGET_PREFS)].copy()
    df["households"] = pd.to_numeric(df["households"], errors="coerce")

    det = (
        df[df["build_type_name"].isin(["一戸建", "長屋建"])]
        .groupby("area_name", as_index=False)["households"]
        .sum()
        .rename(columns={"households": "detached"})
    )
    apt = (
        df[df["build_type_name"] == "共同住宅"]
        .groupby("area_name", as_index=False)["households"]
        .sum()
        .rename(columns={"households": "apartment"})
    )
    summary = det.merge(apt, on="area_name", how="outer").fillna(0)
    summary["total"] = summary["detached"] + summary["apartment"]
    summary["detached_share"] = summary["detached"] / summary["total"] * 100
    summary["apartment_share"] = summary["apartment"] / summary["total"] * 100

    summary = summary.rename(
        columns={
            "area_name": "都道府県",
            "detached": "一戸建・長屋建（世帯数）",
            "apartment": "共同住宅（世帯数）",
        }
    )
    summary["総数"] = summary["一戸建・長屋建（世帯数）"] + summary["共同住宅（世帯数）"]
    summary["一戸建・長屋建（%）"] = _format_percent(summary["detached_share"])
    summary["共同住宅（%）"] = _format_percent(summary["apartment_share"])
    for col in ["一戸建・長屋建（世帯数）", "共同住宅（世帯数）", "総数"]:
        summary[col] = _format_int(summary[col])

    summary = summary[
        [
            "都道府県",
            "一戸建・長屋建（%）",
            "共同住宅（%）",
            "一戸建・長屋建（世帯数）",
            "共同住宅（世帯数）",
            "総数",
        ]
    ]
    summary = summary.sort_values("都道府県", key=lambda s: s.map({name: i for i, name in enumerate(TARGET_PREFS)}))
    return summary


def _prepare_build_period_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["area_name"].isin(TARGET_PREFS)].copy()
    df["area_order"] = df["area_name"].map({name: i for i, name in enumerate(TARGET_PREFS)})
    df["build_order"] = df["build_type_name"].map({name: i for i, name in enumerate(BUILD_TYPE_2CAT_ORDER)})
    df["period_order"] = df["build_period_name"].map({name: i for i, name in enumerate(BUILD_PERIOD_ORDER)})
    df = df.sort_values(["area_order", "build_order", "period_order"])

    table = df[
        [
            "area_name",
            "build_type_name",
            "build_period_name",
            "households",
            "share_percent",
            "total_households",
        ]
    ].copy()
    table = table.rename(
        columns={
            "area_name": "都道府県",
            "build_type_name": "建て方",
            "build_period_name": "建築時期",
            "households": "世帯数",
            "share_percent": "構成比（%）",
            "total_households": "建て方合計",
        }
    )
    table["構成比（%）"] = _format_percent(table["構成比（%）"])
    table["世帯数"] = _format_int(table["世帯数"])
    table["建て方合計"] = _format_int(table["建て方合計"])
    return table


def _write_report(
    out_md: Path,
    *,
    meta: dict[str, Any],
    share_4cat: pd.DataFrame,
    counts_4cat: pd.DataFrame,
    summary_2cat: pd.DataFrame,
    build_period: pd.DataFrame,
) -> None:
    share_4cat_view = share_4cat[
        ["都道府県", "一戸建（%）", "長屋建（%）", "共同住宅（%）", "その他（%）"]
    ].copy()
    summary_2cat_view = summary_2cat[
        ["都道府県", "一戸建・長屋建（%）", "共同住宅（%）"]
    ].copy()
    build_period_view = build_period[
        ["都道府県", "建て方", "建築時期", "構成比（%）"]
    ].copy()

    period_30plus = {"1970年以前", "1971～1980年", "1981～1990年"}
    build_period_30 = build_period[build_period["建築時期"].isin(period_30plus)].copy()
    build_period_30 = (
        build_period_30.groupby(["都道府県", "建て方"], as_index=False)["構成比（%）"].sum()
    )
    bp30_map = {
        (row["都道府県"], row["建て方"]): float(row["構成比（%）"])
        for _, row in build_period_30.iterrows()
    }

    summary_lines: list[tuple[str, str]] = []
    for pref in TARGET_PREFS:
        row2 = summary_2cat.loc[summary_2cat["都道府県"] == pref].iloc[0]
        older_detached = bp30_map.get((pref, "一戸建・長屋建"), float("nan"))
        older_apartment = bp30_map.get((pref, "共同住宅・その他"), float("nan"))
        line = (
            f"2区分は一戸建・長屋建{float(row2['一戸建・長屋建（%）']):.2f}%、"
            f"共同住宅{float(row2['共同住宅（%）']):.2f}%。"
            f"築30年以上（1970年以前-1990年）の割合は、"
            f"一戸建・長屋建{older_detached:.2f}%、共同住宅{older_apartment:.2f}%。"
        )
        summary_lines.append((pref, line))

    lines: list[str] = []
    lines.append("# 北海道・東北6県の住まい（家計主65歳以上）")
    lines.append("")
    lines.append("- 作成日時: " + meta["created_at_local"])
    lines.append(f"- 入力（49-2）: `{meta['input_49_2']}`（sha256={meta['input_49_2_sha256']}）")
    lines.append(f"- 入力（47-2-1）: `{meta['input_47_2_1']}`（sha256={meta['input_47_2_1_sha256']}）")
    lines.append("- 対象: 北海道、青森、秋田、岩手、山形、宮城、福島")
    lines.append("- 不詳は除外")
    lines.append("- 構造は総数（総数がない場合は木造+非木造で合算）")
    lines.append("- 記述ポリシー: `docs/rules/statistical_reporting_policy.md`")
    lines.append("")

    lines.append("## 1. 住宅の建て方（4区分; 住調49-2）")
    lines.append("")
    lines.append("構成比（%）は各都道府県内の総数に対する割合。")
    lines.append("")
    lines.append("### 1.1 構成比（%）")
    lines.append("")
    lines.append(_df_to_md_table(share_4cat_view))
    lines.append("")

    lines.append("## 2. 住宅の建て方（2区分; 住調49-2から再集計）")
    lines.append("")
    lines.append("住調49-2の4区分から「一戸建・長屋建」vs「共同住宅」を再集計（その他は除外）。構成比（%）は各都道府県内の総数に対する割合。")
    lines.append("")
    lines.append(_df_to_md_table(summary_2cat_view))
    lines.append("")

    lines.append("## 3. 建築の時期分布（住調47-2-1）")
    lines.append("")
    lines.append("構成比（%）は都道府県×建て方内の建築時期分布。")
    lines.append("建て方は47-2-1の区分（共同住宅・その他）であり、共同住宅のみへの分離は不可。")
    lines.append("")
    lines.append(_df_to_md_table(build_period_view))
    lines.append("")

    lines.append("## 4. 都道府県別の短い要約")
    lines.append("")
    lines.append("築30年以上（1970年以前-1990年）は、建築時期表の共同住宅・その他を用いた参考値。")
    lines.append("")
    for pref, sentence in summary_lines:
        lines.append(f"### {pref}")
        lines.append("")
        lines.append(sentence)
        lines.append("")

    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="北海道・東北6県の高齢世帯の住まい分布レポートを作成します。")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="estat_elderly_housing の outputs/runs/<tag> を指定（未指定なら最新を使用）",
    )
    parser.add_argument("--tag", type=str, default="", help="outputs/runs/<tag> の tag（未指定なら自動生成）")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_dir = Path(args.input_dir).expanduser() if args.input_dir else _find_latest_estat_run(repo_root)
    input_dir = input_dir.resolve()

    file_49 = input_dir / "pref_building_type_4cat_65plus_49-2.csv"
    file_47 = input_dir / "pref_build_period_by_type_65plus_47-2-1.csv"
    for path in [file_49, file_47]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    df49 = pd.read_csv(file_49, dtype={"area_code": str, "build_type_code": str})
    df47 = pd.read_csv(file_47, dtype={"area_code": str, "build_type_code": str, "build_period_code": str})

    share_4cat, counts_4cat = _prepare_4cat_summary(df49)
    summary_2cat = _prepare_2cat_summary_from_49(df49)
    build_period = _prepare_build_period_table(df47)

    tag = args.tag.strip() or _default_tag("hokkaido_tohoku_elderly_housing_report")
    out_dir = repo_root / "outputs" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "input_49_2": str(file_49),
        "input_49_2_sha256": _sha256(file_49),
        "input_47_2_1": str(file_47),
        "input_47_2_1_sha256": _sha256(file_47),
        "targets": TARGET_PREFS,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    share_4cat.to_csv(out_dir / "hokkaido_tohoku_4cat_share.csv", index=False, encoding="utf-8")
    counts_4cat.to_csv(out_dir / "hokkaido_tohoku_4cat_counts.csv", index=False, encoding="utf-8")
    summary_2cat.to_csv(out_dir / "hokkaido_tohoku_2cat_summary.csv", index=False, encoding="utf-8")
    build_period.to_csv(out_dir / "hokkaido_tohoku_build_period_by_type.csv", index=False, encoding="utf-8")

    _write_report(
        out_dir / "report.md",
        meta=meta,
        share_4cat=share_4cat,
        counts_4cat=counts_4cat,
        summary_2cat=summary_2cat,
        build_period=build_period,
    )

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
