from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PREF_NAMES = [
    "北海道",
    "青森県",
    "岩手県",
    "宮城県",
    "秋田県",
    "山形県",
    "福島県",
    "茨城県",
    "栃木県",
    "群馬県",
    "埼玉県",
    "千葉県",
    "東京都",
    "神奈川県",
    "新潟県",
    "富山県",
    "石川県",
    "福井県",
    "山梨県",
    "長野県",
    "岐阜県",
    "静岡県",
    "愛知県",
    "三重県",
    "滋賀県",
    "京都府",
    "大阪府",
    "兵庫県",
    "奈良県",
    "和歌山県",
    "鳥取県",
    "島根県",
    "岡山県",
    "広島県",
    "山口県",
    "徳島県",
    "香川県",
    "愛媛県",
    "高知県",
    "福岡県",
    "佐賀県",
    "長崎県",
    "熊本県",
    "大分県",
    "宮崎県",
    "鹿児島県",
    "沖縄県",
]


STATS_49_2 = "0004021674"
STATS_47_2_1 = "0004021670"


def _slugify_ascii(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "run"


def _default_tag(slug: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_slugify_ascii(slug)}"


def _estat_get_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    base = "https://api.e-stat.go.jp/rest/3.0/app/json"
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{base}/{endpoint}?{query}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    root_key = {
        "getMetaInfo": "GET_META_INFO",
        "getStatsData": "GET_STATS_DATA",
    }.get(endpoint)
    if root_key is None:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    root = data.get(root_key, {})
    result = root.get("RESULT", {})
    status = result.get("STATUS")
    if status not in (0, "0", None):
        raise RuntimeError(f"e-Stat API error ({endpoint}): {result}")
    return root


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_classes(meta_root: dict[str, Any]) -> list[dict[str, Any]]:
    classes = meta_root.get("METADATA_INF", {}).get("CLASS_INF", {}).get("CLASS_OBJ", [])
    classes = _ensure_list(classes)
    parsed: list[dict[str, Any]] = []
    for cls in classes:
        class_id = cls.get("@id")
        class_name = cls.get("@name")
        items = _ensure_list(cls.get("CLASS"))
        entries = []
        for item in items:
            entries.append(
                {
                    "code": item.get("@code"),
                    "name": item.get("@name"),
                    "level": item.get("@level"),
                    "parent_code": item.get("@parentCode"),
                    "unit": item.get("@unit"),
                }
            )
        parsed.append({"id": class_id, "name": class_name, "classes": entries})
    return parsed


def _class_map(parsed_classes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for cls in parsed_classes:
        code_to_name = {c["code"]: c["name"] for c in cls["classes"]}
        mapping[cls["id"]] = {
            "name": cls["name"],
            "code_to_name": code_to_name,
            "classes": cls["classes"],
        }
    return mapping


def _find_class_by_id(parsed_classes: list[dict[str, Any]], class_id: str) -> dict[str, Any] | None:
    for cls in parsed_classes:
        if cls["id"] == class_id:
            return cls
    return None


def _find_class_by_keyword(parsed_classes: list[dict[str, Any]], keyword: str) -> dict[str, Any] | None:
    for cls in parsed_classes:
        if cls["name"] and keyword in cls["name"]:
            return cls
    return None


def _pick_codes(
    cls: dict[str, Any],
    *,
    include_if: callable | None = None,
    exclude_if: callable | None = None,
) -> list[str]:
    codes: list[str] = []
    for item in cls["classes"]:
        name = item.get("name") or ""
        if exclude_if and exclude_if(name):
            continue
        if include_if and not include_if(name):
            continue
        codes.append(item.get("code"))
    return codes


def _pick_single_code_by_name(cls: dict[str, Any], matcher: callable) -> str:
    matches = [item for item in cls["classes"] if matcher(item.get("name") or "")]
    if len(matches) == 1:
        return matches[0]["code"]
    if not matches:
        raise ValueError(f"No matching code for {cls['name']}")
    exact = [item for item in matches if matcher(item.get("name") or "") and item.get("name") == "65歳以上"]
    if exact:
        return exact[0]["code"]
    return matches[0]["code"]


def _pick_time_code(cls: dict[str, Any], year: str) -> str:
    for item in cls["classes"]:
        name = item.get("name") or ""
        if year in name:
            return item["code"]
    for item in cls["classes"]:
        code = item.get("code") or ""
        if code.startswith(year):
            return item["code"]
    if len(cls["classes"]) == 1:
        return cls["classes"][0]["code"]
    raise ValueError(f"Time code for {year} not found in {cls['name']}")


def _prefecture_codes(area_cls: dict[str, Any]) -> dict[str, str]:
    pref_set = set(PREF_NAMES)
    mapping: dict[str, str] = {}
    for item in area_cls["classes"]:
        name = item.get("name")
        if name in pref_set:
            mapping[name] = item.get("code")
    missing = [p for p in PREF_NAMES if p not in mapping]
    if missing:
        raise ValueError(f"Prefecture codes not found: {missing}")
    return mapping


def _national_code(area_cls: dict[str, Any]) -> str | None:
    for item in area_cls["classes"]:
        if item.get("name") == "全国":
            return item.get("code")
    return None


def _build_params(app_id: str, stats_id: str, code_map: dict[str, list[str]]) -> dict[str, Any]:
    params: dict[str, Any] = {"appId": app_id, "statsDataId": stats_id, "lang": "J"}
    for class_id, codes in code_map.items():
        if not codes:
            continue
        param_name = "cd" + class_id[:1].upper() + class_id[1:]
        params[param_name] = ",".join(codes)
    return params


def _fetch_stats_data(app_id: str, stats_id: str, code_map: dict[str, list[str]]) -> list[dict[str, Any]]:
    params = _build_params(app_id, stats_id, code_map)
    params["limit"] = 10000
    params["startPosition"] = 1
    all_values: list[dict[str, Any]] = []
    while True:
        root = _estat_get_json("getStatsData", params)
        data_inf = root.get("STATISTICAL_DATA", {}).get("DATA_INF", {})
        values = _ensure_list(data_inf.get("VALUE"))
        all_values.extend(values)
        result_inf = root.get("STATISTICAL_DATA", {}).get("RESULT_INF", {})
        total = result_inf.get("TOTAL_NUMBER")
        if total is None:
            break
        total_num = int(total)
        if len(all_values) >= total_num:
            break
        params["startPosition"] = len(all_values) + 1
    return all_values


def _values_to_df(values: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in values:
        row: dict[str, Any] = {}
        for key, val in value.items():
            if key == "$":
                row["value"] = val
                continue
            if key.startswith("@"):
                row[key[1:]] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _attach_axis_labels(df: pd.DataFrame, class_map: dict[str, dict[str, Any]], axis_id: str, prefix: str) -> pd.DataFrame:
    if axis_id not in df.columns:
        return df
    mapping = class_map.get(axis_id, {}).get("code_to_name", {})
    df[f"{prefix}_code"] = df[axis_id]
    df[f"{prefix}_name"] = df[axis_id].map(mapping)
    return df


def _write_class_csv(out_path: Path, parsed_classes: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for cls in parsed_classes:
        for item in cls["classes"]:
            rows.append(
                {
                    "class_id": cls["id"],
                    "class_name": cls["name"],
                    "code": item.get("code"),
                    "name": item.get("name"),
                    "level": item.get("level"),
                    "parent_code": item.get("parent_code"),
                    "unit": item.get("unit"),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")


def _add_area_order(df: pd.DataFrame) -> pd.DataFrame:
    order = ["全国"] + PREF_NAMES
    order_index = {name: i for i, name in enumerate(order)}
    df["area_order"] = df["area_name"].map(order_index)
    return df


def _prepare_49_2(
    app_id: str,
    *,
    out_dir: Path,
    meta_out: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta_root = _estat_get_json("getMetaInfo", {"appId": app_id, "statsDataId": STATS_49_2, "lang": "J"})
    parsed_classes = _parse_classes(meta_root)
    class_map = _class_map(parsed_classes)

    age_cls = _find_class_by_keyword(parsed_classes, "年齢")
    build_cls = _find_class_by_keyword(parsed_classes, "建て方")
    structure_cls = _find_class_by_keyword(parsed_classes, "構造")
    area_cls = _find_class_by_id(parsed_classes, "area")
    time_cls = _find_class_by_id(parsed_classes, "time")
    tab_cls = _find_class_by_id(parsed_classes, "tab")

    if not all([age_cls, build_cls, structure_cls, area_cls, time_cls]):
        raise ValueError("Required classes not found for statsDataId=0004021674")

    age_code = _pick_single_code_by_name(age_cls, lambda n: "65歳以上" in n)
    build_codes = _pick_codes(
        build_cls,
        exclude_if=lambda n: "総数" in n or "不詳" in n,
    )
    structure_total = [c["code"] for c in structure_cls["classes"] if c.get("name") == "総数"]
    if structure_total:
        structure_codes = [structure_total[0]]
        structure_agg = False
    else:
        structure_codes = _pick_codes(structure_cls, exclude_if=lambda n: "不詳" in n)
        structure_agg = True

    pref_codes = _prefecture_codes(area_cls)
    national = _national_code(area_cls)
    area_codes = [pref_codes[p] for p in PREF_NAMES]
    if national:
        area_codes = [national] + area_codes

    time_code = _pick_time_code(time_cls, "2023")

    code_map = {
        "area": area_codes,
        "time": [time_code],
        "cat01": build_codes,
        "cat02": structure_codes,
        "cat03": [age_code],
    }
    if tab_cls:
        tab_codes = [tab_cls["classes"][0]["code"]] if tab_cls.get("classes") else []
        if tab_codes:
            code_map["tab"] = tab_codes

    values = _fetch_stats_data(app_id, STATS_49_2, code_map)
    df_raw = _values_to_df(values)

    df_raw = _attach_axis_labels(df_raw, class_map, "area", "area")
    df_raw = _attach_axis_labels(df_raw, class_map, "time", "time")
    df_raw = _attach_axis_labels(df_raw, class_map, "cat01", "build_type")
    df_raw = _attach_axis_labels(df_raw, class_map, "cat02", "structure")
    df_raw = _attach_axis_labels(df_raw, class_map, "cat03", "age")
    df_raw = _attach_axis_labels(df_raw, class_map, "tab", "tab")

    df_use = df_raw.copy()
    if structure_agg:
        df_use = (
            df_use.groupby(
                ["area_code", "area_name", "time_code", "time_name", "build_type_code", "build_type_name", "age_code", "age_name"],
                as_index=False,
            )["value"]
            .sum()
            .copy()
        )
        df_use["structure_code"] = "sum"
        df_use["structure_name"] = "木造+非木造"
    else:
        df_use = df_use.loc[df_use["structure_code"].isin(structure_codes)].copy()

    df_use = df_use.loc[df_use["build_type_code"].isin(build_codes)].copy()

    totals = df_use.groupby(["area_code", "area_name"], as_index=False)["value"].sum().rename(columns={"value": "total_households"})
    df_use = df_use.merge(totals, on=["area_code", "area_name"], how="left")
    df_use["share_percent"] = (df_use["value"] / df_use["total_households"] * 100).round(4)

    df_use = _add_area_order(df_use)
    build_order = {code: i for i, code in enumerate(build_codes)}
    df_use["build_type_order"] = df_use["build_type_code"].map(build_order)
    df_use = df_use.sort_values(["area_order", "build_type_order"]).drop(columns=["area_order", "build_type_order"])

    cols = [
        "area_code",
        "area_name",
        "build_type_code",
        "build_type_name",
        "households",
        "share_percent",
        "total_households",
        "structure_name",
        "age_name",
        "time_name",
    ]
    df_use = df_use.rename(columns={"value": "households"})
    df_use = df_use[cols]

    _write_class_csv(out_dir / "meta_0004021674_classes.csv", parsed_classes)

    meta_out["0004021674"] = {
        "title": meta_root.get("METADATA_INF", {}).get("TABLE_INF", {}).get("TITLE", {}).get("$"),
        "updated_date": meta_root.get("METADATA_INF", {}).get("TABLE_INF", {}).get("UPDATED_DATE"),
        "age_code": age_code,
        "build_type_codes": build_codes,
        "structure_codes": structure_codes,
        "structure_aggregated": structure_agg,
        "area_codes": area_codes,
        "time_code": time_code,
    }
    return df_use, meta_out


def _prepare_47_2_1(
    app_id: str,
    *,
    out_dir: Path,
    meta_out: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    meta_root = _estat_get_json("getMetaInfo", {"appId": app_id, "statsDataId": STATS_47_2_1, "lang": "J"})
    parsed_classes = _parse_classes(meta_root)
    class_map = _class_map(parsed_classes)

    build_period_cls = _find_class_by_keyword(parsed_classes, "建築の時期")
    build_cls = _find_class_by_keyword(parsed_classes, "建て方")
    structure_cls = _find_class_by_keyword(parsed_classes, "構造")
    age_cls = _find_class_by_keyword(parsed_classes, "年齢")
    area_cls = _find_class_by_id(parsed_classes, "area")
    time_cls = _find_class_by_id(parsed_classes, "time")
    tab_cls = _find_class_by_id(parsed_classes, "tab")

    if not all([build_period_cls, build_cls, structure_cls, age_cls, area_cls, time_cls]):
        raise ValueError("Required classes not found for statsDataId=0004021670")

    build_period_codes = _pick_codes(
        build_period_cls,
        exclude_if=lambda n: "総数" in n or "不詳" in n,
    )
    build_codes = _pick_codes(
        build_cls,
        exclude_if=lambda n: "総数" in n or "不詳" in n,
    )

    structure_total = [c["code"] for c in structure_cls["classes"] if c.get("name") == "総数"]
    if structure_total:
        structure_codes = [structure_total[0]]
        structure_agg = False
    else:
        structure_codes = _pick_codes(structure_cls, exclude_if=lambda n: "不詳" in n)
        structure_agg = True

    age_code = _pick_single_code_by_name(age_cls, lambda n: "65歳以上" in n)

    pref_codes = _prefecture_codes(area_cls)
    national = _national_code(area_cls)
    area_codes = [pref_codes[p] for p in PREF_NAMES]
    if national:
        area_codes = [national] + area_codes

    time_code = _pick_time_code(time_cls, "2023")

    code_map = {
        "area": area_codes,
        "time": [time_code],
        "cat01": build_period_codes,
        "cat02": build_codes,
        "cat03": structure_codes,
        "cat04": [age_code],
    }
    if tab_cls:
        tab_codes = [tab_cls["classes"][0]["code"]] if tab_cls.get("classes") else []
        if tab_codes:
            code_map["tab"] = tab_codes

    values = _fetch_stats_data(app_id, STATS_47_2_1, code_map)
    df_raw = _values_to_df(values)

    df_raw = _attach_axis_labels(df_raw, class_map, "area", "area")
    df_raw = _attach_axis_labels(df_raw, class_map, "time", "time")
    df_raw = _attach_axis_labels(df_raw, class_map, "cat01", "build_period")
    df_raw = _attach_axis_labels(df_raw, class_map, "cat02", "build_type")
    df_raw = _attach_axis_labels(df_raw, class_map, "cat03", "structure")
    df_raw = _attach_axis_labels(df_raw, class_map, "cat04", "age")
    df_raw = _attach_axis_labels(df_raw, class_map, "tab", "tab")

    df_use = df_raw.copy()
    if structure_agg:
        df_use = (
            df_use.groupby(
                [
                    "area_code",
                    "area_name",
                    "time_code",
                    "time_name",
                    "build_type_code",
                    "build_type_name",
                    "build_period_code",
                    "build_period_name",
                    "age_code",
                    "age_name",
                ],
                as_index=False,
            )["value"]
            .sum()
            .copy()
        )
        df_use["structure_code"] = "sum"
        df_use["structure_name"] = "木造+非木造"
    else:
        df_use = df_use.loc[df_use["structure_code"].isin(structure_codes)].copy()

    df_use = df_use.loc[df_use["build_period_code"].isin(build_period_codes)].copy()
    df_use = df_use.loc[df_use["build_type_code"].isin(build_codes)].copy()

    totals = (
        df_use.groupby(["area_code", "area_name", "build_type_code", "build_type_name"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "total_households"})
    )
    df_use = df_use.merge(totals, on=["area_code", "area_name", "build_type_code", "build_type_name"], how="left")
    df_use["share_percent"] = (df_use["value"] / df_use["total_households"] * 100).round(4)

    df_use = _add_area_order(df_use)
    build_order = {code: i for i, code in enumerate(build_codes)}
    period_order = {code: i for i, code in enumerate(build_period_codes)}
    df_use["build_type_order"] = df_use["build_type_code"].map(build_order)
    df_use["build_period_order"] = df_use["build_period_code"].map(period_order)
    df_use = df_use.sort_values(["area_order", "build_type_order", "build_period_order"]).drop(
        columns=["area_order", "build_type_order", "build_period_order"]
    )

    df_use = df_use.rename(columns={"value": "households"})
    df_use = df_use[
        [
            "area_code",
            "area_name",
            "build_type_code",
            "build_type_name",
            "build_period_code",
            "build_period_name",
            "households",
            "share_percent",
            "total_households",
            "structure_name",
            "age_name",
            "time_name",
        ]
    ]

    # 2区分の建て方分布（建築時期で集約）
    bt = (
        df_use.groupby(["area_code", "area_name", "build_type_code", "build_type_name"], as_index=False)["households"]
        .sum()
        .rename(columns={"households": "total_households"})
    )
    bt_total = bt.groupby(["area_code", "area_name"], as_index=False)["total_households"].sum().rename(columns={"total_households": "area_total"})
    bt = bt.merge(bt_total, on=["area_code", "area_name"], how="left")
    bt["share_percent"] = (bt["total_households"] / bt["area_total"] * 100).round(4)
    bt = _add_area_order(bt)
    build_order = {code: i for i, code in enumerate(build_codes)}
    bt["build_type_order"] = bt["build_type_code"].map(build_order)
    bt = bt.sort_values(["area_order", "build_type_order"]).drop(columns=["area_order", "build_type_order"])
    bt = bt[
        [
            "area_code",
            "area_name",
            "build_type_code",
            "build_type_name",
            "total_households",
            "share_percent",
            "area_total",
        ]
    ]

    _write_class_csv(out_dir / "meta_0004021670_classes.csv", parsed_classes)

    meta_out["0004021670"] = {
        "title": meta_root.get("METADATA_INF", {}).get("TABLE_INF", {}).get("TITLE", {}).get("$"),
        "updated_date": meta_root.get("METADATA_INF", {}).get("TABLE_INF", {}).get("UPDATED_DATE"),
        "age_code": age_code,
        "build_type_codes": build_codes,
        "build_period_codes": build_period_codes,
        "structure_codes": structure_codes,
        "structure_aggregated": structure_agg,
        "area_codes": area_codes,
        "time_code": time_code,
    }
    return df_use, bt, meta_out


def _write_report(out_md: Path, *, meta: dict[str, Any], out_files: list[str]) -> None:
    lines: list[str] = []
    lines.append("# e-Stat 住宅・土地統計調査（家計主65歳以上）の都道府県別分布")
    lines.append("")
    lines.append(f"- 作成日時: {meta['created_at_local']}")
    lines.append("- 出力: outputs/runs/<tag>/ 以下に CSV を保存")
    for name in out_files:
        lines.append(f"  - `{name}`")
    lines.append("")
    lines.append("- 構造は総数（総数がない場合は木造+非木造で合算）")
    lines.append("- 不詳は除外")
    lines.append("- 全国行は e-Stat の全国コード（該当がない場合は都道府県合算）")
    lines.append("- 記述ポリシー: `docs/rules/statistical_reporting_policy.md`")
    lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="e-Stat APIから、家計を主に支える者65歳以上の住宅種別分布と建築時期分布を取得します。"
    )
    parser.add_argument("--app-id", type=str, default="", help="e-Stat API appId（未指定ならESTAT_APP_ID）")
    parser.add_argument("--tag", type=str, default="", help="outputs/runs/<tag> の tag（未指定なら自動生成）")
    args = parser.parse_args()

    app_id = args.app_id.strip() or os.environ.get("ESTAT_APP_ID", "").strip()
    if not app_id:
        print("[error] e-Stat appId is required (--app-id or ESTAT_APP_ID).", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    tag = args.tag.strip() or _default_tag("estat_elderly_housing")
    out_dir = repo_root / "outputs" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_out: dict[str, Any] = {"created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    df_49, meta_out = _prepare_49_2(app_id, out_dir=out_dir, meta_out=meta_out)
    df_47, df_bt, meta_out = _prepare_47_2_1(app_id, out_dir=out_dir, meta_out=meta_out)

    file_49 = "pref_building_type_4cat_65plus_49-2.csv"
    file_47 = "pref_build_period_by_type_65plus_47-2-1.csv"
    file_bt = "pref_building_type_2cat_65plus_from_47-2-1.csv"

    df_49.to_csv(out_dir / file_49, index=False, encoding="utf-8")
    df_47.to_csv(out_dir / file_47, index=False, encoding="utf-8")
    df_bt.to_csv(out_dir / file_bt, index=False, encoding="utf-8")

    (out_dir / "meta_selection.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(out_dir / "report.md", meta=meta_out, out_files=[file_49, file_47, file_bt])

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
