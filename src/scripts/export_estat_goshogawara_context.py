from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    metric_name: str
    stats_data_id: str
    cat01_code: str


CITY_AREA_CODES = ["02201", "02205"]  # 青森市, 五所川原市
CITY_AREA_LABELS = {
    "02201": "青森市",
    "02205": "五所川原市",
}
PREF_AREA_CODE = "02000"  # 青森県
CITY_TIME_CODE = "2020100000"
CLIMATE_TIME_CODE = "2023100000"
TAB_CODE = "00001"

CITY_METRICS: list[MetricSpec] = [
    MetricSpec("pop_total", "総人口（人）", "0000020101", "A1101"),
    MetricSpec("pop_density_habitable", "可住地面積1km2当たり人口密度", "0000020301", "#A01202"),
    MetricSpec("age_0_14_share", "15歳未満人口割合（%）", "0000020301", "#A03504"),
    MetricSpec("age_15_64_share", "15-64歳人口割合（%）", "0000020301", "#A03505"),
    MetricSpec("age_65_plus_share", "65歳以上人口割合（%）", "0000020301", "#A03506"),
    MetricSpec("pop_change_rate", "人口増減率（%）", "0000020301", "#A05101"),
    MetricSpec("single_65_plus_household_share", "65歳以上世帯員の単独世帯割合（%）", "0000020301", "#A06304"),
    MetricSpec("general_hospitals_per_100k", "一般病院数（人口10万人当たり）", "0000020309", "#I0910103"),
    MetricSpec("general_clinics_per_100k", "一般診療所数（人口10万人当たり）", "0000020309", "#I0910105"),
]

CLIMATE_METRICS: list[MetricSpec] = [
    MetricSpec("annual_mean_temp_pref", "年平均気温（℃）", "0000010202", "#B02101"),
    MetricSpec("annual_min_temp_pref", "最低気温（日最低気温の月平均の最低値）（℃）", "0000010202", "#B02103"),
]


def _slugify_ascii(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "run"


def _default_tag(slug: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_slugify_ascii(slug)}"


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if s in ("", "-", "…"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _estat_get_json(endpoint: str, params: dict[str, Any], *, allowed_status: tuple[int | str | None, ...]) -> dict[str, Any]:
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
    if status not in allowed_status:
        raise RuntimeError(f"e-Stat API error ({endpoint}): {result}")
    return root


def _parse_classes(meta_root: dict[str, Any]) -> list[dict[str, Any]]:
    classes = _ensure_list(meta_root.get("METADATA_INF", {}).get("CLASS_INF", {}).get("CLASS_OBJ"))
    out: list[dict[str, Any]] = []
    for cls in classes:
        cid = cls.get("@id")
        cname = cls.get("@name")
        items = _ensure_list(cls.get("CLASS"))
        parsed_items = []
        for item in items:
            parsed_items.append(
                {
                    "code": item.get("@code"),
                    "name": item.get("@name"),
                    "level": item.get("@level"),
                    "parent_code": item.get("@parentCode"),
                    "unit": item.get("@unit"),
                }
            )
        out.append({"id": cid, "name": cname, "classes": parsed_items})
    return out


def _class_lookup(parsed_classes: list[dict[str, Any]], class_id: str) -> dict[str, dict[str, Any]]:
    for cls in parsed_classes:
        if cls["id"] == class_id:
            return {item["code"]: item for item in cls["classes"]}
    return {}


def _build_params(app_id: str, stats_data_id: str, code_map: dict[str, list[str]]) -> dict[str, Any]:
    params: dict[str, Any] = {"appId": app_id, "statsDataId": stats_data_id, "lang": "J"}
    for class_id, codes in code_map.items():
        if not codes:
            continue
        params["cd" + class_id[:1].upper() + class_id[1:]] = ",".join(codes)
    return params


def _fetch_stats_data(
    app_id: str,
    stats_data_id: str,
    code_map: dict[str, list[str]],
    *,
    allow_status_1_empty: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    params = _build_params(app_id, stats_data_id, code_map)
    params["limit"] = 10000
    params["startPosition"] = 1

    allowed = (0, "0", None)
    if allow_status_1_empty:
        allowed = (0, "0", 1, "1", None)

    all_values: list[dict[str, Any]] = []
    first_result: dict[str, Any] = {}
    first_result_inf: dict[str, Any] = {}

    while True:
        root = _estat_get_json("getStatsData", params, allowed_status=allowed)
        result = root.get("RESULT", {})
        result_inf = root.get("STATISTICAL_DATA", {}).get("RESULT_INF", {})
        if not first_result:
            first_result = result
            first_result_inf = result_inf

        status = result.get("STATUS")
        if status in (1, "1"):
            return [], first_result, first_result_inf

        values = _ensure_list(root.get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE"))
        all_values.extend(values)

        total = result_inf.get("TOTAL_NUMBER")
        if total is None:
            break
        total_num = int(total)
        if len(all_values) >= total_num:
            break
        params["startPosition"] = len(all_values) + 1

    return all_values, first_result, first_result_inf


def _values_to_df(values: Iterable[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value in values:
        row: dict[str, Any] = {}
        for key, val in value.items():
            if key == "$":
                row["value_raw"] = val
                row["value"] = _to_float(val)
            elif key.startswith("@"):
                row[key[1:]] = val
            else:
                row[key] = val
        rows.append(row)
    return pd.DataFrame(rows)


def _fetch_city_metrics(app_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric_by_stats: dict[str, list[MetricSpec]] = {}
    for m in CITY_METRICS:
        metric_by_stats.setdefault(m.stats_data_id, []).append(m)

    out_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for stats_data_id, metrics in metric_by_stats.items():
        meta_root = _estat_get_json(
            "getMetaInfo",
            {"appId": app_id, "statsDataId": stats_data_id, "lang": "J"},
            allowed_status=(0, "0", None),
        )
        parsed_classes = _parse_classes(meta_root)
        area_map = _class_lookup(parsed_classes, "area")
        time_map = _class_lookup(parsed_classes, "time")
        cat_map = _class_lookup(parsed_classes, "cat01")

        code_map = {
            "area": CITY_AREA_CODES,
            "time": [CITY_TIME_CODE],
            "cat01": [m.cat01_code for m in metrics],
            "tab": [TAB_CODE],
        }
        values, result, result_inf = _fetch_stats_data(app_id, stats_data_id, code_map)
        df = _values_to_df(values)
        if df.empty:
            raise RuntimeError(f"No data fetched: statsDataId={stats_data_id}")

        for metric in metrics:
            d = df.loc[(df["cat01"] == metric.cat01_code) & (df["area"].isin(CITY_AREA_CODES))].copy()
            if d.shape[0] != len(CITY_AREA_CODES):
                raise RuntimeError(
                    f"Incomplete city rows for metric={metric.metric_id}, statsDataId={stats_data_id}: {d.shape[0]}"
                )

            for _, row in d.iterrows():
                area_code = str(row["area"])
                cat_code = str(row["cat01"])
                time_code = str(row["time"])
                area_name = area_map.get(area_code, {}).get("name", area_code)
                cat_name = cat_map.get(cat_code, {}).get("name", cat_code)
                unit = cat_map.get(cat_code, {}).get("unit")
                time_name = time_map.get(time_code, {}).get("name", time_code)
                out_rows.append(
                    {
                        "metric_id": metric.metric_id,
                        "metric_name": metric.metric_name,
                        "stats_data_id": metric.stats_data_id,
                        "cat01_code": metric.cat01_code,
                        "cat01_name": cat_name,
                        "unit": unit,
                        "area_code": area_code,
                        "area_name": area_name,
                        "area_display_name": CITY_AREA_LABELS.get(area_code, area_name),
                        "time_code": time_code,
                        "time_name": time_name,
                        "value_raw": row.get("value_raw"),
                        "value": row.get("value"),
                    }
                )

            meta_rows.append(
                {
                    "stats_data_id": stats_data_id,
                    "metric_id": metric.metric_id,
                    "metric_name": metric.metric_name,
                    "cat01_code": metric.cat01_code,
                    "time_code": CITY_TIME_CODE,
                    "query_result": result,
                    "query_result_inf": result_inf,
                }
            )

    city_long = pd.DataFrame(out_rows)
    order = {m.metric_id: i for i, m in enumerate(CITY_METRICS)}
    city_long["metric_order"] = city_long["metric_id"].map(order)
    city_long = city_long.sort_values(["metric_order", "area_code"]).drop(columns=["metric_order"]).reset_index(drop=True)
    return city_long, {"city_metric_queries": meta_rows}


def _build_city_wide(city_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric_id, d in city_long.groupby("metric_id", sort=False):
        d = d.copy()
        aomori = d.loc[d["area_code"] == "02201"]
        goshi = d.loc[d["area_code"] == "02205"]
        if aomori.empty or goshi.empty:
            raise RuntimeError(f"Missing area in wide conversion: metric_id={metric_id}")

        a = aomori.iloc[0]
        g = goshi.iloc[0]
        a_val = _to_float(a["value"])
        g_val = _to_float(g["value"])
        diff = None
        if a_val is not None and g_val is not None:
            diff = g_val - a_val

        rows.append(
            {
                "metric_id": metric_id,
                "metric_name": a["metric_name"],
                "stats_data_id": a["stats_data_id"],
                "cat01_code": a["cat01_code"],
                "cat01_name": a["cat01_name"],
                "unit": a["unit"],
                "time_code": a["time_code"],
                "time_name": a["time_name"],
                "aomori_area_code": a["area_code"],
                "aomori_area_name": a["area_display_name"],
                "aomori_value": a_val,
                "goshogawara_area_code": g["area_code"],
                "goshogawara_area_name": g["area_display_name"],
                "goshogawara_value": g_val,
                "diff_goshogawara_minus_aomori": diff,
            }
        )
    wide = pd.DataFrame(rows)
    order = {m.metric_id: i for i, m in enumerate(CITY_METRICS)}
    wide["metric_order"] = wide["metric_id"].map(order)
    wide = wide.sort_values("metric_order").drop(columns=["metric_order"]).reset_index(drop=True)
    return wide


def _fetch_pref_climate(app_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta_root = _estat_get_json(
        "getMetaInfo",
        {"appId": app_id, "statsDataId": "0000010202", "lang": "J"},
        allowed_status=(0, "0", None),
    )
    parsed_classes = _parse_classes(meta_root)
    area_map = _class_lookup(parsed_classes, "area")
    time_map = _class_lookup(parsed_classes, "time")
    cat_map = _class_lookup(parsed_classes, "cat01")

    code_map = {
        "area": [PREF_AREA_CODE],
        "time": [CLIMATE_TIME_CODE],
        "cat01": [m.cat01_code for m in CLIMATE_METRICS],
        "tab": [TAB_CODE],
    }
    values, result, result_inf = _fetch_stats_data(app_id, "0000010202", code_map)
    df = _values_to_df(values)
    if df.shape[0] != len(CLIMATE_METRICS):
        raise RuntimeError(f"Unexpected pref climate rows: expected={len(CLIMATE_METRICS)} got={df.shape[0]}")

    metric_map = {m.cat01_code: m for m in CLIMATE_METRICS}
    out_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cat_code = str(row["cat01"])
        spec = metric_map[cat_code]
        area_code = str(row["area"])
        time_code = str(row["time"])
        out_rows.append(
            {
                "metric_id": spec.metric_id,
                "metric_name": spec.metric_name,
                "stats_data_id": "0000010202",
                "cat01_code": cat_code,
                "cat01_name": cat_map.get(cat_code, {}).get("name", cat_code),
                "unit": cat_map.get(cat_code, {}).get("unit"),
                "area_code": area_code,
                "area_name": area_map.get(area_code, {}).get("name", area_code),
                "time_code": time_code,
                "time_name": time_map.get(time_code, {}).get("name", time_code),
                "value_raw": row.get("value_raw"),
                "value": row.get("value"),
            }
        )

    out = pd.DataFrame(out_rows)
    order = {m.metric_id: i for i, m in enumerate(CLIMATE_METRICS)}
    out["metric_order"] = out["metric_id"].map(order)
    out = out.sort_values("metric_order").drop(columns=["metric_order"]).reset_index(drop=True)
    return out, {"pref_climate_query_result": result, "pref_climate_query_result_inf": result_inf}


def _check_city_climate_unavailable(app_id: str) -> dict[str, Any]:
    code_map = {
        "area": CITY_AREA_CODES,
        "time": [CLIMATE_TIME_CODE],
        "cat01": [m.cat01_code for m in CLIMATE_METRICS],
        "tab": [TAB_CODE],
    }
    values, result, result_inf = _fetch_stats_data(
        app_id,
        "0000010202",
        code_map,
        allow_status_1_empty=True,
    )
    status = result.get("STATUS")
    return {
        "stats_data_id": "0000010202",
        "query_code_map": code_map,
        "result": result,
        "result_inf": result_inf,
        "expected_status": 1,
        "actual_status": status,
        "status_matches_expected": str(status) == "1",
        "returned_value_count": len(values),
    }


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


def _write_report(out_md: Path, *, meta: dict[str, Any], city_wide: pd.DataFrame, pref_climate: pd.DataFrame, out_files: list[str]) -> None:
    city_table = city_wide[
        [
            "metric_name",
            "aomori_value",
            "goshogawara_value",
            "diff_goshogawara_minus_aomori",
            "time_name",
        ]
    ].copy()
    city_table = city_table.rename(
        columns={
            "metric_name": "指標",
            "aomori_value": "青森市",
            "goshogawara_value": "五所川原市",
            "diff_goshogawara_minus_aomori": "差（五所川原市-青森市）",
            "time_name": "年",
        }
    )

    climate_table = pref_climate[["metric_name", "value", "time_name"]].copy()
    climate_table = climate_table.rename(
        columns={
            "metric_name": "指標（青森県値）",
            "value": "値",
            "time_name": "年",
        }
    )

    lines: list[str] = []
    lines.append("# e-Stat 五所川原市調査向け背景指標（青森市比較）")
    lines.append("")
    lines.append(f"- 作成日時: {meta['created_at_local']}")
    lines.append("- 出力先: `outputs/runs/<tag>/`")
    for name in out_files:
        lines.append(f"  - `{name}`")
    lines.append("")
    lines.append("## 市区町村比較（2020年度）")
    lines.append("")
    lines.append(_df_to_md_table(city_table))
    lines.append("")
    lines.append("## 青森県気温（2023年度; 県値のみ）")
    lines.append("")
    lines.append(_df_to_md_table(climate_table))
    lines.append("")
    lines.append("## 気温の市区町村比較可否")
    lines.append("")
    ck = meta["climate_city_constraint_check"]
    lines.append(
        f"- 検証クエリ結果: STATUS={ck['actual_status']}（期待: 1）, "
        f"status_matches_expected={ck['status_matches_expected']}, "
        f"returned_value_count={ck['returned_value_count']}"
    )
    lines.append("- 解釈: e-Statの当該系列では青森市・五所川原市の気温値が返らないため、気温は青森県値を併記。")
    lines.append("")
    lines.append("- 記述ポリシー: `docs/rules/statistical_reporting_policy.md`")
    lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="e-Stat APIから五所川原市調査向け背景指標（青森市比較）を取得します。"
    )
    parser.add_argument("--app-id", type=str, default="", help="e-Stat API appId（未指定ならESTAT_APP_ID）")
    parser.add_argument("--tag", type=str, default="", help="outputs/runs/<tag> の tag（未指定なら自動生成）")
    args = parser.parse_args()

    app_id = args.app_id.strip() or os.environ.get("ESTAT_APP_ID", "").strip()
    if not app_id:
        print("[error] e-Stat appId is required (--app-id or ESTAT_APP_ID).", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    tag = args.tag.strip() or _default_tag("estat_goshogawara_context")
    out_dir = repo_root / "outputs" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    city_long, city_meta = _fetch_city_metrics(app_id)
    city_wide = _build_city_wide(city_long)
    pref_climate, climate_meta = _fetch_pref_climate(app_id)
    climate_city_constraint = _check_city_climate_unavailable(app_id)

    city_long_file = "city_metrics_long.csv"
    city_wide_file = "city_metrics_wide.csv"
    pref_climate_file = "pref_climate_2023.csv"
    meta_file = "meta_selection.json"
    report_file = "report.md"

    city_long.to_csv(out_dir / city_long_file, index=False, encoding="utf-8")
    city_wide.to_csv(out_dir / city_wide_file, index=False, encoding="utf-8")
    pref_climate.to_csv(out_dir / pref_climate_file, index=False, encoding="utf-8")

    meta_out = {
        "created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "city_area_codes": CITY_AREA_CODES,
        "city_area_labels": CITY_AREA_LABELS,
        "pref_area_code": PREF_AREA_CODE,
        "city_time_code": CITY_TIME_CODE,
        "climate_time_code": CLIMATE_TIME_CODE,
        "city_metrics": [m.__dict__ for m in CITY_METRICS],
        "climate_metrics": [m.__dict__ for m in CLIMATE_METRICS],
        "city_query_meta": city_meta,
        "climate_query_meta": climate_meta,
        "climate_city_constraint_check": climate_city_constraint,
    }
    (out_dir / meta_file).write_text(json.dumps(meta_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(
        out_dir / report_file,
        meta=meta_out,
        city_wide=city_wide,
        pref_climate=pref_climate,
        out_files=[city_long_file, city_wide_file, pref_climate_file, meta_file],
    )

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
