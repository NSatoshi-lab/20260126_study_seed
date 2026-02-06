#!/usr/bin/env python3
"""Tabulate Goshogawara paper survey results for Step 5."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


EXPECTED_COLUMNS = [
    "response_id",
    "q1_age_group",
    "q2_residence_area",
    "q3_housing_type",
    "q4_building_age_band",
    "q5_tenure",
    "q6_window_insulation_proxy",
    "q7_winter_home_bath_freq",
    "q8_bath_heater_installed",
    "q9_central_heating_use",
    "q10_alt_heating_types",
    "q10_alt_heating_other_text",
    "q11_bath_heater_heating_winter_use",
    "q12_preheat_before_bath",
    "q13a_bathroom_cold_7pt",
    "q13b_dressingroom_cold_7pt",
    "q14_reason_codes",
    "q14_reason_other_text",
]

NUMERIC_COLUMNS = [
    "q1_age_group",
    "q2_residence_area",
    "q3_housing_type",
    "q4_building_age_band",
    "q5_tenure",
    "q6_window_insulation_proxy",
    "q7_winter_home_bath_freq",
    "q8_bath_heater_installed",
    "q9_central_heating_use",
    "q11_bath_heater_heating_winter_use",
    "q12_preheat_before_bath",
    "q13a_bathroom_cold_7pt",
    "q13b_dressingroom_cold_7pt",
]

NO_NEED_REASON_CODES = {1, 6}
BARRIER_REASON_CODES = {2, 3, 4, 5, 7, 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate required tabulations for Aomori paper survey."
    )
    parser.add_argument("--input-csv", required=True, help="Path to input CSV")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    return parser.parse_args()


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    raise RuntimeError(f"Failed to read CSV with fallback encodings: {last_error}")


def ensure_columns(df: pd.DataFrame, expected: Iterable[str]) -> None:
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input CSV: {missing}")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    ensure_columns(d, EXPECTED_COLUMNS)
    d = d[EXPECTED_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        d[col] = pd.to_numeric(d[col], errors="coerce").astype("Int64")
    d["q10_alt_heating_types"] = (
        d["q10_alt_heating_types"].fillna("").astype(str).str.strip()
    )
    d["q10_alt_heating_other_text"] = (
        d["q10_alt_heating_other_text"].fillna("").astype(str).str.strip()
    )
    d["q14_reason_codes"] = d["q14_reason_codes"].fillna("").astype(str).str.strip()
    d["q14_reason_other_text"] = (
        d["q14_reason_other_text"].fillna("").astype(str).str.strip()
    )
    d["response_id"] = d["response_id"].fillna("").astype(str).str.strip()
    return d


def is_missing_numeric(series: pd.Series) -> pd.Series:
    return series.isna() | series.eq(99)


def derive_validity_flags(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    q8 = out["q8_bath_heater_installed"]
    q11 = out["q11_bath_heater_heating_winter_use"]
    q14 = out["q14_reason_codes"]

    miss_q8 = is_missing_numeric(q8)
    need_q11 = q8.eq(1)
    miss_q11 = need_q11 & is_missing_numeric(q11)

    need_q14 = q8.isin([2, 3]) | q11.isin([4, 5])
    miss_q14 = need_q14 & q14.eq("")

    out["need_q11"] = need_q11
    out["need_q14"] = need_q14
    out["missing_q8"] = miss_q8
    out["missing_q11"] = miss_q11
    out["missing_q14"] = miss_q14
    out["invalid_main3"] = miss_q8 | miss_q11 | miss_q14
    return out


def add_label_columns(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["q8_label"] = out["q8_bath_heater_installed"].map(
        {1: "設置あり", 2: "設置なし", 3: "不明", 99: "無回答"}
    ).fillna("無回答")
    out["q9_label"] = out["q9_central_heating_use"].map(
        {1: "使用あり", 2: "使用なし", 3: "不明", 99: "無回答"}
    ).fillna("無回答")
    out["q13a_label"] = out["q13a_bathroom_cold_7pt"].map(
        {
            1: "1_非常に暖かい",
            2: "2_暖かい",
            3: "3_やや暖かい",
            4: "4_どちらでもない",
            5: "5_やや寒い",
            6: "6_寒い",
            7: "7_非常に寒い",
            99: "99_無回答",
        }
    ).fillna("99_無回答")
    out["q3_label"] = out["q3_housing_type"].map(
        {1: "一戸建て", 2: "集合住宅", 3: "その他", 99: "無回答"}
    ).fillna("無回答")
    out["bathroom_cold_binary"] = out["q13a_bathroom_cold_7pt"].apply(label_cold_binary)
    return out


def label_cold_binary(value: object) -> str:
    if pd.isna(value):
        return "無回答"
    n = int(value)
    if 5 <= n <= 7:
        return "寒い(5-7)"
    if 1 <= n <= 4:
        return "寒くない/中立(1-4)"
    return "無回答"


def parse_reason_codes(value: str) -> List[int]:
    if not value:
        return []
    codes: List[int] = []
    for token in value.split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            codes.append(int(token))
        except ValueError:
            continue
    return codes


def reason_dominance(valid: pd.DataFrame) -> dict:
    target = valid[valid["need_q14"]].copy()
    if target.empty:
        return {
            "reason_target_n": 0,
            "no_need_pct": 0.0,
            "barrier_pct": 0.0,
            "gap_pp": 0.0,
            "dominant_group": "判定不可",
        }

    code_lists = target["q14_reason_codes"].apply(parse_reason_codes)
    has_no_need = code_lists.apply(
        lambda xs: any(code in NO_NEED_REASON_CODES for code in xs)
    )
    has_barrier = code_lists.apply(
        lambda xs: any(code in BARRIER_REASON_CODES for code in xs)
    )

    no_need_pct = float(has_no_need.mean() * 100.0)
    barrier_pct = float(has_barrier.mean() * 100.0)
    gap_pp = abs(no_need_pct - barrier_pct)

    if gap_pp >= 10.0:
        dominant_group = "不要群優勢" if no_need_pct > barrier_pct else "障壁群優勢"
    else:
        dominant_group = "拮抗(差10pp未満)"

    return {
        "reason_target_n": int(len(target)),
        "no_need_pct": round(no_need_pct, 2),
        "barrier_pct": round(barrier_pct, 2),
        "gap_pp": round(gap_pp, 2),
        "dominant_group": dominant_group,
    }


def save_crosstabs(valid: pd.DataFrame, output_dir: Path) -> None:
    table1 = pd.crosstab(valid["q8_label"], valid["q13a_label"], dropna=False)
    table1.to_csv(output_dir / "table1_install_x_bathroom_cold_7pt.csv", encoding="utf-8-sig")

    table2 = pd.crosstab(valid["q8_label"], valid["q9_label"], dropna=False)
    table2.to_csv(output_dir / "table2_install_x_central_heating.csv", encoding="utf-8-sig")

    reasons = valid[valid["q14_reason_codes"].ne("")][["q14_reason_codes", "q3_label"]].copy()
    if reasons.empty:
        table3 = pd.DataFrame(columns=["reason_code", "housing_type", "count"])
    else:
        reasons["reason_code_list"] = reasons["q14_reason_codes"].apply(parse_reason_codes)
        exploded = reasons.explode("reason_code_list")
        exploded = exploded[exploded["reason_code_list"].notna()].copy()
        exploded["reason_code"] = exploded["reason_code_list"].astype(int)
        table3 = pd.crosstab(exploded["reason_code"], exploded["q3_label"], dropna=False)
    table3.to_csv(output_dir / "table3_reason_x_housing_type.csv", encoding="utf-8-sig")


def save_qc_and_report(flagged: pd.DataFrame, output_dir: Path) -> None:
    total = int(len(flagged))
    invalid = int(flagged["invalid_main3"].sum())
    valid = total - invalid
    missing_rate = round((invalid / total * 100.0), 2) if total else 0.0

    quality_ok = valid >= 30 and missing_rate < 20.0
    dominance = reason_dominance(flagged[~flagged["invalid_main3"]])

    qc = pd.DataFrame(
        [
            {"metric": "total_responses", "value": total},
            {"metric": "valid_responses", "value": valid},
            {"metric": "invalid_responses", "value": invalid},
            {"metric": "main_missing_rate_pct", "value": missing_rate},
            {"metric": "quality_gate_valid_ge_30_and_missing_lt_20", "value": int(quality_ok)},
            {"metric": "reason_target_n", "value": dominance["reason_target_n"]},
            {"metric": "no_need_pct", "value": dominance["no_need_pct"]},
            {"metric": "barrier_pct", "value": dominance["barrier_pct"]},
            {"metric": "gap_pp", "value": dominance["gap_pp"]},
            {"metric": "dominant_group", "value": dominance["dominant_group"]},
        ]
    )
    qc.to_csv(output_dir / "qc_summary.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "# 五所川原市 紙アンケート集計レポート",
        "",
        "- 目的: Step 5必須3表の出力と品質ゲート確認",
        f"- 総票数: {total}",
        f"- 有効票数: {valid}",
        f"- 無効票数: {invalid}",
        f"- 主要欠損率: {missing_rate}%",
        f"- 次段階ゲート（有効30以上かつ欠損20%未満）: {'PASS' if quality_ok else 'FAIL'}",
        "",
        "## 優勢判定（不要群 vs 障壁群）",
        "",
        f"- 判定対象票数: {dominance['reason_target_n']}",
        f"- 不要群割合: {dominance['no_need_pct']}%",
        f"- 障壁群割合: {dominance['barrier_pct']}%",
        f"- 差: {dominance['gap_pp']}pp",
        f"- 判定: {dominance['dominant_group']}",
        "",
        "## 出力ファイル",
        "",
        "- `qc_summary.csv`",
        "- `table1_install_x_bathroom_cold_7pt.csv`",
        "- `table2_install_x_central_heating.csv`",
        "- `table3_reason_x_housing_type.csv`",
    ]
    (output_dir / "tabulation_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_csv_with_fallback(input_csv)
    normalized = normalize_dataframe(raw)
    flagged = derive_validity_flags(normalized)
    labeled = add_label_columns(flagged)
    valid = labeled[~labeled["invalid_main3"]].copy()

    save_crosstabs(valid, output_dir)
    save_qc_and_report(labeled, output_dir)


if __name__ == "__main__":
    main()
