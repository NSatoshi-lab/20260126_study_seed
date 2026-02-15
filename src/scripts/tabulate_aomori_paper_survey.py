#!/usr/bin/env python3
"""Tabulate Goshogawara paper survey results for Step 5."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import pandas as pd

try:
    from scipy.stats import chi2_contingency, fisher_exact
except Exception:  # pragma: no cover - optional dependency
    chi2_contingency = None
    fisher_exact = None


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
CONFIDENCE_Z_95 = 1.96
MAIN_ANALYSIS_VALID_THRESHOLD = 80
EXPLORATORY_ANALYSIS_VALID_THRESHOLD = 60
MAIN_MISSING_RATE_THRESHOLD_PCT = 20.0

# Sample-size design constants fixed by Step 5 v2 spec.
SAMPLE_SIZE_DESIGN_P = 0.5
SAMPLE_SIZE_DESIGN_E = 0.12
SAMPLE_SIZE_DESIGN_DEFF = 1.2
SAMPLE_SIZE_DESIGN_INVALID_RATE = 0.15
SAMPLE_SIZE_DESIGN_TARGET_ROUND = 10


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


def wilson_ci(success: int, total: int, z: float = CONFIDENCE_Z_95) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = success / total
    denom = 1.0 + (z * z) / total
    center = (phat + (z * z) / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt(
            (phat * (1.0 - phat) / total) + ((z * z) / (4.0 * total * total))
        )
        / denom
    )
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (lo, hi)


def summarize_q7_q9_main(valid: pd.DataFrame) -> dict:
    target = valid[
        valid["q8_bath_heater_installed"].isin([1, 2])
        & valid["q9_central_heating_use"].isin([1, 2, 3])
    ].copy()
    if target.empty:
        return {
            "target_n": 0,
            "rows": [],
            "test_method": "判定不可",
            "test_p": None,
            "small_expected_cell": None,
        }

    q8_map = {1: "設置あり", 2: "設置なし"}
    q9_map = {1: "24時間使用", 2: "時間限定使用", 3: "不使用"}
    target["q7_main_group"] = target["q8_bath_heater_installed"].map(q8_map)
    target["q9_use_label"] = target["q9_central_heating_use"].map(q9_map)

    table = pd.crosstab(
        target["q7_main_group"], target["q9_use_label"], dropna=False
    ).reindex(index=["設置あり", "設置なし"], columns=["24時間使用", "時間限定使用", "不使用"], fill_value=0)

    rows = []
    for group in table.index:
        row_total = int(table.loc[group].sum())
        for col in table.columns:
            count = int(table.loc[group, col])
            pct = (count / row_total * 100.0) if row_total else 0.0
            ci_lo, ci_hi = wilson_ci(count, row_total)
            rows.append(
                {
                    "group": group,
                    "q9_category": col,
                    "count": count,
                    "row_total": row_total,
                    "pct": round(pct, 2),
                    "ci95_lo_pct": round(ci_lo * 100.0, 2),
                    "ci95_hi_pct": round(ci_hi * 100.0, 2),
                }
            )

    # Test choice follows Step 5 v2 intent:
    # - expected>=5: chi-square
    # - expected<5: exact method where available; otherwise descriptive only.
    test_method = "記述統計のみ"
    test_p = None
    small_expected_cell = None
    if chi2_contingency is not None:
        try:
            _, p_chi, _, expected = chi2_contingency(table.to_numpy())
            small_expected_cell = bool((expected < 5.0).any())
            if not small_expected_cell:
                test_method = "χ2検定"
                test_p = float(p_chi)
            else:
                if fisher_exact is not None:
                    try:
                        fisher_res = fisher_exact(table.to_numpy())
                        # SciPy versions may return tuple or result object.
                        test_p = (
                            float(fisher_res.pvalue)
                            if hasattr(fisher_res, "pvalue")
                            else float(fisher_res[1])
                        )
                        test_method = "Fisher-Freeman-Halton exact"
                    except Exception:
                        test_method = "expected<5: exact法未実装のため記述統計のみ"
                else:
                    test_method = "expected<5: exact法未実装のため記述統計のみ"
        except Exception:
            test_method = "記述統計のみ（検定計算不可）"

    return {
        "target_n": int(len(target)),
        "rows": rows,
        "test_method": test_method,
        "test_p": test_p,
        "small_expected_cell": small_expected_cell,
    }


def analysis_gate(valid_count: int, missing_rate_pct: float) -> str:
    if missing_rate_pct >= MAIN_MISSING_RATE_THRESHOLD_PCT:
        return "記述中心（主要欠損率20%以上）"
    if valid_count >= MAIN_ANALYSIS_VALID_THRESHOLD:
        return "主解析（有効80以上）"
    if valid_count >= EXPLORATORY_ANALYSIS_VALID_THRESHOLD:
        return "探索的解析（有効60-79）"
    return "記述中心（有効60未満）"


def sample_size_requirements(
    e: float,
    p: float = SAMPLE_SIZE_DESIGN_P,
    deff: float = SAMPLE_SIZE_DESIGN_DEFF,
    invalid_rate: float = SAMPLE_SIZE_DESIGN_INVALID_RATE,
    z: float = CONFIDENCE_Z_95,
) -> dict:
    n0 = (z * z * p * (1.0 - p)) / (e * e)
    n0_ceil = int(math.ceil(n0))
    n_valid = int(math.ceil(n0_ceil * deff))
    n_collected = int(math.ceil(n_valid / (1.0 - invalid_rate)))
    n_operational = int(
        math.ceil(n_collected / SAMPLE_SIZE_DESIGN_TARGET_ROUND)
        * SAMPLE_SIZE_DESIGN_TARGET_ROUND
    )
    return {
        "e": e,
        "n0": n0,
        "n0_ceil": n0_ceil,
        "n_valid": n_valid,
        "n_collected": n_collected,
        "n_operational": n_operational,
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

    gate = analysis_gate(valid, missing_rate)
    is_main_analysis = int(gate == "主解析（有効80以上）")
    is_exploratory = int(gate == "探索的解析（有効60-79）")
    is_descriptive_only = int(gate.startswith("記述中心"))
    dominance = reason_dominance(flagged[~flagged["invalid_main3"]])
    q7q9 = summarize_q7_q9_main(flagged[~flagged["invalid_main3"]])

    design_primary = sample_size_requirements(e=SAMPLE_SIZE_DESIGN_E)
    design_sens_tight = sample_size_requirements(e=0.10)
    design_sens_loose = sample_size_requirements(e=0.15)

    qc = pd.DataFrame(
        [
            {"metric": "total_responses", "value": total},
            {"metric": "valid_responses", "value": valid},
            {"metric": "invalid_responses", "value": invalid},
            {"metric": "main_missing_rate_pct", "value": missing_rate},
            {"metric": "analysis_gate_label", "value": gate},
            {"metric": "analysis_gate_main", "value": is_main_analysis},
            {"metric": "analysis_gate_exploratory", "value": is_exploratory},
            {"metric": "analysis_gate_descriptive_only", "value": is_descriptive_only},
            {"metric": "reason_target_n", "value": dominance["reason_target_n"]},
            {"metric": "no_need_pct", "value": dominance["no_need_pct"]},
            {"metric": "barrier_pct", "value": dominance["barrier_pct"]},
            {"metric": "gap_pp", "value": dominance["gap_pp"]},
            {"metric": "dominant_group", "value": dominance["dominant_group"]},
            {"metric": "q7q9_target_n", "value": q7q9["target_n"]},
            {"metric": "q7q9_test_method", "value": q7q9["test_method"]},
            {
                "metric": "q7q9_test_p",
                "value": "" if q7q9["test_p"] is None else round(q7q9["test_p"], 6),
            },
            {
                "metric": "q7q9_small_expected_cell",
                "value": "" if q7q9["small_expected_cell"] is None else int(q7q9["small_expected_cell"]),
            },
            {"metric": "design_n0_e12", "value": round(design_primary["n0"], 2)},
            {"metric": "design_n0_ceil_e12", "value": design_primary["n0_ceil"]},
            {"metric": "design_n_valid_e12", "value": design_primary["n_valid"]},
            {"metric": "design_n_collected_e12", "value": design_primary["n_collected"]},
            {
                "metric": "design_n_operational_e12",
                "value": design_primary["n_operational"],
            },
            {
                "metric": "design_n_collected_e10",
                "value": design_sens_tight["n_collected"],
            },
            {
                "metric": "design_n_collected_e15",
                "value": design_sens_loose["n_collected"],
            },
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
        f"- 解析ゲート: {gate}",
        "",
        "## 目標回答数の計算再現（固定前提）",
        "",
        f"- 入力: E=0.12, p=0.5, deff=1.2, invalid=0.15",
        f"- n0={design_primary['n0']:.2f} -> {design_primary['n0_ceil']}",
        f"- n_valid={design_primary['n_valid']}",
        f"- n_collected={design_primary['n_collected']}",
        f"- n_operational={design_primary['n_operational']}",
        "",
        "## 感度テスト（許容誤差E）",
        "",
        f"- E=0.10: n_collected={design_sens_tight['n_collected']}",
        f"- E=0.15: n_collected={design_sens_loose['n_collected']}",
        "",
        "## Q7-Q9 主解析（設置あり/なし × セントラル暖房使用）",
        "",
        f"- 主解析対象票数: {q7q9['target_n']}",
        f"- 検定法: {q7q9['test_method']}",
        (
            "- p値: 記述統計のみ"
            if q7q9["test_p"] is None
            else f"- p値: {q7q9['test_p']:.6f}"
        ),
        "- 行割合は95%CI（Wilson）を併記",
    ]
    if q7q9["rows"]:
        report_lines.extend(
            [
                "",
                "| 群 | Q9カテゴリ | n | 分母 | 割合(%) | 95%CI(%) |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in q7q9["rows"]:
            report_lines.append(
                "| "
                + f"{row['group']} | {row['q9_category']} | {row['count']} | {row['row_total']} | "
                + f"{row['pct']:.2f} | [{row['ci95_lo_pct']:.2f}, {row['ci95_hi_pct']:.2f}] |"
            )
    report_lines.extend(
        [
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
    )
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
