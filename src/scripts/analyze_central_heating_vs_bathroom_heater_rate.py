from __future__ import annotations

import argparse
import hashlib
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
import scipy
import statsmodels
import statsmodels.api as sm
from matplotlib import font_manager
from scipy import stats

DEFAULT_BASE_RELATIVE = Path("..") / "入浴統計" / "data" / "processed" / "full_panel.csv"

DEFAULT_ESTAT_STAT_INF_ID = "000040277713"  # co2r5_3-1_00全国.xlsx
DEFAULT_ESTAT_FILE_KIND = 4  # xlsx

REGIONS = ["北海道", "東北", "関東甲信", "北陸", "東海", "近畿", "中国", "四国", "九州", "沖縄"]

# e-Stat「地方別」の区分に合わせる想定（注: 新潟県は北陸に含める想定）
PREF_TO_REGION: dict[str, str] = {
    "北海道": "北海道",
    "青森県": "東北",
    "岩手県": "東北",
    "宮城県": "東北",
    "秋田県": "東北",
    "山形県": "東北",
    "福島県": "東北",
    "茨城県": "関東甲信",
    "栃木県": "関東甲信",
    "群馬県": "関東甲信",
    "埼玉県": "関東甲信",
    "千葉県": "関東甲信",
    "東京都": "関東甲信",
    "神奈川県": "関東甲信",
    "山梨県": "関東甲信",
    "長野県": "関東甲信",
    "新潟県": "北陸",
    "富山県": "北陸",
    "石川県": "北陸",
    "福井県": "北陸",
    "岐阜県": "東海",
    "静岡県": "東海",
    "愛知県": "東海",
    "三重県": "東海",
    "滋賀県": "近畿",
    "京都府": "近畿",
    "大阪府": "近畿",
    "兵庫県": "近畿",
    "奈良県": "近畿",
    "和歌山県": "近畿",
    "鳥取県": "中国",
    "島根県": "中国",
    "岡山県": "中国",
    "広島県": "中国",
    "山口県": "中国",
    "徳島県": "四国",
    "香川県": "四国",
    "愛媛県": "四国",
    "高知県": "四国",
    "福岡県": "九州",
    "佐賀県": "九州",
    "長崎県": "九州",
    "熊本県": "九州",
    "大分県": "九州",
    "宮崎県": "九州",
    "鹿児島県": "九州",
    "沖縄県": "沖縄",
}


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


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.0001:
        return "p<0.0001"
    if p < 0.001:
        return f"p={p:.4f}"
    if p < 0.2:
        return f"p={p:.3f}"
    if p <= 0.99:
        return f"p={p:.2f}"
    return "p>0.99"


def _fisher_ci_pearson(r: float, *, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if not np.isfinite(r):
        return (float("nan"), float("nan"))
    if np.isclose(abs(r), 1.0):
        return (float(r), float(r))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    lo, hi = z - zcrit * se, z + zcrit * se
    return (float(np.tanh(lo)), float(np.tanh(hi)))


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


def _estat_download_url(*, stat_inf_id: str, file_kind: int) -> str:
    return f"https://www.e-stat.go.jp/stat-search/file-download?statInfId={stat_inf_id}&fileKind={file_kind}"


def _download_file(url: str, *, dest: Path, timeout_sec: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=timeout_sec, stream=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)


CENTRAL_HEATING_LABELS = {
    "central_electric_boiler_pp": "使用している（電気温水ボイラ）",
    "central_gas_boiler_pp": "使用している（ガス温水ボイラ）",
    "central_kerosene_boiler_pp": "使用している（灯油温水ボイラ）",
    "central_duct_pp": "使用している（ダクト式セントラル空調）",
    "central_not_use_pp": "使用していない",
    "central_unknown_pp": "不明",
}


def _find_header_row(df: pd.DataFrame, *, must_contain: str) -> int:
    for r in range(df.shape[0]):
        row = df.iloc[r, :].astype(str)
        if (row == must_contain).any():
            return r
    raise ValueError(f"header row not found: {must_contain}")


def _extract_central_heating_region_rates(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, header=None)
    header_row = _find_header_row(raw, must_contain=CENTRAL_HEATING_LABELS["central_electric_boiler_pp"])

    col_idx: dict[str, int] = {}
    header = raw.iloc[header_row, :].astype(str)
    for key, label in CENTRAL_HEATING_LABELS.items():
        hits = np.where(header.to_numpy() == label)[0]
        if len(hits) == 0:
            raise ValueError(f"label not found in header: {label}")
        col_idx[key] = int(hits[0])

    region_rows = raw[(raw.iloc[:, 0] == 2) & (raw.iloc[:, 2].isin(REGIONS))].copy().reset_index(drop=True)
    if region_rows.shape[0] != len(REGIONS):
        found = sorted(set(region_rows.iloc[:, 2].dropna().astype(str).tolist()))
        raise ValueError(f"unexpected region rows: got={region_rows.shape[0]} found={found}")

    out = pd.DataFrame({"region": region_rows.iloc[:, 2].astype(str).tolist()})
    for key, idx in col_idx.items():
        out[key] = (
            pd.to_numeric(region_rows.iloc[:, idx], errors="coerce")
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )

    out["central_use_pp"] = out[
        [
            "central_electric_boiler_pp",
            "central_gas_boiler_pp",
            "central_kerosene_boiler_pp",
            "central_duct_pp",
        ]
    ].sum(axis=1)
    # 「使用していない」以外の合算（= 使用している + 不明）
    out["central_non_not_use_pp"] = out[
        [
            "central_electric_boiler_pp",
            "central_gas_boiler_pp",
            "central_kerosene_boiler_pp",
            "central_duct_pp",
            "central_unknown_pp",
        ]
    ].sum(axis=1)
    out = out.sort_values("region", key=lambda s: s.map({r: i for i, r in enumerate(REGIONS)})).reset_index(drop=True)
    return out


def _region_level_from_base(base_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(base_csv)
    d = df.loc[pd.to_numeric(df["year"], errors="coerce") == 2023].copy()
    d["heater_pp"] = pd.to_numeric(d["bathroom_heater_rate"], errors="coerce") * 100.0
    d["temp_min"] = pd.to_numeric(d["temp_annual_min"], errors="coerce")
    d["pop_total"] = pd.to_numeric(d["pop_total"], errors="coerce")
    d["region"] = d["pref_name"].map(PREF_TO_REGION)

    missing = sorted(set(d.loc[d["region"].isna(), "pref_name"].dropna().astype(str).tolist()))
    if missing:
        raise ValueError(f"region mapping missing for: {missing}")

    # Population-weighted (approx.) region summaries
    w = d.dropna(subset=["region", "heater_pp", "temp_min", "pop_total"]).copy()
    w["heater_w"] = w["heater_pp"] * w["pop_total"]
    w["temp_min_w"] = w["temp_min"] * w["pop_total"]
    g = (
        w.groupby("region", as_index=False)
        .agg(
            pop_total=("pop_total", "sum"),
            heater_w=("heater_w", "sum"),
            temp_min_w=("temp_min_w", "sum"),
        )
        .copy()
    )
    g["bathroom_heater_pp_popw"] = g["heater_w"] / g["pop_total"]
    g["temp_annual_min_popw"] = g["temp_min_w"] / g["pop_total"]

    # Unweighted region means (sensitivity)
    m = (
        d.dropna(subset=["region", "heater_pp"])
        .groupby("region", as_index=False)
        .agg(
            bathroom_heater_pp_unw=("heater_pp", "mean"),
        )
        .copy()
    )

    out = g.merge(m, on="region", how="left")
    out = out.sort_values("region", key=lambda s: s.map({r: i for i, r in enumerate(REGIONS)})).reset_index(drop=True)
    out = out[["region", "pop_total", "bathroom_heater_pp_popw", "bathroom_heater_pp_unw", "temp_annual_min_popw"]].copy()
    return out


@dataclass(frozen=True)
class AssocOut:
    n: int
    pearson_r: float
    pearson_ci: tuple[float, float]
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    ols_slope_pp_per_1pp: float
    ols_ci: tuple[float, float]
    ols_p: float
    ols_r2: float


def _assoc_unadjusted(d: pd.DataFrame, *, x_col: str, y_col: str) -> AssocOut:
    x = pd.to_numeric(d[x_col], errors="coerce")
    y = pd.to_numeric(d[y_col], errors="coerce")
    ok = x.notna() & y.notna()
    x = x.loc[ok].astype(float)
    y = y.loc[ok].astype(float)
    n = int(x.shape[0])

    pear = stats.pearsonr(x.to_numpy(), y.to_numpy())
    pear_ci = _fisher_ci_pearson(float(pear.statistic), n=n)
    spear = stats.spearmanr(x.to_numpy(), y.to_numpy())

    X = sm.add_constant(pd.DataFrame({"central": x.to_numpy()}))
    ols = sm.OLS(y.to_numpy(), X).fit()
    slope = float(ols.params["central"])
    ci = tuple(float(z) for z in ols.conf_int().loc["central"].tolist())

    return AssocOut(
        n=n,
        pearson_r=float(pear.statistic),
        pearson_ci=pear_ci,
        pearson_p=float(pear.pvalue),
        spearman_rho=float(spear.statistic),
        spearman_p=float(spear.pvalue),
        ols_slope_pp_per_1pp=slope,
        ols_ci=ci,
        ols_p=float(ols.pvalues["central"]),
        ols_r2=float(ols.rsquared),
    )


@dataclass(frozen=True)
class AdjOut:
    n: int
    coef_central: float
    coef_temp_min: float
    ci_central: tuple[float, float]
    ci_temp_min: tuple[float, float]
    p_central: float
    p_temp_min: float
    r2: float


def _assoc_adjusted_temp(d: pd.DataFrame, *, x_col: str, y_col: str, temp_col: str) -> AdjOut:
    x = pd.to_numeric(d[x_col], errors="coerce")
    y = pd.to_numeric(d[y_col], errors="coerce")
    t = pd.to_numeric(d[temp_col], errors="coerce")
    ok = x.notna() & y.notna() & t.notna()
    x = x.loc[ok].astype(float)
    y = y.loc[ok].astype(float)
    t = t.loc[ok].astype(float)
    n = int(x.shape[0])

    X = sm.add_constant(pd.DataFrame({"central": x.to_numpy(), "temp_min": t.to_numpy()}))
    ols = sm.OLS(y.to_numpy(), X).fit()
    ci = ols.conf_int()
    return AdjOut(
        n=n,
        coef_central=float(ols.params["central"]),
        coef_temp_min=float(ols.params["temp_min"]),
        ci_central=(float(ci.loc["central"].iloc[0]), float(ci.loc["central"].iloc[1])),
        ci_temp_min=(float(ci.loc["temp_min"].iloc[0]), float(ci.loc["temp_min"].iloc[1])),
        p_central=float(ols.pvalues["central"]),
        p_temp_min=float(ols.pvalues["temp_min"]),
        r2=float(ols.rsquared),
    )


def _scatter_region(
    d: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    out_png: Path,
    title: str,
) -> None:
    x = pd.to_numeric(d[x_col], errors="coerce")
    y = pd.to_numeric(d[y_col], errors="coerce")
    ok = x.notna() & y.notna()
    x = x.loc[ok].astype(float)
    y = y.loc[ok].astype(float)
    labels = d.loc[ok, "region"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    ax.scatter(x.to_numpy(), y.to_numpy(), s=60, alpha=0.85, edgecolors="none")

    X = sm.add_constant(x.to_numpy())
    ols = sm.OLS(y.to_numpy(), X).fit()
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ys = float(ols.params[0]) + float(ols.params[1]) * xs
    ax.plot(xs, ys, linewidth=2)

    for lx, ly, lab in zip(x.to_numpy(), y.to_numpy(), labels, strict=True):
        ax.annotate(lab, (lx, ly), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("セントラル暖房システム使用率（%）")
    ax.set_ylabel("浴室暖房乾燥機設置率（%; 人口加重の地域平均）")
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _write_report(
    out_md: Path,
    *,
    meta: dict[str, Any],
    region_table: pd.DataFrame,
    assoc_all: AssocOut,
    adj_all: AdjOut,
    cold_thr: float,
    cold_regions: list[str],
    cold_table: pd.DataFrame,
    fig_scatter: Path,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fig_rel = fig_scatter.relative_to(out_md.parent).as_posix()

    md: list[str] = []
    md.append("# セントラル暖房システム使用率と浴室暖房乾燥機設置率の関連（地域別; e-Stat×基盤repo）")
    md.append("")
    md.append(f"- 作成日時: {meta['created_at_local']}")
    md.append(f"- 解析ソフト: Python {meta['python_version']}")
    md.append(
        "- 主要パッケージ: "
        + ", ".join(f"{k} {v}" for k, v in meta.get("versions", {}).items())
    )
    md.append(f"- 入力データ: `{meta['base_csv']}`（sha256={meta['base_csv_sha256']}）")
    md.append(
        f"- e-Stat入力（xlsx）: `{meta['estat_xlsx']}`（statInfId={meta['estat_stat_inf_id']}, sha256={meta['estat_xlsx_sha256']}）"
    )
    md.append("- 記述ポリシー: `docs/rules/statistical_reporting_policy.md`")
    md.append("")
    md.append("---")
    md.append("")

    md.append("## 1. 解析目的")
    md.append("")
    md.append("- 仮説: セントラル暖房システムの使用率が高い地域ほど、浴室暖房乾燥機の設置率が低い。")
    md.append("- e-Statの該当統計表は「地方別（10区分）」のため、都道府県別ではなく**地域別の生態学的関連**として検討する。")
    md.append("")

    md.append("## 2. データと変数定義")
    md.append("")
    md.append("### 2.1 セントラル暖房システム使用率（e-Stat）")
    md.append("")
    md.append("- 対象: e-Stat「令和5年度 家庭部門のCO2排出実態統計調査」第3-1表（全国）内の「地方別」集計。")
    md.append("- 使用率（%）は、表の「セントラル暖房システムの使用状況」について、**「使用していない」以外（= 使用している4区分 + 不明）を合算**して作成した。")
    md.append("")

    md.append("### 2.2 浴室暖房乾燥機設置率（基盤repo）")
    md.append("")
    md.append("- `full_panel.csv` の `bathroom_heater_rate`（2023年住宅・土地統計調査に基づく都道府県値）を使用。")
    md.append("- 地域別の設置率（%）は、2023年の `pop_total` による人口加重平均として集計した（住宅数ではなく人口で加重している点は近似）。")
    md.append("")

    md.append("### 2.3 寒冷地（地域）定義")
    md.append("")
    md.append("- 2023年の `temp_annual_min` を人口加重平均して地域代表値を作り、10地域の下位1/3分位点以下を寒冷地とした。")
    md.append(f"- 閾値: 年平均最低気温（人口加重） <= {cold_thr:.2f}℃")
    md.append(f"- 寒冷地（{len(cold_regions)}/10地域）: {', '.join(cold_regions)}")
    md.append("")

    md.append("## 3. 方法")
    md.append("")
    md.append("- 解析単位: 地域（10地域）。")
    md.append("- 主解析: `浴室暖房乾燥機設置率（%） ~ セントラル暖房システム使用率（%）` の単回帰（OLS）。")
    md.append("- 参考: `temp_annual_min`（人口加重）で調整したOLS（小標本のため探索的）。")
    md.append("- 補助指標: Pearson相関（Fisher変換による95%CI）とSpearman順位相関。")
    md.append("")

    md.append("## 4. 結果")
    md.append("")
    md.append("### 4.1 地域別の値")
    md.append("")
    show_cols = [
        "region",
        "central_non_not_use_pp",
        "central_not_use_pp",
        "bathroom_heater_pp_popw",
        "temp_annual_min_popw",
    ]
    disp = region_table[show_cols].copy()
    disp = disp.rename(
        columns={
            "region": "地域",
            "central_non_not_use_pp": "セントラル暖房使用率（%）",
            "central_not_use_pp": "セントラル暖房「使用していない」（%）",
            "bathroom_heater_pp_popw": "浴室暖房乾燥機設置率（%）",
            "temp_annual_min_popw": "年平均最低気温（℃）",
        }
    )
    for c in ["セントラル暖房使用率（%）", "セントラル暖房「使用していない」（%）", "浴室暖房乾燥機設置率（%）", "年平均最低気温（℃）"]:
        disp[c] = pd.to_numeric(disp[c], errors="coerce").round(2)
    md.append(_df_to_md_table(disp))
    md.append("")
    md.append(f"![figures/scatter_central_heating_pp_vs_bathroom_heater_pp.png](./{fig_rel})")
    md.append("")

    md.append("### 4.2 全国（10地域）での関連")
    md.append("")
    md.append(
        f"- Pearson r={assoc_all.pearson_r:.3f}（95%CI [{assoc_all.pearson_ci[0]:.3f}, {assoc_all.pearson_ci[1]:.3f}], {_fmt_p(assoc_all.pearson_p)}）"
    )
    md.append(f"- Spearman ρ={assoc_all.spearman_rho:.3f}（{_fmt_p(assoc_all.spearman_p)}）")
    md.append(
        f"- OLS（非調整）: 1ppの使用率増加あたりの設置率差={assoc_all.ols_slope_pp_per_1pp:.3f}pp（95%CI [{assoc_all.ols_ci[0]:.3f}, {assoc_all.ols_ci[1]:.3f}], {_fmt_p(assoc_all.ols_p)}）, R²={assoc_all.ols_r2:.3f}"
    )
    md.append(
        f"- OLS（参考: 年平均最低気温で調整）: 1ppの使用率増加あたりの設置率差={adj_all.coef_central:.3f}pp（95%CI [{adj_all.ci_central[0]:.3f}, {adj_all.ci_central[1]:.3f}], {_fmt_p(adj_all.p_central)}）, R²={adj_all.r2:.3f}"
    )
    md.append("")

    md.append("### 4.3 寒冷地（3地域）内の比較（記述的）")
    md.append("")
    md.append("- 寒冷地は3地域のため、統計的推測は行わず記述に留める。")
    md.append("")
    cold_disp = cold_table[["region", "central_non_not_use_pp", "bathroom_heater_pp_popw", "temp_annual_min_popw"]].copy()
    cold_disp = cold_disp.rename(
        columns={
            "region": "地域",
            "central_non_not_use_pp": "セントラル暖房使用率（%）",
            "bathroom_heater_pp_popw": "浴室暖房乾燥機設置率（%）",
            "temp_annual_min_popw": "年平均最低気温（℃）",
        }
    )
    for c in ["セントラル暖房使用率（%）", "浴室暖房乾燥機設置率（%）", "年平均最低気温（℃）"]:
        cold_disp[c] = pd.to_numeric(cold_disp[c], errors="coerce").round(2)
    md.append(_df_to_md_table(cold_disp))
    md.append("")

    md.append("## 5. 解釈と限界")
    md.append("")
    md.append("- e-Stat側が地域別集計のため、都道府県別の検証ではない（地域内の異質性を平均化している）。")
    md.append("- 設置率の地域集計は人口加重であり、住宅数等による加重ではない（近似）。")
    md.append("- 地域数が10と小さく、推定は外れ値（例: 北海道・沖縄）に影響されやすい。")
    md.append("- 本解析は観察データの生態学的関連であり、因果効果を意味しない。")
    md.append("")
    md.append("---")
    md.append("")

    out_md.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="e-Stat（家庭部門CO2排出実態統計調査）の地域別セントラル暖房使用率と、浴室暖房乾燥機設置率（基盤repo）との関連を記述します。"
    )
    parser.add_argument("--base-csv", type=str, default="", help="基盤repoの full_panel.csv パス（未指定なら ../入浴統計/... を使用）")
    parser.add_argument("--estat-xlsx", type=str, default="", help="e-Statのxlsxをローカル指定（未指定ならダウンロード）")
    parser.add_argument("--estat-stat-inf-id", type=str, default=DEFAULT_ESTAT_STAT_INF_ID, help="e-StatのstatInfId")
    parser.add_argument("--estat-file-kind", type=int, default=DEFAULT_ESTAT_FILE_KIND, help="e-StatのfileKind（xlsx=4）")
    parser.add_argument("--tag", type=str, default="", help="outputs/runs/<tag> の tag（未指定なら自動生成）")
    parser.add_argument("--out-dir", type=str, default="", help="出力ディレクトリ（未指定なら outputs/runs/<tag>/）")
    args = parser.parse_args()

    _set_japanese_font()

    repo_root = Path(__file__).resolve().parents[2]
    base_csv = Path(args.base_csv).expanduser() if args.base_csv else (repo_root / DEFAULT_BASE_RELATIVE)
    base_csv = base_csv.resolve()

    tag = args.tag.strip() or _default_tag("central_heating_vs_bathroom_heater_rate")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (repo_root / "outputs" / "runs" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    if args.estat_xlsx.strip():
        estat_xlsx = Path(args.estat_xlsx).expanduser().resolve()
    else:
        inputs_dir = out_dir / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        estat_xlsx = inputs_dir / f"estat_{args.estat_stat_inf_id}_fileKind{args.estat_file_kind}.xlsx"
        url = _estat_download_url(stat_inf_id=args.estat_stat_inf_id, file_kind=args.estat_file_kind)
        _download_file(url, dest=estat_xlsx, timeout_sec=60)

    central = _extract_central_heating_region_rates(estat_xlsx)
    central.to_csv(out_dir / "central_heating_region_rates.csv", index=False, encoding="utf-8")

    region = _region_level_from_base(base_csv)
    region.to_csv(out_dir / "region_level_from_base_2023.csv", index=False, encoding="utf-8")

    d = central.merge(region, on="region", how="left")
    d.to_csv(out_dir / "analysis_dataset_region.csv", index=False, encoding="utf-8")

    assoc_all = _assoc_unadjusted(d, x_col="central_non_not_use_pp", y_col="bathroom_heater_pp_popw")
    adj_all = _assoc_adjusted_temp(d, x_col="central_non_not_use_pp", y_col="bathroom_heater_pp_popw", temp_col="temp_annual_min_popw")

    cold_thr = float(pd.to_numeric(d["temp_annual_min_popw"], errors="coerce").quantile(1.0 / 3.0))
    cold_regions = d.loc[pd.to_numeric(d["temp_annual_min_popw"], errors="coerce") <= cold_thr, "region"].astype(str).tolist()
    cold_regions = [r for r in REGIONS if r in set(cold_regions)]
    cold_table = d.loc[d["region"].isin(cold_regions)].copy()

    fig_scatter = figs_dir / "scatter_central_heating_pp_vs_bathroom_heater_pp.png"
    _scatter_region(
        d,
        x_col="central_non_not_use_pp",
        y_col="bathroom_heater_pp_popw",
        out_png=fig_scatter,
        title="セントラル暖房システム使用率（%）と浴室暖房乾燥機設置率（%）の関係（10地域）",
    )

    meta = {
        "created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "base_csv": str(base_csv),
        "base_csv_sha256": _sha256(base_csv),
        "estat_stat_inf_id": str(args.estat_stat_inf_id),
        "estat_file_kind": int(args.estat_file_kind),
        "estat_xlsx": str(estat_xlsx),
        "estat_xlsx_sha256": _sha256(estat_xlsx),
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "requests": requests.__version__,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(
        out_dir / "report.md",
        meta=meta,
        region_table=d,
        assoc_all=assoc_all,
        adj_all=adj_all,
        cold_thr=cold_thr,
        cold_regions=cold_regions,
        cold_table=cold_table,
        fig_scatter=fig_scatter,
    )

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
