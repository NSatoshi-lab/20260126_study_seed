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

    header = raw.iloc[header_row, :].astype(str)
    col_idx: dict[str, int] = {}
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
    # 「使用していない」以外の合算（= 使用している4区分 + 不明）
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


def _pref_level_from_base(base_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(base_csv)
    pref = (
        df.groupby(["pref_code", "pref_name"], as_index=False)
        .agg(
            bathroom_heater_rate=("bathroom_heater_rate", "first"),
            double_glazing_window_rate=("double_glazing_window_rate", "first"),
            temp_annual_min=("temp_annual_min", "mean"),
            temp_annual_mean=("temp_annual_mean", "mean"),
        )
        .copy()
    )
    pref["heater_pp"] = pd.to_numeric(pref["bathroom_heater_rate"], errors="coerce") * 100.0
    pref["dg_pp"] = pd.to_numeric(pref["double_glazing_window_rate"], errors="coerce") * 100.0
    pref["temp_min"] = pd.to_numeric(pref["temp_annual_min"], errors="coerce")

    pop23 = df.loc[pd.to_numeric(df["year"], errors="coerce") == 2023, ["pref_code", "pref_name", "pop_total"]].copy()
    pop23["pop_total"] = pd.to_numeric(pop23["pop_total"], errors="coerce")
    pop23 = pop23.groupby(["pref_code", "pref_name"], as_index=False).agg(pop_total=("pop_total", "first")).copy()
    pref = pref.merge(pop23, on=["pref_code", "pref_name"], how="left")

    pref["region"] = pref["pref_name"].map(PREF_TO_REGION)
    missing = sorted(set(pref.loc[pref["region"].isna(), "pref_name"].dropna().astype(str).tolist()))
    if missing:
        raise ValueError(f"region mapping missing for: {missing}")

    return pref


@dataclass(frozen=True)
class AssocOut:
    n: int
    pearson_r: float
    pearson_ci: tuple[float, float]
    pearson_p: float
    spearman_rho: float
    spearman_p: float


def _assoc(pref: pd.DataFrame, *, x_col: str, y_col: str) -> AssocOut:
    x = pd.to_numeric(pref[x_col], errors="coerce")
    y = pd.to_numeric(pref[y_col], errors="coerce")
    ok = x.notna() & y.notna()
    x = x.loc[ok].astype(float)
    y = y.loc[ok].astype(float)
    n = int(x.shape[0])
    pear = stats.pearsonr(x.to_numpy(), y.to_numpy())
    pear_ci = _fisher_ci_pearson(float(pear.statistic), n=n)
    spear = stats.spearmanr(x.to_numpy(), y.to_numpy())
    return AssocOut(
        n=n,
        pearson_r=float(pear.statistic),
        pearson_ci=pear_ci,
        pearson_p=float(pear.pvalue),
        spearman_rho=float(spear.statistic),
        spearman_p=float(spear.pvalue),
    )


@dataclass(frozen=True)
class ModelOut:
    n: int
    coef: dict[str, float]
    ci: dict[str, tuple[float, float]]
    p: dict[str, float]
    r2: float


def _ols_cluster(
    pref: pd.DataFrame,
    *,
    y_col: str,
    x_cols: list[str],
    cluster_col: str,
) -> ModelOut:
    d = pref[[y_col, cluster_col, *x_cols]].copy()
    for c in [y_col, *x_cols]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[y_col, cluster_col, *x_cols]).copy()
    n = int(d.shape[0])

    y = d[y_col].astype(float)
    X = sm.add_constant(d[x_cols].astype(float), has_constant="add")
    ols = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col]})
    ci = ols.conf_int()

    coef_out = {k: float(v) for k, v in ols.params.to_dict().items()}
    p_out = {k: float(v) for k, v in ols.pvalues.to_dict().items()}
    ci_out = {k: (float(ci.loc[k].iloc[0]), float(ci.loc[k].iloc[1])) for k in ci.index.tolist()}
    return ModelOut(n=n, coef=coef_out, ci=ci_out, p=p_out, r2=float(ols.rsquared))


def _scatter_pref(
    pref: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    out_png: Path,
    title: str,
    annotate_prefs: list[str],
) -> None:
    d = pref.copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col]).copy()

    x = d[x_col].astype(float).to_numpy()
    y = d[y_col].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    ax.scatter(x, y, s=34, alpha=0.85, edgecolors="none")

    X = sm.add_constant(pd.DataFrame({"central": x}))
    ols = sm.OLS(y, X).fit()
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ys = float(ols.params["const"]) + float(ols.params["central"]) * xs
    ax.plot(xs, ys, linewidth=2)

    for pref_name in annotate_prefs:
        hit = d.loc[d["pref_name"] == pref_name]
        if hit.empty:
            continue
        hx = float(hit[x_col].iloc[0])
        hy = float(hit[y_col].iloc[0])
        ax.scatter([hx], [hy], s=60)
        ax.annotate(pref_name, (hx, hy), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("セントラル暖房システム使用率（%）〔e-Stat地方別値を割当〕")
    ax.set_ylabel("浴室暖房乾燥機設置率（%）")
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
    pref_table: pd.DataFrame,
    assoc: AssocOut,
    m0: ModelOut,
    m1: ModelOut,
    m2: ModelOut,
    fig_scatter: Path,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fig_rel = fig_scatter.relative_to(out_md.parent).as_posix()

    md: list[str] = []
    md.append("# セントラル暖房システム使用率（地方別）と浴室暖房乾燥機設置率の関連（都道府県別; e-Stat×基盤repo）")
    md.append("")
    md.append(f"- 作成日時: {meta['created_at_local']}")
    md.append(f"- 解析ソフト: Python {meta['python_version']}")
    md.append("- 主要パッケージ: " + ", ".join(f"{k} {v}" for k, v in meta.get("versions", {}).items()))
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
    md.append("- 仮説: セントラル暖房システムの使用率が高い地域（地方）ほど、浴室暖房乾燥機の設置率が低い。")
    md.append(
        "- 目的: 地方に集計せず、浴室暖房乾燥機設置率は**都道府県別**に示す（セントラル暖房使用率は地方別値を都道府県へ割り当てる）。"
    )
    md.append("")

    md.append("## 2. データと変数定義")
    md.append("")
    md.append("### 2.1 セントラル暖房システム使用率（e-Stat; 地方別）")
    md.append("")
    md.append("- e-Statの公開表は「地方別（10区分）」であり、都道府県別の使用率は直接得られない。")
    md.append("- そのため本解析では、地方別の使用率を同一地方内の都道府県へ割り当てた（同一地方内では同一値）。")
    md.append("- 使用率（%）は、表の「セントラル暖房システムの使用状況」について、**「使用していない」以外（= 使用している4区分 + 不明）を合算**して作成した。")
    md.append("")

    md.append("### 2.2 浴室暖房乾燥機設置率（基盤repo; 都道府県別）")
    md.append("")
    md.append("- `full_panel.csv` の `bathroom_heater_rate`（2023年住宅・土地統計調査に基づく都道府県値）を使用。")
    md.append("")

    md.append("### 2.3 調整変数（参考）")
    md.append("")
    md.append("- 年平均最低気温（℃）: `temp_annual_min` の2015-2023平均（都道府県別）。")
    md.append("- 複層ガラス化率（%）: `double_glazing_window_rate`（都道府県別）。")
    md.append("")

    md.append("## 3. 方法")
    md.append("")
    md.append("- 解析単位: 都道府県（47）。")
    md.append("- 主解析: `設置率（%） ~ セントラル暖房使用率（%）` のOLS。")
    md.append("- 参考: 年平均最低気温、複層ガラス化率で調整したOLS。")
    md.append("- セントラル暖房使用率は地方別値の割当のため、誤差は地方内で相関する可能性がある。そこで、標準誤差は地方（10クラスタ）でクラスタロバスト推定とした。")
    md.append("")

    md.append("## 4. 結果")
    md.append("")

    md.append("### 4.1 地方別のセントラル暖房使用率（e-Stat）")
    md.append("")
    region_disp = region_table[["region", "central_non_not_use_pp", "central_not_use_pp"]].copy()
    region_disp = region_disp.rename(
        columns={
            "region": "地方",
            "central_non_not_use_pp": "セントラル暖房使用率（%）",
            "central_not_use_pp": "セントラル暖房「使用していない」（%）",
        }
    )
    for c in ["セントラル暖房使用率（%）", "セントラル暖房「使用していない」（%）"]:
        region_disp[c] = pd.to_numeric(region_disp[c], errors="coerce").round(2)
    md.append(_df_to_md_table(region_disp))
    md.append("")

    md.append("### 4.2 都道府県別の値（47）")
    md.append("")
    pref_disp = pref_table[
        [
            "pref_name",
            "region",
            "central_non_not_use_pp",
            "heater_pp",
            "temp_min",
            "dg_pp",
        ]
    ].copy()
    pref_disp = pref_disp.rename(
        columns={
            "pref_name": "都道府県",
            "region": "地方",
            "central_non_not_use_pp": "セントラル暖房使用率（%）",
            "heater_pp": "浴室暖房乾燥機設置率（%）",
            "temp_min": "年平均最低気温（℃）",
            "dg_pp": "複層ガラス化率（%）",
        }
    )
    for c in ["セントラル暖房使用率（%）", "浴室暖房乾燥機設置率（%）", "年平均最低気温（℃）", "複層ガラス化率（%）"]:
        pref_disp[c] = pd.to_numeric(pref_disp[c], errors="coerce").round(2)
    md.append(_df_to_md_table(pref_disp))
    md.append("")

    md.append(f"![figures/scatter_pref_central_heating_pp_vs_bathroom_heater_pp.png](./{fig_rel})")
    md.append("")

    md.append("### 4.3 相関（参考）")
    md.append("")
    md.append(
        f"- Pearson r={assoc.pearson_r:.3f}（95%CI [{assoc.pearson_ci[0]:.3f}, {assoc.pearson_ci[1]:.3f}], {_fmt_p(assoc.pearson_p)}）"
    )
    md.append(f"- Spearman ρ={assoc.spearman_rho:.3f}（{_fmt_p(assoc.spearman_p)}）")
    md.append("")

    md.append("### 4.4 回帰（地方クラスタロバストSE）")
    md.append("")
    md.append(
        f"- M0: 設置率（%） ~ セントラル暖房使用率（%）: 1pp増加あたり {m0.coef['central_non_not_use_pp']:.3f}pp（95%CI [{m0.ci['central_non_not_use_pp'][0]:.3f}, {m0.ci['central_non_not_use_pp'][1]:.3f}], {_fmt_p(m0.p['central_non_not_use_pp'])}）, R²={m0.r2:.3f}"
    )
    md.append(
        f"- M1: + 年平均最低気温（℃）: セントラル暖房 {m1.coef['central_non_not_use_pp']:.3f}pp（95%CI [{m1.ci['central_non_not_use_pp'][0]:.3f}, {m1.ci['central_non_not_use_pp'][1]:.3f}], {_fmt_p(m1.p['central_non_not_use_pp'])}）, R²={m1.r2:.3f}"
    )
    md.append(
        f"- M2: + 年平均最低気温（℃） + 複層ガラス化率（%）: セントラル暖房 {m2.coef['central_non_not_use_pp']:.3f}pp（95%CI [{m2.ci['central_non_not_use_pp'][0]:.3f}, {m2.ci['central_non_not_use_pp'][1]:.3f}], {_fmt_p(m2.p['central_non_not_use_pp'])}）, R²={m2.r2:.3f}"
    )
    md.append("")

    md.append("## 5. 解釈と限界")
    md.append("")
    md.append("- セントラル暖房使用率が都道府県別ではなく地方別のため、同一地方内の都道府県差（例: 青森県と宮城県の差）をこの説明変数では表現できない。")
    md.append("- 地方数が10と小さく、推定は外れ値（例: 北海道・沖縄）に影響されやすい。")
    md.append("- 本解析は観察データの関連であり、因果効果を意味しない。")
    md.append("")
    md.append("---")
    md.append("")
    out_md.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="e-Stat（地方別セントラル暖房）を都道府県へ割り当て、浴室暖房乾燥機設置率（都道府県別）との関連を記述します。"
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

    tag = args.tag.strip() or _default_tag("pref_bathroom_heater_vs_central_heating_region_proxy")
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

    central_region = _extract_central_heating_region_rates(estat_xlsx)
    central_region.to_csv(out_dir / "central_heating_region_rates.csv", index=False, encoding="utf-8")

    pref = _pref_level_from_base(base_csv)
    pref = pref.merge(
        central_region[
            [
                "region",
                "central_non_not_use_pp",
                "central_not_use_pp",
                "central_use_pp",
            ]
        ],
        on="region",
        how="left",
    )
    pref.to_csv(out_dir / "analysis_dataset_pref.csv", index=False, encoding="utf-8")

    assoc = _assoc(pref, x_col="central_non_not_use_pp", y_col="heater_pp")
    m0 = _ols_cluster(pref, y_col="heater_pp", x_cols=["central_non_not_use_pp"], cluster_col="region")
    m1 = _ols_cluster(pref, y_col="heater_pp", x_cols=["central_non_not_use_pp", "temp_min"], cluster_col="region")
    m2 = _ols_cluster(pref, y_col="heater_pp", x_cols=["central_non_not_use_pp", "temp_min", "dg_pp"], cluster_col="region")

    fig_scatter = figs_dir / "scatter_pref_central_heating_pp_vs_bathroom_heater_pp.png"
    annotate = ["北海道", "青森県", "東京都", "沖縄県"]
    _scatter_pref(
        pref,
        x_col="central_non_not_use_pp",
        y_col="heater_pp",
        out_png=fig_scatter,
        title="セントラル暖房使用率（地方別値を割当）と浴室暖房乾燥機設置率（都道府県）",
        annotate_prefs=annotate,
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
    (out_dir / "model_m0.json").write_text(json.dumps(m0.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "model_m1.json").write_text(json.dumps(m1.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "model_m2.json").write_text(json.dumps(m2.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(
        out_dir / "report.md",
        meta=meta,
        region_table=central_region,
        pref_table=pref,
        assoc=assoc,
        m0=m0,
        m1=m1,
        m2=m2,
        fig_scatter=fig_scatter,
    )

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
