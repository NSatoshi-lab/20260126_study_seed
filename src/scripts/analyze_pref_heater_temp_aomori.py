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
import scipy
import statsmodels
import statsmodels.api as sm
from matplotlib import font_manager
from scipy import stats


DEFAULT_BASE_RELATIVE = Path("..") / "入浴統計" / "data" / "processed" / "full_panel.csv"


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


def _fmt_pp(x: float) -> str:
    return f"{x:.2f}"


def _fmt_c(x: float) -> str:
    return f"{x:.2f}"


def _fmt_r(x: float) -> str:
    return f"{x:.3f}"


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


@dataclass(frozen=True)
class AssocOut:
    n: int
    pearson_r: float
    pearson_ci: tuple[float, float]
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    ols_slope_pp_per_1c: float
    ols_ci: tuple[float, float]
    ols_p: float
    ols_r2: float


def _assoc(pref: pd.DataFrame, *, temp_col: str) -> AssocOut:
    d = pref[["bathroom_heater_rate", temp_col]].copy()
    d["heater_pp"] = pd.to_numeric(d["bathroom_heater_rate"], errors="coerce") * 100.0
    d["temp"] = pd.to_numeric(d[temp_col], errors="coerce")
    d = d.dropna(subset=["heater_pp", "temp"]).copy()
    n = int(d.shape[0])

    pear = stats.pearsonr(d["temp"], d["heater_pp"])
    pear_ci = _fisher_ci_pearson(float(pear.statistic), n=n)
    spear = stats.spearmanr(d["temp"], d["heater_pp"])

    X = sm.add_constant(d["temp"].astype(float))
    ols = sm.OLS(d["heater_pp"].astype(float), X).fit()
    slope = float(ols.params["temp"])
    ci = tuple(float(x) for x in ols.conf_int().loc["temp"].tolist())

    return AssocOut(
        n=n,
        pearson_r=float(pear.statistic),
        pearson_ci=pear_ci,
        pearson_p=float(pear.pvalue),
        spearman_rho=float(spear.statistic),
        spearman_p=float(spear.pvalue),
        ols_slope_pp_per_1c=slope,
        ols_ci=ci,
        ols_p=float(ols.pvalues["temp"]),
        ols_r2=float(ols.rsquared),
    )


def _pref_level_mean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["pref_code", "pref_name"], as_index=False)
        .agg(
            bathroom_heater_rate=("bathroom_heater_rate", "first"),
            temp_annual_min=("temp_annual_min", "mean"),
            temp_annual_mean=("temp_annual_mean", "mean"),
        )
        .copy()
    )


def _pref_level_2023(df: pd.DataFrame) -> pd.DataFrame:
    d = df.loc[pd.to_numeric(df["year"], errors="coerce") == 2023].copy()
    return d[["pref_code", "pref_name", "bathroom_heater_rate", "temp_annual_min", "temp_annual_mean"]].copy()


def _rank_table(pref: pd.DataFrame) -> pd.DataFrame:
    out = pref.copy()
    out["heater_pp"] = pd.to_numeric(out["bathroom_heater_rate"], errors="coerce") * 100.0
    out = out.sort_values("heater_pp", ascending=True).reset_index(drop=True)
    out["rank_low"] = out.index + 1
    return out


def _cold_tertile(pref: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    d = pref.copy()
    d["temp_annual_min"] = pd.to_numeric(d["temp_annual_min"], errors="coerce")
    thr = float(d["temp_annual_min"].quantile(1.0 / 3.0))
    cold = d.loc[d["temp_annual_min"] <= thr].copy()
    return cold, thr


def _scatter_with_fit(
    pref: pd.DataFrame,
    *,
    temp_col: str,
    out_png: Path,
    title: str,
    annotate_prefs: list[str],
) -> None:
    d = pref.copy()
    d["heater_pp"] = pd.to_numeric(d["bathroom_heater_rate"], errors="coerce") * 100.0
    d[temp_col] = pd.to_numeric(d[temp_col], errors="coerce")
    d = d.dropna(subset=["heater_pp", temp_col]).copy()

    x = d[temp_col].astype(float).to_numpy()
    y = d["heater_pp"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.scatter(x, y, s=28, alpha=0.8, edgecolors="none")

    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ys = float(ols.params[0]) + float(ols.params[1]) * xs
    ax.plot(xs, ys, linewidth=2)

    for pref_name in annotate_prefs:
        hit = d.loc[d["pref_name"] == pref_name]
        if hit.empty:
            continue
        hx = float(hit[temp_col].iloc[0])
        hy = float(hit["heater_pp"].iloc[0])
        ax.scatter([hx], [hy], s=46)
        ax.annotate(
            pref_name,
            (hx, hy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
        )

    xlabel = "年平均最低気温（℃; 都道府県平均との差）" if temp_col == "temp_annual_min" else "年平均気温（℃; 都道府県平均との差）"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("浴室暖房乾燥機設置率（%）")
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


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


def _write_report(
    out_md: Path,
    *,
    meta: dict[str, Any],
    mean_assoc: dict[str, AssocOut],
    yr2023_assoc: dict[str, AssocOut],
    mean_rank: pd.DataFrame,
    mean_cold: pd.DataFrame,
    mean_cold_thr: float,
    figs: dict[str, Path],
) -> None:
    def assoc_table(assoc: dict[str, AssocOut]) -> str:
        lines: list[str] = []
        lines.append("| 指標 | OLS: 1℃増加あたりの設置率差（pp） | 95%CI | p | Pearson r | 95%CI | p | Spearman ρ | p | R² |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for k in ["temp_annual_min", "temp_annual_mean"]:
            a = assoc[k]
            lines.append(
                "| {label} | {slope} | [{lo}, {hi}] | {p} | {r} | [{rlo}, {rhi}] | {pr} | {rho} | {ps} | {r2} |".format(
                    label="年平均最低気温" if k == "temp_annual_min" else "年平均気温",
                    slope=_fmt_pp(a.ols_slope_pp_per_1c),
                    lo=_fmt_pp(a.ols_ci[0]),
                    hi=_fmt_pp(a.ols_ci[1]),
                    p=_fmt_p(a.ols_p),
                    r=_fmt_r(a.pearson_r),
                    rlo=_fmt_r(a.pearson_ci[0]),
                    rhi=_fmt_r(a.pearson_ci[1]),
                    pr=_fmt_p(a.pearson_p),
                    rho=_fmt_r(a.spearman_rho),
                    ps=_fmt_p(a.spearman_p),
                    r2=_fmt_r(a.ols_r2),
                )
            )
        return "\n".join(lines)

    aomori = mean_rank.loc[mean_rank["pref_name"] == "青森県"].iloc[0]
    aomori_pp = float(aomori["heater_pp"])
    aomori_rank = int(aomori["rank_low"])
    n_pref = int(mean_rank.shape[0])
    aomori_pct = float(aomori_rank / n_pref * 100.0)

    cold_rank = _rank_table(mean_cold)
    aomori_cold = cold_rank.loc[cold_rank["pref_name"] == "青森県"].iloc[0]
    aomori_cold_rank = int(aomori_cold["rank_low"])
    cold_n = int(cold_rank.shape[0])

    bottom10 = mean_rank.head(10).copy()
    top10 = mean_rank.sort_values("heater_pp", ascending=False).head(10).copy()

    def simple_rank_md(df: pd.DataFrame) -> str:
        d = df[["rank_low", "pref_name", "heater_pp"]].copy()
        d["heater_pp"] = d["heater_pp"].map(lambda x: _fmt_pp(float(x)))
        lines = ["| rank(low) | 都道府県 | 設置率（%） |", "| --- | --- | --- |"]
        for _, r in d.iterrows():
            lines.append(f"| {int(r['rank_low'])} | {r['pref_name']} | {r['heater_pp']} |")
        return "\n".join(lines)

    def high_rank_md(df: pd.DataFrame) -> str:
        d = df.copy()
        d["heater_pp"] = pd.to_numeric(d["heater_pp"], errors="coerce").astype(float)
        d = d.sort_values("heater_pp", ascending=False).reset_index(drop=True)
        d["rank_high"] = d.index + 1
        d = d[["rank_high", "pref_name", "heater_pp"]].copy()
        d["heater_pp"] = d["heater_pp"].map(lambda x: _fmt_pp(float(x)))
        lines = ["| rank(high) | 都道府県 | 設置率（%） |", "| --- | --- | --- |"]
        for _, r in d.iterrows():
            lines.append(f"| {int(r['rank_high'])} | {r['pref_name']} | {r['heater_pp']} |")
        return "\n".join(lines)

    out_md.parent.mkdir(parents=True, exist_ok=True)

    fig_min_rel = figs["temp_annual_min"].relative_to(out_md.parent).as_posix()
    fig_mean_rel = figs["temp_annual_mean"].relative_to(out_md.parent).as_posix()

    md: list[str] = []
    md.append("# 都道府県別の浴室暖房乾燥機設置率と気温（基盤repoデータ）")
    md.append("")
    md.append(f"- 作成日時: {meta['created_at_local']}")
    md.append(f"- 解析ソフト: Python {meta['python_version']}")
    md.append(f"- 主要パッケージ: pandas {meta['versions']['pandas']}, numpy {meta['versions']['numpy']}, scipy {meta['versions']['scipy']}, statsmodels {meta['versions']['statsmodels']}, matplotlib {meta['versions']['matplotlib']}")
    md.append(f"- 入力データ: `{meta['base_csv']}`（sha256={meta['base_csv_sha256']}）")
    md.append(f"- 記述ポリシー: `docs/rules/statistical_reporting_policy.md`")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. 解析目的")
    md.append("")
    md.append("- 都道府県別の浴室暖房乾燥機設置率が、年平均最低気温または年平均気温と単調に関連するか（低温ほど高い傾向があるか）を確認する。")
    md.append("- 青森県の設置率が全都道府県の中でどの位置にあるか（順位）を確認する。")
    md.append("")
    md.append("## 2. データと方法")
    md.append("")
    md.append("- 基盤repoの `full_panel.csv` は都道府県×年（2015-2023）のパネルだが、浴室暖房乾燥機設置率は2023年住宅・土地統計調査に基づく都道府県値で、年に関わらず各県で一定値として付与されている。")
    md.append("- 気温（年平均最低気温・年平均気温）は年ごとに値があるため、以下の2通りで都道府県の代表値を作り、設置率（%）との関連を記述的に確認した。")
    md.append("")
    md.append("- 主解析: 2015-2023の都道府県内平均（都道府県ごとに年平均をさらに平均）")
    md.append("- 参考: 2023年のみの都道府県値（クロスセクション）")
    md.append("")
    md.append("- 解析: 設置率（%）を目的変数、気温（℃）を説明変数とする単回帰（OLS）で、1℃増加あたりの設置率差（pp）と95%CIを算出した。補助としてPearson相関とSpearman順位相関を算出した。")
    md.append("- 寒冷地（下位1/3）は、都道府県内平均の年平均最低気温 `temp_annual_min` の1/3分位点以下（`<=`）とした。")
    md.append("")
    md.append("## 3. 結果: 設置率と気温の関連")
    md.append("")
    md.append("### 3.1 主解析（2015-2023の都道府県内平均）")
    md.append("")
    md.append(assoc_table(mean_assoc))
    md.append("")
    md.append(f"![{fig_min_rel}](./{fig_min_rel})")
    md.append("")
    md.append(f"![{fig_mean_rel}](./{fig_mean_rel})")
    md.append("")
    md.append("### 3.2 参考（2023年のみ）")
    md.append("")
    md.append(assoc_table(yr2023_assoc))
    md.append("")
    md.append("## 4. 結果: 青森県の設置率順位（全都道府県）")
    md.append("")
    md.append(f"- 青森県の設置率: {_fmt_pp(aomori_pp)}%")
    md.append(f"- 全47都道府県での低い順順位: {aomori_rank}/{n_pref}（下位{_fmt_pp(aomori_pct)}%）")
    md.append("")
    md.append("### 4.1 低い順（下位10）")
    md.append("")
    md.append(simple_rank_md(bottom10))
    md.append("")
    md.append("### 4.2 高い順（上位10）")
    md.append("")
    md.append(high_rank_md(top10))
    md.append("")
    md.append("## 5. 寒冷地（年平均最低気温の下位1/3）内での青森県の位置")
    md.append("")
    md.append(f"- 寒冷地の閾値（年平均最低気温の1/3分位点）: {_fmt_c(mean_cold_thr)}℃")
    md.append(f"- 寒冷地の都道府県数: {cold_n}")
    md.append(f"- 寒冷地内での青森県の低い順順位: {aomori_cold_rank}/{cold_n}")
    md.append("")
    md.append("---")
    md.append("")
    out_md.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="基盤repoの都道府県データから、設置率×気温の関連と青森県順位を記述します。")
    parser.add_argument("--base-csv", type=str, default="", help="基盤repoの full_panel.csv パス（未指定なら ../入浴統計/... を使用）")
    parser.add_argument("--tag", type=str, default="", help="outputs/runs/<tag> の tag（未指定なら自動生成）")
    parser.add_argument("--out-dir", type=str, default="", help="出力ディレクトリ（未指定なら outputs/runs/<tag>/）")
    parser.add_argument("--years-mean", type=str, default="2015-2023", help="主解析の年範囲表示ラベル（計算自体はCSV内の全期間平均）")
    args = parser.parse_args()

    _set_japanese_font()

    repo_root = Path(__file__).resolve().parents[2]
    base_csv = Path(args.base_csv).expanduser() if args.base_csv else (repo_root / DEFAULT_BASE_RELATIVE)
    base_csv = base_csv.resolve()

    tag = args.tag.strip() or _default_tag("heater_temp_aomori")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (repo_root / "outputs" / "runs" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_csv)
    pref_mean = _pref_level_mean(df)
    pref_2023 = _pref_level_2023(df)

    pref_mean.to_csv(out_dir / "pref_level_mean_2015_2023.csv", index=False, encoding="utf-8")
    pref_2023.to_csv(out_dir / "pref_level_2023.csv", index=False, encoding="utf-8")

    mean_assoc = {k: _assoc(pref_mean, temp_col=k) for k in ["temp_annual_min", "temp_annual_mean"]}
    yr2023_assoc = {k: _assoc(pref_2023, temp_col=k) for k in ["temp_annual_min", "temp_annual_mean"]}

    mean_rank = _rank_table(pref_mean)
    mean_cold, cold_thr = _cold_tertile(pref_mean)

    figs = {
        "temp_annual_min": figs_dir / "scatter_heater_pp_vs_temp_annual_min.png",
        "temp_annual_mean": figs_dir / "scatter_heater_pp_vs_temp_annual_mean.png",
    }
    annotate = ["青森県", "北海道", "沖縄県", "東京都"]
    _scatter_with_fit(
        pref_mean,
        temp_col="temp_annual_min",
        out_png=figs["temp_annual_min"],
        title=f"設置率（%）と年平均最低気温（主解析: {args.years_mean}平均）",
        annotate_prefs=annotate,
    )
    _scatter_with_fit(
        pref_mean,
        temp_col="temp_annual_mean",
        out_png=figs["temp_annual_mean"],
        title=f"設置率（%）と年平均気温（主解析: {args.years_mean}平均）",
        annotate_prefs=annotate,
    )

    meta = {
        "created_at_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "base_csv": str(base_csv),
        "base_csv_sha256": _sha256(base_csv),
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(
        out_dir / "report.md",
        meta=meta,
        mean_assoc=mean_assoc,
        yr2023_assoc=yr2023_assoc,
        mean_rank=mean_rank,
        mean_cold=mean_cold,
        mean_cold_thr=cold_thr,
        figs=figs,
    )

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
