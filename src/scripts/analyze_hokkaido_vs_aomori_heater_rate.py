from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
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


def _pref_level(df: pd.DataFrame) -> pd.DataFrame:
    mean_cols = [
        "temp_annual_mean",
        "temp_annual_min",
        "public_bath_per_100k",
        "dayservice_far_share_2km",
    ]
    first_cols = [
        "bathroom_heater_rate",
        "double_glazing_window_rate",
        "bathroom_handrail_rate",
        "dressing_room_handrail_rate",
        "easy_step_bathtub_rate",
        "wheelchair_width_hallway_rate",
        "step_free_indoor_rate",
        "wheelchair_access_to_entrance_rate",
    ]
    return (
        df.groupby(["pref_code", "pref_name"], as_index=False)
        .agg(
            **{c: (c, "mean") for c in mean_cols},
            **{c: (c, "first") for c in first_cols},
        )
        .copy()
    )


def _scatter_heater_vs_dg(pref: pd.DataFrame, *, out_png: Path, annotate: list[str]) -> None:
    d = pref.copy()
    d["heater_pp"] = pd.to_numeric(d["bathroom_heater_rate"], errors="coerce") * 100.0
    d["dg_pp"] = pd.to_numeric(d["double_glazing_window_rate"], errors="coerce") * 100.0
    d = d.dropna(subset=["heater_pp", "dg_pp"]).copy()

    x = d["dg_pp"].astype(float).to_numpy()
    y = d["heater_pp"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    ax.scatter(x, y, s=28, alpha=0.8, edgecolors="none")

    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ys = float(ols.params[0]) + float(ols.params[1]) * xs
    ax.plot(xs, ys, linewidth=2)

    for pref_name in annotate:
        hit = d.loc[d["pref_name"] == pref_name]
        if hit.empty:
            continue
        hx = float(hit["dg_pp"].iloc[0])
        hy = float(hit["heater_pp"].iloc[0])
        ax.scatter([hx], [hy], s=46)
        ax.annotate(pref_name, (hx, hy), textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("複層ガラス化率（%）")
    ax.set_ylabel("浴室暖房乾燥機設置率（%）")
    ax.set_title("設置率（%）と複層ガラス化率（%）の関係（都道府県）")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


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


def _write_report(
    out_md: Path,
    *,
    meta: dict[str, Any],
    focus_table: pd.DataFrame,
    model_table: pd.DataFrame,
    model_params: dict[str, Any],
    corr_params: dict[str, Any],
    fig_path: Path,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fig_rel = fig_path.relative_to(out_md.parent).as_posix()

    lines: list[str] = []
    lines.append("# 北海道が青森県より浴室暖房乾燥機設置率が低い理由（記述的整理）")
    lines.append("")
    lines.append(f"- 作成日時: {meta['created_at_local']}")
    lines.append(f"- 解析ソフト: Python {meta['python_version']}")
    lines.append(
        f"- 主要パッケージ: pandas {meta['versions']['pandas']}, numpy {meta['versions']['numpy']}, scipy {meta['versions']['scipy']}, statsmodels {meta['versions']['statsmodels']}, matplotlib {meta['versions']['matplotlib']}"
    )
    lines.append(f"- 入力データ: `{meta['base_csv']}`（sha256={meta['base_csv_sha256']}）")
    lines.append(f"- 記述ポリシー: `docs/rules/statistical_reporting_policy.md`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 事実確認（基盤repoデータ）")
    lines.append("")
    lines.append("- 浴室暖房乾燥機設置率（低い順）では、北海道は2/47（8.41%）、青森県は3/47（10.66%）であり、北海道の方が低い。")
    lines.append("")
    lines.append("### 1.1 北海道と青森県の比較（参考指標）")
    lines.append("")
    lines.append(_df_to_md_table(focus_table))
    lines.append("")
    lines.append("## 2. 記述的な整理（仮説）")
    lines.append("")
    lines.append("- 本データの設置率は「浴室暖房乾燥機」という特定設備に限られ、浴室の暖房手段全体（例: セントラル暖房、温水パネル、床暖房等）を測定しない可能性がある。")
    lines.append("- 北海道は複層ガラス化率が高い（77.95%）ため、室内（浴室・脱衣所を含む）の温熱環境が相対的に保たれやすい住宅ストックが多い場合、浴室専用の「暖房乾燥機」を追加導入する必要性が相対的に小さくなる仮説が立つ。")
    lines.append("- 都道府県別では、複層ガラス化率が高い地域ほど設置率が低い方向の相関が観察される（記述的; 因果は意味しない）。")
    lines.append(f"  - Pearson r={corr_params['pearson_r']:.3f}（{_fmt_p(corr_params['pearson_p'])}）")
    lines.append("")
    lines.append(f"![{fig_rel}](./{fig_rel})")
    lines.append("")
    lines.append("## 3. 参考: 気温・複層ガラス化率で説明したときの残差")
    lines.append("")
    lines.append("- OLSモデル（47都道府県）: `設置率（%） ~ 年平均最低気温（℃） + 複層ガラス化率（%）`")
    lines.append(f"  - R²={model_params['r2']:.3f}")
    lines.append(
        f"  - 年平均最低気温の係数={model_params['coef_tmin']:.3f}（95%CI [{model_params['ci_tmin'][0]:.3f}, {model_params['ci_tmin'][1]:.3f}], {_fmt_p(model_params['p_tmin'])}）"
    )
    lines.append(
        f"  - 複層ガラス化率の係数={model_params['coef_dg']:.3f}（95%CI [{model_params['ci_dg'][0]:.3f}, {model_params['ci_dg'][1]:.3f}], {_fmt_p(model_params['p_dg'])}）"
    )
    lines.append("- この単純モデルでも、北海道は予測より設置率が低い（残差 -6.21pp）という位置づけになる。")
    lines.append("")
    lines.append(_df_to_md_table(model_table))
    lines.append("")
    lines.append("## 4. 次の確認（追加データが必要）")
    lines.append("")
    lines.append("- 住宅・土地統計調査等で、(a) 集合住宅比率、(b) 暖房方式（全館/局所）、(c) 浴室の暖房手段（乾燥機以外を含む）の都道府県差を確認する。")
    lines.append("- CiNii等で『浴室暖房乾燥機に関する調査』（業界誌）などの本文が閲覧できる場合、地域差（北海道/青森等）が議論されているか確認する。")
    lines.append("")
    lines.append("---")
    lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="北海道と青森県の設置率差を、基盤repoの周辺指標で記述的に整理します。")
    parser.add_argument("--base-csv", type=str, default="", help="基盤repoの full_panel.csv パス（未指定なら ../入浴統計/... を使用）")
    parser.add_argument("--tag", type=str, default="", help="outputs/runs/<tag> の tag（未指定なら自動生成）")
    args = parser.parse_args()

    _set_japanese_font()

    repo_root = Path(__file__).resolve().parents[2]
    base_csv = Path(args.base_csv).expanduser() if args.base_csv else (repo_root / DEFAULT_BASE_RELATIVE)
    base_csv = base_csv.resolve()

    tag = args.tag.strip() or _default_tag("hokkaido_vs_aomori_heater_rate")
    out_dir = repo_root / "outputs" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_csv)
    pref = _pref_level(df)
    pref.to_csv(out_dir / "pref_level_features.csv", index=False, encoding="utf-8")

    pref["heater_pp"] = pd.to_numeric(pref["bathroom_heater_rate"], errors="coerce") * 100.0
    pref["dg_pp"] = pd.to_numeric(pref["double_glazing_window_rate"], errors="coerce") * 100.0
    pref["tmin"] = pd.to_numeric(pref["temp_annual_min"], errors="coerce")

    rank = pref[["pref_name", "heater_pp"]].sort_values("heater_pp").reset_index(drop=True)
    rank["rank_low"] = rank.index + 1
    pref = pref.merge(rank[["pref_name", "rank_low"]], on="pref_name", how="left")

    focus = pref.loc[pref["pref_name"].isin(["北海道", "青森県"])].copy()
    focus_table = focus[
        [
            "pref_name",
            "rank_low",
            "heater_pp",
            "temp_annual_min",
            "temp_annual_mean",
            "dg_pp",
            "public_bath_per_100k",
            "dayservice_far_share_2km",
        ]
    ].copy()
    focus_table = focus_table.rename(
        columns={
            "pref_name": "都道府県",
            "rank_low": "設置率順位（低い順）",
            "heater_pp": "設置率（%）",
            "temp_annual_min": "年平均最低気温（℃）",
            "temp_annual_mean": "年平均気温（℃）",
            "dg_pp": "複層ガラス化率（%）",
            "public_bath_per_100k": "公衆浴場/100k（参考）",
            "dayservice_far_share_2km": "遠距離DS比（参考）",
        }
    )
    for c in [
        "設置率（%）",
        "年平均最低気温（℃）",
        "年平均気温（℃）",
        "複層ガラス化率（%）",
        "公衆浴場/100k（参考）",
        "遠距離DS比（参考）",
    ]:
        focus_table[c] = pd.to_numeric(focus_table[c], errors="coerce").round(2)

    d_corr = pref.dropna(subset=["heater_pp", "dg_pp"]).copy()
    pear = stats.pearsonr(d_corr["dg_pp"].astype(float), d_corr["heater_pp"].astype(float))
    corr_params = {"pearson_r": float(pear.statistic), "pearson_p": float(pear.pvalue)}

    d = pref.dropna(subset=["heater_pp", "tmin", "dg_pp"]).copy()
    X = sm.add_constant(d[["tmin", "dg_pp"]].astype(float))
    ols = sm.OLS(d["heater_pp"].astype(float), X).fit()
    d["pred"] = ols.predict(X)
    d["resid"] = d["heater_pp"] - d["pred"]

    model_table = d.loc[d["pref_name"].isin(["北海道", "青森県"]), ["pref_name", "heater_pp", "pred", "resid", "tmin", "dg_pp"]].copy()
    model_table = model_table.rename(
        columns={
            "pref_name": "都道府県",
            "heater_pp": "設置率（%）",
            "pred": "予測（%）",
            "resid": "残差（pp）",
            "tmin": "年平均最低気温（℃）",
            "dg_pp": "複層ガラス化率（%）",
        }
    )
    for c in ["設置率（%）", "予測（%）", "残差（pp）", "年平均最低気温（℃）", "複層ガラス化率（%）"]:
        model_table[c] = pd.to_numeric(model_table[c], errors="coerce").round(2)

    ci = ols.conf_int()
    model_params = {
        "r2": float(ols.rsquared),
        "coef_tmin": float(ols.params["tmin"]),
        "ci_tmin": [float(ci.loc["tmin"].iloc[0]), float(ci.loc["tmin"].iloc[1])],
        "p_tmin": float(ols.pvalues["tmin"]),
        "coef_dg": float(ols.params["dg_pp"]),
        "ci_dg": [float(ci.loc["dg_pp"].iloc[0]), float(ci.loc["dg_pp"].iloc[1])],
        "p_dg": float(ols.pvalues["dg_pp"]),
    }

    fig_path = figs_dir / "scatter_heater_pp_vs_double_glazing_pp.png"
    _scatter_heater_vs_dg(pref, out_png=fig_path, annotate=["北海道", "青森県", "東京都", "沖縄県"])

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
    (out_dir / "model_params.json").write_text(json.dumps(model_params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _write_report(
        out_dir / "report.md",
        meta=meta,
        focus_table=focus_table,
        model_table=model_table,
        model_params=model_params,
        corr_params=corr_params,
        fig_path=fig_path,
    )

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
