# Step 5 集計仕様（v2）

## 目的

- 紙アンケート回収後に、必須3表と品質指標を再現可能に作成する。
- Q7-Q9を主解析対象として、便宜抽出の限界を明示した上で解釈可能な要約を作成する。
- 一次集計（Excel/スプレッドシート）と、Python再集計の一致を確認する。

## サンプルサイズ前提（固定）

- 信頼水準: 95%
- 許容誤差: ±12pp
- 比率推定の保守設定: `p=0.5`
- 便宜抽出補正: `deff=1.2`
- 無効・主要欠損見込み: 15%
- 計算結果:
  - `n0 = 1.96^2 * p*(1-p) / E^2 = 66.7 -> 67`
  - `n_valid = ceil(67 * 1.2) = 81`
  - `n_collected = ceil(81 / 0.85) = 96`
  - 運用目標回収票: 100

## 入力データ

- 入力CSV:
  - `aomori_survey/deliverables/20260206_aomori_survey_data_entry_template.csv` 形式
- 必須カラム:
  - `response_id`
  - `q8_bath_heater_installed`
  - `q9_central_heating_use`
  - `q11_bath_heater_heating_winter_use`
  - `q13a_bathroom_cold_7pt`
  - `q14_reason_codes`
  - `q3_housing_type`
- 実行前提:
  - Pythonスクリプトは、上記に加えて入力テンプレート定義の全カラムを要求する。

## 主解析仕様（Q7-Q9）

- 主解析のQ7は二値化して扱う。
  - `q7_main_group=installed_any`: `q8_bath_heater_installed=1`
  - `q7_main_group=not_installed`: `q8_bath_heater_installed=2`
  - `q8_bath_heater_installed in {3,99}` は主解析から除外し、件数のみ報告する。
- 補助記述として、設置あり内での低使用を示す。
  - `installed_low_use`: `q8_bath_heater_installed=1` かつ `q11_bath_heater_heating_winter_use in {4,5}`
  - 上記は件数・割合・95%CIのみ提示し、優劣の結論には用いない。
- Q7×Q9は記述統計（割合＋95%CI）を主とする。
  - 小標本で期待度数が小さい場合は exact 法を優先する。
  - exact 法の実装がない環境では、検定結果を省略し記述統計のみで解釈する。
- Q8理由（`q14_reason_codes`）は多重回答率で集計し、群差は探索的解釈に限定する。

## 出力物（必須）

- `qc_summary.csv`
  - 総票数、有効票数、無効票数、主要欠損率、解析ゲート判定
- `table1_install_x_bathroom_cold_7pt.csv`
  - 表1: 設置 × 浴室寒さ7段階
- `table2_install_x_central_heating.csv`
  - 表2: Q7主解析二値群 × セントラル暖房使用
- `table3_reason_x_housing_type.csv`
  - 表3: 未導入/低使用理由 × 住宅種別
- `tabulation_report.md`
  - 実行ログ、主要指標、判定ルール適用結果、主解析の割合・95%CI

## 無効票判定

- 次のいずれかに該当する票を無効とする。
  - `q8_bath_heater_installed` 欠損
  - Q8=1 かつ `q11_bath_heater_heating_winter_use` 欠損
  - Q8 in {2,3} または Q11 in {4,5} で `q14_reason_codes` 欠損

## 優勢判定（Step 6用）

- `no_need`（不要群）と `barrier`（障壁群）を集計し、差が10pp以上なら優勢と判定する。
- 10pp未満は「拮抗」として扱う。

## 解析ゲート（Q7-Q9主解析）

- `valid_responses >= 80` かつ `main_missing_rate_pct < 20`:
  - 主解析（Q7-Q9）を実施し、結果を本文で解釈する。
- `60 <= valid_responses < 80` かつ `main_missing_rate_pct < 20`:
  - 探索的解析として提示し、結論は限定的記述に留める。
- `valid_responses < 60` または `main_missing_rate_pct >= 20`:
  - 記述統計中心とし、比較結論は保留する。

## 実行コマンド

```powershell
python aomori_survey/src/scripts/tabulate_aomori_paper_survey.py `
  --input-csv aomori_survey/deliverables/20260206_aomori_survey_data_entry_template.csv `
  --output-dir aomori_survey/outputs/runs/20260331_aomori_paper_survey_tabulation
```

## 受け入れ基準

- 計算再現テスト: `E=0.12, deff=1.2, invalid=0.15` で `有効81/回収96/運用100` が再現できる
- 感度テスト: `E=0.10` で回収目標が100票超、`E=0.15` で回収目標が大きく減る
- 解析ゲート判定が `>=80`, `60-79`, `<60` の3区分で一意に出力される
- 必須3表を生成
- Excel一次集計とPython再集計が一致
