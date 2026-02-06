# Step 5 集計仕様（v1）

## 目的

- 紙アンケート回収後に、必須3表と品質指標を再現可能に作成する。
- 一次集計（Excel/スプレッドシート）と、Python再集計の一致を確認する。

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

## 出力物（必須）

- `qc_summary.csv`
  - 総票数、有効票数、無効票数、主要欠損率
- `table1_install_x_bathroom_cold_7pt.csv`
  - 表1: 設置 × 浴室寒さ7段階
- `table2_install_x_central_heating.csv`
  - 表2: 設置 × セントラル暖房使用
- `table3_reason_x_housing_type.csv`
  - 表3: 未導入/低使用理由 × 住宅種別
- `tabulation_report.md`
  - 実行ログ、主要指標、判定ルール適用結果

## 無効票判定

- 次のいずれかに該当する票を無効とする。
  - `q8_bath_heater_installed` 欠損
  - Q8=1 かつ `q11_bath_heater_heating_winter_use` 欠損
  - Q8 in {2,3} または Q11 in {4,5} で `q14_reason_codes` 欠損

## 優勢判定（Step 6用）

- `no_need`（不要群）と `barrier`（障壁群）を集計し、差が10pp以上なら優勢と判定する。
- 10pp未満は「拮抗」として扱う。

## 実行コマンド

```powershell
python aomori_survey/src/scripts/tabulate_aomori_paper_survey.py `
  --input-csv aomori_survey/deliverables/20260206_aomori_survey_data_entry_template.csv `
  --output-dir aomori_survey/outputs/runs/20260331_aomori_paper_survey_tabulation
```

## 受け入れ基準

- 有効票数30以上
- 主要欠損率20%未満
- 必須3表を生成
- Excel一次集計とPython再集計が一致
