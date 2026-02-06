# 五所川原市 紙アンケート コードブック（v1）

## ドキュメント情報

- 版: v1
- 作成日: 2026-02-06
- 対応調査票: `aomori_survey/deliverables/20260206_aomori_survey_questionnaire_v1.md`
- 目的: データ入力・品質管理・クロス集計の仕様を固定する

## 基本ルール

- 1行=1回答票（匿名ID単位）
- 欠損コード:
  - `99`: 無回答/不明（単一選択）
  - `""`（空文字）: 条件分岐で非該当
- 複数選択項目（Q14）は、`;` 区切りでコードを並べる（例: `2;3;5`）
- 自由記述は `q14_reason_other_text` にそのまま記録する

## 変数定義

| 変数名 | 対応設問 | 型 | コード定義 |
| --- | --- | --- | --- |
| `response_id` | 匿名ID | string | `GS-0001` のように一意 |
| `q1_age_group` | Q1 | int | 1=18-29, 2=30-39, 3=40-49, 4=50-59, 5=60-69, 6=70+, 99=無回答 |
| `q2_residence_area` | Q2 | int | 1=五所川原市, 2=県内他市町村, 3=県外, 99=無回答 |
| `q3_respondent_fit` | Q3 | int | 1=把握している, 2=あまり把握していない, 99=無回答 |
| `q4_housing_type` | Q4 | int | 1=一戸建て, 2=集合住宅, 3=その他, 99=無回答 |
| `q5_building_age_band` | Q5 | int | 1=10年未満, 2=10-19年, 3=20-29年, 4=30年以上, 5=不明, 99=無回答 |
| `q6_tenure` | Q6 | int | 1=持家, 2=賃貸, 3=その他, 99=無回答 |
| `q7_window_insulation_proxy` | Q7 | int | 1=ほとんど二重/複層, 2=一部二重/複層, 3=ほとんど単板, 4=不明, 99=無回答 |
| `q8_winter_home_bath_freq` | Q8 | int | 1=ほぼ毎日, 2=週4-6, 3=週1-3, 4=月1-3, 5=ほとんどなし, 99=無回答 |
| `q9_bath_heater_installed` | Q9 | int | 1=設置あり, 2=設置なし, 3=不明, 99=無回答 |
| `q10_central_heating_use` | Q10 | int | 1=使用あり, 2=使用なし, 3=不明, 99=無回答 |
| `q11_bath_heater_winter_use` | Q11 | int | 1=ほぼ毎回, 2=週4-6, 3=週1-3, 4=月1-3, 5=ほぼ使わない, 99=無回答, 空文字=非該当 |
| `q12_preheat_before_bath` | Q12 | int | 1=ほぼ毎回, 2=ときどき, 3=ほぼしない, 99=無回答, 空文字=非該当 |
| `q13a_bathroom_cold_7pt` | Q13(浴室) | int | 1=非常に暖かい ... 7=非常に寒い, 99=無回答 |
| `q13b_dressingroom_cold_7pt` | Q13(脱衣所) | int | 1=非常に暖かい ... 7=非常に寒い, 99=無回答 |
| `q14_reason_codes` | Q14 | string | `1-9` を`;`区切り、空文字=非該当 |
| `q14_reason_other_text` | Q14自由記述 | string | 任意記述、非該当は空文字 |

## Q14 理由コード詳細

| コード | 理由 |
| --- | --- |
| 1 | 既に十分暖かいと感じる |
| 2 | 設置費用が高い |
| 3 | 電気代が気になる |
| 4 | 工事が難しい（構造上の制約） |
| 5 | 賃貸で工事できない |
| 6 | 必要性を感じない |
| 7 | 使い方がわからない |
| 8 | 故障中/メンテナンスの問題 |
| 9 | その他（自由記述） |

## 無効票判定（計画準拠）

- 「主要3項目欠損のみ無効」を以下で実装する。
- 主要項目:
  - `q9_bath_heater_installed`
  - `q11_bath_heater_winter_use`（Q9=1の時のみ必須）
  - `q14_reason_codes`（Q9 in {2,3} または Q11 in {4,5} の時のみ必須）
- 無効票の条件:
  - `q9_bath_heater_installed` が `99` または空
  - Q9=1 かつ `q11_bath_heater_winter_use` が `99` または空
  - Q9 in {2,3} または Q11 in {4,5} で、`q14_reason_codes` が空

## クロス集計キー（必須3表）

### 表1: 設置 × 寒さ（浴室）

- 行: `q9_bath_heater_installed`（1,2,3）
- 列: `q13a_bathroom_cold_7pt`（1-7）
- 補助二値化: `bathroom_cold_binary`（5-7を「寒い」, 1-4を「寒くない/中立」）

### 表2: 設置 × セントラル暖房使用

- 行: `q9_bath_heater_installed`（1,2,3）
- 列: `q10_central_heating_use`（1,2,3）

### 表3: 未導入/低使用理由 × 住宅条件

- 行: `q14_reason_codes`（explode後の各理由コード）
- 列: `q4_housing_type`（1,2,3）

## 自由記述のカテゴリ化（Step 5）

- `q14_reason_other_text` は以下カテゴリに再分類し、記録列 `q14_reason_other_category` を追加する。
  - `cost`: 費用・電気代関連
  - `construction`: 工事制約・設備制約
  - `tenure`: 賃貸制約
  - `no_need`: 不要感・優先度低
  - `operation`: 使い方不明・運用負担
  - `maintenance`: 故障・保守
  - `other`: 上記以外

## 品質管理チェック

- `response_id` 重複なし
- `q2_residence_area=3`（県外）は別途フラグ（除外せず注記）
- `q3_respondent_fit=2` は品質注意フラグ
- 有効票数が30以上かつ主要欠損率20%未満かをStep 6で判定する
