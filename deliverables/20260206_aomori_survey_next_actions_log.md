# 実行ログ: 青森調査 次アクション進行管理

## 作成情報

- 作成日: 2026-02-06
- 最終更新日: 2026-02-06
- 元レポート: `aomori_survey/deliverables/20260205_aomori_survey_pre_survey_report.md`
- 目的: 次アクション（Step 1-6）を順次実行し、意思決定・成果物・判定を一元管理する。

## 進行ステータス

- Step 1（設計確定）: `completed`
- Step 2（調査票v1作成）: `completed`
- Step 3（認知テスト）: `pending`
- Step 4（本調査実施）: `pending`
- Step 5（集計・解釈）: `pending`
- Step 6（レポート更新）: `pending`

## Step 1 設計確定（完了）

### 1-1 固定前提

- 調査目的は「青森県全体の普及率推定」ではなく、「限定サンプル内での内訳把握」とする。
- 主手法は紙アンケート（自己記入）とし、機器を使った室温実測は対象外とする。
- 分析は記述統計とクロス集計に限定し、因果推定は行わない。

### 1-2 運用仕様（確定値）

- 対象地域: 五所川原市（特定市町村1自治体）
- 回答対象者: 原則として五所川原市在住、18歳以上・1世帯1名（入浴実態を把握する成人）
- 居住地設問運用: Q2は「五所川原市/五所川原市以外の青森県内」の2区分。県外回答は `q2_residence_area=99` + 備考注記で扱う。
- 配布方法: 紙のみ（協力者手渡し配布・回収）
- 回収期間: 2026-03-01から2026-03-31
- 匿名性: 完全匿名（氏名・連絡先は取得しない）
- 目標票数: 50票（実務下限30票）
- 票管理: 匿名ID事前印字（`GS-0001` 形式）

### 1-3 主要アウトカム（固定）

1. 浴室暖房乾燥機の設置有無
2. 浴室暖房乾燥機の冬季使用頻度
3. 未導入/低使用の理由

### 1-4 補助アウトカム（固定）

- セントラル暖房システム（全館暖房）の使用有無
- 浴室・脱衣所の主観的寒さ
- 代替暖房機器の使用状況
- 住宅条件（種別、築年帯、持家/賃貸、窓断熱proxy）

### 1-5 測定・判定仕様（固定）

- 寒さ尺度: 7段階
- 冬季自宅入浴頻度: 5カテゴリ
- 無効票ルール: 主要3項目欠損のみ無効
- 優勢判定: 不要群と障壁群の差が10pp以上
- 次段階ゲート: 有効票30以上かつ主要欠損率20%未満
- 共有範囲: 内部利用のみ

## Step 2 調査票v1・コードブックv1（完了）

### 2-1 作成成果物

- 調査票v1: `aomori_survey/deliverables/20260206_aomori_survey_questionnaire_v1.md`
- コードブックv1: `aomori_survey/deliverables/20260206_aomori_survey_codebook_v1.md`
- 入力テンプレート: `aomori_survey/deliverables/20260206_aomori_survey_data_entry_template.csv`
- 集計仕様: `aomori_survey/deliverables/20260206_aomori_survey_step5_tabulation_spec_v1.md`

### 2-2 受け入れ確認

- [x] 設問数が10-15問の範囲に収まっている
- [x] 主要3項目が定義されている
- [x] セントラル暖房使用有無を必須補助項目に含む
- [x] 分岐条件（Q8/Q11/Q14）が明示されている
- [x] 無効票判定仕様がコードブックに定義されている

## Step 3 認知テスト（未着手）

### 3-1 実施仕様（確定）

- 対象: 3名
- 形式: 記入後に口頭確認
- 修正基準: 1名でも詰まれば修正

### 3-2 用意済み資料

- 実施手順: `aomori_survey/deliverables/20260206_aomori_survey_cognitive_test_protocol_v1.md`
- 課題ログテンプレート: `aomori_survey/deliverables/20260206_aomori_survey_cognitive_test_issue_log_template.md`
- 結果記録テンプレート: `aomori_survey/deliverables/20260206_aomori_survey_cognitive_test_results.md`

### 3-3 完了条件

- 重大理解障害が0件になるまで修正し、v2を確定する。

## Step 4 本調査実施（未着手）

### 4-1 実施仕様（確定）

- 配布チャネル: 協力者ネットワーク中心
- 票管理: 匿名ID事前印字
- 締切ルール: 50票到達または2026-03-31到来（早い方）

### 4-2 用意済み資料

- 本調査運用手順: `aomori_survey/deliverables/20260206_aomori_survey_fieldwork_manual_v1.md`
- 回収進捗テンプレート: `aomori_survey/deliverables/202603_aomori_survey_collection_tracker_template.csv`

## Step 5 集計・解釈（未着手）

### 5-1 集計仕様（確定）

- 必須3表:
  - 表1: `設置 × 浴室寒さ`
  - 表2: `設置 × セントラル暖房使用`
  - 表3: `未導入理由 × 住宅条件`
- 自由記述: 事前カテゴリへ再分類
- 集計ツール: Excel/スプレッドシート + Python再集計

### 5-2 用意済み資料

- 集計スクリプト: `aomori_survey/src/scripts/tabulate_aomori_paper_survey.py`
- ドライラン出力（テンプレートCSVで確認）: `aomori_survey/outputs/runs/20260206_implement_plan_dryrun_tabulation`

## Step 6 レポート更新（未着手）

- 対象文書:
  - `aomori_survey/deliverables/20260205_aomori_survey_pre_survey_report.md`
  - 本ログファイル
- 実施内容:
  - 有効票数、主要欠損率、必須3表の結果、優勢判定、次段階判定を追記する。
  - 標本偏りと一般化限界を明記する。

## 決定ログ

- 2026-02-06: Step 1の全仕様を確定（五所川原市、紙調査、匿名、期間固定、目標50票）。
- 2026-02-06: Step 2成果物（調査票v1、コードブックv1、入力テンプレート、集計仕様）を作成。
- 2026-02-06: Step 3-5の運用文書・スクリプトを準備し、集計ドライランを通過。
- 2026-02-06: 調査票・コードブック・CSV・集計仕様・運用文書・集計スクリプトを再採番仕様（Q8/Q11/Q14分岐, Q10新設）へ同期改訂。
