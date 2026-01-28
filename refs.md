# 文献メモ（refs）: 新研究（暫定）

- 最終更新: 2026-01-28
- 目的:
  - 本研究の文献探索ログと証拠メモを管理する。
  - 文献探索ログは `refs/search/` に保存し、本文へ反映する前にここへ記録する。

---

## 1. 検索プロトコル（簡易）

### 1.1 データベース/サイト

- J-STAGE
- CiNii Research
- PubMed

### 1.2 実施日

- 2026-01-26

### 1.3 主な検索語

- 日本語（例）: 青森, 浴室暖房, 浴室暖房乾燥機, 浴室 換気 乾燥 暖房, 風呂, 普及率, 設置率
- 英語（例）: Aomori, bathroom heating, bathroom heater, bathroom heater dryer, installation rate

---

## 2. 検索ログ（青森県の浴室暖房普及率が低いことに関する探索）

### 2.1 J-STAGE

- 検索日: 2026-01-26
- ログ: `refs/search/20260126_125511_aomori_bathroom_heater_literature_jstage.json`
- 要約: 「青森」×「浴室」×「暖房/乾燥機」×「設置率/普及率」などの組み合わせで検索したが、該当0件（検索式はログ参照）。

### 2.2 CiNii Research

- 検索日: 2026-01-26
- ログ: `refs/search/20260126_125401_aomori_bathroom_heater_literature_cinii.json`
- 要約:
  - 「青森」×「浴室」×「暖房」×「設置率/普及率」などの組み合わせで検索したが、該当0件（検索式はログ参照）。
  - 参考として、「浴室 暖房 乾燥機」（23件）、「浴室 暖房 乾燥機 設置率」（1件）などの広めの語でも検索したが、青森県の普及率の低さを直接扱う資料は確認できなかった（詳細はログ参照）。
  - 上位ヒットの中身確認（CRIDスクリーニング）: `refs/search/20260126_132614_cinii_top_hit_screening_cinii_top_hit_screening.json`
    - 例（上位ヒットの性質）:
      - 業界誌（リビングアメニティ協会のニュースレター）: 「浴室暖房乾燥機に関する調査について」（CRID:1520294496047988224）
      - 消費者向け雑誌記事（製品/コスト文脈）: 「ガス代節約ならecoジョーズ…」（CRID:1521980705369193856）
      - 安全・火災事例: 「浴室暖房乾燥機から出火…」（CRID:1520572358896181760）
      - 工学系（給湯・浴室乾燥等の技術）: 「太陽熱を利用した給湯・浴室乾燥システム…」（CRID:1390001277377136384）
    - いずれも、青森県の「普及率が低い」こと自体を直接扱う資料としては確認できなかった（本文閲覧可否の制約あり）。

### 2.3 PubMed

- 検索日: 2026-01-26
- ログ: `refs/search/20260126_125226_aomori_bathroom_heater_literature_pubmed.json`
- 要約:
  - 「Aomori」×「bathroom heating/heater」などの組み合わせで検索したが、該当0件（検索式はログ参照）。
  - 広めの語として「bathroom heating Japan」（9件）も確認したが、青森県の普及率の低さに直結する資料は確認できなかった（詳細はログ参照）。

---

## 3. データソース（e-Stat）

### 3.1 令和5年度 家庭部門のCO2排出実態統計調査（セントラル暖房システム使用状況）

- 取得日: 2026-01-26
- 対象表: <第3-1表>基本項目（世帯、住宅、機器使用状況等）別-暖房使用状況（…セントラル暖房システムの使用状況…）
- e-Stat: statInfId=000040277713（ファイル名: `co2r5_3-1_00全国.xlsx`）
- 本repoの解析run: `outputs/runs/20260126_143043_central_heating_vs_bathroom_heater_rate/report.md`
- 備考: 公開表は「地方別（10区分）」のため、都道府県別ではなく地域別に集計して検討した。

### 3.2 都道府県別のセントラル暖房使用率（公開表）可否

- 実施日: 2026-01-26
- 方法:
  - e-Stat「統計データを探す（ファイル）」で `query=セントラル暖房` を用いてヒットしたデータセットを確認。
  - DB表示（DBView）のAPI（`/dbview/api_get_model?sid=...`）で、地理区分（area）に都道府県が含まれるかを点検。
- 結果:
  - 「セントラル暖房」関連のヒットは、地理区分が「全国」または「地方（10区分）」の公開表のみで、都道府県別の公開表は確認できなかった（2026-01-26時点）。
- 備考:
  - 都道府県別の解析は、地方別値を都道府県へ割り当てたproxyで実施した: `outputs/runs/20260126_153831_pref_bathroom_heater_vs_central_heating_region_proxy/report.md`

### 3.3 e-Stat掲載年次（セントラル暖房システムの使用状況）

- 確認日: 2026-01-28
- 環境省「家庭CO2統計」ページにて、平成29年度〜令和5年度の確報値統計表がe-Statに掲載されている旨を確認。
- 同ページにて、令和6年度以降の偶数年度は調査を実施しない（隔年実施）旨を確認。
- e-Stat「統計データを探す（ファイル）」の令和5年度/令和4年度ページで、<第3-1表>基本項目（世帯、住宅、機器使用状況等）別-暖房使用状況に「セントラル暖房システムの使用状況」が掲載されていることを確認。

### 3.4 北海道のセントラル暖房使用率の年次（平成29-令和5年度）

- 確認日: 2026-01-28
- 方法:
  - e-Stat `retrieve/api_file` で `toukei=00650408` と `query="第3-1表 北海道 暖房"` を用いて該当表を抽出。
  - 各年度の「全国」ファイル（第3-1表）から、地方別の「北海道」行を使用。
  - 使用率（%）は、表の「セントラル暖房システムの使用状況」について、**「使用していない」以外（= 使用している3区分または4区分 + 不明）を合算**して作成した。
- 使用statInfId（全国・第3-1表）:
  - 平成29年度: 000031788023（fileKind=0）
  - 平成30年度: 000031912817（fileKind=0）
  - 平成31年度: 000032051935（fileKind=4）
  - 令和2年度: 000032170981（fileKind=4）
  - 令和3年度: 000040033327（fileKind=4）
  - 令和4年度: 000040161427（fileKind=4）
  - 令和5年度: 000040277713（fileKind=4）
- 生成データ:
  - `outputs/runs/20260126_153831_pref_bathroom_heater_vs_central_heating_region_proxy/hokkaido_central_heating_timeseries.csv`
  - `outputs/runs/20260126_153831_pref_bathroom_heater_vs_central_heating_region_proxy/hokkaido_central_heating_timeseries_table.md`

---

## 4. 検索ログ（セントラル暖房システム×浴室暖房の関係）

### 4.1 J-STAGE

- 検索日: 2026-01-26
- ログ: `refs/search/20260126_174114_central_heating_bathroom_heater_literature_jstage.json`
- 要約: 「セントラルヒーティング/セントラル暖房/全館・全室・温水暖房」「浴室暖房/浴室暖房乾燥機」「脱衣室温度」「ヒートショック」等で多数ヒット。上位タイトルには住宅の温熱環境や暖房方式、浴室・脱衣室の温熱環境、ヒートショック関連が含まれるが、セントラル暖房と浴室暖房の**関係**や都道府県別普及率を直接扱う記述は、タイトル・メタ情報の範囲では未確認（本文確認が必要）。

### 4.2 CiNii Research

- 検索日: 2026-01-26
- ログ: `refs/search/20260126_174215_central_heating_bathroom_heater_literature_cinii.json`
- 要約: セントラル暖房・全館/全室暖房・温水暖房に関する技術/実測研究が複数ヒット。浴室暖房乾燥機の実態調査（業界誌/ニュースレター）や、浴室・脱衣室温熱環境/入浴事故に関する研究題目が上位に含まれる。セントラル暖房と浴室暖房の**関係**や普及率（地域差）を直接扱うものは、タイトル・メタ情報の範囲では未確認（本文確認が必要）。

### 4.3 PubMed

- 検索日: 2026-01-26
- ログ: `refs/search/20260126_174305_central_heating_bathroom_heater_literature_pubmed.json`
- 要約: 日本語語句は該当0件。英語語句は0件が中心で、2件は検索結果ページが判定できず `parse_failed` となった（検索語とURLはログ参照）。セントラル暖房と浴室暖房の関係を直接扱う論文は現時点では未確認。

### 4.4 上位ヒットの本文スクリーニング（J-STAGE/CiNii）

- スクリーニング日: 2026-01-26
- ログ: `refs/search/20260126_213530_central_heating_bathroom_heater_screening.json`
- 要約:
  - J-STAGEの浴室/脱衣室関連論文（PDF本文）では、「セントラル/全館/中央暖房」と「浴室/脱衣室暖房」の直接的な関係記述は確認できなかった。
  - CiNiiのセントラル暖房住宅に関する記録は、題名・要旨の範囲では浴室/脱衣室暖房との直接的関係は未確認（本文アクセスの可否は別途確認が必要）。
- 取得PDF:
  - `refs/pdfs/bathroom_dressingroom_cfd_shasetaikai2016_6_17.pdf`
  - `refs/pdfs/dressingroom_heating_elderly_aija63_509.pdf`
  - （未取得）J-STAGE 500エラー: https://www.jstage.jst.go.jp/article/shasetaikai/2017.6/0/2017.6_219/_pdf/-char/ja

### 4.5 追加本文スクリーニング（J-STAGE）

- スクリーニング日: 2026-01-26
- ログ: `refs/search/20260126_214303_central_heating_bathroom_heater_screening_followup.json`
- 要約:
  - セントラル暖房住宅の運転実態調査（北海道）の要旨では浴室/脱衣室・浴室暖房への直接言及は未確認。PDFリンクは取得できず本文未確認。
  - 東北地域のセントラル暖房住宅の実測調査PDFでは、脱衣室を含む非居室の測定点が示されるが、浴室暖房に関する直接言及は確認できなかった。
- 取得PDF:
  - `refs/pdfs/central_heating_hokkaido_aije73_628_767.pdf`
  - `refs/pdfs/central_heating_tohoku_aijt8_15.pdf`

### 4.6 追加本文スクリーニング（CiNii経由: DOI/IR/NDL）

- スクリーニング日: 2026-01-26
- ログ: `refs/search/20260126_215129_central_heating_bathroom_heater_screening_cinii_route.json`
- 要約:
  - 北海道のセントラル暖房住宅の運転実態調査（IR PDF）では、非居室（洗面・トイレ等）の暖房空間化が言及されるが、浴室/脱衣室暖房との直接的関係記述は確認できなかった。
  - 東北地域のセントラル暖房住宅の実測調査（J-STAGE PDF）では、測定点に脱衣室が含まれる旨が記載されるが、浴室暖房との直接的関係は確認できなかった。
- 取得PDF:
  - `refs/pdfs/central_heating_hokkaido_aije73_628_767.pdf`
  - `refs/pdfs/central_heating_tohoku_aijt8_15.pdf`

### 4.7 追加本文スクリーニング（CiNii経由: DOI/IR/NDL その2）

- スクリーニング日: 2026-01-26
- ログ: `refs/search/20260126_215744_central_heating_bathroom_heater_screening_cinii_route2.json`
- 要約:
  - 浴室・脱衣室温熱環境に関する研究（長野県の実態調査）では、浴室/脱衣室暖房の有無は扱うが、セントラル/全館/中央暖房との直接的関係は確認できなかった。
  - 全国の浴室温熱環境調査では地域差は示すが、セントラル/全館/中央暖房との直接的関係は確認できなかった。
  - 近畿地区の浴室環境調査では浴室設備（換気・乾燥・暖房機能等）の有無が中心で、セントラル/全館/中央暖房との直接的関係は確認できなかった。
- 取得PDF:
  - `refs/pdfs/bathroom_dressingroom_nagano_shasetaikai2019_6_317.pdf`
  - `refs/pdfs/bathroom_environment_national_jhesj14_1_11.pdf`
  - `refs/pdfs/bathroom_environment_kinki_jhej52_10_995.pdf`
