# 平均スペクトルを補助入力にするホストPoC

2026-11-19。特徴/整数モデル/学習処理を先行TDDで実装した。**固定1構成の実データ学習は完了。17/24条件の合格で基準未達のため不採用。C/ESP32-S3は未確認。** [元の設計案](spectral-envelope-proposal.md)の未実装JSONは当時の記録として保持する。

## 入力と形式

`envelope_frontend.features`は旧2,113 float32値のprefixをそのまま残し、平均除去前のlog-mel32bandの時間平均を4で割り、±16に制限して末尾へ追加する。前後10frame/HP160/level4096/発話長の意味は維持する。補助値はpeak正規化後のスペクトルで、物理的な音量ではない。

| 項目 | ホストで確認した値 |
|---|---:|
| 入力 | 2,145 float32 / 8,580B |
| Head | 7×3,169 |
| 学習parameter | 23,134 |
| 数値モデル＋5閾値 | 23,635B |
| C用activation交互領域 | 推計12,288B。Cの実測ではない |

実験NPZは18配列で、head shapeと`logmel-dscnn-envelope-v1`/`context-logmel64-envelope`を識別する。`dscnn_model`/Torch処理の`envelope=True`を明示した場合だけ新形式を受け付け、既定の旧2,113入力/旧headと相互に拒否する。旧形式の通常テストも維持した。正式なC用reader/export/freezeはまだない。

## TDDと回帰

| 対象 | 先行Red | Green |
|---|---|---|
| 特徴4契約 | 未実装import | 4件/0.051秒 |
| 整数モデル5契約 | 未対応keyword | 5件/0.133秒 |
| Torch構成/export/fit4契約 | 未対応keyword | 4件/1.083秒 |
| 配列/保留3契約 | 新frontend未対応の2失敗。既存拒否1件は初回から成功 | 3件/1.812秒 |
| 候補処理4契約 | 未対応keyword | 4件/0.054秒 |

新規20契約。prefix byte不変、独立したスペクトル/整数scalar計算、無音/境界/−32768、署名/形状/値域、入力不変、BN fold、CPU repeat、step/期限、保留の元ID、元archive不変、元neural検証での拒否を確認する。

回帰309件/34.834秒、学習25件/3.581秒、SDK2件/2.583秒がskipなし成功。旧v3凍結13ファイルも不変。別の[MPS小型fixture](../experiments/neural-domain-20261119/spectral-envelope-mps-smoke.json)では2回の4step学習/量子化層/整数logitが一致した。実装後の追加検証であり、新規TDDや実データの品質保証とは扱わない。

```sh
uv run --locked python -m unittest discover -s tests -p 'test_envelope_*.py' -v
uv run --locked --group train python -m unittest discover -s tests/training -p test_envelope_training.py -v
```

[旧形式の実データ互換](../experiments/neural-domain-20261119/spectral-envelope-legacy-compatibility.json)も追加確認した。変更前の固定実装と既定モードで4,808行の全整数logit/24混同行列が一致。旧モデルは不採用のままで、追加検証を新規TDDや実機保証とは扱わない。

## 固定した1構成

[実学習前のrecipe](../experiments/neural-domain-20261119/spectral-envelope-plan.json)。同じ元archive/4合成条件と20無音の保留、30epoch/batch256/seed42、ローカルMPS、最大8,000step/fit900秒。float学習後の整数版でcheckpoint/点差を選び、全24domain＋結合8条件＋選定結合8条件を要求する。ASRの不一致を理由に評価行を削除/再ラベルせず、過去の最終2組をfit/選定/校正へ使わない。

実sourceと容量を確認してからrecipeを保存した。事前JSONの`real_training_started:false`を後から変更しない。通過した場合だけ、新形式のC照合/固定、新しい独立最終評価、S3 build/資源確認へ進む。実機a553/16app・実声6/15と過去の不採用は維持する。新FWの個別許可は別工程。

## 固定1構成の結果

[数値と別照合](../experiments/neural-domain-20261119/spectral-envelope-result.json)。source `9260efbd1a65165ffce6f771209fcfa3f8f6995d`をarchiveし、同じ事前recipeで30epoch/7,350stepを完走。全処理376.242秒、選定epoch6。23,635B、process最大RSS1,758,101,504Bはホスト値で、S3資源の実測ではない。

| 開発domain | 最小F1（基準0.90） | 最大誤反応（上限0.05） | 最小操作recall（基準0.80） | 未達profile |
|---|---:|---:|---:|---|
| legacy | 0.919074 | 0.018018 | 0.854167 | なし |
| 保留neural | 0.886878 | 0.048544 | 0.800000 | quiet/hum/noise/early/late |
| 元neural検証 | 0.868433 | 0.060000 | 0.800000 | noise/early |

全24条件のうち17条件が合格し、旧モデルの21/24を下回った。selectionも未達で、不採用として保存する。56組の指標を混同行列から再計算し、30source/8元入力/重み/recipe hashと容量、旧modeで新headを拒否することも照合した。quality driverのexit1は基準未達で、学習処理の例外ではない。

追加のclean「まって」12例では、棄却前1位の正解が旧モデル6/12から11/12へ増えた。ただし同じ4元発話のvariantで、実声や独立最終評価ではない。局所的な改善と他条件の悪化を分け、平均スペクトルを除いていたことだけが主因とは断定しない。

次は旧CNNの既存係数を固定し、追加32値に対応する224係数だけを学習する仮説を検討する。全体の再学習による判定の変化を切り分け、全評価基準を維持する。別の学習recipe/契約はまだ未固定で、追加探索は開始していない。

新しい最終データ/C移植/target build/実機変更は行わず、実機a553/16app・実声6/15、既存の不採用とVast.ai累計0米ドルを維持する。

次の[固定係数による補助入力案](frozen-prefix-residual-proposal.md)は未実装/未学習。元の不採用を保持し、recipeは実装後に別途固定する。
