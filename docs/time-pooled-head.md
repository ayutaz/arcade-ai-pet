# 時間方向pooling headのホスト実装

2026-11-19。先行TDD15契約と回帰374件が成功し、実データの前に[1構成のrecipe](../experiments/neural-domain-20261119/time-pool-plan.json)を固定した。固定1構成の実学習は完了したが15/24条件で不採用。C/ESP32-S3は未確認。[初期設計](time-pooled-head-proposal.md)の未実装JSONは作成時の記録として維持する。

## 方式と容量

入力2,145値はenvelope版と同じ。5層の出力N×24channel×16time×8bandを、連続する時間4cellずつで平均化する。整数版はuint16へ和を取り、4でfloor除算してuint8 Q5へ戻す。band方向は平均化しない。

| 項目 | ホストで確認した値 |
|---|---:|
| pool後 | 24×4×8＝768値 |
| 補助入力 | energy64＋duration1＋band平均32＝97値 |
| head | 7×865 |
| 学習parameter | 7,006 |
| 数値モデル＋5閾値 | 7,507B |

`envelope=True,time_pool=True`の両方を明示した場合だけ新headを受け付ける。classifierは`logmel-dscnn-timepool-v1`、特徴のfrontendは既存`context-logmel64-envelope`。旧headとの混在と、固定prefix補助学習との混在を拒否する。新方式はfloatで全体を学習し、既存係数を固定する方式ではない。

Torchの学習では連続値の平均を使い、export後の整数版ではQ5/floorを使う。両方の数値が同一とは主張せず、checkpoint/校正/最終の開発指標は整数版で評価する。Cのreader/export/freezeと総RAM/速度は未実装/未測定。activationの12,288Bは従来の設計推計のままで、S3実測ではない。

## TDDと回帰

| 対象 | 意図したRed | Green |
|---|---|---|
| 整数処理5契約 | 未対応time_poolと未実装pool関数 | 5件/0.216秒 |
| Torch/export/fit5契約 | 未対応time_pool | 5件/1.109秒 |
| 候補処理5契約 | 未対応time_pool | 5件/0.071秒 |

```sh
uv run --locked python -m unittest discover -s tests -p 'test_time_pool_*.py' -v
uv run --locked --group train python -m unittest discover -s tests/training -p test_time_pool_training.py -v
```

独立scalarによる全整数CNN/軸/4cell平均/floor、signed tailの順序、shape/容量/旧形式拒否、Torch側のfloat平均、BN/export不変、小型CPU repeat、step/期限、保留の元IDと元neural回帰での拒否を確認した。回帰337件/36.394秒・学習35件/3.946秒・固定SDK2件/2.769秒がskipなし成功。旧v3凍結13sourceもbyte不変。

[旧2形式の追加互換検査](../experiments/neural-domain-20261119/time-pool-legacy-compatibility.json)は各4,808件、計9,616行で旧実装と全整数logitが一致。envelope側の追加32値は決定的な範囲内のprobeで、実音声から再計算した平均値ではない。[MPS小型fixture](../experiments/neural-domain-20261119/time-pool-mps-smoke.json)の2回/各4stepも全量子化層が一致した。いずれも実装後の追加検証で、TDDや実データ品質の代わりとはしない。

## 実学習前の固定条件

同じ元archive・保留4合成条件/20無音・全24domain・結合8条件・選定結合8条件を維持する。30epoch/batch256/seed42、ローカルMPS、最大8,000step/fit900秒、予定7,350step。Adam0.003/weight_decay0.0001、逆頻度class weight。語別の手作業ルーティングや評価行の削除/再ラベル、消費済み最終2組の再利用を行わない。

全基準通過後だけ、新形式のC照合/正式freeze/新独立最終評価/S3資源へ進む。実機a553/16app・実声6/15、既存候補の不採用とVast.ai累計0米ドルを維持する。新FWの個別許可は別工程。

## 固定1構成の結果

[実測と別照合](../experiments/neural-domain-20261119/time-pool-result.json)。source `7d4660ab7d918913ea5a55d353e5a10341257276`をarchiveし、30epoch/7,350stepを完走。全処理381.421秒、fit299.585秒、選定epoch26。数値7,507B、ホスト最大RSS1,518,747,648Bで、S3の総RAM/速度の実測ではない。

| 開発domain | 最小F1 | 最大誤反応 | 最小操作recall | 未達profile |
|---|---:|---:|---:|---|
| legacy | 0.886875 | 0.018018 | 0.812500 | noise |
| 保留neural | 0.871754 | 0.033981 | 0.800000 | 全8条件 |
| 元neural検証 | 0.917413 | 0.020000 | 0.866667 | なし |

**全24条件中15条件が合格し、基準未達で不採用。** 旧log-melの21/24、平均スペクトル版の17/24を下回った。小型化は確認できたが、未知の合成条件への認識は改善していない。

56組の指標と採否を混同行列から再計算し、32source/8元入力/重み/recipe hash、7×865のhead/容量/旧形式での拒否を照合した。quality driverのexit1は品質未達による終了で、学習処理の例外ではない。新C/正式freeze/新独立最終/target/実機へ進めない。

次は[floatと整数の対比較](float-checkpoint-probe-proposal.md)を先行TDDで用意する。今回も元のfloat checkpointの判定を残していないため、学習した分類自体と量子化のどちらが影響したかは確定できない。既存のdequantized shadowは、量子化前の重みを復元したものではない。新しいhead形状を追加探索する前に、この観測不足を解消する。

実機a553/16app・実声6/15、過去の不採用とVast.ai累計0米ドルを維持する。
