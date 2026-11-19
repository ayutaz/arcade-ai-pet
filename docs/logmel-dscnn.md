# log-melと2D小型CNNのホストPoC

2026-11-19。**特徴/整数演算/学習停止の契約とMacの予備計測を確認。実データの学習/品質とC/実機は未確認。** [実学習前の固定条件](../experiments/neural-domain-20261119/logmel-dscnn-plan.json)・[前候補の不合格](caption-holdout.md)。

低資源KWSのDS-CNNは[Hello Edgeの原論文](https://arxiv.org/abs/1711.07128)を参考にした独自構成。原論文の成績を本機/日本語の実績とは扱わない。

## 入力と整数演算

確認済みのHP160/peak4096と同じ発話区間を使い、前後100msを保持する。log-melを64時間区間×32bandへ面積で重み付け平均し、短い区間でも空binを作らない。band平均を除いて4で割り、energy64値/発話長1値を加えた2,113値。旧261入力とは別形式で扱う。

| 項目 | ホストで確認した内容 |
|---|---|
| 特徴 | 2,113 float32値、8,452B |
| CNN | Conv2d＋2組のdepthwise/pointwise、BatchNorm、ReLU6、7出力 |
| 学習parameter | 22,910 |
| export | BatchNorm fold、channel別int8重み/int32 bias/shift |
| 整数演算 | signed Q5入力、uint8 Q5/ReLU6中間値、Q10出力 |
| 数値モデル＋5閾値 | 23,411B |
| C用の交互作業領域 | 設計上12,288B。C実装/総RAMの実測ではない |

NumPyの整数演算を別のPython整数/ループ実装と全6層で照合した。負の除算、同点丸め、出力channel別shift、ReLU6、int32飽和、batch分割、入力不変を検査する。NumPyの一時配列が12KiBで動くという意味ではない。

## 先行TDD

| 対象 | 先行Red | Green |
|---|---|---|
| log-mel特徴6契約 | module未実装 | 6件/0.093秒 |
| 整数CNN6契約 | module未実装 | 6件/0.232秒 |
| 量子化4契約 | module未実装 | 4件/0.035秒 |
| 配列2契約＋保留dispatch2契約 | module未実装/未対応引数 | 2件/1.711秒、保留の新旧10件/2.380秒 |
| Torch構成/BN fold3契約 | module未実装 | 3件/0.413秒 |
| CPU fit/期限/step上限3契約 | 未実装属性/関数 | 3件/1.780秒 |
| 候補処理7契約 | module未実装 | 7件/0.101秒 |

新規は計33契約。回帰279件/33.608秒、training21件/3.485秒、pinned SDK2件/3.076秒がskipなし成功した。旧v3凍結13ファイルは不変。実装後の追加計測を新規TDDとは数えない。

## Macの予備計測

[数値](../experiments/neural-domain-20261119/logmel-dscnn-host-profile.json)・[source関係](../experiments/neural-domain-20261119/logmel-dscnn-profile-provenance.json)。M1/arm64、NumPy2.5.5/Torch2.15.0で、乱数入力のbatch256を3step warmup後に5step計測した。実データや本体音声を使わず、モデルを保存していない。

| backend | 学習1stepの中央値 | 整数参照64例 |
|---|---:|---:|
| CPU | 0.152秒 | 0.017秒 |
| ローカルMPS | 0.029秒 | 0.030秒 |

process最大RSSは最終時点548,945,920B。先に行ったCPU処理も含むhigh-waterで、MPS単体や全実データの消費量ではない。別の[小型fixtureのMPS repeat](../experiments/neural-domain-20261119/logmel-dscnn-mps-smoke.json)も2回完走し、量子化層/整数logitが一致した。ESP32-S3の速度や認識精度は未測定。

## 次の1比較

同じcaption保留と元archiveを使い、30epoch/batch256/seed42、MPS、最大8,000step/fit900秒の1構成を固定した。float学習後のBN fold/量子化結果で候補を選び、全24domain＋結合8条件と選定2domainの結合8条件を要求する。実学習前の`training_started:false`を後から変更しない。

```sh
uv run --locked --group train python -m unittest discover -s tests -p 'test_dscnn_*.py'
uv run --locked --group train python -m unittest discover -s tests -p 'test_logmel_*.py'
uv run --locked --group train python -m unittest discover -s tests/training -p 'test_dscnn_*.py'
```

学習の配列/fit/選定はホスト専用で、期限超過や入力/source改変時は途中の候補を保存しない。旧reader/runtimeへ流用せず、採用には新形式のC照合/固定と新しい独立最終評価、S3資源確認が必要。実機a553/16app、実声6/15、過去の失敗と課金0米ドルを維持する。
