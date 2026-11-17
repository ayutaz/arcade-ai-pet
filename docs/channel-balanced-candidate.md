# ハム音と独立雑音の比率調整

2026-11-18。**humを3回・独立noiseを5回にした候補が、validationの全8条件を通過した。**
独立した最終合成データ、C照合、S3 build、実機/実声はまだ未検証。配布モデルと実機FWは変更していない。

## 固定した候補

| 項目 | 値 |
|---|---|
| 前処理 | HP160→平均除去→ピーク4,096へ整数正規化 |
| 特徴 | adaptive区間選択。noise power差引きは不採用 |
| モデル | 260→88→7、int8 weight/int32 bias、23,884B |
| 学習 | CPU2threads/seed42、1,000epochs上限から856を選択 |
| train / validation | 27,958 / 2,160例 |
| 拒否閾値 | 995。validationの整数境界から選択 |
| train比率 | hum3回、noise5回、その他各1回。validationは各1回のまま |

humは同じ入力の繰り返しで、多様なハム音を合成したとは呼ばない。
noiseは元の1回＋独立seedの4回。元発話splitとvalidation入力は以前から維持する。

| 条件 | F1 | 操作recall | 対象外/無音の誤起動率 |
|---|---:|---:|---:|
| clean / quiet / dc / late | 1.000000 | 1.000000 | 0.000000 |
| drift | 0.993071 | 1.000000 | 0.000000 |
| hum | 0.922442 | 0.833333 | 0.004505 |
| noise | 0.905626 | 0.854167 | 0.009009 |
| early | 0.986653 | 0.979167 | 0.000000 |

[候補の数値](experiments/channel-hum3-noise5-validation-candidate.json)。
hum5/noise5の対照候補は不採用。[対照の数値](experiments/channel-hum5-noise5-rejected.json)。
どの条件もF1>=0.90、誤起動<=0.05、操作recall>=0.80の固定ゲートを満たす候補だけを次へ進める。

`weights.npz` SHA-256: `374987a02446e805f18427d7de8de3cd955c4833efc6e0a9dbd9fa655c3c59f2`。
実験重みはGit対象外checkpointsに保存した。既存KWS v2をこの重みへ置き換えていない。

## TDDと次の工程

humのtrainだけの件数、noiseとの独立した比率、validation不変、同じhumの再現、過大値のI/O前拒否を先にテスト。
未対応keywordのRed後に実装し、標準83件/学習8件が成功した。
品質の通過は実験結果に基づき、契約テストの成功だけで認識品質を保証しない。

```sh
uv run --locked --group train python -m arcade_ai_pet.channel_training \
  --dataset datasets/jtalk-aligned-v3 --candidate hp160 --frontend adaptive \
  --hidden 88 --selection worst --noise-replicas 5 --noise-mode fresh \
  --hum-replicas 3 --epochs 1000 --output checkpoints/fresh-balanced
```

次は重み/閾値/コード/前処理を固定し、最終合成データを評価する。結果へ合わせて閾値を再調整しない。
通過後だけC移植/数値照合、S3 build/資源検査、具体的な固定レビューを揃える。新FW適用は別の個別許可が必要。
新規GPU/API課金は0米ドル。本体録音の外部取得は行っていない。
