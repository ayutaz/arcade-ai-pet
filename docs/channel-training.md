# 入力条件を加えた学習

2026-11-18。入力条件8種類を追加し、音量正規化を含む2候補をCPUで学習した。
**両方ともvalidation不合格。配布モデル・OS・実機は更新していない。**

## 固定した手順

[入力条件の比較](channel-robustness.md)と同じ8条件、同じ採用ゲートを使う。
元発話のtrain/validation分離を保ち、各例へ8条件を適用。加工後、平均除去またはHP80→平均除去を行い、ピーク4,096へ整数正規化して既存MFCCへ渡す。
正規化は失われた情報を復元する処理ではない。全ゼロ入力は全ゼロのまま維持する。

| 項目 | 条件 |
|---|---|
| fitへ渡す例数 | train 15,976、validation 2,160 |
| 元データのtest | fit/閾値選択に使用しない |
| 最終評価用データ | 未評価。選択条件が不合格のため開かない |
| 学習 | PyTorch CPU、2threads、seed42、600epochs、260→64→7 |
| 選択 | validationでepochを選び、その後8条件を満たすか確認。閾値もvalidationだけで選択 |
| 重み | 両候補とも17,380B。24KiB以内だが、容量だけで採用しない |
| 料金 | 新規GPU/APIとも0米ドル |

## validation結果

| 条件 | 平均除去＋正規化 F1 | HP80＋正規化 F1 |
|---|---:|---:|
| clean | 1.000 | 0.986 |
| quiet | 1.000 | 0.966 |
| dc | 0.991 | 0.966 |
| drift | 0.214 | 0.907 |
| hum | 0.587 | 0.701 |
| noise | 0.530 | 0.480 |
| early | 0.976 | 0.967 |
| late | 1.000 | 0.986 |

HP80候補でもnoiseの操作recallは0.333、誤起動0.104。driftの操作recallは0.792で、F1だけを見て合格にはしない。
単純なデータ追加と音量正規化では不十分だった。次は雑音がある録音での発話区間選択を切り分ける。
この仮想ストレスが実機録音と同じとは未確認で、実機の原因確定や実声の精度として使わない。

[平均除去候補](experiments/channel-center-v1-rejected.json)・[HP80候補](experiments/channel-hp80-v1-rejected.json)に数値を保存。
既存KWS v2のJSON/重みは当時のまま維持する。今回の実験重みはGit対象外checkpointsに保持する。

## 再現とTDD

```sh
uv run --locked python -m unittest discover -s tests -p test_channel_training.py -v
uv run --locked --group train python -m unittest discover -s tests/training -p test_channel_training.py -v
uv run --locked --group train python -m arcade_ai_pet.channel_training \
  --dataset datasets/jtalk-aligned-v3 --candidate hp80 \
  --epochs 600 --output checkpoints/fresh-channel-hp80
```

ホスト3件はmodule未実装の先行Red、学習recipe1件は`train_channel_candidate`未実装の先行Redから実装した。
試験入力の振幅を小さくしすぎ、加工時の整数化で全ゼロになった初期fixtureは修正した。この失敗を製品のRedや情報復元の成功として記録しない。
学習recipeは学習用依存groupの試験に置き、標準CIへPyTorchを暗黙要求しない。
本体音声の外部取得、Flash操作、GPIO変更、クラウドGPU契約を行わない。
