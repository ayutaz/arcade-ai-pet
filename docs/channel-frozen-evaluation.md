# 固定候補の最終合成評価

2026-11-18。**重み・前処理・閾値・13ファイルのコード/lockを固定した後、8条件すべてで最終合成評価を通過した。**
評価に合わせた学習/閾値の再調整は行っていない。C移植・S3 build・実機・実声は別工程として残る。

## 固定と検査

[hum3/noise5候補](channel-balanced-candidate.md)をdevelopment validationから再計算し、保存された成功flagだけで固定しない。
重みは23,884B、threshold=995、HP160→平均除去→peak4,096→adaptive MFCC。

- `weights.npz` SHA: `374987a02446e805f18427d7de8de3cd955c4833efc6e0a9dbd9fa655c3c59f2`。
- コード/lock 13件と重みの同一性、元発話ID/WAV hashの重複を最終波形を開く前に検査。
- 固定JSONは作成時の`final_evaluated=false`のまま保持し、結果を別のファイルへ保存する。
- 旧KWS v2にも使った固定合成holdoutを再利用する。この新候補の学習/選択には使用していない。新しく収録した実声とは呼ばない。
- 956例×8条件=7,648評価。元の音声は2合成話者/452source。条件間で同じ元発話を再利用し、7,648独立話者とは呼ばない。

## 結果

| 条件 | F1 | 操作recall | 対象外/無音の誤起動率 |
|---|---:|---:|---:|
| clean | 0.997864 | 0.994444 | 0.000000 |
| quiet | 0.996414 | 0.994444 | 0.000000 |
| dc | 0.997382 | 0.994444 | 0.000000 |
| drift | 0.995928 | 0.994444 | 0.000000 |
| hum | 0.927506 | 0.861111 | 0.009021 |
| noise | 0.916143 | 0.855556 | 0.012887 |
| early | 0.987067 | 0.972222 | 0.001289 |
| late | 0.997864 | 0.994444 | 0.000000 |

[最終JSON](experiments/channel-balanced-final-v1.json)。全条件でF1>=0.90、誤起動<=0.05、操作recall>=0.80を通過。
ホストのPython/整数推論の結果。Cとの全経路一致やESP32-S3の速度、マイク/実声の品質を保証する結果ではない。
既存配布KWS v2の結果・重みと、実機c032は変更していない。

## TDDと再現

先行3試験のmodule未実装Red後、固定時の独立再計算、源の変更/重複拒否、閾値を再選択しない最終評価を実装した。
さらに別ID同一WAV・改変weightの拒否を既存実装へ追加検証。後者を先行TDDに数えない。
標準87件/学習8件が成功。模擬finalの満点はツールの契約だけで、上の実験結果とは分ける。

学習・原本生成は[候補手順](channel-balanced-candidate.md)・[合成手順](reproduce-model.md)に従う。
次は`uv run --locked python`から実行する例。毎回未存在の出力名を使い、失敗も削除しない。

```python
from arcade_ai_pet.channel_freeze import freeze_channel, evaluate_channel_frozen

freeze_channel("checkpoints/fresh-balanced", "datasets/jtalk-aligned-v3",
               "checkpoints/fresh-frozen")
evaluate_channel_frozen("checkpoints/fresh-frozen", "datasets/jtalk-heldout-v1",
                        "reports/fresh-final")
```

最終データを見た後の新候補は、この結果へ閾値を合わせず別の独立評価も用意する。
本体PCM外部取得、Flash操作、GPU/API契約は行っていない。今回の新規GPU/API課金は0米ドル。
