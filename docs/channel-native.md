# 固定候補のC実装と数値照合

2026-11-18。**全7,648評価でPythonとCの最終判定が一致し、C側も品質ゲートを通過した。**
この監査はホスト検証。後続OS候補のS3 buildは成功したが、実機認識や処理時間の保証ではない。

| 検査 | 結果 |
|---|---|
| 同じ特徴を渡した整数logit | 7,648 / 7,648完全一致 |
| PCM→前処理→特徴→分類の最終判定 | 7,648 / 7,648一致 |
| 連続PCMと8分割PCMの特徴 | 7,648 / 7,648完全一致 |
| Pythonとの特徴最大差 | 0.000004291535 |
| Cの重み / scratch | 23,884B / 5,680B |
| 品質 | 全8条件でF1>=0.90、誤起動<=0.05、操作recall>=0.80 |

[Native JSON](../models/kws-v3/native-evaluation.json)・[モデルカード](../models/kws-v3/model-card.md)。
モデル/閾値は[最終合成評価](channel-frozen-evaluation.md)の前に固定したまま。C移植のために重みや閾値を変更していない。

## 実装の範囲

- `channel_prepare.c`: HP160→平均除去→peak4,096。連続/分割のprivate PCMをin-place処理。
- `channel_features.c`: adaptive区間と既存MFCC数学処理。FFTのreal領域をunionで198個のdoubleの並べ替えにも使う。
- sortは最大19,503比較の挿入sort。qsort/malloc系の未定義symbolがないことをホストbinaryで確認。
- `inference.c`: 既存の整数engineを再利用。生成headerのhidden88/threshold995を読む。
- 8bankは先頭7本が4,096samples、最後が3,328samples以上。trusted caller-owned pointerのみを扱い、guest pointerや任意長PCMを受けない。
- v3は専用`channel-metadata.json`を使い、v2 runtimeへ誤って渡すmetadataを生成しない。

Pythonの比較wrapperは入力を保護するためcopyする。C本体の追加PCM buffer不要という性質を、Python wrapperのRAM量やOS全体の実heapへ拡大しない。
モデルcardに記載した出力1,040B・整数推論stack等も、5,680B scratchとは別。

## 公開用の成果物

`models/kws-v3`へ重み・数学table・model header・固定/評価JSON・56照合特徴を追加した。
weights/model header/派生特徴はCC BY 3.0、手書きcodeと数学tableはApache-2.0。[条件](data-license.md)。
本体録音・原本TTS weight・資格情報・firmware binaryは含めない。
このC監査時点では既存のv2と実機c032を維持した。後続のe591/v3適用と未達の音響比較は[実機記録](device-validation.md#v3の適用と音響比較)へ。

## TDDと回帰

| 対象 | Red → Green |
|---|---|
| export/prepare/特徴/推論 | 新module未実装 → 3契約成功 |
| 追加allocator依存 | qsortのsymbol検出 → workspace内の挿入sortで成功 |
| 全経路benchmark | runner未実装 → C一致と低品質モデルを別判定する契約成功 |
| v3公開asset | asset/照合NPZ未作成 → identity/予算/56整数照合成功 |

benchmarkの最初のmock参照不足は準備の不備として除外し、importを整えた後の未実装APIでRedを取り直した。
既存v2の数値table exporterは共通rendererへ整理し、既存asset再生成/改変拒否の試験も維持する。

```sh
uv run --locked python -m unittest discover -s tests -p test_channel_native.py -v
uv run --locked python -m unittest discover -s tests -p test_published_channel_model.py -v
```

全体のC監査は、ローカルの合成原本と固定checkpointを使い`uv run --locked python`から実行する例。

```python
from arcade_ai_pet.channel_native import export_channel, evaluate_native_channel

export_channel("checkpoints/fresh-frozen", "build/fresh-channel-native")
evaluate_native_channel("checkpoints/fresh-frozen", "build/fresh-channel-native",
                        "datasets/jtalk-heldout-v1",
                        "reports/fresh-final/channel-final-evaluation.json",
                        "reports/fresh-native-audit")
```

比較は固定モデルの再現/移植試験。同じ最終データで新しい重みを選び直す処理ではない。
新規GPU/API課金は0米ドル。監査時点ではtarget build未実施。その後、固定source48bf239のOS候補はS3 build/715件を通過した。[後続統合](../models/kws-v3/model-card.md#後続のos統合候補)。実heap/stack・遅延・実声30試行は未実施。

最終回帰は標準94件/8.082秒、学習8件/2.222秒、固定SDKの両VM統合2件/3.203秒で成功。
ホストcompilerはApple clang21.0.0。公開候補のC source5件と生成headerのSHAが、保存した実測JSONと一致することを別途照合した。
モデル/数学tableのlicense、lock、Markdown参照、差分、staged secret scanも確認する。
