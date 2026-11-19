# 合成音声の処理段階ごとの内容点検

2026-11-19。[固定した条件](../experiments/neural-domain-20261119/utterance-stage-plan.json)で、保留4合成条件の「まって」を元音声から切出しまで追跡する。元の不採用や評価行を変更せず、既存のASRで内容がどう認識されるかを点検する。

## TDDと取り出す段階

`utterance_stages.prepare`は16kHzへresample済みのPCMを受け取り、既存のseedによる3variantと、HP160/level4096後の特徴抽出範囲を取り出す。認識・学習・ファイル保存は行わない。

| 段階 | 4条件の点検数 | 内容 |
|---|---:|---|
| source | 4 | 元の48kHz WAV |
| resampled | 4 | 既存`read_pcm`の16kHz化 |
| trimmed | 4 | 既存の無音除去後 |
| augmented | 12 | 既存3variantの拡張後 |
| conditioned | 12 | HP160/level4096後、2秒全体 |
| cropped | 12 | 特徴抽出と同じ発話区間＋前後10frame |
| control-source | 4 | 同じ条件の「こんにちは」の元WAV |

先に4契約を追加し、未実装moduleのimport失敗をRedとして確認。最小実装後4件/0.485秒が成功した。元の拡張/conditionとbyte一致、別scalarのenergy/閾値/区間計算、frame窓の終端、入力不変、−32768を含む固定gain、不正入力の早期拒否を検査する。回帰290件/33.239秒、学習21件/3.496秒、SDK2件/2.922秒もskipなし成功。

```sh
uv run --locked python -m unittest discover -s tests -p test_utterance_stages.py -v
```

## 内容点検の条件と限界

最大52呼出し/CPU600秒、既存Whisper model/lockをhashで固定し、ダウンロードやTTS再生成を行わない。ASR用コピーだけをpeak16,384へ正規化し、元データと学習入力は保持する。ASRには音声とsample rateだけを渡し、prompt/labelを与えない。事前の日本語/beam5/VADなし等の設定を維持する。

ASRは分類器から独立したモデルだが、元音声を採用するときに使ったASRと同じ。元音声との一致を新しい独立最終評価や人の正解判定とは扱わない。変換後に不一致が出ても誤ラベルと即断せず、評価行を削除/変更しない。PCMはローカルの合成音声だけで、実機録音の取得や外部転送は行わない。

内容の一致と小型CNNの正解は別の指標。[丸めなし比較](dscnn-shadow.md)や実声6/15を成功へ置き換えず、次の仮説を絞る診断として使う。
