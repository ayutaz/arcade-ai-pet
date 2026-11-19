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

## 初回52件の点検結果

[結果](../experiments/neural-domain-20261119/utterance-stage-result.json)。source `9b68fe0a3c5778ccdf54958da6fefea9c7ccf7d5`のarchiveを使い、52呼出し/CPU66.947秒で完走した。12variantのconditioned特徴は保存済みvalidationと一致し、元音声/manifest/モデル/ASR資産を保持した。

| 段階 | ASRとの完全一致 |
|---|---:|
| 元source / resampled / trimmed | 各3/4 |
| augmented | 6/12 |
| conditioned | 5/12 |
| cropped | 4/12 |
| 対照の「こんにちは」 | 4/4 |

extension-03のvariant0は全段階で「待って」と認識されたが、小型CNNは対象外1位だった。extension-13のvariant1はconditionedで「まって」、croppedで「まい」となった。ASRの不一致だけで誤ラベルとは断定せず、評価行の除外/変更は行っていない。元音声の1条件でも不一致があり、gainやASR側の影響も残る。

## gainと文脈の追加比較

[追加条件](../experiments/neural-domain-20261119/utterance-context-plan.json)は28呼出し。元の音量のsource4件と、同じ2秒/同じ位置のまま文脈だけを変えた各12件を比較する。conditioned全体を一度だけpeak16,384に合わせ、発話区間±10frame/±20frameの外側を0にする。可変長入力と位置の変化を避け、二つの文脈でgainを変えない。

`padded_context`の3契約も先行の未実装import Red後に最小実装。3件/0.019秒が成功し、同じ位置の保持、frame窓の終端、gainなし、範囲外0、入力不変と不正入力を確認した。初回52件のsource/事前条件を後続の成功で上書きしない。この比較も診断用で、長い文脈を学習候補へ採用する許可や精度保証ではない。

追加後の回帰293件/33.705秒、学習21件/3.161秒、SDK2件/2.397秒もskipなし成功。`uv run --locked python -m unittest discover -s tests -p test_padded_context.py -v`で追加3契約を再現できる。

## 追加28件の点検結果

[結果と別照合](../experiments/neural-domain-20261119/utterance-context-result.json)。source `7e84cf9bc8829ed40224f393d2ee6c49b3d108c1`を固定し、28呼出し/CPU36.348秒で完走した。両実行の80件の転記/一致数を別再計算し、12組の比較音声は同じ位置/長さでgainが共通、文脈の外側だけが変わることをPCMで照合した。

| 追加比較 | ASRとの一致 |
|---|---:|
| 元の音量のsource | 4/4 |
| 同じ2秒/位置、前後10frame | 5/12 |
| 同じ2秒/位置、前後20frame | 4/12 |

元音声はgain正規化の有無で3/4と4/4に変わり、ASR側の感度がある。文脈20frameで新たに一致した組は0、逆に不一致となった組は1だった。したがって文脈を単に倍にする再学習は行わない。全80件を人の正解判定や新しい最終評価とは扱わず、評価行の削除/再ラベルもしない。

一方、extension-03/variant0は元の処理/切出しでもASRと一致し、CNNは対象外だった。入力内容の不確かさだけでは説明できない分類失敗も残る。次の仮説は、現行特徴が除いたband別の平均スペクトルを補助入力として残し、短い語の定常的な音の違いを保持できるかというもの。新しい特徴/数値形式のTDDと容量確認後に1構成を事前固定し、元の全評価条件を保持して比較する。現段階では設計仮説で、精度向上を確認した結果ではない。

元の不採用、実機a553/16app・実声6/15を維持する。新しい学習/閾値探索/C移植/最終評価/実機変更は行っていない。Vast.ai累計0米ドル。

次の[平均スペクトルを残すPoC案](spectral-envelope-proposal.md)は設計だけで、未実装/未学習。容量計算を実測や固定recipeと混同しない。
