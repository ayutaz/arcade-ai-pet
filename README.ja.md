# Arcade AI Pet

[Scratch Arcade OS](https://github.com/ayutaz/scratch-arcade-os)向けの、通信なしで遊べる相棒アプリです。
学習した行動で動くペットと遊び、短い日本語の5操作を試せます。
アプリは小さなWASM package、音声認識はOSのサービスとして動きます。

[English](README.md)

![学習した相棒](docs/media/companion.gif)

*実WASMのホスト再生です。認識結果は模擬イベントで、実機の録画ではありません。*

## 現在の状態

**実験版ソースとして公開します。実声の受入は未完了です。**
[リリースノート](docs/release-notes/v0.1.0.md)。

| 対象 | 確認結果 |
|---|---|
| ペット1.0.1 | package 5,176B。移動・認識結果への動作・終了を両ホストVMで確認 |
| 学習した移動 | policy 472B。未知64盤面で平均8.64点、固定乱数のランダム操作は0.97点 |
| [固定音声候補](models/kws-logmel-v1/model-card.md) | 数値パラメーター23,411B。C/Python全3,448判定が一致、予約した合成8条件に合格 |
| 最新の実機試験（3522版） | 操作4/5、負例誤反応0/5、内部エラー0。受理4本の返答完走・16アプリ保持を確認 |
| 既知の制約 | Wi-Fi使用後にPetが起動できない場合があり、通常再起動で復帰。作業領域を減らす修正版は未適用。新モデルの実声結果は未取得 |

旧モデルの実声は**6/15**で、基準12/15に届きませんでした。
新しい合成結果は実声試験の代わりにはなりません。
[現在の実機試験](docs/logmel-device-check.md)と[過去の実声記録](docs/voice-validation.md)に範囲を分けて記録しています。

## PCでの試用

[uv](https://docs.astral.sh/uv/)と固定Python 3.12.13を使います。テストにはClangが必要です。
以下の確認には実機・GPU・APIキー・音声エンジンのダウンロードは不要です。

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

[Macシミュレーターでのビルドと試遊](docs/pet-app.md)。
シミュレーターはボタンと模擬認識結果を使い、Macのマイクは録音しません。
本体への導入には、SDK更新とは別に、固定OSのレビューと個別許可が必要です。

## モデルと再現

操作語は**あそぼ / おいで / まって / おやすみ / ありがとう**の5つです。
対象外の発話と無音も分類します。日本語の全文認識や会話モデルではありません。

- [固定log-melモデル・評価・ホスト実行例](models/kws-logmel-v1/model-card.md)
- [学習した移動モデル](models/behavior-v1/model-card.md)
- [合成音声の内容点検](docs/generated-audio-audit.md)・[データの利用条件](docs/data-license.md)
- [以前のCPU学習レシピ](docs/reproduce-model.md)・[実験履歴](docs/validation.md)

コードは[Apache-2.0](LICENSE)。音声の重み・派生した数値には[別の帰属/ライセンス条件](models/kws-logmel-v1/LICENSE.txt)があります。
上流TTSの重み・生成音声・本体録音は同梱しません。[ソース配布手順](docs/releases.md)。
