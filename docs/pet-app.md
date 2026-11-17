# ペットWASMの使い方と検証範囲

2026-11-17 JST。固定FW/Petを1台へ適用し、基本の自動実機試験は成功。[実機結果](device-validation.md)。実声の品質・聴取・実LCD/物理入力は未確認。

## build

公開OSのcommit `3bb9548846505eac303927b9064c6a8d0eb44f79`をcleanなcheckoutで用意し、
SDK側で`uv run --locked python tools/bootstrap.py`を実行する。音声serviceとMac native USB修正は公開mainへマージ済み。
旧FWや旧SDKでは新しい音声importを使えず、SDKを更新するだけでは本体FWは変わらない。

```sh
uv sync --locked
uv run --locked python -m arcade_ai_pet.build \
  --sdk ../scratch-arcade-os --output build/my-pet
PYTHONPATH=../scratch-arcade-os/prototypes/lab-os uv run --locked python \
  -m labos.project run build/my-pet
```

2つ目は既存SDKのMacウィンドウ。HOMEでZを押して起動する。
通常のホストウィンドウはMacのマイク/スピーカーへ接続せず、手動操作で試す。
生成先は新規フォルダのみ。packageは5,142B、WASM4,886B、権限draw/speaker/microphone。
同じsourceからのhash: `14f657f0fe5955d916f6a6811b2975bda0e9bd944a30f968aadbc9eb7c43a23f`。

## 操作

| 本体 | Mac | 操作 |
|---|---|---|
| A | Z | 2秒録音。新しい実機音声serviceが必要 |
| X | R | 学習した相棒との収集遊び |
| 方向 | 矢印 | 遊び中は青い自分を操作。それ以外は上おいで/左まって/下おやすみ/右ありがとう |
| B / FN+B | X / F+X | 終了/HOME |

緑は相棒、青は自分、黄は回収物、灰は壁。左が相棒の得点、右が自分の得点。
BFSを毎frame実行せず、学習した472BのMLPが相棒の移動を選ぶ。
声はあそぼ/おいで/まって/おやすみ/ありがとうの5操作。対象外/無音は状態を変えない。
ありがとうで好感度が増えるが、今回の版は電源をまたぐ育成保存をしない。

## 確認したこと

- Native Cの状態/移動と独立Pythonの照合。未学習16seed×120stepも一致。
- 既存50,000fuel/2pagesで起動。126framesの描画と状態がWasmtime/WAMRで一致し正常終了。
- 各7classの模擬認識結果を実APIへ流し、5操作と棄却/返答要求/終了が両VMで一致。
- ネイティブVM poolは160KiB。実音声I/Oではなくstubサービスなので、音質やS3速度は保証しない。
- 画像は実WASMから生成。実LCDやMacウィンドウの目視、実キー、実声、実機音声の聴取は未確認。

学習/合成の評価を人間の声の精度へ拡大しない。アプリ追加前にも専用ID/前hash/他app保持を確認する。
固定レビュー/個別許可を受けたimageを1回適用した。公開mainの任意buildを書込む許可ではない。[OS音声service](https://github.com/ayutaz/scratch-arcade-os/pull/6)。
