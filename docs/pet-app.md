# ペットWASMの使い方と検証範囲

2026-11-19。現行appは1.0.1。固定log-mel FWで合成5語/返答5本が成功したが、Desk併用の音声確保エラーが残る。新モデルの実声品質・聴取・実LCD/物理入力は未確認。[現在の試験](logmel-device-check.md)。以前のc032/操作0/5は[履歴](device-validation.md)へ保持する。

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
生成先は新規フォルダのみ。packageは5,176B、権限draw/speaker/microphone。
同じsourceからのhash: `a6b624c9684642448f673bf782e2e777953b506c523e95d8a4f5376f281155dd`。

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
