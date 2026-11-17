# 固定モデルの再現とC照合

Python3.12.13/NumPy2.5.5/PyTorch2.15.0、seed42、CPU2threads。
Open JTalkの6voice設定は2話者。モデルカードのhashは取得時の固定重みであり、
別の音声エンジン/数学実装でのbit一致を保証しない。

## 開発データとvalidationだけの学習

各出力は未存在の名前を使う。Homebrew版の実行ファイル/辞書/voiceのhashはmanifestへ記録される。

```sh
brew install open-jtalk
uv run --locked --group train python -m arcade_ai_pet.data \
  --jtalk-root /opt/homebrew/opt/open-jtalk --output datasets/reproduce-development
uv run --locked --group train python -m arcade_ai_pet.training \
  --dataset datasets/reproduce-development --output checkpoints/reproduce-kws \
  --epochs 600 --hidden 64 --validation-only
```

元の再現試行では、同じPCM原本をhash確認して再処理したaligned-v3を使用した。
現行generatorは同じ既定prosody/seedとaligned-mfcc-v2を使う。
新manifestの追加メタデータにより、JSON hashが当時と同じになるとは限らない。

## 固定後の独立合成

以下は一つの`uv run --locked python`から実行するPython例。評価前にモデルを固定し、
開発source IDとWAV hashが重なれば拒否する。重み/閾値を評価結果へ合わせて変更しない。

```python
from arcade_ai_pet.data import create
from arcade_ai_pet.frozen import freeze_model, evaluate_frozen
from arcade_ai_pet.native import export_assets

freeze_model("checkpoints/reproduce-kws", "datasets/reproduce-development",
             "checkpoints/reproduce-frozen")
create("datasets/reproduce-heldout", "/opt/homebrew/opt/open-jtalk",
       rates=(0.95, 1.2), pitches=(0,), noise_namespace="kws-holdout-v1")
evaluate_frozen("checkpoints/reproduce-frozen", "datasets/reproduce-heldout",
                "reports/reproduce-quality")
export_assets("checkpoints/reproduce-frozen", "build/reproduce-native")
```

`export_assets`はC表現を作るだけで、Flashへ書かない。
配布済みモデルのC監査には次を使う。必要なのはローカルの合成原本で、本体録音は使わない。

```sh
uv run --locked python -m arcade_ai_pet.native_evaluation \
  --model models/kws-v2 --dataset datasets/reproduce-heldout \
  --output reports/reproduce-native
```

全原本/加工/雑音を再構成し、保存した特徴と一致してからCへ渡す。
元wave/hash・特徴・整数logit・最終classを分けて検査する。
native predictorへ任意headerをそのまま渡さず、検査済み数値から再生成したheaderだけをcompileする。

## 未完了

後続の[入力条件の比較](channel-robustness.md)では、固定モデルに対する4前処理はいずれも不採用だった。
既存の合成956例の成功を、低音量/雑音や実声の認識品質へ拡大しない。

ESP32-S3 build/実測、OS内の録音→認識→返答、実声の品質評価は別工程。
Macでの0.228msを本体の応答速度として使用しない。GPU課金・録音の外部送信は不要。
