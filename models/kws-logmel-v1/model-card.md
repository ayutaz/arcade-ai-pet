# Fixed Japanese command model — log-mel v1

2026-11-19。5操作語（あそぼ/おいで/まって/おやすみ/ありがとう）と対象外/無音の7classを扱う、小型の整数CNN。一般的な日本語ASRや、連続した会話の認識器ではない。

| 項目 | 固定値 |
|---|---|
| runtime | context-logmel-dscnn-v1 |
| 入力 | 16kHz mono PCM16、32,000frames（2秒） |
| 特徴 | HP160/peak4096、log-mel64×32＋energy64＋duration1（計2,113） |
| 分類器 | 6層、int8/int32、Q5 activation/Q10 logits |
| 数値容量 | 重み23,391B＋5閾値20B＝23,411B（NPZ容器/数学table/コードを除く） |
| 棄却閾値 | [3505,1220,3063,552,1052]。対象外/無音は操作しない |
| C scratch | 特徴26,672B。推論12,288Bは特徴の完了後に同じ領域を再利用 |
| weights SHA-256 | 194c5fceaeb9e51f83d8637033680a779c50c614becc3ff7bc37339ec31d27a8 |

Open JTalkの合成音声と、Irodori VoiceDesignで生成/独立ASR点検した一般的な成人声の合成データを混合した。実機録音・実在人物の参照録音・個人の声は学習に使用していない。時間/周波数maskはtrainだけに適用し、モデル/checkpoint/5閾値を固定後に未使用4条件を初めて生成した。最終評価からの追加学習や閾値変更はない。

[evaluation.json](evaluation.json)の77発話＋200対照/8profile/3,448行で最小F1=0.924751、最大負例誤反応率=0.013477、最小操作recall=0.883333。全C特徴/Q5/logit/判定がPythonと一致した。合成ホスト評価であり、未知の人/部屋/距離の精度を保証しない。

[decision-frozen.json](decision-frozen.json)に固定52source・元入力hash・最終planとのbindingを保存する。モデルの数値とheaderを再生成照合してから使う。旧v3/CMNの失敗や新FW/Pet1.0.0の返答停止は別の履歴であり、このモデルの合格へ書き換えない。実機の最新受入結果は別文書へ記録する。

## 使用例

```python
from pathlib import Path
import numpy as np
from arcade_ai_pet.dscnn_assets import DscnnNativeAssets
with DscnnNativeAssets(Path("models/kws-logmel-v1")) as model:
    label, logits = model.predict(model.extract(np.zeros(32000, np.int16)))
    assert label == 6  # silence; this is a host smoke check
```

ESP-IDF側は公開OSの明示選択[log-mel backend](https://github.com/ayutaz/scratch-arcade-os/blob/main/docs/voice-logmel-runtime.md)へこの数値exportを渡す。実機書込みを行うコマンドは含めない。GPIO/別基板の互換性やPSRAMを仮定しない。

## 条件

重み・数値C表は保守的に[CC BY 3.0と帰属](LICENSE.txt)を添えて公開する。数学table/手書きcodeはApache-2.0。Irodori/codecはMIT表記と追加の倫理条件を固定model cardで点検済みで、人物名/参照録音/意図的な声まね/誤認目的を使用していない。上流の原本重み/音声PCMは再配布しない。[各データの条件](../../docs/data-license.md)。
