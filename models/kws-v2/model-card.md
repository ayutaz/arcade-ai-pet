# 日本語5操作 KWS v2

2026-11-17。合成音声で成立した候補。**実声・実機での品質は未検証**。
日本語の自由文ASRや話者認識ではない。

| 項目 | 値 |
|---|---|
| 操作 | あそぼ / おいで / まって / おやすみ / ありがとう |
| 追加class | 対象外 / 無音・noise |
| 入力 | 16kHz mono PCM16、2秒 |
| 特徴 | 発話区間を合わせた20bin×12MFCC＋20energy、計260 |
| モデル | MLP 260→64→7、ReLU6 |
| 整数化 | int8 weight / int32 bias / Q10 activation、2の累乗scale |
| 重み | 17,380B（NPZファイル18,874B） |
| 選択 | seed42、600epochsのvalidationからepoch51、拒否閾値0 |
| C scratch | 5,680B。出力1,040B/関数stack等は別 |
| 実行 | ホストCとPythonの判定956/956一致、整数logit956/956完全一致 |

## 新規合成の評価

モデルを固定してから、別速度/基準pitchと新しいnoise seedで生成した。
252原発話×3加工＋200noise＝956例、原source計452。
開発時の全source ID・音声WAV hashとの重複は0。元のvoice人物は共有する。

| 指標 | 結果 |
|---|---|
| macro F1（7class） | 0.978908 |
| 全体正解率 | 0.988494 |
| 対象外/無音の誤起動率 | 0.002577 |
| 音声モデルの採用基準 | F1>=0.90、誤起動率<=0.05を通過 |
| C/Python特徴の最大差 | 0.000003219 |

行列等は[evaluation.json](evaluation.json)、Cの比較は[native-evaluation.json](native-evaluation.json)。
ホストCの中央値0.224ms/p95 0.847msはこのMacでの値で、ESP32-S3の性能ではない。
このモデルをOSへ適用・本体へ書き込み済みとは扱わない。

## データと制限

Open JTalkのHTS Voice Mei 5感情とNIT ATR503 M001を使用。
6voiceファイルだが、人物はMeiとM001の2種。実人間の未知話者テストではない。
開発データは756原発話×3加工＋200noise、train1997/validation270/test201。
初回の失敗testを設計変更に使ったため、最終判断には新規合成データを用意した。

未知の人・距離・部屋・雑音・言い方への強さ、連続待受、実機遅延、
声が小さい場合の誤拒否は未検証。対象外の文章は16種類に限る。
2秒録音と認識を別工程にし、まずpush-to-talkで実声評価する。

## 配布条件と同一性

コードはApache-2.0。学習した重み/派生特徴は保守的にCC BY 3.0として配布する。
HTS Voice Mei: MMDAgent Project Team、©2009–2018 Nagoya Institute of Technology,
Department of Computer Science。NIT ATR503 M001: HTS Working Group、
©2003–2012 Nagoya Institute of Technology / ©2003–2008 Tokyo Institute of Technology。
合成、augmentation、学習、量子化、C変換を行った。機関の推薦は受けていない。
[詳細条件](../../docs/data-license.md)・[ライセンス](LICENSE.txt)。

`weights.npz` SHA-256:
`91251ae7be454a62f9cd35607f3e2dbf6a4be8ce0b80342bd81f947e4a8e5527`。
`pet_model.h`は同じ重みのC表現、`feature_tables.h`は生成した数学定数。
`conformance.npz`は新規合成からの38特徴列とlogitで、音声PCMを含まない。
上流TTS重み・本体録音・実機ID・資格情報は含まない。
