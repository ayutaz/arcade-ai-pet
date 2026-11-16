# 事前生成した日本語の返答5種

「あそぼ/おいで/まって/おやすみ/ありがとう」に対する短い返答。
NIT ATR503 M001で合成し、DC除去、31tap低域通過、8kHz化、PCM8量子化、5ms fadeを適用。
25,418BをFlashへ置き、再生時は各sampleを2回出して16kHzへ合わせる。
PCM16への倍率28、assetの絶対値100以下で最大2,800。本体再生品質は未確認。
実時間TTSや自由文読み上げではない。コード内の声は人/本体録音ではない。

| 応答ID | 返答 | 8kHzのsample数 |
|---|---|---:|
| 0 | いっしょにあそぼ | 7,839 |
| 1 | はーい | 3,116 |
| 2 | わかった | 4,859 |
| 3 | おやすみ | 4,619 |
| 4 | ありがとう | 4,985 |

HTS Working Group、©2003–2012 Nagoya Institute of Technology /
©2003–2008 Tokyo Institute of Technology。生成と加工を行った。機関の推薦はない。
[CC BY 3.0条件](LICENSE.txt)。engine/dictionary/voice/原PCM/hashは[metadata.json](metadata.json)。
上流の合成モデルや原WAVは含めず、数値C表だけを配布する。

```sh
uv run --locked python -m arcade_ai_pet.replies \
  --jtalk-root /opt/homebrew/opt/open-jtalk --output assets/new-replies
```
