# 雑音下の発話区間選択

2026-11-18。元の区間選択は最大frame energyの6%を閾値にする。
背景音がこの値を超えると、発話以外の部分も20binへ含む可能性がある。この仮説をホストだけで比較した。
**改善は見られたが、4候補とも全条件の採用ゲートは未達。** 実機原因や未知話者への一般精度は未確認。

## 比較した変更

| frontend | 変更 |
|---|---|
| aligned | 前の[入力条件付き学習](channel-training.md)。基準 |
| adaptive | 下位20%のframe energyも使って背景より高い区間を選択。最低20framesを保持 |
| subtract | adaptiveに加え、静かなframesの平均FFT powerを差し引く。各成分の5%をfloorとして残す |

入力はHP80またはHP160→平均除去→ピーク4,096への正規化。
話者/原発話split・8条件・品質ゲートは前回と同じ。train 15,976/validation 2,160例、CPU2threads/seed42、重み17,380B。
閾値・epoch・候補の比較はvalidationだけで行い、最終評価用データは開いていない。

| 候補 | epochs上限 | drift F1 | hum F1 | noise F1 | 採用 |
|---|---:|---:|---:|---:|---|
| HP80 / adaptive | 600 | 0.908 | 0.844 | 0.873 | 不採用 |
| HP80 / subtract | 600 | 0.923 | 0.810 | 0.842 | 不採用 |
| HP80 / adaptive | 1,000 | 0.908 | 0.856 | 0.860 | 不採用 |
| HP160 / adaptive | 1,000 | 0.997 | 0.891 | 0.894 | 不採用 |

HP160/adaptiveでもhumの操作recallは0.792で、0.80未満。数値を丸めて採用扱いにしない。
雑音を差し引く処理は、この比較では区間選択だけの変更を上回らなかった。先に複雑な処理を実機へ入れない。
各JSONは[experiments](experiments/)に保存。既存の失敗・モデル・固定FWは維持する。

## TDDと再現

```sh
uv run --locked python -m unittest discover -s tests -p 'test_channel*.py' -v
uv run --locked --group train python -m unittest discover -s tests/training -p test_channel_training.py -v
uv run --locked --group train python -m arcade_ai_pet.channel_training \
  --dataset datasets/jtalk-aligned-v3 --candidate hp160 --frontend adaptive \
  --epochs 1000 --output checkpoints/fresh-adaptive
```

- 新frontendは先行3試験を作り、module未実装のRed後にGreen。背景の除外、最低frame数、全無音、既知のpower差引き、数値範囲を検査。
- 学習側への切替は先行テストで未対応keywordによるRedを確認後に実装。未知frontendはI/O前に拒否する。
- 標準74件、学習6件が成功。既存SDK/appは変更していない。
- NumPyの実験実装。C移植・S3 build・実機認識は未実施。合成PCMも本体録音ではない。

次は24KiBの重み予算内でモデル容量の不足も比較する。validation結果だけで実声の保証へ進まない。
