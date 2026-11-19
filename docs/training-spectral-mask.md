# 学習時だけの時間・周波数mask

2026-11-19。先行14契約と回帰358/学習48/固定SDK2件が成功。Mac MPSの小型2fitでも整数重み・選定float stateが再現し、CPU/MPSのmask結果が一致した。実機の確認ではない。

## 変更範囲

既存logmel-dscnn-v1の64×32特徴だけに、例ごとに連続する時間区間と周波数区間を0埋めする。時間幅0〜6、周波数幅0〜3を一様抽選し、その幅が収まる開始位置を抽選する。幅0は変更なし。末尾のenergy64/duration1、ラベル、元IDは保持する。0は正規化特徴の平均値で、物理的な無音ではない。

既定は無効。明示した学習batchだけで有効にし、検証・校正・推論では適用しない。augmentation用のNumPy Generatorはseed+1で独立させ、Torchの学習順序用RNGを消費しない。今回の対応範囲は既存2,113入力のみ。envelope/time-pool/frozen-prefixとの混用を拒否する。

候補保存前と校正後に設定を照合し、未宣言・不足・変更されたmask設定を拒否する。float診断との併用、途中の期限超過、step上限も検証。モデル形状と数値23,411Bは維持する。

## TDD

| 対象 | 意図したRed | Green |
|---|---|---|
| NumPy plan 4契約 | spectral_mask未実装のImportError | 4件 / 0.036秒 |
| Torch/fit 6契約 | spectral_mask未実装のImportError | 6件 / 1.684秒 |
| 候補保存 4契約 | 未対応keyword 3件、未宣言設定を拒否しない1件 | 4件 / 0.090秒 |

```sh
uv run --locked python -m unittest discover -s tests -p test_spectral_mask.py -v
uv run --locked --group train python -m unittest discover -s tests/training -p test_spectral_mask.py -v
uv run --locked python -m unittest discover -s tests -p test_spectral_mask_candidate.py -v
```

全回帰は358件/36.409秒、学習48件/6.469秒、固定SDK2件/3.111秒。skipなし。既存13固定sourceは原本とbyte一致。MPSの小型試験は追加検証であり、後付けTDDとは記録しない。

## 固定比較

実装・容量確認後、同じtrain/保留/validation、30epoch、seed42、batch256、neural比重2の1構成だけを固定する。最大8,000step/fit900秒。元float診断を併用するが、選定は従来の整数品質に限定する。評価行の削除/再ラベル、消費済み最終2組の利用、幅だけの反復探索は行わない。

実機a553/16app・実声6/15と既存不採用を維持する。新C・正式freeze・新独立最終・S3 build・実機適用は別工程。Vast.ai累計0米ドル。

[固定条件](../experiments/neural-domain-20261119/spectral-mask-plan.json)・[MPS小型試験](../experiments/neural-domain-20261119/spectral-mask-mps-smoke.json)。
