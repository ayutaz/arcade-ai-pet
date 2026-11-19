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

## 固定1構成の結果

固定source `0e2db8f7e2bf6f096a8095120a9d585a9cdfec10`。30epoch/7,350step、全処理372.727秒、fit318.016秒。整数品質でepoch16を選定した。元データ・保留・評価行は変更していない。

| 指標 | maskなしの前候補 | 学習時mask |
|---|---:|---:|
| 開発用domain/profile合格 | 21/24 | 24/24 |
| 最低macro F1 | 0.884537 | 0.909556 |
| 最大負例誤反応率 | 3.8835% | 3.3981% |
| 最低操作recall | 83.3333% | 85.4167% |
| 選定用操作の生順位正解 | 794/864 | 827/864 |
| 数値容量 | 23,411B | 23,411B |

全24domain/profileと結合8条件、および選定16条件/結合8条件の56指標を混同行列から別計算し、採否が一致した。既存13固定source/今回34source/元入力8ファイル/固定条件/保存重みの照合も成功。選定float stateを再ロードし、4,288行のlogit hashと112クラス行を再現した。選定用の操作864例はfloat/整数とも827正解で、同じ37誤順位が残る。独立発話数・実声の成績ではない。

重みNPZ SHA-256: `194c5fceaeb9e51f83d8637033680a779c50c614becc3ff7bc37339ec31d27a8`。元float stateと重みファイルはprivateに保管し、公開するのはコードと匿名集計のみ。

## 採否と次の条件

**開発用候補として合格。実機採用・正式Releaseは未合格。** 今回のvalidationは従来候補でも参照しており、未使用の独立最終評価ではない。実声6/15を改善したとは扱わない。

次は同じ固定重み/閾値のまま、log-mel前処理と整数6層のC参照・group単位の一致検査を先行TDDで実装する。旧v3 source/defaultを保持し、資源上限/型/不正入力も確認する。その後に正式freeze、新しい未使用の最終条件、S3 build/実測資源を順に通す。未達なら固定した失敗を残し、最終評価を使った再調整はしない。新FWの具体的なレビューと個別許可の前に実機へ書き込まない。

[監査付き集計](../experiments/neural-domain-20261119/spectral-mask-result.json)。
