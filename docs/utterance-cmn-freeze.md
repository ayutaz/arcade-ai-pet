# CMN候補の固定と新しい独立評価

後続の[新4条件による最終評価](utterance-cmn-final.md)は品質未達で不採用。以下の生成/評価未完了は各工程時点の記録。実機の採用モデルを変更していない。

2026-11-19。CMN専用の固定/評価経路をTDDで追加した。後続の実データによる正式freezeは成功した。新しい独立評価は未完了。[C照合](utterance-cmn-native.md)・[新しい4条件の計画](../experiments/neural-domain-20261119/utterance-cmn-reserved-plan.json)。

## 固定する内容

`decision_freeze.freeze_candidate(..., runtime='context-duration-cmn-class-v1')`を明示する。省略時は従来のcontext-durationを維持する。CMNは専用の入力契約/C処理を使い、重み、5閾値、前処理・特徴・推論・評価source、元発話ID/hash、開発manifest、新最終計画を固定する。

source38ファイルにはCMNのPython/Cとtrain配列生成・量子化も含む。固定前に元音声から開発用全16条件＋結合8条件の品質とC一致を再計算し、失敗時は固定出力を作らない。検証時には数値metadataのruntime/契約も照合する。過去の固定物は当時のsource archiveで確認し、新コードで黙って読み替えない。

生成/特徴化/評価は固定物のruntimeを使い、重み/計画/freezeのhashへ結び付ける。新4条件は開発用選定後・生成前に定義した。旧最終4条件は消費済みで、fit・候補選定・再校正へ使わない。今回も最終結果を見て閾値を変えない。

## TDDと検証

先行6契約で、CMN処理へのdispatch、sourceと形式の固定、異常時の出力禁止、生成前の改変拒否、最終評価の固定frontendと校正禁止、最終C照合の経路を検査した。初回Redはfixtureのpatch解除漏れも含んだため、fixtureを直して未対応属性/引数のRedを再確認。最小実装後6件/0.864秒が成功した。

配列生成・量子化sourceの固定も先にassertを追加し、`duration_arrays.py`未登録のRedを確認して追加。6件/0.836秒が成功。回帰237件/28.911秒が成功した。既存v3凍結13ファイルと旧形式の既定動作を維持する。

```sh
uv run --locked --group train python -m unittest discover -s tests -p test_cmn_freeze.py
uv run --locked --group train python -m unittest discover -s tests -v
```

生成テストのproviderは決定的なfixtureであり、実音声の独立評価ではない。固定したsourceで正式freezeを実行済み。新しい4条件の生成/内容点検を進める。実機の認識結果は6/15のまま。target build・新FW適用・実声の再試験を完了したものではなく、実機書込みには固定imageの別許可が必要。

## 実候補の固定結果

[実行前の条件](../experiments/neural-domain-20261119/utterance-cmn-freeze-preflight.json)と[結果](../experiments/neural-domain-20261119/utterance-cmn-freeze-result.json)を分けて保存した。前者の`frozen:false`は実行前の記録で、後から書き換えない。

| 固定項目 | 結果 |
|---|---|
| source / 公開main | 667d650 / 271dae1（tree一致） |
| source / 元音声 | 38ファイル / 1,236音声 |
| validationの整数logit・分割bank・最終判定 | 各2,680/2,680一致 |
| 開発用条件 | 全16domain＋結合8条件が合格 |
| 重み＋5閾値 / scratch | 23,992B / 5,680B |
| Mac arm64でのfreeze時間 | 12.132秒 |

重み`0292ba74…`、5閾値2538/3401/326/2312/871、元発話ID/hash、開発manifest、新最終計画を固定した。freeze JSON SHAは`fa84e9fbda884341af804b3e8b0c9b343fd5621bce8c741e14475d7edb8324c8`。新計画のcanonical SHAは`d357efa61c2085d221582407e99f892654ff56c2642d366a4c8d0ced442323c0`。

[PR46](https://github.com/ayutaz/arcade-ai-pet/pull/46)のhead667d650は必須4CI（workflow44475409688）成功後に通常mergeした。実際の固定も同じsource archiveを使用し、検証対象のコードを変更しない。

その後、新しい4条件だけのCPU生成/ASR内容点検を開始した。上限2,400秒/168試行、課金0米ドル。音声・重み・端末データはこの記録に含めない。これは独立評価の合格やESP32-S3の実測ではない。
