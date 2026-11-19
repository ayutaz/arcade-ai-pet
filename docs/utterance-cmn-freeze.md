# CMN候補の固定と新しい独立評価

2026-11-19。CMN専用の固定/評価経路をTDDで追加した。実データの正式freezeと新しい独立評価は次の工程で、ここでは完了と扱わない。[C照合](utterance-cmn-native.md)・[新しい4条件の計画](../experiments/neural-domain-20261119/utterance-cmn-reserved-plan.json)。

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

生成テストのproviderは決定的なfixtureであり、実音声の独立評価ではない。次は固定したsourceで正式freezeを実行し、新しい4条件だけを生成/内容点検する。実機の認識結果は6/15のまま。target build・新FW適用・実声の再試験を完了したものではなく、実機書込みには固定imageの別許可が必要。
