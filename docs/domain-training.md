# 旧・新の声を別条件として学習する処理

600/1000epochsの混合学習は同じcheckpointを選択し、新domainの基準に未達だった。次の候補は、元発話や最終条件を追加せず、学習時の重みと開発条件別の選定を変更する。

## 契約

`domain_training.train_candidate(domains, mixed, output, ...)` はlegacy/neuralの2つのdevelopment datasetと、その正確な結合manifestを要求する。final role、元ID/WAVの重複、結合の不一致、既存出力を拒否する。

- 学習行はlegacy1倍/neural4倍（上限8倍）。繰返すのはtrainだけで、独立した発話数やvalidation件数は増やさない。
- checkpointは旧/新×8profileの16groupでraw判定を比較し、全条件の基準と最悪F1を優先する。全体平均だけで選ばない。
- その固定checkpointの閾値を、同じ16groupのdevelopment validationだけで整数境界から選ぶ。全16groupのF1≥0.90、誤反応≤0.05、操作recall≥0.80と、結合8profileの基準を要求する。
- 260→hidden88→7、ReLU6、int8量子化、最大1000epochs/2CPU threads。hiddenは既存上限90を超えない。旧v3の凍結対象13ファイルは変更しない。
- メタデータへtrainer hash、元manifest hash、学習行の重み、選択epoch/閾値、条件別の結果を保存する。基準未達の候補を凍結・実機適用しない。

実際の追加学習で良くなることを保証するコードではない。最終評価・人の発話・実機の判定は別工程。[事前計画と失敗](../experiments/neural-domain-20261119/README.md)。

## TDD（2026-11-19）

通常4契約と実学習1契約を先に追加。通常側は未実装moduleのImportError1件、training側はModuleNotFoundError1件でRedを確認。最小実装後、通常4件/0.011秒、実学習1件/9.904秒が成功。

契約は新domain/profileの失敗を平均で隠さない順位、16groupの閾値選定、欠けたgroup/型の拒否、trainだけの繰返し・validation分離・final拒否。実学習fixtureは同じCPU環境で2回学習し、量子化後スコアの一致と16groupの判定を確認した。fixtureの成功を実声の品質へ拡大しない。

コマンド: `uv run --locked python -m unittest discover -s tests -p test_domain_training.py`、`uv run --locked --group train python -m unittest discover -s tests/training -p test_domain_training.py`。

回帰は通常135件/15.737秒、training9件/1.457秒、SDK2件/3.374秒が成功（skipなし）。実際の新domain候補の学習はこの記録後に行う。
