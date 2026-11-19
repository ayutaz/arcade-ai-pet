# 旧・新の声を別条件として学習する処理

600/1000epochsの混合学習は同じcheckpointを選択し、新domainの基準に未達だった。次の候補は、元発話や最終条件を追加せず、学習時の重みと開発条件別の選定を変更する。

## 契約

`domain_training.train_candidate(domains, mixed, output, ...)` はlegacy/neuralの2つのdevelopment datasetと、その正確な結合manifestを要求する。final role、元ID/WAVの重複、結合の不一致、既存出力を拒否する。

- 学習行はlegacy1倍/neural4倍（上限8倍）。繰返すのはtrainだけで、独立した発話数やvalidation件数は増やさない。
- 学習途中のraw順位上位8件と、8期間それぞれの最良候補から、重複を除いた最大16checkpointを保持する。各checkpointで閾値を校正し、旧/新×8profileの全条件と最悪F1を比較する。raw順位だけで最終モデルを決めない。
- その固定checkpointの閾値を、同じ16groupのdevelopment validationだけで整数境界から選ぶ。全16groupのF1≥0.90、誤反応≤0.05、操作recall≥0.80と、結合8profileの基準を要求する。
- 260→hidden88→7、ReLU6、int8量子化、最大1000epochs/2CPU threads。hiddenは既存上限90を超えない。旧v3の凍結対象13ファイルは変更しない。
- メタデータへtrainer hash、元manifest hash、学習行の重み、選択epoch/閾値、条件別の結果を保存する。基準未達の候補を凍結・実機適用しない。

実際の追加学習で良くなることを保証するコードではない。最終評価・人の発話・実機の判定は別工程。[事前計画と失敗](../experiments/neural-domain-20261119/README.md)。

## TDD（2026-11-19）

通常4契約と実学習1契約を先に追加。通常側は未実装moduleのImportError1件、training側はModuleNotFoundError1件でRedを確認。最小実装後、通常4件/0.011秒、実学習1件/9.904秒が成功。

契約は新domain/profileの失敗を平均で隠さない順位、16groupの閾値選定、欠けたgroup/型の拒否、trainだけの繰返し・validation分離・final拒否。実学習fixtureは同じCPU環境で2回学習し、量子化後スコアの一致と16groupの判定を確認した。fixtureの成功を実声の品質へ拡大しない。

コマンド: `uv run --locked python -m unittest discover -s tests -p test_domain_training.py`、`uv run --locked --group train python -m unittest discover -s tests/training -p test_domain_training.py`。

回帰は通常135件/15.737秒、training9件/1.457秒、SDK2件/3.374秒が成功（skipなし）。実際の新domain候補の学習はこの記録後に行う。


## 閾値を含むcheckpoint選定の修正

初版の1/4学習は新しい声のclean操作recall1.0まで改善したが、最大誤反応0.08・最悪F1=0.8659で未達。[初版結果](../experiments/neural-domain-20261119/domain-weighted-initial-result.json)を保持する。

raw順位では良くても閾値で修正できないモデルと、raw順位では劣っても閾値で全条件を満たせるモデルの反例を先に追加した。bounded shortlistの契約も追加し、両helper未実装のAttributeError2件を確認。実学習fixtureには候補数/選定方式の契約を足し、既存実装のKeyError1件を確認してから修正した。

Greenは通常6件/0.014秒、実学習1件/1.241秒。最大16候補を校正し、全条件・最悪F1・平均F1・小さい閾値・早いepochの順で選ぶ。候補数を固定し、最終条件は使わない。新しい実モデルの成績は次の実験で確認する。

修正後の回帰: 通常137件/15.536秒、training9件/1.820秒、SDK2件/2.298秒が成功（skipなし）。
