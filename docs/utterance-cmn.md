# 発話ごとのMFCC平均を除くPoC

後続の[新4条件による最終評価](utterance-cmn-final.md)は品質未達で不採用。以下の生成/評価未完了は各工程時点の記録。実機の採用モデルを変更していない。

2026-11-19。**固定1構成で開発用24条件が全合格。独立評価/実機の改善は未確認。** [前候補の最終評価の不合格](reserved-final-evaluation.md)を保持し、開発データだけで別の前処理を比較した。[実行前の条件](../experiments/neural-domain-20261119/utterance-cmn-plan.json)。

## 仮説と契約

前後100msを保持した20bin×12係数の各列から、その発話の平均を除く。energy20値と発話長1値は維持し、出力を−16〜16に制限する。分散による拡大や語別の例外は加えない。

CMNは[Juliusの公式資料](https://github.com/julius-speech/julius/blob/master/doc/Normalize.md)に記載される係数平均の正規化が動機。ただし今回の処理は**集約/clip後の20bin**への適用で、原フレームのCMNと同一とは主張しない。[Kaldiの資料](https://kaldi-asr.org/doc/transform.html)も単純な正規化を一律には推奨していない。短い発話の識別情報を失う可能性もあるため、成功は実測で判断する。

| 項目 | 契約 |
|---|---|
| 入力/出力 | float32の261値、またはそのbatch。有限値/範囲を検査 |
| 平均の範囲 | 発話ごと・係数ごと。batchや別splitを混ぜない |
| 保持 | energy、duration、元の値/PCM/ファイル、ラベル、profile、split |
| データ | developmentのtrain/validationのみ。finalはbuilder前に拒否 |
| モード | `context-duration-cmn`を明示。既存モードの意味を変えない |

平均値が残す話声/入力条件への依存を減らせるかという仮説で、前回の失敗原因が確定したわけではない。最終結果を見て現在の閾値を引き下げる対応は行わない。

## TDD

先行7契約のmodule未実装import Red2件と、新しいdispatch契約の未対応frontend Red1件を確認してから実装。CMN7件/1.826秒、dispatch3件/0.016秒が成功（dispatchの2件は既存モードの回帰）。

新規計8契約は、係数ごとのoffsetへの不変性、発話ごとの独立性、energy/duration保持、入力非変更、silence/clip/不正型/形/NaN、実PCMからの構築、元データのhash保持、final/test混入拒否、261入力のfit選択とfeature source記録を検査する。

```sh
uv run --locked python -m unittest discover -s tests -p 'test_cmn_*.py'
uv run --locked python -m unittest discover -s tests -p test_duration_dispatch.py
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group train python -m unittest discover -s tests/training -v
```

回帰225件/26.866秒・training15件/3.990秒・SDK2件/3.063秒、skipなし。旧v3の凍結13ファイルは不変。後続の30ファイルfreeze対象に含むtrainerには明示モードを追加したため、前候補の再現は固定した727f9d6のsourceを使用する。

## 固定した比較

同じ1,236音声源/開発データ、seed42、hidden88、epochs1000、neural比重2、trainのnoise5/hum3、fresh noiseを維持する。1モデル構成だけを比較し、校正はdevelopment validationのglobal/zerosを初期値とする最大4passの既存手順を使う。

全16domain/profileと結合8profileでF1≥0.90、誤反応≤0.05、操作recall≥0.80が必要。平均や旧domainの件数で未達を隠さない。重み＋閾値は24KiB以下。失敗は結果として保存し、同じ比重/閾値だけの追加探索へ広げない。

```python
from arcade_ai_pet.domain_training import train_candidate

result = train_candidate(
    {'legacy': legacy_data, 'neural': neural_data}, mixed_data, output,
    epochs=1000, hidden=88, neural_weight=2, frontend='context-duration-cmn')
```

出力は新しい前処理を示すホストcheckpoint。261値でも従来の`context-duration-class-v1`とは意味が異なるため、旧freeze/export/OS runtimeへ流用しない。C版/新形式の固定処理/独立最終評価/実機適用は、このPoCの成功とは別工程。

前回の最終データは消費済みで、このfit/選定に使わない。次の採用候補には新しい事前固定の独立条件が必要。実機・録音の外部転送・GPIO/PSRAM/security・課金は変更しない。

## 固定1構成の結果

[初回の数値](../experiments/neural-domain-20261119/utterance-cmn-initial-result.json)。source906a070、同じ元データ/固定1構成を104.333秒で学習・校正した。選択epoch991、重み23,972B、5閾値は2538/3401/326/2312/871、合計23,992B。

開発用16domain条件と結合8条件がすべて合格。最小F1=0.916793・最大誤反応0.02・最小操作recall0.854167。独立したscalar判定と全2,680行が一致し、全24groupの数値も再計算して照合した。元の6つのdevelopment NPZはhash不変。

実行driverは実builderをラップし、返されたvalidationを保持して校正へ使った。入力/乱数/profile/分割を置き換えず、学習と校正は同じ配列を使用した。この実データ比較は追加検証で、新規TDDとは数えない。

前の最終4条件は使用していない。この合格だけでは汎化/実機の改善を証明しない。次は新しい意味を区別するC実装/数値形式・固定処理をTDDで整え、候補固定後に別の独立条件を生成する。

## C処理の照合

CMN専用の[C処理と数値形式](utterance-cmn-native.md)を追加し、開発用2,680条件のPython/C判定がすべて一致した。新しい独立最終評価・target build・実機は未確認。

[正式freeze](utterance-cmn-freeze.md)も同じ開発データの再計算/C照合後に成功。新しい4条件の独立評価は進行中で、旧最終不合格と実声6/15は維持する。
