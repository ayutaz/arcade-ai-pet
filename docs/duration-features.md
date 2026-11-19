# 発話時間を追加するホストPoC

**既存260特徴を維持し、発話時間を1値追加した261入力のホストPoC。固定1構成の実学習は基準未達で不採用。C移植・実機適用は未実施。** 旧v3の凍結対象13ファイル、旧量子化器/reader/exportは変更しない。

## 根拠

[16条件の追加結果](../experiments/neural-domain-20261119/extension-results.json)。430試行からASR一致328元発話を採用し、trainだけへ984例追加。旧validation/testのbyte/hashは保持した。計1,236元発話の固定2構成は最悪F1=0.865432/0.860771で不採用。データ量・同じ比重/閾値だけの探索は続けない。

[誤判定と時間分布](../experiments/neural-domain-20261119/extension-failure-diagnostics.json)では、neural validation各520行の誤りは両方33行。そのうち24〜25行が未知語「かわいい」。最多の誤反応元のclean活動区間は約0.995〜1.005秒、neural trainの「おいで」の中央値は0.345秒/90%点0.565秒だった。元発話の3variantや8profileを独立した話者数へ数えない。

現行frontendは活動区間を20分割して平均し、実時間を明示した値を持たない。時間が判別に役立つという仮説で、成功の証明ではない。「おいで」のtrainにも最大1.055秒、「ありがとう」に1.895秒の例があり、長さの閾値で特定語を拒否しない。

## 契約

- `duration_frontend.features` は前260値を旧adaptiveと同一にし、末尾へ `4 * (last-first) / 198` を追加。無音/peak32未満は261値とも0。
- `duration_arrays.make_arrays` は旧split/元発話/profile/ノイズseedを保持。final入力は読込み前に拒否し、評価行の比重を増やさない。
- `domain_training.train_candidate(..., frontend='duration')` だけが261入力を明示する。既存のadaptive/contextは260入力のまま。
- `duration_quantize` は261入力・最大90hiddenを別に検査。旧量子化器の260上限を解除しない。hidden88なら23,972B、最大90なら24,516B。
- 261入力は旧reader/exportが拒否する。旧C版や実機へそのまま渡さない。

[固定した比較条件](../experiments/neural-domain-20261119/duration-plan.json)は同じ1,236元発話、hidden88/1000epochs/seed42/neural比重2の1構成。旧/新16条件と結合8条件の全gateを維持し、最終4captionは未使用。

```python
from arcade_ai_pet.domain_training import train_candidate

train_candidate(
    {'legacy': legacy_development, 'neural': extended_neural},
    mixed_development, fresh_output,
    frontend='duration', neural_weight=2, epochs=1000, hidden=88,
)
```

## TDD（2026-11-19）

| 先行テスト | 意図したRed | Green |
|---|---|---|
| 特徴の保持/時間/無音/入力 | module未実装のimport error1件 | 4件/0.080秒 |
| split/profileの保持・final拒否 | module未実装のimport error1件 | 2件/2.119秒 |
| 261入力のdispatch/旧export拒否 | frontend未対応のValueError1件 | 1件/0.007秒 |
| 261入力の実学習/幅の拒否 | feature_count未対応のTypeError2件 | 2件/1.950秒 |
| 別量子化器の同一性/演算/容量 | module未実装のimport error1件 | 通常側10件/1.831秒に含む |

実学習の最初のGreen試行は旧量子化器の260入力制限で1件停止した。上限は維持し、261専用の3契約を先に追加して別moduleへ実装した。各失敗ログを保持し、この中間失敗を成功へ書き換えない。

コマンドは `uv run --locked python -m unittest discover -s tests -p 'test_duration_*.py'`、実学習は `uv run --locked --group train python -m unittest discover -s tests/training -p test_duration_fit.py`。

回帰は通常185件/19.043秒、training15件/2.970秒、実SDK2件/3.403秒、skipなし。fixtureの学習成功は実際の音声認識品質や実機動作の保証ではない。

## データと公開範囲

[生成plan](../experiments/neural-domain-20261119/extension-source-plan.json)と[学習前の固定条件](../experiments/neural-domain-20261119/extension-predeclared-training.json)を保存。後者の未学習flagは実行前snapshot。旧/新データと教師モデルのlicenseはそれぞれ適用し、Irodoriのmodel cardの倫理的制限も維持する。元WAV・教師重み・失敗音源・本体PCMは配布しない。新しいAPI課金/GPU契約/FW書込みは行っていない。


## 固定1構成の実行結果

[初回の数値](../experiments/neural-domain-20261119/duration-initial-result.json)。事前条件どおり、source5ee49feで既存の1,236元発話を学習した。104.844秒、23,972B、最悪F1=0.865901・最大誤反応0.12・最小操作recall0.833333で**不採用**。最終4caption・元WAV・評価行・実機は変更していない。

最小recallは従来の比重2の0.729167より上がったが、複数条件のgateに未達で、発話時間1値の追加だけでは全体の問題を解決できなかった。追加の比重/閾値探索で合格を作らず、残る誤判定の音声情報を調べる。C版/モデル凍結/独立最終評価/target buildへ進めない。


次の制御した比較は[語頭の100ms保持と発話時間の組合せ](context-duration-features.md)。単独候補の未達を保持し、元データ/評価条件を変えずに比較する。
