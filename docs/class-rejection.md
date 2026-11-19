# 固定モデルのクラス別棄却PoC

後続の[C版照合](decision-native.md)では開発用2,680件の全判定一致を確認した。以下は校正時点の記録で、正式freeze・独立最終評価・実機は引き続き未完了。

**固定したcontext-durationモデルのクラス別棄却により、開発用16条件と結合8条件の全基準に合格。独立最終評価・C対応・実機適用は未確認。** 旧global棄却処理と凍結済み13ファイルを変更せず、別moduleで比較する。

## 根拠と範囲

[同じ固定モデル](context-duration-features.md#固定1構成の結果)は14/16条件に合格し、旧hum/noiseが未達。2,680検証行を元WAVから再計算し、保存済みの16条件の全数値を再現した。humで正解1位だが棄却5件・誤順位5件、noiseで7件・5件が残る。

旧humの正解点差はclass0で最小2336、class2で953、class3で211だった一方、誤ったclass1には1309までの例があった。一律の閾値を下げず、全5クラスへ同じ手順を適用して、低信頼の誤りと正解を分けられるか検証する。特定語/元発話ID/文字列による例外処理はない。

校正には開発用validationだけを使う。モデルを再学習せず、最後の独立4captionや本体の音声を開かない。

## 契約

- `class_rejection.decisions` はQ10 logit7値と整数閾値5値を受け取る。1位がcommand0〜4で、点差がその閾値未満ならunknown5へ棄却。元のunknown5/silence6は維持する。
- 同値の閾値を並べると旧global判定と一致する。argmaxの同点は先頭を選び、int32の両端の点差はint64で計算する。
- `calibrate` は0〜4096の整数境界だけを検討。既存global閾値の一律表とzerosの2初期状態、固定クラス順0〜4、この比較では最大4passで比較する。変化しなければ早く終える。
- 全16domain/profileと結合8profileのgateを同時に判定。未達groupを平均で隠さない。局所的な座標探索で、全組合せの最適性を保証しない。
- 5値は20B、固定重み23,972Bとの合計23,992B。旧exportに渡せる形式とは扱わない。

[事前条件](../experiments/neural-domain-20261119/class-rejection-plan.json)は固定重み/検証行のhash、最大4pass、同じ品質gateを保存する。合格時も独立したデータや実声の成功とは扱わない。

```python
import numpy as np
from arcade_ai_pet.class_rejection import calibrate, decisions

result = calibrate(labels, logits, groups, initial_threshold=2531, max_passes=4)
predictions = decisions(logits, np.asarray(result['thresholds_q10'], dtype=np.int32))
```

## TDD（2026-11-19）

先行7契約を追加し、module未実装importエラー1件でRed。最小実装後7件/0.023秒が成功。符号なし整数のlabels/groupsの契約も追加し、混合型がbincountへ入るTypeError1件のRed後、検証後にint64へ統一した。最新8件/0.028秒が成功。

契約は旧判定との一致・unknown/silence・同点・overflow・不正入力、単一閾値で解けない例の校正、1groupの未達拒否、入力不変と行順不変、探索回数の上限。単一閾値では両方満たせない例を、独立した旧校正器とも照合する。

コマンド: `uv run --locked python -m unittest discover -s tests -p test_class_rejection.py`。回帰198件/21.033秒、training15件/3.265秒、SDK2件/2.426秒、skipなし。実データの校正効果はこの記録後に判断する。


## 固定モデルでの校正結果

[初回の数値](../experiments/neural-domain-20261119/class-rejection-initial-result.json)。sourceef50810で、固定重み/固定2,680行を校正した。5閾値は `[977, 2531, 1780, 181, 768]`、366候補・各2/3pass・0.035秒。重みは再学習していない。

全16domain/profileと結合8profileが合格。最悪F1=0.903890、最大誤反応0.04、最小操作recall0.833333。独立したscalar判定と2,680行の全予測が一致し、全24groupの数値を別に再計算して照合した。元の重み/検証NPZのhashも不変。重み＋棄却表は23,992B。

これは校正に使ったdevelopment validationでの選定結果で、汎化性能や実声/実機の合格ではない。候補の固定・Cでの特徴/判定の照合・未使用の最終4条件・target build/資源確認が残る。最終データを見て再校正せず、実機へ書く場合は固定imageの個別許可を必要とする。
