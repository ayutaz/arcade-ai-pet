# 語頭の余白と発話時間のホストPoC

**既存context特徴260値と、余白を含まない活動時間1値を組み合わせる261入力の実験。実データの改善/C移植/実機動作は未確認。** 新たな音声生成・評価gate変更・本体更新を行わず、既存1,236元発話の1構成だけで比較する。

## 根拠と比較範囲

[以前のASR点検](../experiments/neural-domain-20261119/window-asr-analysis.json)で、最多の誤反応元は元音声/全体が「かわいい」、活動区間だけが「さわいい」、前後100msを残すと「かわいい」へ一致した。ASR結果を人の聴取や実機の認識結果とは扱わない。

[旧context候補](../experiments/neural-domain-20261119/context-initial-result.json)も[時間1値の候補](duration-features.md#固定1構成の実行結果)も不採用だった。今回の仮説は、追加328元発話を含む同じデータで、語頭の余白と時間が補完するかどうか。両方の単独失敗を消さず、組合せが成功すると仮定しない。

## 契約

| 項目 | 固定する内容 |
|---|---|
| 最初の260値 | 既存 `context_frontend.features` と同一。録音範囲内で前後100msを保持 |
| 末尾の1値 | 従来 `duration` と同じ、余白前の活動frame数/198×4 |
| 学習 | `frontend='context-duration'` を明示。旧modeの既定値は維持 |
| 評価 | 元発話/split/profile/ノイズseedを保持。finalは読込み前に拒否 |
| 容量/形式 | 261入力/hidden88で23,972B見込み。旧reader/exportは拒否 |

[事前条件](../experiments/neural-domain-20261119/context-duration-plan.json)はhidden88/1000epochs/seed42/neural比重2の1構成。旧/新16条件・結合8条件のF1≥0.90、誤反応≤0.05、操作recall≥0.80、数値重み24KiBを維持する。特定語を長さで拒否する規則はない。

```python
from arcade_ai_pet.domain_training import train_candidate

train_candidate(
    {'legacy': legacy_development, 'neural': extended_neural},
    mixed_development, fresh_output,
    frontend='context-duration', neural_weight=2, epochs=1000, hidden=88,
)
```

録音範囲外の音を補う処理ではない。

## TDD（2026-11-19）

変更前に5契約を追加した。context keyword未対応のTypeError2件と、arrays/dispatchのmode未対応ValueError各1件で意図したRed。残り1件はfinal拒否の継続検査で、既存のmode拒否でも成功していたため、新しいRedとは数えない。

最小実装後、既存を含むduration15件/3.656秒が成功。回帰190件/20.817秒、training15件/3.386秒、SDK2件/2.360秒、skipなし。既存context特徴と非padding時間の一致、無音、bool以外の拒否、split/profile保持、旧reader/export拒否を確認した。

コマンドは `uv run --locked python -m unittest discover -s tests -p 'test_duration_*.py'`。実学習の回帰は `uv run --locked --group train python -m unittest discover -s tests/training -v`。fixtureの成功を認識品質へ拡大しない。旧v3の凍結対象13ファイルは変更しない。
