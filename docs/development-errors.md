# 開発用分類の誤順位と棄却の切り分け

2026-11-19。`development_errors.audit(labels, logits, thresholds)`は、固定した7分類の整数logitと5操作の点差閾値を読み、誤順位と正解棄却を分けて集計する。学習・閾値変更・モデル保存・実機操作は行わない。

| 集計 | 意味 |
|---|---|
| raw_top1_correct | 棄却前の1位が正しい操作 |
| correct_top1_rejected | 1位は正しいが点差不足で棄却された操作 |
| wrong_top1 | 1位が別操作/対象外/無音だった操作。棄却閾値を下げても正解に変わらない |
| wrong_command_accepted | 別操作を最終的に受け付けた操作 |
| raw / final false_activation | 対象外/無音を操作とした件数（棄却前/後） |

元/最終の7×7混同行列、クラス別内訳、正解棄却の点差の範囲を返す。公開する場合もPCM・個別の録音・実機識別情報は含めない。呼出し側でdevelopment由来/固定source/モデルhashを確認し、最終評価を次の調整へ流用しない。

## TDD

先に3契約を追加し、未実装moduleの`ModuleNotFoundError`をRedとして確認。その後に最小実装した。

```sh
uv run --locked python -m unittest discover -s tests -p test_development_errors.py -v
```

3件/0.021秒が成功。手計算fixtureで正解・正解棄却・誤順位・誤操作受付・負例棄却を分離し、同点の先頭index、閾値と等しい点差、int32両端の差、入力不変、無効/過大配列を検査する。回帰282件/33.525秒、学習21件/3.225秒、SDK2件/3.070秒もskipなし成功した。

これは診断処理の保証で、音声認識の品質改善や実機動作の合格ではない。[固定CNNの不採用結果](logmel-dscnn.md)を維持する。実モデルの再計算は追加検証として別に記録する。
