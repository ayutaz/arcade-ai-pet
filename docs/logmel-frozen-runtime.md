# log-mel CNNの固定と最終評価への接続

2026-11-19。既存2,113入力/6層/5閾値だけの数値export、full nativeの再検査、同じ予約4条件を使う最終生成/評価を接続する。モデル/閾値の探索は追加しない。

- 18個のNPZ memberを、dtype/shape/payload/上限/重複/shiftまで確認してからload。headerは数値から再生成し、申告されたhashだけを信用しない。
- freezeは元のlegacy validation・保留neural train条件・元neural validationの24groupをPCMから再計算。全行のC/Q5/logit/判定・bankedと品質が揃った場合だけ出力する。
- 数値/全source/教師callback/元入力/plan/生成物のbindingを固定する。既存MLP/CMNを既定のまま残し、新runtimeは明示選択だけにする。
- 未使用の4captionはモデル固定後にだけ生成。内容点検は既存の独立ASR、最終評価での校正/学習は禁止。

## 検証記録

数値asset7契約、freeze/dispatch6契約を先に追加し、未実装moduleのImportErrorでRed→各7件/1.024秒、6件/1.166秒でGreen。教師callbackのsource固定不足も先に失敗を確認して追加した。

最終評価のCNN切替1件は初回の変更後に契約を追加したため、TDD実施には数えない。未コミットの切替を元のMLP経路へ戻して、2113入力のmatmul不一致を再現し、切替を適用して7件/1.729秒Greenを確認した。以後の変更は先行契約で進める。

全host391件/44.573秒、学習48件/6.691秒、固定SDK2件/3.261秒がskipなし成功。元v3固定13sourceはbyte不変。教師binding追加後の7契約も再確認する。生ログは非公開Labのreports/へ。

## 固定した実験

候補194c5fce…・5閾値・holdout seed2026111904を維持する。新しい4条件のplanはf81bdc42…で、消費済み2plan/captionを再利用しない。CPU生成は最大2,400秒/168試行、追加モデル学習は行わない。

これは事前条件の固定で、正式freeze/独立最終/target/実機の完了報告ではない。実機a553/16app・実声6/15と既存の失敗を維持する。

[事前条件](../experiments/neural-domain-20261119/logmel-freeze-plan.json)・[新しい最終plan](../experiments/neural-domain-20261119/logmel-reserved-plan.json)。
