# 24KiB以内のモデル容量比較

2026-11-18。HP160/adaptiveの誤認識を調べるため、隠れ層を64から88/90へ増やした。
**88版はhumを改善したが、noise F1は0.8942で基準未達。両候補とも不採用。** 最終評価と実機更新は行っていない。

| 隠れ層 | 整数重み/バイアス等 | hum F1 / 操作recall | noise F1 / 操作recall |
|---|---:|---|---|
| 64 | 17,380B | 0.891 / 0.792 | 0.894 / 0.875 |
| 88 | 23,884B | 0.912 / 0.833 | 0.894 / 0.854 |
| 90 | 24,426B | 0.868 / 0.771 | 0.870 / 0.792 |

8条件、元発話split、CPU2threads/seed42、1,000epochs上限は同じ。候補/閾値の選択に最終データを使わない。
[88版の数値](experiments/channel-hp160-adaptive-88-v2-rejected.json)・[90版の数値](experiments/channel-hp160-adaptive-90-v2-rejected.json)。
容量を増やすだけでは全条件を通らず、大きい方が常に良い結果にもならなかった。
validationの小さい差を実声の改善と扱わない。次はcheckpoint選択の目的と入力診断を切り分ける。

## 資源の上限

- 260→90→7のint8 weight/int32 bias/shiftは24,426B。既存の24KiB上限を変えない。
- 学習・量子化・archive読込のhidden上限を90へ揃えた。91以上は拒否する。
- 学習CLIは不正な幅/epochを波形読込前に拒否する。archiveの巨大shape/型/メンバー検査も維持する。
- 88版の既存C整数推論とPythonのlogit一致を追加確認。既存Cは生成headerのhiddenサイズを使い、実装変更は不要だった。
- この比較でOSへ新しい特徴処理/重みを組み込んでいない。target build、実heap/stack、実機の速度は未測定。

## TDDと失敗記録

先行テストで幅88の学習/モデル読込を拒否するRed、未対応のrecipe引数によるRedを確認した。
最初のGreen試行は量子化側に64上限が残り2件失敗。直接の量子化テストも先に追加してRedを確認し、上限を一元化した。
波形読込前の予算拒否も4条件の先行Redから実装。標準78件、学習7件、既存SDKの両VM統合2件が成功した。
C推論88版のテストは既存Cへの追加検証であり、CをTDDで変更したとは記載しない。

量子化の上限修正前に開始した最初の2実験はValueErrorで終了し、weights/resultは未生成だった。
そのlogを残し、修正とテスト成功後は別のv2出力先で学習を実行した。これらの実験失敗を認識精度の結果や製品の先行Redへ流用しない。

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group train python -m unittest discover -s tests/training -v
uv run --locked --group train python -m arcade_ai_pet.channel_training \
  --dataset datasets/jtalk-aligned-v3 --candidate hp160 --frontend adaptive \
  --hidden 88 --epochs 1000 --output checkpoints/fresh-capacity88
```

新規GPU/API課金は0米ドル。本体録音の外部取得・Flash操作・GPIO/PSRAM変更は行わない。
