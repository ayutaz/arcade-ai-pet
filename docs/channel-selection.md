# 苦手な条件を優先したcheckpoint選択

2026-11-18。全条件をまとめたF1だけで選ぶと少数条件の失敗を隠す可能性があるため、validationの条件別最悪F1を優先する選択を追加した。
**今回の64/88版は従来と同じ整数重みを選び、認識結果は改善しなかった。** どちらも不採用で、最終評価と実機更新は行っていない。

従来の選択を既定値として維持し、`--selection worst`を明示した時だけ切り替える。
各checkpointで条件別F1/誤起動を計算して順位付けし、採用時は従来通り操作recallも含む全条件のゲートを別に検査する。
候補の選択はvalidationのみ。実声の受入条件や最終データへ合わせて基準を変えていない。

| 版 | 選択epoch | noise F1 | 全条件採用 |
|---|---:|---:|---|
| 64 / worst | 896 | 0.8939 | 不合格 |
| 88 / worst | 1,000 | 0.8942 | 不合格 |

[64版](experiments/channel-hp160-adaptive-64-worst-v1-rejected.json)・[88版](experiments/channel-hp160-adaptive-88-worst-v1-rejected.json)。
従来版との比較はNPZ内の全integer arrayで一致を検査した。容器ファイルの時刻等を重みの差と扱わない。

## 学習データとvalidationの差

88版の保存重みを変更せず、同じ入力条件でtrain側も追加確認した。

| 条件 | train F1 / 操作recall | validation F1 / 操作recall |
|---|---|---|
| hum | 0.9995 / 1.000 | 0.9115 / 0.833 |
| noise | 0.9903 / 1.000 | 0.8942 / 0.854 |

[比較JSON](experiments/channel-generalization-gap-v1.json)。学習側はほぼ正しく、単にepochや幅を増やすより入力の多様性を調べる根拠になる。
現行noise加工は同じ元発話IDの3加工に同じ追加雑音seedを使う。次はtrainだけの独立雑音/発話バリエーションを比較する。
この差を実機の失敗原因と断定しない。独立した実人間の評価は別工程。

## TDDと検証

先行テストは「全体平均が高くても少数条件が悪いcheckpointより、最悪条件が良い方を優先する」。
新しい順位関数/fit引数/recipe引数が未実装のRed後に実装し、標準79件、学習8件が成功した。
不正なgroup数値・dtype・長さを拒否し、古い選択は既定のまま保持。TDDのGreenを認識品質の合格と混同しない。

```sh
uv run --locked --group train python -m arcade_ai_pet.channel_training \
  --dataset datasets/jtalk-aligned-v3 --candidate hp160 --frontend adaptive \
  --hidden 88 --selection worst --epochs 1000 \
  --output checkpoints/fresh-worst-channel
```

実験はCPUだけ。配布モデル、OS、実機、GPIO/PSRAM、料金は変更していない。新規GPU/API課金は0米ドル。
