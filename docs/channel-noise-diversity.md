# 独立雑音と判定閾値の比較

2026-11-18。学習側のnoise F1=0.9903に対しvalidation=0.8942だったため、trainにだけ雑音のバリエーションを増やした。
**独立雑音は改善したが、humの操作recallは38/48=0.7917で、全条件採用は未達。** 最終評価・実機更新は行っていない。

## 雑音の数と多様性の分離

同じ学習量で2候補を比較する。noiseだけ5倍にし、その他の条件は1回。validationの入力と件数は変えない。

| 候補 | trainのnoise | train / validation例数 |
|---|---|---:|
| repeat | 同じ入力を5回 | 23,964 / 2,160 |
| fresh | 元の1回＋独立seedの4回 | 23,964 / 2,160 |

追加seedは元発話ID・加工後の合成PCM・replica番号のSHAから固定。validationへ追加seedを使わない。
両候補はHP160/adaptive、peak4,096正規化、hidden88、CPU2threads/seed42、1,000epochs上限、worst選択で比較した。
重み23,884Bで、配布済みKWS v2は変更しない。

| 候補 | hum F1 / 操作recall | noise F1 / 操作recall | 全条件採用 |
|---|---|---|---|
| repeat / 粗い閾値 | 0.8855 / 0.8333 | 0.8892 / 0.8542 | 不合格 |
| fresh / 粗い閾値 | 0.9014 / 0.7917 | 0.9049 / 0.8333 | 不合格 |
| fresh / 境界全探索 | 0.9075 / 0.7917 | 0.9049 / 0.8333 | 不合格 |

独立雑音と、noise条件の学習量を増やした効果を混同しない。両候補とも元の8条件すべてで判定し、都合の良い条件だけで採用しない。

## 整数境界での閾値選択

以前は0/128/256/512/…/4,096の粗い候補だった。
整数marginで`margin<threshold`の時だけ対象外へ変わるため、0と各margin+1を比較すると、許可範囲0〜4,096の全ての判定パターンを調べられる。
検証データだけで境界を比較し、最終データを選択に使わない。

freshは閾値1,888を選んだが、humの操作recall不足は解消しなかった。F1だけで合格へ読み替えない。
再選択は新しい出力先へweightの同じbyteをコピーし、旧validationとweight/data hashを確認して行う。旧結果を上書きしない。
モデルの学習をやり直した結果と、同じ重みでの閾値再選択を分ける。

数値: [fresh初回](experiments/channel-noise-fresh-v1-rejected.json)・[repeat初回](experiments/channel-noise-repeat-v1-rejected.json)・
[fresh再選択](experiments/channel-noise-fresh-exact-v1-rejected.json)・[repeat再選択](experiments/channel-noise-repeat-exact-v1-rejected.json)。
仮想条件のvalidationであり、実機の入力原因や未知話者精度の証明ではない。

## TDDと再現

- trainだけを増やす、validationのbyte相当の配列は不変、freshは異なりrepeatは同一、再生成が決定的、過大条件はI/O前に拒否する先行テスト。
- 粗い刻みだと見落とす1,279/1,400のmarginを使い、閾値1,280で正しく分離する先行テスト。
- 再選択は原本保持・同じweightのコピー・改変拒否を先にテストした。
- 未実装APIのRed後に実装。標準82件、学習8件が成功。TDDのGreenと品質ゲートを区別する。

```sh
uv run --locked --group train python -m arcade_ai_pet.channel_training \
  --dataset datasets/jtalk-aligned-v3 --candidate hp160 --frontend adaptive \
  --hidden 88 --selection worst --noise-replicas 5 --noise-mode fresh \
  --epochs 1000 --output checkpoints/fresh-diverse-noise
```

上の現行CLIは境界全探索を含む。旧2学習実験の粗い閾値は保存した履歴で、現行の出力と同じ選択規則とは呼ばない。
既存checkpointの再選択は`uv run --locked python`から`reselect_threshold(checkpoint, dataset, fresh_output)`を呼ぶ。
新規GPU/API課金は0米ドル。本体PCM、未知GPIO、PSRAM、Flashは触らない。

次は学習内の条件別比率と、独立した発話バリエーションを分けて検証する。実機入力の未確定部分も残しておき、単一仮説へ決め打ちしない。
