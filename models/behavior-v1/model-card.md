# 収集ゲームの相棒 policy v1

BFSの最短移動を教師とする模倣学習。強化学習ではない。生成した8×6盤面、壁4個、
相対goalと隣接4方向の障害物から4行動を選ぶ。MLP 6→32→4、ReLU6。
学習seed0〜255（各64状態）、validation10000〜10031、test20000〜20063。
seed73、300epochs、整数validation正解率でepoch221を選択後に重みを固定した。

| policy | 未学習64seed、各120stepの平均得点 |
|---|---:|
| 学習済みint8/Q10 | 8.640625 |
| 固定seedのランダム | 0.96875 |
| 隣接障害を避けるgreedy | 7.375 |
| 全盤面を使うBFS参照 | 24.734375 |

採用条件はランダムを超えること。greedy/BFSを一般的に上回るモデルとは主張しない。
近傍情報だけでは壁の向こうを回り込めず、同じ場所に止まることがある。
各policyは同じseed/初期配置/step数で別rolloutし、goal列は回収数に応じて進む。
1手ずつBFSを呼ぶ実装でなく、学習済み整数MLPで推論する。

パラメータ472B（int8重み320、int32 bias144、shift8）。NPZのコンテナ容量は別。
学習データ/重みは本プロジェクト生成、Apache-2.0。声の重みCC BY 3.0とは別。
CPUのみ、Vast.ai費用0米ドル。実機の速度/ゲーム体験は未検証。
詳細seed/各得点/hashは[evaluation.json](evaluation.json)。

```sh
uv run --locked --group train python -m arcade_ai_pet.behavior_training \
  --output checkpoints/new-behavior --epochs 300
```

凍結NPZ SHA-256: `5260061fddf6642c3d0b10583faaadf1aec6729564cd5035326f45d78f9744ba`。
