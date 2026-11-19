# 既存係数を固定する補助入力のホストPoC

2026-11-19。先行TDDによる実装と回帰は成功。実データ学習前に[1構成のrecipe](../experiments/neural-domain-20261119/frozen-prefix-plan.json)を固定した。品質の合格やC/実機の動作は未確認。[初期案](frozen-prefix-residual-proposal.md)のJSONは作成時の状態を保持する。

## 変更範囲

旧21/24条件の不採用モデル8595204e…を固定した土台とし、平均スペクトル32値に対する7×32=224係数だけを0から学習する。5畳み込み層とhead既存3,137列の重み/bias/shiftをbyte不変にする。実機で採用しているモデルとは別物。

入力2,145値・数値モデル/5閾値23,635Bは前回の全再学習版と同じ形式。固定row shiftを使い、追加係数はint8の−127〜127へ制約する。CPU float64の量子化を考慮した勾配近似で学習し、checkpointは実際の整数logitで選ぶ。第0epochの無変更モデルも候補に含める。

## TDDと追加照合

| 対象 | 意図したRed | Green |
|---|---|---|
| 固定層/丸め前accumulator/整数化6契約 | 未実装moduleのimport失敗 | 6件/0.205秒 |
| CPU学習5契約 | 未実装moduleのimport失敗 | 5件/0.945秒 |
| 候補処理5契約 | 未実装fit_residual属性 | 5件/0.063秒 |
| 校正中の土台変更を拒否する1契約 | ValueError未発生 | 6件/0.069秒（上記5件を含む） |

新規17契約。独立scalar畳み込み、負値のfloor/飽和、ties-even、固定prefix、入力不変、追加係数の更新/CPU repeat、step/期限、校正前後の土台変更拒否、元neural回帰での不採用を確認した。

```sh
uv run --locked python -m unittest discover -s tests -p test_dscnn_residual.py -v
uv run --locked python -m unittest discover -s tests -p test_residual_candidate.py -v
uv run --locked --group train python -m unittest discover -s tests/training -p test_dscnn_residual_training.py -v
```

回帰321件/35.324秒、学習30件/3.735秒、SDK2件/3.202秒がskipなし成功。SDKは既存の固定版を指定。旧v3凍結13sourceもbyte不変。

[追加の4,808行照合](../experiments/neural-domain-20261119/frozen-prefix-zero-proof.json)では、追加係数0で旧モデル/新モデル/分解した整数headの全logitが一致した。旧実データprefixに決定的な範囲内の32値を追加した試験で、追加値は実音声から再計算した平均スペクトルではない。実装後の追加検証であり、TDDや認識精度の保証とは扱わない。

## 実学習前の固定条件

CPU・30epoch・batch512・seed42、最大4,000step/fit300秒。予定3,690step、Adam lr0.003/weight_decay0.01、逆頻度のclass weight。epoch0/1/6/11/16/21/26/30から整数演算で選定する。

データ/4合成条件・20無音の保留/全24domain・結合8条件・選定結合8条件は前回と同じ。ASR不一致で評価行を削除/再ラベルしない。消費済み独立最終2組はfit/選定/校正に使わない。全基準通過後だけC照合/正式freeze/新独立最終/S3資源へ進む。

実機a553/16app・実声6/15、旧候補の不採用を維持する。新FW適用や録音外部取得は行わず、Vast.ai累計0米ドル。
