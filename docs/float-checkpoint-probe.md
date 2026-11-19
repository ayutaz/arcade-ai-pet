# 元float checkpointと整数版を対にするホスト診断

2026-11-19。先行TDD20契約と回帰394件が成功し、実データ前に[1構成のrecipe](../experiments/neural-domain-20261119/float-probe-plan.json)を固定した。実データの再計測/C/実機は未確認。[初期案](float-checkpoint-probe-proposal.md)のJSONは当時の記録を保持する。

## 記録と選定

`float_probe=True`を明示した場合だけ、各checkpointの実際のTorch evalと整数版を同じ選定用配列で比較する。16条件/7クラス別に、両方正解・floatだけ正解・整数だけ正解・両方誤り・生順位の一致を記録する。floatのlogitは生値、整数はQ10なので、数値差の集計ではfloatを1,024倍する。

選定/校正/採用は従来の整数版だけを使う。元neuralを含む全24domainと結合条件も維持する。floatの判定を使った閾値探索や、保留/最終データの用途変更は行わない。

選定された元float stateだけを非公開の実験領域に`float-checkpoint.npz`として保存する。固定5層/BN epsilon1e-5/headに対応したfloat32/int64の32配列で、ファイルは0600。BNのrunning値/カウンターも含み、source/入力/整数重み/epochとhashで対応を残す。optimizer stateは含まないので、学習の完全な再開を保証する形式ではない。

stateからBNをfoldして整数化し、選ばれた整数層と全配列が一致することを保存前と校正後に検査する。stateと整数層の不一致、途中のstate変更、未宣言のstateは拒否する。配列本体をJSONやGitへ含めず、公開するのは集計/来歴だけ。未選定checkpointは集計/元logitのhashを残し、float stateは保存しない。

## 先行TDDと回帰

| 対象 | 意図したRed | Green |
|---|---|---|
| 集計/stateの6契約 | 未実装moduleのImportError | 6件/0.031秒 |
| Torch/学習の7契約 | 未実装capture/restore/float_scores、未対応float_probe | 7件/2.235秒 |
| 候補保存の7契約 | 未対応float_probe。未宣言stateは拒否できずJSON化でTypeError | 7件/0.125秒 |

```sh
uv run --locked python -m unittest discover -s tests -p 'test_float*probe*.py' -v
uv run --locked --group train python -m unittest discover -s tests/training -p test_float_checkpoint_probe.py -v
```

手計算の順位差/同点、型/shape/値域/BN/state digest/整数binding、全3形式のfloat再現・入力/BN/勾配/RNG不変、診断有無での整数層/選定一致、診断統計を選定に使わないこと、上限/期限/校正中の変更/途中保存拒否を確認した。回帰350件/36.518秒・学習42件/5.876秒・固定SDK2件/3.409秒がskipなし成功。旧v3凍結13sourceもbyte不変。

[MPS小型fixture](../experiments/neural-domain-20261119/float-probe-mps-smoke.json)でも各4stepの対照/診断fitの全整数層と選定epochが一致し、選定float stateを再ロードして元logitのhash/比較集計を再現した。実装後の追加検証で、先行TDDや実データ品質の保証とは分ける。

## 再計測前の固定条件

既存のlog-mel構成（2,113入力、数値23,411B）を全体学習する。30epoch/batch256/seed42、ローカルMPS、上限8,000step/fit900秒、予定7,350step。元archive/保留4合成条件と20無音/全評価行は維持する。新headを追加探索せず、消費済み最終2組を使わない。

今回観測するのは新しい学習の元float stateであり、過去の未保存float値の復元ではない。同じ学習構成でも、旧重みhashの一致を成功条件にしない。品質に合格した場合もC/正式freeze/新独立最終/S3資源を別途確認する。実機a553/16app・実声6/15、既存の不採用とVast.ai累計0米ドルを維持する。
