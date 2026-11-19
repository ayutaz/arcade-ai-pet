# 固定CNNの中間値丸めの比較

2026-11-19。`dscnn_shadow.scores`は、保存済みのint8重み/int32 bias/shiftを変えず、float64で計算するホスト専用の診断。入力Q5化・中間値の切捨て・出力整数化を外し、ReLU6と層構成は維持する。Q10相当の単位を持つfloat64の7出力を返す。

**元のfloat学習checkpointの復元ではない。** 重み/biasの量子化前の値は戻せず、その量子化による精度低下は判定できない。shadowを採用モデルやESP32-S3の実装とは扱わない。

## TDD

4契約を先に追加し、未実装moduleのimport失敗をRedとして確認。その後に最小実装した。

```sh
uv run --locked python -m unittest discover -s tests -p test_dscnn_shadow.py -v
```

4件/0.141秒が成功。全5畳み込みとheadを別のPython scalarループ/`math.fsum`に照合し、biasのQ5基準、出力channel別shift、ReLU6、NCHWのflatten、tailの半Q5値、int32での未飽和、batch分割、入力/重み不変を検査する。回帰286件/33.108秒、学習21件/3.176秒、SDK2件/2.489秒もskipなし成功。

## 比較の固定条件

[比較前の条件](../experiments/neural-domain-20261119/logmel-shadow-plan.json)で、同じ重み/5閾値と開発用24条件/4,808行を使う。旧整数版の全混同行列を再現してからshadowと比較する。shadowの閾値判定用変換はties-to-evenでQ10整数へ丸め、int32へ飽和する。float出力からの順位変化も別に数え、入力/中間値の丸めと混同しない。

学習・閾値変更・新しい最終データ・C移植・実機操作は行わない。分類の誤りが同じままなら丸めだけで直るとは判断しない。差があっても、新しい整数候補/独立評価/S3資源の検証は別工程にする。[前の誤り診断](development-errors.md)を保持する。
