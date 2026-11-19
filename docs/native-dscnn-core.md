# 整数6層CNNのC参照

2026-11-19。既存logmel-dscnn-v1の整数推論を独立したC11実装へ移す。PCM処理・モデル採用・OS統合は含まない。

## 契約

入力はfloat32の2,113特徴、固定6層のint8重み/int32 bias/出力channel別shift。Q5への偶数丸め、3×3/1×1・depthwise・padding/stride、ReLU6、Q5→Q10、負数の床除算、int32飽和を既存Python参照と照合する。5閾値の棄却は同点の最初のclassを優先し、差の計算にint64を使う。

APIはアクセス可能なバッファを渡すtrusted caller用で、guest用sandboxではない。null/数/整列/shift/値域/特徴tail/重なりを確認してから書き込む。workspaceはcaller所有の12,288Bで、推論内のheap確保・可変長配列・グローバル可変状態を使わない。host wrapperはモデルをcopyし、同じworkspaceの呼出しは直列化をcallerの責任とする。

## TDDと回帰

先行9契約。未実装moduleのImportErrorでRedを確認してから、C/header/host wrapperを実装し9件/3.451秒でGreen。

```sh
uv run --locked python -m unittest discover -s tests -p test_dscnn_native.py -v
```

| 検査 | 結果 |
|---|---|
| 全host回帰 | 367件 / 39.121秒、skipなし |
| 学習回帰 | 48件 / 6.606秒、skipなし |
| 固定SDK両VM | 2件 / 3.125秒、skipなし |
| 元v3の固定13source | 原本とbyte一致 |
| ASan/UBSan追加検査 | 64vector、極端なbias/shift、短いworkspace/重なり/NaN、終端32B canaryが成功 |

ASan/UBSanと実モデルの後続照合は追加検証で、後付けTDDと呼ばない。hostの秒数をS3の推論速度と扱わない。

## 次の照合

固定した学習時mask候補の数値23,411B/5閾値と、元の全4,808行を1回照合する。評価行と閾値を変えず、24group/56指標を個別に確認する。CのPCM→log-mel前処理、正式freeze、新しい独立最終評価、S3資源は未完了。実機a553/16app・実声6/15・Vast.ai累計0米ドルを維持する。

[照合条件](../experiments/neural-domain-20261119/native-dscnn-core-plan.json)。
