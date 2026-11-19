# PCMからlog-mel特徴量を作るC参照

2026-11-19。先行10契約の未実装ImportErrorを確認後、C/frontend wrapperを追加。10件/3.904秒でGreen。全host377件/41.621秒、学習48件/6.355秒、固定SDK2件/3.039秒がskipなし成功した。

- HP160/mean除去/peak4096は既存Cを再利用し、元13固定sourceはbyte不変。
- doubleの数学table/FFT、適応VAD、前後10frame、64bin分数重なり、32band中心化とenergy64/duration1を実装。モデル・閾値・評価行は変えない。
- 代表PCMの特徴差2e-6以下/Q5一致、contiguous/bankedのbit一致、無音/最大振幅/位置境界、入力不変、不正pointer/count/整列/重なり/不足workspace/終端canaryを先行検査。
- 特徴抽出workspace26,672B（上限32KiB）。推論12,288Bとは別。target上で同時に必要な全RAMや処理時間の保証ではない。
- C APIはアクセス可能なcaller所有buffer用で、guest sandboxではない。同じworkspaceの同時呼出しはcallerが直列化する。

```sh
uv run --locked python -m unittest discover -s tests -p test_logmel_native.py -v
```

次は固定候補と元の全4,808行をPCMから再構築し、既存cacheの再現、特徴差、全Q5/整数logit/最終判定、banked、24group/56指標を照合する。実施前の条件を固定し、数値差が出たら記録して止める。新しいモデル探索/追加学習はしない。

正式freeze・未使用の独立最終・S3全体build/実行は未完了。実機a553/16app、最新実声6/15、Vast.ai累計0米ドルを維持する。

[先行条件](../experiments/neural-domain-20261119/native-logmel-plan.json)。
