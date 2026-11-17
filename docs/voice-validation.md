# 実声試験の記録と集計

このツールは音声を録音せず、USBにも接続しない。事前に固定した[受入条件](acceptance.md)の30試行を作り、数値の結果だけを検査/集計する。集計ツール単独ではFW適用や実機検証を証明しない。現在の基本実機試験は[別の結果](device-validation.md)を参照する。

## 固定する試行

| 種類 | 回数 | 目標 |
|---|---:|---|
| 5操作 | 各3回、計15回 | 正解12回以上 |
| 対象外の言葉 | 10回 | 下の無音と合わせて誤起動1回以下 |
| 無音 | 5回 | 同上 |

「あそぼ/おいで/まって/おやすみ/ありがとう」を1周して3周。次に対象外10語、無音5回。固定IDごとに一度記録し、失敗した試行も残す。調整後の再試験は別ファイルへ保存する。

```sh
uv run --locked python -m arcade_ai_pet.voice_trials plan
uv run --locked python -m arcade_ai_pet.voice_trials summarize reports/voice-results.json
```

[空の記録例](../examples/voice-results.example.json)はnullを含むため、そのままでは集計を拒否する。Codexが実際の観測を記録し、人にJSON編集を依頼しない。
`records`にはtrial番号、observed、latency_msだけを入れる。録音・名前・MAC・SSID等は含めない。30件の欠落/重複、余分な項目、非有限値、過大なファイルは拒否する。

| observed | 意味 |
|---:|---|
| 0〜4 | 上記の5操作、記載順 |
| 5 | 対象外 |
| 6 | 無音 |
| -1 | 推論等のエラー。1件でも採用条件は不合格 |
| 7 | schema2のみ。画面で操作なしを確認。nativeの対象外/無音/エラー内訳は不明 |

latency_msは録音終了から判定を観測するまでの時間。採取時の開始/終了点と観測周期は、別の実機結果資料にも残す。このツールは時間そのものを測定しない。

## 合成と人の声の区分

| scope | 対象 |
|---|---|
| host_fixture | ホストの模擬データ |
| device_synthetic | 合成音声を実機のマイクへ聞かせる音響試験 |
| device_human | 人が実機へ話す試験 |

`thresholds_met`は入力された数値が基準内かを示す。scopeは申告された区分であり、ツールだけでは由来を保証できない。`hardware_acceptance_proven`は常にfalseにし、集計だけで実機合格にはしない。固定FW/app/model、実際の観測経路、実声・聴取・表示の証拠を別に照合して最終判断する。

本体の録音を外部へ取得・送信する必要はない。FW個別許可、対象外appの保持、資源解放の条件は維持する。

## 画面から操作を観測する場合

nativeの戻り値を取得できる記録は従来のschema1を使い、7×7行列を維持する。
描画命令CRCから操作だけを観測する場合はschema2にし、無反応を7として記録する。
対象外5や無音6を推測して埋めない。schema2の行列は7×8、列名はobserved_labelsで示す。
`unresolved_native_trials`と`native_labels_fully_reported`で内訳を区別する。

`errors`は明示的に記録された-1の数で、無反応7の内部でエラーが無かったと保証する値ではない。
操作の成功/誤起動とnative分類の測定は別。録音/再生の正常完了・所定app/CRCの由来も別の実機結果で照合する。
