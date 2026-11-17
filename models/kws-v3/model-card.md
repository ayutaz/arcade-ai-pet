# 日本語5操作 KWS v3 — 合成ストレス評価済み候補

2026-11-18。**固定後の合成評価・ホストC照合と、後続OS候補のESP32-S3 buildを通過。実機・実声は未検証。**
日本語の自由文ASRではなく、5操作と対象外/無音の7分類。
既存実機c032のKWS v2や、`models/kws-v2`の成果物は置き換えていない。

| 項目 | 値 |
|---|---|
| 操作 | あそぼ / おいで / まって / おやすみ / ありがとう |
| 入力 | 16kHz mono PCM16、32,000samples |
| 前処理 | HP160→平均除去→peak4,096。毎録音で状態初期化、PCM16へ丸め/飽和 |
| 特徴 | 背景の下位20%energyを使うadaptive区間、20bin×12MFCC＋20energy |
| モデル | 260→88→7、ReLU6、int8 weight/int32 bias/Q10 activation |
| Tensor予算 | 23,884B。NPZ容器のサイズと別 |
| C scratch | 5,680B。特徴出力1,040B、推論stack等は別 |
| 選択 | CPU2threads/seed42、1,000epochsから856、閾値995 |
| train比率 | hum3回、独立noise5回、他各1回。validationは各1回 |
| train / validation | 27,958 / 2,160。元発話splitは加工前に分離 |

## 固定後の評価

14学習候補をvalidationで比較して本候補を固定。その後は重み・閾値・前処理を変更せず評価した。
既存の固定合成holdoutを再利用。この新候補のfit/選択には未使用だが、新しく収録した実声とは呼ばない。

| 条件 | F1 | 操作recall | 対象外/無音の誤起動率 |
|---|---:|---:|---:|
| clean | 0.997864 | 0.994444 | 0.000000 |
| quiet | 0.996414 | 0.994444 | 0.000000 |
| dc | 0.997382 | 0.994444 | 0.000000 |
| drift | 0.995928 | 0.994444 | 0.000000 |
| hum | 0.927506 | 0.861111 | 0.009021 |
| noise | 0.916143 | 0.855556 | 0.012887 |
| early | 0.987067 | 0.972222 | 0.001289 |
| late | 0.997864 | 0.994444 | 0.000000 |

956例×8条件=7,648評価。2合成話者/452sourceを条件間で再利用し、7,648独立話者とは呼ばない。
[評価JSON](evaluation.json)、[固定JSON](channel-frozen.json)、[固定/再現手順](../../docs/channel-frozen-evaluation.md)。

## Cの照合

[Native JSON](native-evaluation.json)では7,648例すべてで、同じ特徴に対する整数logit、全経路の最終判定、連続/分割PCMの特徴が一致した。
Pythonとの特徴最大差は0.000004292。Cは追加PCM bufferを確保せず、FFT workspaceのunionを区間選択にも再利用する。
198値の挿入sortは最大19,503比較。ホストCの未定義symbolにqsort/malloc系がないことも検査したが、targetでの実heap/stack/速度は別途測定が必要。

`conformance.npz`は固定holdoutからラベル別の先頭1例×8条件の56特徴/logit/prediction。
PCMは含めず、実機の録音でもない。品質はこの小さい照合用データではなく、全7,648評価で判定した。
`channel-metadata.json`は専用runtime/入力契約を持つ。v2のmetadataとして誤って使わない。

## 来歴・ライセンス・制限

HTS Voice Mei 5感情とNIT ATR503 M001のOpen JTalk合成を使用。6voice設定だが人物は2人。
元のvoice著作権、CC BY 3.0の帰属/変更内容は[LICENSE](LICENSE.txt)・[データ条件](../../docs/data-license.md)を参照。
weights、model header、派生照合特徴はCC BY 3.0。手書きcodeと数学定数feature_tables.hはApache-2.0。
原本TTS重み、本体録音、実機ID、資格情報を含めない。機関の推薦を主張しない。

`weights.npz` SHA-256: `374987a02446e805f18427d7de8de3cd955c4833efc6e0a9dbd9fa655c3c59f2`。
未知の人・実環境・長時間待受・録再同時動作の保証はない。実声の正式30試行と実機動画は未完了。
新規GPU/API課金は0米ドル。ファームウェアbinaryの配布やFlash書込み許可とは別の成果物。


## 後続のOS統合候補

[Scratch Arcade OS PR #11](https://github.com/ayutaz/scratch-arcade-os/pull/11)は4必須CI成功後に通常merge（main5d33e3f）。モデル元source595f5d7を固定source48bf239へ取り込み、OS715ホスト試験とS3 buildを通過した。
固定版48bf239を埋め込んだimageは974,080B、SHA `e59158619b0a3956c771d0b099a5d7201ff6d05e86c593c0b27e0250effd4c0c`。imageの実機適用は追加個別許可待ち。binaryはこのrepoへ含めない。
分類結果/入力集計を現在のmic leaseからだけ読むAPIも加えたが、PCM/特徴量の外部取得、GPIO/PSRAM/partition変更はない。実heap/stack・推論時間・実声品質は未測定。[OSの検証範囲](https://github.com/ayutaz/scratch-arcade-os/blob/5d33e3f/docs/voice-service.md)。
