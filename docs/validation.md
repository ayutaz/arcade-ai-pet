# 実装・評価の記録

2026-11-17 JST、macOS arm64、uv、Python3.12.13、NumPy2.5.5、PyTorch2.15.0 CPU。
Vast.ai費用0米ドル。GPU契約・FW更新・本体録音の取得/転送はない。

## TDDと追加検証

| 動作 | Red | Green |
|---|---|---|
| 特徴抽出/発話分割 | features module欠落 | 3件、0.044秒 |
| 加工・WAV読取/整数層 | data/quantize module欠落 | 全8件、0.425秒 |
| 前後無音の除去 | trim_silence import欠落 | 全9件、0.046秒 |
| 学習・混同行列・閾値 | training module欠落 | 全12件、10.023秒 |
| 発話位置の補正 | 同じ音の配置だけで特徴差8.862314 | 4件、0.008秒、差0.001未満 |
| 合成原本cache | source_cache引数未対応 | data 5件、0.663秒。hash改変も拒否 |
| dataset契約 | load_dataset未実装 | 発話ID/分割、非有限値、ラベル照合の試験を追加 |

dataの最初の一括合成CLIは加工/読取の先行試験に対する追加実装で、
CLI全体に先行Redが揃っていたとは記録しない。cache追加時に一括処理の契約試験を先に追加した。
データv1は最長音源28,320samplesが暫定上限28,000を超えて中断。
語を切らず前後の無音だけを除く再現テストから修正した。途中出力は未完成として保持。

## 初回モデル: 不採用

`datasets/jtalk-v2`、6voice、各3速度×2pitch、756原発話×3加工＋200noise。
train1997 / validation270 / test201。発話IDで分割し、同じ原発話の加工版は跨がせない。
voiceの人物は各splitで共有するため、未知話者評価ではない。

```sh
uv run --locked --group train python -m arcade_ai_pet.training \
  --dataset datasets/jtalk-v2 --output checkpoints/kws-v1 --epochs 400
```

48hiddenのMLP、Q10 activation/int8 weight/int32 bias。重み13,044B。
validationで選択したepoch381、拒否閾値2048。
test macro F1=0.708741、accuracy=0.796020、対象外/無音の誤起動率=0.037879。
F1>=0.90に届かず、`accepted_synthetic=false`。このtestを最終の独立評価へ再利用しない。

## 発話位置の補正: 開発用validationのみ

同じPCM原本をhash確認して再処理し、発話のある区間へ20binを合わせた。
`datasets/jtalk-aligned-v3`、64hidden、seed42、600epochsからvalidationだけでepoch51を選択。
validation macro F1=1.0 / 誤起動率0.0、閾値0。
今回testは参照していない。以前の評価結果を見て設計を変更したため、
採用判定には新しく生成して凍結モデルへ適用する評価データを用意する。
このvalidation値を実声の認識率や実機での成功へ拡大しない。

## 実行方法

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group train python -m unittest discover -s tests/training -v
```

現在の資源目標と実声の確認条件は[受入条件](acceptance.md)に固定している。
次は新規合成の独立評価、Cへの特徴/整数推論移植、返答素材、ペットと行動学習を進める。

現在の分離後のunit 13件Green（0.679秒）。学習試験はtests/trainingで別実行する。

## 2026-11-17: 新規合成評価とC移植

| 動作 | 意図したRed | Green/証拠 |
|---|---|---|
| 候補固定/新規評価 | frozen module欠落 | 固定hash、旧ID/改変重み/出力上書き拒否 |
| 新しいprosody/noise | rates引数未対応 | PCMも新seed、開発IDと重複0 |
| 元waveの重複防止 | 別IDなら同じwaveが通過 | WAV hash重複を拒否 |
| validation独立照合 | 重みを壊しても申告成功だけで通過 | 直接推論して未達なら固定拒否 |
| C export | native module欠落 | 表/重みの決定的出力 |
| C特徴/整数推論 | 実Cファイル欠落によるcompile失敗 | Native 4件、0.772秒 |
| 配布モデル | 配布メタデータ欠落 | 固定重み/38特徴とCの整数logit照合 |
| 全C経路監査 | native_evaluation module欠落 | synthetic原本の再構成、改変header/wave拒否 |
| 数値exportの境界 | threshold文字列からC注入が可能 | int範囲外を出力前に拒否 |
| NPZ資源 | NumPyのallocation前検査なし | 全6memberを揃えた巨大shape headerをallocation前に拒否 |
| 再現学習 | hidden/validation_only未対応 | testをscoreせず指定幅のcheckpointを保存 |

固定17,380Bモデルを新しく作った956例で評価し、macro F1=0.978908、
対象外/無音の誤起動率=0.002577。評価後にモデルを変更していない。
全956例のC/Python整数logitと最終classが一致、特徴差の最大は3.219e-6、C scratch5,680B。
ホスト中央値0.224ms/p95 0.847msで、本体の速度ではない。
公開範囲と完全な結果は[モデルカード](../models/kws-v2/model-card.md)へ。

最新のunit29件Green（3.202秒）、学習3件Green（1.799秒）は再現学習追加前の値。
学習の4件目は先行Red確認後に追加実装し、別に再実行して記録する。
実声/実機/GPU利用は未実施。古い不採用結果と開発validationは履歴として保持する。

再現学習の先行Redはhidden引数未対応、実装後は学習4件Green（1.670秒）。
修正済みunitの再実行は29件Green。Cの数値比較、固定後の合成評価、公開モデルの同一性を含む。

## 未使用末尾のエネルギー床の回帰修正

32000samplesのうちFFT窓に含まれない最後の80samplesだけに信号がある場合を先行試験。
Cだけenergyが16となり、Pythonとの差16.0でRed。最大energyにも1e-8の床を適用し、
Native 6件Green（0.440秒）。凍結重みは変更せず、全956例を再監査した。
整数logit/最終判定956/956一致、特徴差最大3.219e-6を維持。
公開native-evaluation.jsonは修正後のC hashと測定値（Mac中央値0.224ms/p95 0.847ms）へ更新。
旧監査はローカルreportsに保持し、実声/実機の証拠へは拡大しない。

```sh
uv run --locked python -m unittest discover -s tests -p test_native.py -v
uv run --locked python -m arcade_ai_pet.native_evaluation \
  --model models/kws-v2 --dataset datasets/jtalk-heldout-v1 \
  --output reports/native-kws-v2-tail-fix
```

修正後の全unit30件Green（2.836秒）、学習4件Green（1.944秒）。

## CIのrunner供給不足

GitHubの旧run 43924536580はMac arm64 runnerを取得できず失敗。
annotationに`The job was not acquired by Runner of type hosted even after multiple attempts`と
arm64のcapacity constraintsが記録され、テストの実行失敗とは区別した。
ホストMac契約はmacos-15-intel、CPU学習はubuntu-24.04へ変更。
必須check名とPR/管理者保護を維持し、skip/失敗無視/迂回mergeは使わない。
CI構成の宣言変更で、新しいTDD実装とは数えない。手元のMac arm64検証は上の結果を保持する。

## 収集行動のCPU学習と整数C

World/teacher/benchmark、行動学習、数値export/C推論を先行module欠落のRedから実装。
World 2件Green（0.017秒）、学習の再現1件Green（1.723秒）、export/C 1件Green（0.382秒）。
公開候補がない先行試験のRed後に固定成果を置き、2件Greenへ。

```sh
uv run --locked python -m unittest discover -s tests -p 'test_behavior*.py' -v
uv run --locked --group train python -m unittest discover -s tests/training -p test_behavior_training.py -v
uv run --locked --group train python -m arcade_ai_pet.behavior_training \
  --output checkpoints/behavior-v1 --epochs 300
```

新しい合成盤面のBFS模倣学習で、音声と別のモデル。固定後の未知64seedに対して
平均得点8.640625（random0.96875/greedy7.375/BFS24.734375）。C/Python全7680stepの行動一致。
ruleを一般的に超えたとは主張せず、局所観測だけで行き詰まる制限も記録した。
[モデルカード](../models/behavior-v1/model-card.md)と固定JSON/NPZ/C表を公開、外部データなし/Apache-2.0。

全unit34件Green（3.597秒）、学習5件Green（1.672秒）。CPUのみ、GPU費用0米ドル。
行動のWASM/app/実機統合はまだない。

## ペットの状態と返答素材

- ペット状態はmodule欠落の先行Redから実装。5操作/対象外棄却/ボタン代替/終了、
  学習した移動、待機/睡眠で時計停止、プレイヤー別得点/壁/外周を3件Green（0.007秒）。
- 返答のcodec/exportはmodule欠落の先行Redから実装し、3件Green（0.033秒）。
  合成器のfixture追加は実装前だが、その追加後の独立Redを再実行していないため、
  合成CLI全体まで先行TDD済みとは拡大しない。公開モデル/原WAV hashを記録する追加検証も実施。
- 分割PCMはCのbanked symbol欠落でRed。8bankを直接読む共通coreへ変更、1件Green（0.344秒）。
  連続PCMのNative6件もGreen（0.407秒）、956例の全C経路を再監査し判定/logit一致。
  C scratch5,680Bのまま、2つ目の64KB PCMを確保しない。

```sh
uv run --locked python -m unittest discover -s tests -p test_pet.py -v
uv run --locked python -m unittest discover -s tests -p test_replies.py -v
uv run --locked python -m unittest discover -s tests -p test_features_banked.py -v
uv run --locked python -m arcade_ai_pet.native_evaluation \
  --model models/kws-v2 --dataset datasets/jtalk-heldout-v1 --output reports/native-kws-v2-banked
```

返答5種は8kHz PCM8で25,569B、事前合成であり本体内TTSではない。
ペットはPythonの状態モデルまで。描画/WASM/OS音声サービス/実声/本体録音の取得はまだない。

この段階の全unit41件Green（4.220秒）、学習5件Green（1.658秒）。

## 返答の出力専用バッファへの適合

OS接続前に最長返答8010×2framesが既存speaker専用16000framesを超えると判明。
公開素材の全長<=8000を先行試験してRed。語を切らず合成速度を1.10→1.15へ調整し、
5素材を再生成。合計25,418B、各sample数[7839, 3116, 4859, 4619, 4985]、1秒以内に収めた。
旧素材/metadataはGit履歴へ保持し、本体への書込みは行っていない。

OS向けGCC buildで2箇所の連続ifがmisleading-indentationとして拒否された。
警告を無効にせず、文の改行だけを修正。動作変更や新しいTDDとは扱わない。

改行後の全956例もC/Pythonの整数logit/判定が一致。unit42件Green（3.916秒）。

## ペットのC/WASM化

app.c欠落を先行Redにし、模擬audio I/O下の5操作/棄却/終了・行動・プレイヤー入力を
3件Green（0.509秒）。ビルダーはmodule欠落でRed→2件Green（0.003秒）。
最初のbuilder fixtureはmacOSの/var aliasを含んで失敗したためreal pathに修正し、機能のRedとは数えない。

実WASMで起動fuel超過を再現する統合試験を先に固定し、initの到達性BFSを
同じ判定のbitset floodへ置換。fuelは50,000を維持。nativeの盤面/行動照合も通過した。
さらに実SDKの66px高の数字で画面外描画が発生し、実数字assetをfixtureへ持ち込んで
先行再現（2件Red）。盤面と数値の座標を調整し、native3件Green（0.385秒）。

数字assetを無描画stubにした初版native試験は描画範囲を保証していなかった。
改めてSDK同一asset/実WASM両VMで確認。終了後debugを読もうとした統合fixtureも
終了前の値を保持するよう修正し、productionのRedとは数えない。

```sh
uv run --locked python -m unittest discover -s tests -p test_pet_native.py -v
uv run --locked python -m unittest discover -s tests -p test_build.py -v
uv run --locked python -m unittest discover -s tests/integration -v
```

実WASM126frames、模擬voice全7classを両VM/160KiB poolで確認。
統合2件Green（2.433秒）、全unit47件Green（4.453秒）、学習5件Green（1.870秒）。
package5,142B、実機未適用。[使い方と画像の区分](pet-app.md)。

## ホストSDKの接続修正

SDKを3bb9548846505eac303927b9064c6a8d0eb44f79へ固定。Mac native USBの再接続で稼働appがresetされる共通修正を取り込む。pin更新であり、新しいTDDのRedがあったとは記録しない。
unit47件（4.271秒）/両VM統合2件（2.432秒）が成功。Petの再build SHAは14f657f0…と完全一致し、固定レビューの5,142B packageとFW候補は変更しない。追加の照合scriptは最初にMacの一時path aliasで拒否され、実pathへ解決して再実行した。実装のRedではない。実機への新FW書込みは未実施。

## 実声30試行の集計

`uv run --locked python -m unittest discover -s tests -p test_voice_trials.py -v`でmodule欠落Redを確認後、固定30試行/採用境界/推論エラー/欠落重複/余分な情報/CLI/FIFO拒否を最小実装し6件Green。
巨大な整数latencyを追加してOverflowErrorのRed→上限検査を先に行ってGreen。実機へ合成音声を聞かせる区分も未対応のRed→人の声と別scopeにして7件Green（0.031秒）。
集計は録音・USB・FW操作を行わず、scopeを自動証明しない。模擬入力が満点でもhardware_acceptance_provenはfalse。新しい実声結果を得たとは記載しない。

最終回帰は標準54件/4.569秒、両VM統合2件/2.406秒が成功。nullの記録例はCLIがexit2で拒否。文書リンクも確認した。

導入監査でpet-app.mdが旧SDK3bb154bを案内し、builderの固定3bb9548と異なることを発見。guide/CI/builderの整合テストを先行追加し、旧guideでRed→現行pinへの修正でGreen。過去の固定FW source3bb154bと当時の記録は変更しない。

SDK案内の回帰追加後、標準55件/4.495秒成功。app/VMコードは変更せず、直前の両VM統合2件も成功済み。
