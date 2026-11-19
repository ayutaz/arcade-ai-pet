# ニューラル合成音声の追加学習計画

2026-11-19、開発分割v2・学習選定v4（最終評価前）。**学習・最終評価より先に、条件と選定方法を固定する。** 旧v3の実声結果は6/15で未達。本計画は新モデルや実声改善の完成報告ではない。

## データ

[source-plan.json](source-plan.json)のcanonical JSON SHA-256は `67598d206e18b8d78bdf133fe8c9a0a4fc72b690b090ec88298926e5ab06554d`。生成を開始した実planと一致する。エンジンcommit、モデル/codec/ASR/生成環境lockのhash、seed、variant数、最大試行数を固定する。

| 用途 | 条件 | 扱い |
|---|---|---|
| 既存開発データ | Open JTalkの既存756元発話＋200音/無音対照、variants3 | 元のID/split/帰属を維持 |
| 新開発データ | 一般的な成人の声の8caption、各21prompt、最大2試行、variants3 | 独立ASR一致だけを採用。棄却・再試行を保存。全21promptの採用を要求。学習前に開発分割v2を作る |
| 新最終評価 | 別の4caption、各21prompt、最大2試行、variants3 | モデル凍結後まで生成しない。選定には使わない |

captionは異なる条件の指定であり、異なる実在話者を証明しない。21promptの内訳は操作語5件と対象外16件。元音源hashの重複と発話IDの漏洩を拒否し、同じ元発話のvariantを別splitへ分けない。旧データのID/splitは維持する。新raw開発データはvalidationの操作語不足が判明したため、[開発分割v2](../../docs/development-domain-validation.md)で元ID/source/splitを保存しつつ層化した別dataset版へ準備する。新raw側の旧test所属も変更し得るが、それを独立評価とは呼ばない。モデルの予測で所属を選ばず、最終4条件は変更しない。既存のOpen JTalk最終データは再利用する場合も回帰試験と呼び、新たな独立評価とは区別する。

## 最初の候補

新旧の開発データを結合し、既存v3と同じ構造・前処理で学習する。最初の比較ではデータの効果を調べ、最終条件を見て構造を選ばない。

| 項目 | 固定値 |
|---|---|
| 前処理 / frontend | hp160 / adaptive / peak4096 |
| 構造 | 260入力、hidden88、7出力 |
| 学習 | seed42、最大600epochs、worst-profile checkpoint selection |
| 学習ノイズ | fresh、noise5、hum3 |
| 閾値 | development validationだけで既存の整数境界探索 |
| 既存baseline | v3、threshold995、重みSHA `374987a02446e805f18427d7de8de3cd955c4833efc6e0a9dbd9fa655c3c59f2` |

baselineと新候補は同じ開発validationの8profileで比較する。新候補がvalidationを通らなければ凍結・最終生成へ進まない。失敗を記録し、開発データの範囲で分析する。

## 次へ進む条件

1. source/hash/内容点検・再構築の契約が成功する。
2. 混合validationに加え、旧・新を別々のvalidation domainとして検査する。全7クラスの存在と、両domain全8profileでmacro F1≥0.90、false activation≤0.05、command recall≥0.80を要求する。合算の好成績で新しい声の未達を隠さない。
3. 検証済みモデルを凍結してから新final条件を生成し、同じ基準で一度評価する。結果を見て選定し直した場合は独立評価を使い切ったと記録する。
4. Python/Cのスコアと判定、C frontend、従来データの回帰、重みとscratchの資源制約を検査する。同じhidden88でも対象buildのFlash/RAMを別途測る。
5. 新FWが必要なら具体的なimage/hash/領域/保全・復旧条件をレビューし、個別の許可後にだけ実機へ適用する。
6. 合成評価の合格と実声の正解12/15・負例誤反応1/15以下・内部エラー0を分ける。実表示・聴取・実キーも代替したとは扱わない。

CPUでの最初の比較であり新規クラウド課金は0。声の原音・教師重み・本体録音は公開しない。[生成契約](../../docs/neural-corpus.md)・[最終条件の契約](../../docs/final-corpus.md)・[利用条件](../../docs/data-license.md)。


## 生成完了と最初の2候補

[生成記録](development-generation-summary.json): 開発8条件/203試行で152元発話がASR一致、51試行は不一致。全656例の特徴量再構築を確認し、旧756発話と合わせ908発話＋400対照、計3124例を用意した。最終4条件は未生成。

[混合学習の比較](pooled-training-comparison.json): 600/1000epochsはともにepoch571・同じ重み/閾値652を選択。新domainの最悪F1はv3の0.522から0.807へ上がったが、誤反応最大0.12・操作recall最低0.733で未達。旧domainは基準を通過した。2候補は凍結・最終評価・実機適用しない。1000側の判定は、同一重み/閾値と入力hashを確認して600側を参照した。

## 次の開発候補

[条件別学習](../../docs/domain-training.md)でlegacy/neuralの学習重みを1/4にし、checkpointと閾値を旧/新×8profileで選ぶ。構造hidden88、seed42、1000epochs、前処理・ノイズ条件は維持する。繰返すtrain行を新しい発話数と数えず、独立した4final条件は未生成のまま維持する。これは開発データ内の次候補で、最終評価の成績から方針を変更したものではない。


## 条件別学習初版と次の選定

[1/4学習初版](domain-weighted-initial-result.json)はepoch381/threshold1188を選択。新domainのclean操作recallは1.0だが、誤反応最大0.08・earlyのF1=0.8659、旧domainのnoise F1=0.8791で未達だった。この候補も凍結・最終生成・実機適用しない。

次は同じデータ・1/4学習・hidden88/1000epochsのまま、raw順位上位と学習期間別の候補から最大16checkpointを取り、それぞれ校正した16groupの成績で選定する。閾値適用後に順位が逆転する反例をTDDで確認した変更で、合格基準や最終4条件は変更しない。
