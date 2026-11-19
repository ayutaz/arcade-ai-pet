# 261入力モデルの固定と独立最終評価

実候補の固定後に実施した[予約4条件の最終評価](reserved-final-evaluation.md)は不合格だった。以下はAPIと契約テストの説明。開発用条件の合格を汎化性能へ拡大しない。

2026-11-19。新しい[context-duration/C実装](decision-native.md)専用の固定処理。旧260入力のfreeze/最終評価APIとは混ぜない。以下の契約テスト成功は、実モデルの最終評価や実機の合格ではない。

## 固定するもの

`decision_freeze.freeze_candidate(weights, thresholds, domains, plan, output)` は、2〜4個の開発domainを検査し、元音声からvalidationを再構成する。

| 対象 | 検査 |
|---|---|
| データ | development限定、元発話ID/WAV重複なし、予約final条件の混入なし |
| モデル | 261入力/整数重み、5閾値、前処理/C実装のsource hash |
| 品質 | 各domainの8条件と結合8条件すべてでF1≥0.90、誤反応≤0.05、操作recall≥0.80 |
| C照合 | 直接整数logit/判定、全経路判定、連続/分割bank一致、特徴誤差<0.001 |
| 資源 | 数値モデル＋5閾値≤24KiB、特徴scratch≤24KiB |
| 最終計画 | 予約4条件・seed・attempts・provider等を含む計画全体のhash |

品質やC一致の成功flagを入力として受け取らない。検査中に重み/source/manifestが変わった場合も拒否する。不合格なら要求された出力先は作らない。

出力は数値assets、`decision-frozen.json`と`final-plan.json`。`verify`で数値・コード・計画・ヘッダーを再照合する。数値export側の`frozen:false`はexport単体がfreezeを保証しない意味で、正式な固定記録は別ファイル。

生成モデルのライセンスはコードと別。既存HTSの帰属条件とニューラルproviderのモデルカード/利用条件を、各入力データの来歴から引き継ぐ。新しい重みや音声の公開許可を与えるAPIではない。

## 最終生成と評価

`decision_final.generate_final` は、検証済みfreezeと**完全に同じ計画のfinal4条件だけ**を生成する。source生成/独立ASR点検と、core環境での特徴化を分ける。

```python
from arcade_ai_pet.decision_freeze import freeze_candidate
from arcade_ai_pet.decision_final import generate_final, finalize_final, evaluate_final

freeze_candidate(weights, thresholds, {'legacy': old_data, 'neural': new_data}, plan, frozen)
generate_final(plan, final_data, frozen=frozen, synthesize=teacher, recognize=independent_asr)
# 固定したcoreのuv環境から実行。teacher環境で特徴量を作らない。
finalize_final(final_data, frozen=frozen)
result = evaluate_final(frozen, final_data, report_directory)
```

生成結果には重み・freeze JSON・計画のhashを結び付ける。特徴化前に同じbindingと4条件の存在を確認する。評価時もbinding、provider/sourceのhash、元発話ID、condition/label/promptを含むsource形式、WAV重複を検査する。変更/再ラベル/重複はPCM読込み前に拒否する。

最終評価は全新規元発話/加工版を使う。8条件の品質とC照合を再検査し、未達なら未達の結果を保存する。校正/学習は呼ばない。評価中のfreeze/manifest変更は、結果の出力前に拒否する。

同じproviderによる未使用captionの評価であり、未知の実在話者や実声の評価ではない。最終結果を見た後で再調整したモデルを、その最終データで独立評価したとは扱わない。

## TDDと検証範囲

基本9契約を先行追加し、未実装module importのRed2件後に最小実装した。最初の試行では模擬音源の重複による1回の正当な再試行を「84呼出し固定」としたfixtureが失敗したため、事前上限168内と特徴化時の追加生成なしを検査する形へ修正した。importした既存TestCaseの重複実行も除外した。

追加の整合性2契約では、再ラベル等がPCMへ到達する4つのRedを確認。さらに評価中のmanifest変更が拒否されないRedを確認してから、来歴/ID/条件と開始終了hashの照合を追加した。最新11件/2.931秒が成功。

```sh
uv run --locked python -m unittest discover -s tests -p 'test_decision_f*.py'
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group train python -m unittest discover -s tests/training -v
```

回帰217件/25.268秒・training15件/3.530秒・SDK2件/2.346秒、skipなし。旧凍結対象13ファイルは不変。小さな数値fixtureと模擬音源であり、実モデルのfreeze・最終生成・品質・target buildは別の実行記録へ分離する。
