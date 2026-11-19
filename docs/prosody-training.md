# 速度・音程変化を加える開発実験

学習用の合成音声だけに速度0.9/1.1倍の変化を加える。線形補間なので音程も変わり、音程を保つtime stretchではない。推論のadaptive frontendと既存v3の凍結対象13ファイルは変更しない。

## 採用条件

予備点検16件はASRとの一致14件だった。遅い「まって」は「待て」、「こんにちは」は「おんにちわ」になったため、加工音声を一律に正解ラベルで学習しない。読みの対応表や合格基準も変更しない。

| 段階 | 処理 |
|---|---|
| 準備 | `prosody_arrays.prepare_sources(dataset, output)`。developmentのtrain元音声だけを読み、加工WAVとplanを新規保存 |
| 点検 | `qualify_sources(output, recognize=callback)`。正解文を渡さないASRで全加工WAVを点検し、不一致を保存 |
| 学習 | `domain_training.train_candidate(..., neural_prosody=True, prosody_qualification=output)`。一致した加工WAVだけをneural trainへ追加 |

2秒を超える加工は切らずに除外する。元のtrainは維持し、validationは変更しない。元発話数は増えず、独立した新話者や実声の評価にはならない。

planは元manifest・元WAV・加工WAV・生成コードをhashで結ぶ。点検結果はplanと点検コードのhashを持ち、学習時に再照合する。加工し直さず、ASRが点検したWAVのPCMを読む。ASRエラーがある点検は未完了として拒否する。既存planと確定した点検結果への上書きも拒否する。

## 比較の範囲

最初の比較はhidden88、1000epochs、neural比重2/4の2構成。旧/新の各8条件と結合8条件の全てでF1≥0.90・誤反応≤0.05・操作recall≥0.80、数値重み24KiB以下を維持する。最終4条件はモデル凍結後まで未生成。受入に届かなければC検証・実機更新へ進めない。

合成音声・ASR一致・ホスト学習の成功を、実声や実機の改善とは扱わない。実声の最新結果は6/15で、実声の合格条件12/15は変更しない。

## TDD記録（2026-11-19）

1. rateの長さ/端点/切断拒否、train限定/validation不変、元WAV変更/final拒否の3契約を先に追加。未実装moduleでRed、最小実装後3件が成功。
2. neuralのみのdispatch契約を先に追加。未対応引数でRed、追加後既存を含む8件が成功。
3. ASR点検を必須にする契約を先に追加。6件中、未実装APIのエラー5件と未点検を拒否しない失敗1件でRed。dispatchは未対応`prosody_qualification`でエラー1件。
4. 最小実装後、点検を含む6件/0.501秒、dispatch8件/0.026秒が成功。元WAV/加工WAVの改変、内容不一致の除外、ASRエラー、上書き拒否を検査した。

コマンドはいずれも`uv run --locked --group train python -m unittest discover -s tests -p test_prosody_arrays.py`と`-p test_domain_training.py`。実際のASR一致率・学習精度は契約テストとは別に記録する。

回帰は通常149件/15.488秒、training9件/1.785秒、SDK2件が成功し、skipなし。`uv lock --check`と`git diff --check`も成功。SDKのメモリ上限拒否メッセージは、上限超過を拒否するfixtureの出力である。

実データの準備で48kHz元音声を16kHz限定として拒否した。WAV hashは一致し、元データの破損ではなかった。48kHz fixtureを先に追加し、同じ拒否でRedを確認。既存`read_pcm`と同じ16kHz変換を加工前に使う修正後、7件が成功した。加工・点検対象の出力WAVは16kHzのまま。失敗時は出力先を作成していない。

48kHz修正後の通常回帰は150件/15.515秒が成功。training/SDKは直前の同じ学習・VMコードで成功した結果と区別する。
