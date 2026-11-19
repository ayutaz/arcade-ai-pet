# ニューラル合成音声の開発データ

開発用の音源を生成し、独立ASRと内容が一致した発話だけを特徴量へ変換する。実声の正解や実機性能を保証する処理ではない。既存v3のデータ・モデル・凍結対象ファイルは変更しない。

## 生成条件の分離

`neural_corpus.validate_plan` はエンジンcommit、モデル/codec/実行環境lock/ASRモデルのSHA-256、seed、試行数、特徴量variant数、条件を固定する。条件はdevelopmentとfinalに分け、同じcaptionの共有を拒否する。

`generate_development` はdevelopment条件だけを生成する。発話ごとのseed/元音源ID/hash/splitを保存し、ASRが異なる内容を返した音源は棄却する。期待文をASRへ渡さず、合成の再試行は最大3回。インフラ例外では停止し、途中の記録を残す。棄却音源を別ラベルへ付け替えない。

これで話者の独立性を証明したことにはならない。同じモデルからの異なるcaptionという条件分離であり、finalの生成・モデル凍結後評価は別工程。

## 2段階の実行

1. TTS/ASR用の隔離uv環境から `generate_development(plan, output, synthesize=..., recognize=..., finalize=False)` を呼ぶ。
2. 本プロジェクトのロック済みuv環境から `finalize_development(output)` を呼ぶ。

前半はWAVと点検記録、`qualified-sources.json`を作る。後半は元音源と点検結果のhash/ID/ラベル/条件を再照合し、既存のaligned-mfcc-v2形式でtrain/validation/testのNPZとmanifestを作る。特徴量計算時のNumPy版をmanifestへ記録する。元音源ごとのsplitを維持し、異なるvariantを別splitへ移さない。

TTS用依存が本体側のNumPy版と異なっても、特徴量計算は本体側の固定環境で行える。生成済みの音源は再合成しない。既存NPZ/manifestの上書きを拒否する。部分失敗は新しい出力先で扱う。

## ホスト用アダプター

`host_teacher.source_callbacks` は明示的にロード済みのTTS/ASRを受け取る。モデルのダウンロード・自動ロードは行わない。TTS要求は参照音声なし、2秒以内、40steps。有限値のmono音声だけを新規PCM16 WAVへ書く。ASRには16kHzの実音源と日本語設定を渡し、正解文やhotwordを渡さない。

推奨モデルの互換性や利用許諾を自動判定するAPIではない。プロバイダーはライセンスと実際のasset hashを別途確認して注入する。WAV/重み/生成データをこのPRへ含めない。

## TDD記録（2026-11-19）

| 対象 | Red | Green |
|---|---|---|
| 開発条件だけの生成、棄却/再試行/途中停止、既存形式との一致 | 5契約を追加、module未実装で読込エラー1件 | 5件成功 |
| TTS/ASRの境界、prompt非提供、波形制約 | 4契約を追加、module未実装で読込エラー1件 | 4件成功 |
| 生成と特徴計算の分離、変更音源の拒否 | `finalize`未対応による2件のエラー | 対象7件成功 |
| 点検記録の欠落/不合格/hash変更の拒否 | ValueError未発生、3 subtest失敗 | 対象8件成功 |

コマンドは `uv run --locked python -m unittest discover -s tests -p test_neural_corpus.py` と `-p test_host_teacher.py`。実音声の品質試験はこの契約テストに含まない。

回帰: 通常118件、training8件、SDK integration2件が成功（skipなし）。ホスト上の検証で、実機の認識改善は未確認。
