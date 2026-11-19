# 短い音声のASR制限と生成済みWAVの再利用

2秒以下の固定prompt音声を点検するため、ホストASRへ`max_new_tokens=32`を指定する。language=ja/beam5、全文の読み一致、未知語や不一致の不採用は変えず、正解文やhotwordも渡さない。

## 変更の根拠

16条件の追加生成中、短い音声へのtranscriptが最大223文字に達した。処理を止めた時点で45試行（33一致/12不一致）を保存している。観測timeoutだけで停止したものではなく、不要に長いデコードを抑えるための変更。

同じ保存WAVを新設定で再点検すると、「あそぼ」は一致のまま1.305秒、「ゲーム」の長いデコードは不一致のまま1.481秒だった。2件の追加検査であり、全音声での速度や品質の保証ではない。

## WAVの再利用

```python
from arcade_ai_pet.generation_cache import source_callback
from arcade_ai_pet.neural_corpus import generate_development

cached_synthesis, counts = source_callback(plan, prior_output, synthesize=teacher)
generate_development(
    plan, fresh_output, synthesize=cached_synthesis,
    recognize=bounded_blind_asr, finalize=False,
)
```

- 同じ固定planのdevelopment ID/prompt/caption/seedだけを受け付ける。
- progressに保存したWAV hashを使用時に再検査し、確認できたWAVだけを新しい出力先へコピーする。
- ASR判定は再利用せず、すべて点検し直す。最終条件や既に特徴量を作ったデータを入力にしない。
- metadataにない未完了WAVは再利用しない。元のファイル/失敗/進捗/ログを変更しない。
- 既存出力や変更WAVを検出しても、黙って生成し直して上書きしない。

元のprocessを停止したかどうかは実行側で確認する。ファイルのstatusだけをprocessの終了証拠としない。コピーしたWAVを別の元発話として二重計上しない。

## TDD（2026-11-19）

ASRのkwargs契約を先に変更し、max_new_tokens欠落の失敗1件でRed。最小実装後4件/0.007秒が成功。cacheの4契約は未実装moduleのimportエラー1件でRed、実装後4件/0.433秒が成功。

通常166件/19.196秒、training13件/10.445秒、SDK2件/3.162秒、skipなし。実ASRの2件は既存コードの追加照合で、TDD Redとは数えない。生成音声/元WAV/モデル重みはGitへ入れず、実機FWと既存v3は変更しない。
