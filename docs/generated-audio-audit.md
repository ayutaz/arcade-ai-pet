# 生成音声の内容点検

2026-11-19。生成要求と独立ASRの結果を照合するホスト用API。
**内容の一致だけを確認し、利用条件・データ分離・モデル精度の合格までは判定しない。**

## 契約

`arcade_ai_pet.generated_audio.audit_generated_wave(path, prompt=..., recognize=...)`を使う。
`recognize`はsample rateとmono PCM16の配列を受け、文字列を返す関数。
認識器へ正解のpromptやlabelを渡さない。学習対象のKWSを照合役にしない。

| 検査 | 動作 |
|---|---|
| 入力 | 既知の21生成prompt、0.1〜2秒、8〜96kHzのmono PCM16 |
| 由来 | 読み取ったWAVのSHA-256、rate、frames、peak、飽和数を記録 |
| 一致 | `content_check_passed=true`。promptに対応するlabelを維持 |
| 不一致/空文字 | false。不一致をASR側の別labelへ付け替えない |
| 認識器の失敗 | false。例外本文や不正な戻り値を保存しない |
| 無音/形式違反/2秒超 | ASRを呼ぶ前に拒否。切り捨てて通さない |

NFKC・カナ/かな・空白/句読点を整え、固定辞書の表記だけを対応させる。
例えば「待って」→「まって」、「遊ぼう/あそぼう」→操作「あそぼ」。
「まってない」や複数の操作語を含む文、似ているだけの表現は一致としない。
曖昧一致・部分一致・評価中のKWSを使った正解付けは行わない。

ASR一致でも音声の正確さが人に確認されたことにはならない。
TTS/モデルごとの利用条件と帰属、生成条件/元発話の分離、独立評価は別に確認する。
原音・本体録音・機器ID・絶対パスはAPIの結果に含めない。
呼出し側が音源の由来とASRの版/hashを別に管理する。

## TDD

新module未実装のModuleNotFoundErrorで先行Redを確認してから実装。
最初のGreenではmacOSの一時パス`/var`がsymlinkを含むfixtureで4 errorとなり、fixtureを実パスへ直した。実装のsymlink拒否は維持した。
次に、内容一致を学習利用の全面承認と誤解しない契約を先に追加し、旧キーのRedを確認してから結果名を`content_check_passed`へ変更した。

```sh
uv run --locked python -m unittest discover -s tests -p test_generated_audio.py
```

7件/0.005秒が成功。実ASRや音声モデルをダウンロードせず、模擬callbackで境界を検査する。
標準106件/10.063秒、CPU学習8件/9.464秒、実SDKの両VM2件/3.013秒も成功、skipなし。
FW・認識器の重み・閾値・app packageは変更しない。

追加生成では[短い音声のデコード制限とWAV再利用](generation-resume.md)を使う。保存した音声のhashを照合し、旧ASR判定を使い回さず再点検する。
