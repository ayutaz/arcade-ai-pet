# 最終条件の生成

ここでは旧260入力のchannel freezeを扱う。261入力/5閾値には別の[decision freeze/最終評価API](decision-freeze.md)を使い、旧処理を拡張解釈しない。

`final_corpus.generate_final(plan, output, frozen=..., synthesize=..., recognize=...)` は、検証済みのモデル凍結後に予約済みfinal条件だけを生成する。development条件は生成しない。既存のchannel freeze検証でソース/重み/設定を照合し、凍結JSONと重みのhashを生成結果へ結び付ける。

音源生成と内容点検の後、特徴量はまだ作らない。本体側の固定uv環境から `finalize_final(output, frozen=...)` を呼び、同じモデルの凍結状態を再照合してから特徴量を作る。途中でthresholdなどの凍結JSONが変わった場合も拒否する。manifestのroleはfinalであり、開発データ結合には使用できない。

これはcaption条件の分離であって、異なる実在話者や実声の独立性ではない。ASRとの一致も、人が発音を確認した意味には使わない。最終結果を見て再調整した場合は、そのデータを新しい最終評価と呼ばない。

生成後の評価は既存 `channel_freeze.evaluate_channel_frozen` で、元音源ID/WAVの重複、固定threshold/前処理、モデルソースhashを検査する。旧Open JTalkの評価データを比較へ使った場合は回帰試験として区別する。

## TDD（2026-11-19）

モデル凍結後のfinal条件限定生成、生成前の変更重み拒否、特徴量計算前の変更凍結JSON拒否の3契約を追加。未実装moduleのImportErrorによる先行Red後、最小実装で3件/0.732秒が成功。通常回帰121件/15.459秒が成功（skipなし）。

コマンドは `uv run --locked python -m unittest discover -s tests -p test_final_corpus.py`、回帰は `uv run --locked python -m unittest discover -s tests`。実際の最終音源生成やモデル評価はこの契約試験に含まない。進行中の開発音源生成は先に固定したsource snapshotを使い、この変更から影響を受けない。
