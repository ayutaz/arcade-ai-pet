# 開発データの結合

`mixed_corpus.merge_development(inputs, output)` は、元音源のID・ラベル・splitと音源hashを変えず、複数の開発データを結合する。既存のOpen JTalkデータと、新しい内容点検付きデータの併用を想定する。

- 1〜8個の実ディレクトリを受け取り、新規の出力先だけを作る。
- aligned-mfcc-v2と同じvariant数を要求する。明示されたfinal roleは拒否する。roleのない既存manifestは従来の開発データとして扱うため、呼出側で来歴を確認する。
- source IDの重複、異なるIDの同一発話WAV、音源hash不一致を拒否する。
- 各NPZと元音源の再構築を照合してから出力する。既存の元発話splitを維持し、variantの分離漏洩を防ぐ。
- componentごとの元manifest hash、voice/engineなどの来歴、licenseを保持する。複数の利用条件を単一ライセンスへ変換しない。

同じゼロPCMの対照例は話者独立性の根拠に数えない。最終評価用の条件は結合しない。従来のtest splitを学習・閾値選定へ使う機能ではない。

## TDD（2026-11-19）

先行テスト4件を追加し、未実装moduleによるRedを確認。最小実装後、fixtureのbetaにvalidation例がなかったため1件が既存データ契約で失敗した。fixtureを各splitのある60元発話へ修正し、Greenは4件/2.308秒。

契約はsplit/再構築/来歴の保持、final・重複・変更音源の拒否、variant不一致・既存出力の保護。コマンド: `uv run --locked python -m unittest discover -s tests -p test_mixed_corpus.py`。fixture修正を追加TDDとは数えない。合成データの結合テストであり、実声認識の改善を示すものではない。

通常回帰122件/19.097秒が成功（skipなし）。先行PRのtraining8件/SDK2件は成功済みで、本変更は結合データの契約を通常テストで検証する。
