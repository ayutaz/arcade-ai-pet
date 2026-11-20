# ソースリリース

2026-11-20。ユーザーの「100%にしなくていい、ここから出す」により、v0.1.0を**実験版のSource pre-release**として公開する。既知の音声4/5・Wi-Fi後のPet起動失敗・実声未確認を[リリースノート](release-notes/v0.1.0.md)へ記載する。安定版の受入を達成したとは扱わず、追加FW適用/再学習は行わない。

2026-11-18に作成したsource配布の安全条件は維持する。

## 配布物

- `<project>-<version>-source.tar.gz`: Git追跡source、LICENSE、文書/公開デモ素材。
- `SHA256SUMS`: archiveのSHA-256。

firmwareやpaired app package、本体録音、鍵、build/cacheは含めない。
公開済みの小型モデルNPZ・CC BY 3.0の帰属/ライセンス・評価JSONもarchiveへ含める。候補のsource公開と正式Releaseを区別し、モデルの再学習や実機の自動更新は行わない。

## 公開前の条件

| 条件 | 検査 |
|---|---|
| 版 | `vMAJOR.MINOR.PATCH`とpyproject.tomlが一致 |
| source | cleanなcheckout、タグがHEADを指し、origin/mainの履歴に含まれる |
| 試験 | タグ上で通常の全CIを再実行。失敗/skipで公開へ進めない |
| 配布境界 | 追跡済み秘密/生成物パス、firmware/録音/鍵、symlink、認証情報形式を拒否 |
| 出力 | dist内の新規実ディレクトリのみ。既存archiveやReleaseを上書きしない |

archiveは同じcommitでtimestampを固定して再生成できる。バイナリのビルド再現性とは別の検査。

## タグ作成後の処理

[Source release workflow](../.github/workflows/release.yml)が[通常CI](../.github/workflows/tests.yml)を呼び、全jobの成功後だけpublishへ進む。
publish jobだけにcontents:writeを与え、checkoutの資格情報を保存しない。Actionsとuvは既存CIと同じ固定版。
Source archiveとchecksumだけを`gh release create --verify-tag --prerelease --notes-file`で登録する。タグごとの明示ノートを必須にし、自動生成ノートだけでは公開しない。タグ/既存Releaseは削除・上書きしない。

ローカルで同じarchiveを作る場合は、CI成功後に通常mergeしたmainの対象タグをcheckoutしたclean環境で実行する。

```sh
uv sync --locked
uv run --locked python tools/source_release.py --tag v0.1.0
```

このコマンド自体はGitHubへ送信せず、実機にも接続しない。タグ未作成/不一致なら停止する。
タグpushは公開操作。今回の実験版公開はユーザーが明示的に指示した範囲で行う。

## TDDと範囲

新tool未作成のFileNotFoundErrorで先行Redを確認後、最小実装。
5件の契約試験は一時Git repoを実際に作り、main/tag/clean、同一archiveとSHA、秘密パス/symlink/出力境界、CI公開順序を検査する。
初回のGreen試行はtarfileがrootディレクトリ名の末尾slashを除くことをfixtureが考慮せず1失敗。
archiveのsource範囲を変えず期待値を修正し、5件すべて成功した。これは製品のRedへ数えない。

```sh
uv run --locked python -m unittest discover -s tests -p test_source_release.py -v
```

ローカルarchive生成はfixtureで検証。実Releaseへのアップロードは未実行で、workflowの実発火は正式タグの公開時に検証する。

最終回帰: 標準99件/10.567秒、実SDK統合2件/1.964秒（実SDK両VM）が成功（skipなし）。uv lock・actionlint1.7.12・差分検査も成功。Actionsの実発火や本体の動作を保証する結果とは区別する。
学習系の回帰も8件/1.930秒で成功。今回の変更でモデルを再学習していない。
単一backslashのファイル名も拒否する既存動作を追加照合した。これは先行Redを伴う新実装ではない。

## 2026-11-19の公開準備

固定log-mel候補の数値・数学table・freeze/独立最終評価・帰属条件をsourceへ収録する。先行2契約はファイル欠落でRed、追加後2件/1.270秒でGreen。全通常395件/44.197秒が成功した。上流TTS重み・生成音声・本体PCM・FW/app binaryは含めない。

[現在の実機試験](logmel-device-check.md)は合成5語/返答まで成功したが、Desk併用の修正と実声/実物受入が未完了。これは11-19時点の準備記録。11-20の実験版公開方針は冒頭に記載した。


## 実験版の公開契約

先行テストで、publishコマンドにpre-release指定と版別ノートがないことをRedとして確認。指定を追加後、既存のsource境界/タグ/再現性を含む6件がGreen。モデル/閾値/凍結sourceは変更していない。全CI成功後の通常merge→mainのタグ→タグCI→source公開→取得後のSHA/byte照合の順で進める。

今回の検証: `uv run --locked python -m unittest discover -s tests -p test_source_release.py -v` は6件/2.358秒Green（先行Redは6件中1失敗）。通常全401件/47.784秒も成功、skipなし。actionlint、source境界423ファイル、文書リンク269件、gitleaks/差分検査が成功。公開CIとRelease後の取得照合は別に記録する。
