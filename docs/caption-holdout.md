# 声条件を保留する開発用検証

2026-11-19。[CMN候補の独立評価不合格](utterance-cmn-final.md)を受け、開発時に未使用captionへの汎化を検査する。**処理の契約と実データの分離は確認済み。後続の固定1構成も学習したが、品質未達で不採用。** [事前条件](../experiments/neural-domain-20261119/caption-holdout-plan.json)・[配列の確認結果](../experiments/neural-domain-20261119/caption-holdout-arrays-result.json)。

## データの分離

元の24caption中、train専用16条件からseed2026111904のhash順でextension-13/14/03/02を保留する。train由来の無音20元レコードも保留し、その元発話と全variant/profileをtrainから除外する。元のvalidation/test archiveは書き換えない。過去の最終評価2組は使用しない。

| 新しい配列 | 使用目的 | 実データの行数 |
|---|---|---:|
| train | 保留条件/無音を除外したneural学習 | 17,332 |
| validation | 元のneural検証を追加回帰として維持 | 520 |
| held_validation | 保留captionと無音。学習用のnoise/hum増量なし | 2,128 |

保留は82元音声＋20無音。trainに保留元IDが存在せず、3配列の元ID集合は互いに重ならない。旧6archiveのhashが不変で、元の検証520行のlogitも保存済み結果と一致した。このlogit再現は旧モデル/元の検証だけの後方確認で、旧モデルを新保留データで評価していない。確認時間25.574秒（Mac arm64）。

captionは生成時の条件で、実在する話者IDではない。条件を分けたことだけで未知の人への精度を保証しない。

## 選定と必須回帰

`caption_training.train_candidate`は既存の学習器を使い、旧legacy検証＋保留neural検証でcheckpointと5閾値を選ぶ。元のneural検証は選定に使わず、追加の必須回帰にする。全24domain/profile、全体の結合8profile、選定2domainの結合8profileで既存のF1/誤反応/recall基準を要求する。

```python
from arcade_ai_pet.caption_training import train_candidate

result = train_candidate(
    {'legacy': '/real/legacy-data', 'neural': '/real/neural-data'},
    '/fresh/output', holdout_seed=2026111904,
    epochs=1000, hidden=88, neural_weight=2, max_passes=4)
```

パスは手元の開発データと未使用の保存先へ置き換える。関数はfinal-role、元発話/WAVの重複、test配列、保留元IDの混入を拒否する。入力やsourceが途中で変わると候補を保存しない。品質未達の候補は結果とともに保存するが、合格flagを付けない。前処理/モデル構造/学習seedはCMN候補と共通で、今回の比較は事前固定した1構成のみ。

## TDD

配列生成の7契約を先行追加し、module未実装のimportでRedを確認した。最小実装後7件/2.323秒が成功。途中のmanifest置換で古い分割を使う問題も先行テストで再現（ValueError未発生のRed）し、同じbytesからのparse/hashと選択fingerprint照合で修正。8件/2.223秒が成功した。

学習処理も7契約を先行追加し、module未実装のRed後に実装。7件/0.121秒が成功した。選定への入力は実際のfit引数で検査し、元neural検証だけが失敗すると全体も不合格になることを確認する。既存の学習器をmockした契約試験で、新モデルの精度確認ではない。

```sh
uv run --locked --group train python -m unittest discover -s tests -p 'test_caption_*.py'
uv run --locked --group train python -m unittest discover -s tests -v
```

旧v3凍結13ファイルは変更しない。次は固定sourceによる1構成の学習/校正/追加回帰。独立最終評価・C再照合・S3 build・実機の認識改善を、この処理のテスト成功へ置き換えない。実機a553/実声6/15と最終不合格を維持し、新FW適用には別の個別許可が必要。

回帰252件/31.293秒、training15件/4.073秒、pinned SDK2件/3.267秒がskipなしで成功した。uv lock・文書参照・公開対象とsecret scanも提出前に確認する。コードの契約成功と候補品質の合格は区別する。

## 固定1構成の結果

[全数値](../experiments/neural-domain-20261119/caption-holdout-training-result.json)。固定source443f8df、同じCMN/hidden88/seed42/epochs1000/比重2で1構成だけを学習・校正した。108.091秒、選定epoch976。学習62,622行、選定用4,288行、元neuralの追加回帰520行で評価した。

| domain | 最小F1 | 最大誤反応率 | 最小操作recall |
|---|---:|---:|---:|
| legacy | 0.890414 | 0.018018 | 0.854167 |
| 保留neural | 0.886260 | 0.029126 | 0.800000 |
| 元neural | 0.828352 | 0.120000 | 0.866667 |

既存基準F1≥0.90/誤反応≤0.05/recall≥0.80を維持し、不採用とした。選定に使う2domainも一部未達で、追加回帰の元neuralでは全8profileの誤反応率が超過した。保留側だけに合わせる調整は行わない。

閾値は2336/675/0/872/3791、数値モデル＋表23,992B、重みSHAは`a34c2900431957ee853970df1d5978b7a1bd7f84038b63fe2255109c63849e4b`。失敗も出力に保存し、実行driverは品質gateのexit1で終了した。実装のテスト失敗や処理中断ではない。

56組の指標を混同行列から別に再計算し、保留/元neuralの16組も保存配列から再現した。元8入力ファイル（manifest含む）とsource23ファイルは不変。これは開発用の比較で、独立最終評価ではない。

新しい最終データの生成・C/固定処理・S3 build・実機適用へ進めない。次は20binのMFCCを前提とする既存方式に対し、より細かい時間/周波数の情報を扱う特徴と分類器のホストPoCを検討する。過去の最終2組、実声6/15、課金0米ドルを維持する。
