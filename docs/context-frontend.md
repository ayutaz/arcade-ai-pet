# 語頭の文脈を残すホストPoC

**Python上の実験。対応するC frontend・freeze/export・実機版はまだない。** 既存v3の凍結対象13ファイル、C実装、重みは変更しない。contextのモデルを旧exportへ渡すことは拒否する。

## 調査の根拠

[固定4構成](../experiments/neural-domain-20261119/bounded-grid-results.json)も不採用だったため、developmentの失敗例を調べた。「かわいい」1元発話の全3variantを「おいで」とする誤反応と、earlyの「おやすみ」「まって」の失敗が繰り返された。

同じ固定Whisperで元音源・処理後全区間・特徴量に使った区間・前後100ms付き区間を点検した。[ASRと範囲/hash](../experiments/neural-domain-20261119/window-asr-analysis.json)。正解文はASRへ与えず、すべてpeak0.8へ正規化して比較した。入力は合成音声だけで、本体PCMではない。

| 発話 | 元音声 | 処理後全区間 | 現在の区間 | 前後100ms |
|---|---|---|---|---|
| かわいい | 一致 | 一致 | 「さわいい」で不一致 | 一致 |
| おやすみ、early | 一致 | 不一致 | 空文字 | 不一致 |
| まって、early | 一致 | 一致 | 一致 | 一致 |

この観測は区間の影響を疑う根拠で、単独のASRを人の正解とは扱わない。「おやすみ」は意図した150msの切断stressで元音声から変わっており、文脈を足しても復元したとは言えない。元の失敗と全8条件の合格基準は維持する。

## PoCの契約

`context_frontend.active_span` は既存の活性範囲の前後10frame（各100ms）を追加し、記録の0〜198frame内へ収める。FFT/Mel/DCT、HP160、peak4096、260特徴量、7クラスは変えない。

`context_arrays.make_arrays` は同じ元音源/ID/splitからcontext特徴を作る。trainだけに指定したノイズ複製を行い、validationは増やさない。test PCMは基礎データの整合検査に読まれ得るが、context学習配列には入れない。

`domain_training.train_candidate(..., frontend='context')` で明示的に選び、featureのhashをメタデータへ保存する。既定adaptiveの動作は維持する。旧`freeze_channel`はcontextを拒否し、別の対応C実装なしで実機へ流れない。

最初の比較は同じ908元発話・比重1/4・hidden88・1000epochs・校正候補最大16件。旧/新×8条件と結合8条件をすべて検査し、F1≥0.90・誤反応≤0.05・操作recall≥0.80を下げない。最終4条件は未生成のまま。

## TDD（2026-11-19）

- frontend3契約、配列1契約を先に追加。各module未実装の読込エラー1件でRed。
- context dispatchと旧freeze拒否の契約を追加。未対応`frontend`引数のTypeError1件でRed。
- 最小実装後、frontend3件/0.121秒、配列1件/0.649秒、dispatchを含む7件/0.056秒が成功。
- 弱い語頭と強い後半音を持つfixtureでは、旧特徴量が語頭の有無で同じになり、context特徴量は変化することを確認した。特定の人の発音を保証するテストではない。
- 回帰は通常142件/15.655秒、training9件/9.617秒、SDK2件/3.285秒、skipなし。

実モデルの改善、C/Python一致、S3 build、実機の認識・表示・聴取はこの契約テストに含まない。
