# 開発データの分割と条件別検証

2026-11-19、追加学習前の変更。生成途中の新しい音源では、hashによる従来の分割だけだと一部の操作語がvalidationに存在しなかった。旧データと合算した成績だけでは、この不足や新しい声の失敗が隠れる。

## 開発データの版

`development_split.prepare_development(base, output)` は明示的なdevelopment corpusだけを受け取り、音源は再生成せず、新しい開発データ版を作る。

- 操作5クラスと対象外クラスに、それぞれ3元発話以上を要求する。
- 各クラスを元ID順に並べ、validation/testへ各max(1, n//10)、残りをtrainへ割り当てる。予測結果を使って選ばない。
- 新IDのhashがそのsplitに入るnonceを決定的に探す。元のID/source/splitとWAV hashを保持し、同じ元発話のvariantを分けない。
- 新IDに対応するaugmentation seedで特徴量を再計算する。対照の無音/雑音は既存のID・PCM・splitを維持する。
- 全7クラスが各splitにあることと元音源の再構築を検査する。元のフォルダーは変更しない。

これは開発用の分割計画v2で、最初のraw開発datasetのtest所属も変更し得る。その旧testを独立評価とは主張しない。モデル学習前に変更理由を記録し、別に予約した4つのfinal条件は生成・分割変更をしない。最終評価の結果を見て分割を変えるためのAPIではない。

## 旧・新ドメインの別判定

`domain_validation.validate_domains(checkpoint, domains)` は同じ固定重み/閾値で、各development domainのvalidationを8profileずつ評価する。旧・新データを別々に判定し、各profileのF1≥0.90、誤反応≤0.05、操作語recall≥0.80をすべて満たした場合だけ`accepted_all_domains`とする。合算のsuccess flagは採用根拠にしない。

final role、領域間の元音源ID/WAV重複、変更重みは特徴量計算前に拒否する。学習・閾値再調整を行うAPIではない。旧Open JTalkのrole省略形式をdevelopmentとして扱うため、呼出側で元の開発datasetであることを確認する。

## TDD

| 対象 | 先行Red | Green |
|---|---|---|
| domain別判定 | 3契約を追加し、module未実装の読込エラー1件 | 3件/0.024秒 |
| 元発話単位の層化/保持/拒否 | 3契約を追加し、module未実装の読込エラー1件 | 3件/4.428秒 |

`uv run --locked python -m unittest discover -s tests -p test_domain_validation.py` と `-p test_development_split.py`。ホストの契約試験で、追加学習・実声改善・実機適用の結果ではない。旧v3の凍結対象13ファイルと重みは変更しない。

回帰: 通常131件/22.459秒、training8件/13.997秒、SDK2件/5.097秒が成功（skipなし）。
