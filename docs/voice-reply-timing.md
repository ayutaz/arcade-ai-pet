# 音声分類後の返答開始タイミング

2026-11-19。固定log-mel FW6a1f/revision5では、最初の既知音源の32,000frame録音・特徴生成・あそぼの分類が成功した（result0/errors0）。返答再生は15,678frames中12,288で失敗し、全体の予備試験は未合格。HOME/音声解放/16app保持を確認した。

アプリ1.0.0は分類と返答開始を同じtick内で呼んでいた。音声serviceの時刻は外側のpollで更新されるため、同期分類の所要時間が返答の期限へ入る経路がある。1.0.1は分類結果を直ちに画面へ反映し、返答開始を次のtickへ送る。Bによる終了は待機中の返答より優先し、ボタン操作の返答は従来どおり開始する。

## 先行契約と検証

分類直後の返答回数0/次tickで1/重複なし/Bで取消、負例非反応、package version1.0.1を先に契約化。旧コードで同tickの返答回数1と旧versionが原因のRedを確認した。C4件/0.363秒、固定SDKのWasmtime/WAMRを含む2件/2.18秒でGreen。全通常392件/44.426秒がskipなし成功した。学習モデルの固定52sourceは全てbyte不変。

packageは5,176B/SHA-256 a6b624c9684642448f673bf782e2e777953b506c523e95d8a4f5376f281155dd。アプリ更新の実機適用/返答完走は未確認であり、FWを再書込みしたものではない。実機の更新は所有版/hash・他15app保持と具体的な許可を別に確認する。新たな学習/モデル/閾値変更は行わない。

## CIランナー

結果文書PR64のwasmはmacos-15でrunner未割当のまま15分後cancelled（実行stepなし）。PR65へ統合し、同じ必須job名/固定SDK/全2VM試験/15分上限を保って、標準ARM64のmacos-26へ移行する。先行contractで旧labelをRed→1件Green。無料の標準公開repo用labelで、有料large/self-hostedやCI迂回は使わない。[GitHub公式runner仕様](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)。実際のCI成功は別に確認する。
