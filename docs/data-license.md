# 合成音声と学習成果物の条件

2026-11-17。実験にはOpen JTalkと同梱の6種類のHTS voiceを使用。
本体録音・個人の声・非公開の純正素材は使用していない。

| 対象 | 出所・条件 | 取扱い |
|---|---|---|
| Open JTalk | [公式](https://open-jtalk.sourceforge.net/)、Modified BSD | ホストの音声合成。コードをvendorしていない |
| HTS Voice Mei 1.5、5感情 | MMDAgent Project Team、©2009–2018 Nagoya Institute of Technology, Department of Computer Science、CC BY 3.0 | 生成音声と派生素材に帰属を表示 |
| NIT ATR503 M001 | HTS Working Group、©2003–2012 Nagoya Institute of Technology, Department of Computer Science / ©2003–2008 Tokyo Institute of Technology, Interdisciplinary Graduate School of Science and Engineering、CC BY 3.0 | 生成音声と派生素材に帰属を表示 |
| 自作実装 | 本repo、Apache-2.0 | 上流voiceの条件と分ける |

[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)の条件を維持し、
生成、加工、学習、量子化したことをモデルカードへ記載する。
学習重みも保守的にCC BY 3.0として帰属を添えて扱い、コードのApache-2.0へ一括で変更しない。
声の提供者や機関からの推薦・保証を主張しない。

今回参照した実ファイルはHomebrew open-jtalk/1.11の
`voice/mei/README.txt`と`voice/m100/COPYING`。
[Homebrewの公開式](https://formulae.brew.sh/formula/open-jtalk)もBSD-3-Clause AND CC-BY-3.0を記載する。
Homebrewのpackage名は1.11だが、取得した実行ファイルの`-h`表示はOpen JTalk Version 1.10。
manifestには実行ファイル・辞書・各voiceのSHA-256を残し、版名だけで同一性を判断しない。

学習途中の合成WAV/重みはローカルのGit対象外。公開候補には帰属、変更内容、モデルカード、
評価範囲と照合済みhashを揃える。原本voice重みの再配布は本実験に不要。

その後、`models/kws-v2`へ自作の小型KWS重み・C表現・照合用の派生特徴を追加した。
原本TTS重みや音声PCMは含めず、同ディレクトリのmodel-card/LICENSEに帰属・変更内容・評価範囲を記録する。

`assets/replies-v1`はM001だけを使う事前生成の短い返答。CC BY 3.0の
帰属/変更内容/hashを添え、PCM8の数値C表を配布する。実機録音ではない。
`models/behavior-v1`は自作盤面のBFS模倣学習で、外部音声由来ではなくApache-2.0。
