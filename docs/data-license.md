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


## 新しい合成音声の開発データ

Irodoriの参照録音を使わないVoiceDesign出力を、新しい小型認識器の開発データとして検証する。2026-11-19時点ではデータ生成の工程で、旧v3モデルの置換や新しい実声合格を意味しない。重み・生成WAVは本変更の配布物へ含めない。

| 対象 | 固定資料 | 確認内容 |
|---|---|---|
| Irodori-TTS v2 VoiceDesign / Chihiro Arata | [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v2-VoiceDesign/tree/456e55708e7183f5c7faa1448209d54aa8991451) | MIT表記。実在人物への無断の意図的な声の模倣、誤認・偽情報を目的とする生成への追加制限も記載 |
| Semantic-DACVAE Japanese / Aratako | [model card](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim/tree/47376ee24834d7a05a48ebabfe3cde29b3c5e214) | MIT表記 |
| faster-whisper small / Systran | [model card](https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120) | MIT表記。音源点検のホスト推論だけに使用 |

[取得したmodel cardのhash](neural-provider-model-cards.json)を保持する。Irodoriの条件は一般的な成人の声質・高さ・速さだけとし、人物名・他人の参照録音は使わない。固定した短い操作語と対象外の語を生成し、点検では合成音声であることを保持する。

planの`license: MIT`は確認したモデルの識別情報であり、出力・混合データ・派生モデル全体の利用許諾を自動承認する意味ではない。Open JTalk音声との混合では元のCC BY 3.0の帰属と、各componentの来歴を維持する。生成元の表示や派生成果物の条件を公開前に照合し、本リポジトリのApache-2.0だけへ置き換えない。
