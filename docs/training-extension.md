# 評価データを保持した学習音声の追加

新しい合成音声をtrainだけに追加する。既存validation/testは元ファイルをそのままコピーし、レコードも変更しない。追加学習によって認識品質が改善するかは別の検証で判断する。

```python
from arcade_ai_pet.training_extension import append_qualified_training

append_qualified_training(
    neural_development, qualified_extension, extended_neural,
    reserved_plan=original_source_plan,
)
```

`uv run --locked python ...`で既存のNumPy2.5.5環境から実行する。生成/ASR側の別環境で特徴量を計算しない。

## 入力条件

| 入力 | 必須条件 |
|---|---|
| 元データ | role=development、既存plan hash、全元WAVと特徴の再構築が一致 |
| 追加音声 | development用の新しい条件、同じ確認済みprovider、全21promptの点検一致を含む |
| 未分割 | `generate_development(..., finalize=False)`の完了結果。特徴量ファイルを既に作った追加データは拒否 |
| 最終条件 | 元planの4条件等をそのまま保持。新しいdevelopment条件とID/captionを重ねない |
| 重複・改変 | 元/追加のID・WAV hash・成功audit・生成/点検コードのhashを照合。不一致は拒否 |

生成時の仮のhash所属は来歴に残し、特徴量を作る前に追加分だけtrainの新IDへ割り当てる。既存のvalidation/testや最終評価データをtrainへ移さない。追加の元WAVと失敗音源は変更しない。

## 出力と保持

元manifestのレコードを先頭にそのまま置き、追加分だけ後ろへ加える。追加レコードは元ID/source/splitを保持する。`validation.npz`/`test.npz`はbyte単位で同じ、`train.npz`だけが増える。旧/新の全評価条件のgateを別々に残し、古い失敗を平均で薄めない。

同じ出力先の再利用、final/incomplete入力、元の最終条件の変更、未確認provider、同一音声、変更WAVを拒否する。元ファイルのsymlinkと不正IDは出力作成/外部WAVの読込み前に拒否する。

## TDD（2026-11-19）

1. 新規4契約を先に追加し、未実装moduleのimportエラー1件でRed。初回Greenは12件だったが、fixture参照で既存8件を重複実行していた。参照をmodule経由へ直し、新規件数と区別した。
2. 元WAVのsymlink拒否が出力作成後になる反例を先に追加。出力が残る失敗1件のRed後、事前検査へ移して5件が成功。
3. 不正な元IDが外部WAVへ達する反例を先に追加。外部hash読込みを検出するAssertionErrorでRed。元ID/データの検証を先に行う修正後、6件/2.192秒が成功。

コマンドは`uv run --locked python -m unittest discover -s tests -p test_training_extension.py`。通常161件/17.731秒、training13件/10.224秒、SDK2件/3.214秒の回帰も成功。最後のID検査修正後の通常回帰は別途記録する。小fixtureのASRは模擬で、実際の発音や認識品質の保証ではない。

## 実データの計画

現在の新TTSのtrainは各指示5/6/4/6/6元発話。新しいdevelopment16条件を生成/ASR点検し、合格した元発話だけを追加する。既存の評価16条件・最終4caption・実声6/15の記録は維持する。まだ追加学習や実機での改善を確認した段階ではない。

ID検査修正後の通常回帰162件/22.659秒も成功。skipなし。lock・Markdown参照・diffと旧v3の凍結対象13ファイル不変を確認した。
