# 固定log-melモデルの配布loader

2026-11-19。凍結済みモデルの数値やC runtimeは変更せず、配布時の読み込みを分ける。

## CIで検出した差

最初の公開候補PR66は、Linuxの全395件で1 errorとなった。`dscnn_freeze.verify`の再生成header照合で`frozen header changed`を検出。Macでは同じ395件が成功していた。失敗したCIはrun44540480602で、成功として扱わない。

数学tableはNumPy/CPUのsin/cos等からdouble値を生成するため、環境により末尾bitが変わり得る。固定済みheaderと異なる再生成tableを渡す再現テストでは、従来loaderが同じ理由で拒否する。元のfreezeを変更したり、近似一致を許可したりしない。

## 配布時の検査

`FrozenLogmelModel`はこの公開v1の固定freeze専用。一般の学習checkpointには既存のexport/検証手順を使う。

| 対象 | 検査 |
|---|---|
| decision-frozen.json | 公開した固定SHA `89e06e7e…`との完全一致 |
| 凍結時の52source | 元のsource hash一覧と完全一致 |
| 重み/metadata/plan | 固定freezeのhashと型/容量/plan契約を照合 |
| 学習した整数header | 重みから再生成し、固定hashと完全一致 |
| 数学table | 固定SHA `bd9ff5df…`と完全一致した配布byteを使用 |
| 実行 | 元のC frontendと整数CNNをコンパイル。sourceを前後に照合 |

数値table、重み、5閾値、52sourceは変更しない。既存の汎用loaderや凍結処理も変更しない。新しい配布loaderは別ファイルで、モデルの再学習・再校正・追加最終評価を行うものではない。Flash操作も含めない。

## TDDと範囲

先行5契約を追加し、module欠落のRedを確認（loader 1 error）。最小実装後5件/1.765秒でGreen。

- 数学tableの再生成差があっても、固定byteを使用すること
- 重み/header/metadata/planの改変やsymlinkをコンパイル前に拒否すること
- 凍結時sourceの相違を「移植の例外」にしないこと
- 無音と2種類の合成正弦波で両PCM layoutの特徴が一致し、整数logitが独立Pythonと一致すること
- 入力/52sourceを変更せず、終了後の再使用を拒否すること

```sh
uv run --locked python -m unittest discover -s tests -p test_logmel_release_loader.py -v
```

テストはローカル生成tableと固定tableのhashも出力する。新しいCIの各OSで差を確認するための診断で、同じhashになる環境を失敗にはしない。初回Linuxの直接原因をこのMac上の再現だけで断定しない。

ホストCの移植検査であり、実声の受入や実機のメモリ不足解消を保証しない。[実機の残作業](logmel-device-check.md)。

ローカル全400件/45.882秒、skipなしで成功。固定52sourceとモデル/数学header/metadata/plan/独立最終JSONのbyte不変も照合した。Linux/Mac CIの結果は後続で確認する。
