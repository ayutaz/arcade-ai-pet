# CMN専用のC処理とモデル形式

2026-11-19。開発用2,680条件でPython/Cの最終判定がすべて一致した。[全数値](../experiments/neural-domain-20261119/utterance-cmn-native-result.json)。独立最終評価や実声の改善を示す結果ではない。

## 形式と処理

261入力でも旧context-durationとは意味が異なる。runtimeを`context-duration-cmn-class-v1`、frontendを`context-duration-cmn`、CMNを`post-bin-mean-20-clip16`として分離した。旧runtimeと相互に拒否し、Cヘッダーは検査済み整数から再生成・照合してからコンパイルする。

既存のC特徴処理の後に、20binの各MFCC係数の平均をdoubleで求めて除去し、±16に制限する。energy20値とduration1値は変更しない。作業領域を再利用し、CMNで新しいheap領域を確保しない。これはホストの処理とscratchの検査で、ESP32-S3の総RAM/stack実測ではない。

| 確認 | 結果 |
|---|---|
| 前処理PCM、連続/8分割bank特徴 | 各2,680/2,680一致 |
| 同一特徴の整数logit、保存済みPython logit | 各2,680/2,680一致 |
| 全経路の最終判定 | 2,680/2,680一致 |
| Python/Cの最大特徴誤差 | 0.000002355 |
| 開発品質 | 16domain条件＋結合8条件に合格 |
| 数値モデル＋5閾値 / scratch | 23,992B / 5,680B |
| Mac arm64照合時間 | 12.890秒 |

重み`0292ba74…`と5閾値2538/3401/326/2312/871を変更せず、既存2domainの元音声から再構築した。保存logit/ラベル/groupも全行照合し、元PCMを変更しない。特徴のbit単位のPython/C一致やターゲット速度を主張しない。

## TDDと検証

6契約を先行追加し、未実装`cmn_assets`のimportでRedを確認した。最初のGreen試行は5件成功、1件がテストの閾値list型でエラーとなり、Python判定が要求するint32配列へfixtureを直した。次のGreenは6件/2.054秒で成功。どちらのログも保持する。

```sh
uv run --locked --group train python -m unittest discover -s tests -p test_cmn_native.py
uv run --locked --group train python -m unittest discover -s tests -v
uv run --locked --group train python -m unittest discover -s tests/training -v
ARCADE_OS_ROOT=/absolute/path/to/pinned/scratch-arcade-os uv run --locked --group train python -m unittest discover -s tests/integration -v
```

回帰231件/28.483秒、training15件/3.828秒が成功。SDK試験は相対パスの`..`を実パス検査で拒否した初回失敗を保存し、clean/pinned SDKの絶対パスでは2件/2.437秒、skipなしで成功した。旧v3凍結13ファイルは不変。2,680条件の照合は追加検証で、新規TDDや独立最終評価には数えない。

後続の[固定/独立評価経路](utterance-cmn-freeze.md)もTDDで追加した。後続の正式freezeも成功した。以前の最終4条件は消費済みで、再校正や選定へ使わない。新最終評価・target build・実機は未完了。新規課金0米ドル。
