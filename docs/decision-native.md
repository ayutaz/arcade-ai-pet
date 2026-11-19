# 新しい音声候補のC照合

2026-11-19。**開発用2,680件の全判定がPython/Cで一致した。独立最終評価と実機の合格ではない。** 旧260入力・単一閾値のAPIと凍結対象13ファイルは変更していない。

## 形式と安全境界

| 対象 | 新しい契約 |
|---|---|
| 特徴 | HP160→平均除去→peak4096。100msの前後を保持する260特徴＋元の発話長1値 |
| モデル | 261→1〜90→7、int8重み/int32 bias/Q10、整数shift0〜16 |
| 棄却 | command0〜4の各閾値。unknown5/silence6は維持 |
| 数値容量 | 重み・bias・shift・5閾値の合計24KiB以下 |
| PCM | 16kHz/mono/int16/32,000samples。連続または既存8bank |
| Cのメモリ | trusted callerの整列済み領域。確保・I/O・可変globalなし |

`decision_assets.read_layers` はZIPの6メンバー、NPYの型/形/長さ/shift値を`np.load`前に検査する。旧260入力、過大配列、object/float、破損archiveとsymlink経路は拒否する。新規出力だけを作り、既存ファイルを更新しない。

`DecisionNativeModel` は数値からCヘッダーを再生成し、内容とhashを照合してから固定CソースだけをClangでコンパイルする。外部ヘッダーをそのままコンパイルしない。checksumだけを書き換えた改変も拒否する。C APIはtrustedな内部呼出し用で、WASM guestのポインターや同時呼出しを保護するAPIではない。

これは**数値assetの出力**で、正式freezeでも配布モデルでもない。生成モデルのライセンスはコードとは別で、旧HTSだけのCC BY表記を新しい混合データへ転用しない。

```python
from arcade_ai_pet.decision_assets import export_assets
from arcade_ai_pet.decision_native import DecisionNativeModel

# 自分の検証済み261入力モデル。fresh-outputは未作成の実パスを指定。
export_assets('checkpoints/candidate/weights.npz', [977, 2531, 1780, 181, 768], 'build/fresh-output')
with DecisionNativeModel('build/fresh-output') as model:
    values = model.extract(pcm, banked=True)
    label, logits = model.predict(values)
```

## TDDと回帰

8契約を先に追加し、未実装moduleのimportエラー2件でRed。その後に最小実装した。初回Green試行はmacOSの`/var` symlinkを含む一時パスで6エラーになり、fixtureの実パスを解決してからGreen8件/1.873秒。symlinkを拒否する契約は維持する。

```sh
uv run --locked python -m unittest discover -s tests -p 'test_decision_*.py'
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group train python -m unittest discover -s tests/training -v
```

新規契約は新旧形式の分離、容量、事前入力拒否、決定的出力、ヘッダー改変拒否、C特徴/準備の一致、分割bank、整数推論/同点/最大点差、null/不整列/寸法/NaN拒否を検査する。回帰206件/21.873秒・training15件/3.404秒・SDK2件/2.443秒がskipなしで成功。

## 選定済みモデルの照合

[クラス別棄却](class-rejection.md)で選定した重みと閾値を変えず、開発元WAV/制御音から全validation行を再構成した。各元発話ID・variant・domain/profile・ラベルの順序を保存済みlogit NPZへ照合している。

各行でPythonのHP/peak/261特徴/整数logit/棄却を再計算。Cの前処理、連続/分割特徴、Python特徴からの直接整数推論、C特徴からの最終判定を比較した。[匿名の全数値](../experiments/neural-domain-20261119/decision-native-result.json)に入力hashとC source hashを保存する。生成物や生音声は含めない。

| 検査 | 結果 |
|---|---|
| 前処理のPCM一致 | 2,680/2,680 |
| 連続/分割bankの特徴一致 | 2,680/2,680（bit単位） |
| 同じ特徴からの整数logit一致 | 2,680/2,680 |
| 保存済みPython logitの再現 | 2,680/2,680 |
| 全経路の最終判定一致 | 2,680/2,680 |
| Python/Cの最大特徴誤差 | 0.000002742 |
| C判定の品質 | 16/16 domain条件・8/8結合条件が既存基準に合格 |
| 特徴の作業領域 | 5,680B |
| 重み＋棄却表 | 23,992B |
| 照合時間 | 12.160秒、Mac arm64/Python3.12.13 |

浮動小数点の特徴値そのものがPythonとbit単位で一致したという結果ではない。同じ特徴を渡した整数logitは完全一致し、全経路の判定も一致した。時間と領域はホストの測定で、ESP32-S3の速度・総RAM・stackではない。

## 残る工程

固定処理と未使用の最終4条件へのbindingをTDDで実装し、候補と処理を固定した後に独立最終データを開く。最終結果を見た再校正はしない。合格時のみtarget build/資源確認/具体的なimageレビューへ進む。

実機は従来a553/16app、実声6/15のまま。新FWの適用、録音の外部取得、PSRAM/GPIO/security変更は行っていない。Vast.ai/API課金0米ドル。
