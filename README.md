# amtlab — AMT 車両データ解析・変速適合最適化基盤

ローカルの [Ollama](https://ollama.com/) を前提にした、**AMT(自動MT: 乾式単板クラッチ + 有段ギヤ)車両の
変速品質データ解析システム**です。計測データは外に出さず、手元で

```
プラントモデル/実車ログ  →  変速イベント切り出し  →  KPI 抽出
        →  scikit-learn 代理モデル  →  Optuna 感度解析・適合最適化
        →  seaborn 可視化  →  Ollama によるレポート生成
```

までを一気通貫で回します。

---

## なにが解けるのか

AMT は変速中にクラッチを切るため、**必ず駆動トルクが抜けます(トルクホール)**。
そこでこのリポジトリは次の 3 点にフォーカスして解析します。

1. **変速時間** — 指令からトルク復帰完了まで
2. **車速の落ち込み** — 変速前後で車速がどれだけ削られたか
3. **カレントギア(現在ギア)** — どの段の変速が一番効いているか

### 車速の落ち込みは 2 通りで測る

| KPI | 意味 | 使いどころ |
| --- | --- | --- |
| `speed_drop_kmh` | 波形のピーク → 谷の落ち込み幅 | 実測ログからそのまま読める量 |
| `speed_loss_kmh` | 「変速しなかった場合」との最大差 | 変速そのものが奪った車速 |

全開アップシフトのように**車速は落ちないが伸びが止まる**条件では前者が 0 になり、
惰行ダウンシフトでは前者に走行抵抗ぶんの減速が混ざります。両方を常に併記し、
**最適化の目的関数には `speed_loss_kmh`** を使います
(理由と実測値は [`docs/methodology.md` §3.2](docs/methodology.md) に記載)。

副作用の監視として、変速ショック `jerk_rms` / `jerk_peak`、トルク抜け時間、
クラッチすべり仕事 `clutch_energy_j` も毎試行記録します。

## クイックスタート

```bash
pip install -e .            # 依存: numpy / pandas / scipy / scikit-learn / optuna / seaborn / pyyaml

amtlab run --quick          # 小規模で一通り動かす(20 秒程度)
amtlab run --config configs/default.yaml   # 本番設定(数分)
```

`outputs/` に次の成果物が出ます。

```
outputs/
├── config.used.yaml         実行時の設定スナップショット
├── summary.json             LLM/レポートに渡した解析サマリ
├── report.md                Ollama(または内蔵テンプレート)による解析レポート
├── data/dataset.csv         DoE で生成した変速イベント(1 行 = 1 変速)
├── models/surrogate_*.joblib 学習済み代理モデル
├── tables/                  重要度・パレート解・改善率などの表
└── figures/                 seaborn による図一式
```

### サンプル結果(`configs/default.yaml`、600 イベント / 実行 92 秒)

Optuna(NSGA-II, 150 試行)による適合最適化のベースライン比:

| 指標 | ベースライン | 最適化後 | 変化 |
| --- | --- | --- | --- |
| `shift_time_s` [s] | 1.000 | 0.822 | **−17.8 %** |
| `speed_loss_kmh` [km/h] | 7.91 | 6.46 | **−18.3 %** |
| `speed_drop_kmh` [km/h] | 1.14 | 0.88 | −22.6 % |
| `torque_interrupt_s` [s] | 0.737 | 0.608 | −17.5 % |
| `jerk_peak` [m/s³] | 43.0 | 23.3 | −45.9 % |
| `clutch_energy_j` [J] | 12.8 | 34.6 | **+169 %**(速さの対価) |

代理モデルの CV 決定係数は `shift_time_s` **R²=0.92** / `speed_loss_kmh` **R²=0.94**。
**変速時間は運転条件だけでは R²=0.26 しか説明できず(適合値込みで 0.92)、
ほぼ適合値で決まる**ことが分かります。

<p align="center">
  <img src="docs/images/shift_trace.png" width="52%" alt="変速時系列">
  <img src="docs/images/pareto_front.png" width="46%" alt="パレートフロント">
</p>

左: 2→3 速アップシフトの時系列。車速パネルに**落ち込み幅(ピーク→谷)と
「変速しなかった場合」の基準線**を重ねています。
右: 変速時間 × 車速損失のパレートフロント(色はジャーク、★ が重み付き最良解)。

#### カレントギア別の傾向

![カレントギア別](docs/images/gear_analysis.png)

| 現在ギア | 変速時間 [s] | `speed_drop_kmh` | `speed_loss_kmh` |
| --- | --- | --- | --- |
| 1速 → 2速 | 1.35 | 1.45 | **8.81** |
| 2速 → 3速 | 1.12 | 1.05 | 5.06 |
| 3速 → 4速 | 1.00 | 1.31 | 2.84 |
| 4速 → 5速 | 0.84 | 1.32 | 1.89 |
| 5速 → 6速 | 0.87 | **2.08** | 1.36 |

2 つの指標は**ギヤに対して逆方向**に効きます。損失で見れば最悪は 1→2 速
(駆動力が大きいぶん奪われる車速も大きい)、実測の落ち込み幅で見れば高速段
(走行抵抗が大きく実際に減速する)。落ち込み幅だけを見ていると 1→2 速の問題を
見落とします。

![変速時間と車速の関係](docs/images/speed_time_relation.png)

変速時間の重要度 1 位は**カレントギア**(`from_gear`)、次いで適合値の
`sync_gain` と `clutch_close_rate` でした。

![感度解析](docs/images/importance.png)

### 主なコマンド

| コマンド | 用途 |
| --- | --- |
| `amtlab run` | 解析パイプライン一式 |
| `amtlab simulate --scenario 2-3@45,0.7` | 単一変速の時系列シミュレーションと波形描画 |
| `amtlab dataset --samples 800` | ランダム DoE データセット生成のみ |
| `amtlab analyze --dataset path.csv` | 既存データセットから解析のみ |
| `amtlab calibrate --trials 200 --mode pareto` | 適合値の多目的最適化のみ |
| `amtlab inspect --log log.csv` | ログの信号構成・サンプルレート診断(J1939 自動検出) |
| `amtlab ingest --log log.csv` | 実車ログから変速イベントを切り出して KPI 化 |
| `amtlab demo-log --j1939` | J1939 形式のデモログ生成 |
| `amtlab report --summary outputs/summary.json` | レポートだけ再生成(モデル差し替え検証に) |
| `amtlab ollama` | Ollama の疎通確認 |

## 実車ログを解析する(J1939 対応)

シミュレーションを使わず、**実測ログをそのまま**投入できます。
J1939(DBC デコード済み)なら列名を自動検出します。

```bash
amtlab inspect --log mylog.csv    # まず何が取れているか診断
amtlab ingest  --log mylog.csv    # 変速イベントを切り出して KPI 化
```

`inspect` は検出できた SPN・**実効サンプルレート**・注意点を出します。

```
  SPN  内部名                   列名                                   実効Hz  信号
   84  speed_kmh             CCVS1_WheelBasedVehicleSpeed         10.0  ホイールベース車速
  190  engine_speed_rpm      EEC1_EngineSpeed                     50.0  エンジン回転数
  191  output_shaft_rpm      ETC1_TransmissionOutputShaftSpeed    50.0  アウトプットシャフト回転数
  522  clutch_slip_pct       ETC1_PercentClutchSlip               50.0  クラッチ滑り率
  523  gear                  ETC2_TransmissionCurrentGear           離散  現在のギア位置
  574  shift_in_process      ETC1_TransmissionShiftInProcess        離散  トランスシフトインプロセス
```

### この信号セットで何が測れるか

| 指標 | J1939 のみ | 使う SPN |
| --- | --- | --- |
| 変速時間(トルク相 / 締結相に分解) | ◎ 実測 | 574 + 522 |
| 車速の落ち込み・車速損失 | ◎ 実測 | 84 / 904 |
| トルク抜け時間・トルク復帰時間 | ◎ 実測 | 513 × 544 × 526、512 |
| クラッチすべり時間 | ◎ 実測 | 522 |
| 吹け上がり | ◎ 実測 | 190 vs 191 × 526 |
| **ジャーク(乗り心地)** | **△ 参考値** | 車体前後加速度の追加計測が必要 |

**ジャークだけは J1939 では定量評価できません。** 車速(SPN 84)は 10 Hz 程度でしか
更新されず、階段状の信号を微分すると量子化が偽のジャークになるためです
(自動でローパスを実効レートの 1/3 まで下げて暴走は防ぎます)。
50 Hz で来るアウトプットシャフト回転(SPN 191)は**駆動側の速度**なので、
微分すると乗り心地ではなく駆動軸のねじり振動を測ることになり、代用になりません
(`driveline_speed_kmh` として振動観察用にだけ保持)。
各イベントに `jerk_quality`(`measured`/`derived`/`low_rate`/`unusable`)を記録します。

→ **変速時間・車速の落ち込み・トルク抜けにフォーカスするなら、追加計測なしで
そのまま回せます。**

### 変速イベントの切り出し

**SPN 574(シフトインプロセス)があれば推定不要**で、フラグの立ち上がりが変速開始、
立ち下がりがギヤ入り。締結完了はクラッチ滑り率(SPN 522)が 1% を切った時刻。
したがって変速時間が

```
変速時間 = gear_engage_time_s (トルク相 + ギヤ入りまで) + clutch_close_time_s (締結)
```

に分解され、遅れがどちらの相にあるか切り分けられます。
SPN 574 が無い場合はギヤ信号(523)の遷移から推定します(ニュートラル経由に対応)。

### 列名を明示する / 手を入れる

自動検出が外れる場合は明示できます(明示 > 自動検出)。

```bash
amtlab ingest --log mylog.csv --col-gear GearPos --col-speed Vsp
```

```python
from amtlab.ingest import SignalMap, analyze_log

trace, events = analyze_log("mylog.csv", SignalMap(gear="GearPos"))
print(events[["current_gear", "to_gear", "shift_time_s",
              "speed_drop_kmh", "speed_loss_kmh", "torque_interrupt_s"]])
```

出力はシミュレーションと**同じ KPI スキーマ**なので、`amtlab analyze` 以降
(代理モデル・カレントギア別集計・可視化)をそのまま適用できます。

### 手元にログが無いとき

J1939 形式のデモログ(信号名・単位・PGN ごとの更新周期・分解能を模擬)を作れます。

```bash
amtlab demo-log --j1939 --out demo.csv && amtlab inspect --log demo.csv
```

## Python API

```python
from amtlab import ShiftControlParams, ShiftScenario, simulate_shift, extract_kpis
from amtlab.calibration import calibrate, CalibrationSetting
from amtlab.dataset import build_dataset
from amtlab.features import ObjectiveSpec, gear_summary

# 1 回の変速をシミュレーション
result = simulate_shift(ShiftControlParams(), ShiftScenario(2, 3, speed_kmh=45, throttle=0.6))
kpis = extract_kpis(result)
print(kpis["shift_time_s"], kpis["speed_drop_kmh"], kpis["speed_loss_kmh"])

# カレントギア別の傾向
print(gear_summary(build_dataset(300)))

# 目的を「変速時間 + 車速損失」の 2 つに絞って多目的最適化
spec = ObjectiveSpec.from_config(weights={"shift_time_s": 0.5, "speed_loss_kmh": 0.5})
outcome = calibrate(CalibrationSetting(objectives=spec), n_trials=150, mode="pareto")
print(outcome.improvement())   # 目的に入れなかった KPI も併記される
print(outcome.best_control)
```

目的 KPI は YAML でも差し替えられます。

```yaml
calibration:
  objectives: [shift_time_s, speed_loss_kmh]
  weights: {shift_time_s: 0.5, speed_loss_kmh: 0.5}
```

## Ollama の設定

`configs/default.yaml` の `ollama` セクション、または環境変数 `OLLAMA_HOST` で指定します。

```bash
ollama serve
ollama pull qwen2.5:7b
amtlab ollama          # 疎通確認
```

Ollama に接続できない場合は、**内蔵テンプレートによる決定論的レポート**へ自動フォールバックするため、
CI や LLM 無し環境でもパイプラインは止まりません。

## 構成

| モジュール | 役割 |
| --- | --- |
| `amtlab.simulation.vehicle` | エンジン・変速機・駆動系・車体の物理パラメータ |
| `amtlab.simulation.controller` | 変速シーケンス(状態機械)と適合パラメータ 8 種 |
| `amtlab.simulation.plant` | 4 状態の数値積分(クラッチすべり/ロック、駆動軸ねじり) |
| `amtlab.dataset` | 実験計画(DoE)とバッチ実行 |
| `amtlab.features` | 変速品質 KPI の抽出・カレントギア別集計・目的関数 |
| `amtlab.j1939` | J1939 信号の定義・自動検出・物理量変換・レート診断 |
| `amtlab.ingest` | 実車ログの取り込み・イベント切り出し |
| `amtlab.modeling` | scikit-learn 代理モデル・感度解析 |
| `amtlab.tuning` | Optuna によるハイパーパラメータ探索 |
| `amtlab.calibration` | Optuna(TPE / NSGA-II)による適合最適化 |
| `amtlab.viz` | seaborn 可視化 |
| `amtlab.reporting` | Ollama クライアントとレポート生成 |
| `amtlab.pipeline` | 上記を束ねたパイプライン |

モデルの詳細(運動方程式・変速シーケンス・KPI 定義)は [`docs/methodology.md`](docs/methodology.md) を参照してください。

## 開発

```bash
pip install -e ".[dev]"
pytest -q          # 130 tests / 1 分程度
ruff check src tests
```

## ライセンス

MIT
