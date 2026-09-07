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
適合(キャリブレーション)エンジニアは次の 3 つの相反する要求のバランスを取ります。

| 目的 | KPI | 小さくしたい理由 |
| --- | --- | --- |
| 変速ショック | `jerk_rms` / `jerk_peak` [m/s³] | 乗り心地・車両振動 |
| 変速の速さ | `shift_time_s` [s] | もたつき感・トルク抜け時間 |
| クラッチ耐久 | `clutch_energy_j` [J] | 発熱・摩耗 |

本リポジトリは、この 3 目的のトレードオフを **データで可視化し、Optuna で最適な適合値を探索**します。

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

### サンプル結果(`configs/default.yaml`、600 イベント / 実行 90 秒)

| 指標 | ベースライン適合 | Optuna 最適化後 | 変化 |
| --- | --- | --- | --- |
| `jerk_rms` [m/s³] | 8.83 | 6.21 | **−29.6 %** |
| `jerk_peak`(最悪条件)[m/s³] | 79.4 | 23.6 | **−70.3 %** |
| `shift_time_s` [s] | 1.000 | 0.948 | −5.2 % |
| `clutch_energy_j` [J] | 12.8 | 18.2 | +42 %(ショック低減の対価) |

代理モデル(`jerk_rms`)の CV 決定係数は **R² = 0.83**。
運転条件だけで説明した場合は R² = 0.51 なので、**残りは適合値で決まっている**ことが分かります。

<p align="center">
  <img src="docs/images/shift_trace.png" width="52%" alt="変速時系列">
  <img src="docs/images/pareto_front.png" width="46%" alt="パレートフロント">
</p>

左: 2→3 速アップシフトの時系列(フェーズを色分け。トルクホールと再締結後のシャッフル振動が見える)。
右: 変速ショック・変速時間・クラッチ仕事のパレートフロント(★ が重み付き最良解)。

![感度解析](docs/images/importance.png)

`throttle` と `speed_kmh`(運転条件)に次いで、`sync_slip_offset` や `torque_reduce_rate` といった
**適合値がジャークを支配している**ことが permutation importance から読み取れます。

### 主なコマンド

| コマンド | 用途 |
| --- | --- |
| `amtlab run` | 解析パイプライン一式 |
| `amtlab simulate --scenario 2-3@45,0.7` | 単一変速の時系列シミュレーションと波形描画 |
| `amtlab dataset --samples 800` | ランダム DoE データセット生成のみ |
| `amtlab analyze --dataset path.csv` | 既存データセットから解析のみ |
| `amtlab calibrate --trials 200 --mode pareto` | 適合値の多目的最適化のみ |
| `amtlab demo-log` / `amtlab ingest --log log.csv` | 実車ログ形式の取り込み・イベント切り出し |
| `amtlab report --summary outputs/summary.json` | レポートだけ再生成(モデル差し替え検証に) |
| `amtlab ollama` | Ollama の疎通確認 |

## 実車ログを解析する

シミュレーションを使わず、**実測ログをそのまま**投入できます。必要な信号は
`時刻 / エンジン回転 / 車速 / 選択ギヤ` の 4 本(あればスロットル・駆動軸トルクも使用)。

```bash
amtlab ingest --log mylog.csv \
  --col-time t --col-engine-speed Ne --col-speed Vsp --col-gear GearPos
```

ギヤ信号の遷移(ニュートラル経由を含む)から変速イベントを自動抽出し、
シミュレーションと**同じ KPI スキーマ**の表を作るので、`amtlab analyze` 以降の
解析コードをそのまま流用できます。

```python
from amtlab.ingest import SignalMap, analyze_log

trace, events = analyze_log("mylog.csv", SignalMap(gear="GearPos"))
print(events[["from_gear", "to_gear", "shift_time_s", "jerk_rms", "engine_flare_rpm"]])
```

## Python API

```python
from amtlab import ShiftControlParams, ShiftScenario, simulate_shift, extract_kpis
from amtlab.calibration import calibrate, CalibrationSetting

# 1 回の変速をシミュレーション
result = simulate_shift(ShiftControlParams(), ShiftScenario(2, 3, speed_kmh=45, throttle=0.6))
print(extract_kpis(result))

# 代表条件セットに対して適合値を多目的最適化(NSGA-II)
outcome = calibrate(CalibrationSetting(), n_trials=150, mode="pareto")
print(outcome.improvement())
print(outcome.best_control)
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
| `amtlab.features` | 変速品質 KPI の抽出と目的関数 |
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
pytest -q          # 85 tests / 1 分程度
```

## ライセンス

MIT
