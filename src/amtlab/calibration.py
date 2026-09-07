"""Optuna による AMT 変速制御の適合(キャリブレーション)最適化。

2 つのモードを持つ:
  * ``scalar``  : 重み付きスカラー目的を TPE で最小化(適合値を 1 組決める)
  * ``pareto``  : (ジャーク, 変速時間, クラッチ仕事) を NSGA-II で多目的最適化

評価は「代表運転条件セット」に対する平均 + 最悪値で行う。実際の適合と
同様に、1 組の適合値が全条件で破綻しないことを要求する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

from .features import (
    DEFAULT_WEIGHTS,
    OBJECTIVE_KPIS,
    OBJECTIVE_SCALES,
    extract_kpis,
    scalar_objective,
)
from .modeling import SurrogateModel
from .simulation.controller import CONTROL_BOUNDS, ShiftControlParams
from .simulation.plant import ShiftScenario, SimSettings, simulate_shift
from .simulation.vehicle import VehicleParams

optuna.logging.set_verbosity(optuna.logging.WARNING)


def default_scenarios() -> list[ShiftScenario]:
    """適合の代表条件(発進直後の低速段〜高速段、上り勾配、ダウンシフト)。"""
    return [
        ShiftScenario(1, 2, speed_kmh=18.0, throttle=0.85),
        ShiftScenario(2, 3, speed_kmh=42.0, throttle=0.60),
        ShiftScenario(3, 4, speed_kmh=70.0, throttle=0.45),
        ShiftScenario(4, 5, speed_kmh=95.0, throttle=0.35),
        ShiftScenario(2, 3, speed_kmh=38.0, throttle=0.75, grade_pct=6.0, payload_kg=300.0),
        ShiftScenario(4, 3, speed_kmh=55.0, throttle=0.40),
        ShiftScenario(5, 4, speed_kmh=80.0, throttle=0.30),
    ]


@dataclass
class CalibrationSetting:
    scenarios: list[ShiftScenario] = field(default_factory=default_scenarios)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    sim: SimSettings = field(default_factory=SimSettings)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    worst_case_weight: float = 0.3  # 平均に対する最悪条件のブレンド率
    n_jobs: int = -1


def _kpis_for(control: ShiftControlParams, scenario: ShiftScenario,
              setting: CalibrationSetting) -> dict[str, float]:
    result = simulate_shift(control, scenario, vehicle=setting.vehicle, settings=setting.sim)
    kpis = extract_kpis(result)
    kpis.update(scenario.as_dict())
    return kpis


def evaluate_control(
    control: ShiftControlParams, setting: CalibrationSetting | None = None
) -> pd.DataFrame:
    """代表条件すべてに対する KPI を DataFrame で返す。"""
    setting = setting or CalibrationSetting()
    rows = Parallel(n_jobs=setting.n_jobs, prefer="processes")(
        delayed(_kpis_for)(control, s, setting) for s in setting.scenarios
    )
    df = pd.DataFrame(rows)
    for k, v in control.as_dict().items():
        df[k] = v
    return df


def aggregate(df: pd.DataFrame, setting: CalibrationSetting) -> dict[str, float]:
    """条件横断の集約指標(平均と最悪値のブレンド)。"""
    agg: dict[str, float] = {}
    for key in OBJECTIVE_KPIS:
        mean = float(df[key].mean())
        worst = float(df[key].max())
        agg[key] = (1.0 - setting.worst_case_weight) * mean + setting.worst_case_weight * worst
        agg[f"{key}_mean"] = mean
        agg[f"{key}_worst"] = worst
    agg["completed_ratio"] = float(df["completed"].mean())
    agg["jerk_peak_worst"] = float(df["jerk_peak"].max())
    agg["score"] = scalar_objective(
        {**{k: agg[k] for k in OBJECTIVE_KPIS}, "completed": agg["completed_ratio"] >= 1.0},
        setting.weights,
    )
    return agg


def suggest_control(trial: optuna.Trial) -> ShiftControlParams:
    """探索空間から適合値をサンプリングする。"""
    return ShiftControlParams(
        **{name: trial.suggest_float(name, lo, hi) for name, (lo, hi) in CONTROL_BOUNDS.items()}
    )


def _surrogate_predict(
    surrogates: dict[str, SurrogateModel],
    control: ShiftControlParams,
    setting: CalibrationSetting,
) -> dict[str, float]:
    """代理モデルで KPI を高速予測する(シミュレーション不要)。"""
    rows = []
    for sc in setting.scenarios:
        row = {**sc.as_dict(), **control.as_dict()}
        rows.append(row)
    X = pd.DataFrame(rows)
    out: dict[str, float] = {}
    for key, model in surrogates.items():
        pred = model.predict(X)
        out[f"{key}_pred_mean"] = float(np.mean(pred))
        out[f"{key}_pred_worst"] = float(np.max(pred))
        out[key] = (1.0 - setting.worst_case_weight) * float(np.mean(pred)) + (
            setting.worst_case_weight * float(np.max(pred))
        )
    out["completed_ratio"] = 1.0
    return out


@dataclass
class CalibrationOutcome:
    study: optuna.Study
    mode: str
    best_control: ShiftControlParams
    best_metrics: dict[str, float]
    baseline_control: ShiftControlParams
    baseline_metrics: dict[str, float]
    trials: pd.DataFrame
    pareto: pd.DataFrame = field(default_factory=pd.DataFrame)

    def improvement(self) -> pd.DataFrame:
        """ベースライン比の改善率(%)。"""
        rows = []
        for key in (*OBJECTIVE_KPIS, "score", "jerk_peak_worst"):
            base = self.baseline_metrics.get(key, np.nan)
            best = self.best_metrics.get(key, np.nan)
            rows.append(
                {
                    "metric": key,
                    "baseline": base,
                    "optimized": best,
                    "improvement_pct": (base - best) / base * 100.0 if base else np.nan,
                }
            )
        return pd.DataFrame(rows)


def calibrate(
    setting: CalibrationSetting | None = None,
    n_trials: int = 120,
    mode: Literal["scalar", "pareto"] = "scalar",
    seed: int = 0,
    baseline: ShiftControlParams | None = None,
    surrogates: dict[str, SurrogateModel] | None = None,
    timeout: float | None = None,
) -> CalibrationOutcome:
    """変速制御の適合値を Optuna で最適化する。

    Parameters
    ----------
    surrogates:
        与えると KPI をシミュレーションではなく代理モデルで評価する(高速)。
        最終的な best/baseline 指標は必ずシミュレーションで再評価する。
    """

    setting = setting or CalibrationSetting()
    baseline = baseline or ShiftControlParams()

    def metrics_for(control: ShiftControlParams) -> dict[str, float]:
        if surrogates:
            return _surrogate_predict(surrogates, control, setting)
        return aggregate(evaluate_control(control, setting), setting)

    if mode == "scalar":
        def objective(trial: optuna.Trial) -> float:
            control = suggest_control(trial)
            m = metrics_for(control)
            for k, v in m.items():
                trial.set_user_attr(k, float(v))
            if surrogates:
                return scalar_objective(
                    {**{k: m[k] for k in OBJECTIVE_KPIS}, "completed": 1.0}, setting.weights
                )
            return m["score"]

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=20),
            study_name="amt-calibration-scalar",
        )
    else:
        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            control = suggest_control(trial)
            m = metrics_for(control)
            for k, v in m.items():
                trial.set_user_attr(k, float(v))
            return tuple(float(m[k]) for k in OBJECTIVE_KPIS)

        study = optuna.create_study(
            directions=["minimize"] * len(OBJECTIVE_KPIS),
            sampler=optuna.samplers.NSGAIISampler(seed=seed, population_size=min(40, n_trials)),
            study_name="amt-calibration-pareto",
        )

    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    pareto = pd.DataFrame()
    if mode == "scalar":
        best_control = ShiftControlParams.from_dict(study.best_trial.params)
    else:
        pareto = pareto_frame(study, setting.weights)
        best_row = pareto.sort_values("score").iloc[0]
        best_control = ShiftControlParams.from_dict(best_row.to_dict())

    # 最終評価は常にシミュレーションで実施(代理モデルの誤差を持ち込まない)
    best_metrics = aggregate(evaluate_control(best_control, setting), setting)
    baseline_metrics = aggregate(evaluate_control(baseline, setting), setting)

    from .tuning import study_to_frame

    return CalibrationOutcome(
        study=study,
        mode=mode,
        best_control=best_control,
        best_metrics=best_metrics,
        baseline_control=baseline,
        baseline_metrics=baseline_metrics,
        trials=study_to_frame(study),
        pareto=pareto,
    )


def pareto_frame(study: optuna.Study, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """多目的探索のパレート解を DataFrame 化する。"""
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for t in study.best_trials:
        row = {"trial": t.number}
        row.update({k: v for k, v in zip(OBJECTIVE_KPIS, t.values)})
        row.update(t.params)
        row["score"] = sum(
            weights.get(k, 0.0) * v / OBJECTIVE_SCALES.get(k, 1.0)
            for k, v in zip(OBJECTIVE_KPIS, t.values)
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def compare_controls(
    controls: dict[str, ShiftControlParams], setting: CalibrationSetting | None = None
) -> pd.DataFrame:
    """複数の適合値セットを同一条件で比較する(seaborn 可視化用の長形式)。"""
    setting = setting or CalibrationSetting()
    frames = []
    for label, control in controls.items():
        df = evaluate_control(control, setting)
        df["calibration"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
