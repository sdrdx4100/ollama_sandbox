"""解析パイプライン全体のオーケストレーション。

    DoE シミュレーション → 代理モデル学習(Optuna+scikit-learn)
      → 感度解析 → 適合値の多目的最適化 → seaborn 可視化 → Ollama レポート

各ステップの成果物は ``output_dir`` 配下に保存され、単体でも再利用できる。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from . import viz
from .calibration import (
    CalibrationOutcome,
    CalibrationSetting,
    calibrate,
    compare_controls,
    default_scenarios,
)
from .config import PipelineConfig
from .dataset import build_dataset
from .features import gear_summary
from .modeling import (
    SurrogateModel,
    condition_only_baseline,
    fit_surrogate,
    importance_frame,
    partial_dependence_frame,
)
from .reporting import OllamaClient, build_summary, generate_report, write_report
from .simulation.controller import ShiftControlParams
from .simulation.plant import ShiftScenario, simulate_shift
from .tuning import tune_surrogate


@dataclass
class PipelineArtifacts:
    config: PipelineConfig
    output_dir: Path
    dataset: pd.DataFrame
    surrogates: dict[str, SurrogateModel]
    importance: pd.DataFrame
    calibration: CalibrationOutcome
    summary: dict[str, Any]
    report_path: Path
    report_source: str
    figures: list[Path] = field(default_factory=list)
    elapsed_s: float = 0.0


def _dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "data": root / "data",
        "models": root / "models",
        "tables": root / "tables",
        "figures": root / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def run_pipeline(
    config: PipelineConfig | None = None,
    dataset: pd.DataFrame | None = None,
    verbose: bool = True,
) -> PipelineArtifacts:
    """設定に従って解析パイプラインを最後まで実行する。"""

    cfg = config or PipelineConfig()
    t_start = time.time()
    out = _dirs(Path(cfg.output_dir))
    log = (lambda msg: print(f"[amtlab] {msg}", flush=True)) if verbose else (lambda msg: None)

    cfg.dump(out["root"] / "config.used.yaml")
    vehicle = cfg.vehicle_params()
    log(f"車両: {cfg.vehicle} ({vehicle.transmission.n_gears()} 速 / "
        f"{vehicle.total_mass:.0f} kg)")

    # 1) DoE データセット ------------------------------------------------
    if dataset is None:
        log(f"1/6 DoE シミュレーション ({cfg.dataset.n_samples} events) ...")
        dataset = build_dataset(
            n_samples=cfg.dataset.n_samples,
            seed=cfg.dataset.seed,
            n_jobs=cfg.dataset.n_jobs,
            vehicle=vehicle,
        )
    dataset.to_csv(out["data"] / "dataset.csv", index=False)
    log(f"    -> {len(dataset)} events / 完了率 {dataset['completed'].mean():.1%}")

    gears = gear_summary(dataset)
    gears.to_csv(out["tables"] / "gear_summary.csv", index=False)
    worst = gears.sort_values("speed_loss_kmh_mean", ascending=False).iloc[0]
    log(f"    -> カレントギア別で車速損失が最大なのは {int(worst['current_gear'])} 速 "
        f"{worst['direction']} (平均 {worst['speed_loss_kmh_mean']:.2f} km/h / "
        f"変速時間 {worst['shift_time_s_mean']:.2f} s)")

    # 2) 代理モデル(Optuna でハイパーパラメータ探索) ---------------------
    log(f"2/6 代理モデル学習 + Optuna 探索 ({cfg.model.n_trials} trials) ...")
    tuning = tune_surrogate(
        dataset,
        target=cfg.model.primary_target,
        n_trials=cfg.model.n_trials,
        model_names=cfg.model.model_names,
        n_splits=cfg.model.n_splits,
        seed=cfg.dataset.seed,
    )
    tuning.history().to_csv(out["tables"] / "tuning_history.csv", index=False)
    surrogates: dict[str, SurrogateModel] = {cfg.model.primary_target: tuning.surrogate}
    for target in cfg.model.targets:
        if target == cfg.model.primary_target:
            continue
        surrogates[target] = fit_surrogate(
            dataset, target=target, model_name=tuning.best_model_name,
            n_splits=cfg.model.n_splits, seed=cfg.dataset.seed, **tuning.best_params,
        )
    for name, model in surrogates.items():
        model.save(out["models"] / f"surrogate_{name}.joblib")
        log(f"    -> {name}: {model.model_name} R2={model.metrics['r2']:.3f} "
            f"RMSE={model.metrics['rmse']:.3f}")

    # 3) 感度解析 --------------------------------------------------------
    log("3/6 感度解析 (permutation importance / partial dependence) ...")
    importance = importance_frame(
        surrogates[cfg.model.primary_target], dataset,
        n_repeats=cfg.model.permutation_repeats, seed=cfg.dataset.seed,
    )
    importance.to_csv(out["tables"] / "importance.csv", index=False)
    pdp = partial_dependence_frame(surrogates[cfg.model.primary_target], dataset)
    pdp.to_csv(out["tables"] / "partial_dependence.csv", index=False)
    cond_only = condition_only_baseline(
        dataset, cfg.model.primary_target, n_splits=cfg.model.n_splits,
        seed=cfg.dataset.seed,
    )
    log(f"    -> 運転条件のみの R2={cond_only['r2']:.3f} / "
        f"適合値込み R2={surrogates[cfg.model.primary_target].metrics['r2']:.3f}")

    # 4) 適合値の最適化 --------------------------------------------------
    log(f"4/6 適合最適化 ({cfg.calibration.mode}, {cfg.calibration.n_trials} trials) ...")
    setting = CalibrationSetting(
        scenarios=default_scenarios(vehicle),
        vehicle=vehicle,
        objectives=cfg.calibration.objective_spec(),
        worst_case_weight=cfg.calibration.worst_case_weight,
    )
    outcome = calibrate(
        setting=setting,
        n_trials=cfg.calibration.n_trials,
        mode=cfg.calibration.mode,  # type: ignore[arg-type]
        seed=cfg.calibration.seed,
        surrogates=surrogates if cfg.calibration.use_surrogate else None,
    )
    outcome.trials.to_csv(out["tables"] / "calibration_trials.csv", index=False)
    if len(outcome.pareto):
        outcome.pareto.to_csv(out["tables"] / "pareto_front.csv", index=False)
    improvement = outcome.improvement()
    improvement.to_csv(out["tables"] / "improvement.csv", index=False)
    pd.DataFrame(
        [
            {"parameter": k, "baseline": outcome.baseline_control.as_dict()[k], "optimized": v}
            for k, v in outcome.best_control.as_dict().items()
        ]
    ).to_csv(out["tables"] / "calibration_params.csv", index=False)
    for _, row in improvement.iterrows():
        log(f"    -> {row['metric']:>16}: {row['baseline']:.3f} -> {row['optimized']:.3f} "
            f"({row['improvement_pct']:+.1f}%)")

    compare = compare_controls(
        {"baseline": outcome.baseline_control, "optimized": outcome.best_control}, setting
    )
    compare.to_csv(out["tables"] / "calibration_comparison.csv", index=False)

    # 5) 可視化 ----------------------------------------------------------
    log("5/6 seaborn による可視化 ...")
    figs = [
        viz.plot_kpi_distributions(dataset, out["figures"]),
        viz.plot_gear_analysis(dataset, out["figures"]),
        viz.plot_speed_time_relation(dataset, out["figures"]),
        viz.plot_kpi_correlation(dataset, out["figures"]),
        viz.plot_condition_map(dataset, out["figures"], kpi=cfg.model.primary_target),
        viz.plot_model_diagnostics(surrogates[cfg.model.primary_target], out["figures"]),
        viz.plot_importance(importance, out["figures"]),
        viz.plot_partial_dependence(pdp, out["figures"]),
        viz.plot_calibration_comparison(compare, out["figures"]),
    ]
    if "objective" in outcome.trials.columns:
        figs.append(viz.plot_optuna_history(outcome.trials, out["figures"]))
    if len(outcome.pareto):
        figs.append(
            viz.plot_pareto(outcome.pareto, out["figures"], keys=setting.objectives.keys)
        )

    demo_scenario = setting.scenarios[1]
    traces = {
        label: simulate_shift(control, demo_scenario, vehicle=vehicle,
                              settings=setting.sim)
        for label, control in (
            ("baseline", outcome.baseline_control),
            ("optimized", outcome.best_control),
        )
    }
    figs.append(viz.plot_shift_trace(traces["baseline"], out["figures"], "trace_baseline"))
    figs.append(viz.plot_shift_trace(traces["optimized"], out["figures"], "trace_optimized"))
    figs.append(viz.plot_trace_comparison(traces, out["figures"]))
    log(f"    -> {len(figs)} 図を出力")

    # 6) レポート --------------------------------------------------------
    log("6/6 レポート生成 ...")
    summary = build_summary(
        dataset, surrogates, importance, outcome, gears=gears,
        tuning={
            "best_model": tuning.best_model_name,
            "best_rmse": round(tuning.best_rmse, 4),
            "n_trials": len(tuning.study.trials),
            "condition_only_r2": round(cond_only["r2"], 4),
        },
    )
    summary["objectives"] = {
        "keys": list(setting.objectives.keys),
        "weights": dict(setting.objectives.weights),
    }
    (out["root"] / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    client = OllamaClient(
        host=cfg.ollama.host, model=cfg.ollama.model,
        temperature=cfg.ollama.temperature, timeout_s=cfg.ollama.timeout_s,
    )
    text, source = generate_report(
        summary, client=client, enabled=cfg.ollama.enabled, language=cfg.ollama.language
    )
    report_path = write_report(text, out["root"] / "report.md")
    log(f"    -> {report_path} (source: {source})")

    elapsed = time.time() - t_start
    log(f"完了: {elapsed:.1f}s / 成果物 {out['root'].resolve()}")
    return PipelineArtifacts(
        config=cfg,
        output_dir=out["root"],
        dataset=dataset,
        surrogates=surrogates,
        importance=importance,
        calibration=outcome,
        summary=summary,
        report_path=report_path,
        report_source=source,
        figures=figs,
        elapsed_s=elapsed,
    )


def quick_scenario(text: str) -> ShiftScenario:
    """``2-3@45,0.6`` 形式の文字列を ShiftScenario に変換する。"""
    gears, _, rest = text.partition("@")
    g_from, _, g_to = gears.partition("-")
    parts = [p for p in rest.split(",") if p]
    speed = float(parts[0]) if parts else 45.0
    throttle = float(parts[1]) if len(parts) > 1 else 0.6
    grade = float(parts[2]) if len(parts) > 2 else 0.0
    payload = float(parts[3]) if len(parts) > 3 else 0.0
    return ShiftScenario(int(g_from), int(g_to), speed, throttle, grade, payload)


def load_control(path: str | Path | None) -> ShiftControlParams:
    """JSON/YAML の適合値ファイルを読み込む(未指定ならベースライン)。"""
    if path is None:
        return ShiftControlParams()
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):  # calibration_params.csv 相当の形式
        data = {row["parameter"]: row["optimized"] for row in data}
    return ShiftControlParams.from_dict(data)
