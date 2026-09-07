"""Optuna による代理モデルのハイパーパラメータ探索。

目的関数は K-fold CV の RMSE(小さいほど良い)。モデル種別そのものも
探索対象に含め、AMT の変速品質という非線形・交互作用の強い応答に
最も合うモデルを選ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd

from .dataset import FEATURE_COLUMNS
from .modeling import SurrogateModel, cross_validate_model, fit_surrogate

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_model(trial: optuna.Trial, model_names: list[str]) -> tuple[str, dict[str, Any]]:
    name = trial.suggest_categorical("model", model_names)
    if name == "hist_gbr":
        params = {
            "learning_rate": trial.suggest_float("hgb_learning_rate", 0.01, 0.3, log=True),
            "max_iter": trial.suggest_int("hgb_max_iter", 100, 600, step=50),
            "max_leaf_nodes": trial.suggest_int("hgb_max_leaf_nodes", 8, 64, log=True),
            "min_samples_leaf": trial.suggest_int("hgb_min_samples_leaf", 5, 40),
            "l2_regularization": trial.suggest_float("hgb_l2", 1e-6, 1.0, log=True),
        }
    elif name in ("random_forest", "extra_trees"):
        params = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 24),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 12),
            "max_features": trial.suggest_float("rf_max_features", 0.3, 1.0),
        }
    else:  # ridge
        params = {}
    return name, params


@dataclass
class TuningOutcome:
    study: optuna.Study
    best_model_name: str
    best_params: dict[str, Any]
    best_rmse: float
    surrogate: SurrogateModel

    def history(self) -> pd.DataFrame:
        return study_to_frame(self.study)


def tune_surrogate(
    df: pd.DataFrame,
    target: str = "jerk_rms",
    n_trials: int = 40,
    features: list[str] | None = None,
    model_names: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    timeout: float | None = None,
) -> TuningOutcome:
    """代理モデルのハイパーパラメータを Optuna(TPE)で最適化する。"""

    features = features or FEATURE_COLUMNS
    model_names = model_names or ["hist_gbr", "random_forest", "extra_trees", "ridge"]

    def objective(trial: optuna.Trial) -> float:
        name, params = _suggest_model(trial, model_names)
        metrics, _ = cross_validate_model(
            df, target, model_name=name, features=features,
            n_splits=n_splits, seed=seed, **params
        )
        trial.set_user_attr("r2", metrics["r2"])
        trial.set_user_attr("model_name", name)
        trial.set_user_attr("model_params", params)
        return metrics["rmse"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=f"surrogate::{target}",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best = study.best_trial
    name = best.user_attrs["model_name"]
    params = best.user_attrs["model_params"]
    surrogate = fit_surrogate(
        df, target=target, model_name=name, features=features,
        n_splits=n_splits, seed=seed, **params
    )
    return TuningOutcome(
        study=study,
        best_model_name=name,
        best_params=params,
        best_rmse=float(best.value),
        surrogate=surrogate,
    )


def study_to_frame(study: optuna.Study) -> pd.DataFrame:
    """試行履歴を DataFrame 化(可視化用)。"""
    multi = len(study.directions) > 1
    rows = []
    for t in study.trials:
        values = t.values
        if not values:
            continue
        row: dict[str, Any] = {"trial": t.number, "state": t.state.name}
        if multi:
            for i, v in enumerate(values):
                row[f"objective_{i}"] = v
        else:
            row["objective"] = values[0]
        row.update({f"param_{k}": v for k, v in t.params.items()})
        row.update({k: v for k, v in t.user_attrs.items() if not isinstance(v, dict)})
        rows.append(row)
    df = pd.DataFrame(rows)
    if "objective" in df.columns:
        df["best_so_far"] = df["objective"].cummin()
    return df
