"""scikit-learn による変速品質の代理モデル(サロゲート)構築と評価。

適合値 + 運転条件 -> 変速品質 KPI の写像を学習する。
学習済みモデルは
  * 感度解析(permutation importance / 部分依存)
  * Optuna 適合探索の高速な代理評価
の両方に使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import CONDITION_FEATURES, CONTROL_FEATURES, FEATURE_COLUMNS

CATEGORICAL_FEATURES = ["from_gear", "to_gear"]

MODEL_FACTORY: dict[str, Any] = {
    "hist_gbr": lambda **kw: HistGradientBoostingRegressor(random_state=0, **kw),
    "random_forest": lambda **kw: RandomForestRegressor(random_state=0, n_jobs=-1, **kw),
    "extra_trees": lambda **kw: ExtraTreesRegressor(random_state=0, n_jobs=-1, **kw),
    "ridge": lambda **kw: RidgeCV(**kw),
}


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    cats = [c for c in CATEGORICAL_FEATURES if c in features]
    nums = [c for c in features if c not in cats]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
            ("num", StandardScaler(), nums),
        ],
        remainder="drop",
    )


def build_pipeline(
    model_name: str = "hist_gbr",
    features: list[str] | None = None,
    **model_kwargs: Any,
) -> Pipeline:
    """前処理 + 回帰器のパイプラインを構築する。"""
    if model_name not in MODEL_FACTORY:
        raise KeyError(f"未知のモデル: {model_name} (選択肢: {sorted(MODEL_FACTORY)})")
    features = features or FEATURE_COLUMNS
    return Pipeline(
        [
            ("prep", make_preprocessor(features)),
            ("model", MODEL_FACTORY[model_name](**model_kwargs)),
        ]
    )


@dataclass
class SurrogateModel:
    """学習済み代理モデルと評価結果。"""

    pipeline: Pipeline
    target: str
    features: list[str]
    metrics: dict[str, float]
    cv_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_name: str = "hist_gbr"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[self.features])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "SurrogateModel":
        return joblib.load(Path(path))


def cross_validate_model(
    df: pd.DataFrame,
    target: str,
    model_name: str = "hist_gbr",
    features: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    **model_kwargs: Any,
) -> tuple[dict[str, float], pd.DataFrame]:
    """K-fold CV で汎化性能を評価し、指標と OOF 予測を返す。"""

    features = features or FEATURE_COLUMNS
    X, y = df[features], df[target].to_numpy()
    pipe = build_pipeline(model_name, features, **model_kwargs)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)
    metrics = {
        "rmse": float(root_mean_squared_error(y, oof)),
        "mae": float(mean_absolute_error(y, oof)),
        "r2": float(r2_score(y, oof)),
        "target_std": float(np.std(y)),
    }
    preds = pd.DataFrame({"actual": y, "predicted": oof})
    preds["residual"] = preds["predicted"] - preds["actual"]
    for col in ("is_upshift", "from_gear", "to_gear", "throttle"):
        if col in df.columns:
            preds[col] = df[col].to_numpy()
    return metrics, preds


def fit_surrogate(
    df: pd.DataFrame,
    target: str = "jerk_rms",
    model_name: str = "hist_gbr",
    features: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    **model_kwargs: Any,
) -> SurrogateModel:
    """CV 評価 + 全データ再学習で代理モデルを作る。"""

    features = features or FEATURE_COLUMNS
    metrics, preds = cross_validate_model(
        df, target, model_name, features, n_splits, seed, **model_kwargs
    )
    pipe = build_pipeline(model_name, features, **model_kwargs)
    pipe.fit(df[features], df[target])
    return SurrogateModel(
        pipeline=pipe,
        target=target,
        features=features,
        metrics=metrics,
        cv_predictions=preds,
        model_name=model_name,
    )


def importance_frame(
    surrogate: SurrogateModel,
    df: pd.DataFrame,
    n_repeats: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """permutation importance を DataFrame で返す(適合値/条件を区別)。"""

    result = permutation_importance(
        surrogate.pipeline,
        df[surrogate.features],
        df[surrogate.target],
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
        scoring="r2",
    )
    # 図中の凡例に使うため英語表記(フォント依存を避ける)
    kind = {
        f: ("calibration" if f in CONTROL_FEATURES else "condition")
        for f in surrogate.features
    }
    out = pd.DataFrame(
        {
            "feature": surrogate.features,
            "importance": result.importances_mean,
            "std": result.importances_std,
            "kind": [kind[f] for f in surrogate.features],
            "target": surrogate.target,
        }
    )
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def partial_dependence_frame(
    surrogate: SurrogateModel,
    df: pd.DataFrame,
    features: list[str] | None = None,
    grid_resolution: int = 25,
) -> pd.DataFrame:
    """主要な適合値の部分依存(PDP)を長形式 DataFrame で返す。"""

    features = features or CONTROL_FEATURES
    rows = []
    for feat in features:
        disp = PartialDependenceDisplay.from_estimator(
            surrogate.pipeline,
            df[surrogate.features],
            [feat],
            grid_resolution=grid_resolution,
            kind="average",
        )
        pd_result = disp.pd_results[0]
        grid = np.asarray(pd_result["grid_values"][0])
        avg = np.asarray(pd_result["average"]).ravel()
        rows.append(
            pd.DataFrame(
                {
                    "feature": feat,
                    "value": grid,
                    "partial_dependence": avg,
                    "target": surrogate.target,
                }
            )
        )
        import matplotlib.pyplot as plt

        plt.close("all")
    return pd.concat(rows, ignore_index=True)


def condition_only_baseline(df: pd.DataFrame, target: str, **kw: Any) -> dict[str, float]:
    """運転条件のみで説明した場合のベースライン性能(適合値の寄与を測る)。"""
    metrics, _ = cross_validate_model(df, target, features=CONDITION_FEATURES, **kw)
    return metrics
