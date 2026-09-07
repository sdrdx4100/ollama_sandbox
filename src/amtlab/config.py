"""解析パイプラインの設定(YAML で外出し)。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .features import ObjectiveSpec


@dataclass
class DatasetConfig:
    n_samples: int = 600
    seed: int = 0
    n_jobs: int = -1


@dataclass
class ModelConfig:
    targets: list[str] = field(
        default_factory=lambda: ["shift_time_s", "speed_drop_kmh", "speed_loss_kmh", "jerk_rms"]
    )
    primary_target: str = "shift_time_s"
    n_trials: int = 40
    n_splits: int = 5
    model_names: list[str] = field(
        default_factory=lambda: ["hist_gbr", "random_forest", "extra_trees", "ridge"]
    )
    permutation_repeats: int = 10


@dataclass
class CalibrationConfig:
    n_trials: int = 150
    mode: str = "pareto"  # "scalar" or "pareto"
    seed: int = 0
    use_surrogate: bool = False
    worst_case_weight: float = 0.3
    #: 目的 KPI(順序は多目的最適化の軸順)と重み
    objectives: list[str] = field(default_factory=lambda: list(ObjectiveSpec().keys))
    weights: dict[str, float] = field(default_factory=lambda: dict(ObjectiveSpec().weights))

    def objective_spec(self) -> ObjectiveSpec:
        return ObjectiveSpec.from_config(keys=self.objectives, weights=self.weights)


@dataclass
class OllamaConfig:
    enabled: bool = True
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    temperature: float = 0.3
    timeout_s: float = 180.0
    language: str = "ja"


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)

    # ------------------------------------------------------------------
    @staticmethod
    def _build(cls: type, data: dict[str, Any]) -> Any:
        """辞書から入れ子のデータクラスを構築する(未知キーはエラー)。"""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"{cls.__name__} に未知の設定キー: {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items()})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        cfg = cls()
        for key, value in (data or {}).items():
            if not hasattr(cfg, key):
                raise KeyError(f"未知の設定キー: {key}")
            current = getattr(cfg, key)
            if is_dataclass(current) and isinstance(value, dict):
                setattr(cfg, key, cls._build(type(current), value))
            else:
                setattr(cfg, key, value)
        return cfg

    @classmethod
    def load(cls, path: str | Path | None) -> "PipelineConfig":
        if path is None:
            return cls()
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return path
