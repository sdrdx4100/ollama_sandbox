import pytest
import yaml

from amtlab.config import PipelineConfig


def test_defaults_are_consistent():
    cfg = PipelineConfig()
    assert cfg.model.primary_target in cfg.model.targets
    assert set(cfg.calibration.weights) == {"jerk_rms", "shift_time_s", "clutch_energy_j"}
    assert cfg.calibration.mode in ("scalar", "pareto")


def test_shipped_config_loads():
    cfg = PipelineConfig.load("configs/default.yaml")
    assert cfg.dataset.n_samples > 0
    assert cfg.ollama.host.startswith("http")


def test_partial_override_keeps_other_defaults():
    cfg = PipelineConfig.from_dict({"dataset": {"n_samples": 10}})
    assert cfg.dataset.n_samples == 10
    assert cfg.dataset.seed == PipelineConfig().dataset.seed


def test_unknown_keys_are_rejected():
    with pytest.raises(KeyError):
        PipelineConfig.from_dict({"nope": 1})
    with pytest.raises(KeyError):
        PipelineConfig.from_dict({"dataset": {"nope": 1}})


def test_dump_roundtrip(tmp_path):
    cfg = PipelineConfig.from_dict({"output_dir": "x", "ollama": {"model": "llama3.1"}})
    path = cfg.dump(tmp_path / "c.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert PipelineConfig.from_dict(data).ollama.model == "llama3.1"
