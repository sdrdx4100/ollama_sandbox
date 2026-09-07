import json

from amtlab.config import PipelineConfig
from amtlab.pipeline import load_control, quick_scenario, run_pipeline
from amtlab.simulation import ShiftControlParams


def test_quick_scenario_parsing():
    sc = quick_scenario("2-3@45,0.6,3.5,250")
    assert (sc.from_gear, sc.to_gear) == (2, 3)
    assert sc.speed_kmh == 45.0 and sc.throttle == 0.6
    assert sc.grade_pct == 3.5 and sc.payload_kg == 250.0
    assert quick_scenario("4-3@60").throttle == 0.6


def test_load_control_defaults_and_yaml(tmp_path):
    assert load_control(None) == ShiftControlParams()
    path = tmp_path / "cal.yaml"
    path.write_text("clutch_close_rate: 7.5\n", encoding="utf-8")
    assert load_control(path).clutch_close_rate == 7.5


def test_pipeline_produces_every_artifact(tmp_path, small_dataset):
    cfg = PipelineConfig.from_dict(
        {
            "output_dir": str(tmp_path / "out"),
            "model": {"targets": ["shift_time_s", "speed_loss_kmh"], "n_trials": 3,
                      "n_splits": 3, "model_names": ["ridge"], "permutation_repeats": 2},
            "calibration": {"n_trials": 8, "mode": "scalar"},
            "ollama": {"enabled": False},
        }
    )
    art = run_pipeline(cfg, dataset=small_dataset, verbose=False)
    root = art.output_dir

    assert (root / "data" / "dataset.csv").exists()
    assert (root / "config.used.yaml").exists()
    for name in ("importance.csv", "improvement.csv", "calibration_params.csv",
                 "tuning_history.csv", "calibration_comparison.csv", "gear_summary.csv"):
        assert (root / "tables" / name).exists()
    for target in cfg.model.targets:
        assert (root / "models" / f"surrogate_{target}.joblib").exists()
    assert art.figures and all(p.exists() for p in art.figures)

    assert art.report_source == "fallback"
    assert art.report_path.read_text(encoding="utf-8").startswith("# AMT")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset"]["n_events"] == len(small_dataset)
    assert summary["calibration"]["mode"] == "scalar"
    assert summary["objectives"]["keys"] == list(cfg.calibration.objectives)
    assert summary["by_current_gear"], "カレントギア別の集計がレポートに含まれること"
    assert {"gear_analysis.png", "speed_time_relation.png"} <= {
        p.name for p in art.figures
    }
    assert art.calibration.best_control.clipped() == art.calibration.best_control
