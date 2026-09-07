import json

import pytest

from amtlab.reporting import (
    OllamaClient,
    build_summary,
    fallback_report,
    generate_report,
    write_report,
)


@pytest.fixture
def summary():
    return {
        "dataset": {"n_events": 120, "upshift_ratio": 0.65,
                    "speed_range_kmh": [15.0, 180.0], "kpi_statistics": {}},
        "surrogate_models": {"jerk_rms": {"r2": 0.72, "rmse": 1.1, "model": "hist_gbr"}},
        "top_importance": [{"feature": "throttle", "importance": 0.5, "kind": "condition"}],
        "calibration": {
            "mode": "pareto", "n_trials": 30,
            "baseline_params": {"clutch_close_rate": 3.5},
            "optimized_params": {"clutch_close_rate": 6.2},
            "improvement": [{"metric": "jerk_rms", "baseline": 8.8,
                             "optimized": 5.3, "improvement_pct": 39.5}],
            "n_pareto_solutions": 5,
        },
    }


def test_fallback_report_contains_all_sections(summary):
    text = fallback_report(summary)
    for heading in ("# AMT 変速品質 解析レポート", "## 1. サマリ", "## 2.", "## 4.", "## 5."):
        assert heading in text
    assert "clutch_close_rate" in text
    assert "39.5" in text


def test_generate_report_falls_back_when_disabled(summary):
    text, source = generate_report(summary, enabled=False)
    assert source == "fallback"
    assert text == fallback_report(summary)


def test_generate_report_falls_back_when_ollama_is_unreachable(summary):
    client = OllamaClient(host="http://127.0.0.1:9", timeout_s=1.0)
    text, source = generate_report(summary, client=client)
    assert source == "fallback"
    assert text.startswith("# AMT")


def test_unreachable_client_reports_unavailable():
    client = OllamaClient(host="http://127.0.0.1:9", timeout_s=1.0)
    assert client.is_available() is False
    assert client.available_models() == []


def test_host_is_normalised(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaClient(host="localhost:11434/").host == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    assert OllamaClient().host == "http://gpu-box:11434"


def test_write_report_creates_parent_directories(tmp_path, summary):
    path = write_report(fallback_report(summary), tmp_path / "nested" / "report.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# AMT")


def test_summary_is_json_serialisable(small_dataset):
    from amtlab.calibration import CalibrationSetting, calibrate, default_scenarios
    from amtlab.modeling import fit_surrogate, importance_frame

    surrogate = fit_surrogate(small_dataset, "jerk_rms", "ridge", n_splits=3)
    importance = importance_frame(surrogate, small_dataset, n_repeats=2)
    outcome = calibrate(
        setting=CalibrationSetting(scenarios=default_scenarios()[:2], n_jobs=1),
        n_trials=4, mode="scalar", seed=0,
    )
    summary = build_summary(small_dataset, {"jerk_rms": surrogate}, importance, outcome)
    json.dumps(summary, ensure_ascii=False)
    assert summary["dataset"]["n_events"] == len(small_dataset)
    assert "jerk_rms" in summary["surrogate_models"]
