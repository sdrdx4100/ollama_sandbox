import json

import pandas as pd
import pytest

from amtlab.cli import build_parser, main


def test_parser_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_simulate_writes_figure_and_trace(tmp_path, capsys):
    out = tmp_path / "figs"
    code = main(["simulate", "--scenario", "2-3@45,0.6", "--out", str(out), "--trace-csv"])
    assert code == 0
    assert (out / "shift_trace.png").exists()
    assert (out / "trace.csv").exists()
    printed = capsys.readouterr().out
    assert "jerk_rms" in printed


def test_dataset_command(tmp_path):
    path = tmp_path / "doe.csv"
    assert main(["dataset", "--samples", "6", "--seed", "1", "--jobs", "1",
                 "--out", str(path)]) == 0
    assert len(pd.read_csv(path)) == 6


def test_demo_log_and_ingest(tmp_path, capsys):
    log = tmp_path / "log.csv"
    assert main(["demo-log", "--out", str(log), "--seed", "3"]) == 0
    assert main(["ingest", "--log", str(log), "--out", str(tmp_path / "ing")]) == 0
    assert (tmp_path / "ing" / "tables" / "log_events.csv").exists()
    assert (tmp_path / "ing" / "figures" / "log_overview.png").exists()


def test_j1939_demo_log_inspect_and_ingest(tmp_path, capsys):
    log = tmp_path / "j1939.csv"
    assert main(["demo-log", "--j1939", "--out", str(log), "--seed", "5"]) == 0
    assert main(["inspect", "--log", str(log), "--save", str(tmp_path / "ch.csv")]) == 0
    printed = capsys.readouterr().out
    assert "SPN" in printed and "574" in printed
    assert (tmp_path / "ch.csv").exists()

    assert main(["ingest", "--log", str(log), "--out", str(tmp_path / "ing")]) == 0
    kpi = pd.read_csv(tmp_path / "ing" / "tables" / "log_events.csv")
    assert (kpi["detection_source"] == "shift_in_process").all()


def test_batch_ingest_over_a_directory(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    from amtlab.ingest import make_j1939_demo_log

    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(3):
        make_j1939_demo_log(n_shifts=3, seed=20 + i).to_parquet(
            logs / f"drive_{i}.parquet", index=False
        )

    assert main(["inspect", "--log", str(logs), "--save", str(tmp_path / "ch.csv")]) == 0
    assert (tmp_path / "ch_presence.csv").exists()
    assert "全ファイルにある信号" in capsys.readouterr().out

    out = tmp_path / "batch"
    assert main(["ingest", "--log", str(logs), "--out", str(out), "--jobs", "1"]) == 0
    events = pd.read_csv(out / "tables" / "log_events.csv")
    assert events["source_file"].nunique() == 3
    assert (out / "tables" / "log_files.csv").exists()
    assert (out / "tables" / "signal_presence.csv").exists()
    assert (out / "figures" / "file_comparison.png").exists()


def test_report_command_uses_fallback(tmp_path):
    summary = {
        "dataset": {"n_events": 1, "upshift_ratio": 1.0, "speed_range_kmh": [10, 20],
                    "kpi_statistics": {}},
        "surrogate_models": {"jerk_rms": {"r2": 0.5, "rmse": 1.0, "model": "ridge"}},
        "top_importance": [{"feature": "throttle", "importance": 0.1, "kind": "condition"}],
        "calibration": {"mode": "scalar", "n_trials": 1,
                        "baseline_params": {"clutch_close_rate": 3.5},
                        "optimized_params": {"clutch_close_rate": 4.0},
                        "improvement": [{"metric": "jerk_rms", "baseline": 2.0,
                                         "optimized": 1.0, "improvement_pct": 50.0}],
                        "n_pareto_solutions": 0},
    }
    spath = tmp_path / "summary.json"
    spath.write_text(json.dumps(summary), encoding="utf-8")
    rpath = tmp_path / "report.md"
    assert main(["report", "--summary", str(spath), "--out", str(rpath),
                 "--no-ollama"]) == 0
    assert "AMT" in rpath.read_text(encoding="utf-8")


def test_ollama_command_reports_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    assert main(["ollama"]) == 1
