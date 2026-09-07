import pandas as pd
import pytest

from amtlab import viz
from amtlab.ingest import analyze_log, make_demo_log
from amtlab.modeling import fit_surrogate, importance_frame, partial_dependence_frame
from amtlab.simulation import ShiftControlParams, ShiftScenario, simulate_shift


@pytest.fixture(scope="module")
def surrogate(small_dataset):
    return fit_surrogate(small_dataset, "jerk_rms", "ridge", n_splits=3)


def test_shift_trace_figure(tmp_path):
    result = simulate_shift(ShiftControlParams(), ShiftScenario(2, 3, 45.0, 0.6))
    path = viz.plot_shift_trace(result, tmp_path)
    assert path.exists() and path.stat().st_size > 0


def test_trace_comparison_figure(tmp_path):
    scenario = ShiftScenario(2, 3, 45.0, 0.6)
    results = {
        "baseline": simulate_shift(ShiftControlParams(), scenario),
        "fast": simulate_shift(ShiftControlParams(clutch_close_rate=9.0), scenario),
    }
    assert viz.plot_trace_comparison(results, tmp_path).exists()


def test_dataset_figures(small_dataset, tmp_path):
    assert viz.plot_kpi_distributions(small_dataset, tmp_path).exists()
    assert viz.plot_kpi_correlation(small_dataset, tmp_path).exists()
    assert viz.plot_condition_map(small_dataset, tmp_path).exists()


def test_gear_figures(small_dataset, tmp_path):
    assert viz.plot_gear_analysis(small_dataset, tmp_path).exists()
    assert viz.plot_speed_time_relation(small_dataset, tmp_path).exists()


def test_model_figures(surrogate, small_dataset, tmp_path):
    assert viz.plot_model_diagnostics(surrogate, tmp_path).exists()
    imp = importance_frame(surrogate, small_dataset, n_repeats=2)
    assert viz.plot_importance(imp, tmp_path).exists()
    pdp = partial_dependence_frame(surrogate, small_dataset,
                                   features=["clutch_close_rate"], grid_resolution=5)
    assert viz.plot_partial_dependence(pdp, tmp_path).exists()


def test_optimisation_figures(tmp_path):
    trials = pd.DataFrame({"trial": [0, 1, 2], "objective": [1.0, 0.8, 0.9],
                           "best_so_far": [1.0, 0.8, 0.8]})
    assert viz.plot_optuna_history(trials, tmp_path).exists()
    pareto = pd.DataFrame({"shift_time_s": [1.0, 0.8], "speed_loss_kmh": [2.0, 3.5],
                           "jerk_rms": [5.0, 6.0], "score": [0.9, 1.0]})
    assert viz.plot_pareto(pareto, tmp_path).exists()
    assert viz.plot_pareto(pareto, tmp_path, "pareto2",
                           keys=("shift_time_s", "speed_loss_kmh")).exists()


def test_calibration_comparison_figure(tmp_path):
    from amtlab.calibration import CalibrationSetting, compare_controls, default_scenarios

    setting = CalibrationSetting(scenarios=default_scenarios()[:2], n_jobs=1)
    compare = compare_controls(
        {"baseline": ShiftControlParams(),
         "optimized": ShiftControlParams(clutch_close_rate=8.0)}, setting
    )
    assert viz.plot_calibration_comparison(compare, tmp_path).exists()


def test_log_overview_figure(tmp_path):
    trace, kpi = analyze_log(make_demo_log(n_shifts=3, seed=1))
    assert viz.plot_log_overview(trace, kpi, tmp_path).exists()
