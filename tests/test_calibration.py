import pytest

from amtlab.calibration import (
    CalibrationSetting,
    aggregate,
    calibrate,
    compare_controls,
    default_scenarios,
    evaluate_control,
    pareto_frame,
)
from amtlab.features import OBJECTIVE_KPIS
from amtlab.simulation import ShiftControlParams


@pytest.fixture(scope="module")
def setting():
    return CalibrationSetting(scenarios=default_scenarios()[:3], n_jobs=1)


def test_evaluate_control_covers_every_scenario(setting):
    df = evaluate_control(ShiftControlParams(), setting)
    assert len(df) == len(setting.scenarios)
    assert {"jerk_rms", "shift_time_s", "clutch_energy_j"} <= set(df.columns)


def test_aggregate_blends_mean_and_worst(setting):
    df = evaluate_control(ShiftControlParams(), setting)
    agg = aggregate(df, setting)
    for key in OBJECTIVE_KPIS:
        assert agg[f"{key}_mean"] <= agg[key] <= agg[f"{key}_worst"] + 1e-9
    assert agg["score"] > 0


def test_scalar_calibration_improves_on_a_poor_baseline(setting):
    poor = ShiftControlParams(
        torque_reduce_rate=250.0, clutch_open_rate=3.0, sync_gain=0.4,
        sync_slip_offset=24.0, clutch_close_rate=1.0, torque_recovery_rate=200.0,
    )
    outcome = calibrate(setting=setting, n_trials=25, mode="scalar", seed=1, baseline=poor)
    assert outcome.best_metrics["score"] < outcome.baseline_metrics["score"]
    improvement = outcome.improvement()
    assert set(improvement["metric"]) >= set(OBJECTIVE_KPIS)


def test_optimised_parameters_stay_within_bounds(setting):
    outcome = calibrate(setting=setting, n_trials=12, mode="scalar", seed=0)
    clipped = outcome.best_control.clipped().as_dict()
    assert clipped == pytest.approx(outcome.best_control.as_dict())


def test_pareto_mode_returns_a_front(setting):
    outcome = calibrate(setting=setting, n_trials=16, mode="pareto", seed=0)
    assert len(outcome.pareto) >= 1
    assert set(OBJECTIVE_KPIS) <= set(outcome.pareto.columns)
    assert outcome.pareto["score"].is_monotonic_increasing
    assert pareto_frame(outcome.study).shape[0] == outcome.pareto.shape[0]


def test_compare_controls_is_long_format(setting):
    df = compare_controls(
        {"a": ShiftControlParams(), "b": ShiftControlParams(clutch_close_rate=8.0)}, setting
    )
    assert set(df["calibration"]) == {"a", "b"}
    assert len(df) == 2 * len(setting.scenarios)
