import numpy as np
import pytest

from amtlab.features import (
    DEFAULT_WEIGHTS,
    OBJECTIVE_KPIS,
    extract_kpis,
    objective_vector,
    result_to_row,
    scalar_objective,
)
from amtlab.simulation import ShiftControlParams, ShiftScenario, simulate_shift


@pytest.fixture(scope="module")
def kpis():
    return extract_kpis(simulate_shift(ShiftControlParams(), ShiftScenario(2, 3, 45.0, 0.6)))


def test_kpi_keys_and_signs(kpis):
    for key in ("shift_time_s", "jerk_rms", "jerk_peak", "clutch_energy_j",
                "torque_interrupt_s", "engine_flare_rpm", "quality_score"):
        assert key in kpis
    assert kpis["shift_time_s"] > 0
    assert kpis["jerk_peak"] >= kpis["jerk_rms"] > 0
    assert kpis["clutch_energy_j"] >= 0
    assert kpis["engine_flare_rpm"] >= 0
    assert kpis["speed_loss_kmh"] >= 0


def test_torque_interruption_is_shorter_than_the_shift(kpis):
    assert 0 < kpis["torque_interrupt_s"] <= kpis["shift_time_s"] + 0.6


def test_objective_vector_matches_declared_kpis(kpis):
    assert objective_vector(kpis) == tuple(kpis[k] for k in OBJECTIVE_KPIS)


def test_scalar_objective_rewards_better_kpis():
    good = {"jerk_rms": 3.0, "shift_time_s": 0.6, "clutch_energy_j": 50.0, "completed": 1.0}
    bad = {"jerk_rms": 9.0, "shift_time_s": 1.4, "clutch_energy_j": 400.0, "completed": 1.0}
    assert scalar_objective(good) < scalar_objective(bad)


def test_incomplete_shift_is_penalised():
    base = {"jerk_rms": 3.0, "shift_time_s": 0.6, "clutch_energy_j": 50.0}
    assert scalar_objective({**base, "completed": 0.0}) > scalar_objective(
        {**base, "completed": 1.0}
    )


def test_nan_kpi_gives_infinite_score():
    assert scalar_objective({"jerk_rms": np.nan, "shift_time_s": 1.0,
                             "clutch_energy_j": 1.0, "completed": 1.0}) == float("inf")


def test_weights_cover_all_objectives():
    assert set(DEFAULT_WEIGHTS) == set(OBJECTIVE_KPIS)


def test_result_row_contains_conditions_and_calibration():
    row = result_to_row(simulate_shift(ShiftControlParams(), ShiftScenario(3, 4, 70.0, 0.4)))
    assert row["from_gear"] == 3 and row["to_gear"] == 4 and row["is_upshift"] == 1
    for key in ShiftControlParams().as_dict():
        assert key in row
