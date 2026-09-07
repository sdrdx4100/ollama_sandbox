import numpy as np
import pytest

from amtlab.dataset import (
    CONTROL_FEATURES,
    FEATURE_COLUMNS,
    ScenarioSpace,
    build_dataset,
    run_batch,
)
from amtlab.simulation import ShiftControlParams
from amtlab.simulation.controller import CONTROL_BOUNDS


def test_dataset_shape_and_columns(small_dataset):
    assert len(small_dataset) == 150
    for col in FEATURE_COLUMNS + ["jerk_rms", "shift_time_s", "clutch_energy_j"]:
        assert col in small_dataset.columns


def test_dataset_is_reproducible():
    a = build_dataset(n_samples=8, seed=3, n_jobs=1)
    b = build_dataset(n_samples=8, seed=3, n_jobs=1)
    assert np.allclose(a["jerk_rms"], b["jerk_rms"])


def test_sampled_calibration_stays_within_bounds(small_dataset):
    for name in CONTROL_FEATURES:
        lo, hi = CONTROL_BOUNDS[name]
        assert small_dataset[name].between(lo, hi).all()


def test_sampled_scenarios_keep_engine_speed_feasible():
    space = ScenarioSpace()
    rng = np.random.default_rng(0)
    veh = space.vehicle
    for _ in range(50):
        sc = space.sample(rng)
        rpm_to = veh.engine_speed_for(sc.speed_kmh / 3.6, sc.to_gear) * 9.5493
        assert space.min_engine_rpm <= rpm_to <= space.max_engine_rpm
        assert abs(sc.from_gear - sc.to_gear) == 1


def test_run_batch_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        run_batch([ShiftControlParams()], [], n_jobs=1)


def test_most_shifts_complete(small_dataset):
    assert small_dataset["completed"].mean() > 0.9
