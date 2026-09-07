import numpy as np
import pytest

from amtlab.simulation import (
    ShiftControlParams,
    ShiftScenario,
    SimSettings,
    simulate_shift,
)


@pytest.fixture(scope="module")
def upshift():
    return simulate_shift(ShiftControlParams(), ShiftScenario(2, 3, 45.0, 0.6))


def test_shift_completes_and_reaches_target_gear(upshift):
    assert upshift.completed
    assert upshift.trace["gear"].iloc[-1] == 3
    assert upshift.shift_end_time > upshift.shift_start_time


def test_phase_sequence_is_monotonic(upshift):
    order = [
        "in_gear", "torque_reduce", "clutch_open", "gear_out", "speed_sync",
        "gear_in", "clutch_close", "torque_recovery", "done",
    ]
    seen = list(dict.fromkeys(upshift.trace["phase"]))
    assert seen == [p for p in order if p in seen]
    assert "speed_sync" in seen and "clutch_close" in seen


def test_trace_has_no_nan_in_key_signals(upshift):
    cols = ["engine_speed_rpm", "speed_kmh", "accel", "jerk", "clutch_torque"]
    assert not upshift.trace[cols].isna().any().any()


def test_neutral_phase_has_no_drive_torque(upshift):
    neutral = upshift.trace[upshift.trace["gear"] == 0]
    assert len(neutral) > 0
    assert np.allclose(neutral["clutch_torque"], 0.0)


def test_upshift_reduces_engine_speed(upshift):
    tr = upshift.trace
    assert tr["engine_speed_rpm"].iloc[-1] < tr["engine_speed_rpm"].iloc[0]


def test_downshift_raises_engine_speed():
    res = simulate_shift(ShiftControlParams(), ShiftScenario(4, 3, 60.0, 0.35))
    tr = res.trace
    assert res.completed
    assert tr["engine_speed_rpm"].iloc[-1] > tr["engine_speed_rpm"].iloc[0]


def test_pre_shift_state_is_settled(upshift):
    pre = upshift.trace[upshift.trace["time"] < upshift.shift_start_time]
    assert pre["jerk"].abs().max() < 1.0  # 初期条件の過渡が乗っていない


def test_faster_clutch_closing_shortens_the_shift():
    slow = simulate_shift(
        ShiftControlParams(clutch_close_rate=1.0), ShiftScenario(2, 3, 45.0, 0.6)
    )
    fast = simulate_shift(
        ShiftControlParams(clutch_close_rate=10.0), ShiftScenario(2, 3, 45.0, 0.6)
    )
    assert fast.shift_end_time < slow.shift_end_time


def test_uphill_shift_loses_more_speed():
    flat = simulate_shift(ShiftControlParams(), ShiftScenario(2, 3, 45.0, 0.4))
    hill = simulate_shift(
        ShiftControlParams(), ShiftScenario(2, 3, 45.0, 0.4, grade_pct=10.0)
    )
    assert hill.trace["accel"].min() < flat.trace["accel"].min()


def test_simulation_terminates_with_extreme_calibration():
    control = ShiftControlParams(
        torque_reduce_rate=200.0, clutch_open_rate=2.0, sync_gain=0.3,
        clutch_close_rate=0.8, torque_recovery_rate=150.0,
    )
    res = simulate_shift(control, ShiftScenario(1, 2, 20.0, 0.9),
                         settings=SimSettings(max_time=5.0))
    assert res.trace["time"].iloc[-1] <= 5.0


def test_identical_gears_are_rejected():
    with pytest.raises(ValueError):
        ShiftScenario(3, 3, 50.0, 0.5)
