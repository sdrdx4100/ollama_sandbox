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
    good = {"jerk_rms": 3.0, "shift_time_s": 0.6, "speed_loss_kmh": 1.0, "completed": 1.0}
    bad = {"jerk_rms": 9.0, "shift_time_s": 1.4, "speed_loss_kmh": 8.0, "completed": 1.0}
    assert scalar_objective(good) < scalar_objective(bad)


def test_incomplete_shift_is_penalised():
    base = {"jerk_rms": 3.0, "shift_time_s": 0.6, "speed_loss_kmh": 1.0}
    assert scalar_objective({**base, "completed": 0.0}) > scalar_objective(
        {**base, "completed": 1.0}
    )


def test_nan_kpi_gives_infinite_score():
    assert scalar_objective({"jerk_rms": np.nan, "shift_time_s": 1.0,
                             "speed_loss_kmh": 1.0, "completed": 1.0}) == float("inf")


def test_weights_cover_all_objectives():
    assert set(DEFAULT_WEIGHTS) == set(OBJECTIVE_KPIS)


def test_result_row_contains_conditions_and_calibration():
    row = result_to_row(simulate_shift(ShiftControlParams(), ShiftScenario(3, 4, 70.0, 0.4)))
    assert row["from_gear"] == 3 and row["to_gear"] == 4 and row["is_upshift"] == 1
    for key in ShiftControlParams().as_dict():
        assert key in row


# --- 車速の落ち込み ------------------------------------------------------------
def test_speed_drop_metrics_on_a_dip():
    from amtlab.features import speed_drop_metrics

    t = np.linspace(0.0, 2.0, 201)
    v = 50.0 + np.where(t < 0.2, 2.0 * t, np.where(t < 1.0, 0.4 - 3.0 * (t - 0.2), -2.0))
    m = speed_drop_metrics(t, v, pre_accel=0.0)
    assert m["speed_drop_kmh"] == pytest.approx(2.4, abs=0.1)
    assert m["speed_drop_pct"] == pytest.approx(2.4 / 50.4 * 100, abs=0.2)
    assert np.isnan(m["speed_recovery_s"])  # 窓内でピークまで戻らない


def test_speed_loss_uses_the_no_shift_reference():
    from amtlab.features import speed_drop_metrics

    # 加速中に伸びが止まるだけで、車速自体は落ちないケース
    t = np.linspace(0.0, 1.0, 101)
    v = 30.0 + 3.6 * 2.0 * np.minimum(t, 0.2)  # 0.2 s で加速が止まる
    m = speed_drop_metrics(t, v, pre_accel=2.0)
    assert m["speed_drop_kmh"] == pytest.approx(0.0, abs=1e-6)
    assert m["speed_loss_kmh"] == pytest.approx(3.6 * 2.0 * 0.8, abs=0.1)


def test_speed_metrics_are_present_in_kpis(kpis):
    for key in ("speed_drop_kmh", "speed_drop_pct", "speed_loss_kmh", "current_gear"):
        assert key in kpis
    assert kpis["speed_drop_kmh"] >= 0.0
    assert kpis["speed_loss_kmh"] >= 0.0


def test_short_window_returns_nan_metrics():
    from amtlab.features import speed_drop_metrics

    m = speed_drop_metrics(np.array([0.0]), np.array([50.0]), pre_accel=1.0)
    assert all(np.isnan(v) for v in m.values())


# --- 目的仕様 -------------------------------------------------------------------
def test_objective_spec_rejects_missing_weight():
    from amtlab.features import ObjectiveSpec

    with pytest.raises(KeyError):
        ObjectiveSpec(keys=("shift_time_s", "speed_loss_kmh"),
                      weights={"shift_time_s": 1.0})


def test_objective_spec_rejects_unknown_kpi():
    from amtlab.features import ObjectiveSpec

    with pytest.raises(KeyError):
        ObjectiveSpec(keys=("mystery_kpi",), weights={"mystery_kpi": 1.0})


def test_objective_vector_follows_the_spec():
    from amtlab.features import ObjectiveSpec

    spec = ObjectiveSpec.from_config(weights={"speed_loss_kmh": 0.5, "shift_time_s": 0.5})
    kpis = {"shift_time_s": 0.9, "speed_loss_kmh": 4.0, "jerk_rms": 6.0, "completed": 1.0}
    assert objective_vector(kpis, spec) == (4.0, 0.9)


# --- カレントギア別の集計 --------------------------------------------------------
def test_gear_summary_groups_by_current_gear(small_dataset):
    from amtlab.features import gear_summary

    g = gear_summary(small_dataset)
    assert {"current_gear", "direction"} <= set(g.columns)
    assert set(g["direction"]) <= {"upshift", "downshift"}
    assert g["shift_time_s_count"].sum() == len(small_dataset)
    assert set(g["current_gear"]) == set(small_dataset["from_gear"].astype(int))


def test_gear_long_format_is_tidy(small_dataset):
    from amtlab.features import GEAR_SUMMARY_KPIS, gear_long_format

    long = gear_long_format(small_dataset)
    assert set(long["kpi"]) == set(GEAR_SUMMARY_KPIS)
    assert len(long) == len(small_dataset) * len(GEAR_SUMMARY_KPIS)
