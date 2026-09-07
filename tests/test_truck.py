import numpy as np
import pytest

from amtlab.dataset import ScenarioSpace, build_dataset
from amtlab.features import extract_kpis
from amtlab.simulation import (
    ShiftControlParams,
    ShiftScenario,
    get_vehicle,
    passenger_6speed,
    scaled_bounds,
    simulate_shift,
    truck_12speed,
)
from amtlab.simulation.controller import CONTROL_BOUNDS


@pytest.fixture(scope="module")
def truck():
    return truck_12speed()


# --- プリセット ------------------------------------------------------------------
def test_truck_has_twelve_gears(truck):
    assert truck.transmission.n_gears() == 12
    ratios = truck.transmission.gear_ratios
    assert all(b < a for a, b in zip(ratios, ratios[1:]))
    assert ratios[-1] == pytest.approx(1.0)  # トップは直結


def test_gear_beyond_the_transmission_is_rejected(truck):
    with pytest.raises(IndexError, match="12 段"):
        truck.transmission.total_ratio(13)
    assert truck.transmission.has_gear(12)
    assert not truck.transmission.has_gear(0)


def test_presets_are_addressable_by_name():
    assert get_vehicle("truck12").transmission.n_gears() == 12
    assert get_vehicle("passenger6").transmission.n_gears() == 6
    assert get_vehicle(None).transmission.n_gears() == 6
    with pytest.raises(KeyError):
        get_vehicle("spaceship")


def test_truck_cruises_at_a_realistic_engine_speed(truck):
    rpm = truck.engine_speed_for(80.0 / 3.6, 12) * 9.5493
    assert 1200 < rpm < 1600  # 12 速 80 km/h でおよそ 1400 rpm


def test_truck_is_much_heavier_than_the_car(truck):
    assert truck.total_mass > 10 * passenger_6speed().total_mass


# --- 適合値のスケーリング ----------------------------------------------------------
def test_bounds_scale_with_engine_torque(truck):
    car_lo, car_hi = CONTROL_BOUNDS["torque_reduce_rate"]
    lo, hi = scaled_bounds(truck)["torque_reduce_rate"]
    factor = max(truck.engine.wot_torque) / 190.0
    assert lo == pytest.approx(car_lo * factor)
    assert hi == pytest.approx(car_hi * factor)
    assert factor > 10  # トラックは 1 桁大きい


def test_stroke_rates_are_not_scaled(truck):
    assert scaled_bounds(truck)["clutch_close_rate"] == CONTROL_BOUNDS["clutch_close_rate"]


def test_slip_offsets_scale_with_the_speed_range(truck):
    lo, hi = scaled_bounds(truck)["sync_slip_offset"]
    assert hi < CONTROL_BOUNDS["sync_slip_offset"][1]  # 回転域が狭い


def test_default_control_is_scaled_for_the_vehicle(truck):
    default = ShiftControlParams.default_for(truck)
    assert default.torque_reduce_rate > ShiftControlParams().torque_reduce_rate * 5
    assert default.clutch_close_rate == ShiftControlParams().clutch_close_rate
    # 既定値はスケール後の範囲に収まっている
    assert default.clipped(scaled_bounds(truck)).as_dict() == pytest.approx(
        default.as_dict()
    )


def test_car_bounds_are_unchanged():
    assert scaled_bounds(passenger_6speed()) == pytest.approx(
        {k: v for k, v in CONTROL_BOUNDS.items()}
    )


# --- シミュレーション --------------------------------------------------------------
@pytest.mark.parametrize("gear", [3, 6, 9, 11])
def test_truck_upshifts_complete(truck, gear):
    space = ScenarioSpace.for_vehicle(truck)
    rpm = float(np.mean(space.upshift_rpm))
    speed = (
        rpm / 9.5493 * truck.wheel_radius / truck.transmission.total_ratio(gear) * 3.6
    )
    result = simulate_shift(
        ShiftControlParams.default_for(truck),
        ShiftScenario(gear, gear + 1, speed_kmh=speed, throttle=0.8),
        vehicle=truck,
    )
    assert result.completed
    kpis = extract_kpis(result)
    assert 0.3 < kpis["shift_time_s"] < 3.0
    assert kpis["speed_loss_kmh"] > 0


def test_truck_scenario_space_uses_the_diesel_band(truck):
    space = ScenarioSpace.for_vehicle(truck)
    assert 1200 < space.upshift_rpm[0] < 1500
    assert 1700 < space.upshift_rpm[1] < 2100
    assert space.downshift_rpm[0] > truck.engine.idle_rpm


def test_truck_doe_covers_high_gears(truck):
    df = build_dataset(n_samples=40, seed=5, vehicle=truck, n_jobs=1)
    assert df["from_gear"].max() >= 9
    assert df["completed"].mean() > 0.85
    # 適合値は車両スケール後の範囲でサンプリングされている
    lo, hi = scaled_bounds(truck)["torque_reduce_rate"]
    assert df["torque_reduce_rate"].between(lo, hi).all()
    assert df["torque_reduce_rate"].max() > CONTROL_BOUNDS["torque_reduce_rate"][1]
