import numpy as np
import pytest

from amtlab.simulation.vehicle import VehicleParams, rads_to_rpm, rpm_to_rads


def test_rpm_conversion_roundtrip():
    assert rads_to_rpm(rpm_to_rads(3000.0)) == pytest.approx(3000.0)


def test_wot_torque_within_map_range():
    eng = VehicleParams().engine
    torques = [eng.wot(rpm_to_rads(rpm)) for rpm in (800, 2500, 3500, 6500)]
    assert min(torques) > 0
    assert max(torques) == pytest.approx(max(eng.wot_torque))


def test_drag_torque_is_negative_and_grows_with_speed():
    eng = VehicleParams().engine
    assert eng.drag(rpm_to_rads(1000)) < 0
    assert eng.drag(rpm_to_rads(5000)) < eng.drag(rpm_to_rads(1000))


def test_steady_torque_interpolates_between_limits():
    eng = VehicleParams().engine
    w = rpm_to_rads(3000)
    lo, hi = eng.torque_limits(w)
    assert eng.steady_torque(0.0, w) == pytest.approx(lo)
    assert eng.steady_torque(1.0, w) == pytest.approx(hi)
    assert lo < eng.steady_torque(0.5, w) < hi


def test_clutch_capacity_monotonic_and_zero_below_kiss():
    trm = VehicleParams().transmission
    assert trm.clutch_capacity_at(trm.clutch_kiss_point * 0.5) == 0.0
    caps = [trm.clutch_capacity_at(x) for x in np.linspace(0, 1, 21)]
    assert all(b >= a for a, b in zip(caps, caps[1:]))
    assert caps[-1] == pytest.approx(trm.clutch_capacity)


def test_gear_ratios_are_descending():
    ratios = VehicleParams().transmission.gear_ratios
    assert all(b < a for a, b in zip(ratios, ratios[1:]))


def test_road_load_components():
    veh = VehicleParams()
    assert veh.road_load(0.0) == pytest.approx(0.0, abs=1e-6)
    assert veh.road_load(30.0) > veh.road_load(10.0)
    uphill = VehicleParams(grade=np.arctan(0.1))
    assert uphill.road_load(10.0) > veh.road_load(10.0)


def test_payload_increases_total_mass():
    assert VehicleParams(payload=300).total_mass == pytest.approx(1800.0)
