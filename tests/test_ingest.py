import numpy as np
import pandas as pd
import pytest

from amtlab.ingest import (
    SignalMap,
    analyze_log,
    detect_shift_events,
    event_kpis,
    make_demo_log,
    standardize,
)


@pytest.fixture(scope="module")
def demo_log():
    return make_demo_log(n_shifts=4, seed=3)


def test_demo_log_is_continuous(demo_log):
    assert demo_log["time"].is_monotonic_increasing
    assert demo_log["speed_kmh"].diff().abs().max() < 2.0  # 段差のない 1 本のログ


def test_standardize_requires_core_signals():
    with pytest.raises(KeyError):
        standardize(pd.DataFrame({"time": [0.0, 0.1], "gear": [1, 1]}))


def test_standardize_resamples_to_uniform_grid(demo_log):
    trace = standardize(demo_log, resample_hz=50.0)
    dt = np.diff(trace["time"].to_numpy())
    assert np.allclose(dt, 0.02, atol=1e-9)
    assert {"accel", "accel_filt", "jerk"} <= set(trace.columns)


def test_gear_signal_is_resampled_without_phantom_gears(demo_log):
    trace = standardize(demo_log, resample_hz=25.0)
    assert set(trace["gear"]) <= set(demo_log["gear"])


def test_detects_every_upshift(demo_log):
    trace = standardize(demo_log)
    events = detect_shift_events(trace)
    assert [(e.from_gear, e.to_gear) for e in events] == [(1, 2), (2, 3), (3, 4), (4, 5)]
    assert all(e.is_upshift for e in events)
    assert all(e.t_end > e.t_start for e in events)


def test_event_kpis_match_the_simulation_schema(demo_log):
    trace = standardize(demo_log)
    kpis = event_kpis(trace, detect_shift_events(trace)[0])
    for key in ("shift_time_s", "jerk_rms", "jerk_peak", "engine_flare_rpm",
                "speed_drop_kmh", "speed_drop_pct", "speed_loss_kmh",
                "current_gear", "is_upshift"):
        assert key in kpis
    assert kpis["current_gear"] == kpis["from_gear"]
    assert kpis["speed_drop_kmh"] >= 0.0
    assert kpis["shift_time_s"] > 0
    assert kpis["neutral_time_s"] > 0


def test_analyze_log_end_to_end(demo_log, tmp_path):
    path = tmp_path / "log.csv"
    demo_log.to_csv(path, index=False)
    trace, kpi = analyze_log(path, SignalMap(shaft_torque="shaft_torque"))
    assert len(kpi) == 4
    assert kpi["torque_interrupt_s"].notna().all()
    assert (kpi["t_end"] > kpi["t_start"]).all()


def test_no_events_when_gear_never_changes():
    n = 200
    df = pd.DataFrame(
        {
            "time": np.arange(n) * 0.01,
            "engine_speed_rpm": np.full(n, 2000.0),
            "speed_kmh": np.linspace(50, 55, n),
            "gear": np.full(n, 3),
        }
    )
    trace, kpi = analyze_log(df)
    assert kpi.empty
