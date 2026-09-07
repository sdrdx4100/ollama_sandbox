"""クラッチの ON-OFF(切り方)指標のテスト。"""

import numpy as np
import pandas as pd
import pytest

from amtlab.ingest import (
    CLUTCH_RELEASED_PCT,
    ShiftEvent,
    analyze_log,
    clutch_metrics,
    clutch_profile,
    make_j1939_demo_log,
    standardize,
)


def _trace(slip: np.ndarray, dt: float = 0.01) -> pd.DataFrame:
    trace = pd.DataFrame(
        {
            "time": np.arange(len(slip)) * dt,
            "clutch_slip_pct": slip,
            "gear": np.where(slip > 50, 0, 3),
            "engine_speed_rpm": np.full(len(slip), 1500.0),
            "speed_kmh": np.linspace(50, 52, len(slip)),
        }
    )
    trace.attrs["dt"] = dt
    return trace


def _square_slip(release_s, open_s, engage_s, dt=0.01, lead_s=0.3, tail_s=0.5):
    """台形のすべり波形(切り→全切り→繋ぎ)を作る。"""
    lead = np.zeros(int(lead_s / dt))
    release = np.linspace(0, 100, int(release_s / dt), endpoint=False)
    hold = np.full(int(open_s / dt), 100.0)
    engage = np.linspace(100, 0, int(engage_s / dt), endpoint=False)
    tail = np.zeros(int(tail_s / dt))
    return np.concatenate([lead, release, hold, engage, tail])


@pytest.fixture
def square_event():
    slip = _square_slip(release_s=0.10, open_s=0.40, engage_s=0.20)
    trace = _trace(slip)
    event = ShiftEvent(
        index=0, from_gear=3, to_gear=4, t_start=0.30, t_end=1.00, t_settle=1.50
    )
    return trace, event


# --- 指標の定義どおりに測れているか --------------------------------------------------
def test_cycle_and_phase_times(square_event):
    trace, event = square_event
    m = clutch_metrics(trace, event)
    assert m["clutch_release_time_s"] == pytest.approx(0.10, abs=0.02)
    # 全切り時間はプラトー 0.40 s + ランプのうち 90% を超える肩の分
    assert 0.40 <= m["clutch_open_time_s"] <= 0.47
    assert m["clutch_engage_time_s"] == pytest.approx(0.20, abs=0.03)
    assert m["clutch_cycle_time_s"] == pytest.approx(0.70, abs=0.03)


def test_full_release_ratio_and_mean_slip(square_event):
    trace, event = square_event
    m = clutch_metrics(trace, event)
    assert m["clutch_full_release_ratio"] == pytest.approx(
        m["clutch_open_time_s"] / m["clutch_cycle_time_s"]
    )
    assert 0.55 < m["clutch_full_release_ratio"] < 0.70
    assert m["clutch_peak_slip_pct"] == pytest.approx(100.0)
    # 台形の平均: (立上り + 全切り + 立下り) / サイクル時間
    assert 60.0 < m["clutch_mean_slip_pct"] < 90.0


def test_shorter_cycle_with_the_same_depth_scores_higher(square_event):
    """短時間でしっかり切る = 全切り比率が高い。"""
    short = clutch_metrics(*(_trace(_square_slip(0.05, 0.40, 0.05)),
                             square_event[1]))
    slow = clutch_metrics(*(_trace(_square_slip(0.30, 0.40, 0.30)),
                            square_event[1]))
    assert short["clutch_cycle_time_s"] < slow["clutch_cycle_time_s"]
    assert short["clutch_full_release_ratio"] > slow["clutch_full_release_ratio"]
    assert short["clutch_release_rate_pct_s"] > slow["clutch_release_rate_pct_s"]


def test_partial_release_is_reported_without_open_time():
    """全切りに達しない(半クラのまま)場合。"""
    slip = _square_slip(0.10, 0.30, 0.10) * 0.5  # 最大 50%
    trace = _trace(slip)
    event = ShiftEvent(0, 3, 4, t_start=0.30, t_end=1.0, t_settle=1.4)
    m = clutch_metrics(trace, event)
    assert m["clutch_open_time_s"] == 0.0
    assert m["clutch_full_release_ratio"] == 0.0
    assert np.isnan(m["clutch_release_time_s"])
    assert m["clutch_peak_slip_pct"] < CLUTCH_RELEASED_PCT


def test_no_clutch_action_at_all():
    trace = _trace(np.zeros(200))
    m = clutch_metrics(trace, ShiftEvent(0, 3, 4, 0.3, 1.0, 1.4))
    assert m["clutch_engaged_throughout"] == 1.0
    assert "clutch_cycle_time_s" not in m


def test_missing_signal_returns_nothing():
    trace = pd.DataFrame({"time": np.arange(50) * 0.01, "gear": 3})
    trace.attrs["dt"] = 0.01
    assert clutch_metrics(trace, ShiftEvent(0, 3, 4, 0.1, 0.3, 0.4)) == {}


def test_brief_dip_does_not_end_the_cycle():
    """ノイズで一瞬 0 に落ちてもサイクルを打ち切らない。"""
    slip = _square_slip(0.05, 0.40, 0.05)
    slip[int(0.42 / 0.01)] = 0.0  # 全切り中に 1 サンプルだけ欠測
    m = clutch_metrics(_trace(slip), ShiftEvent(0, 3, 4, 0.30, 1.0, 1.4))
    assert m["clutch_cycle_time_s"] > 0.4


# --- 波形の切り出し ----------------------------------------------------------------
def test_profile_uses_a_fixed_time_grid(square_event):
    trace, event = square_event
    profile = clutch_profile(trace, event, n_points=100, lead_time=0.2, span=2.0)
    assert len(profile) == 100
    assert profile["rel_time_s"].iloc[0] == pytest.approx(-0.2)
    assert profile["rel_time_s"].iloc[-1] == pytest.approx(2.0)
    assert profile["clutch_slip_pct"].isna().any()  # 区間外は NaN
    assert profile["current_gear"].eq(3).all()


def test_profiles_from_different_events_share_the_grid():
    log = make_j1939_demo_log(n_shifts=3, seed=4)
    trace = standardize(log)
    from amtlab.ingest import detect_shift_events

    events = detect_shift_events(trace)
    grids = [clutch_profile(trace, ev)["rel_time_s"].to_numpy() for ev in events]
    for grid in grids[1:]:
        assert np.allclose(grid, grids[0])


# --- 実ログ経路 --------------------------------------------------------------------
def test_clutch_metrics_reach_the_event_table():
    _, kpi = analyze_log(make_j1939_demo_log(n_shifts=3, seed=3))
    for column in ("clutch_cycle_time_s", "clutch_open_time_s",
                   "clutch_full_release_ratio", "clutch_mean_slip_pct"):
        assert column in kpi.columns
        assert kpi[column].notna().all()
    assert kpi["clutch_full_release_ratio"].between(0.0, 1.0).all()
    assert (kpi["clutch_cycle_time_s"] <= kpi["shift_time_s"] + 0.3).all()


def test_slip_is_derived_when_spn_522_is_missing():
    """SPN 522 が無い場合は回転差から代用するが、出所を記録する。

    ニュートラル中は入力軸がクラッチから切り離されるため、回転差からは
    「どれだけ深く切ったか」は測れない。代用値であることを必ず残す。
    """
    log = make_j1939_demo_log(n_shifts=2, seed=8).drop(
        columns=["ETC1::PercentClutchSlip"]
    )
    trace = standardize(log)
    assert "clutch_slip_pct" in trace.columns
    assert trace.attrs["clutch_slip_source"].startswith("derived")

    measured = standardize(make_j1939_demo_log(n_shifts=2, seed=8))
    assert measured.attrs["clutch_slip_source"] == "SPN 522"
    # 代用値は切り深さを過小評価する
    assert trace["clutch_slip_pct"].max() < measured["clutch_slip_pct"].max()


def test_derived_slip_is_flagged_in_the_event_table():
    log = make_j1939_demo_log(n_shifts=2, seed=8).drop(
        columns=["ETC1::PercentClutchSlip"]
    )
    _, kpi = analyze_log(log)
    assert kpi["clutch_slip_source"].str.startswith("derived").all()


def test_inspect_warns_about_a_derived_slip():
    from amtlab.j1939 import inspect_log

    log = make_j1939_demo_log(n_shifts=2, seed=8).drop(
        columns=["ETC1::PercentClutchSlip"]
    )
    warnings = inspect_log(log).warnings
    assert any("深く切ったかは測れません" in w for w in warnings)


def test_mixed_slip_sources_block_the_clutch_comparison():
    from amtlab.compare import check_comparability

    events = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "source_file": ["a1", "a2", "b1", "b2"],
            "clutch_slip_source": ["SPN 522", "SPN 522", "derived (x)", "derived (x)"],
            "shift_time_s": [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert any("すべり率の出所" in w for w in check_comparability(events))
