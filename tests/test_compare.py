import numpy as np
import pandas as pd
import pytest

from amtlab.batch import analyze_logs, iter_log_groups, signal_presence
from amtlab.compare import (
    check_comparability,
    cliffs_delta,
    cluster_bootstrap_ci,
    compare_by_gear,
    compare_groups,
    condition_overlap,
    hedges_g,
    overlap_ratio,
)
from amtlab.ingest import make_j1939_demo_log
from amtlab.simulation import ShiftControlParams

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    """A 社(速い締結)と B 社(遅い締結)のログ群を作る。"""
    root = tmp_path_factory.mktemp("fleet")
    specs = {
        "A": dict(clutch_close_rate=9.0, torque_reduce_rate=2500.0),
        "B": dict(clutch_close_rate=1.6, torque_reduce_rate=500.0),
    }
    for company, kwargs in specs.items():
        directory = root / company
        directory.mkdir()
        rng = np.random.default_rng(1 if company == "A" else 2)
        for i in range(6):
            controls = [
                ShiftControlParams(**kwargs, sync_gain=float(rng.uniform(1.5, 3.0)))
                for _ in range(3)
            ]
            make_j1939_demo_log(
                n_shifts=3, seed=i, start_speed_kmh=25.0, controls=controls
            ).to_parquet(directory / f"{company}_{i}.parquet", index=False)
    return root


@pytest.fixture(scope="module")
def batch(fleet):
    return analyze_logs({"A": fleet / "A", "B": fleet / "B"}, n_jobs=1)


# --- グループ付きの読み込み --------------------------------------------------------
def test_groups_are_expanded_with_labels(fleet):
    pairs = iter_log_groups({"A": fleet / "A", "B": fleet / "B"})
    assert len(pairs) == 12
    assert {g for g, _ in pairs} == {"A", "B"}


def test_the_same_file_cannot_belong_to_two_groups(fleet):
    with pytest.raises(ValueError, match="複数グループ"):
        iter_log_groups({"A": fleet / "A", "duplicate": fleet / "A"})


def test_events_carry_the_group_label(batch):
    assert "group" in batch.events.columns
    assert set(batch.events["group"]) == {"A", "B"}
    assert batch.events.groupby("group")["source_file"].nunique().to_dict() == {"A": 6, "B": 6}


def test_signal_presence_keeps_the_group(fleet):
    presence = signal_presence({"A": fleet / "A", "B": fleet / "B"})
    assert set(presence["group"]) == {"A", "B"}
    assert presence["engine_speed_rpm"].all()


# --- 統計量 ----------------------------------------------------------------------
def test_hedges_g_direction_and_zero():
    a = np.random.default_rng(0).normal(0, 1, 200)
    assert hedges_g(a, a + 1.0) > 0.8
    assert hedges_g(a, a) == pytest.approx(0.0, abs=1e-9)


def test_cliffs_delta_bounds():
    assert cliffs_delta(np.array([1, 2, 3]), np.array([4, 5, 6])) == pytest.approx(1.0)
    assert cliffs_delta(np.array([4, 5, 6]), np.array([1, 2, 3])) == pytest.approx(-1.0)
    a = np.arange(50.0)
    assert abs(cliffs_delta(a, a)) < 1e-9


def test_cluster_bootstrap_is_wider_than_naive_resampling():
    """同じ走行のイベントは相関するので、ファイル単位の CI は広くなるはず。"""
    rng = np.random.default_rng(0)
    rows = []
    for group, offset in (("A", 0.0), ("B", 0.2)):
        for f in range(8):
            file_effect = rng.normal(0, 0.5)  # 走行ごとの癖
            for _ in range(10):
                rows.append(
                    {
                        "group": group,
                        "file": f"{group}{f}",
                        "value": offset + file_effect + rng.normal(0, 0.1),
                    }
                )
    df = pd.DataFrame(rows)
    clustered = cluster_bootstrap_ci(df, "value", "group", "file", "A", "B", n_boot=400)
    df["event"] = np.arange(len(df))  # 1 イベント 1 クラスタ = 素のブートストラップ
    naive = cluster_bootstrap_ci(df, "value", "group", "event", "A", "B", n_boot=400)
    assert (clustered[1] - clustered[0]) > (naive[1] - naive[0])


def test_bootstrap_returns_nan_for_an_empty_group():
    df = pd.DataFrame({"group": ["A"] * 3, "file": ["f"] * 3, "value": [1.0, 2.0, 3.0]})
    low, high = cluster_bootstrap_ci(df, "value", "group", "file", "A", "B", n_boot=10)
    assert np.isnan(low) and np.isnan(high)


# --- 比較 ------------------------------------------------------------------------
def test_slower_calibration_is_detected_as_longer_shift_time(batch):
    result = compare_groups(batch.events, reference="A", files=batch.files, n_boot=300)
    row = result.table[
        (result.table["kpi"] == "shift_time_s") & (result.table["group"] == "B")
    ].iloc[0]
    assert row["diff"] > 0
    assert row["ci_low"] > 0  # 有意に悪化
    assert "悪化" in row["verdict"]
    assert row["adjusted_diff"] > 0  # 条件補正後も同じ向き


def test_comparison_needs_two_groups():
    events = pd.DataFrame({"group": ["A", "A"], "shift_time_s": [1.0, 1.1],
                           "source_file": ["x", "y"]})
    with pytest.raises(ValueError, match="2 グループ以上"):
        compare_groups(events)


def test_unknown_reference_is_rejected(batch):
    with pytest.raises(ValueError, match="基準グループ"):
        compare_groups(batch.events, reference="C")


def test_result_table_has_both_raw_and_adjusted(batch):
    result = compare_groups(batch.events, reference="A", n_boot=200)
    for column in ("median_ref", "median_group", "diff", "ci_low", "ci_high",
                   "cliffs_delta", "hedges_g", "adjusted_diff", "verdict", "n_files"):
        assert column in result.table.columns
    assert (result.table["n_files"] == 6).all()


def test_verdict_is_conservative_when_the_interval_spans_zero():
    rng = np.random.default_rng(3)
    events = pd.DataFrame(
        {
            "group": ["A"] * 60 + ["B"] * 60,
            "source_file": [f"a{i // 10}" for i in range(60)]
            + [f"b{i // 10}" for i in range(60)],
            "shift_time_s": np.concatenate([rng.normal(1.0, 0.2, 60),
                                            rng.normal(1.0, 0.2, 60)]),
            "current_gear": np.tile([1, 2, 3], 40),
            "is_upshift": 1,
            "speed_kmh": rng.uniform(20, 80, 120),
            "throttle": rng.uniform(0.2, 0.9, 120),
        }
    )
    result = compare_groups(events, reference="A", kpis=("shift_time_s",), n_boot=300)
    assert result.table["verdict"].iloc[0] == "差は有意でない"


# --- 比較可能性の検証 --------------------------------------------------------------
def test_mixed_detection_sources_block_the_comparison():
    events = pd.DataFrame(
        {
            "group": ["A"] * 3 + ["B"] * 3,
            "source_file": ["a1", "a2", "a3", "b1", "b2", "b3"],
            "detection_source": ["shift_in_process"] * 3 + ["gear_change"] * 3,
            "shift_time_s": [1.0, 1.1, 1.0, 1.4, 1.5, 1.45],
            "current_gear": [1, 2, 3, 1, 2, 3],
        }
    )
    warnings = check_comparability(events)
    assert any("切り出し方法" in w for w in warnings)


def test_mixed_jerk_quality_is_flagged():
    events = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "source_file": ["a1", "a2", "b1", "b2"],
            "jerk_quality": ["measured", "measured", "low_rate", "low_rate"],
            "shift_time_s": [1.0, 1.0, 1.0, 1.0],
        }
    )
    assert any("ジャーク" in w for w in check_comparability(events))


def test_small_sample_is_flagged():
    events = pd.DataFrame(
        {"group": ["A", "B"], "source_file": ["a", "b"], "shift_time_s": [1.0, 1.2]}
    )
    warnings = check_comparability(events)
    assert any("ファイルしかありません" in w for w in warnings)


def test_condition_overlap_detects_disjoint_operating_ranges():
    events = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "source_file": ["a"] * 20 + ["b"] * 20,
            "current_gear": [1] * 20 + [5] * 20,  # まったく重ならない
            "speed_kmh": np.concatenate([np.linspace(10, 30, 20),
                                         np.linspace(90, 120, 20)]),
            "shift_time_s": np.linspace(0.8, 1.2, 40),
        }
    )
    overlap = condition_overlap(events)
    assert overlap_ratio(overlap) == pytest.approx(0.0)
    result = compare_groups(events, reference="A", kpis=("shift_time_s",), n_boot=100)
    assert any("共通サポート" in w for w in result.comparability)


def test_per_gear_comparison(batch):
    per_gear = compare_by_gear(batch.events, reference="A")
    assert {"current_gear", "direction", "kpi", "group", "diff"} <= set(per_gear.columns)
    shift = per_gear[per_gear["kpi"] == "shift_time_s"]
    assert (shift["diff"] > 0).all()  # どのギヤでも B の方が遅い
