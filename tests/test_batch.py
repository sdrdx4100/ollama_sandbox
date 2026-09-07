import numpy as np
import pandas as pd
import pytest

from amtlab.batch import (
    analyze_logs,
    inspect_logs,
    iter_log_paths,
    load_log,
    read_log,
    read_log_schema,
    signal_presence,
)
from amtlab.ingest import SignalMap, make_j1939_demo_log
from amtlab.j1939 import to_seconds

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def log_dir(tmp_path_factory):
    """複数の parquet ログ + 信号欠けファイル + 壊れたファイル。"""
    root = tmp_path_factory.mktemp("logs")
    for i in range(4):
        make_j1939_demo_log(n_shifts=3, seed=10 + i, start_speed_kmh=22 + 3 * i).to_parquet(
            root / f"drive_{i:02d}.parquet", index=False
        )
    legacy = make_j1939_demo_log(n_shifts=3, seed=99).drop(
        columns=["ETC1::TransmissionShiftInProcess", "ETC1::PercentClutchSlip"]
    )
    legacy.to_parquet(root / "legacy.parquet", index=False)
    (root / "broken.parquet").write_bytes(b"not a parquet file")
    make_j1939_demo_log(n_shifts=2, seed=7).to_csv(root / "as_csv.csv", index=False)
    return root


# --- パス展開 --------------------------------------------------------------------
def test_directory_expands_to_every_log(log_dir):
    paths = iter_log_paths(log_dir)
    assert len(paths) == 7
    assert {p.suffix for p in paths} == {".parquet", ".csv"}


def test_glob_and_explicit_paths(log_dir):
    assert len(iter_log_paths(str(log_dir / "drive_*.parquet"))) == 4
    both = iter_log_paths([log_dir / "drive_00.parquet", log_dir / "drive_01.parquet"])
    assert len(both) == 2


def test_duplicates_are_removed(log_dir):
    path = log_dir / "drive_00.parquet"
    assert len(iter_log_paths([path, path, str(path)])) == 1


def test_missing_pattern_returns_nothing(tmp_path):
    assert iter_log_paths(tmp_path / "nope*.parquet") == []


# --- 列だけ読む ------------------------------------------------------------------
def test_schema_is_read_without_loading_data(log_dir):
    schema = read_log_schema(log_dir / "drive_00.parquet")
    assert "EEC1::EngineSpeed" in schema
    assert "ETC1::TransmissionShiftInProcess" in schema


def test_csv_schema_too(log_dir):
    assert "EEC1::EngineSpeed" in read_log_schema(log_dir / "as_csv.csv")


def test_load_log_reads_only_the_mapped_columns(log_dir, tmp_path):
    wide = pd.read_parquet(log_dir / "drive_00.parquet")
    for i in range(30):  # 使わない列を大量に足す
        wide[f"OTHER::Channel{i}"] = 0.0
    path = tmp_path / "wide.parquet"
    wide.to_parquet(path, index=False)

    assert len(read_log_schema(path)) == len(wide.columns)
    df, mapping = load_log(path)
    assert set(df.columns) == set(mapping.values())
    assert len(df.columns) < len(wide.columns)
    assert not any(c.startswith("OTHER::") for c in df.columns)


def test_read_log_dispatches_on_suffix(log_dir):
    assert len(read_log(log_dir / "as_csv.csv")) > 0
    assert len(read_log(log_dir / "drive_00.parquet")) > 0


# --- 時刻の単位吸収 ---------------------------------------------------------------
def test_to_seconds_from_datetime():
    stamps = pd.to_datetime(
        ["2024-01-01T00:00:00", "2024-01-01T00:00:01.5"], format="ISO8601"
    )
    assert list(to_seconds(pd.Series(stamps))) == [0.0, 1.5]


def test_to_seconds_from_milliseconds():
    ms = pd.Series(np.arange(0, 10_000_000, 10, dtype=float))
    assert to_seconds(ms).diff().median() == pytest.approx(0.01)


def test_to_seconds_keeps_plain_seconds():
    seconds = pd.Series(np.arange(0, 10, 0.01))
    assert to_seconds(seconds).diff().median() == pytest.approx(0.01)


# --- 一括解析 --------------------------------------------------------------------
@pytest.fixture(scope="module")
def batch(log_dir):
    return analyze_logs(log_dir, n_jobs=1)


def test_every_readable_file_is_processed(batch):
    assert batch.n_files == 7
    assert batch.n_ok == 6  # broken.parquet だけ失敗
    assert len(batch.failures()) == 1
    assert "ArrowInvalid" in batch.failures()["error"].iloc[0]


def test_events_carry_their_source_file(batch):
    assert len(batch.events) > 0
    assert batch.events["source_file"].nunique() == 6
    assert batch.events["global_event_id"].is_unique
    assert (batch.events.groupby("source_file")["event_id"].min() == 0).all()


def test_a_broken_file_does_not_stop_the_batch(batch):
    assert "読み込みに失敗" in batch.summary()
    assert len(batch.events) > 0


def test_mixed_detection_sources_are_flagged(batch):
    sources = set(batch.events["detection_source"])
    assert sources == {"shift_in_process", "gear_change"}
    assert any("切り出し方法" in w for w in batch.warnings)


def test_signal_layout_difference_is_flagged(batch):
    assert any("信号構成" in w for w in batch.warnings)


def test_missing_pattern_returns_empty_result(tmp_path):
    result = analyze_logs(tmp_path / "none*.parquet")
    assert result.events.empty
    assert result.warnings


def test_kpis_are_comparable_across_files(batch):
    for column in ("shift_time_s", "speed_drop_kmh", "speed_loss_kmh", "current_gear"):
        assert column in batch.events.columns
    assert batch.events["shift_time_s"].gt(0).all()


# --- 信号の有無マトリクス ---------------------------------------------------------
def test_signal_presence_matrix(log_dir):
    presence = signal_presence(log_dir)
    assert len(presence) == 7
    assert presence["engine_speed_rpm"].dtype == bool

    ok = presence[presence["error"] == ""]
    assert ok["engine_speed_rpm"].all()
    missing = ok.loc[~ok["shift_in_process"], "file"].tolist()
    assert missing == ["legacy.parquet"]
    assert (presence["error"] != "").sum() == 1


def test_inspect_logs_collects_rates(log_dir):
    frame = inspect_logs(log_dir, limit=2)
    assert set(frame["file"]).issubset({p.name for p in iter_log_paths(log_dir)})
    rates = frame[frame["key"] == "engine_speed_rpm"]["effective_rate_hz"]
    assert (rates > 40).all()


def test_explicit_signal_map_still_applies(log_dir):
    sm = SignalMap(speed_kmh="EBC2::FrontAxleSpeed")
    result = analyze_logs(log_dir / "drive_00.parquet", signal_map=sm, n_jobs=1)
    assert result.n_ok == 1
    assert len(result.events) > 0
