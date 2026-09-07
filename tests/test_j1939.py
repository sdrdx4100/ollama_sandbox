import numpy as np
import pandas as pd
import pytest

from amtlab.ingest import (
    J1939_DEMO_COLUMNS,
    SignalMap,
    analyze_log,
    detect_shift_events,
    make_j1939_demo_log,
    standardize,
)
from amtlab.j1939 import (
    CATALOG_BY_KEY,
    J1939_CATALOG,
    _normalize,
    decode,
    detect_columns,
    format_report,
    inspect_log,
    split_prefix,
)


@pytest.fixture(scope="module")
def j1939_log():
    return make_j1939_demo_log(n_shifts=4, seed=3)


# --- 列名の自動検出 --------------------------------------------------------------
def test_catalog_keys_are_unique():
    keys = [s.key for s in J1939_CATALOG]
    assert len(keys) == len(set(keys))
    spns = [s.spn for s in J1939_CATALOG if s.spn]
    assert len(spns) == len(set(spns))


def test_detects_every_demo_signal(j1939_log):
    mapping = detect_columns(j1939_log)
    for key, column in J1939_DEMO_COLUMNS.items():
        assert mapping.get(key) == column, key


# --- 列名の正規化(UpperCamelCase + メッセージ名プレフィックス) ------------------
@pytest.mark.parametrize(
    "column,prefix,body",
    [
        ("EEC1::EngineSpeed", "EEC1", "EngineSpeed"),
        ("ETC2::TransmissionCurrentGear", "ETC2", "TransmissionCurrentGear"),
        ("EEC1.ActualEnginePercentTorque", "EEC1", "ActualEnginePercentTorque"),
        ("EEC1_EngineSpeed", "EEC1", "EngineSpeed"),
        ("CCVS1/WheelBasedVehicleSpeed", "CCVS1", "WheelBasedVehicleSpeed"),
        ("engine_speed_rpm", "", "engine_speed_rpm"),  # 小文字は割らない
        ("speed_kmh", "", "speed_kmh"),
        ("EngineSpeed", "", "EngineSpeed"),
    ],
)
def test_prefix_is_split_only_when_it_looks_like_a_message_name(column, prefix, body):
    assert split_prefix(column) == (prefix, body)


@pytest.mark.parametrize(
    "raw,normalized",
    [
        ("TransmissionCurrentGear", "transmission_current_gear"),
        ("AcceleratorPedalPosition1", "accelerator_pedal_position_1"),
        ("WheelBasedVehicleSpeed", "wheel_based_vehicle_speed"),
        ("PercentClutchSlip", "percent_clutch_slip"),
        ("engine_speed_rpm", "engine_speed_rpm"),
    ],
)
def test_camel_case_is_split_into_words(raw, normalized):
    assert _normalize(raw) == normalized


def test_upper_camel_case_with_prefix_is_detected():
    columns = [
        "Timestamp",
        "EEC1::EngineSpeed",
        "CCVS1::WheelBasedVehicleSpeed",
        "ETC1::TransmissionShiftInProcess",
        "ETC2::TransmissionCurrentGear",
        "ETC2::TransmissionSelectedGear",
        "ETC2::TransmissionActualGearRatio",
        "ETC1::PercentClutchSlip",
        "ETC1::TransmissionInputShaftSpeed",
        "ETC1::TransmissionOutputShaftSpeed",
        "EEC2::AcceleratorPedalPosition1",
        "EEC1::ActualEnginePercentTorque",
        "EEC1::DriversDemandEnginePercentTorque",
        "EC1::EngineReferenceTorque",
        "EBC2::FrontAxleSpeed",
    ]
    mapping = detect_columns(pd.DataFrame(columns=columns))
    assert mapping["engine_speed_rpm"] == "EEC1::EngineSpeed"
    assert mapping["shift_in_process"] == "ETC1::TransmissionShiftInProcess"
    assert mapping["gear"] == "ETC2::TransmissionCurrentGear"
    assert mapping["selected_gear"] == "ETC2::TransmissionSelectedGear"
    assert mapping["gear_ratio"] == "ETC2::TransmissionActualGearRatio"
    assert len(mapping) == len(columns)  # 全列が別々の信号に割り当たる


def test_standard_dbc_short_names_are_detected():
    """``Transmission`` → ``Trans``、``Engine`` → ``Eng`` の短縮形に対応する。"""
    expected = {
        "Timestamp": "time",
        "EEC1::EngSpeed": "engine_speed_rpm",
        "EEC1::ActualEngPercentTorque": "actual_torque_pct",
        "EEC1::DriversDemandEngPercentTorque": "demand_torque_pct",
        "EEC1::EngNominalFrictionPercentTorque": "friction_torque_pct",
        "EC1::EngReferenceTorque": "reference_torque_nm",
        "EEC2::AccelPedalPos1": "accel_pedal_pct",
        "CCVS1::WheelBasedVehicleSpeed": "speed_kmh",
        "EBC2::FrontAxleSpeed": "front_axle_speed_kmh",
        "ETC1::TransInputShaftSpeed": "input_shaft_rpm",
        "ETC1::TransOutputShaftSpeed": "output_shaft_rpm",
        "ETC1::PercentClutchSlip": "clutch_slip_pct",
        "ETC1::TransShiftInProcess": "shift_in_process",
        "ETC2::TransCurrentGear": "gear",
        "ETC2::TransSelectedGear": "selected_gear",
        "ETC2::TransActualGearRatio": "gear_ratio",
    }
    mapping = detect_columns(pd.DataFrame(columns=list(expected)))
    assert {v: k for k, v in mapping.items()} == expected


def test_short_and_long_names_resolve_identically():
    short = make_j1939_demo_log(n_shifts=2, seed=1, naming="short")
    long = make_j1939_demo_log(n_shifts=2, seed=1, naming="long")
    assert set(detect_columns(short)) == set(detect_columns(long))
    assert list(short.columns) != list(long.columns)


@pytest.mark.parametrize(
    "raw,expanded",
    [
        ("TransShiftInProcess", "transmission_shift_in_process"),
        ("EngSpeed", "engine_speed"),
        ("AccelPedalPos1", "accelerator_pedal_position_1"),
        ("ActualEngPercentTorque", "actual_engine_percent_torque"),
        # 部分一致では展開しない(Engaged の eng を engine にしない)
        ("TransDrivelineEngaged", "transmission_driveline_engaged"),
        ("EngineSpeed", "engine_speed"),
    ],
)
def test_abbreviations_are_expanded_token_wise(raw, expanded):
    assert _normalize(raw) == expanded


def test_abbreviated_decoy_is_still_rejected():
    """短縮形でも SPN 512 と SPN 2432 を取り違えない。"""
    columns = ["Timestamp", "EEC1::EngSpeed", "CCVS1::WheelBasedVehicleSpeed",
               "EEC1::DriversDemandEngPercentTorque", "EEC1::EngDemandPercentTorque"]
    mapping = detect_columns(pd.DataFrame(columns=columns))
    assert mapping["demand_torque_pct"] == "EEC1::DriversDemandEngPercentTorque"


def test_specific_alias_beats_a_similar_signal():
    """SPN 512(Driver's Demand)と SPN 2432(Engine Demand)を取り違えない。"""
    columns = [
        "Timestamp",
        "EEC1::EngineSpeed",
        "CCVS1::WheelBasedVehicleSpeed",
        "EEC1::DriversDemandEnginePercentTorque",
        "EEC1::EngineDemandPercentTorque",
    ]
    mapping = detect_columns(pd.DataFrame(columns=columns))
    assert mapping["demand_torque_pct"] == "EEC1::DriversDemandEnginePercentTorque"
    assert "EEC1::EngineDemandPercentTorque" not in mapping.values()


def test_each_column_is_used_by_at_most_one_signal():
    columns = ["Timestamp", "EEC1::EngineSpeed", "CCVS1::WheelBasedVehicleSpeed",
               "ETC2::TransmissionCurrentGear", "ETC2::TransmissionActualGearRatio"]
    mapping = detect_columns(pd.DataFrame(columns=columns))
    assert len(set(mapping.values())) == len(mapping)


def test_unrelated_columns_stay_unmapped():
    columns = ["Timestamp", "EEC1::EngineSpeed", "CCVS1::WheelBasedVehicleSpeed",
               "ETC2::TransmissionCurrentGear", "CCVS1::ParkingBrakeSwitch",
               "ETC1::TransmissionDrivelineEngaged", "AMB::AmbientAirTemperature"]
    mapping = detect_columns(pd.DataFrame(columns=columns))
    for noise in ("CCVS1::ParkingBrakeSwitch", "ETC1::TransmissionDrivelineEngaged",
                  "AMB::AmbientAirTemperature"):
        assert noise not in mapping.values()


@pytest.mark.parametrize(
    "column,expected",
    [
        ("ETC2_TransmissionCurrentGear", "gear"),
        ("Transmission Selected Gear", "selected_gear"),
        ("ETC2_TransmissionActualGearRatio", "gear_ratio"),
        ("SPN_574", "shift_in_process"),
        ("エンジン回転数", "engine_speed_rpm"),
        ("クラッチ滑り率", "clutch_slip_pct"),
        ("ホイールベース車速", "speed_kmh"),
        ("アクセルペダルポジション", "accel_pedal_pct"),
        ("Engine_Reference_Torque", "reference_torque_nm"),
        ("input shaft speed", "input_shaft_rpm"),
    ],
)
def test_alias_matching(column, expected):
    mapping = detect_columns(pd.DataFrame(columns=["time", column]))
    assert mapping.get(expected) == column


def test_current_gear_alias_does_not_swallow_gear_ratio():
    df = pd.DataFrame(columns=["time", "ActualGear", "ActualGearRatio"])
    mapping = detect_columns(df)
    assert mapping["gear"] == "ActualGear"
    assert mapping["gear_ratio"] == "ActualGearRatio"


# --- 物理量への変換 --------------------------------------------------------------
def test_decode_builds_torque_in_nm(j1939_log):
    out = decode(j1939_log)
    assert "engine_torque_nm" in out and "demand_torque_nm" in out
    ref = out["reference_torque_nm"]
    expected = out["actual_torque_pct"] * ref / 100.0
    assert np.allclose(out["engine_torque_nm"], expected)


def test_decode_marks_neutral_from_gear_ratio(j1939_log):
    out = decode(j1939_log)
    neutral = out["gear_ratio"].abs() < 1e-3
    assert neutral.any()
    assert (out.loc[neutral, "gear"] == 0).all()


def test_decode_converts_pedal_to_throttle(j1939_log):
    out = decode(j1939_log)
    assert out["throttle"].between(0.0, 1.0).all()


def test_shift_in_process_is_boolean(j1939_log):
    out = decode(j1939_log)
    assert set(out["shift_in_process"].unique()) <= {0, 1}


# --- 診断レポート ----------------------------------------------------------------
def test_inspect_reports_rates_and_discrete_channels(j1939_log):
    report = inspect_log(j1939_log)
    rates = {c.key: c.effective_rate_hz for c in report.channels}
    assert rates["engine_speed_rpm"] == pytest.approx(50.0, abs=1.0)
    assert rates["speed_kmh"] == pytest.approx(10.0, abs=1.0)
    assert np.isnan(rates["gear"])  # 離散信号は周期を推定しない
    assert "rows=" in format_report(report)


def test_inspect_warns_that_jerk_is_not_quantitative(j1939_log):
    report = inspect_log(j1939_log)
    assert not report.jerk_is_reliable
    assert any("ジャーク" in w for w in report.warnings)
    # 駆動側の速度をジャークに使わないことが明示されている
    assert any("駆動側の速度" in w for w in report.warnings)


def test_best_speed_channel_excludes_the_driveline(j1939_log):
    report = inspect_log(j1939_log)
    best = report.best_speed_channel()
    assert best is not None
    assert best.key in ("front_axle_speed_kmh", "speed_kmh")


def test_inspect_flags_raw_gear_values():
    df = pd.DataFrame(
        {
            "time": np.arange(100) * 0.01,
            "EEC1_EngineSpeed": np.full(100, 2000.0),
            "CCVS1_WheelBasedVehicleSpeed": np.linspace(50, 51, 100),
            "ETC2_TransmissionCurrentGear": np.full(100, 128.0),  # offset -125 未適用
        }
    )
    report = inspect_log(df)
    assert any("offset -125" in w for w in report.warnings)


# --- イベント切り出し ------------------------------------------------------------
def test_events_come_from_the_shift_flag(j1939_log):
    trace = standardize(j1939_log)
    events = detect_shift_events(trace)
    assert [(e.from_gear, e.to_gear) for e in events] == [(1, 2), (2, 3), (3, 4), (4, 5)]
    assert all(e.source == "shift_in_process" for e in events)
    assert all(e.t_start < e.t_engage <= e.t_end for e in events)
    assert all(np.isfinite(e.ratio_to) and e.ratio_to > 0 for e in events)


def test_flag_based_shift_time_covers_the_torque_phase(j1939_log):
    """フラグ起点の変速時間は、ギヤ抜き起点より必ず長い(トルクダウンを含む)。"""
    trace = standardize(j1939_log)
    flag_events = detect_shift_events(trace)
    gear_only = trace.drop(columns=["shift_in_process"])
    gear_only.attrs.update(trace.attrs)
    gear_events = detect_shift_events(gear_only)
    assert len(flag_events) == len(gear_events)
    for flag, gear in zip(flag_events, gear_events):
        assert flag.t_start < gear.t_start
        assert flag.t_end - flag.t_start > gear.t_end - gear.t_start


def test_kpis_use_measured_signals(j1939_log):
    _, kpi = analyze_log(j1939_log)
    assert len(kpi) == 4
    assert (kpi["detection_source"] == "shift_in_process").all()
    assert (kpi["torque_source"] == "est_wheel_torque_nm").all()
    assert (kpi["flare_source"] == "output_shaft_rpm x gear_ratio").all()
    assert (kpi["jerk_quality"] == "low_rate").all()
    for column in ("gear_engage_time_s", "clutch_close_time_s", "clutch_slip_time_s",
                   "torque_recovery_s"):
        assert kpi[column].notna().all(), column
    assert (kpi["clutch_close_time_s"] >= 0).all()


def test_low_rate_speed_does_not_explode_the_jerk(j1939_log):
    """車速 10 Hz の階段状データを微分してもジャークが発散しないこと。"""
    trace = standardize(j1939_log)
    assert trace.attrs["jerk_quality"] == "low_rate"
    assert trace.attrs["jerk_cutoff_hz"] < 10.0
    assert trace["jerk"].abs().max() < 60.0


def test_driveline_speed_is_kept_but_not_used_for_accel(j1939_log):
    trace = standardize(j1939_log)
    assert "driveline_speed_kmh" in trace.columns
    assert "output_shaft" not in trace.attrs["accel_source"]


# --- 明示指定と自動検出の併用 ------------------------------------------------------
def test_explicit_columns_win_over_auto_detection(j1939_log):
    sm = SignalMap(speed_kmh="EBC2::FrontAxleSpeed")
    mapping = sm.resolve(j1939_log)
    assert mapping["speed_kmh"] == "EBC2::FrontAxleSpeed"


def test_auto_detection_can_be_disabled(j1939_log):
    sm = SignalMap(auto_detect=False)
    with pytest.raises(KeyError):
        sm.resolve(j1939_log)


def test_missing_signals_raise_with_a_helpful_message():
    df = pd.DataFrame({"time": [0.0, 0.1], "ETC2_TransmissionCurrentGear": [1, 1]})
    with pytest.raises(KeyError, match="engine_speed_rpm"):
        SignalMap().resolve(df)


def test_report_lists_unmapped_columns(j1939_log):
    log = j1939_log.copy()
    log["CCVS1::ParkingBrakeSwitch"] = 0
    report = inspect_log(log)
    assert "CCVS1::ParkingBrakeSwitch" in report.unmapped_columns
    assert "マッピングされなかった列" in format_report(report)


def test_signal_map_file_overrides_auto_detection(j1939_log, tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text("speed_kmh: 'EBC2::FrontAxleSpeed'\n", encoding="utf-8")
    sm = SignalMap.from_file(path)
    mapping = sm.resolve(j1939_log)
    assert mapping["speed_kmh"] == "EBC2::FrontAxleSpeed"
    assert mapping["engine_speed_rpm"] == "EEC1::EngSpeed"  # 残りは自動検出


def test_signal_map_file_rejects_unknown_keys(tmp_path):
    path = tmp_path / "map.yaml"
    path.write_text("nonexistent_signal: X\n", encoding="utf-8")
    with pytest.raises(KeyError):
        SignalMap.from_file(path)


def test_unknown_keys_do_not_break_the_report(j1939_log):
    mapping = detect_columns(j1939_log)
    mapping["shaft_torque"] = "EC1_EngineReferenceTorque"  # カタログ外のキー
    report = inspect_log(j1939_log, mapping)
    assert all(c.key in CATALOG_BY_KEY for c in report.channels)
