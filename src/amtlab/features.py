"""変速イベントの時系列から特徴量(変速品質 KPI)を抽出する。

実車の適合で使われる代表的な指標:
  * 変速時間      : トルクダウン開始 〜 トルク復帰完了
  * トルク抜け時間 : 駆動トルクが変速前の一定割合を下回る時間
  * ジャーク       : 加速度の時間微分(乗り心地の代理指標)
  * クラッチ仕事   : すべり伝達エネルギ(耐久・発熱)
  * 吹け上がり     : 同期先回転に対するエンジン回転のオーバーシュート
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .simulation.plant import ShiftResult
from .simulation.vehicle import RPM_PER_RADS

#: 目的関数として扱う KPI (小さいほど良い)
OBJECTIVE_KPIS = ("jerk_rms", "shift_time_s", "clutch_energy_j")

#: スカラー化の際の正規化スケール(代表的な良品レベル)
OBJECTIVE_SCALES = {"jerk_rms": 6.0, "shift_time_s": 1.0, "clutch_energy_j": 500.0}

DEFAULT_WEIGHTS = {"jerk_rms": 0.5, "shift_time_s": 0.35, "clutch_energy_j": 0.15}


def _window(trace: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    return trace[(trace["time"] >= t0) & (trace["time"] <= t1)]


def extract_kpis(result: ShiftResult, settle_time: float = 0.6) -> dict[str, float]:
    """1 変速イベントの KPI を辞書で返す。"""

    tr = result.trace
    t0 = result.shift_start_time
    t1 = result.shift_end_time
    dt = result.settings.dt
    win = _window(tr, t0, t1 + settle_time)
    pre = _window(tr, max(t0 - 0.15, tr["time"].iloc[0]), t0)

    pre_torque = float(pre["shaft_torque"].mean()) if len(pre) else 0.0
    pre_accel = float(pre["accel"].mean()) if len(pre) else 0.0
    pre_speed = float(pre["speed_kmh"].mean()) if len(pre) else float(tr["speed_kmh"].iloc[0])

    jerk = win["jerk"].to_numpy()
    accel = win["accel"].to_numpy()

    # トルク抜け: 変速前トルクの 20% を下回っている時間
    thr = 0.2 * pre_torque
    if pre_torque > 0:
        interrupt = float((win["shaft_torque"].to_numpy() < thr).sum() * dt)
    else:
        interrupt = float((win["shaft_torque"].to_numpy() > thr).sum() * dt)

    # クラッチすべり仕事
    slip = win["slip"].to_numpy()
    ct = win["clutch_torque"].to_numpy()
    power = np.abs(np.nan_to_num(ct * slip, nan=0.0))
    clutch_energy = float(np.trapezoid(power, dx=dt))
    slipping = np.abs(np.nan_to_num(slip, nan=0.0)) > result.lock_slip_tol
    slip_time = float((slipping & (np.abs(ct) > 1.0)).sum() * dt)

    # 吹け上がり: 変速中のエンジン回転が「開始回転 or 同期先回転」をどれだけ超えたか
    target_rpm = win["wheel_speed"].to_numpy() * result.ratio_to * RPM_PER_RADS
    start_rpm = float(pre["engine_speed_rpm"].mean()) if len(pre) else float(
        tr["engine_speed_rpm"].iloc[0]
    )
    reference = np.maximum(target_rpm, start_rpm)
    during_shift = win["phase"].isin(
        ["gear_out", "speed_sync", "gear_in", "clutch_close"]
    ).to_numpy()
    flare = win["engine_speed_rpm"].to_numpy() - reference
    engine_flare_rpm = float(np.max(flare[during_shift])) if during_shift.any() else 0.0

    phase_time = win.groupby("phase")["time"].agg(lambda s: (len(s) * dt))

    kpis = {
        "shift_time_s": float(t1 - t0),
        "torque_interrupt_s": interrupt,
        "jerk_rms": float(np.sqrt(np.mean(jerk**2))) if len(jerk) else np.nan,
        "jerk_peak": float(np.max(np.abs(jerk))) if len(jerk) else np.nan,
        "accel_drop": float(pre_accel - np.min(accel)) if len(accel) else np.nan,
        "min_accel": float(np.min(accel)) if len(accel) else np.nan,
        "speed_loss_kmh": max(float(pre_speed - np.min(win["speed_kmh"].to_numpy())), 0.0)
        if len(win)
        else np.nan,
        "clutch_energy_j": clutch_energy,
        "clutch_slip_time_s": slip_time,
        "engine_flare_rpm": max(engine_flare_rpm, 0.0),
        "sync_time_s": float(phase_time.get("speed_sync", 0.0)),
        "clutch_close_time_s": float(phase_time.get("clutch_close", 0.0)),
        "pre_shaft_torque_nm": pre_torque,
        "completed": float(result.completed),
    }
    kpis["quality_score"] = scalar_objective(kpis)
    return kpis


def scalar_objective(
    kpis: dict[str, float], weights: dict[str, float] | None = None
) -> float:
    """複数 KPI を正規化して重み付き和にしたスカラー目的関数(小さいほど良い)。"""

    w = weights or DEFAULT_WEIGHTS
    score = 0.0
    for key, weight in w.items():
        value = kpis.get(key, np.nan)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("inf")
        score += weight * float(value) / OBJECTIVE_SCALES.get(key, 1.0)
    if not kpis.get("completed", 1.0):
        score += 1.0  # 変速未完了はペナルティ
    return float(score)


def objective_vector(kpis: dict[str, float]) -> tuple[float, ...]:
    """多目的最適化用の目的値ベクトル。"""
    return tuple(float(kpis[k]) for k in OBJECTIVE_KPIS)


def result_to_row(result: ShiftResult) -> dict[str, float]:
    """1 実験 = 1 行(条件 + 適合値 + KPI)に変換する。"""
    row: dict[str, float] = {}
    row.update(result.scenario.as_dict())
    row.update(result.control.as_dict())
    row.update(extract_kpis(result))
    return row


def results_to_frame(results) -> pd.DataFrame:
    """複数の ShiftResult を解析用 DataFrame にまとめる。"""
    return pd.DataFrame([result_to_row(r) for r in results])
