"""AMT 車両プラントモデルと 1 変速イベントの数値シミュレーション。

構成:
    エンジン(1慣性) --クラッチ(すべり/ロック)-- 変速機 -- 駆動軸ねじり(k, c) -- 車体質量

駆動軸のねじり要素を入れているため、クラッチ再締結時のシャッフル振動
(2〜8 Hz)が再現でき、ジャーク指標が適合パラメータに感度を持つ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .controller import (
    PlantFeedback,
    ShiftController,
    ShiftControlParams,
    ShiftPhase,
)
from .vehicle import KMH_PER_MS, VehicleParams, rads_to_rpm


@dataclass
class ShiftScenario:
    """1 回の変速を行う運転条件。"""

    from_gear: int = 2
    to_gear: int = 3
    speed_kmh: float = 45.0
    throttle: float = 0.6
    grade_pct: float = 0.0
    payload_kg: float = 0.0

    def __post_init__(self) -> None:
        if self.from_gear == self.to_gear:
            raise ValueError("from_gear と to_gear は異なる必要があります")

    @property
    def is_upshift(self) -> bool:
        return self.to_gear > self.from_gear

    def as_dict(self) -> dict[str, float]:
        return {
            "from_gear": self.from_gear,
            "to_gear": self.to_gear,
            "speed_kmh": self.speed_kmh,
            "throttle": self.throttle,
            "grade_pct": self.grade_pct,
            "payload_kg": self.payload_kg,
            "is_upshift": int(self.is_upshift),
        }


@dataclass
class SimSettings:
    dt: float = 0.001  # 積分刻み [s]
    pre_time: float = 0.30  # 変速指令前の助走 [s]
    post_time: float = 1.20  # 変速完了後の観測 [s]
    max_time: float = 6.00  # 打ち切り [s]
    jerk_cutoff_hz: float = 10.0  # ジャーク算出用ローパス [Hz]


@dataclass
class ShiftResult:
    trace: pd.DataFrame
    scenario: ShiftScenario
    control: ShiftControlParams
    shift_start_time: float
    shift_end_time: float
    completed: bool
    ratio_from: float = 1.0
    ratio_to: float = 1.0
    lock_slip_tol: float = 1.5
    settings: SimSettings = field(default_factory=SimSettings)


def _lowpass(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    """ゼロ位相ローパス(信号長が足りない場合は素通し)。"""
    fs = 1.0 / dt
    nyq = fs / 2.0
    wn = min(cutoff_hz / nyq, 0.99)
    if len(x) < 30 or wn <= 0:
        return x
    b, a = butter(2, wn)
    return filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), len(x) - 1))


def simulate_shift(
    control: ShiftControlParams,
    scenario: ShiftScenario,
    vehicle: VehicleParams | None = None,
    settings: SimSettings | None = None,
) -> ShiftResult:
    """1 回の変速イベントを時系列シミュレーションする。"""

    settings = settings or SimSettings()
    base = vehicle or VehicleParams()
    veh = VehicleParams(
        mass=base.mass,
        payload=scenario.payload_kg,
        wheel_radius=base.wheel_radius,
        drag_area=base.drag_area,
        air_density=base.air_density,
        rolling_resistance=base.rolling_resistance,
        grade=float(np.arctan(scenario.grade_pct / 100.0)),
        engine=base.engine,
        transmission=base.transmission,
        driveline=base.driveline,
    )
    trm, eng, dl = veh.transmission, veh.engine, veh.driveline
    dt = settings.dt
    n_max = int(settings.max_time / dt) + 1

    # --- 初期条件: from_gear で走行抵抗と釣り合った定常状態 -----------------
    v = scenario.speed_kmh / KMH_PER_MS
    w_w = veh.wheel_speed_for(v)
    ratio0 = trm.total_ratio(scenario.from_gear)
    w_e = max(w_w * ratio0, eng.idle_rpm / 9.5493)
    t_engine = eng.steady_torque(scenario.throttle, w_e)
    # 定常「加速中」状態のねじり角: 駆動軸両端の角加速度が一致する軸トルクを解く
    eta0 = trm.efficiency if t_engine >= 0.0 else 1.0
    a_side = veh.wheel_side_inertia(scenario.from_gear) + eng.inertia * ratio0**2 * eta0
    b_side = veh.total_mass * veh.wheel_radius**2
    t_shaft0 = (
        b_side * t_engine * ratio0 * eta0 + a_side * veh.road_load(v) * veh.wheel_radius
    ) / (a_side + b_side)
    theta = t_shaft0 / dl.stiffness
    locked = True

    ctrl = ShiftController(veh, control, scenario.from_gear, scenario.to_gear)

    rec = {
        k: np.zeros(n_max)
        for k in (
            "time", "engine_speed_rpm", "input_speed_rpm", "wheel_speed",
            "speed_kmh", "accel", "engine_torque", "torque_command",
            "clutch_position", "clutch_capacity", "clutch_torque",
            "shaft_torque", "wheel_torque", "slip", "gear",
        )
    }
    phases: list[str] = []

    shift_start = settings.pre_time
    shift_end = np.nan
    accel = 0.0
    n = 0

    for i in range(n_max):
        t = i * dt
        if t >= shift_start:
            ctrl.request_shift()

        driver_torque = eng.steady_torque(scenario.throttle, w_e)
        gear_now = ctrl.gear
        w_in = w_w * trm.total_ratio(gear_now) if gear_now else np.nan
        fb = PlantFeedback(
            engine_speed=w_e,
            wheel_speed=w_w,
            driver_torque=driver_torque,
            engine_torque=t_engine,
            locked=locked,
        )
        cmd = ctrl.step(dt, fb)

        # --- エンジントルク応答(一次遅れ + マップ制限) -------------------
        lo, hi = eng.torque_limits(w_e)
        t_cmd = float(np.clip(cmd.torque_command, lo, hi))
        t_engine += (t_cmd - t_engine) * dt / eng.torque_time_constant

        # --- 駆動軸ねじり ------------------------------------------------
        w_v = v / veh.wheel_radius
        t_shaft = dl.stiffness * theta + dl.damping * (w_w - w_v)

        gear = cmd.gear
        j_wheel = veh.wheel_side_inertia(gear)
        capacity = trm.clutch_capacity_at(cmd.clutch_position)

        if gear is None:
            # ニュートラル: 駆動トルクは完全に抜ける(トルクホール)
            locked = False
            t_clutch = 0.0
            dw_e = t_engine / eng.inertia
            a_w = -t_shaft / j_wheel
        else:
            ratio = trm.total_ratio(gear)
            eta = trm.efficiency if t_engine >= 0.0 else 1.0
            w_in = w_w * ratio
            if locked:
                a_w = (t_engine * ratio * eta - t_shaft) / (
                    j_wheel + eng.inertia * ratio**2 * eta
                )
                t_clutch = t_engine - eng.inertia * ratio * a_w
                if abs(t_clutch) > capacity:  # 容量不足 -> すべりへ遷移
                    locked = False
                else:
                    dw_e = ratio * a_w
            if not locked:
                t_clutch = capacity * np.sign(w_e - w_in) if capacity > 0 else 0.0
                dw_e = (t_engine - t_clutch) / eng.inertia
                a_w = (t_clutch * ratio * eta - t_shaft) / j_wheel

        # --- 車体 --------------------------------------------------------
        f_drive = t_shaft / veh.wheel_radius
        accel_new = (f_drive - veh.road_load(v)) / veh.total_mass

        rec["time"][i] = t
        rec["engine_speed_rpm"][i] = rads_to_rpm(w_e)
        rec["input_speed_rpm"][i] = rads_to_rpm(w_in) if gear else np.nan
        rec["wheel_speed"][i] = w_w
        rec["speed_kmh"][i] = v * KMH_PER_MS
        rec["accel"][i] = accel_new
        rec["engine_torque"][i] = t_engine
        rec["torque_command"][i] = t_cmd
        rec["clutch_position"][i] = cmd.clutch_position
        rec["clutch_capacity"][i] = capacity
        rec["clutch_torque"][i] = t_clutch
        rec["shaft_torque"][i] = t_shaft
        rec["wheel_torque"][i] = t_shaft
        rec["slip"][i] = (w_e - w_in) if gear else np.nan
        rec["gear"][i] = gear if gear else 0
        phases.append(cmd.phase.value)
        n = i + 1

        # --- 積分 (semi-implicit Euler) -----------------------------------
        accel = accel_new
        w_e = max(w_e + dw_e * dt, eng.idle_rpm / 9.5493 * 0.5)
        w_w = w_w + a_w * dt
        v = max(v + accel * dt, 0.0)
        theta += (w_w - v / veh.wheel_radius) * dt

        # クラッチロック判定(すべり収束 + 容量充足)
        if gear is not None and not locked and capacity > 0.0:
            ratio = trm.total_ratio(gear)
            if abs(w_e - w_w * ratio) < trm.lock_slip_tol:
                j_ref = j_wheel / ratio**2
                w_common = (eng.inertia * w_e + j_ref * w_w * ratio) / (eng.inertia + j_ref)
                w_e = w_common
                w_w = w_common / ratio
                locked = True

        if ctrl.phase is ShiftPhase.DONE:
            if np.isnan(shift_end):
                shift_end = t
            if t - shift_end >= settings.post_time:
                break

    trace = pd.DataFrame({k: v_[:n] for k, v_ in rec.items()})
    trace["phase"] = phases[:n]
    trace["accel_filt"] = _lowpass(trace["accel"].to_numpy(), dt, settings.jerk_cutoff_hz)
    trace["jerk"] = np.gradient(trace["accel_filt"].to_numpy(), dt)

    return ShiftResult(
        trace=trace,
        scenario=scenario,
        control=control.clipped(),
        shift_start_time=shift_start,
        shift_end_time=(
            float(shift_end) if not np.isnan(shift_end) else float(trace["time"].iloc[-1])
        ),
        completed=bool(not np.isnan(shift_end)) and not ctrl.timed_out,
        ratio_from=trm.total_ratio(scenario.from_gear),
        ratio_to=trm.total_ratio(scenario.to_gear),
        lock_slip_tol=trm.lock_slip_tol,
        settings=settings,
    )
