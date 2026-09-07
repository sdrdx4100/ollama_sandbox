"""AMT 変速制御(シフトシーケンス)の状態機械と適合パラメータ。

AMT は乾式単板クラッチを電動アクチュエータで操作するため、変速中は必ず
駆動トルクが抜ける(トルクホール)。したがって適合の主戦場は

    「変速時間」 ・ 「変速ショック(ジャーク)」 ・ 「クラッチ発熱(すべり仕事)」

のトレードオフになる。ここではその競合を再現できる最小限の制御則を実装する。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum

import numpy as np


class ShiftPhase(str, Enum):
    IN_GEAR = "in_gear"
    TORQUE_REDUCE = "torque_reduce"
    CLUTCH_OPEN = "clutch_open"
    GEAR_OUT = "gear_out"
    SPEED_SYNC = "speed_sync"
    GEAR_IN = "gear_in"
    CLUTCH_CLOSE = "clutch_close"
    TORQUE_RECOVERY = "torque_recovery"
    DONE = "done"


#: 適合パラメータの探索範囲 (Optuna / ランダム実験計画で共用)。
#: **乗用車(最大トルク 190 Nm / 最高 6800 rpm)基準**の値。
#: トラックのようにトルク・回転域が違う車両では :func:`scaled_bounds` で
#: スケールしたものを使う。
CONTROL_BOUNDS: dict[str, tuple[float, float]] = {
    "torque_reduce_rate": (200.0, 4000.0),
    "clutch_open_rate": (2.0, 25.0),
    "sync_gain": (0.3, 6.0),
    "sync_slip_offset": (0.0, 25.0),
    "kiss_approach_rate": (4.0, 30.0),
    "clutch_close_rate": (0.8, 12.0),
    "torque_recovery_slip": (2.0, 45.0),
    "torque_recovery_rate": (150.0, 4000.0),
}


#: エンジントルクに比例してスケールする適合値 [Nm/s], [Nm/(rad/s)]
TORQUE_SCALED_PARAMS = ("torque_reduce_rate", "torque_recovery_rate", "sync_gain")
#: 回転域に比例してスケールする適合値 [rad/s]
SPEED_SCALED_PARAMS = ("sync_slip_offset", "torque_recovery_slip")

#: CONTROL_BOUNDS が基準にしている乗用車の代表値
REFERENCE_TORQUE_NM = 190.0
REFERENCE_MAX_RPM = 6800.0


def vehicle_scales(vehicle) -> tuple[float, float]:
    """車両のトルク倍率・回転域倍率を返す(乗用車基準)。"""
    torque_scale = max(vehicle.engine.wot_torque) / REFERENCE_TORQUE_NM
    speed_scale = vehicle.engine.max_rpm / REFERENCE_MAX_RPM
    return float(torque_scale), float(speed_scale)


def scaled_bounds(vehicle) -> dict[str, tuple[float, float]]:
    """車両に合わせて適合値の探索範囲をスケールする。

    トラックはエンジントルクが 1 桁大きいので、乗用車基準のトルク勾配
    (Nm/s)では変速が成立しない。逆に回転域は狭いので、すべりオフセットの
    ような回転量は小さくする。
    """

    torque_scale, speed_scale = vehicle_scales(vehicle)
    out: dict[str, tuple[float, float]] = {}
    for name, (lo, hi) in CONTROL_BOUNDS.items():
        if name in TORQUE_SCALED_PARAMS:
            out[name] = (lo * torque_scale, hi * torque_scale)
        elif name in SPEED_SCALED_PARAMS:
            out[name] = (lo * speed_scale, hi * speed_scale)
        else:
            out[name] = (lo, hi)
    return out


@dataclass
class ShiftControlParams:
    """変速制御の適合パラメータ(いわゆる適合値/キャリブレーション定数)。"""

    torque_reduce_rate: float = 1800.0  # トルクダウン勾配 [Nm/s]
    clutch_open_rate: float = 12.0  # クラッチ解放速度 [1/s]
    sync_gain: float = 2.0  # 回転同期 P ゲイン [Nm/(rad/s)]
    sync_slip_offset: float = 8.0  # 同期目標のすべりオフセット [rad/s]
    kiss_approach_rate: float = 15.0  # ミート点までの早戻し速度 [1/s]
    clutch_close_rate: float = 3.5  # ミート後の締結速度 [1/s]
    torque_recovery_slip: float = 15.0  # トルク復帰を開始するすべり [rad/s]
    torque_recovery_rate: float = 1200.0  # トルク復帰勾配 [Nm/s]

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def clipped(self, bounds: dict[str, tuple[float, float]] | None = None
                ) -> "ShiftControlParams":
        """探索範囲内にクリップした新しいインスタンスを返す。"""
        bounds = bounds or CONTROL_BOUNDS
        values = {}
        for f in fields(self):
            lo, hi = bounds[f.name]
            values[f.name] = float(np.clip(getattr(self, f.name), lo, hi))
        return ShiftControlParams(**values)

    @staticmethod
    def default_for(vehicle) -> "ShiftControlParams":
        """車両に合わせて既定の適合値をスケールする(ベースライン用)。"""
        torque_scale, speed_scale = vehicle_scales(vehicle)
        base = ShiftControlParams()
        values = {}
        for f in fields(base):
            value = getattr(base, f.name)
            if f.name in TORQUE_SCALED_PARAMS:
                value *= torque_scale
            elif f.name in SPEED_SCALED_PARAMS:
                value *= speed_scale
            values[f.name] = float(value)
        return ShiftControlParams(**values)

    @staticmethod
    def from_dict(d: dict[str, float]) -> "ShiftControlParams":
        known = {f.name for f in fields(ShiftControlParams)}
        return ShiftControlParams(**{k: float(v) for k, v in d.items() if k in known})

    @staticmethod
    def sample(
        rng: np.random.Generator,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> "ShiftControlParams":
        """探索範囲から一様乱数でサンプリング(実験計画用)。"""
        bounds = bounds or CONTROL_BOUNDS
        return ShiftControlParams(
            **{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in bounds.items()}
        )


@dataclass
class ControlCommand:
    """1 ステップ分の制御指令。"""

    torque_command: float  # エンジントルク要求 [Nm]
    clutch_position: float  # クラッチストローク比 [-]
    gear: int | None  # 噛み合っているギヤ (None = ニュートラル)
    phase: ShiftPhase


@dataclass
class PlantFeedback:
    """制御器が参照するプラント状態。"""

    engine_speed: float  # [rad/s]
    wheel_speed: float  # [rad/s]
    driver_torque: float  # ドライバ要求トルク [Nm]
    engine_torque: float  # 実エンジントルク [Nm]
    locked: bool


class ShiftController:
    """アップ/ダウンシフト 1 回分のシーケンスを司る状態機械。"""

    #: 各フェーズのタイムアウト [s] (異常適合値で無限ループしないための保険)
    PHASE_TIMEOUT = {
        ShiftPhase.TORQUE_REDUCE: 0.60,
        ShiftPhase.CLUTCH_OPEN: 0.60,
        ShiftPhase.GEAR_OUT: 0.30,
        ShiftPhase.SPEED_SYNC: 1.50,
        ShiftPhase.GEAR_IN: 0.30,
        ShiftPhase.CLUTCH_CLOSE: 1.50,
        ShiftPhase.TORQUE_RECOVERY: 1.00,
    }
    SYNC_TOLERANCE = 5.0  # 同期完了とみなす回転数差 [rad/s]

    def __init__(self, vehicle, params: ShiftControlParams, from_gear: int, to_gear: int):
        self.vehicle = vehicle
        self.params = params.clipped(scaled_bounds(vehicle))
        self.from_gear = from_gear
        self.to_gear = to_gear
        self.phase = ShiftPhase.IN_GEAR
        self.phase_time = 0.0
        self.total_time = 0.0
        self.torque_command = 0.0
        self.clutch_position = 1.0
        self.gear: int | None = from_gear
        self.shift_started = False
        self.shift_finished_time: float | None = None
        self.timed_out = False

    # ------------------------------------------------------------------
    @property
    def is_upshift(self) -> bool:
        return self.to_gear > self.from_gear

    def sync_target_speed(self, wheel_speed: float) -> float:
        """締結先ギヤでの入力軸回転 + すべりオフセット [rad/s]。"""
        w_in = wheel_speed * self.vehicle.transmission.total_ratio(self.to_gear)
        return w_in + self.params.sync_slip_offset

    def request_shift(self) -> None:
        if not self.shift_started:
            self.shift_started = True
            self._enter(ShiftPhase.TORQUE_REDUCE)

    # ------------------------------------------------------------------
    def _enter(self, phase: ShiftPhase) -> None:
        self.phase = phase
        self.phase_time = 0.0

    def _ramp(self, value: float, target: float, rate: float, dt: float) -> float:
        step = rate * dt
        if value < target:
            return min(value + step, target)
        return max(value - step, target)

    def step(self, dt: float, fb: PlantFeedback) -> ControlCommand:
        p = self.params
        self.total_time += dt
        self.phase_time += dt
        timeout = self.PHASE_TIMEOUT.get(self.phase, np.inf)
        expired = self.phase_time >= timeout

        if self.phase is ShiftPhase.IN_GEAR:
            self.torque_command = fb.driver_torque
            self.clutch_position = 1.0
            self.gear = self.from_gear

        elif self.phase is ShiftPhase.TORQUE_REDUCE:
            self.torque_command = self._ramp(self.torque_command, 0.0, p.torque_reduce_rate, dt)
            if abs(self.torque_command) < 5.0 or expired:
                self._enter(ShiftPhase.CLUTCH_OPEN)

        elif self.phase is ShiftPhase.CLUTCH_OPEN:
            self.torque_command = self._ramp(self.torque_command, 0.0, p.torque_reduce_rate, dt)
            self.clutch_position = self._ramp(self.clutch_position, 0.0, p.clutch_open_rate, dt)
            kiss = self.vehicle.transmission.clutch_kiss_point
            if self.clutch_position <= kiss * 0.98 or expired:
                self._enter(ShiftPhase.GEAR_OUT)

        elif self.phase is ShiftPhase.GEAR_OUT:
            self.clutch_position = self._ramp(self.clutch_position, 0.0, p.clutch_open_rate, dt)
            self.gear = None
            if self.phase_time >= self.vehicle.transmission.gear_out_time or expired:
                self._enter(ShiftPhase.SPEED_SYNC)

        elif self.phase is ShiftPhase.SPEED_SYNC:
            target = self.sync_target_speed(fb.wheel_speed)
            err = target - fb.engine_speed
            lo, hi = self.vehicle.engine.torque_limits(fb.engine_speed)
            self.torque_command = float(np.clip(p.sync_gain * err, lo, hi))
            if abs(err) <= self.SYNC_TOLERANCE or expired:
                self.timed_out |= expired
                self._enter(ShiftPhase.GEAR_IN)

        elif self.phase is ShiftPhase.GEAR_IN:
            target = self.sync_target_speed(fb.wheel_speed)
            lo, hi = self.vehicle.engine.torque_limits(fb.engine_speed)
            self.torque_command = float(np.clip(p.sync_gain * (target - fb.engine_speed), lo, hi))
            if self.phase_time >= self.vehicle.transmission.gear_in_time or expired:
                self.gear = self.to_gear
                self._enter(ShiftPhase.CLUTCH_CLOSE)

        elif self.phase is ShiftPhase.CLUTCH_CLOSE:
            self.gear = self.to_gear
            kiss = self.vehicle.transmission.clutch_kiss_point
            if self.clutch_position < kiss:
                rate = p.kiss_approach_rate
            else:
                rate = p.clutch_close_rate
            self.clutch_position = self._ramp(self.clutch_position, 1.0, rate, dt)

            w_in = fb.wheel_speed * self.vehicle.transmission.total_ratio(self.to_gear)
            slip = abs(fb.engine_speed - w_in)
            if slip <= p.torque_recovery_slip or fb.locked:
                self.torque_command = self._ramp(
                    self.torque_command, fb.driver_torque, p.torque_recovery_rate, dt
                )
            else:
                lo, hi = self.vehicle.engine.torque_limits(fb.engine_speed)
                target = self.sync_target_speed(fb.wheel_speed)
                self.torque_command = float(
                    np.clip(p.sync_gain * (target - fb.engine_speed), lo, hi)
                )
            if (fb.locked and self.clutch_position > 0.995) or expired:
                self.timed_out |= expired
                self._enter(ShiftPhase.TORQUE_RECOVERY)

        elif self.phase is ShiftPhase.TORQUE_RECOVERY:
            self.gear = self.to_gear
            self.clutch_position = self._ramp(self.clutch_position, 1.0, p.clutch_close_rate, dt)
            self.torque_command = self._ramp(
                self.torque_command, fb.driver_torque, p.torque_recovery_rate, dt
            )
            recovered = abs(self.torque_command - fb.driver_torque) < 2.0
            if (recovered and fb.locked) or expired:
                self.timed_out |= expired
                self.shift_finished_time = self.total_time
                self._enter(ShiftPhase.DONE)

        else:  # DONE
            self.torque_command = fb.driver_torque
            self.clutch_position = 1.0
            self.gear = self.to_gear

        return ControlCommand(
            torque_command=self.torque_command,
            clutch_position=float(np.clip(self.clutch_position, 0.0, 1.0)),
            gear=self.gear,
            phase=self.phase,
        )
